import copy
import importlib
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch import Tensor
import yaml

from models.diffusion_utils import (
    ddim_sample_x0_prediction,
    make_beta_schedule,
    q_sample_x0,
    sample_query_noise,
)
from utils.ExpConfigs import ExpConfigs
from utils.irregular_grid import gather_grid_queries, gridify_future_queries
from utils.query_adapter import flatten_apn_batch, masked_mse, scatter_query_values


class TimeEncoding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        if dim <= 1:
            self.scale = nn.Linear(1, dim)
            self.periodic = None
        else:
            self.scale = nn.Linear(1, 1)
            self.periodic = nn.Linear(1, dim - 1)

    def forward(self, t: Tensor) -> Tensor:
        t = t.unsqueeze(-1)
        if self.periodic is None:
            return self.scale(t)
        return torch.cat([self.scale(t), torch.sin(self.periodic(t))], dim=-1)


class DiffusionStepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, steps: Tensor) -> Tensor:
        half_dim = self.dim // 2
        if half_dim == 0:
            emb = steps.float().unsqueeze(-1)
        else:
            exponent = -math.log(10000.0) * torch.arange(
                half_dim, device=steps.device, dtype=torch.float32
            ) / max(half_dim - 1, 1)
            freqs = torch.exp(exponent)
            args = steps.float().unsqueeze(-1) * freqs.unsqueeze(0)
            emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
            if emb.shape[-1] < self.dim:
                emb = torch.nn.functional.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.proj(emb.to(dtype=self.proj[0].weight.dtype))


class QueryDenoisingBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm_self = nn.LayerNorm(d_model)
        self.norm_cross = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        history: Tensor,
        query_key_padding_mask: Tensor,
        history_key_padding_mask: Tensor,
        query_mask: Tensor,
    ) -> Tensor:
        self_out, _ = self.self_attn(
            query,
            query,
            query,
            key_padding_mask=query_key_padding_mask,
            need_weights=False,
        )
        query = self.norm_self(query + self.dropout(self_out))
        query = query * query_mask.unsqueeze(-1)

        cross_out, _ = self.cross_attn(
            query,
            history,
            history,
            key_padding_mask=history_key_padding_mask,
            need_weights=False,
        )
        query = self.norm_cross(query + self.dropout(cross_out))
        query = query * query_mask.unsqueeze(-1)

        ffn_out = self.ffn(query)
        query = self.norm_ffn(query + self.dropout(ffn_out))
        return query * query_mask.unsqueeze(-1)


class QueryTemporalVariableContext(nn.Module):
    """Condition a query on local same-variable and cross-variable history."""

    def __init__(
        self,
        d_model: int,
        mode: str,
        lengths: tuple[float, ...],
        use_query_group: bool,
        dropout: float,
    ):
        super().__init__()
        self.mode = mode
        self.use_query_group = use_query_group
        length_tensor = torch.tensor(lengths, dtype=torch.float32).clamp_min(1e-6)
        self.register_buffer("lengths", length_tensor, persistent=False)
        n_scales = len(lengths)
        # Per scale: local mean, effective support, and nearest-observation gap.
        self.same_var_proj = nn.Sequential(
            nn.LayerNorm(3 * n_scales),
            nn.Linear(3 * n_scales, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        if mode == "temporal_variable":
            self.history_value_proj = nn.Linear(1, d_model)
            self.temporal_proj = nn.Sequential(
                nn.LayerNorm(n_scales * d_model),
                nn.Linear(n_scales * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
        else:
            self.history_value_proj = None
            self.temporal_proj = None
        self.group_proj = nn.Linear(d_model, d_model) if use_query_group else None
        self.output_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hist_value: Tensor,
        hist_time: Tensor,
        hist_var: Tensor,
        hist_mask: Tensor,
        query_time: Tensor,
        query_var: Tensor,
        query_row: Tensor,
        query_mask: Tensor,
        var_embedding: nn.Embedding,
    ) -> Tensor:
        dtype = hist_value.dtype
        valid_history = hist_mask[:, None, :].to(dtype)
        delta = query_time[:, :, None] - hist_time[:, None, :]
        causal = (delta >= 0).to(dtype)
        delta = delta.clamp_min(0.0)
        base_weight = valid_history * causal
        same_variable = hist_var[:, None, :].eq(query_var[:, :, None]).to(dtype)
        same_weight = base_weight * same_variable

        same_features = []
        temporal_tokens = []
        history_token = None
        if self.temporal_proj is not None:
            assert self.history_value_proj is not None
            history_token = self.history_value_proj(hist_value.unsqueeze(-1)) + var_embedding(hist_var)

        for length in self.lengths.to(device=hist_value.device, dtype=dtype):
            kernel = torch.exp(-delta / length)
            local_weight = same_weight * kernel
            local_mass = local_weight.sum(dim=-1).clamp_min(1e-6)
            local_mean = (local_weight * hist_value[:, None, :]).sum(dim=-1) / local_mass
            nearest_gap = delta.masked_fill(local_weight <= 0, float("inf")).amin(dim=-1)
            nearest_gap = torch.where(torch.isfinite(nearest_gap), nearest_gap, torch.ones_like(nearest_gap))
            same_features.extend((local_mean, torch.log1p(local_mass), nearest_gap))

            if history_token is not None:
                temporal_weight = base_weight * kernel
                temporal_mass = temporal_weight.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                temporal_tokens.append(
                    torch.einsum("bqh,bhd->bqd", temporal_weight / temporal_mass, history_token)
                )

        context = self.same_var_proj(torch.stack(same_features, dim=-1))
        if temporal_tokens:
            assert self.temporal_proj is not None
            context = context + self.temporal_proj(torch.cat(temporal_tokens, dim=-1))

        if self.group_proj is not None:
            same_row = query_row[:, :, None].eq(query_row[:, None, :]).to(dtype)
            group_weight = same_row * query_mask[:, None, :]
            group_mass = group_weight.sum(dim=-1, keepdim=True).clamp_min(1.0)
            group_token = torch.einsum(
                "bqk,bkd->bqd",
                group_weight / group_mass,
                var_embedding(query_var),
            )
            context = context + self.group_proj(group_token)

        return self.dropout(self.output_norm(context)) * query_mask.unsqueeze(-1)


def _safe_key_padding_mask(valid_mask: Tensor) -> Tensor:
    valid = valid_mask.bool()
    padding = ~valid
    empty_rows = valid.sum(dim=1) == 0
    if empty_rows.any():
        padding = padding.clone()
        padding[empty_rows, 0] = False
    return padding


class QSDiffDenoiser(nn.Module):
    def __init__(self, configs: ExpConfigs):
        super().__init__()
        d_model = configs.d_model
        n_heads = configs.n_heads
        num_layers = configs.n_layers
        d_ff = configs.d_ff if configs.d_ff > 0 else d_model * 4
        dropout = configs.dropout

        self.value_proj = nn.Linear(1, d_model)
        self.noisy_value_proj = nn.Linear(1, d_model)
        self.use_prior_condition = bool(configs.qsdiff_use_prior and configs.qsdiff_prior_condition)
        self.prior_residual_mode = configs.qsdiff_prior_residual_mode
        self.query_context_mode = configs.qsdiff_query_context_mode
        if self.use_prior_condition:
            self.prior_value_proj = nn.Linear(1, d_model)
        if self.prior_residual_mode == "gated_residual":
            self.prior_residual_gate = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 1),
            )
            # Start with a conservative correction around the frozen prior.
            nn.init.zeros_(self.prior_residual_gate[-1].weight)
            nn.init.constant_(self.prior_residual_gate[-1].bias, configs.qsdiff_prior_gate_init)
        self.time_encoding = TimeEncoding(d_model)
        self.time_proj = nn.Linear(d_model, d_model)
        self.var_emb = nn.Embedding(configs.enc_in, d_model)
        self.diff_step_proj = DiffusionStepEmbedding(d_model)
        if self.query_context_mode != "none":
            context_lengths = tuple(
                float(item.strip())
                for item in configs.qsdiff_query_context_lengths.split(",")
                if item.strip()
            )
            self.query_context = QueryTemporalVariableContext(
                d_model=d_model,
                mode=self.query_context_mode,
                lengths=context_lengths or (0.03, 0.10, 0.30),
                use_query_group=bool(configs.qsdiff_query_context_group),
                dropout=configs.qsdiff_query_context_dropout,
            )
        else:
            self.query_context = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.history_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.blocks = nn.ModuleList(
            [QueryDenoisingBlock(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        hist_value: Tensor,
        hist_time: Tensor,
        hist_var: Tensor,
        hist_mask: Tensor,
        query_noisy: Tensor,
        query_time: Tensor,
        query_var: Tensor,
        query_row: Tensor,
        query_mask: Tensor,
        diff_step: Tensor,
        prior_mean: Tensor | None = None,
    ) -> Tensor:
        hist_emb = (
            self.value_proj(hist_value.unsqueeze(-1))
            + self.time_proj(self.time_encoding(hist_time))
            + self.var_emb(hist_var)
        )
        hist_emb = hist_emb * hist_mask.unsqueeze(-1)
        hist_key_padding_mask = _safe_key_padding_mask(hist_mask)
        history = self.history_encoder(hist_emb, src_key_padding_mask=hist_key_padding_mask)

        step_emb = self.diff_step_proj(diff_step).unsqueeze(1)
        query_emb = (
            self.noisy_value_proj(query_noisy.unsqueeze(-1))
            + self.time_proj(self.time_encoding(query_time))
            + self.var_emb(query_var)
            + step_emb
        )
        if self.use_prior_condition and prior_mean is not None:
            query_emb = query_emb + self.prior_value_proj(prior_mean.unsqueeze(-1))
        if self.query_context is not None:
            query_emb = query_emb + self.query_context(
                hist_value=hist_value,
                hist_time=hist_time,
                hist_var=hist_var,
                hist_mask=hist_mask,
                query_time=query_time,
                query_var=query_var,
                query_row=query_row,
                query_mask=query_mask,
                var_embedding=self.var_emb,
            )
        query_emb = query_emb * query_mask.unsqueeze(-1)
        query_key_padding_mask = _safe_key_padding_mask(query_mask)

        query = query_emb
        for block in self.blocks:
            query = block(
                query=query,
                history=history,
                query_key_padding_mask=query_key_padding_mask,
                history_key_padding_mask=hist_key_padding_mask,
                query_mask=query_mask,
            )

        residual = self.output_head(query).squeeze(-1)
        if prior_mean is not None and self.prior_residual_mode == "residual":
            pred = prior_mean + residual
        elif prior_mean is not None and self.prior_residual_mode == "gated_residual":
            gate = torch.sigmoid(self.prior_residual_gate(query).squeeze(-1))
            pred = prior_mean + gate * residual
        else:
            pred = residual
        return pred * query_mask


class DynamicQueryNoiseProcess(nn.Module):
    _OLD_DYNAMIC_TYPES = {"channel_ou", "dynamic_lowrank", "dqspn"}
    _MULTISCALE_TYPES = {"multi_scale_ou", "ou_multiscale"}
    _ANNEALED_TYPES = {"annealed_ou", "ou_annealed"}
    _GAP_TYPES = {"gap_adaptive_ou", "ou_adaptive"}

    def __init__(self, configs: ExpConfigs):
        super().__init__()
        self.noise_type = configs.qsdiff_noise_type
        self.diffusion_steps = configs.qsdiff_diffusion_steps
        self.ou_gamma = configs.qsdiff_ou_gamma
        self.ou_sigma0 = configs.qsdiff_ou_sigma0
        self.ou_jitter = configs.qsdiff_ou_jitter
        self.hidden_dim = configs.qsdiff_noise_hidden_dim
        self.rank = configs.qsdiff_noise_rank
        self.dyn_weight = configs.qsdiff_noise_dyn_weight
        self.residual_weight = configs.qsdiff_noise_residual_weight
        self.iid_residual_weight = configs.qsdiff_noise_iid_residual_weight
        self.gap_weight = configs.qsdiff_noise_gap_weight
        self.scale_mixture_mode = configs.qsdiff_noise_scale_mixture_mode
        self.min_length = configs.qsdiff_noise_min_length
        self.min_scale = configs.qsdiff_noise_min_scale
        self.anneal_min_length = configs.qsdiff_noise_anneal_min_length
        self.anneal_max_length = configs.qsdiff_noise_anneal_max_length
        self.anneal_direction = configs.qsdiff_noise_anneal_direction
        self.max_cholesky_tries = configs.qsdiff_noise_cholesky_tries
        self.register_buffer("noise_alpha_bar", make_beta_schedule(self.diffusion_steps), persistent=False)

        multiscale_lengths = self._parse_float_list(configs.qsdiff_noise_multiscale_lengths, (0.03, 0.10, 0.30))
        multiscale_weights = self._parse_float_list(configs.qsdiff_noise_multiscale_weights, ())
        if len(multiscale_weights) != len(multiscale_lengths):
            multiscale_weights = tuple(1.0 for _ in multiscale_lengths)
        length_tensor = torch.tensor(multiscale_lengths, dtype=torch.float32).clamp_min(1e-6)
        weight_tensor = torch.tensor(multiscale_weights, dtype=torch.float32).clamp_min(1e-8)
        self.register_buffer("multiscale_lengths", length_tensor, persistent=False)
        initial_mixture_logits = torch.log(weight_tensor)
        if self.scale_mixture_mode == "learned_global":
            self.multiscale_logits = nn.Parameter(initial_mixture_logits)
        else:
            self.register_buffer("multiscale_logits", initial_mixture_logits, persistent=False)

        self.var_emb = nn.Embedding(configs.enc_in, self.hidden_dim)
        self.scalar_proj = nn.Linear(4, self.hidden_dim)
        self.length_head = nn.Linear(self.hidden_dim, 1)
        self.scale_head = nn.Linear(self.hidden_dim, 1)
        self.dynamic_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.rank),
        )
        if self.scale_mixture_mode == "query_adaptive":
            self.scale_mixture_step_proj = nn.Linear(1, self.hidden_dim)
            self.scale_mixture_gate = nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, len(multiscale_lengths)),
            )
            # Begin from the supplied fixed mixture and learn query-specific deviations.
            nn.init.zeros_(self.scale_mixture_step_proj.weight)
            nn.init.zeros_(self.scale_mixture_step_proj.bias)
            nn.init.zeros_(self.scale_mixture_gate[-1].weight)
            with torch.no_grad():
                self.scale_mixture_gate[-1].bias.copy_(initial_mixture_logits)
        else:
            self.scale_mixture_step_proj = None
            self.scale_mixture_gate = None
        self.channel_factor = nn.Parameter(torch.randn(configs.enc_in, self.rank) / math.sqrt(self.rank))

    @staticmethod
    def _parse_float_list(value: str, default: tuple[float, ...]) -> tuple[float, ...]:
        if value is None or str(value).strip() == "":
            return default
        parsed = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
        return parsed if parsed else default

    def _history_stats(self, flat: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        query_time = flat["query_time"]
        query_var = flat["query_var"]
        query_mask = flat["query_mask"]
        hist_time = flat["hist_time"]
        hist_var = flat["hist_var"]
        hist_mask = flat["hist_mask"]

        gaps = torch.ones_like(query_time)
        densities = torch.zeros_like(query_time)

        for b in range(query_time.shape[0]):
            hist_valid = hist_mask[b] > 0
            hist_total = hist_valid.to(query_time.dtype).sum().clamp_min(1.0)
            for j in range(query_time.shape[1]):
                if query_mask[b, j] <= 0:
                    continue
                same_var = hist_valid & hist_var[b].eq(query_var[b, j])
                if same_var.any():
                    last_time = hist_time[b, same_var].max()
                    gaps[b, j] = (query_time[b, j] - last_time).abs()
                    densities[b, j] = same_var.to(query_time.dtype).sum() / hist_total
        return gaps * query_mask, densities * query_mask

    def _query_features(self, flat: dict[str, Tensor], prior_mean: Tensor | None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if prior_mean is None:
            prior_mean = torch.zeros_like(flat["query_time"])
        gaps, densities = self._history_stats(flat)
        scalars = torch.stack(
            [
                flat["query_time"],
                prior_mean,
                gaps,
                densities,
            ],
            dim=-1,
        )
        z = torch.tanh(self.scalar_proj(scalars) + self.var_emb(flat["query_var"]))
        length = torch.nn.functional.softplus(self.length_head(z)).squeeze(-1) + self.min_length
        scale = torch.nn.functional.softplus(self.scale_head(z)).squeeze(-1) + self.min_scale
        phi = self.dynamic_proj(z)
        return z, length * flat["query_mask"], scale * flat["query_mask"], phi * flat["query_mask"].unsqueeze(-1)

    def _channel_covariance(self, dtype: torch.dtype, device: torch.device) -> Tensor:
        factors = self.channel_factor.to(device=device, dtype=dtype)
        cov = factors @ factors.transpose(0, 1) / max(self.rank, 1)
        cov = cov + torch.eye(cov.shape[0], device=device, dtype=dtype)
        return cov

    def _safe_cholesky(self, cov: Tensor) -> Tensor:
        eye = torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
        jitter = self.ou_jitter
        for _ in range(max(int(self.max_cholesky_tries), 1)):
            try:
                return torch.linalg.cholesky(cov + jitter * eye)
            except RuntimeError:
                jitter *= 10.0
        return torch.linalg.cholesky(cov + jitter * eye)

    def _sample_same_channel_ou(self, flat: dict[str, Tensor], gamma: float | Tensor) -> Tensor:
        if not torch.is_tensor(gamma):
            return sample_query_noise(
                query_value=flat["query_value"],
                query_time=flat["query_time"],
                query_var=flat["query_var"],
                query_mask=flat["query_mask"],
                noise_type="ou",
                ou_gamma=float(gamma),
                ou_sigma0=self.ou_sigma0,
                ou_jitter=self.ou_jitter,
            )

        query_time = flat["query_time"]
        query_var = flat["query_var"]
        query_mask = flat["query_mask"]
        noise = torch.zeros_like(query_time)
        gamma = gamma.to(device=query_time.device, dtype=query_time.dtype)
        for b in range(query_time.shape[0]):
            valid = query_mask[b] > 0
            if not valid.any():
                continue
            times = query_time[b, valid]
            vars_ = query_var[b, valid]
            time_dist = (times[:, None] - times[None, :]).abs()
            same_var = vars_[:, None].eq(vars_[None, :]).to(query_time.dtype)
            cov = torch.exp(-gamma[b] * time_dist) * same_var
            eye = torch.eye(times.shape[0], device=query_time.device, dtype=query_time.dtype)
            cov = cov + (self.ou_sigma0 ** 2) * eye
            eta = torch.randn(times.shape[0], device=query_time.device, dtype=query_time.dtype)
            noise[b, valid] = self._safe_cholesky(cov) @ eta
        return noise * query_mask

    def _noise_level(self, diff_step: Tensor | None, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        if diff_step is None:
            return torch.ones(batch_size, device=device, dtype=dtype)
        steps = diff_step.to(device=device).clamp(0, self.diffusion_steps)
        alpha_bar = self.noise_alpha_bar.to(device=device, dtype=dtype)[steps]
        return (1.0 - alpha_bar).clamp(0.0, 1.0)

    def _annealed_lengths(self, diff_step: Tensor | None, flat: dict[str, Tensor]) -> Tensor:
        query_time = flat["query_time"]
        level = self._noise_level(diff_step, query_time.shape[0], query_time.device, query_time.dtype)
        min_length = max(float(self.anneal_min_length), float(self.min_length))
        max_length = max(float(self.anneal_max_length), min_length)
        if self.anneal_direction == "reverse":
            return max_length - (max_length - min_length) * level
        return min_length + (max_length - min_length) * level

    def _multiscale_mixture_weights(
        self,
        flat: dict[str, Tensor],
        prior_mean: Tensor | None,
        diff_step: Tensor | None,
    ) -> Tensor:
        """Return global or per-query nonnegative weights over OU scales."""
        logits = self.multiscale_logits.to(device=flat["query_time"].device, dtype=flat["query_time"].dtype)
        if self.scale_mixture_mode != "query_adaptive":
            return torch.softmax(logits, dim=0)

        assert self.scale_mixture_gate is not None
        assert self.scale_mixture_step_proj is not None
        z, _, _, _ = self._query_features(flat, prior_mean)
        level = self._noise_level(
            diff_step,
            flat["query_time"].shape[0],
            flat["query_time"].device,
            flat["query_time"].dtype,
        ).view(-1, 1)
        step_embedding = self.scale_mixture_step_proj(level).unsqueeze(1)
        return torch.softmax(self.scale_mixture_gate(torch.tanh(z + step_embedding)), dim=-1)

    def _sample_multiscale_ou(
        self,
        flat: dict[str, Tensor],
        prior_mean: Tensor | None = None,
        diff_step: Tensor | None = None,
        annealed: bool = False,
    ) -> Tensor:
        query_time = flat["query_time"]
        query_var = flat["query_var"]
        query_mask = flat["query_mask"]
        lengths = self.multiscale_lengths.to(device=query_time.device, dtype=query_time.dtype).clamp_min(self.min_length)
        weights = self._multiscale_mixture_weights(flat, prior_mean, diff_step)
        noise = torch.zeros_like(query_time)
        if annealed:
            base_mid = lengths.median().clamp_min(self.min_length)
            step_scale = (self._annealed_lengths(diff_step, flat) / base_mid).clamp_min(self.min_length)
        else:
            step_scale = torch.ones(query_time.shape[0], device=query_time.device, dtype=query_time.dtype)

        for b in range(query_time.shape[0]):
            valid = query_mask[b] > 0
            n_valid = int(valid.sum().item())
            if n_valid == 0:
                continue
            times = query_time[b, valid]
            vars_ = query_var[b, valid]
            time_dist = (times[:, None] - times[None, :]).abs()
            same_var = vars_[:, None].eq(vars_[None, :]).to(query_time.dtype)
            cov = torch.zeros(n_valid, n_valid, device=query_time.device, dtype=query_time.dtype)
            for scale_index, length in enumerate(lengths):
                scaled_length = (length * step_scale[b]).clamp_min(self.min_length)
                kernel = torch.exp(-time_dist / scaled_length) * same_var
                if weights.dim() == 1:
                    cov = cov + weights[scale_index] * kernel
                else:
                    local_weight = weights[b, valid, scale_index].clamp_min(0.0).sqrt()
                    # D K D remains PSD for every scale; summing these terms preserves PSD.
                    cov = cov + (local_weight[:, None] * local_weight[None, :]) * kernel
            eye = torch.eye(n_valid, device=query_time.device, dtype=query_time.dtype)
            cov = cov + (self.ou_sigma0 ** 2) * eye
            eta = torch.randn(n_valid, device=query_time.device, dtype=query_time.dtype)
            noise[b, valid] = self._safe_cholesky(cov) @ eta
        return noise * query_mask

    def _sample_annealed_ou(self, flat: dict[str, Tensor], diff_step: Tensor | None) -> Tensor:
        lengths = self._annealed_lengths(diff_step, flat).clamp_min(self.min_length)
        return self._sample_same_channel_ou(flat, gamma=1.0 / lengths)

    def _rms_normalize(self, values: Tensor, mask: Tensor) -> Tensor:
        denom = ((values.square() * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp_min(1.0))
        denom = torch.sqrt(denom.clamp_min(1e-8))
        return values / denom

    def _sample_gap_adaptive_ou(self, flat: dict[str, Tensor]) -> Tensor:
        base = self._sample_same_channel_ou(flat, gamma=self.ou_gamma)
        gaps, _ = self._history_stats(flat)
        amplitude = 1.0 + max(float(self.gap_weight), 0.0) * torch.log1p(gaps.clamp_min(0.0))
        amplitude = self._rms_normalize(amplitude, flat["query_mask"])
        return base * amplitude * flat["query_mask"]

    def _sample_dynamic_lowrank_residual(self, flat: dict[str, Tensor], prior_mean: Tensor | None) -> Tensor:
        _, _, _, phi = self._query_features(flat, prior_mean)
        query_mask = flat["query_mask"]
        residual = torch.zeros_like(query_mask)
        latent = torch.randn(
            query_mask.shape[0],
            self.rank,
            device=query_mask.device,
            dtype=query_mask.dtype,
        )
        for b in range(query_mask.shape[0]):
            valid = query_mask[b] > 0
            if not valid.any():
                continue
            phi_b = phi[b, valid].to(query_mask.dtype)
            norm = phi_b.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            phi_b = phi_b * (math.sqrt(max(self.rank, 1)) / norm)
            residual[b, valid] = (phi_b @ latent[b]) / math.sqrt(max(self.rank, 1))
        return residual * query_mask

    def _sample_dynamic_diag_residual(self, flat: dict[str, Tensor], prior_mean: Tensor | None) -> Tensor:
        _, _, scale, _ = self._query_features(flat, prior_mean)
        scale = self._rms_normalize(scale, flat["query_mask"])
        return torch.randn_like(scale) * scale * flat["query_mask"]

    def _combine_with_residual(
        self,
        base: Tensor,
        residual: Tensor,
        flat: dict[str, Tensor],
        diff_step: Tensor | None,
        step_aware: bool = True,
    ) -> Tensor:
        query_mask = flat["query_mask"]
        if step_aware:
            level = self._noise_level(diff_step, base.shape[0], base.device, base.dtype).unsqueeze(-1)
        else:
            level = torch.ones(base.shape[0], 1, device=base.device, dtype=base.dtype)
        residual_weight = max(float(self.residual_weight), 0.0) * level
        iid_weight = max(float(self.iid_residual_weight), 0.0)
        combined = base + torch.sqrt(residual_weight.clamp_min(0.0)) * residual
        denom = 1.0 + residual_weight
        if iid_weight > 0:
            combined = combined + math.sqrt(iid_weight) * torch.randn_like(base) * query_mask
            denom = denom + iid_weight
        return combined / torch.sqrt(denom.clamp_min(1e-8)) * query_mask

    @torch.no_grad()
    def transport_scale_aware_noise(
        self,
        flat: dict[str, Tensor],
        source_noise: Tensor,
        source_step: Tensor,
        target_step: Tensor,
        prior_mean: Tensor | None = None,
    ) -> Tensor:
        """Transport a scale-aware Gaussian noise draw between its exact step covariances.

        Both marginals use the same latent standard normal: eps_r = L_r L_s^{-1} eps_s,
        where L_u L_u^T is the composite scale-aware covariance at step u.
        """
        if self.noise_type != "scale_aware_ou" or self.scale_mixture_mode != "fixed":
            raise ValueError(
                "covariance_trajectory consistency requires qsdiff_noise_type=scale_aware_ou "
                "with qsdiff_noise_scale_mixture_mode=fixed."
            )

        query_time = flat["query_time"]
        query_var = flat["query_var"]
        query_mask = flat["query_mask"]
        lengths = self.multiscale_lengths.to(device=query_time.device, dtype=query_time.dtype).clamp_min(self.min_length)
        mixture = torch.softmax(
            self.multiscale_logits.to(device=query_time.device, dtype=query_time.dtype),
            dim=0,
        )
        _, _, _, phi = self._query_features(flat, prior_mean)
        transported = torch.zeros_like(source_noise)
        base_mid = lengths.median().clamp_min(self.min_length)
        identity_dtype = query_time.dtype

        def covariance_for_step(batch_index: int, valid: Tensor, step: Tensor) -> Tensor:
            times = query_time[batch_index, valid]
            variables = query_var[batch_index, valid]
            time_distance = (times[:, None] - times[None, :]).abs()
            same_variable = variables[:, None].eq(variables[None, :]).to(identity_dtype)
            level = self._noise_level(
                step.view(1),
                1,
                query_time.device,
                identity_dtype,
            )[0]
            annealed_length = self._annealed_lengths(step.view(1), flat)[0]
            scale_factor = (annealed_length / base_mid).clamp_min(self.min_length)
            covariance = torch.zeros(
                times.shape[0],
                times.shape[0],
                device=query_time.device,
                dtype=identity_dtype,
            )
            for scale_index, length in enumerate(lengths):
                covariance = covariance + mixture[scale_index] * torch.exp(
                    -time_distance / (length * scale_factor).clamp_min(self.min_length)
                ) * same_variable

            eye = torch.eye(times.shape[0], device=query_time.device, dtype=identity_dtype)
            # _sample_multiscale_ou adds sigma0^2 I, then its Cholesky helper adds
            # one initial ou_jitter I.  Mirror that actual marginal covariance here.
            covariance = covariance + (self.ou_sigma0 ** 2 + self.ou_jitter) * eye
            phi_b = phi[batch_index, valid].to(identity_dtype)
            phi_b = phi_b * (
                math.sqrt(max(self.rank, 1)) / phi_b.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            )
            residual_covariance = phi_b @ phi_b.transpose(0, 1) / max(self.rank, 1)
            residual_weight = max(float(self.residual_weight), 0.0) * level
            iid_weight = max(float(self.iid_residual_weight), 0.0)
            return (covariance + residual_weight * residual_covariance + iid_weight * eye) / (
                1.0 + residual_weight + iid_weight
            )

        for batch_index in range(query_mask.shape[0]):
            valid = query_mask[batch_index] > 0
            if not valid.any():
                continue
            source_covariance = covariance_for_step(batch_index, valid, source_step[batch_index])
            target_covariance = covariance_for_step(batch_index, valid, target_step[batch_index])
            source_factor = torch.linalg.cholesky(source_covariance)
            target_factor = torch.linalg.cholesky(target_covariance)
            latent = torch.linalg.solve_triangular(
                source_factor,
                source_noise[batch_index, valid].detach().unsqueeze(-1),
                upper=False,
            ).squeeze(-1)
            transported[batch_index, valid] = target_factor @ latent
        return transported * query_mask

    def forward(
        self,
        flat: dict[str, Tensor],
        prior_mean: Tensor | None = None,
        diff_step: Tensor | None = None,
    ) -> Tensor:
        if self.noise_type in {"iid", "ou"}:
            return sample_query_noise(
                query_value=flat["query_value"],
                query_time=flat["query_time"],
                query_var=flat["query_var"],
                query_mask=flat["query_mask"],
                noise_type=self.noise_type,
                ou_gamma=self.ou_gamma,
                ou_sigma0=self.ou_sigma0,
                ou_jitter=self.ou_jitter,
            )

        if self.noise_type in self._MULTISCALE_TYPES:
            return self._sample_multiscale_ou(flat, prior_mean=prior_mean)
        if self.noise_type in self._ANNEALED_TYPES:
            return self._sample_annealed_ou(flat, diff_step)
        if self.noise_type in self._GAP_TYPES:
            return self._sample_gap_adaptive_ou(flat)
        if self.noise_type == "ou_dynamic_lowrank":
            base = self._sample_same_channel_ou(flat, gamma=self.ou_gamma)
            residual = self._sample_dynamic_lowrank_residual(flat, prior_mean)
            return self._combine_with_residual(base, residual, flat, diff_step)
        if self.noise_type == "ou_dynamic_diag":
            base = self._sample_same_channel_ou(flat, gamma=self.ou_gamma)
            residual = self._sample_dynamic_diag_residual(flat, prior_mean)
            return self._combine_with_residual(base, residual, flat, diff_step)
        if self.noise_type == "scale_aware_ou":
            base = self._sample_multiscale_ou(flat, prior_mean=prior_mean, diff_step=diff_step, annealed=True)
            # Avoid constructing query features and a low-rank residual when
            # both optional residual weights are disabled. This is exactly the
            # same distribution as _combine_with_residual(base, ..., ...), but
            # is much faster for the selected MIMIC-III configuration.
            if (
                not self.training
                and self.residual_weight <= 0.0
                and self.iid_residual_weight <= 0.0
            ):
                # Preserve the legacy sampler's RNG stream: the skipped residual
                # used exactly one [batch, rank] standard-normal draw.
                torch.randn(
                    flat["query_mask"].shape[0],
                    self.rank,
                    device=base.device,
                    dtype=base.dtype,
                )
                return base
            residual = self._sample_dynamic_lowrank_residual(flat, prior_mean)
            return self._combine_with_residual(base, residual, flat, diff_step)

        if self.noise_type not in self._OLD_DYNAMIC_TYPES:
            raise ValueError(f"Unknown QSDiff noise_type: {self.noise_type}")

        _, length, scale, phi = self._query_features(flat, prior_mean)
        query_time = flat["query_time"]
        query_var = flat["query_var"]
        query_mask = flat["query_mask"]
        channel_cov = self._channel_covariance(query_time.dtype, query_time.device)
        noise = torch.zeros_like(query_time)

        for b in range(query_time.shape[0]):
            valid = query_mask[b] > 0
            n_valid = int(valid.sum().item())
            if n_valid == 0:
                continue

            times = query_time[b, valid]
            vars_ = query_var[b, valid]
            scale_b = scale[b, valid]
            length_b = length[b, valid]
            phi_b = phi[b, valid]

            cov_inner = torch.zeros(n_valid, n_valid, device=query_time.device, dtype=query_time.dtype)
            if self.noise_type in {"channel_ou", "dqspn"}:
                time_dist = (times[:, None] - times[None, :]).abs()
                time_scale = torch.sqrt(length_b[:, None] * length_b[None, :]).clamp_min(self.min_length)
                k_time = torch.exp(-time_dist / time_scale)
                k_channel = channel_cov[vars_][:, vars_]
                cov_inner = cov_inner + k_time * k_channel

            if self.noise_type in {"dynamic_lowrank", "dqspn"}:
                k_dyn = phi_b @ phi_b.transpose(0, 1) / max(self.rank, 1)
                dyn_weight = 1.0 if self.noise_type == "dynamic_lowrank" else self.dyn_weight
                cov_inner = cov_inner + dyn_weight * k_dyn

            cov = cov_inner * (scale_b[:, None] * scale_b[None, :])
            eye = torch.eye(n_valid, device=query_time.device, dtype=query_time.dtype)
            cov = cov + (self.ou_sigma0 ** 2 + self.ou_jitter) * eye
            eta = torch.randn(n_valid, device=query_time.device, dtype=query_time.dtype)
            noise[b, valid] = self._safe_cholesky(cov) @ eta

        return noise * query_mask


def _resolve_checkpoint(path: str, itr_i: int) -> Path:
    checkpoint_path = Path(path)
    if checkpoint_path.is_dir():
        itr_file = checkpoint_path / f"iter{itr_i}" / "pytorch_model.bin"
        if itr_file.exists():
            return itr_file
        direct_file = checkpoint_path / "pytorch_model.bin"
        if direct_file.exists():
            return direct_file
        fallback_file = checkpoint_path / "iter0" / "pytorch_model.bin"
        if fallback_file.exists():
            return fallback_file
    return checkpoint_path


def _load_yaml_namespace(path: Path, prior_model_name: str) -> SimpleNamespace:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if prior_model_name == "APN":
        # PyOmniTS renamed APN's model-specific fields while retaining the architecture.
        if "apn_te_dim" in data:
            data["tpatchgnn_te_dim"] = data["apn_te_dim"]
        if "apn_npatch" in data:
            data["n_patches_list"] = [data["apn_npatch"]]
        if "apn_nlayer" in data:
            data["n_layers"] = data["apn_nlayer"]
        if "apn_attn_heads" in data:
            data["n_heads"] = data["apn_attn_heads"]
    return SimpleNamespace(**data)


def _canonical_prior_model_name(name: str) -> str:
    model_names = {
        "apn": "APN",
        "hyperimts": "HyperIMTS",
        "grafiti": "GraFITi",
        "warpformer": "Warpformer",
    }
    key = name.strip().lower()
    if key not in model_names:
        raise ValueError(
            f"Unknown QSDiff prior model '{name}'. Choose from APN, HyperIMTS, "
            "GraFITi, or Warpformer."
        )
    return model_names[key]


class PriorModelWrapper(nn.Module):
    def __init__(self, configs: ExpConfigs):
        super().__init__()
        self.model_name = _canonical_prior_model_name(configs.qsdiff_prior_model)
        checkpoint = _resolve_checkpoint(configs.qsdiff_prior_checkpoint, getattr(configs, "itr_i", 0))
        if not checkpoint.exists():
            raise FileNotFoundError(f"QSDiff prior checkpoint not found: {checkpoint}")

        config_path = checkpoint.parent / "configs.yaml"
        prior_configs = (
            _load_yaml_namespace(config_path, self.model_name)
            if config_path.exists()
            else SimpleNamespace(**vars(configs))
        )
        # Irregular datasets determine their padded history/query lengths only
        # after the current dataset is loaded. Historical checkpoint YAML files
        # were saved before that discovery and can therefore contain ``null``.
        # Reuse the current split's discovered lengths so priors such as GRU-D
        # build the same number of recurrent steps as in standalone training.
        for field in ("seq_len_max_irr", "pred_len_max_irr", "patch_len_max_irr"):
            current_value = getattr(configs, field, None)
            if current_value is not None:
                setattr(prior_configs, field, current_value)

        model_module = importlib.import_module(f"models.{self.model_name}")
        self.model = model_module.Model(prior_configs)
        state_dict = torch.load(checkpoint, map_location="cpu")
        load_result = self.model.load_state_dict(state_dict, strict=bool(configs.qsdiff_prior_strict))
        if bool(configs.qsdiff_prior_strict) and (load_result.missing_keys or load_result.unexpected_keys):
            raise RuntimeError(
                f"Failed to strictly load {self.model_name} prior: missing={load_result.missing_keys}, "
                f"unexpected={load_result.unexpected_keys}"
            )

        if bool(configs.qsdiff_freeze_prior):
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x: Tensor, x_mark: Tensor, x_mask: Tensor, y: Tensor, y_mark: Tensor, y_mask: Tensor) -> Tensor:
        # Prior models receive query locations and masks, but never future target values.
        prior_inputs = {
            "x": x,
            "x_mark": x_mark,
            "x_mask": x_mask,
            "y": torch.zeros_like(y),
            "y_mark": y_mark,
            "y_mask": y_mask,
            "exp_stage": "test",
        }
        was_training = self.model.training
        if not any(param.requires_grad for param in self.model.parameters()):
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(**prior_inputs)
        else:
            outputs = self.model(**prior_inputs)
        if was_training and any(param.requires_grad for param in self.model.parameters()):
            self.model.train()
        return outputs["pred"] * y_mask


class Model(nn.Module):
    def __init__(self, configs: ExpConfigs):
        super().__init__()
        self.configs = configs
        self.diffusion_steps = configs.qsdiff_diffusion_steps
        self.sampling_steps = configs.qsdiff_sampling_steps
        self.num_samples = configs.qsdiff_num_samples
        self.target_mode = str(configs.qsdiff_target_mode)
        self.grid_future_bins = int(configs.qsdiff_grid_future_bins) or int(
            configs.pred_len
        )
        if self.target_mode not in {"query", "grid_mask"}:
            raise ValueError(f"Unknown QSDiff target mode: {self.target_mode}")
        if self.grid_future_bins <= 0:
            raise ValueError(
                f"QSDiff grid future bins must be positive, got {self.grid_future_bins}"
            )
        self.noise_type = configs.qsdiff_noise_type
        self.ou_gamma = configs.qsdiff_ou_gamma
        self.ou_sigma0 = configs.qsdiff_ou_sigma0
        self.ou_jitter = configs.qsdiff_ou_jitter
        self.use_prior = bool(configs.qsdiff_use_prior)
        self.prior_model_name = _canonical_prior_model_name(configs.qsdiff_prior_model)
        self.prior_center = bool(configs.qsdiff_prior_center)
        self.prior_loss_weight = configs.qsdiff_prior_loss_weight
        self.prior_anchor_weight = max(float(configs.qsdiff_prior_anchor_weight), 0.0)
        self.prior_anchor_gap_scale = max(float(configs.qsdiff_prior_anchor_gap_scale), 1e-6)
        self.prior_sample_residual_scale = float(configs.qsdiff_prior_sample_residual_scale)
        self.sample_temperature = max(float(configs.qsdiff_sample_temperature), 0.0)
        self.consistency_weight = configs.qsdiff_consistency_weight
        self.consistency_keep_prob = configs.qsdiff_consistency_keep_prob
        self.consistency_mode = configs.qsdiff_consistency_mode
        self.consistency_views = max(int(configs.qsdiff_consistency_views), 1)
        self.consistency_variance_weight = configs.qsdiff_consistency_variance_weight
        self.consistency_stopgrad = bool(configs.qsdiff_consistency_stopgrad)
        self.consistency_stage_weighting = configs.qsdiff_consistency_stage_weighting
        self.consistency_stage_power = max(float(configs.qsdiff_consistency_stage_power), 0.0)
        self.consistency_stage_max_step = int(configs.qsdiff_consistency_stage_max_step)
        self.consistency_teacher_decay = min(max(float(configs.qsdiff_consistency_teacher_decay), 0.0), 0.999999)
        self.consistency_teacher_warmup_epochs = max(int(configs.qsdiff_consistency_teacher_warmup_epochs), 0)
        self.consistency_trajectory_ratio = min(max(float(configs.qsdiff_consistency_trajectory_ratio), 0.0), 1.0)
        self.consistency_trajectory_min_step = max(int(configs.qsdiff_consistency_trajectory_min_step), 1)
        self.consistency_teacher_confidence = configs.qsdiff_consistency_teacher_confidence
        self.consistency_teacher_confidence_temperature = max(
            float(configs.qsdiff_consistency_teacher_confidence_temperature), 1e-6
        )
        self.consistency_graph_weight = max(float(configs.qsdiff_consistency_graph_weight), 0.0)
        self.clean_loss_weight = max(float(configs.qsdiff_clean_loss_weight), 0.0)
        self.clean_loss_step = int(configs.qsdiff_clean_loss_step)
        self.mean_loss_weight = max(float(configs.qsdiff_mean_loss_weight), 0.0)
        self.low_noise_step_prob = min(max(float(configs.qsdiff_low_noise_step_prob), 0.0), 1.0)
        self.low_noise_step_max = int(configs.qsdiff_low_noise_step_max)
        self.denoiser = QSDiffDenoiser(configs)
        # The teacher is created only for EMA consistency, so legacy
        # configurations retain the original parameterization and behavior.
        self.ema_teacher = (
            copy.deepcopy(self.denoiser)
            if self.consistency_mode in {"ema_trajectory", "covariance_trajectory", "ema_same_step"}
            else None
        )
        if self.ema_teacher is not None:
            self.ema_teacher.requires_grad_(False)
            self.ema_teacher.eval()
            self.register_buffer("ema_teacher_ready", torch.tensor(False), persistent=True)
        else:
            self.ema_teacher_ready = None
        self.noise_process = DynamicQueryNoiseProcess(configs)
        self.prior_model = PriorModelWrapper(configs) if self.use_prior else None
        alpha_bar = make_beta_schedule(self.diffusion_steps)
        self.register_buffer("alpha_bar", alpha_bar)

    def _sample_diff_steps(self, batch_size: int, device: torch.device) -> Tensor:
        full_steps = torch.randint(
            1,
            self.diffusion_steps + 1,
            (batch_size,),
            device=device,
            dtype=torch.long,
        )
        if self.low_noise_step_prob <= 0:
            return full_steps
        low_max = max(1, min(self.diffusion_steps, self.low_noise_step_max))
        low_steps = torch.randint(
            1,
            low_max + 1,
            (batch_size,),
            device=device,
            dtype=torch.long,
        )
        use_low = torch.rand(batch_size, device=device) < self.low_noise_step_prob
        return torch.where(use_low, low_steps, full_steps)

    def train(self, mode: bool = True):
        """Keep the EMA teacher deterministic while the student is trained."""
        super().train(mode)
        if self.ema_teacher is not None:
            self.ema_teacher.eval()
        return self

    @torch.no_grad()
    def update_consistency_teacher(self) -> None:
        """Update the trajectory teacher after one optimizer step."""
        if self.ema_teacher is None or self.ema_teacher_ready is None or not bool(self.ema_teacher_ready.item()):
            return
        decay = self.consistency_teacher_decay
        for teacher_param, student_param in zip(self.ema_teacher.parameters(), self.denoiser.parameters()):
            teacher_param.mul_(decay).add_(student_param.detach(), alpha=1.0 - decay)
        # Denoiser currently uses no running-statistics buffers, but copying them
        # makes the teacher safe if normalization layers are changed later.
        for teacher_buffer, student_buffer in zip(self.ema_teacher.buffers(), self.denoiser.buffers()):
            teacher_buffer.copy_(student_buffer.detach())
        self.ema_teacher.eval()

    @torch.no_grad()
    def _activate_consistency_teacher(self, current_epoch: int) -> bool:
        """Synchronize the EMA teacher after its supervised burn-in period."""
        if self.ema_teacher is None or self.ema_teacher_ready is None:
            return False
        if current_epoch < self.consistency_teacher_warmup_epochs:
            return False
        if not bool(self.ema_teacher_ready.item()):
            self.ema_teacher.load_state_dict(self.denoiser.state_dict(), strict=True)
            self.ema_teacher_ready.fill_(True)
            self.ema_teacher.eval()
        return True

    def _predict_flat(
        self,
        flat: dict[str, Tensor],
        query_noisy: Tensor,
        diff_step: Tensor,
        prior_mean: Tensor | None = None,
    ) -> Tensor:
        return self.denoiser(
            flat["hist_value"],
            flat["hist_time"],
            flat["hist_var"],
            flat["hist_mask"],
            query_noisy,
            flat["query_time"],
            flat["query_var"],
            flat["query_row"],
            flat["query_mask"],
            diff_step,
            prior_mean,
        )

    @torch.no_grad()
    def _predict_teacher_flat(
        self,
        flat: dict[str, Tensor],
        query_noisy: Tensor,
        diff_step: Tensor,
        prior_mean: Tensor | None = None,
    ) -> Tensor:
        if self.ema_teacher is None:
            raise RuntimeError("EMA teacher prediction was requested without ema_trajectory consistency.")
        self.ema_teacher.eval()
        return self.ema_teacher(
            flat["hist_value"],
            flat["hist_time"],
            flat["hist_var"],
            flat["hist_mask"],
            query_noisy,
            flat["query_time"],
            flat["query_var"],
            flat["query_row"],
            flat["query_mask"],
            diff_step,
            prior_mean,
        )

    def _ddim_transition(
        self,
        query_noisy: Tensor,
        pred_x0: Tensor,
        source_step: Tensor,
        target_step: Tensor,
        query_mask: Tensor,
        prior_mean: Tensor | None = None,
    ) -> Tensor:
        """Move a noisy query state along the model's deterministic DDIM path."""
        alpha_bar = self.alpha_bar.to(device=query_noisy.device, dtype=query_noisy.dtype)
        alpha_source = alpha_bar[source_step].unsqueeze(-1)
        alpha_target = alpha_bar[target_step].unsqueeze(-1)
        sqrt_source = torch.sqrt(alpha_source)
        sqrt_noise_source = torch.sqrt((1.0 - alpha_source).clamp_min(1e-12))
        sqrt_target = torch.sqrt(alpha_target)
        sqrt_noise_target = torch.sqrt((1.0 - alpha_target).clamp_min(1e-12))
        if prior_mean is None:
            eps = (query_noisy - sqrt_source * pred_x0) / sqrt_noise_source
            target_noisy = sqrt_target * pred_x0 + sqrt_noise_target * eps
        else:
            eps = (query_noisy - sqrt_source * pred_x0 - (1.0 - sqrt_source) * prior_mean) / sqrt_noise_source
            target_noisy = (
                sqrt_target * pred_x0
                + (1.0 - sqrt_target) * prior_mean
                + sqrt_noise_target * eps
            )
        return target_noisy * query_mask

    def _dense_prior(
        self,
        x: Tensor,
        x_mark: Tensor,
        x_mask: Tensor,
        y: Tensor,
        y_mark: Tensor,
        y_mask: Tensor,
    ) -> Tensor:
        if self.prior_model is None:
            return torch.zeros_like(y)
        return self.prior_model(x=x, x_mark=x_mark, x_mask=x_mask, y=y, y_mark=y_mark, y_mask=y_mask)

    def _flatten_prior(self, prior_dense: Tensor, flat: dict[str, Tensor]) -> Tensor:
        prior_flat = torch.zeros_like(flat["query_value"])
        for b in range(prior_dense.shape[0]):
            valid = flat["query_mask"][b] > 0
            if valid.any():
                prior_flat[b, valid] = prior_dense[b, flat["query_row"][b, valid], flat["query_var"][b, valid]]
        return prior_flat * flat["query_mask"]

    def _sample_noise(
        self,
        flat: dict[str, Tensor],
        prior_mean: Tensor | None = None,
        diff_step: Tensor | None = None,
    ) -> Tensor:
        return self.noise_process(flat, prior_mean if self.use_prior else None, diff_step)

    def _sample_flat(self, flat: dict[str, Tensor], prior_mean: Tensor | None = None) -> Tensor:
        def denoise_fn(y_t: Tensor, diff_step: Tensor) -> Tensor:
            return self._predict_flat(flat, y_t, diff_step, prior_mean)

        initial_step = torch.full(
            (flat["query_value"].shape[0],),
            self.diffusion_steps,
            device=flat["query_value"].device,
            dtype=torch.long,
        )
        initial_noise = self._sample_noise(flat, prior_mean, initial_step)
        centered_prior = prior_mean if (self.use_prior and self.prior_center) else None
        sample = ddim_sample_x0_prediction(
            denoise_fn=denoise_fn,
            query_shape=flat["query_value"].shape,
            query_mask=flat["query_mask"],
            alpha_bar=self.alpha_bar,
            diffusion_steps=self.diffusion_steps,
            sampling_steps=self.sampling_steps,
            initial_noise=initial_noise,
            prior_mean=centered_prior,
        )
        if self.use_prior and prior_mean is not None and self.prior_sample_residual_scale != 1.0:
            sample = prior_mean + self.prior_sample_residual_scale * (sample - prior_mean)
            sample = sample * flat["query_mask"]
        return sample

    def _make_subset_mask(self, query_mask: Tensor) -> Tensor:
        keep_prob = min(max(float(self.consistency_keep_prob), 0.0), 1.0)
        subset_mask = (torch.rand_like(query_mask) < keep_prob).to(query_mask.dtype) * query_mask
        for b in range(query_mask.shape[0]):
            if query_mask[b].sum() > 0 and subset_mask[b].sum() == 0:
                valid_idx = torch.nonzero(query_mask[b] > 0, as_tuple=False).flatten()
                choice = valid_idx[torch.randint(valid_idx.numel(), (1,), device=query_mask.device)]
                subset_mask[b, choice] = 1
        return subset_mask

    def _consistency_mse(
        self,
        prediction: Tensor,
        target: Tensor,
        subset_mask: Tensor,
        diff_step: Tensor,
        reliability: Tensor | None = None,
    ) -> Tensor:
        """Match query predictions while optionally downweighting unreliable noisy stages."""
        if reliability is None and self.consistency_stage_weighting == "none" and self.consistency_stage_max_step <= 0:
            return masked_mse(prediction, target, subset_mask)

        confidence = torch.ones((diff_step.shape[0], 1), device=subset_mask.device, dtype=subset_mask.dtype)
        if self.consistency_stage_weighting == "alpha_bar":
            alpha = self.alpha_bar.to(subset_mask.device)[diff_step].to(subset_mask.dtype)
            confidence = confidence * alpha.clamp_min(0.0).pow(self.consistency_stage_power).unsqueeze(-1)
        if self.consistency_stage_max_step > 0:
            low_noise = (diff_step <= self.consistency_stage_max_step).to(subset_mask.dtype).unsqueeze(-1)
            confidence = confidence * low_noise
        if reliability is not None:
            confidence = confidence * reliability.detach().to(device=subset_mask.device, dtype=subset_mask.dtype)
        squared_error = (prediction - target).square() * subset_mask * confidence
        # Keep the unweighted valid-query denominator so high-noise samples reduce
        # the auxiliary supervision instead of merely being renormalized away.
        return squared_error.sum() / subset_mask.sum().clamp_min(1.0)

    def _multiscale_graph_consistency(
        self,
        prediction: Tensor,
        teacher_target: Tensor,
        flat: dict[str, Tensor],
        diff_step: Tensor,
        reliability: Tensor | None = None,
    ) -> Tensor:
        """Match prediction-target residuals over a same-variable multi-scale query graph."""
        query_time = flat["query_time"]
        query_var = flat["query_var"]
        query_mask = flat["query_mask"]
        time_distance = (query_time.unsqueeze(-1) - query_time.unsqueeze(-2)).abs()
        same_variable = query_var.unsqueeze(-1).eq(query_var.unsqueeze(-2)).to(query_mask.dtype)
        valid_pair = query_mask.unsqueeze(-1) * query_mask.unsqueeze(-2)
        diagonal = torch.eye(query_mask.shape[1], device=query_mask.device, dtype=query_mask.dtype).unsqueeze(0)

        lengths = self.noise_process.multiscale_lengths.to(
            device=query_time.device,
            dtype=query_time.dtype,
        ).clamp_min(self.noise_process.min_length)
        mixture = torch.softmax(
            self.noise_process.multiscale_logits.to(device=query_time.device, dtype=query_time.dtype),
            dim=0,
        ).detach()
        kernels = torch.exp(-time_distance.unsqueeze(1) / lengths.view(1, -1, 1, 1))
        adjacency = (kernels * mixture.view(1, -1, 1, 1)).sum(dim=1)
        adjacency = adjacency * same_variable * valid_pair * (1.0 - diagonal)

        stage_confidence = torch.ones(
            (diff_step.shape[0], 1, 1),
            device=query_mask.device,
            dtype=query_mask.dtype,
        )
        if self.consistency_stage_weighting == "alpha_bar":
            alpha = self.alpha_bar.to(query_mask.device)[diff_step].to(query_mask.dtype)
            stage_confidence = stage_confidence * alpha.clamp_min(0.0).pow(self.consistency_stage_power).view(-1, 1, 1)
        if self.consistency_stage_max_step > 0:
            stage_confidence = stage_confidence * (
                diff_step <= self.consistency_stage_max_step
            ).to(query_mask.dtype).view(-1, 1, 1)

        residual = prediction - teacher_target
        residual_difference = residual.unsqueeze(-1) - residual.unsqueeze(-2)
        edge_confidence = stage_confidence
        if reliability is not None:
            edge_reliability = torch.sqrt(
                (reliability.detach().unsqueeze(-1) * reliability.detach().unsqueeze(-2)).clamp_min(0.0)
            )
            edge_confidence = edge_confidence * edge_reliability.to(dtype=query_mask.dtype)
        squared_error = residual_difference.square() * adjacency * edge_confidence
        # The base graph mass stays in the denominator so unreliable teacher
        # regions attenuate the auxiliary signal instead of being renormalized.
        return squared_error.sum() / adjacency.sum().clamp_min(1.0)

    def forward(
        self,
        x: Tensor,
        x_mark: Tensor,
        x_mask: Tensor,
        y: Tensor,
        y_mark: Tensor,
        y_mask: Tensor,
        exp_stage: str = "train",
        **kwargs,
    ) -> dict[str, Tensor]:
        original_y = y
        original_y_mask = y_mask
        target_grid = None
        if self.target_mode == "grid_mask":
            target_grid = gridify_future_queries(
                y=y,
                y_mark=y_mark,
                y_mask=y_mask,
                future_bins=self.grid_future_bins,
            )
            y = target_grid["future_value"]
            y_mark = target_grid["future_time"].unsqueeze(-1)
            y_mask = target_grid["future_mask"]

        use_target_values = exp_stage in {"train", "val"}
        flat = flatten_apn_batch(
            x=x,
            x_mark=x_mark,
            x_mask=x_mask,
            y=y if use_target_values else None,
            y_mark=y_mark,
            y_mask=y_mask,
        )
        prior_dense = self._dense_prior(x, x_mark, x_mask, y, y_mark, y_mask)
        prior_mean = self._flatten_prior(prior_dense, flat) if self.use_prior else torch.zeros_like(flat["query_value"])
        centered_prior = prior_mean if (self.use_prior and self.prior_center) else None

        if exp_stage in {"train", "val"}:
            batch_size = x.shape[0]
            diff_step = self._sample_diff_steps(batch_size, x.device)
            noise = self._sample_noise(flat, prior_mean if self.use_prior else None, diff_step)
            query_noisy, _ = q_sample_x0(
                query_value=flat["query_value"],
                query_mask=flat["query_mask"],
                diff_step=diff_step,
                alpha_bar=self.alpha_bar,
                noise=noise,
                prior_mean=centered_prior,
            )
            pred_query = self._predict_flat(flat, query_noisy, diff_step, prior_mean if self.use_prior else None)
            pred = scatter_query_values(
                pred_query,
                flat["query_row"],
                flat["query_var"],
                flat["query_mask"],
                y.shape,
            )
            aux_loss = pred.new_tensor(0.0)
            if self.use_prior and self.prior_loss_weight > 0:
                prior_loss = masked_mse(prior_mean, flat["query_value"], flat["query_mask"])
                aux_loss = aux_loss + self.prior_loss_weight * prior_loss
            else:
                prior_loss = pred.new_tensor(0.0)

            if self.use_prior and self.prior_anchor_weight > 0:
                gaps, _ = self.noise_process._history_stats(flat)
                # Sparse or remote same-variable history should preserve more of the prior forecast.
                anchor_mask = (1.0 - torch.exp(-gaps / self.prior_anchor_gap_scale)) * flat["query_mask"]
                prior_anchor_loss = masked_mse(pred_query, prior_mean, anchor_mask)
                aux_loss = aux_loss + self.prior_anchor_weight * prior_anchor_loss
            else:
                prior_anchor_loss = pred.new_tensor(0.0)

            if self.clean_loss_weight > 0:
                clean_step_value = max(0, min(self.diffusion_steps, self.clean_loss_step))
                clean_step = torch.full(
                    (batch_size,),
                    clean_step_value,
                    device=x.device,
                    dtype=torch.long,
                )
                clean_noise = self._sample_noise(flat, prior_mean if self.use_prior else None, clean_step)
                clean_noisy, _ = q_sample_x0(
                    query_value=flat["query_value"],
                    query_mask=flat["query_mask"],
                    diff_step=clean_step,
                    alpha_bar=self.alpha_bar,
                    noise=clean_noise,
                    prior_mean=centered_prior,
                )
                clean_pred = self._predict_flat(
                    flat,
                    clean_noisy,
                    clean_step,
                    prior_mean if self.use_prior else None,
                )
                clean_loss = masked_mse(clean_pred, flat["query_value"], flat["query_mask"])
                aux_loss = aux_loss + self.clean_loss_weight * clean_loss
            else:
                clean_loss = pred.new_tensor(0.0)

            if self.mean_loss_weight > 0:
                mean_step = torch.zeros(batch_size, device=x.device, dtype=torch.long)
                mean_query = torch.zeros_like(flat["query_value"])
                mean_pred = self._predict_flat(
                    flat,
                    mean_query,
                    mean_step,
                    prior_mean if self.use_prior else None,
                )
                mean_loss = masked_mse(mean_pred, flat["query_value"], flat["query_mask"])
                aux_loss = aux_loss + self.mean_loss_weight * mean_loss
            else:
                mean_loss = pred.new_tensor(0.0)

            if exp_stage == "train" and self.consistency_weight > 0:
                if self.consistency_mode == "kernel_pairwise":
                    # Compare relative denoised values across same-variable query
                    # pairs.  It stays entirely in x0 space and therefore does not
                    # require transporting a scale-aware noisy state between steps.
                    consistency_loss = self._multiscale_graph_consistency(
                        pred_query,
                        flat["query_value"],
                        flat,
                        diff_step,
                    )
                elif self.consistency_mode in {"ema_trajectory", "covariance_trajectory", "ema_same_step"}:
                    # Both EMA modes retain the complete query set used at
                    # inference. The same-step variant avoids a cross-stage state;
                    # covariance mode transports an exact Gaussian latent between
                    # scale-aware marginals; DDIM mode follows the sampler path.
                    current_epoch = int(kwargs.get("current_epoch", 0))
                    active_teacher = self._activate_consistency_teacher(current_epoch)
                    if active_teacher:
                        with torch.no_grad():
                            if self.consistency_mode in {"ema_trajectory", "covariance_trajectory"}:
                                teacher_step = torch.floor(
                                    diff_step.to(dtype=torch.float32) * self.consistency_trajectory_ratio
                                ).long().clamp(min=1, max=self.diffusion_steps)
                                teacher_step = torch.minimum(teacher_step, diff_step)
                                if self.consistency_mode == "ema_trajectory":
                                    teacher_noisy = self._ddim_transition(
                                        query_noisy=query_noisy.detach(),
                                        pred_x0=pred_query.detach(),
                                        source_step=diff_step,
                                        target_step=teacher_step,
                                        query_mask=flat["query_mask"],
                                        prior_mean=centered_prior.detach() if centered_prior is not None else None,
                                    )
                                else:
                                    teacher_noise = self.noise_process.transport_scale_aware_noise(
                                        flat=flat,
                                        source_noise=noise.detach(),
                                        source_step=diff_step,
                                        target_step=teacher_step,
                                        prior_mean=prior_mean.detach() if self.use_prior else None,
                                    )
                                    teacher_noisy, _ = q_sample_x0(
                                        query_value=flat["query_value"],
                                        query_mask=flat["query_mask"],
                                        diff_step=teacher_step,
                                        alpha_bar=self.alpha_bar,
                                        noise=teacher_noise,
                                        prior_mean=centered_prior.detach() if centered_prior is not None else None,
                                    )
                                consistency_mask = flat["query_mask"] * (
                                    diff_step >= self.consistency_trajectory_min_step
                                ).to(flat["query_mask"].dtype).unsqueeze(-1)
                            else:
                                teacher_step = diff_step
                                teacher_noisy = query_noisy.detach()
                                consistency_mask = flat["query_mask"]
                            teacher_pred = self._predict_teacher_flat(
                                flat=flat,
                                query_noisy=teacher_noisy,
                                diff_step=teacher_step,
                                prior_mean=prior_mean.detach() if self.use_prior else None,
                            )
                        if self.consistency_teacher_confidence == "advantage":
                            # The teacher guides only where its detached x0 estimate
                            # is currently more accurate than the student estimate.
                            student_error = (pred_query.detach() - flat["query_value"]).square()
                            teacher_error = (teacher_pred - flat["query_value"]).square()
                            reliability = torch.sigmoid(
                                (student_error - teacher_error) / self.consistency_teacher_confidence_temperature
                            )
                        else:
                            reliability = None
                        consistency_loss = self._consistency_mse(
                            pred_query,
                            teacher_pred,
                            consistency_mask,
                            diff_step,
                            reliability=reliability,
                        )
                        if self.consistency_mode == "ema_same_step" and self.consistency_graph_weight > 0:
                            graph_loss = self._multiscale_graph_consistency(
                                pred_query,
                                teacher_pred,
                                flat,
                                diff_step,
                                reliability=reliability,
                            )
                            consistency_loss = consistency_loss + self.consistency_graph_weight * graph_loss
                    else:
                        consistency_loss = pred.new_tensor(0.0)
                else:
                    subset_mask = self._make_subset_mask(flat["query_mask"])
                    subset_flat = dict(flat)
                    subset_flat["query_mask"] = subset_mask
                    subset_query_noisy = query_noisy * subset_mask
                    subset_prior = prior_mean * subset_mask if self.use_prior else None
                    pred_subset = self._predict_flat(subset_flat, subset_query_noisy, diff_step, subset_prior)
                    if self.consistency_mode == "moments":
                        full_views = [pred_query]
                        subset_views = [pred_subset]
                        for _ in range(1, self.consistency_views):
                            view_noise = self._sample_noise(flat, prior_mean if self.use_prior else None, diff_step)
                            view_noisy, _ = q_sample_x0(
                                query_value=flat["query_value"],
                                query_mask=flat["query_mask"],
                                diff_step=diff_step,
                                alpha_bar=self.alpha_bar,
                                noise=view_noise,
                                prior_mean=centered_prior,
                            )
                            full_views.append(
                                self._predict_flat(flat, view_noisy, diff_step, prior_mean if self.use_prior else None)
                            )
                            subset_views.append(
                                self._predict_flat(subset_flat, view_noisy * subset_mask, diff_step, subset_prior)
                            )

                        full_stack = torch.stack(full_views, dim=0)
                        subset_stack = torch.stack(subset_views, dim=0)
                        if self.consistency_stopgrad:
                            full_stack = full_stack.detach()
                        full_mean = full_stack.mean(dim=0)
                        subset_mean = subset_stack.mean(dim=0)
                        full_var = full_stack.var(dim=0, unbiased=False)
                        subset_var = subset_stack.var(dim=0, unbiased=False)
                        consistency_loss = self._consistency_mse(subset_mean, full_mean, subset_mask, diff_step)
                        consistency_loss = consistency_loss + max(float(self.consistency_variance_weight), 0.0) * self._consistency_mse(
                            subset_var,
                            full_var,
                            subset_mask,
                            diff_step,
                        )
                    else:
                        consistency_target = pred_query.detach() if self.consistency_stopgrad else pred_query
                        consistency_loss = self._consistency_mse(pred_subset, consistency_target, subset_mask, diff_step)
                aux_loss = aux_loss + self.consistency_weight * consistency_loss
            else:
                consistency_loss = pred.new_tensor(0.0)

            return {
                "pred": pred,
                "true": y,
                "mask": y_mask,
                "aux_loss": aux_loss,
                "prior_loss": prior_loss.detach(),
                "prior_anchor_loss": prior_anchor_loss.detach(),
                "consistency_loss": consistency_loss.detach(),
                "clean_loss": clean_loss.detach(),
                "mean_loss": mean_loss.detach(),
            }

        samples_dense = []
        for _ in range(self.num_samples):
            sample_query = self._sample_flat(flat, prior_mean if self.use_prior else None)
            sample_dense = scatter_query_values(
                sample_query,
                flat["query_row"],
                flat["query_var"],
                flat["query_mask"],
                y.shape,
            )
            samples_dense.append(sample_dense)

        pred_samples = torch.stack(samples_dense, dim=1)
        pred = pred_samples.mean(dim=1)
        if self.sample_temperature != 1.0:
            pred_samples = pred.unsqueeze(1) + self.sample_temperature * (pred_samples - pred.unsqueeze(1))
            pred_samples = pred_samples * y_mask.unsqueeze(1)
            # Keep the exact pre-calibration ensemble mean as the point forecast.
            # This makes temperature calibration distribution-only and guarantees
            # that MAE/MSE inputs are unchanged.
        if target_grid is not None:
            batch_size, n_samples, _, n_variables = pred_samples.shape
            pred_samples = gather_grid_queries(
                pred_samples.reshape(
                    batch_size * n_samples, self.grid_future_bins, n_variables
                ),
                target_grid["query_bin"].repeat_interleave(n_samples, dim=0),
                original_y_mask.repeat_interleave(n_samples, dim=0),
            ).reshape(
                batch_size,
                n_samples,
                original_y.shape[1],
                n_variables,
            )
            pred = pred_samples.mean(dim=1)
            y = original_y
            y_mask = original_y_mask
        return {
            "pred": pred,
            "pred_samples": pred_samples,
            "true": y,
            "mask": y_mask,
        }
