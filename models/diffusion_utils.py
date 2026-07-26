import torch
from torch import Tensor


def make_beta_schedule(
    diffusion_steps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    device: torch.device | None = None,
) -> Tensor:
    """Return alpha_bar indexed from 0..diffusion_steps, with alpha_bar[0] = 1."""
    betas = torch.linspace(beta_start, beta_end, diffusion_steps, device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return torch.cat([torch.ones(1, device=device), alpha_bar], dim=0)


def _extract(values: Tensor, steps: Tensor, target: Tensor) -> Tensor:
    out = values.to(target.device)[steps].to(target.dtype)
    return out.view(steps.shape[0], *([1] * (target.dim() - 1)))


def q_sample_x0(
    query_value: Tensor,
    query_mask: Tensor,
    diff_step: Tensor,
    alpha_bar: Tensor,
    noise: Tensor | None = None,
    prior_mean: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    if noise is None:
        noise = torch.randn_like(query_value) * query_mask
    else:
        noise = noise * query_mask
    sqrt_ab = torch.sqrt(_extract(alpha_bar, diff_step, query_value))
    sqrt_one_minus_ab = torch.sqrt((1.0 - _extract(alpha_bar, diff_step, query_value)).clamp_min(1e-12))
    if prior_mean is None:
        prior_term = 0.0
    else:
        prior_term = (1.0 - sqrt_ab) * prior_mean
    noisy = sqrt_ab * query_value + prior_term + sqrt_one_minus_ab * noise
    return noisy * query_mask, noise


def sample_iid_noise(query_value: Tensor, query_mask: Tensor) -> Tensor:
    return torch.randn_like(query_value) * query_mask


def build_ou_cov(
    query_time: Tensor,
    query_var: Tensor,
    gamma: float = 1.0,
    sigma0: float = 1e-2,
    jitter: float = 1e-4,
) -> Tensor:
    """
    Build same-channel OU covariance for one valid query set.

    Sigma_ij = exp(-gamma * |t_i - t_j|) * 1[c_i = c_j]
             + sigma0^2 * 1[i = j] + jitter * 1[i = j]
    """
    time_dist = (query_time[:, None] - query_time[None, :]).abs()
    same_var = query_var[:, None].eq(query_var[None, :])
    cov = torch.exp(-float(gamma) * time_dist) * same_var.to(query_time.dtype)
    diag = torch.eye(query_time.shape[0], device=query_time.device, dtype=query_time.dtype)
    return cov + (float(sigma0) ** 2 + float(jitter)) * diag


def sample_ou_noise(
    query_time: Tensor,
    query_var: Tensor,
    query_mask: Tensor,
    gamma: float = 1.0,
    sigma0: float = 1e-2,
    jitter: float = 1e-4,
) -> Tensor:
    """Sample batch-wise same-channel OU noise in padded query-set layout."""
    batch_size, max_queries = query_mask.shape
    noise = torch.zeros(batch_size, max_queries, device=query_mask.device, dtype=query_mask.dtype)

    for b in range(batch_size):
        valid = query_mask[b] > 0
        n_valid = int(valid.sum().item())
        if n_valid == 0:
            continue

        valid_time = query_time[b, valid].float()
        valid_var = query_var[b, valid]
        cov = build_ou_cov(
            valid_time,
            valid_var,
            gamma=gamma,
            sigma0=sigma0,
            jitter=jitter,
        )
        eta = torch.randn(n_valid, device=query_mask.device, dtype=cov.dtype)
        chol = torch.linalg.cholesky(cov)
        noise[b, valid] = (chol @ eta).to(query_mask.dtype)

    return noise * query_mask


def sample_query_noise(
    query_value: Tensor,
    query_time: Tensor,
    query_var: Tensor,
    query_mask: Tensor,
    noise_type: str = "iid",
    ou_gamma: float = 1.0,
    ou_sigma0: float = 1e-2,
    ou_jitter: float = 1e-4,
) -> Tensor:
    if noise_type == "iid":
        return sample_iid_noise(query_value, query_mask)
    if noise_type == "ou":
        return sample_ou_noise(
            query_time=query_time,
            query_var=query_var,
            query_mask=query_mask,
            gamma=ou_gamma,
            sigma0=ou_sigma0,
            jitter=ou_jitter,
        )
    raise ValueError(f"Unknown QSDiff noise_type: {noise_type}")


def make_ddim_timesteps(diffusion_steps: int, sampling_steps: int, device: torch.device) -> Tensor:
    sampling_steps = min(max(int(sampling_steps), 1), int(diffusion_steps))
    steps = torch.linspace(diffusion_steps, 1, sampling_steps, device=device).round().long()
    # Keep order while removing occasional duplicates caused by rounding.
    unique_steps = []
    previous = None
    for step in steps.tolist():
        if step != previous:
            unique_steps.append(step)
            previous = step
    return torch.tensor(unique_steps, device=device, dtype=torch.long)


def ddim_sample_x0_prediction(
    denoise_fn,
    query_shape: tuple[int, int],
    query_mask: Tensor,
    alpha_bar: Tensor,
    diffusion_steps: int,
    sampling_steps: int,
    initial_noise: Tensor | None = None,
    prior_mean: Tensor | None = None,
) -> Tensor:
    """DDIM-style deterministic reverse process for an x0-prediction model."""
    device = query_mask.device
    if initial_noise is None:
        y_t = torch.randn(query_shape, device=device, dtype=query_mask.dtype) * query_mask
    else:
        y_t = initial_noise.to(device=device, dtype=query_mask.dtype) * query_mask
    if prior_mean is not None:
        prior_mean = prior_mean.to(device=device, dtype=query_mask.dtype) * query_mask
        y_t = (y_t + prior_mean) * query_mask
    timesteps = make_ddim_timesteps(diffusion_steps, sampling_steps, device)

    for idx, step in enumerate(timesteps):
        step_value = int(step.item())
        prev_step = int(timesteps[idx + 1].item()) if idx + 1 < len(timesteps) else 0
        diff_step = torch.full((query_shape[0],), step_value, device=device, dtype=torch.long)

        pred_x0 = denoise_fn(y_t, diff_step) * query_mask

        ab_t = alpha_bar.to(device=device, dtype=y_t.dtype)[step_value]
        ab_prev = alpha_bar.to(device=device, dtype=y_t.dtype)[prev_step]
        sqrt_ab_t = torch.sqrt(ab_t)
        sqrt_one_minus_ab_t = torch.sqrt((1.0 - ab_t).clamp_min(1e-12))
        sqrt_ab_prev = torch.sqrt(ab_prev)
        sqrt_one_minus_ab_prev = torch.sqrt((1.0 - ab_prev).clamp_min(1e-12))

        if prior_mean is None:
            eps = (y_t - sqrt_ab_t * pred_x0) / sqrt_one_minus_ab_t
            y_t = (sqrt_ab_prev * pred_x0 + sqrt_one_minus_ab_prev * eps) * query_mask
        else:
            eps = (y_t - sqrt_ab_t * pred_x0 - (1.0 - sqrt_ab_t) * prior_mean) / sqrt_one_minus_ab_t
            y_t = (
                sqrt_ab_prev * pred_x0
                + (1.0 - sqrt_ab_prev) * prior_mean
                + sqrt_one_minus_ab_prev * eps
            ) * query_mask

    return y_t * query_mask

