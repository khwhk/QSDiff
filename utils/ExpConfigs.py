# Code from: https://github.com/Ladbaby/PyOmniTS
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ExpConfigs:
    '''
    dataclass for argparse typo check, making life easier

    Make sure to update this dataclass after adding new args in argparse
    '''
    @classmethod
    def get_sweep_values(cls, attr_name: str) -> Optional[list]:
        for field_info in cls.__dataclass_fields__.values():
            if field_info.name == attr_name:
                return field_info.metadata.get('sweep')
        return None

    # basic config
    checkpoints: str
    is_training: int
    model_id: str
    model_name: str
    task_name: str

    # dataset & data loader
    augmentation_ratio: int
    collate_fn: str | None
    dataset_file_name: str | None
    dataset_id: str
    dataset_name: str
    dataset_root_path: str
    dataset_subset_name: str | None
    embed: str
    features: str
    freq: str
    missing_rate: float
    target_variable_index: int
    target_variable_name: str
    train_val_loader_drop_last: int
    train_val_loader_shuffle: int

    # forecasting task
    label_len: int
    pred_len: int
    seq_len: int

    # classification task
    n_classes: int

    # GPU
    allow_tf32: int
    gpu_id: int
    gpu_ids: str | None
    use_gpu: int
    use_multi_gpu: int

    # training
    batch_size: int
    itr: int
    seed_base: int
    learning_rate: float
    grad_clip_norm: float
    loss: str
    lr_scheduler: str
    lr_scheduler_gamma: float
    n_train_stages: str
    num_workers: int
    patience: int
    pretrained_checkpoint_file_name: str | None
    pretrained_checkpoint_root_path: str | None
    retain_graph: int
    sweep: int
    skip_test: int
    train_epochs: int
    val_interval: int
    wandb: int

    # testing
    checkpoints_test: str | None
    load_checkpoints_test: int
    save_arrays: int
    save_cache_arrays: int
    test_all: int
    test_dataset_statistics: int
    test_flop: int
    test_gpu_memory: int
    test_query_keep_ratio: float
    test_query_subset_seed: int
    test_split: str
    test_train_time: int

    # model configs
    # common
    activation: str
    c_out: int
    channel_independence: int
    d_ff: int = field(metadata={"sweep": [16, 32, 64, 128, 256, 512, 2048]})
    d_layers: int = field(metadata={"sweep": [1, 2]})
    d_model: int = field(metadata={"sweep": [16, 32, 64, 128, 256, 512]})
    d_timesteps: int
    dec_in: int
    dropout: float = field(metadata={"sweep": [0.0, 0.1, 0.3, 0.4, 0.5]})
    e_layers: int = field(metadata={"sweep": [1, 2, 3, 4, 8]})
    embed_type: int
    enc_in: int
    factor: int
    output_attention: int
    hidden_layers: int
    individual: int
    kernel_size: int = field(metadata={"sweep": [2, 3, 4, 5]})
    moving_avg: int
    n_heads: int = field(metadata={"sweep": [1, 4, 8]})
    n_layers: int = field(metadata={"sweep": [1, 2, 3, 4]})
    n_patches_list: list[int]
    node_dim: int
    patch_len: int
    patch_len_list: list[int]
    patch_stride: int
    revin: int
    revin_affine: int
    scale_factor: int
    top_k: int = field(metadata={"sweep": [3, 5]})
    # Adaptor
    ts_backbone_name: str
    ts_backbone_overwrite_config_list: list[str]
    # CRU
    cru_bandwidth: int
    cru_num_basis: int
    cru_ts: float = field(metadata={"sweep": [0.2, 0.3]})
    # Informer
    informer_distil: int
    # Latent ODE
    latent_ode_classif: int
    latent_ode_gen_layers: int = field(metadata={"sweep": [2, 3]})
    latent_ode_gru_units: int = field(metadata={"sweep": [50, 100]})
    latent_ode_linear_classif: int
    latent_ode_rec_dims: int = field(metadata={"sweep": [30, 40, 100]})
    latent_ode_rec_layers: int = field(metadata={"sweep": [2, 3, 4]})
    latent_ode_units: int = field(metadata={"sweep": [50, 300, 500]})
    latent_ode_z0_encoder: str
    # Mamba
    mamba_d_conv: int
    mamba_expand: int
    # mTAN
    mtan_alpha: float = field(metadata={"sweep": [5., 100.]})
    mtan_num_ref_points: int = field(metadata={"sweep": [8, 16, 32, 64, 128]})
    # NeuralFlows
    neuralflows_flow_layers: int = field(metadata={"sweep": [1, 2, 4, 16]})
    neuralflows_flow_model: str
    neuralflows_latents: int = field(metadata={"sweep": [15, 20]})
    neuralflows_time_hidden_dim: int
    neuralflows_time_net: str
    # Nonstationary Transformer
    nonstationarytransformer_p_hidden_dims: list = field(metadata={"sweep": [[8, 8], [16, 16], [32, 32], [64, 64], [128, 128], [256, 256]]})
    nonstationarytransformer_p_hidden_layers: int = field(metadata={"sweep": [2, 4]})
    # PatchTST
    patchtst_decomposition: int
    patchtst_fc_dropout: float
    patchtst_head_dropout: float
    patchtst_padding_patch: str
    patchtst_subtract_last: int
    # PrimeNet
    primenet_pooling: str
    # ReIMTS
    reimts_pad_time_emb: int
    # ScaleFormer
    scaleformer_scales: list[int]
    # TimeMixer
    timemixer_decomp_method: str
    timemixer_down_sampling_layers: int = field(metadata={"sweep": [1, 3]})
    timemixer_down_sampling_method: str
    timemixer_use_norm: int
    # tPatchGNN
    tpatchgnn_te_dim: int
    # CSDI
    csdi_diffusion_steps: int
    csdi_beta_schedule: str
    csdi_num_samples: int
    csdi_channels: int
    csdi_layers: int
    csdi_nheads: int
    csdi_time_embedding_dim: int
    csdi_feature_embedding_dim: int
    csdi_diffusion_embedding_dim: int
    csdi_beta_start: float
    csdi_beta_end: float
    csdi_sample_batch_size: int
    csdi_history_bins: int
    csdi_future_bins: int
    csdi_max_history_bins: int
    # TMDM
    tmdm_diffusion_steps: int
    tmdm_num_samples: int
    tmdm_history_bins: int
    tmdm_future_bins: int
    tmdm_max_history_bins: int
    tmdm_prior_loss_weight: float
    tmdm_beta_start: float
    tmdm_beta_end: float
    # Diffusion-TS
    diffusionts_source_root: str
    diffusionts_history_bins: int
    diffusionts_future_bins: int
    diffusionts_max_history_bins: int
    diffusionts_diffusion_steps: int
    diffusionts_sampling_steps: int
    diffusionts_num_samples: int
    diffusionts_sample_batch_size: int
    diffusionts_n_layer_enc: int
    diffusionts_n_layer_dec: int
    diffusionts_d_model: int
    diffusionts_n_heads: int
    diffusionts_mlp_hidden_times: int
    diffusionts_eta: float
    diffusionts_attn_dropout: float
    diffusionts_kernel_size: int
    diffusionts_padding_size: int
    diffusionts_beta_schedule: str
    diffusionts_loss_type: str
    diffusionts_use_langevin: int
    diffusionts_langevin_coef: float
    diffusionts_langevin_lr: float
    # TimeGrad
    timegrad_history_bins: int
    timegrad_future_bins: int
    timegrad_max_history_bins: int
    timegrad_diffusion_steps: int
    timegrad_num_samples: int
    timegrad_sample_batch_size: int
    timegrad_rnn_hidden_size: int
    timegrad_rnn_type: str
    timegrad_num_layers: int
    timegrad_static_dim: int
    timegrad_conditioning_length: int
    timegrad_residual_layers: int
    timegrad_residual_channels: int
    timegrad_dilation_cycle: int
    timegrad_beta_end: float
    timegrad_beta_schedule: str
    timegrad_include_history_loss: int
    # LLapDiff
    llapdiff_source_root: str
    llapdiff_max_history_len: int
    llapdiff_latent_channels: int
    llapdiff_vae_latent_dim: int
    llapdiff_vae_layers: int
    llapdiff_vae_heads: int
    llapdiff_vae_ff: int
    llapdiff_vae_beta: float
    llapdiff_vae_anneal_epochs: int
    llapdiff_vae_input_dropout: float
    llapdiff_vae_noise_std: float
    llapdiff_vae_entity_conditioned: int
    llapdiff_summary_mix_dim: int
    llapdiff_summary_tv_hidden: int
    llapdiff_summary_tokens: int
    llapdiff_summary_layers: int
    llapdiff_summary_time2vec_dim: int
    llapdiff_summary_irreg_pooling: str
    llapdiff_summary_irreg_hidden: int
    llapdiff_summary_irreg_scale: float
    llapdiff_summary_t_token_mode: str
    llapdiff_summary_t_token_scale: float
    llapdiff_summary_pos_encoding: str
    llapdiff_summary_channel_balanced: int
    llapdiff_summary_loss_x: float
    llapdiff_summary_loss_v: float
    llapdiff_summary_loss_t: float
    llapdiff_summary_loss_dt: float
    llapdiff_summary_loss_obs: float
    llapdiff_hidden_dim: int
    llapdiff_num_layers: int
    llapdiff_num_heads: int
    llapdiff_laplace_k: int
    llapdiff_diffusion_steps: int
    llapdiff_sampling_steps: int
    llapdiff_num_samples: int
    llapdiff_schedule: str
    llapdiff_predict_type: str
    llapdiff_attn_dropout: float
    llapdiff_drop_cond_p: float
    llapdiff_guidance_strength: float
    llapdiff_guidance_power: float
    llapdiff_sample_eta: float
    llapdiff_loss_weight_scheme: str
    llapdiff_minsnr_gamma: float
    # ProFITi
    profiti_source_root: str
    profiti_latent_dim: int
    profiti_conditioning_layers: int
    profiti_flow_layers: int
    profiti_attention_heads: int
    profiti_marginal_training: int
    profiti_num_samples: int
    profiti_point_num_samples: int
    # Marginalization-consistent separable flows
    separable_flows_source_root: str
    separable_flows_latent_dim: int
    separable_flows_encoder_layers: int
    separable_flows_flow_layers: int
    separable_flows_attention_heads: int
    separable_flows_num_components: int
    separable_flows_num_bins: int
    separable_flows_bounds: float
    separable_flows_encoder: str
    separable_flows_marginal_training: int
    separable_flows_num_samples: int
    # QSDiff
    qsdiff_diffusion_steps: int
    qsdiff_sampling_steps: int
    qsdiff_num_samples: int
    qsdiff_sample_temperature: float
    qsdiff_eval_temperatures: str
    qsdiff_target_mode: str
    qsdiff_grid_future_bins: int
    qsdiff_noise_type: str
    qsdiff_ou_gamma: float
    qsdiff_ou_sigma0: float
    qsdiff_ou_jitter: float
    qsdiff_noise_hidden_dim: int
    qsdiff_noise_rank: int
    qsdiff_noise_dyn_weight: float
    qsdiff_noise_residual_weight: float
    qsdiff_noise_iid_residual_weight: float
    qsdiff_noise_gap_weight: float
    qsdiff_noise_multiscale_lengths: str
    qsdiff_noise_multiscale_weights: str
    qsdiff_noise_scale_mixture_mode: str
    qsdiff_noise_anneal_min_length: float
    qsdiff_noise_anneal_max_length: float
    qsdiff_noise_anneal_direction: str
    qsdiff_noise_min_length: float
    qsdiff_noise_min_scale: float
    qsdiff_noise_cholesky_tries: int
    qsdiff_use_prior: int
    qsdiff_prior_model: str
    qsdiff_prior_checkpoint: str
    qsdiff_prior_strict: int
    qsdiff_freeze_prior: int
    qsdiff_prior_center: int
    qsdiff_prior_condition: int
    qsdiff_prior_residual_mode: str
    qsdiff_prior_gate_init: float
    qsdiff_prior_loss_weight: float
    qsdiff_prior_anchor_weight: float
    qsdiff_prior_anchor_gap_scale: float
    qsdiff_prior_sample_residual_scale: float
    qsdiff_query_context_mode: str
    qsdiff_query_context_lengths: str
    qsdiff_query_context_group: int
    qsdiff_query_context_dropout: float
    qsdiff_consistency_weight: float
    qsdiff_consistency_keep_prob: float
    qsdiff_consistency_mode: str
    qsdiff_consistency_views: int
    qsdiff_consistency_variance_weight: float
    qsdiff_consistency_stopgrad: int
    qsdiff_consistency_stage_weighting: str
    qsdiff_consistency_stage_power: float
    qsdiff_consistency_stage_max_step: int
    qsdiff_consistency_teacher_decay: float
    qsdiff_consistency_teacher_warmup_epochs: int
    qsdiff_consistency_trajectory_ratio: float
    qsdiff_consistency_trajectory_min_step: int
    qsdiff_consistency_teacher_confidence: str
    qsdiff_consistency_teacher_confidence_temperature: float
    qsdiff_consistency_graph_weight: float
    qsdiff_clean_loss_weight: float
    qsdiff_clean_loss_step: int
    qsdiff_mean_loss_weight: float
    qsdiff_ema_decay: float
    qsdiff_low_noise_step_prob: float
    qsdiff_low_noise_step_max: int
    qsdiff_sample_val_interval: int
    qsdiff_sample_val_num_samples: int
    qsdiff_sample_val_sampling_steps: int
    qsdiff_sample_val_max_batches: int
    qsdiff_sample_val_seed: int
    qsdiff_crps_scale: str

    # Used to be compatible with ipython. Never used
    f: int = 1

    # args not presented in argparse
    seq_len_max_irr: int | None = None # maximum number of observations along time dimension of x, set in irregular time series datasets
    pred_len_max_irr: int | None = None # maximum number of observations along time dimension of y, set in irregular time series datasets
    patch_len_max_irr: int | None = None # maximum number of observations along time dimension in a patch of x, set in irregular time series datasets
    subfolder_train: str = "" # timestamp of training in format %Y_%m%d_%H%M
    itr_i: int = 0 # current training iteration. [0, itr-1]

class ExpConfigsTracker:
    """Wrapper that tracks which ExpConfigs attributes are accessed"""
    
    def __init__(self, configs: ExpConfigs):
        object.__setattr__(self, '_config', configs)
        object.__setattr__(self, '_accessed_attrs', set())
    
    def __getattr__(self, name: str) -> Any:
        if hasattr(self._config, name):
            self._accessed_attrs.add(name)
            return getattr(self._config, name)
        raise AttributeError(f"'{type(self._config).__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            self._accessed_attrs.add(name)
            setattr(self._config, name, value)
    
    def get_accessed_attributes(self) -> set[str]:
        """Return set of accessed attribute names"""
        return self._accessed_attrs.copy()
    
    def get_unused_attributes(self) -> set[str]:
        """Return set of unused attribute names"""
        all_attrs = {field.name for field in self._config.__dataclass_fields__.values()}
        return all_attrs - self._accessed_attrs
    
    def print_access_report(self):
        """Print a report of accessed vs unused attributes"""
        accessed = self.get_accessed_attributes()
        unused = self.get_unused_attributes()
        
        print("=== ExpConfigs Access Report ===")
        print(f"Accessed attributes ({len(accessed)}):")
        for attr in sorted(accessed):
            print(f"  ✓ {attr}")

    @property
    def __dict__(self):
        return self._config.__dict__
