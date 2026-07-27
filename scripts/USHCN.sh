#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

set_dataset_args USHCN
set_qsdiff_engineering_args USHCN

# Only method-defining parameters are exposed here. This operating point uses
# Scale-Aware Continuous Noise with the Low-Rank Query Residual and no prior.
QSDIFF_ARGS=(
    --qsdiff_target_mode query
    --qsdiff_diffusion_steps 100
    --qsdiff_sampling_steps 20
    --qsdiff_num_samples 25
    --qsdiff_noise_type scale_aware_ou
    --qsdiff_noise_multiscale_lengths 0.03,0.10,0.30
    --qsdiff_noise_multiscale_weights ""
    --qsdiff_noise_scale_mixture_mode fixed
    --qsdiff_noise_anneal_min_length 0.03
    --qsdiff_noise_anneal_max_length 0.30
    --qsdiff_noise_residual_weight 0.10
    --qsdiff_noise_rank 8
    --qsdiff_use_prior 0
    --qsdiff_consistency_weight 0.0
    --qsdiff_consistency_mode x0
)

run_qsdiff_reproduction USHCN QSDiff_USHCN_reproduction 150 3
