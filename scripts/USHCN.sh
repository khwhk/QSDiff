#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

set_dataset_args USHCN
QSDIFF_ARGS=(
    --d_model 32 --d_ff 128 --n_layers 3 --batch_size 32 --learning_rate 0.001 --num_workers 10
    --qsdiff_noise_type scale_aware_ou --qsdiff_noise_scale_mixture_mode fixed
    --qsdiff_noise_multiscale_lengths 0.03,0.10,0.30
    --qsdiff_noise_anneal_min_length 0.03 --qsdiff_noise_anneal_max_length 0.30
    --qsdiff_noise_residual_weight 0.1 --qsdiff_noise_rank 8 --qsdiff_consistency_weight 0
    --qsdiff_sample_val_interval 0
)
EVAL_ARGS=()
run_main
