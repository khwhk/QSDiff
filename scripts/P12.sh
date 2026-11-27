#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

set_dataset_args P12
QSDIFF_ARGS=(
    --d_model 64 --d_ff 256 --n_layers 3 --batch_size 32 --learning_rate 0.001 --num_workers 10
    --qsdiff_noise_type dynamic_lowrank --qsdiff_noise_rank 8
    --qsdiff_consistency_weight 0.1 --qsdiff_consistency_keep_prob 0.5
    --qsdiff_prior_residual_mode absolute --qsdiff_sample_val_interval 0
)
EVAL_ARGS=()
run_main
