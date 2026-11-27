#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

set_dataset_args HumanActivity
QSDIFF_ARGS=(
    --d_model 32 --d_ff 128 --n_layers 1 --batch_size 2 --learning_rate 0.0003 --num_workers 10
    --qsdiff_noise_type ou --qsdiff_consistency_weight 0.1 --qsdiff_consistency_keep_prob 0.5
    --qsdiff_prior_residual_mode gated_residual --qsdiff_prior_gate_init -2.0
    --qsdiff_prior_anchor_weight 0.03 --qsdiff_prior_anchor_gap_scale 0.03
    --qsdiff_prior_sample_residual_scale 0.25 --qsdiff_mean_loss_weight 0.35
    --qsdiff_low_noise_step_prob 0.25 --qsdiff_low_noise_step_max 20
    --qsdiff_sample_val_interval 2 --qsdiff_sample_val_num_samples 10
    --qsdiff_sample_val_sampling_steps 10 --qsdiff_sample_val_max_batches 64
)
EVAL_ARGS=()
run_main
