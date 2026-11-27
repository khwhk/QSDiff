#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

set_dataset_args MIMIC_III
QSDIFF_ARGS=(
    --d_model 96 --d_ff 384 --n_layers 4 --batch_size 16 --learning_rate 0.0007 --num_workers 4
    --qsdiff_noise_type scale_aware_ou --qsdiff_noise_scale_mixture_mode learned_global
    --qsdiff_noise_multiscale_lengths 0.01,0.03,0.08 --qsdiff_noise_multiscale_weights 0.50,0.30,0.20
    --qsdiff_noise_anneal_min_length 0.03 --qsdiff_noise_anneal_max_length 0.30
    --qsdiff_noise_residual_weight 0 --qsdiff_consistency_weight 0
    --qsdiff_prior_residual_mode gated_residual --qsdiff_prior_anchor_weight 0.03
    --qsdiff_prior_anchor_gap_scale 0.10 --qsdiff_mean_loss_weight 0.30
    --qsdiff_low_noise_step_prob 0.50 --qsdiff_low_noise_step_max 20
    --qsdiff_sample_val_interval 2 --qsdiff_sample_val_num_samples 5
    --qsdiff_sample_val_sampling_steps 10 --qsdiff_sample_val_max_batches 24
)
# Residual shrinkage was applied at inference, after validation checkpoint selection.
EVAL_ARGS=(--qsdiff_prior_sample_residual_scale 0.4)
run_main
