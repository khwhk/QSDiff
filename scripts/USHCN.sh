#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

set_dataset_args USHCN
set_qsdiff_engineering_args USHCN

# This is the selected USHCN operating point (five-seed MSE 0.1711). It uses
# Scale-Aware Continuous Noise with the Low-Rank Query Residual enabled. The
# selected configuration is prior-free, so it intentionally does not consume
# the optional HyperIMTS checkpoint.
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

SEEDS="${SEEDS:-2024 2025 2026 2027 2028}"
for seed in $SEEDS; do
    echo "Training and testing QSDiff on USHCN (seed ${seed})..."
    run_model \
        --is_training 1 \
        --model_name QSDiff \
        --model_id "QSDiff_USHCN_best_seed${seed}" \
        --checkpoints storage/results \
        --itr 1 \
        --seed_base "$seed" \
        "${DATASET_ARGS[@]}" \
        "${QSDIFF_ENGINEERING_ARGS[@]}" \
        "${QSDIFF_ARGS[@]}"
done
