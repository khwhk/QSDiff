#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

set_dataset_args MIMIC_III
set_qsdiff_engineering_args MIMIC_III

pointer_file="$ROOT_DIR/storage/priors/MIMIC_III/HyperIMTS.latest"
if [[ -n "${QSDIFF_PRIOR_CHECKPOINT:-}" ]]; then
    prior_checkpoint="$QSDIFF_PRIOR_CHECKPOINT"
elif [[ -f "$pointer_file" ]]; then
    read -r prior_checkpoint < "$pointer_file"
    [[ "$prior_checkpoint" = /* ]] || prior_checkpoint="$ROOT_DIR/$prior_checkpoint"
else
    echo "Missing HyperIMTS prior. Run: bash scripts/pretrain.sh MIMIC_III" >&2
    echo "Alternatively set QSDIFF_PRIOR_CHECKPOINT=/path/to/prior/run." >&2
    exit 1
fi

# The exposed parameters below correspond to QSDiff's query-space diffusion,
# scale-aware continuous noise, optional residual, prior guidance, and
# consistency objective. Routine optimizer/capacity settings live in _common.sh.
QSDIFF_ARGS=(
    --qsdiff_target_mode query
    --qsdiff_diffusion_steps 100
    --qsdiff_sampling_steps 20
    --qsdiff_num_samples 25
    --qsdiff_noise_type scale_aware_ou
    --qsdiff_noise_multiscale_lengths 0.01,0.03,0.08
    --qsdiff_noise_multiscale_weights 0.50,0.30,0.20
    --qsdiff_noise_scale_mixture_mode learned_global
    --qsdiff_noise_anneal_min_length 0.03
    --qsdiff_noise_anneal_max_length 0.30
    --qsdiff_noise_residual_weight 0.0
    --qsdiff_noise_rank 8
    --qsdiff_use_prior 1
    --qsdiff_prior_model HyperIMTS
    --qsdiff_prior_checkpoint "$prior_checkpoint"
    --qsdiff_prior_center 1
    --qsdiff_prior_condition 1
    --qsdiff_freeze_prior 1
    --qsdiff_prior_residual_mode gated_residual
    --qsdiff_prior_anchor_weight 0.03
    --qsdiff_prior_anchor_gap_scale 0.10
    --qsdiff_mean_loss_weight 0.30
    --qsdiff_low_noise_step_prob 0.50
    --qsdiff_low_noise_step_max 20
    --qsdiff_consistency_weight 0.0
    --qsdiff_consistency_mode x0
)

SEEDS="${SEEDS:-2024 2025 2026 2027 2028}"
for seed in $SEEDS; do
    echo "Training and testing QSDiff on MIMIC-III (seed ${seed})..."
    run_model \
        --is_training 1 \
        --model_name QSDiff \
        --model_id "QSDiff_MIMIC_III_best_seed${seed}" \
        --checkpoints storage/results \
        --itr 1 \
        --seed_base "$seed" \
        "${DATASET_ARGS[@]}" \
        "${QSDIFF_ENGINEERING_ARGS[@]}" \
        "${QSDIFF_ARGS[@]}"
done
