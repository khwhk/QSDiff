#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

DATASET="${1:-MIMIC_III}"
set_dataset_args "$DATASET"
set_hyperimts_args "$DATASET"

PRIOR_ID="HyperIMTS_Prior_${DATASET}"
PRIOR_SEED="${PRIOR_SEED:-2024}"

echo "Pre-training HyperIMTS prior on ${DATASET} (seed ${PRIOR_SEED})..."
run_model \
    --is_training 1 \
    --model_name HyperIMTS \
    --model_id "$PRIOR_ID" \
    --checkpoints storage/results \
    --itr 1 \
    --seed_base "$PRIOR_SEED" \
    "${DATASET_ARGS[@]}" \
    "${HYPERIMTS_ARGS[@]}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    exit 0
fi

case "$DATASET" in
    MIMIC_III) shape_dir="72_3" ;;
    USHCN) shape_dir="150_3" ;;
esac

run_base="$ROOT_DIR/storage/results/$DATASET/$DATASET/HyperIMTS/$PRIOR_ID/$shape_dir"
latest_run="$(
    find "$run_base" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' \
        | sort -nr \
        | head -n 1 \
        | cut -d' ' -f2-
)"

if [[ -z "$latest_run" || ! -f "$latest_run/iter0/pytorch_model.bin" ]]; then
    echo "HyperIMTS finished, but no usable checkpoint was found under $run_base." >&2
    exit 1
fi

pointer_dir="$ROOT_DIR/storage/priors/$DATASET"
mkdir -p "$pointer_dir"
printf '%s\n' "${latest_run#"$ROOT_DIR"/}" > "$pointer_dir/HyperIMTS.latest"

echo "Prior checkpoint registered at:"
echo "  ${latest_run#"$ROOT_DIR"/}"
echo "The MIMIC-III QSDiff script will load it automatically."
