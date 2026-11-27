#!/usr/bin/env bash
# Shared training and evaluation steps for the four forecasting scripts.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-${CUDA_VISIBLE_DEVICES-0}}"

set_dataset_args() {
    DATASET="$1"
    case "$DATASET" in
        HumanActivity) SEQ_LEN=3000; PRED_LEN=300; CHANNELS=12 ;;
        MIMIC_III)     SEQ_LEN=72;   PRED_LEN=3;   CHANNELS=96 ;;
        P12)           SEQ_LEN=36;   PRED_LEN=3;   CHANNELS=36 ;;
        USHCN)         SEQ_LEN=150;  PRED_LEN=3;   CHANNELS=5 ;;
        *) echo "Unknown dataset: $DATASET" >&2; return 2 ;;
    esac
    DATASET_ARGS=(
        --dataset_name "$DATASET" --dataset_id "$DATASET"
        --dataset_root_path "${DATASET_ROOT:-storage/datasets}/$DATASET"
        --collate_fn collate_fn --features M --seq_len "$SEQ_LEN" --pred_len "$PRED_LEN"
        --enc_in "$CHANNELS" --dec_in "$CHANNELS" --c_out "$CHANNELS"
    )
    SEED_BASE="${SEED_BASE:-2024}"
    ITR="${ITR:-5}"
    if [[ -n "${SEEDS:-}" ]]; then
        local -a seeds
        local index
        read -r -a seeds <<< "$SEEDS"
        if (( ${#seeds[@]} == 0 )); then echo "SEEDS is empty." >&2; return 2; fi
        SEED_BASE="${seeds[0]}"
        ITR="${#seeds[@]}"
        for ((index=0; index<ITR; index++)); do
            [[ "${seeds[index]}" =~ ^(0|[1-9][0-9]*)$ && "${seeds[index]}" == "$((SEED_BASE + index))" ]] || {
                echo "SEEDS must be consecutive nonnegative integers." >&2; return 2;
            }
        done
    fi
    [[ "$ITR" =~ ^[1-9][0-9]*$ && "$SEED_BASE" =~ ^(0|[1-9][0-9]*)$ ]] || {
        echo "ITR must be positive and SEED_BASE must be a nonnegative integer." >&2; return 2;
    }
}

run_command() {
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        printf '%q ' "$@"; printf '\n'
    else
        "$@"
    fi
}

new_run() {
    local parent="$ROOT_DIR/storage/reproduction/$DATASET/$1"
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        printf '%s/run-preview\n' "$parent"
    else
        mkdir -p "$parent"
        mktemp -d "$parent/run-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX"
    fi
}

# main.py creates one timestamp under this invocation's private checkpoint directory.
trained_group() {
    local parent="$1/$DATASET/$DATASET/$2/$3/${SEQ_LEN}_${PRED_LEN}"
    if [[ "${DRY_RUN:-0}" == 1 ]]; then printf '%s/timestamp\n' "$parent"; return; fi
    local -a groups=("$parent"/*)
    [[ "${#groups[@]}" == 1 && -f "${groups[0]}/iter0/pytorch_model.bin" ]] || {
        echo "Missing completed training run under $parent" >&2; return 1;
    }
    printf '%s\n' "${groups[0]}"
}

prior_protocol() {
    PRIOR_BASE="$SEED_BASE"
    PRIOR_COUNT="$ITR"
    case "$DATASET" in
        HumanActivity) PRIOR_ARGS=(--d_model 128 --n_heads 1 --n_layers 3) ;;
        MIMIC_III)
            PRIOR_ARGS=(--d_model 256 --n_heads 4 --n_layers 2)
            PRIOR_COUNT=1
            PRIOR_BASE="${PRIOR_SEED:-2024}"
            ;;
        P12) PRIOR_ARGS=(--d_model 256 --n_heads 8 --n_layers 1) ;;
        *) echo "$DATASET runs directly without a prior." >&2; return 2 ;;
    esac
    PRIOR_POINTER="$ROOT_DIR/storage/priors/$DATASET/HyperIMTS_${PRIOR_BASE}_${PRIOR_COUNT}.latest"
}

train_prior() {
    local output pointer_tmp index
    output="$(new_run prior)"
    run_command "$PYTHON_BIN" main.py --is_training 1 --model_name HyperIMTS --model_id HyperIMTS \
        --checkpoints "$output" --seed_base "$PRIOR_BASE" --itr "$PRIOR_COUNT" --skip_test 1 \
        "${DATASET_ARGS[@]}" "${PRIOR_ARGS[@]}" --d_ff 2048 --dropout 0.05 \
        --batch_size 32 --learning_rate 0.001 --num_workers 10 --loss MSE \
        --lr_scheduler DelayedStepDecayLR --train_epochs 300 --patience 10 --val_interval 1
    PRIOR_ROOT="$(trained_group "$output" HyperIMTS HyperIMTS)"
    if [[ "${DRY_RUN:-0}" != 1 ]]; then
        for ((index=0; index<PRIOR_COUNT; index++)); do
            [[ -f "$PRIOR_ROOT/iter$index/pytorch_model.bin" ]] || {
                echo "Missing prior checkpoint at $PRIOR_ROOT/iter$index" >&2; return 1;
            }
        done
        mkdir -p "$(dirname "$PRIOR_POINTER")"
        pointer_tmp="$(mktemp "${PRIOR_POINTER}.XXXXXX")"
        printf '%s\n' "${PRIOR_ROOT#"$ROOT_DIR"/}" > "$pointer_tmp"
        mv -f "$pointer_tmp" "$PRIOR_POINTER"
    fi
}

resolve_prior() {
    prior_protocol
    if [[ -n "${QSDIFF_PRIOR_CHECKPOINT:-}" ]]; then
        PRIOR_ROOT="$QSDIFF_PRIOR_CHECKPOINT"
    elif [[ -s "$PRIOR_POINTER" ]]; then
        read -r PRIOR_ROOT < "$PRIOR_POINTER"
    else
        train_prior
    fi
    if [[ "$DATASET" == MIMIC_III && -d "$PRIOR_ROOT/iter0" ]]; then
        PRIOR_ROOT="$PRIOR_ROOT/iter0"
    fi
    if [[ "${DRY_RUN:-0}" != 1 && ! -e "$PRIOR_ROOT" ]]; then
        echo "Missing HyperIMTS checkpoint: $PRIOR_ROOT" >&2; return 1
    fi
    PRIOR_FLAGS=(--qsdiff_use_prior 1 --qsdiff_prior_model HyperIMTS
        --qsdiff_prior_checkpoint "$PRIOR_ROOT" --qsdiff_freeze_prior 1
        --qsdiff_prior_center 1 --qsdiff_prior_condition 1)
}

run_main() {
    local output trained index seed split checkpoint source
    local -a splits=(val test) common eval_prior sources=()
    PRIOR_FLAGS=(--qsdiff_use_prior 0)
    if [[ "$DATASET" != USHCN ]]; then resolve_prior; fi
    if [[ "$DATASET" == MIMIC_III ]]; then splits=(test); fi
    output="$(new_run forecast)"
    common=(--model_name QSDiff --model_id "QSDiff_${DATASET}" "${DATASET_ARGS[@]}"
        --loss MSE --lr_scheduler DelayedStepDecayLR --train_epochs 200 --patience 10 --val_interval 1
        --dropout 0.1 --n_heads 4 --qsdiff_target_mode query --qsdiff_diffusion_steps 100
        --qsdiff_sampling_steps 20 --qsdiff_num_samples 25 --qsdiff_crps_scale model
        "${QSDIFF_ARGS[@]}")
    if [[ "$DATASET" == MIMIC_III && -z "${QSDIFF_CHECKPOINT:-}" ]]; then
        # The learned-mixture checkpoints used independent processes and the same frozen prior.
        for ((index=0; index<ITR; index++)); do
            seed=$((SEED_BASE + index))
            run_command "$PYTHON_BIN" main.py "${common[@]}" "${PRIOR_FLAGS[@]}" \
                --checkpoints "$output/seed$seed" --is_training 1 --itr 1 --seed_base "$seed" --skip_test 1
            trained="$(trained_group "$output/seed$seed" QSDiff "QSDiff_${DATASET}")"
            sources+=("$trained/iter0")
        done
    else
        trained="${QSDIFF_CHECKPOINT:-}"
        if [[ -z "$trained" ]]; then
            run_command "$PYTHON_BIN" main.py "${common[@]}" "${PRIOR_FLAGS[@]}" \
                --checkpoints "$output" --is_training 1 --itr "$ITR" --seed_base "$SEED_BASE" --skip_test 1
            trained="$(trained_group "$output" QSDiff "QSDiff_${DATASET}")"
        fi
        for ((index=0; index<ITR; index++)); do sources+=("$trained/iter$index"); done
    fi
    for ((index=0; index<ITR; index++)); do
        seed=$((SEED_BASE + index))
        checkpoint="$output/forecasts/iter$index"
        if [[ "${DRY_RUN:-0}" != 1 ]]; then
            source="${sources[index]}"
            [[ -f "$source/pytorch_model.bin" && -f "$source/configs.yaml" ]] || {
                echo "Missing checkpoint or configs.yaml at $source" >&2; return 1;
            }
            mkdir -p "$checkpoint"
            ln -s "$(realpath "$source/pytorch_model.bin")" "$checkpoint/pytorch_model.bin"
            ln -s "$(realpath "$source/configs.yaml")" "$checkpoint/configs.yaml"
        fi
        eval_prior=("${PRIOR_FLAGS[@]}")
        if [[ "$DATASET" != USHCN && -d "$PRIOR_ROOT/iter$index" ]]; then
            eval_prior+=(--qsdiff_prior_checkpoint "$PRIOR_ROOT/iter$index")
        fi
        for split in "${splits[@]}"; do
            run_command "$PYTHON_BIN" main.py "${common[@]}" "${eval_prior[@]}" "${EVAL_ARGS[@]}" \
                --is_training 0 --itr 1 --seed_base "$seed" --num_workers 2 \
                --checkpoints_test "$checkpoint" --test_split "$split" --save_arrays 1
        done
    done
    run_command "$PYTHON_BIN" scripts/_calibrate.py --dataset "$DATASET" \
        --experiment-root "$output/forecasts" --expected-seeds "$ITR" \
        --output-prefix "$output/reproduction_metrics" --save-calibrated
    echo "Results: $output/reproduction_metrics.md"
}
