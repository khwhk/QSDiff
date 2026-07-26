#!/usr/bin/env bash

# Shared dataset and optimization settings. Public experiment scripts keep the
# QSDiff method parameters visible while centralizing routine engineering
# settings here.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"

run_model() {
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        printf 'CUDA_VISIBLE_DEVICES=%q %q %q' "$GPU_ID" "$PYTHON_BIN" "$ROOT_DIR/main.py"
        printf ' %q' "$@"
        printf '\n'
        return 0
    fi

    (
        cd "$ROOT_DIR"
        CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" main.py "$@"
    )
}

set_dataset_args() {
    local dataset="$1"
    case "$dataset" in
        MIMIC_III)
            DATASET_ARGS=(
                --dataset_name MIMIC_III
                --dataset_id MIMIC_III
                --dataset_root_path storage/datasets/MIMIC_III
                --collate_fn collate_fn
                --features M
                --seq_len 72
                --pred_len 3
                --enc_in 96
                --dec_in 96
                --c_out 96
            )
            ;;
        USHCN)
            DATASET_ARGS=(
                --dataset_name USHCN
                --dataset_id USHCN
                --dataset_root_path storage/datasets/USHCN
                --collate_fn collate_fn
                --features M
                --seq_len 150
                --pred_len 3
                --enc_in 5
                --dec_in 5
                --c_out 5
            )
            ;;
        *)
            echo "Unsupported dataset: $dataset (choose MIMIC_III or USHCN)." >&2
            return 2
            ;;
    esac
}

set_hyperimts_args() {
    local dataset="$1"
    case "$dataset" in
        MIMIC_III)
            HYPERIMTS_ARGS=(
                --d_model 256
                --d_ff 2048
                --n_layers 2
                --n_heads 4
                --dropout 0.05
                --batch_size 32
                --learning_rate 0.001
                --num_workers 10
            )
            ;;
        USHCN)
            HYPERIMTS_ARGS=(
                --d_model 256
                --d_ff 2048
                --n_layers 1
                --n_heads 1
                --dropout 0.05
                --batch_size 16
                --learning_rate 0.001
                --num_workers 10
            )
            ;;
    esac

    HYPERIMTS_ARGS+=(
        --loss MSE
        --lr_scheduler DelayedStepDecayLR
        --train_epochs 300
        --patience 10
        --val_interval 1
    )
}

set_qsdiff_engineering_args() {
    local dataset="$1"
    case "$dataset" in
        MIMIC_III)
            QSDIFF_ENGINEERING_ARGS=(
                --d_model 96
                --d_ff 384
                --n_layers 4
                --n_heads 4
                --dropout 0.1
                --batch_size 16
                --learning_rate 0.0007
                --num_workers 4
                --qsdiff_sample_val_interval 2
                --qsdiff_sample_val_num_samples 5
                --qsdiff_sample_val_sampling_steps 10
                --qsdiff_sample_val_max_batches 24
                --qsdiff_sample_val_seed 2024
            )
            ;;
        USHCN)
            QSDIFF_ENGINEERING_ARGS=(
                --d_model 32
                --d_ff 128
                --n_layers 3
                --n_heads 4
                --dropout 0.1
                --batch_size 32
                --learning_rate 0.001
                --num_workers 10
                --qsdiff_sample_val_interval 0
            )
            ;;
    esac

    QSDIFF_ENGINEERING_ARGS+=(
        --loss MSE
        --lr_scheduler DelayedStepDecayLR
        --train_epochs 200
        --patience 10
        --val_interval 1
    )
}
