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
                --qsdiff_prior_residual_mode gated_residual
                --qsdiff_prior_anchor_weight 0.03
                --qsdiff_prior_anchor_gap_scale 0.10
                --qsdiff_mean_loss_weight 0.30
                --qsdiff_low_noise_step_prob 0.50
                --qsdiff_low_noise_step_max 20
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
        --save_arrays 1
        --qsdiff_crps_scale original
    )
}

resolve_seed_protocol() {
    local -a seeds
    local index

    if [[ -n "${SEEDS:-}" ]]; then
        read -r -a seeds <<< "$SEEDS"
        [[ "${#seeds[@]}" -gt 0 ]] || {
            echo "SEEDS must contain at least one integer seed." >&2
            return 2
        }
        SEED_BASE="${seeds[0]}"
        ITR="${#seeds[@]}"
        for ((index = 0; index < ITR; index++)); do
            if [[ "${seeds[$index]}" != "$((SEED_BASE + index))" ]]; then
                echo "SEEDS must be contiguous because one experiment root stores iter0..iterN." >&2
                return 2
            fi
        done
    else
        SEED_BASE="${SEED_BASE:-2024}"
        ITR="${ITR:-5}"
    fi
}

latest_experiment_root() {
    local dataset="$1"
    local model_id="$2"
    local seq_len="$3"
    local pred_len="$4"
    local base="$ROOT_DIR/storage/results/$dataset/$dataset/QSDiff/$model_id/${seq_len}_${pred_len}"
    local latest

    latest="$(find "$base" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort | tail -n 1)"
    [[ -n "$latest" ]] || {
        echo "Cannot locate the completed experiment under $base." >&2
        return 1
    }
    printf '%s/%s\n' "$base" "$latest"
}

run_qsdiff_reproduction() {
    local dataset="$1"
    local model_id="$2"
    local seq_len="$3"
    local pred_len="$4"
    local experiment_root
    local iteration
    local seed
    local seed_model_id
    local seed_root
    local -a seed_roots

    resolve_seed_protocol
    if [[ "$dataset" == "MIMIC_III" ]]; then
        # The selected MIMIC result used one fresh process per seed. Keeping
        # that organization also preserves each test ensemble's RNG protocol.
        for ((iteration = 0; iteration < ITR; iteration++)); do
            seed="$((SEED_BASE + iteration))"
            seed_model_id="${model_id}_seed${seed}"
            echo "Training QSDiff on ${dataset}, seed ${seed}..."
            run_model \
                --is_training 1 \
                --model_name QSDiff \
                --model_id "$seed_model_id" \
                --checkpoints storage/results \
                --itr 1 \
                --seed_base "$seed" \
                "${DATASET_ARGS[@]}" \
                "${QSDIFF_ENGINEERING_ARGS[@]}" \
                "${QSDIFF_ARGS[@]}"

            if [[ "${DRY_RUN:-0}" == "1" ]]; then
                continue
            fi
            seed_root="$(
                latest_experiment_root \
                    "$dataset" "$seed_model_id" "$seq_len" "$pred_len"
            )"
            seed_roots+=("$seed_root/iter0")
            echo "Saving validation forecasts for ${dataset}, seed ${seed}..."
            run_model \
                --is_training 0 \
                --model_name QSDiff \
                --model_id "$seed_model_id" \
                --checkpoints storage/results \
                --checkpoints_test "$seed_root/iter0" \
                --itr 1 \
                --seed_base "$seed" \
                --test_split val \
                "${DATASET_ARGS[@]}" \
                "${QSDIFF_ENGINEERING_ARGS[@]}" \
                "${QSDIFF_ARGS[@]}"
        done
        if [[ "${DRY_RUN:-0}" == "1" ]]; then
            return 0
        fi

        # Present independently trained seeds to the reporting code through a
        # single auditable iter0..iterN manifest.
        experiment_root="$ROOT_DIR/storage/results/$dataset/$dataset/QSDiff/${model_id}_manifest/${seq_len}_${pred_len}/$(date -u +%Y_%m%d_%H%M%S)_$$"
        mkdir -p "$experiment_root"
        for ((iteration = 0; iteration < ITR; iteration++)); do
            ln -s "${seed_roots[$iteration]}" "$experiment_root/iter${iteration}"
        done
    else
        # The selected USHCN result used the framework's native contiguous
        # five-iteration protocol.
        echo "Training QSDiff on ${dataset}: ${ITR} seed(s), starting from ${SEED_BASE}..."
        run_model \
            --is_training 1 \
            --model_name QSDiff \
            --model_id "$model_id" \
            --checkpoints storage/results \
            --itr "$ITR" \
            --seed_base "$SEED_BASE" \
            "${DATASET_ARGS[@]}" \
            "${QSDIFF_ENGINEERING_ARGS[@]}" \
            "${QSDIFF_ARGS[@]}"

        if [[ "${DRY_RUN:-0}" == "1" ]]; then
            return 0
        fi
        experiment_root="$(
            latest_experiment_root \
                "$dataset" "$model_id" "$seq_len" "$pred_len"
        )"

        for ((iteration = 0; iteration < ITR; iteration++)); do
            seed="$((SEED_BASE + iteration))"
            echo "Saving validation forecasts for ${dataset}, seed ${seed}..."
            run_model \
                --is_training 0 \
                --model_name QSDiff \
                --model_id "$model_id" \
                --checkpoints storage/results \
                --checkpoints_test "$experiment_root/iter${iteration}" \
                --itr 1 \
                --seed_base "$seed" \
                --test_split val \
                "${DATASET_ARGS[@]}" \
                "${QSDIFF_ENGINEERING_ARGS[@]}" \
                "${QSDIFF_ARGS[@]}"
        done
    fi

    "$PYTHON_BIN" "$ROOT_DIR/scripts/_calibrate.py" \
        --dataset "$dataset" \
        --experiment-root "$experiment_root" \
        --expected-seeds "$ITR" \
        --output-prefix "$experiment_root/reproduction_metrics" \
        --save-calibrated

    echo "Reproduction report: $experiment_root/reproduction_metrics.md"
}
