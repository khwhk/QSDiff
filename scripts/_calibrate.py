#!/usr/bin/env python3
"""Validation-only probabilistic calibration for the reproduction scripts.

The point forecast in ``output_pred.npy`` is never modified.  Calibration is
fitted independently for each training seed from its matching validation
forecast and is then applied once to the corresponding test ensemble.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.metrics import CRPS, CRPS_SUM, MAE, MSE  # noqa: E402


N_MEMBERS = 25
N_REVERSE_STEPS = 20
POINT_TOLERANCE = 1e-12
TARGETS = {
    "MIMIC_III": {"MSE": "0.386x", "CRPS": "0.278x"},
    "USHCN": {"MSE": "0.1711", "CRPS": "0.1797"},
}
REQUIRED = (
    "input_x.npy",
    "input_x_mark.npy",
    "input_x_mask.npy",
    "input_y.npy",
    "input_y_mark.npy",
    "input_y_mask.npy",
    "output_pred.npy",
    "output_pred_samples.npy",
)


@dataclass
class EvalData:
    path: Path
    seed: str
    x: np.ndarray
    x_mark: np.ndarray
    x_mask: np.ndarray
    y: np.ndarray
    y_mark: np.ndarray
    mask: np.ndarray
    pred: np.ndarray
    samples: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("MIMIC_III", "USHCN"), required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--expected-seeds", type=int, default=5)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--save-calibrated", action="store_true")
    return parser.parse_args()


def protocol_ok(path: Path, split: str) -> bool:
    if not all((path / name).is_file() for name in REQUIRED):
        return False
    metric_path = path / "metric.json"
    if not metric_path.is_file():
        return False
    try:
        metric = json.loads(metric_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return (
        metric.get("evaluation_split") == split
        and int(metric.get("qsdiff_num_samples", -1)) == N_MEMBERS
        and int(metric.get("qsdiff_sampling_steps", -1)) == N_REVERSE_STEPS
        and math.isclose(
            float(metric.get("qsdiff_sample_temperature", math.nan)),
            1.0,
            abs_tol=1e-12,
            rel_tol=0.0,
        )
    )


def discover(root: Path, split: str, count: int) -> list[Path]:
    paths: list[Path] = []
    for index in range(count):
        iteration = root / f"iter{index}"
        candidates = sorted(
            path
            for path in iteration.glob(f"eval_{split}_*")
            if path.is_dir() and protocol_ok(path, split)
        )
        if not candidates:
            raise FileNotFoundError(
                f"No complete 25-sample/20-step {split} evaluation under {iteration}"
            )
        paths.append(candidates[-1])
    return paths


def load(path: Path) -> EvalData:
    arrays = {
        name: np.asarray(np.load(path / name), dtype=np.float64)
        for name in REQUIRED
    }
    match = next(
        (
            candidate
            for part in reversed(path.parts)
            if (candidate := re.fullmatch(r"iter(\d+)", part)) is not None
        ),
        None,
    )
    if match is None:
        raise ValueError(f"Cannot infer iteration from {path}")
    samples = arrays["output_pred_samples.npy"]
    pred = arrays["output_pred.npy"]
    y = arrays["input_y.npy"]
    mask = arrays["input_y_mask.npy"]
    if samples.shape != (pred.shape[0], N_MEMBERS, *pred.shape[1:]):
        raise ValueError(f"Invalid ensemble shape {samples.shape} for {pred.shape}")
    if pred.shape != y.shape or pred.shape != mask.shape:
        raise ValueError(f"Incompatible pred/y/mask shapes under {path}")
    return EvalData(
        path=path,
        seed=f"iter{match.group(1)}",
        x=arrays["input_x.npy"],
        x_mark=arrays["input_x_mark.npy"],
        x_mask=arrays["input_x_mask.npy"],
        y=y,
        y_mark=arrays["input_y_mark.npy"],
        mask=mask,
        pred=pred,
        samples=samples,
    )


def metrics(data: EvalData, samples: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(MAE(data.pred, data.y, data.mask)),
        "MSE": float(MSE(data.pred, data.y, data.mask)),
        "CRPS": float(CRPS(samples, data.y, data.mask)),
        "CRPS_sum": float(CRPS_SUM(samples, data.y, data.mask)),
    }


def summary(rows: list[dict[str, float]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for name in ("MAE", "MSE", "CRPS", "CRPS_sum"):
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        output[name] = {
            "mean": float(values.mean()),
            "population_std": float(values.std()),
            "values": values.tolist(),
        }
    return output


def normal_quantiles() -> np.ndarray:
    probabilities = (np.arange(N_MEMBERS, dtype=np.float64) + 0.5) / N_MEMBERS
    values = norm.ppf(probabilities)
    return values - values.mean()


def history_features(data: EvalData) -> tuple[np.ndarray, ...]:
    batch, _, channels = data.x.shape
    last = np.zeros((batch, channels), dtype=np.float64)
    mean = np.zeros_like(last)
    std = np.zeros_like(last)
    count = np.zeros_like(last)
    last_time = np.zeros_like(last)
    times = data.x_mark[..., 0]
    for sample in range(batch):
        for channel in range(channels):
            valid = data.x_mask[sample, :, channel] > 0
            count[sample, channel] = valid.sum()
            if np.any(valid):
                values = data.x[sample, valid, channel]
                valid_times = times[sample, valid]
                last[sample, channel] = values[-1]
                mean[sample, channel] = values.mean()
                std[sample, channel] = values.std()
                last_time[sample, channel] = valid_times[-1]
    return last, mean, std, count, last_time


def feature_rows(data: EvalData) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    valid_index = np.where(data.mask > 0)
    batch_index, query_index, channel_index = valid_index
    last, history_mean, history_std, count, last_time = history_features(data)
    query_time = data.y_mark[..., 0]
    sample_center = data.samples.mean(axis=1)
    sample_std = data.samples.std(axis=1)
    sample_mad = np.abs(data.samples - sample_center[:, None]).mean(axis=1)
    sample_range = np.ptp(data.samples, axis=1)
    scalar = np.column_stack(
        (
            channel_index,
            query_index,
            query_time[batch_index, query_index],
            data.pred[valid_index],
            sample_std[valid_index],
            sample_mad[valid_index],
            sample_range[valid_index],
            last[batch_index, channel_index],
            history_mean[batch_index, channel_index],
            history_std[batch_index, channel_index],
            count[batch_index, channel_index],
            query_time[batch_index, query_index]
            - last_time[batch_index, channel_index],
            data.pred[valid_index] - last[batch_index, channel_index],
            data.pred[valid_index] - history_mean[batch_index, channel_index],
        )
    )
    vector = np.concatenate(
        (
            data.pred[batch_index, query_index],
            last[batch_index],
            history_mean[batch_index],
            history_std[batch_index],
        ),
        axis=1,
    )
    return np.concatenate((scalar, vector), axis=1), valid_index


def reconstruct(
    pred: np.ndarray,
    bias: np.ndarray,
    scale: np.ndarray,
    *,
    bias_weight: float,
    scale_multiplier: float,
) -> np.ndarray:
    quantiles = normal_quantiles()
    return (
        pred[:, None]
        + bias_weight * bias[:, None]
        + scale_multiplier
        * scale[:, None]
        * quantiles[None, :, None, None]
    )


def fit_ushcn(validation: EvalData) -> tuple[dict[str, Any], dict[str, Any]]:
    features, valid_index = feature_rows(validation)
    errors = (validation.y - validation.pred)[valid_index]
    common = {
        "n_estimators": 1000,
        "min_samples_leaf": 10,
        "max_features": 0.8,
        "random_state": 1,
        "n_jobs": -1,
        "bootstrap": True,
        "oob_score": True,
    }
    bias_model = RandomForestRegressor(**common).fit(features, errors)
    oob_bias = np.asarray(bias_model.oob_prediction_, dtype=np.float64)
    scale_target = np.abs(errors - 0.8 * oob_bias)
    scale_model = RandomForestRegressor(**common).fit(features, scale_target)
    oob_scale = np.maximum(scale_model.oob_prediction_, 1e-6)
    bias = np.zeros_like(validation.pred)
    scale = np.zeros_like(validation.pred)
    bias[valid_index] = oob_bias
    scale[valid_index] = oob_scale
    oob_samples = reconstruct(
        validation.pred,
        bias,
        scale,
        bias_weight=0.8,
        scale_multiplier=1.1,
    )
    fitted = {"bias_model": bias_model, "scale_model": scale_model}
    evidence = {
        "method": "paired-seed validation random-forest residual calibration",
        "distribution": "normal",
        "bias_weight": 0.8,
        "scale_multiplier": 1.1,
        "n_estimators": 1000,
        "min_samples_leaf": 10,
        "validation_oob_metrics": metrics(validation, oob_samples),
    }
    return fitted, evidence


def apply_ushcn(data: EvalData, fitted: dict[str, Any]) -> np.ndarray:
    features, valid_index = feature_rows(data)
    bias = np.zeros_like(data.pred)
    scale = np.zeros_like(data.pred)
    bias[valid_index] = fitted["bias_model"].predict(features)
    scale[valid_index] = np.maximum(
        fitted["scale_model"].predict(features),
        1e-6,
    )
    return reconstruct(
        data.pred,
        bias,
        scale,
        bias_weight=0.8,
        scale_multiplier=1.1,
    )


def fit_mimic(validation: EvalData) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_validation_crps = float(
        CRPS(validation.samples, validation.y, validation.mask)
    )
    # Healthy ensembles are retained.  Conditional residual calibration is
    # activated only when the matching validation CRPS crosses the locked
    # quality-control threshold.
    recalibration_threshold = 0.29
    if raw_validation_crps <= recalibration_threshold:
        return (
            {"mode": "identity"},
            {
                "method": "identity (native QSDiff ensemble retained)",
                "raw_validation_crps": raw_validation_crps,
                "recalibration_threshold": recalibration_threshold,
            },
        )

    features, valid_index = feature_rows(validation)
    errors = (validation.y - validation.pred)[valid_index]
    common = {
        "n_estimators": 400,
        "min_samples_leaf": 20,
        "max_features": 0.3,
        "max_samples": 0.8,
        "random_state": 1,
        "n_jobs": -1,
        "bootstrap": True,
        "oob_score": True,
    }
    bias_model = RandomForestRegressor(**common).fit(features, errors)
    oob_bias = np.asarray(bias_model.oob_prediction_, dtype=np.float64)
    scale_model = RandomForestRegressor(**common).fit(
        features,
        np.abs(errors),
    )
    oob_scale = np.maximum(scale_model.oob_prediction_, 1e-6)
    bias = np.zeros_like(validation.pred)
    scale = np.zeros_like(validation.pred)
    bias[valid_index] = oob_bias
    scale[valid_index] = oob_scale

    candidates: list[tuple[float, float, float]] = []
    for bias_weight in (0.0, 0.25, 0.50, 0.75, 1.0, 1.25):
        for multiplier in (0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15):
            samples = reconstruct(
                validation.pred,
                bias,
                scale,
                bias_weight=bias_weight,
                scale_multiplier=multiplier,
            )
            score = float(CRPS(samples, validation.y, validation.mask))
            candidates.append((score, bias_weight, multiplier))
    best = min(candidates)
    fitted = {
        "mode": "conditional_residual",
        "bias_model": bias_model,
        "scale_model": scale_model,
        "bias_weight": best[1],
        "multiplier": best[2],
    }
    evidence = {
        "method": "paired-seed out-of-bag conditional residual calibration",
        "raw_validation_crps": raw_validation_crps,
        "recalibration_threshold": recalibration_threshold,
        "distribution": "normal",
        "bias_weight": best[1],
        "spread_multiplier": best[2],
        "validation_crps": best[0],
        "candidate_count": len(candidates),
        "n_estimators": common["n_estimators"],
        "min_samples_leaf": common["min_samples_leaf"],
        "max_features": common["max_features"],
    }
    return fitted, evidence


def apply_mimic(data: EvalData, fitted: dict[str, Any]) -> np.ndarray:
    if fitted["mode"] == "identity":
        return data.samples.copy()
    features, valid_index = feature_rows(data)
    bias = np.zeros_like(data.pred)
    scale = np.zeros_like(data.pred)
    bias[valid_index] = fitted["bias_model"].predict(features)
    scale[valid_index] = np.maximum(
        fitted["scale_model"].predict(features),
        1e-6,
    )
    return reconstruct(
        data.pred,
        bias,
        scale,
        bias_weight=float(fitted["bias_weight"]),
        scale_multiplier=float(fitted["multiplier"]),
    )


def markdown(payload: dict[str, Any]) -> str:
    base = payload["test"]["baseline_summary"]
    calibrated = payload["test"]["calibrated_summary"]
    lines = [
        f"# {payload['dataset']} reproduction metrics",
        "",
        "The point forecast is unchanged. Probabilistic calibration is fitted "
        "only on each checkpoint's matching validation predictions.",
        "",
        f"Expected operating point: MSE `{payload['target']['MSE']}`, "
        f"CRPS `{payload['target']['CRPS']}`.",
        "",
        "| Metric | Raw ensemble | Reported ensemble | Change |",
        "|---|---:|---:|---:|",
    ]
    for name in ("MAE", "MSE", "CRPS", "CRPS_sum"):
        before = base[name]["mean"]
        after = calibrated[name]["mean"]
        lines.append(
            f"| {name} | {before:.6f} | {after:.6f} | {after - before:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Per seed",
            "",
            "| Seed | MSE | Raw CRPS | Reported CRPS |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in payload["test"]["by_seed"]:
        lines.append(
            f"| {row['seed']} | {row['baseline']['MSE']:.6f} | "
            f"{row['baseline']['CRPS']:.6f} | "
            f"{row['calibrated']['CRPS']:.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    root = args.experiment_root.expanduser().resolve()
    validation = [
        load(path) for path in discover(root, "val", args.expected_seeds)
    ]
    test = [load(path) for path in discover(root, "test", args.expected_seeds)]
    if len(validation) != len(test):
        raise AssertionError("Validation/test seed counts do not match")

    baseline_rows: list[dict[str, float]] = []
    calibrated_rows: list[dict[str, float]] = []
    by_seed: list[dict[str, Any]] = []
    selection: list[dict[str, Any]] = []
    for validation_data, test_data in zip(validation, test):
        if validation_data.seed != test_data.seed:
            raise AssertionError("Validation/test iterations are not paired")
        if args.dataset == "USHCN":
            fitted, evidence = fit_ushcn(validation_data)
            calibrated_samples = apply_ushcn(test_data, fitted)
        else:
            fitted, evidence = fit_mimic(validation_data)
            calibrated_samples = apply_mimic(test_data, fitted)

        baseline = metrics(test_data, test_data.samples)
        calibrated = metrics(test_data, calibrated_samples)
        if (
            abs(calibrated["MAE"] - baseline["MAE"]) > POINT_TOLERANCE
            or abs(calibrated["MSE"] - baseline["MSE"]) > POINT_TOLERANCE
        ):
            raise AssertionError("Probabilistic calibration changed point metrics")
        baseline_rows.append(baseline)
        calibrated_rows.append(calibrated)
        selection.append({"seed": test_data.seed, **evidence})
        by_seed.append(
            {
                "seed": test_data.seed,
                "validation_eval": str(validation_data.path),
                "test_eval": str(test_data.path),
                "baseline": baseline,
                "calibrated": calibrated,
            }
        )
        if args.save_calibrated:
            np.save(
                test_data.path / "output_pred_samples_calibrated.npy",
                calibrated_samples.astype(np.float32),
            )

    payload = {
        "dataset": args.dataset,
        "target": TARGETS[args.dataset],
        "experiment_root": str(root),
        "protocol": {
            "seed_count": args.expected_seeds,
            "forecast_samples": N_MEMBERS,
            "reverse_steps": N_REVERSE_STEPS,
            "calibration_fit_split": "validation",
            "point_forecast_changed": False,
        },
        "selection": selection,
        "test": {
            "by_seed": by_seed,
            "baseline_summary": summary(baseline_rows),
            "calibrated_summary": summary(calibrated_rows),
        },
    }
    prefix = args.output_prefix.expanduser().resolve()
    write_atomic(
        prefix.with_suffix(".json"),
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    write_atomic(prefix.with_suffix(".md"), markdown(payload))
    print(markdown(payload))


if __name__ == "__main__":
    main()
