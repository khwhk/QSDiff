"""Score the raw and calibrated ensembles from the main forecasting experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import laplace, norm, t
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.metrics import CRPS, CRPS_SUM, MAE, MSE


N_MEMBERS = 25
PROBABILITIES = (np.arange(N_MEMBERS) + 0.5) / N_MEMBERS
DATASETS = ("HumanActivity", "MIMIC_III", "P12", "USHCN")


@dataclass
class EvalData:
    path: Path
    seed: int
    pred: np.ndarray
    samples: np.ndarray
    y: np.ndarray
    mask: np.ndarray

    def array(self, name: str) -> np.ndarray:
        return np.load(self.path / f"{name}.npy", allow_pickle=False)


def load(iteration: Path, split: str, dataset: str) -> EvalData:
    paths = sorted(iteration.glob(f"eval_{split}_*/metric.json"))
    if len(paths) != 1:
        raise ValueError(f"Expected one {split} evaluation in {iteration}, found {len(paths)}")
    path = paths[0].parent
    info = json.loads(paths[0].read_text())
    if (
        info["evaluation_split"],
        info["qsdiff_num_samples"],
        info["qsdiff_sampling_steps"],
        info["qsdiff_sample_temperature"],
    ) != (split, N_MEMBERS, 20, 1.0):
        raise ValueError(f"Unexpected evaluation protocol at {path}")
    scale = {"MIMIC_III": 0.4, "HumanActivity": 0.25}.get(dataset, 1.0)
    if (
        "qsdiff_prior_sample_residual_scale" in info
        and info["qsdiff_prior_sample_residual_scale"] != scale
    ):
        raise ValueError(f"Unexpected prior residual scale at {path}")

    def read(name: str) -> np.ndarray:
        return np.asarray(np.load(path / f"{name}.npy", allow_pickle=False), dtype=np.float64)

    data = EvalData(
        path,
        int(info["evaluation_seed"]),
        read("output_pred"),
        read("output_pred_samples"),
        read("input_y"),
        read("input_y_mask"),
    )
    if (
        data.pred.shape != data.y.shape
        or data.pred.shape != data.mask.shape
        or data.samples.shape != (data.pred.shape[0], N_MEMBERS, *data.pred.shape[1:])
    ):
        raise ValueError(f"Incompatible forecast shapes at {path}")
    return data


def metrics(data: EvalData, samples: np.ndarray) -> dict[str, float]:
    return {
        "MSE": float(MSE(data.pred, data.y, data.mask)),
        "MAE": float(MAE(data.pred, data.y, data.mask)),
        "CRPS": float(CRPS(samples, data.y, data.mask)),
        "CRPS_sum": float(CRPS_SUM(samples, data.y, data.mask)),
    }


def history_features(data: EvalData) -> tuple[np.ndarray, ...]:
    x = np.asarray(data.array("input_x"), dtype=np.float64)
    x_mask = data.array("input_x_mask")
    batch, _, channels = x.shape
    last = np.zeros((batch, channels), dtype=np.float64)
    mean = np.zeros_like(last)
    std = np.zeros_like(last)
    count = np.zeros_like(last)
    last_time = np.zeros_like(last)
    times = data.array("input_x_mark")[..., 0]
    for sample in range(batch):
        for channel in range(channels):
            valid = x_mask[sample, :, channel] > 0
            count[sample, channel] = valid.sum()
            if np.any(valid):
                values = x[sample, valid, channel]
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
    query_time = data.array("input_y_mark")[..., 0]
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
            query_time[batch_index, query_index] - last_time[batch_index, channel_index],
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


def forest_ensemble(dataset: str, validation: EvalData, test: EvalData) -> np.ndarray:
    # Both forests use the matching seed's validation residuals.
    human = dataset == "HumanActivity"
    features, index = feature_rows(validation)
    errors = (validation.y - validation.pred)[index]
    options = dict(
        n_estimators=500 if human else 1000,
        min_samples_leaf=15 if human else 10,
        max_features=0.8,
        random_state=1,
        n_jobs=-1,
        bootstrap=True,
        oob_score=True,
    )
    bias = RandomForestRegressor(**options).fit(features, errors)
    weight = 0.7 if human else 0.8
    scale_target = np.abs(errors) if human else np.abs(errors - weight * bias.oob_prediction_)
    scale = RandomForestRegressor(**options).fit(features, scale_target)
    features, index = feature_rows(test)
    center = test.pred.copy()
    center[index] += weight * bias.predict(features)
    width = np.zeros_like(center)
    width[index] = (0.9 if human else 1.1) * np.maximum(scale.predict(features), 1e-6)
    quantiles = (laplace if human else norm).ppf(PROBABILITIES)
    return center[:, None] + width[:, None] * (quantiles - quantiles.mean())[None, :, None, None]


def p12_scales(validation: list[EvalData]) -> tuple[np.ndarray, np.ndarray]:
    """Minimize empirical CRPS on pooled validation residuals for each position/channel."""
    quantiles = t.ppf(PROBABILITIES, df=3)
    quantiles -= quantiles.mean()
    grid = np.arange(401, dtype=np.float64) * 0.005
    draws = grid[:, None] * quantiles
    weights = (2 * np.arange(1, N_MEMBERS + 1) - N_MEMBERS - 1) / N_MEMBERS**2
    half_pairwise_distance = draws @ weights
    scales = np.zeros(validation[0].pred.shape[1:])
    for pos, channel in np.ndindex(scales.shape):
        residuals = []
        for data in validation:
            valid = data.mask[:, pos, channel] > 0
            residuals.append(data.y[valid, pos, channel] - data.pred[valid, pos, channel])
        residuals = np.sort(np.concatenate(residuals))
        if not residuals.size:
            continue
        # Prefix sums give sum |draw - residual| without allocating an observation axis.
        prefix = np.r_[0.0, np.cumsum(residuals)]
        below = np.searchsorted(residuals, draws)
        absolute_error = (2 * below - residuals.size) * draws + prefix[-1] - 2 * prefix[below]
        scores = absolute_error.mean(axis=1) / residuals.size - half_pairwise_distance
        scales[pos, channel] = grid[np.argmin(scores)]
    channel_quantiles = np.stack([np.roll(quantiles, i) for i in range(scales.shape[1])], axis=1)
    return scales, channel_quantiles


def mimic_ensemble(data: EvalData) -> np.ndarray:
    """Apply the validation-selected OU spread floor to each patient's 25 trajectories."""
    times = np.asarray(data.array("input_y_mark")[..., 0], dtype=np.float64)
    ids = data.array("input_sample_ID")
    center = data.samples.mean(axis=1, keepdims=True)
    deviations = data.samples - center
    deviations -= deviations.mean(axis=1, keepdims=True)
    output = center + 1.5 * deviations
    half = N_MEMBERS // 2
    for row, valid in enumerate(data.mask > 0):
        count = valid.sum()
        if not count:
            continue
        variance = np.square(deviations[row][:, valid]).sum() / (N_MEMBERS * count)
        missing_variance = 0.45**2 - 1.5**2 * variance
        if missing_variance <= 0:
            continue
        positive = np.zeros_like(deviations[row, :half])
        for channel in np.flatnonzero(valid.any(axis=0)):
            indices = np.flatnonzero(valid[:, channel])
            query_times = times[row, indices]
            gaps = np.abs(query_times[:, None] - query_times[None, :])
            covariance = sum(
                w * np.exp(-gaps / length)
                for w, length in ((0.5, 0.01), (0.3, 0.03), (0.2, 0.08))
            )
            chol = np.linalg.cholesky(covariance + 1e-12 * np.eye(len(indices)))
            digest = hashlib.blake2b(digest_size=8, person=b"QSDiffVF")
            digest.update((2024).to_bytes(8, "little", signed=True))
            digest.update(np.ascontiguousarray(ids[row]).tobytes())
            digest.update(int(channel).to_bytes(4, "little"))
            rng = np.random.default_rng(int.from_bytes(digest.digest(), "little"))
            positive[:, :, channel][:, indices] = (
                rng.standard_normal((half, len(indices))) @ chol.T
            )
        # Project out the base ensemble so that the combined variance equals the chosen floor.
        paired = (deviations[row, :half] - deviations[row, half : 2 * half]) * valid
        norm_squared = np.square(paired).sum()
        if norm_squared > 1e-24:
            positive -= (positive * paired).sum() / norm_squared * paired
        rms_squared = 2 * np.square(positive).sum() / (N_MEMBERS * count)
        correction = positive * np.sqrt(missing_variance / rms_squared)
        output[row, :half] += correction
        output[row, half : 2 * half] -= correction
    return output


def summarize(rows: list[dict[str, float]]) -> dict:
    result = {}
    for metric in rows[0]:
        values = [row[metric] for row in rows]
        result[metric] = {"mean": float(np.mean(values)), "population_std": float(np.std(values))}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--expected-seeds", type=int, default=5)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--save-calibrated", action="store_true")
    args = parser.parse_args()
    iterations = [args.experiment_root / f"iter{i}" for i in range(args.expected_seeds)]
    if not iterations:
        parser.error("--expected-seeds must be positive")
    validation = []
    if args.dataset != "MIMIC_III":
        validation = [load(path, "val", args.dataset) for path in iterations]
    if args.dataset == "P12":
        scales, quantiles = p12_scales(validation)
    rows = []
    for i, path in enumerate(iterations):
        test = load(path, "test", args.dataset)
        if validation and validation[i].seed != test.seed:
            raise ValueError(f"Validation and test seeds differ at {path}")
        if args.dataset == "MIMIC_III":
            samples = mimic_ensemble(test)
        elif args.dataset == "P12":
            samples = test.pred[:, None] + scales[None, None] * quantiles[None, :, None, :]
        else:
            samples = forest_ensemble(args.dataset, validation[i], test)
        rows.append(
            {
                "seed": test.seed,
                "raw": metrics(test, test.samples),
                "calibrated": metrics(test, samples),
                "test_dir": str(test.path),
                "validation_dir": str(validation[i].path) if validation else None,
            }
        )
        if args.save_calibrated:
            np.save(test.path / "output_pred_samples_calibrated.npy", samples.astype(np.float32))
    if len({row["seed"] for row in rows}) != len(rows):
        raise ValueError("Repeated evaluation seed in summary")
    summary = {label: summarize([row[label] for row in rows]) for label in ("raw", "calibrated")}
    report = {
        "dataset": args.dataset,
        "forecast_samples": N_MEMBERS,
        "reverse_steps": 20,
        "metric_scale": "model",
        "seeds": rows,
        "summary": summary,
    }
    lines = [
        f"# {args.dataset}",
        "",
        f"Mean ± population SD over {len(rows)} seed(s).",
        "",
        "| Metric | Raw | Calibrated |",
        "|---|---:|---:|",
    ]
    for name in summary["raw"]:
        cells = [
            f"{summary[key][name]['mean']:.6f} ± {summary[key][name]['population_std']:.6f}"
            for key in ("raw", "calibrated")
        ]
        lines.append(f"| {name} | {' | '.join(cells)} |")
    lines += ["", "| Seed | MSE | MAE | CRPS | CRPS_sum |", "|---:|---:|---:|---:|---:|"]
    for row in rows:
        cells = [f"{row['calibrated'][name]:.6f}" for name in ("MSE", "MAE", "CRPS", "CRPS_sum")]
        lines.append(f"| {row['seed']} | {' | '.join(cells)} |")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    markdown = "\n".join(lines) + "\n"
    args.output_prefix.with_suffix(".md").write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
