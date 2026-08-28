"""Aggregate common numeric evaluation histories across arbitrary training runs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import numpy as np


T_CRITICAL_95 = (
    0.0,
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262,
    2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101,
    2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052,
    2.048, 2.045, 2.042,
)
CI_METHOD = "two-sided 95% Student t interval; df 1-30 table, df > 30 uses 1.96"


def t_critical_95(degrees_of_freedom: int) -> float:
    if degrees_of_freedom < 1:
        raise ValueError("degrees_of_freedom must be positive")
    if degrees_of_freedom <= 30:
        return T_CRITICAL_95[degrees_of_freedom]
    return 1.96


def summarize_values(values: list[float]) -> dict[str, float | int]:
    if len(values) < 2:
        raise ValueError("at least two values are required for multi-run statistics")
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("aggregation values must be finite")
    mean = float(array.mean())
    std = float(array.std(ddof=1))
    sem = std / math.sqrt(len(array))
    half_width = t_critical_95(len(array) - 1) * sem
    return {
        "n": len(array),
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_lower": mean - half_width,
        "ci95_upper": mean + half_width,
    }


def read_history(run_dir: Path) -> tuple[list[str], dict[int, dict[str, str]]]:
    path = run_dir / "evaluation_history.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "sampled_steps" not in reader.fieldnames:
            raise RuntimeError(f"missing sampled_steps column: {path}")
        rows = {
            int(row["sampled_steps"]): {
                key: value for key, value in row.items() if key != "sampled_steps"
            }
            for row in reader
        }
        return [key for key in reader.fieldnames if key != "sampled_steps"], rows


def _finite_float(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError
    return number


def aggregate_training_histories(
    run_dirs: list[Path], output_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(run_dirs) < 2:
        raise ValueError("at least two run directories are required")
    loaded = [read_history(path) for path in run_dirs]
    columns = [set(item[0]) for item in loaded]
    histories = [item[1] for item in loaded]
    common_steps = sorted(set.intersection(*(set(history) for history in histories)))
    if not common_steps:
        raise RuntimeError("training runs have no common sampled_steps")
    common_columns = set.intersection(*columns)
    numeric_metrics = []
    for metric in sorted(common_columns):
        try:
            for history in histories:
                for step in common_steps:
                    _finite_float(history[step][metric])
        except (KeyError, TypeError, ValueError):
            continue
        numeric_metrics.append(metric)
    if not numeric_metrics:
        raise RuntimeError("training runs have no common numeric metrics")
    all_columns = set.union(*columns)
    skipped_columns = sorted(all_columns - set(numeric_metrics))

    summary_rows: list[dict[str, Any]] = []
    for step in common_steps:
        for metric in numeric_metrics:
            values = [
                _finite_float(history[step][metric]) for history in histories
            ]
            summary_rows.append({
                "sampled_steps": step,
                "metric": metric,
                **summarize_values(values),
            })
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "training_curve_summary.csv", summary_rows)
    manifest = {
        "run_dirs": [str(path.resolve()) for path in run_dirs],
        "number_of_runs": len(run_dirs),
        "common_sampled_steps": common_steps,
        "aggregated_metrics": numeric_metrics,
        "skipped_columns": skipped_columns,
        "ci_method": CI_METHOD,
    }
    (output_dir / "aggregation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    _write_legacy_outputs(output_dir, summary_rows, numeric_metrics)
    return summary_rows, manifest


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_legacy_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    metrics: list[str],
) -> None:
    by_key = {(row["sampled_steps"], row["metric"]): row for row in rows}
    steps = sorted({int(row["sampled_steps"]) for row in rows})
    if {"average_return", "win_rate"} <= set(metrics):
        figure8 = []
        for step in steps:
            return_row = by_key[(step, "average_return")]
            win_row = by_key[(step, "win_rate")]
            figure8.append({
                "sampled_steps": step,
                "average_return_mean": return_row["mean"],
                "average_return_ci95": (
                    return_row["ci95_upper"] - return_row["mean"]
                ),
                "win_rate_mean": win_row["mean"],
                "win_rate_ci95": win_row["ci95_upper"] - win_row["mean"],
            })
        write_csv(output_dir / "figure8_data.csv", figure8)
    if "average_red_loss" in metrics:
        figure9 = []
        for step in steps:
            row = by_key[(step, "average_red_loss")]
            figure9.append({
                "sampled_steps": step,
                "average_red_loss_mean": row["mean"],
                "average_red_loss_ci95": row["ci95_upper"] - row["mean"],
            })
        write_csv(output_dir / "figure9_data.csv", figure9)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/aggregate"))
    args = parser.parse_args()
    run_dirs = [
        path if path.is_absolute() else PROJECT_ROOT / path for path in args.run_dirs
    ]
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else PROJECT_ROOT / args.output_dir
    )
    rows, manifest = aggregate_training_histories(run_dirs, output_dir)
    print(
        f"aggregated {manifest['number_of_runs']} runs, "
        f"{len(manifest['common_sampled_steps'])} steps, "
        f"{len(manifest['aggregated_metrics'])} metrics into {output_dir} "
        f"({len(rows)} summary rows)"
    )


if __name__ == "__main__":
    main()

