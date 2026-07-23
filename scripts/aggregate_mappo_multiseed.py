"""Aggregate MAPPO seed summaries with sample uncertainty statistics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import yaml


METRICS = (
    "red_win_rate", "mean_team_episode_return", "mean_agent_sum_episode_return",
    "timeout_rate", "red_crash_rate", "blue_crash_rate", "mean_effective_damage",
    "mean_hits", "mean_episode_steps", "policy_entropy_mean", "mean_observation_saturation_ratio",
    "terminal_reward_proportion",
)


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("Aggregation requires a nonempty finite sample")
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    half = 1.96 * std / np.sqrt(array.size)
    return {"mean": mean, "sample_std": std, "ci95_low": float(mean-half), "ci95_high": float(mean+half),
            "median": float(np.median(array)), "min": float(array.min()), "max": float(array.max())}


def aggregate(input_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = sorted(input_root.glob("seed_*/run/seed_summary.yaml"))
    if not paths:
        raise FileNotFoundError(f"No seed summaries under {input_root}")
    payloads = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in paths]
    rows: list[dict[str, Any]] = []
    grouped: dict[str, list[float]] = {}
    for payload in payloads:
        for checkpoint, metrics in payload["evaluations"].items():
            for metric in METRICS:
                value = metrics.get(metric)
                if value is None:
                    continue
                rows.append({"seed": payload["seed"], "checkpoint": checkpoint, "metric": metric, "value": float(value), "run_dir": payload["run_dir"]})
                grouped.setdefault(f"{checkpoint}.{metric}", []).append(float(value))
    summary = {"input_root": str(input_root.resolve()), "num_seeds": len(payloads), "statistics": {key: summarize(values) for key, values in grouped.items()}}
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-summary", required=True)
    args = parser.parse_args()
    rows, summary = aggregate(Path(args.input_root))
    csv_path, yaml_path = Path(args.output_csv), Path(args.output_summary)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    yaml_path.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    print(csv_path.resolve()); print(yaml_path.resolve())


if __name__ == "__main__":
    main()
