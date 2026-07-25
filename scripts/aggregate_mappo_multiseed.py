"""Aggregate MAPPO seed summaries with sample uncertainty statistics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import yaml


_T_975 = (
    0.0,12.706204736,4.302652730,3.182446305,2.776445105,2.570581836,
    2.446911851,2.364624252,2.306004135,2.262157163,2.228138852,
    2.200985160,2.178812830,2.160368656,2.144786688,2.131449546,
    2.119905299,2.109815578,2.100922040,2.093024054,2.085963447,
    2.079613845,2.073873068,2.068657610,2.063898562,2.059538553,
    2.055529439,2.051830516,2.048407142,2.045229642,2.042272456,
)


def student_t_critical_975(degrees_of_freedom: int) -> float:
    """Return the two-sided 95% Student-t critical value without SciPy."""

    if degrees_of_freedom <= 0: return 0.0
    if degrees_of_freedom < len(_T_975): return _T_975[degrees_of_freedom]
    z=1.959963984540054; n=float(degrees_of_freedom)
    return z+(z**3+z)/(4*n)+(5*z**5+16*z**3+3*z)/(96*n*n)+(3*z**7+19*z**5+17*z**3-15*z)/(384*n**3)


METRICS = (
    "overall_red_win_rate", "elimination_win_rate", "timeout_survival_win_rate", "decisive_win_rate", "draw_rate",
    "mean_team_episode_return", "mean_agent_sum_episode_return",
    "timeout_rate", "red_crash_rate", "blue_crash_rate", "mean_effective_damage",
    "mean_hits", "mean_attack_area_steps", "mean_episode_steps", "mean_red_survivors", "mean_blue_survivors",
    "policy_entropy_mean", "mean_observation_saturation_ratio",
    "terminal_reward_proportion",
)


def summarize(values: list[float]) -> dict[str, float | int | str]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("Aggregation requires a nonempty finite sample")
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    critical = student_t_critical_975(int(array.size - 1)) if array.size > 1 else 0.0
    half = critical * std / np.sqrt(array.size)
    return {"mean": mean, "sample_std": std, "ci95_low": float(mean-half), "ci95_high": float(mean+half),
            "median": float(np.median(array)), "min": float(array.min()), "max": float(array.max()),
            "confidence_interval_method":"student_t","num_training_seeds":int(array.size)}


def aggregate(input_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = sorted(input_root.glob("seed_*/run/seed_summary.yaml"))
    if not paths:
        raise FileNotFoundError(f"No seed summaries under {input_root}")
    payloads = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in paths]
    rows: list[dict[str, Any]] = []
    grouped: dict[str, list[float]] = {}
    for payload in payloads:
        evaluations=payload.get("test",{}).get("evaluations",payload["evaluations"])
        for checkpoint, metrics in evaluations.items():
            for metric in METRICS:
                value = metrics.get(metric)
                if value is None:
                    continue
                rows.append({"seed": payload["seed"], "checkpoint": checkpoint, "metric": metric, "value": float(value), "run_dir": payload["run_dir"]})
                grouped.setdefault(f"{checkpoint}.{metric}", []).append(float(value))
    summary = {"input_root": str(input_root.resolve()), "evaluation_split":"test", "num_training_seeds": len(payloads), "confidence_interval_method":"student_t", "statistics": {key: summarize(values) for key, values in grouped.items()}}
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
