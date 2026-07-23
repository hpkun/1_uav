"""Aggregate the matched paper/project terminal-reward sensitivity probe."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import yaml


EVALUATION_METRICS = (
    "mean_team_episode_return",
    "mean_agent_sum_episode_return",
    "timeout_rate",
    "mean_effective_damage",
    "mean_hits",
    "policy_entropy_mean",
    "terminal_reward_proportion",
)


def _training_value_losses(run_dir: Path) -> tuple[float, float]:
    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as stream:
        values = [float(row["value_loss"]) for row in csv.DictReader(stream)]
    if not values:
        raise ValueError(f"No training rows in {run_dir / 'metrics.csv'}")
    return values[-1], float(np.mean(values[-10:]))


def _load_rows(run_dir: Path) -> list[dict[str, float | int | str]]:
    with (run_dir / "seed_summary.yaml").open(encoding="utf-8") as stream:
        summary = yaml.safe_load(stream)
    final_value_loss, tail_value_loss = _training_value_losses(run_dir)
    rows: list[dict[str, float | int | str]] = []
    for checkpoint in ("initial", "last", "best"):
        evaluation = summary["evaluations"][checkpoint]
        row: dict[str, float | int | str] = {
            "profile": summary["terminal_reward_profile"],
            "seed": int(summary["seed"]),
            "total_env_steps": int(summary["total_env_steps"]),
            "evaluation_episodes": int(summary["evaluation_episodes"]),
            "evaluation_seed_start": int(summary["evaluation_seed_start"]),
            "checkpoint": checkpoint,
            "final_training_value_loss": final_value_loss,
            "last_10_updates_mean_value_loss": tail_value_loss,
        }
        row.update({metric: float(evaluation[metric]) for metric in EVALUATION_METRICS})
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-run", required=True, type=Path)
    parser.add_argument("--project-run", required=True, type=Path)
    parser.add_argument(
        "--output",
        default="outputs/metrics/2v2_terminal_reward_ablation.csv",
        type=Path,
    )
    args = parser.parse_args()
    rows = _load_rows(args.paper_run) + _load_rows(args.project_run)
    matched = {
        (row["seed"], row["total_env_steps"], row["evaluation_episodes"], row["evaluation_seed_start"])
        for row in rows
    }
    if len(matched) != 1:
        raise ValueError("Ablation runs do not share seed, step count, and evaluation protocol")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
