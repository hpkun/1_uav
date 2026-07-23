"""Matched-seed statistical comparison of rule baselines and MAPPO checkpoints."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable

import numpy as np
import torch

try:
    from scripts.run_1v1_episode import run_episode
    from scripts.run_2v2_episode import run_2v2_episode
except ModuleNotFoundError:  # direct script execution
    from run_1v1_episode import run_episode
    from run_2v2_episode import run_2v2_episode
from uav_env.algorithms.mappo.runner import MAPPORunner


def _summary(values: list[float]) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    half = 1.96 * std / np.sqrt(len(array))
    return mean, std, mean - half, mean + half


def _rule_episode(config: dict, policy: str, seed: int) -> dict[str, float]:
    environment = config["environment"]
    if environment["kind"] == "1v1":
        _, result = run_episode(environment["scenario"], environment["opponent"], seed, policy)
        return {
            "episode_return": result.cumulative_reward,
            "red_win": float(result.outcome == "red"),
            "effective_damage": result.red_damage,
            "episode_steps": float(result.decision_steps),
            "crash": float(result.red_ground_crash or result.blue_ground_crash),
        }
    _, result = run_2v2_episode(environment["scenario"], environment["opponent"], seed, policy)
    return {
        "episode_return": result.team_cumulative_reward,
        "red_win": float(result.winner == "red"),
        "effective_damage": float(sum(result.red_effective_damage.values())),
        "episode_steps": float(result.decision_steps),
        "crash": float(result.red_ground_crashes + result.blue_ground_crashes > 0),
    }


def _checkpoint_evaluator(checkpoint: Path, config: dict) -> Callable[[int], dict[str, float]]:
    runner = MAPPORunner(config, f"comparison_{checkpoint.stem}")
    runner.resume(str(checkpoint), actor_only=True)

    def evaluate(seed: int) -> dict[str, float]:
        result = runner.evaluate(1, seed_start=seed, deterministic=True)
        return {
            "episode_return": result["mean_episode_return"],
            "red_win": result["red_win_rate"],
            "effective_damage": result["mean_effective_damage"],
            "episode_steps": result["mean_episode_steps"],
            "crash": float(max(result["red_crash_rate"], result["blue_crash_rate"])),
        }

    return evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--last", required=True)
    parser.add_argument("--best", required=True)
    parser.add_argument("--initial", help="Optional initialized-policy checkpoint; defaults to last checkpoint's sibling initial.pt")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=100000)
    parser.add_argument("--output", default="outputs/metrics/mappo_baseline_comparison.csv")
    args = parser.parse_args()
    if args.episodes <= 1:
        raise ValueError("At least two episodes are required for a sample standard deviation")

    last = Path(args.last)
    best = Path(args.best)
    data = torch.load(last, map_location="cpu", weights_only=False)
    config = data["config"]
    initial = Path(args.initial) if args.initial else last.with_name("initial.pt")
    evaluators: list[tuple[str, str, Callable[[int], dict[str, float]]]] = [
        ("random", "rule", lambda seed: _rule_episode(config, "random", seed)),
        ("straight", "rule", lambda seed: _rule_episode(config, "straight", seed)),
        ("pursuit", "rule", lambda seed: _rule_episode(config, "pursuit", seed)),
    ]
    if initial.exists():
        evaluators.append(("mappo_initial", str(initial), _checkpoint_evaluator(initial, config)))
    evaluators.extend([
        ("mappo_last", str(last), _checkpoint_evaluator(last, config)),
        ("mappo_best", str(best), _checkpoint_evaluator(best, config)),
    ])

    rows: list[dict[str, float | int | str]] = []
    for policy, source, evaluate in evaluators:
        samples = [evaluate(args.seed_start + offset) for offset in range(args.episodes)]
        for metric in ("episode_return", "red_win", "effective_damage", "episode_steps", "crash"):
            mean, std, low, high = _summary([sample[metric] for sample in samples])
            if metric in {"red_win", "crash"}:
                low, high = max(0.0, low), min(1.0, high)
            rows.append({
                "policy": policy, "metric": metric, "episodes": args.episodes,
                "seed_start": args.seed_start, "mean": mean, "std": std,
                "ci95_low": low, "ci95_high": high, "source": source,
            })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output.resolve())


if __name__ == "__main__":
    main()
