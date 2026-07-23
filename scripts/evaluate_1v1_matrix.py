"""Evaluate the complete 3x3x3 homogeneous 1v1 rule-policy matrix."""

from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean

import yaml

try:
    from scripts.run_1v1_episode import run_episode
except ModuleNotFoundError:  # direct script execution
    from run_1v1_episode import run_episode


FIELDS = [
    "scenario", "red_policy", "blue_policy", "episodes", "red_win_rate", "blue_win_rate", "draw_rate",
    "timeout_rate", "red_ground_crash_rate", "blue_ground_crash_rate", "collision_rate", "mean_episode_steps",
    "mean_red_effective_damage", "mean_blue_effective_damage", "mean_red_hits", "mean_blue_hits",
    "mean_red_attack_area_steps", "mean_blue_attack_area_steps", "mean_reward",
]


def _run_case(case: tuple[str, str, str, int]):
    scenario, red_policy, blue_policy, seed = case
    return run_episode(scenario, blue_policy, seed, red_policy)[1]


def evaluate_matrix(episodes: int, seed_start: int = 0, workers: int = 1) -> list[dict[str, float | int | str]]:
    """Return deterministic aggregate rows for every required combination."""

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    rows: list[dict[str, float | int | str]] = []
    for scenario in ("tail_chase", "head_on", "balanced_random"):
        for red_policy in ("straight", "random", "pursuit"):
            for blue_policy in ("straight", "random", "pursuit"):
                cases = [(scenario, red_policy, blue_policy, seed_start + index) for index in range(episodes)]
                if workers == 1:
                    summaries = [_run_case(case) for case in cases]
                else:
                    with ProcessPoolExecutor(max_workers=workers) as executor:
                        summaries = list(executor.map(_run_case, cases))
                red_wins = sum(s.outcome == "red" for s in summaries)
                blue_wins = sum(s.outcome == "blue" for s in summaries)
                draws = episodes - red_wins - blue_wins
                row = {
                    "scenario": scenario, "red_policy": red_policy, "blue_policy": blue_policy, "episodes": episodes,
                    "red_win_rate": red_wins / episodes, "blue_win_rate": blue_wins / episodes, "draw_rate": draws / episodes,
                    "timeout_rate": mean(s.timeout for s in summaries), "red_ground_crash_rate": mean(s.red_ground_crash for s in summaries),
                    "blue_ground_crash_rate": mean(s.blue_ground_crash for s in summaries), "collision_rate": mean(s.collision for s in summaries),
                    "mean_episode_steps": mean(s.decision_steps for s in summaries), "mean_red_effective_damage": mean(s.red_damage for s in summaries),
                    "mean_blue_effective_damage": mean(s.blue_damage for s in summaries), "mean_red_hits": mean(s.red_hits for s in summaries),
                    "mean_blue_hits": mean(s.blue_hits for s in summaries), "mean_red_attack_area_steps": mean(s.red_attack_area_steps for s in summaries),
                    "mean_blue_attack_area_steps": mean(s.blue_attack_area_steps for s in summaries), "mean_reward": mean(s.cumulative_reward for s in summaries),
                }
                rows.append(row)
    return rows


def main() -> None:
    """Evaluate, print, and save the complete matrix and its seed configuration."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    rows = evaluate_matrix(args.episodes, args.seed_start, args.workers)
    output = Path("outputs/metrics")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "1v1_rule_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    snapshot = {"episodes_per_combination": args.episodes, "seed_start": args.seed_start, "seed_end": args.seed_start + args.episodes - 1, "workers": args.workers, "scenarios": ["tail_chase", "head_on", "balanced_random"], "policies": ["straight", "random", "pursuit"]}
    (output / "1v1_rule_matrix_config.yaml").write_text(yaml.safe_dump(snapshot, sort_keys=False), encoding="utf-8")
    writer = csv.DictWriter(__import__("sys").stdout, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
