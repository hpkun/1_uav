"""Batch evaluation for homogeneous 2v2 rule policies."""

from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean

try:
    from scripts.run_2v2_episode import run_2v2_episode
except ModuleNotFoundError:  # direct script execution
    from run_2v2_episode import run_2v2_episode


FIELDS = ["scenario", "red_policy", "blue_policy", "episodes", "red_win_rate", "blue_win_rate", "draw_rate", "average_red_survivors", "average_blue_survivors", "timeout_rate", "crash_rate", "mean_red_effective_damage", "mean_red_hits", "mean_contribution_score", "mean_team_reward", "mean_episode_steps"]


def _run_case(case: tuple[str, str, str, int]):
    scenario, red_policy, blue_policy, seed = case
    return run_2v2_episode(scenario, blue_policy, seed, red_policy)[1]


def evaluate_2v2(episodes: int, seed_start: int = 0, workers: int = 1) -> list[dict[str, float | int | str]]:
    """Evaluate all three scenarios and all three-by-three policy pairs."""

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    rows = []
    for scenario in ("head_on_formation", "offset_formation", "balanced_random"):
        for red_policy in ("straight", "random", "pursuit"):
            for blue_policy in ("straight", "random", "pursuit"):
                cases = [(scenario, red_policy, blue_policy, seed_start + i) for i in range(episodes)]
                if workers == 1:
                    summaries = [_run_case(case) for case in cases]
                else:
                    with ProcessPoolExecutor(max_workers=workers) as executor:
                        summaries = list(executor.map(_run_case, cases))
                red_wins = sum(s.winner == "red" for s in summaries)
                blue_wins = sum(s.winner == "blue" for s in summaries)
                rows.append({
                    "scenario": scenario, "red_policy": red_policy, "blue_policy": blue_policy, "episodes": episodes,
                    "red_win_rate": red_wins / episodes, "blue_win_rate": blue_wins / episodes, "draw_rate": (episodes-red_wins-blue_wins)/episodes,
                    "average_red_survivors": mean(s.red_survivors for s in summaries), "average_blue_survivors": mean(s.blue_survivors for s in summaries),
                    "timeout_rate": mean(s.timeout for s in summaries), "crash_rate": mean((s.red_ground_crashes+s.blue_ground_crashes)>0 for s in summaries),
                    "mean_red_effective_damage": mean(sum(s.red_effective_damage.values()) for s in summaries),
                    "mean_red_hits": mean(sum(s.red_hits.values()) for s in summaries), "mean_contribution_score": mean(sum(s.red_contribution.values()) for s in summaries),
                    "mean_team_reward": mean(s.team_cumulative_reward for s in summaries), "mean_episode_steps": mean(s.decision_steps for s in summaries),
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    rows = evaluate_2v2(args.episodes, args.seed_start, args.workers)
    output = Path("outputs/metrics")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "2v2_rule_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    writer = csv.DictWriter(__import__("sys").stdout, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
