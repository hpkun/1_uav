"""Evaluate rule-policy combinations without any learning algorithm."""

from __future__ import annotations

import argparse
from statistics import mean

from run_1v1_episode import run_episode


def main() -> None:
    """Run a batch and print win/loss/draw and safety statistics."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--scenario", choices=["tail_chase", "head_on", "balanced_random"], default="balanced_random")
    parser.add_argument("--red-policy", choices=["random", "straight", "pursuit"], default="pursuit")
    parser.add_argument("--opponent", choices=["random", "straight", "pursuit"], default="pursuit")
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    summaries = [
        run_episode(args.scenario, args.opponent, args.seed + index, args.red_policy)[1]
        for index in range(args.episodes)
    ]
    wins = sum(summary.outcome == "red" for summary in summaries)
    losses = sum(summary.outcome == "blue" for summary in summaries)
    draws = sum(summary.outcome == "draw" for summary in summaries)
    crashes = sum("ground" in summary.termination_reason or "ceiling" in summary.termination_reason for summary in summaries)
    timeouts = sum(summary.termination_reason == "timeout" for summary in summaries)
    print(f"Episodes: {args.episodes}")
    print(f"Win rate: {wins / args.episodes:.4f}")
    print(f"Loss rate: {losses / args.episodes:.4f}")
    print(f"Draw rate: {draws / args.episodes:.4f}")
    print(f"Average episode length: {mean(summary.decision_steps for summary in summaries):.3f}")
    print(f"Average cumulative reward: {mean(summary.cumulative_reward for summary in summaries):.6f}")
    print(f"Average red hits: {mean(summary.red_hits for summary in summaries):.3f}")
    print(f"Crash rate: {crashes / args.episodes:.4f}")
    print(f"Timeout rate: {timeouts / args.episodes:.4f}")


if __name__ == "__main__":
    main()
