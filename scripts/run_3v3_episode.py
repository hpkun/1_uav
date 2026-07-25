"""Run and summarize one fixed homogeneous 3v3 head-on episode."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json

import numpy as np

from uav_env.envs import make_3v3_env
from uav_env.envs.combat_multi_env import CombatMultiEnv
from uav_env.opponents.pursuit import PursuitOpponent
from uav_env.opponents.team_controller import TeamRuleController


@dataclass(frozen=True)
class Episode3v3Summary:
    winner: str
    termination_reason: str
    decision_steps: int
    red_survivors: int
    blue_survivors: int
    elimination_win: bool
    timeout_survival_win: bool
    red_effective_damage: float
    blue_effective_damage: float
    red_hits: int
    blue_hits: int
    red_attack_area_steps: int
    blue_attack_area_steps: int
    red_crashes: int
    blue_crashes: int
    team_return: float


def run_3v3_episode(seed: int, red_policy: str = "pursuit", max_steps: int | None = None) -> tuple[CombatMultiEnv, Episode3v3Summary]:
    """Run pursuit-vs-pursuit with independent nearest living targets."""

    if red_policy != "pursuit":
        raise ValueError("The fixed 3v3 rule probe currently supports red pursuit only")
    env = make_3v3_env("head_on_formation", "pursuit", seed=seed, multi_terminal_reward_profile="paper_2024_exact")
    env.reset(seed=seed)
    pursuit_cfg = {key: float(value) for key, value in env.config["pursuit"].items()}
    pursuit = PursuitOpponent(
        env.profile, env.attack_config, float(env.config["physics_dt"]), int(env.config["physics_steps_per_action"]),
        float(env.config["gravity"]), float(env.config["max_altitude"]), **pursuit_cfg,
    )
    controller = TeamRuleController("pursuit", pursuit, seed + 1_000_003)
    terminated = truncated = False
    team_return = 0.0
    info: dict[str, object] = {}
    limit = max_steps or int(env.config["max_decision_steps"])
    while not (terminated or truncated) and env.decision_step < limit:
        selected, _ = controller.select_actions(env.red_aircraft, env.blue_aircraft)
        _, reward, terminated, truncated, info = env.step(np.asarray([int(action) for action in selected], dtype=np.int64))
        team_return += float(reward)
    outcome = info.get("outcome", env._outcome(False))
    statistics = env.get_statistics()["aircraft"]
    red_ids = [f"red_{index}" for index in range(3)]
    blue_ids = [f"blue_{index}" for index in range(3)]
    reason = str(outcome.termination_reason)
    winner = str(outcome.winner or "none")
    summary = Episode3v3Summary(
        winner=winner,
        termination_reason=reason,
        decision_steps=env.decision_step,
        red_survivors=int(outcome.red_survivors or 0),
        blue_survivors=int(outcome.blue_survivors or 0),
        elimination_win=winner == "red" and reason == "blue_eliminated",
        timeout_survival_win=winner == "red" and reason == "timeout",
        red_effective_damage=sum(float(statistics[key]["effective_damage"]) for key in red_ids),
        blue_effective_damage=sum(float(statistics[key]["effective_damage"]) for key in blue_ids),
        red_hits=sum(int(statistics[key]["hits"]) for key in red_ids),
        blue_hits=sum(int(statistics[key]["hits"]) for key in blue_ids),
        red_attack_area_steps=sum(int(statistics[key]["attack_area_steps"]) for key in red_ids),
        blue_attack_area_steps=sum(int(statistics[key]["attack_area_steps"]) for key in blue_ids),
        red_crashes=sum(int(statistics[key]["ground_crashes"]) for key in red_ids),
        blue_crashes=sum(int(statistics[key]["ground_crashes"]) for key in blue_ids),
        team_return=team_return,
    )
    return env, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--red-policy", choices=["pursuit"], default="pursuit")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    completed: list[tuple[Episode3v3Summary, int]] = []
    for offset in range(args.episodes):
        env, summary = run_3v3_episode(args.seed + offset, args.red_policy, args.max_steps)
        attempts = sum(len(step.get("attack_attempts", [])) for step in env.get_trajectory())
        completed.append((summary, attempts))
    if args.episodes == 1:
        for key, value in completed[0][0].__dict__.items():
            print(f"{key}: {value}")
        return
    summaries = [item[0] for item in completed]
    reasons = sorted({summary.termination_reason for summary in summaries})
    aggregate = {
        "episodes": len(summaries),
        "termination_reasons": {reason: sum(summary.termination_reason == reason for summary in summaries) for reason in reasons},
        "red_wins": sum(summary.winner == "red" for summary in summaries),
        "blue_wins": sum(summary.winner == "blue" for summary in summaries),
        "draws": sum(summary.winner == "draw" for summary in summaries),
        "attack_area_steps": sum(summary.red_attack_area_steps + summary.blue_attack_area_steps for summary in summaries),
        "attack_attempts": sum(item[1] for item in completed),
        "hits": sum(summary.red_hits + summary.blue_hits for summary in summaries),
        "effective_damage": sum(summary.red_effective_damage + summary.blue_effective_damage for summary in summaries),
        "mean_red_survivors": sum(summary.red_survivors for summary in summaries) / len(summaries),
        "mean_blue_survivors": sum(summary.blue_survivors for summary in summaries) / len(summaries),
        "crashes": sum(summary.red_crashes + summary.blue_crashes for summary in summaries),
    }
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
