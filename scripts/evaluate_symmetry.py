"""Paired mirror/team-swap fairness evaluation for homogeneous combat environments."""

from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from uav_env.core.symmetry import mirror_state_xz
from uav_env.core.enums import Team
from uav_env.envs import make_1v1_env, make_2v2_env
from uav_env.opponents.pursuit import PursuitOpponent
from uav_env.opponents.team_controller import TeamRuleController


@dataclass(frozen=True)
class EpisodeRecord:
    winner: str
    episode_return: float
    steps: int
    red_damage: float
    blue_damage: float
    red_crash: bool
    blue_crash: bool


def _swapped(winner: str) -> str:
    return {"red": "blue", "blue": "red", "draw": "draw"}.get(winner, winner)


def _pursuit(env) -> PursuitOpponent:
    return PursuitOpponent(
        env.profile, env.attack_config, float(env.config["physics_dt"]),
        int(env.config["physics_steps_per_action"]), float(env.config["gravity"]),
        float(env.config["max_altitude"]),
        **{key: float(value) for key, value in env.config["pursuit"].items()},
    )


def _run_1v1(scenario: str, seed: int, transform: str) -> EpisodeRecord:
    env = make_1v1_env(scenario, "pursuit", seed=seed)
    if transform != "original":
        source = make_1v1_env(scenario, "pursuit", seed=seed)
        source.reset(seed=seed)
        if transform == "mirror":
            options = {"red_state": mirror_state_xz(source.red.state), "blue_state": mirror_state_xz(source.blue.state)}
        elif transform == "swap_mirror":
            options = {
                "red_state": replace(mirror_state_xz(source.blue.state), team_id=int(Team.RED)),
                "blue_state": replace(mirror_state_xz(source.red.state), team_id=int(Team.BLUE)),
            }
        else:
            raise ValueError(f"Unknown transform: {transform}")
        env.reset(seed=seed, options=options)
        env.reverse_damage_sample_order = transform == "swap_mirror"
    else:
        env.reset(seed=seed)
    red_policy = _pursuit(env)
    total = 0.0
    terminated = truncated = False
    info = {}
    while not (terminated or truncated):
        action = red_policy.select_action(env.red.state.copy(), env.blue.state.copy())
        _, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
    outcome = info["outcome"]
    stats = info["statistics"]
    reason = str(outcome.termination_reason)
    return EpisodeRecord(
        str(outcome.winner), total, int(outcome.decision_steps),
        float(stats["red_effective_damage"]), float(stats["blue_effective_damage"]),
        "red_ground_crash" in reason, "blue_ground_crash" in reason,
    )


def _run_2v2(scenario: str, seed: int, transform: str) -> EpisodeRecord:
    env = make_2v2_env(scenario, "pursuit", seed=seed)
    if transform != "original":
        source = make_2v2_env(scenario, "pursuit", seed=seed)
        source.reset(seed=seed)
        if transform == "mirror":
            options = {
                "red_states": [mirror_state_xz(u.state) for u in source.red_aircraft],
                "blue_states": [mirror_state_xz(u.state) for u in source.blue_aircraft],
            }
        elif transform == "swap_mirror":
            options = {
                "red_states": [replace(mirror_state_xz(u.state), team_id=int(Team.RED)) for u in source.blue_aircraft],
                "blue_states": [replace(mirror_state_xz(u.state), team_id=int(Team.BLUE)) for u in source.red_aircraft],
            }
        else:
            raise ValueError(f"Unknown transform: {transform}")
        env.reset(seed=seed, options=options)
        if transform == "swap_mirror":
            env.damage_sample_team_order = (int(Team.RED), int(Team.BLUE))
    else:
        env.reset(seed=seed)
    red_controller = TeamRuleController("pursuit", _pursuit(env), seed + 1_000_003)
    total = 0.0
    terminated = truncated = False
    info = {}
    while not (terminated or truncated):
        actions, _ = red_controller.select_actions(env.red_aircraft, env.blue_aircraft)
        _, reward, terminated, truncated, info = env.step(np.asarray(actions, dtype=np.int64))
        total += float(reward)
    outcome = info["outcome"]
    aircraft = info["statistics"]["aircraft"]
    red_damage = sum(float(aircraft[f"red_{i}"]["effective_damage"]) for i in range(2))
    blue_damage = sum(float(aircraft[f"blue_{i}"]["effective_damage"]) for i in range(2))
    red_crash = any(int(aircraft[f"red_{i}"]["ground_crashes"]) for i in range(2))
    blue_crash = any(int(aircraft[f"blue_{i}"]["ground_crashes"]) for i in range(2))
    return EpisodeRecord(str(outcome.winner), total, int(outcome.decision_steps), red_damage, blue_damage, red_crash, blue_crash)


def _run_named_case(name: str, seed: int, transform: str) -> EpisodeRecord:
    if name == "1v1_head_on_pursuit_vs_pursuit":
        return _run_1v1("head_on", seed, transform)
    if name == "2v2_head_on_pursuit_vs_pursuit":
        return _run_2v2("head_on_formation", seed, transform)
    if name == "2v2_offset_and_mirror_pursuit_vs_pursuit":
        return _run_2v2("offset_formation", seed, transform)
    raise ValueError(f"Unknown symmetry case: {name}")


def _evaluate_case(name: str, episodes: int, seed_start: int, workers: int) -> dict[str, float | int | str]:
    jobs = [(name, seed_start + offset, transform) for offset in range(episodes) for transform in ("original", "mirror", "swap_mirror")]
    if workers == 1:
        records = [_run_named_case(*job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            records = list(executor.map(_run_named_case, *zip(*jobs)))
    originals = records[0::3]
    mirrors = records[1::3]
    swaps = records[2::3]
    combined = originals + swaps
    outcome_match = np.mean([a.winner == _swapped(b.winner) for a, b in zip(originals, swaps)])
    damage_differences = [
        0.5 * (abs(a.red_damage - b.red_damage) + abs(a.blue_damage - b.blue_damage))
        for a, b in zip(originals, mirrors)
    ]
    return {
        "case": name,
        "paired_seeds": episodes,
        "paired_outcome_match_rate": float(outcome_match),
        # Pure reflection preserves the learning side and therefore reward viewpoint.
        "mean_return_difference": float(np.mean([abs(a.episode_return - b.episode_return) for a, b in zip(originals, mirrors)])),
        "mean_effective_damage_difference": float(np.mean(damage_differences)),
        "mean_episode_length_difference": float(np.mean([abs(a.steps - b.steps) for a, b in zip(originals, mirrors)])),
        "red_blue_win_rate_gap": float(abs(np.mean([r.winner == "red" for r in combined]) - np.mean([r.winner == "blue" for r in combined]))),
        "crash_rate_gap": float(abs(np.mean([r.red_crash for r in combined]) - np.mean([r.blue_crash for r in combined]))),
    }


def evaluate_symmetry(episodes: int, seed_start: int = 0, workers: int = 1) -> list[dict[str, float | int | str]]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    cases = ["1v1_head_on_pursuit_vs_pursuit", "2v2_head_on_pursuit_vs_pursuit", "2v2_offset_and_mirror_pursuit_vs_pursuit"]
    return [_evaluate_case(name, episodes, seed_start, workers) for name in cases]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100, help="Paired seeds per case")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--output", default="outputs/metrics/symmetry_report.csv")
    args = parser.parse_args()
    rows = evaluate_symmetry(args.episodes, args.seed_start, args.workers)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    writer = csv.DictWriter(__import__("sys").stdout, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
