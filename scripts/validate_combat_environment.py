"""Reproducible V1.1 environment validation without learning."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import numpy as np

from uav_combat.environment.env import MultiUAVCombatEnv
from uav_combat.math_utils import wrap_angle


def distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {
        "count": len(values), "min": float(np.min(array)),
        "p25": float(np.percentile(array, 25)), "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)), "max": float(np.max(array)),
    }


def reward_statistics(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size), "mean": float(np.mean(array)), "std": float(np.std(array)),
        "p10": float(np.percentile(array, 10)), "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)), "max_abs": float(np.max(np.abs(array))),
    }


def reset_statistics(config: Path, count: int) -> dict:
    env = MultiUAVCombatEnv(config)
    speeds, altitudes, horizontal_centers = [], [], []
    center_separations, pair_distances, heading_perturbations = [], [], []
    for seed in range(count):
        _, info = env.reset(seed)
        nominal_red = info["radial_angle"]
        nominal_blue = wrap_angle(nominal_red + np.pi)
        red_center = np.mean([[state.x, state.y] for state in env.red], axis=0)
        blue_center = np.mean([[state.x, state.y] for state in env.blue], axis=0)
        center_separations.append(np.linalg.norm(blue_center - red_center))
        pair_distances.extend(
            np.linalg.norm([blue.x - red.x, blue.y - red.y, blue.z - red.z])
            for red in env.red for blue in env.blue
        )
        heading_perturbations.extend(
            wrap_angle(state.psi - nominal_red) for state in env.red
        )
        heading_perturbations.extend(
            wrap_angle(state.psi - nominal_blue) for state in env.blue
        )
        for team in (env.red, env.blue):
            speeds.extend(state.v for state in team)
            altitudes.extend(state.altitude for state in team)
            horizontal_centers.append(np.mean([[state.x, state.y] for state in team], axis=0))
    centers = np.asarray(horizontal_centers)
    return {
        "resets": count,
        "speed_min": float(np.min(speeds)), "speed_max": float(np.max(speeds)),
        "speed_mean": float(np.mean(speeds)),
        "altitude_min": float(np.min(altitudes)), "altitude_max": float(np.max(altitudes)),
        "altitude_mean": float(np.mean(altitudes)),
        "formation_center_radius_mean": float(np.mean(np.linalg.norm(centers, axis=1))),
        "team_center_separation_min": float(np.min(center_separations)),
        "team_center_separation_max": float(np.max(center_separations)),
        "team_center_separation_mean": float(np.mean(center_separations)),
        "red_blue_pair_distance_min": float(np.min(pair_distances)),
        "red_blue_pair_distance_max": float(np.max(pair_distances)),
        "red_blue_pair_distance_mean": float(np.mean(pair_distances)),
        "heading_perturbation_deg_min": float(np.rad2deg(np.min(heading_perturbations))),
        "heading_perturbation_deg_max": float(np.rad2deg(np.max(heading_perturbations))),
        "heading_perturbation_deg_mean": float(np.rad2deg(np.mean(heading_perturbations))),
    }


def flank_actions(env: MultiUAVCombatEnv, nominal_heading: float) -> np.ndarray:
    """Simple 5-second symmetric lateral break used only by validation."""
    cfg = env.config["blue_policy"]
    actions = []
    for index, own in enumerate(env.red):
        if not own.alive:
            actions.append(np.zeros(3, dtype=np.float32))
            continue
        offset = np.deg2rad(30.0) if index < 2 else -np.deg2rad(30.0)
        desired_heading = wrap_angle(nominal_heading + offset)
        actions.append(np.clip(np.array([
            (260.0 - own.v) / cfg["speed_error_scale"],
            cfg["elevation_gain"] * (0.0 - own.theta) / cfg["elevation_action_scale"],
            cfg["heading_gain"] * wrap_angle(desired_heading - own.psi) / (np.pi / 3.0),
        ], dtype=np.float32), -1.0, 1.0))
    return np.stack(actions)


def run_episode(task: tuple[str, str, int]) -> dict:
    config, scenario, seed = task
    env = MultiUAVCombatEnv(config)
    _, reset_info = env.reset(seed)
    shaping_values: list[float] = []
    event_values: list[float] = []
    while True:
        if scenario == "straight":
            red_actions = np.zeros((4, 3), dtype=np.float32)
            blue_actions = np.zeros((4, 3), dtype=np.float32)
        elif scenario == "rule":
            red_actions = env.fixed_policy.team_actions(env.red, env.blue)
            blue_actions = None
        elif scenario == "flank":
            red_actions = (
                flank_actions(env, reset_info["radial_angle"])
                if env.steps < 50
                else env.fixed_policy.team_actions(env.red, env.blue)
            )
            blue_actions = None
        else:
            raise ValueError(f"unknown scenario: {scenario}")
        _, _, terminated, truncated, info = env.step(red_actions, blue_actions)
        shaping_values.extend(map(float, info["shaping_rewards"]))
        event_values.extend(map(float, info["event_rewards"]))
        if terminated or truncated:
            record = dict(info)
            record["_shaping_values"] = shaping_values
            record["_event_values"] = event_values
            return record


def summarize(records: list[dict], scenario: str) -> dict:
    episodes = len(records)
    mean = lambda key: float(np.mean([record[key] for record in records]))
    first_attackable = [r["first_attackable_step"] for r in records if r["first_attackable_step"] is not None]
    first_lock = [r["first_lock_step"] for r in records if r["first_lock_step"] is not None]
    first_kill = [r["first_kill_step"] for r in records if r["first_kill_step"] is not None]
    red_attack_kills = sum(r["red_attack_kills"] for r in records)
    blue_attack_kills = sum(r["blue_attack_kills"] for r in records)
    total_deaths = sum(r["red_losses"] + r["blue_losses"] for r in records)
    boundary_keys = (
        "red_horizontal_boundary_losses", "blue_horizontal_boundary_losses",
        "red_low_altitude_losses", "blue_low_altitude_losses",
        "red_high_altitude_losses", "blue_high_altitude_losses",
    )
    termination_counts = {
        reason: sum(record["termination_reason"] == reason for record in records)
        for reason in ("red_win", "blue_win", "draw_mutual_destruction", "draw_timeout")
    }
    shaping_values = [value for record in records for value in record["_shaping_values"]]
    event_values = [value for record in records for value in record["_event_values"]]
    result = {
        "episodes": episodes,
        "attackable_episodes": len(first_attackable),
        "attackable_episode_rate": len(first_attackable) / episodes,
        "completed_lock_episodes": len(first_lock),
        "completed_lock_episode_rate": len(first_lock) / episodes,
        "kill_episodes": len(first_kill),
        "first_kill_rate": len(first_kill) / episodes,
        "first_attackable_step_distribution": distribution(first_attackable),
        "first_lock_step_distribution": distribution(first_lock),
        "first_kill_step_distribution": distribution(first_kill),
        "episode_length_mean": mean("episode_length"),
        "episode_length_distribution": distribution([r["episode_length"] for r in records]),
        "red_loss_mean": mean("red_losses"), "blue_loss_mean": mean("blue_losses"),
        "red_attack_kills_total": red_attack_kills,
        "blue_attack_kills_total": blue_attack_kills,
        "combat_kills_total": red_attack_kills + blue_attack_kills,
        "total_deaths": total_deaths,
        "red_attack_kills_mean": mean("red_attack_kills"),
        "blue_attack_kills_mean": mean("blue_attack_kills"),
        "red_boundary_loss_mean": mean("red_boundary_losses"),
        "blue_boundary_loss_mean": mean("blue_boundary_losses"),
        **{f"{key}_total": sum(r[key] for r in records) for key in boundary_keys},
        **{f"{key}_mean": mean(key) for key in boundary_keys},
        "combat_kill_fraction": (red_attack_kills + blue_attack_kills) / max(total_deaths, 1),
        "win_rate": mean("red_win"), "loss_rate": mean("blue_win"),
        "draw_rate": mean("draw"), "termination_counts": termination_counts,
        "shaping_reward_statistics": reward_statistics(shaping_values),
        "event_reward_statistics": reward_statistics(event_values),
    }
    diagnoses = []
    if scenario == "flank" and not first_attackable:
        diagnoses.append("degenerate: flank baseline never entered the attack envelope")
    if scenario == "flank" and not first_kill:
        diagnoses.append("degenerate: flank baseline never completed maneuver-lock-kill")
    if scenario in ("rule", "flank") and result["combat_kill_fraction"] < 0.1 and (
        result["red_boundary_loss_mean"] + result["blue_boundary_loss_mean"] > 1.0
    ):
        diagnoses.append(f"degenerate: {scenario} baseline remains boundary dominated")
    if not np.all(np.isfinite(shaping_values + event_values)):
        diagnoses.append("invalid: non-finite reward component")
    result["diagnoses"] = diagnoses
    return result


def run_scenario(
    config: Path, scenario: str, episodes: int, seed_base: int, workers: int
) -> dict:
    tasks = [(str(config), scenario, seed_base + episode) for episode in range(episodes)]
    if workers == 1:
        records = [run_episode(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            records = list(executor.map(run_episode, tasks, chunksize=1))
    return summarize(records, scenario)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset-count", type=int, default=1000)
    parser.add_argument("--straight-episodes", type=int, default=100)
    parser.add_argument("--rule-episodes", type=int, default=200)
    parser.add_argument("--flank-episodes", type=int, default=200)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--output", default="outputs/combat_environment_validation_v1_1.json")
    args = parser.parse_args()
    if min(args.reset_count, args.straight_episodes, args.rule_episodes, args.flank_episodes, args.workers) <= 0:
        raise ValueError("all validation counts and workers must be positive")
    root = Path(__file__).resolve().parents[1]
    config = root / "configs/combat_environment.yaml"
    result = {
        "reset_statistics": reset_statistics(config, args.reset_count),
        "straight_vs_straight": run_scenario(
            config, "straight", args.straight_episodes, 1_000_000, args.workers
        ),
        "rule_vs_rule": run_scenario(
            config, "rule", args.rule_episodes, 2_000_000, args.workers
        ),
        "flank_then_pursuit_vs_fixed_blue": run_scenario(
            config, "flank", args.flank_episodes, 3_000_000, args.workers
        ),
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
