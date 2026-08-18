"""Statistical validation for the public combat environment (no learning)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from uav_combat.environment.env import MultiUAVCombatEnv


def reset_statistics(config: Path, count: int) -> dict:
    env = MultiUAVCombatEnv(config)
    speeds, altitudes, horizontal_centers = [], [], []
    center_separations, pair_distances, heading_perturbations = [], [], []
    for seed in range(count):
        _, info = env.reset(seed)
        nominal_red = info["radial_angle"]
        nominal_blue = (nominal_red + np.pi + np.pi) % (2 * np.pi) - np.pi
        red_center = np.mean([[state.x, state.y] for state in env.red], axis=0)
        blue_center = np.mean([[state.x, state.y] for state in env.blue], axis=0)
        center_separations.append(np.linalg.norm(blue_center - red_center))
        pair_distances.extend(
            np.linalg.norm([blue.x - red.x, blue.y - red.y, blue.z - red.z])
            for red in env.red for blue in env.blue
        )
        heading_perturbations.extend(
            (state.psi - nominal_red + np.pi) % (2 * np.pi) - np.pi for state in env.red
        )
        heading_perturbations.extend(
            (state.psi - nominal_blue + np.pi) % (2 * np.pi) - np.pi for state in env.blue
        )
        for team in (env.red, env.blue):
            speeds.extend(state.v for state in team)
            altitudes.extend(state.altitude for state in team)
            horizontal_centers.append(np.mean([[state.x, state.y] for state in team], axis=0))
    centers = np.asarray(horizontal_centers)
    return {
        "resets": count,
        "speed_min": float(np.min(speeds)),
        "speed_max": float(np.max(speeds)),
        "speed_mean": float(np.mean(speeds)),
        "altitude_min": float(np.min(altitudes)),
        "altitude_max": float(np.max(altitudes)),
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


def run_scenario(config: Path, episodes: int, rule_red: bool, seed_base: int) -> dict:
    records = []
    for episode in range(episodes):
        env = MultiUAVCombatEnv(config)
        env.reset(seed_base + episode)
        while True:
            red_actions = (
                env.fixed_policy.team_actions(env.red, env.blue)
                if rule_red else np.zeros((4, 3), dtype=np.float32)
            )
            blue_actions = None if rule_red else np.zeros((4, 3), dtype=np.float32)
            _, _, terminated, truncated, info = env.step(red_actions, blue_actions)
            if terminated or truncated:
                records.append(info)
                break

    def mean(key: str) -> float:
        return float(np.mean([record[key] for record in records]))

    first_attackable = [r["first_attackable_step"] for r in records if r["first_attackable_step"] is not None]
    first_kill = [r["first_kill_step"] for r in records if r["first_kill_step"] is not None]
    def distribution(values: list[float]) -> dict:
        if not values:
            return {"count": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None}
        array = np.asarray(values, dtype=float)
        return {
            "count": len(values), "min": float(np.min(array)),
            "p25": float(np.percentile(array, 25)), "median": float(np.median(array)),
            "p75": float(np.percentile(array, 75)), "max": float(np.max(array)),
        }
    termination_counts = {
        reason: sum(record["termination_reason"] == reason for record in records)
        for reason in ("red_win", "blue_win", "draw_mutual_destruction", "draw_timeout")
    }
    result = {
        "episodes": episodes,
        "first_attackable_rate": len(first_attackable) / episodes,
        "first_attackable_step_mean": float(np.mean(first_attackable)) if first_attackable else None,
        "first_kill_rate": len(first_kill) / episodes,
        "first_kill_step_mean": float(np.mean(first_kill)) if first_kill else None,
        "first_kill_step_distribution": distribution(first_kill),
        "episode_length_mean": mean("episode_length"),
        "episode_length_distribution": distribution([r["episode_length"] for r in records]),
        "red_loss_mean": mean("red_losses"),
        "blue_loss_mean": mean("blue_losses"),
        "mutual_destruction_rate": float(np.mean([
            r["termination_reason"] == "draw_mutual_destruction" for r in records
        ])),
        "red_boundary_loss_mean": mean("red_boundary_losses"),
        "blue_boundary_loss_mean": mean("blue_boundary_losses"),
        "total_boundary_loss_distribution": distribution([
            r["red_boundary_losses"] + r["blue_boundary_losses"] for r in records
        ]),
        "win_rate": mean("red_win"),
        "loss_rate": mean("blue_win"),
        "draw_rate": mean("draw"),
        "termination_counts": termination_counts,
    }
    diagnoses = []
    if result["first_attackable_rate"] == 0.0:
        diagnoses.append("degenerate: no episode ever entered the attack envelope")
    if result["first_kill_rate"] == 0.0:
        diagnoses.append("degenerate: no episode produced a completed three-step lock")
    if result["episode_length_mean"] <= 3.0:
        diagnoses.append("degenerate: episodes terminate almost immediately")
    if result["red_boundary_loss_mean"] + result["blue_boundary_loss_mean"] >= 7.5:
        diagnoses.append("degenerate: nearly all aircraft are lost to the boundary")
    result["diagnoses"] = diagnoses
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset-count", type=int, default=1000)
    parser.add_argument("--straight-episodes", type=int, default=100)
    parser.add_argument("--pursuit-episodes", type=int, default=200)
    parser.add_argument("--output", default="outputs/combat_environment_validation.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = root / "configs/combat_environment.yaml"
    result = {
        "reset_statistics": reset_statistics(config, args.reset_count),
        "straight_vs_straight": run_scenario(config, args.straight_episodes, False, 1_000_000),
        "pursuit_vs_pursuit": run_scenario(config, args.pursuit_episodes, True, 2_000_000),
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
