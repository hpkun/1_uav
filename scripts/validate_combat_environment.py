"""Reproducible V1.3 environment validation without learning."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import numpy as np

from uav_combat.environment.env import MultiUAVCombatEnv
from uav_combat.environment.arena import boundary_cost, horizontal_safety_severity
from uav_combat.environment.control import action_to_control
from uav_combat.environment.geometry import engagement_geometry
from uav_combat.environment.reward import combined_potentials
from uav_combat.math_utils import wrap_angle
from uav_combat.models import AircraftState
from uav_combat.diagnostics.action_stability import vertical_balance
from diagnose_action_stability import (
    ACTOR_SEEDS, finalize_rollout, finalize_short, merge_rollouts,
    rollout_worker, short_worker,
)


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
    """Simple 5-second symmetric lateral break through the public maneuver helper."""
    actions = []
    for index, own in enumerate(env.red):
        if not own.alive:
            actions.append(np.zeros(3, dtype=np.float32))
            continue
        offset = np.deg2rad(30.0) if index < 2 else -np.deg2rad(30.0)
        desired_heading = wrap_angle(nominal_heading + offset)
        actions.append(env.fixed_policy.safe_action_toward(own, desired_heading, 0.0, 260.0))
    return np.stack(actions)


def tail_blue_actions(env: MultiUAVCombatEnv, nominal_heading: float) -> np.ndarray:
    """Diagnostic-only straight merge followed by a deterministic level turn."""
    if env.steps < 120:
        return np.zeros((4, 3), dtype=np.float32)
    actions = []
    for own in env.blue:
        if not own.alive:
            actions.append(np.zeros(3, dtype=np.float32))
            continue
        desired_heading = wrap_angle(nominal_heading + np.deg2rad(60.0))
        actions.append(env.fixed_policy.safe_action_toward(
            own, desired_heading, 0.0, 260.0
        ))
    return np.stack(actions)


def run_episode(task: tuple[str, str, int]) -> dict:
    config, scenario, seed = task
    env = MultiUAVCombatEnv(config)
    _, reset_info = env.reset(seed)
    shaping_values: list[float] = []
    event_values: list[float] = []
    boundary_cost_values: list[float] = []
    boundary_shaping_values: list[float] = []
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
        elif scenario == "tail":
            red_actions = env.fixed_policy.team_actions(env.red, env.blue)
            blue_actions = tail_blue_actions(
                env, wrap_angle(reset_info["radial_angle"] + np.pi)
            )
        else:
            raise ValueError(f"unknown scenario: {scenario}")
        _, _, terminated, truncated, info = env.step(red_actions, blue_actions)
        shaping_values.extend(map(float, info["shaping_rewards"]))
        event_values.extend(map(float, info["event_rewards"]))
        boundary_cost_values.extend(map(float, info["boundary_cost"]))
        boundary_shaping_values.extend(map(float, info["boundary_shaping_rewards"]))
        if terminated or truncated:
            record = dict(info)
            record["_shaping_values"] = shaping_values
            record["_event_values"] = event_values
            record["_boundary_cost_values"] = boundary_cost_values
            record["_boundary_shaping_values"] = boundary_shaping_values
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
    boundary_cost_values = [
        value for record in records for value in record["_boundary_cost_values"]
    ]
    boundary_shaping_values = [
        value for record in records for value in record["_boundary_shaping_values"]
    ]
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
        "combat_deaths_total": red_attack_kills + blue_attack_kills,
        "total_deaths": total_deaths,
        "red_attack_kills_mean": mean("red_attack_kills"),
        "blue_attack_kills_mean": mean("blue_attack_kills"),
        "red_boundary_loss_mean": mean("red_boundary_losses"),
        "blue_boundary_loss_mean": mean("blue_boundary_losses"),
        **{f"{key}_total": sum(r[key] for r in records) for key in boundary_keys},
        **{f"{key}_mean": mean(key) for key in boundary_keys},
        "combat_kill_fraction": (red_attack_kills + blue_attack_kills) / max(total_deaths, 1),
        "boundary_deaths_total": sum(
            sum(r[key] for r in records) for key in boundary_keys
        ),
        "win_rate": mean("red_win"), "loss_rate": mean("blue_win"),
        "draw_rate": mean("draw"), "termination_counts": termination_counts,
        "shaping_reward_statistics": reward_statistics(shaping_values),
        "event_reward_statistics": reward_statistics(event_values),
        "boundary_cost_statistics": reward_statistics(boundary_cost_values),
        "boundary_shaping_statistics": reward_statistics(boundary_shaping_values),
    }
    result["boundary_death_fraction"] = result["boundary_deaths_total"] / max(total_deaths, 1)
    diagnoses = []
    if scenario == "flank" and not first_attackable:
        diagnoses.append("degenerate: flank baseline never entered the attack envelope")
    if scenario == "flank" and not first_kill:
        diagnoses.append("degenerate: flank baseline never completed maneuver-lock-kill")
    if scenario in ("rule", "flank") and (
        result["red_boundary_loss_mean"] + result["blue_boundary_loss_mean"] > 0.5
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


def run_action_stability(config: Path, workers: int) -> tuple[dict, dict]:
    """Run five-seed 200-full and 500-short stochastic/uniform regressions."""
    stochastic_tasks = [
        (str(config), "fresh_stochastic", seed, 40_000_000 + i * 40, 40, False)
        for i, seed in enumerate(ACTOR_SEEDS)
    ]
    uniform_tasks = [
        (str(config), "uniform_random", None, 41_000_000 + i * 40, 40, False)
        for i in range(5)
    ]
    stochastic_short_tasks = [
        (str(config), "fresh_stochastic", seed, 42_000_000 + i * 100, 100)
        for i, seed in enumerate(ACTOR_SEEDS)
    ]
    uniform_short_tasks = [
        (str(config), "uniform", None, 43_000_000 + i * 100, 100)
        for i in range(5)
    ]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        stochastic_full = finalize_rollout(merge_rollouts(
            list(executor.map(rollout_worker, stochastic_tasks))
        ))
        uniform_full = finalize_rollout(merge_rollouts(
            list(executor.map(rollout_worker, uniform_tasks))
        ))
        stochastic_short = finalize_short(
            list(executor.map(short_worker, stochastic_short_tasks))
        )
        uniform_short = finalize_short(
            list(executor.map(short_worker, uniform_short_tasks))
        )
    return (
        {"full_episodes": stochastic_full, "short_100_step_rollouts": stochastic_short},
        {"full_episodes": uniform_full, "short_100_step_rollouts": uniform_short},
    )


def fresh_stochastic_short(config: Path, workers: int) -> dict:
    tasks = [
        (str(config), "fresh_stochastic", seed, 42_000_000 + i * 100, 100)
        for i, seed in enumerate(ACTOR_SEEDS)
    ]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return finalize_short(list(executor.map(short_worker, tasks)))


def recovery_trajectory(config: Path, fraction: float, speed: float, angle: float) -> dict:
    env = MultiUAVCombatEnv(config)
    radius = env.radius * fraction
    outward = np.array([np.cos(angle), np.sin(angle)], dtype=float)
    own = AircraftState(
        radius * outward[0], radius * outward[1], -3000.0,
        speed, 0.0, angle, True,
    )
    target_radius = env.radius + 5000.0
    target = AircraftState(
        target_radius * outward[0], target_radius * outward[1], -3000.0,
        speed, 0.0, angle, True,
    )
    maximum_radius = radius
    non_outward_step = None
    clear_turn_step = None
    reenter_step = None
    crossed = False
    for step in range(1, env.max_steps + 1):
        action = env.fixed_policy.action(own, [target])
        control = action_to_control(own, action, env.config["action"])
        own = env.integrator.step(own, control, env.dynamics, env.spec)
        current_radius = float(np.hypot(own.x, own.y))
        maximum_radius = max(maximum_radius, current_radius)
        radial_velocity = float(np.dot(
            np.array([own.x, own.y]), own.velocity_vector()[:2]
        ) / max(current_radius, 1e-12))
        if non_outward_step is None and radial_velocity <= 0.0:
            non_outward_step = step
        if clear_turn_step is None and abs(wrap_angle(own.psi - angle)) >= np.deg2rad(30.0):
            clear_turn_step = step
        if non_outward_step is not None and current_radius <= (
            env.radius * env.config["battlefield"]["horizontal_soft_fraction"]
        ):
            reenter_step = step
            break
        if current_radius > env.radius:
            crossed = True
            break
    return {
        "initial_fraction": fraction, "initial_speed": speed,
        "radial_angle": angle, "max_radius": maximum_radius,
        "crossed_hard_boundary": crossed,
        "steps_to_radial_velocity_nonpositive": non_outward_step,
        "steps_to_heading_change_30deg": clear_turn_step,
        "steps_to_reenter_soft_region": reenter_step,
        "initial_severity": horizontal_safety_severity(
            AircraftState(radius * outward[0], radius * outward[1], -3000.0,
                          speed, 0.0, angle), env.config["battlefield"]
        ),
        "initial_recovery_speed": env.fixed_policy.recovery_speed(
            AircraftState(radius * outward[0], radius * outward[1], -3000.0,
                          speed, 0.0, angle), 260.0
        ),
    }


def arena_recovery_stress(config: Path) -> dict:
    cases = [
        recovery_trajectory(config, fraction, speed, angle)
        for fraction in (0.7, 0.8, 0.9)
        for speed in (225.0, 260.0, 300.0)
        for angle in (0.0, 2.0 * np.pi / 3.0, -2.0 * np.pi / 3.0)
    ]
    benchmark = [case for case in cases if case["initial_speed"] <= 260.0]
    return {
        "cases": cases,
        "case_count": len(cases),
        "hard_crossings": sum(case["crossed_hard_boundary"] for case in cases),
        "benchmark_225_260_hard_crossings": sum(
            case["crossed_hard_boundary"] for case in benchmark
        ),
        "all_cases_clear_turn_before_hard": all(
            case["steps_to_heading_change_30deg"] is not None for case in cases
        ),
        "benchmark_cases_safe": all(
            not case["crossed_hard_boundary"]
            and case["steps_to_radial_velocity_nonpositive"] is not None
            for case in benchmark
        ),
    }


def boundary_potential_validation(config: Path) -> dict:
    env = MultiUAVCombatEnv(config)
    battlefield = env.config["battlefield"]
    radius = env.radius
    empty = [AircraftState(0, 0, -3000, 225, 0, 0, False) for _ in range(4)]
    def potential(x=0.0, altitude=3000.0):
        own = AircraftState(x, 0, -altitude, 225, 0, 0)
        team = [own] + [state.copy() for state in empty[:3]]
        return float(combined_potentials(
            team, empty, 8000.0, battlefield, 1.0
        )[2][0])
    costs = {
        "inside": boundary_cost(AircraftState(0.5 * radius, 0, -3000, 225, 0, 0), battlefield),
        "soft": boundary_cost(AircraftState(0.65 * radius, 0, -3000, 225, 0, 0), battlefield),
        "middle": boundary_cost(AircraftState(0.8 * radius, 0, -3000, 225, 0, 0), battlefield),
        "hard": boundary_cost(AircraftState(radius, 0, -3000, 225, 0, 0), battlefield),
    }
    checks = {
        "inside_zero": costs["inside"] == 0.0,
        "soft_zero": costs["soft"] == 0.0,
        "middle_between": 0.0 < costs["middle"] < 1.0,
        "hard_one": costs["hard"] == 1.0,
        "outward_lowers_potential": potential(0.81 * radius) < potential(0.8 * radius),
        "inward_raises_potential": potential(0.79 * radius) > potential(0.8 * radius),
        "lower_outward_lowers_potential": potential(0, 550) < potential(0, 650),
        "upper_outward_lowers_potential": potential(0, 7950) < potential(0, 7850),
    }
    return {"costs": costs, "checks": checks, "passed": all(checks.values())}


def invariant_checks(config: Path) -> dict:
    env = MultiUAVCombatEnv(config)
    observation, _ = env.reset(12345)
    own = AircraftState(0, 0, -3000, 225, 0, 0)
    head_on = AircraftState(1000, 0, -3000, 225, 0, np.pi)
    tail = AircraftState(1000, 0, -3000, 225, 0, 0)
    theta = np.deg2rad(np.array([-30, 0, 30]))[:, None]
    a2 = np.linspace(-1, 1, 9)[None, :]
    trim_error = np.max(np.abs(vertical_balance(theta, 0.0, a2)))
    return {
        "trim_max_abs_vertical_balance": float(trim_error),
        "trim_passed": bool(trim_error < 1e-12),
        "observation_shape": list(observation.shape),
        "observation_finite": bool(np.all(np.isfinite(observation))),
        "observation_config_is_derived": set(env.config["observation"]) == {
            "speed_center", "speed_scale", "relative_velocity_scale"
        },
        "head_on_not_attackable": not env.weapon.attackable(
            engagement_geometry(own, head_on)
        ),
        "tail_attackable": env.weapon.attackable(
            engagement_geometry(own, tail)
        ),
        "lock_steps_required": env.weapon.lock_steps_required,
    }


def all_finite(value) -> bool:
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite(item) for item in value)
    return not isinstance(value, (float, np.floating)) or np.isfinite(value)


def acceptance(result: dict) -> dict:
    short = result["fresh_stochastic_100_step_stability"]
    rule = result["rule_vs_rule"]
    flank = result["flank_then_pursuit_vs_fixed_blue"]
    checks = {
        "A_trim_relative_action_regression": result["invariant_checks"]["trim_passed"],
        "B_fresh_stochastic_no_systematic_vertical_drift": (
            short["altitude_change"]["mean"] > -25.0
            and short["altitude_change"]["mean"] < 25.0
            and abs(short["theta_change"]["mean"]) < 0.025
        ),
        "C_fixed_rule_arena_trajectory_safety": result["arena_recovery_stress"]["benchmark_cases_safe"],
        "D_rule_no_mass_boundary_self_destruction": (
            rule["red_boundary_loss_mean"] + rule["blue_boundary_loss_mean"] <= 0.5
        ),
        "E_flank_no_mass_boundary_self_destruction": (
            flank["red_boundary_loss_mean"] + flank["blue_boundary_loss_mean"] <= 0.5
        ),
        "F_canonical_combat_chain_reachable": any(
            result[key]["attackable_episodes"] > 0
            and result[key]["completed_lock_episodes"] > 0
            and result[key]["kill_episodes"] > 0
            for key in (
                "rule_vs_rule", "flank_then_pursuit_vs_fixed_blue",
            )
        ),
        "G_weapon_geometry_and_lock_regression": (
            result["invariant_checks"]["head_on_not_attackable"]
            and result["invariant_checks"]["tail_attackable"]
            and result["invariant_checks"]["lock_steps_required"] == 3
        ),
        "H_observation_regression": (
            result["invariant_checks"]["observation_shape"] == [4, 54]
            and result["invariant_checks"]["observation_finite"]
            and result["invariant_checks"]["observation_config_is_derived"]
        ),
        "I_boundary_potential_direction": result["boundary_potential_validation"]["passed"],
        "J_all_values_finite": all_finite(result),
    }
    return {"checks": checks, "passed": all(checks.values())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset-count", type=int, default=1000)
    parser.add_argument("--straight-episodes", type=int, default=100)
    parser.add_argument("--rule-episodes", type=int, default=200)
    parser.add_argument("--flank-episodes", type=int, default=200)
    parser.add_argument("--tail-episodes", type=int, default=200)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--output", default="outputs/combat_environment_validation_v1_3.json")
    args = parser.parse_args()
    if min(args.reset_count, args.straight_episodes, args.rule_episodes, args.flank_episodes, args.tail_episodes, args.workers) <= 0:
        raise ValueError("all validation counts and workers must be positive")
    root = Path(__file__).resolve().parents[1]
    config = root / "configs/combat_environment.yaml"
    result = {
        "reset_statistics": reset_statistics(config, args.reset_count),
        "invariant_checks": invariant_checks(config),
        "boundary_potential_validation": boundary_potential_validation(config),
        "arena_recovery_stress": arena_recovery_stress(config),
        "fresh_stochastic_100_step_stability": fresh_stochastic_short(config, args.workers),
        "straight_vs_straight": run_scenario(
            config, "straight", args.straight_episodes, 1_000_000, args.workers
        ),
        "rule_vs_rule": run_scenario(
            config, "rule", args.rule_episodes, 2_000_000, args.workers
        ),
        "flank_then_pursuit_vs_fixed_blue": run_scenario(
            config, "flank", args.flank_episodes, 3_000_000, args.workers
        ),
        "tail_acquisition_diagnostic": run_scenario(
            config, "tail", args.tail_episodes, 4_000_000, args.workers
        ),
    }
    result["acceptance"] = acceptance(result)
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
