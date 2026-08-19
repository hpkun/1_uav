"""Read-only action/control stability diagnosis for frozen V1.1."""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from uav_combat.diagnostics.action_stability import (
    bank_compensated_actions,
    fresh_actor,
    trim_a1,
    trim_normal_load,
    vertical_balance,
)
from uav_combat.environment.env import MultiUAVCombatEnv
from uav_combat.environment.control import action_to_control


ACTOR_SEEDS = [101, 202, 303, 404, 505]
METRIC_RANGES = {
    "abs_a0": (0.0, 1.0), "a1": (-1.0, 1.0), "abs_a2": (0.0, 1.0),
    "abs_phi": (0.0, np.pi / 3.0), "nz": (-3.0, 5.0),
    "theta": (-np.pi / 3.0, np.pi / 3.0), "altitude": (0.0, 8500.0),
    "speed": (140.0, 310.0), "vertical_balance": (-4.1, 4.1),
    "theta_dot": (-0.25, 0.25),
}
BOUNDARY_KEYS = [
    "red_low_altitude_losses", "blue_low_altitude_losses",
    "red_high_altitude_losses", "blue_high_altitude_losses",
]


def empty_stream() -> dict:
    return {
        name: {"count": 0, "sum": 0.0, "sum_sq": 0.0, "hist": np.zeros(512, dtype=np.int64)}
        for name in METRIC_RANGES
    }


def stream_add(stream: dict, name: str, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return
    low, high = METRIC_RANGES[name]
    entry = stream[name]
    entry["count"] += int(values.size)
    entry["sum"] += float(values.sum())
    entry["sum_sq"] += float(np.square(values).sum())
    entry["hist"] += np.histogram(np.clip(values, low, high), bins=512, range=(low, high))[0]


def stream_merge(target: dict, source: dict) -> None:
    for name in METRIC_RANGES:
        for key in ("count", "sum", "sum_sq"):
            target[name][key] += source[name][key]
        target[name]["hist"] += source[name]["hist"]


def histogram_quantile(hist: np.ndarray, low: float, high: float, q: float) -> float:
    count = int(hist.sum())
    if count == 0:
        return float("nan")
    index = int(np.searchsorted(np.cumsum(hist), q * max(count - 1, 0), side="right"))
    index = min(index, len(hist) - 1)
    return float(low + (index + 0.5) * (high - low) / len(hist))


def stream_finalize(stream: dict) -> dict:
    result = {}
    for name, entry in stream.items():
        count = entry["count"]
        mean = entry["sum"] / max(count, 1)
        variance = max(entry["sum_sq"] / max(count, 1) - mean * mean, 0.0)
        low, high = METRIC_RANGES[name]
        result[name] = {
            "count": count, "mean": mean, "std": float(np.sqrt(variance)),
            "p10": histogram_quantile(entry["hist"], low, high, 0.10),
            "p50": histogram_quantile(entry["hist"], low, high, 0.50),
            "p90": histogram_quantile(entry["hist"], low, high, 0.90),
        }
    return result


def empty_control_summary() -> dict:
    return {
        "stream": empty_stream(), "balance_count": 0, "balance_negative": 0,
        "balance_below_minus_0_1": 0, "balance_positive": 0,
        "bank_over_30": {"count": 0, "nz_sum": 0.0, "balance_sum": 0.0, "negative": 0},
        "bank_over_45": {"count": 0, "nz_sum": 0.0, "balance_sum": 0.0, "negative": 0},
    }


def add_controls(
    summary: dict, states: list, actions: np.ndarray, rings=None, next_step=0,
    action_config: dict | None = None,
) -> None:
    rows = []
    for index, state in enumerate(states):
        if not state.alive:
            continue
        a0, a1, a2 = map(float, actions[index])
        cfg = action_config or {
            "nx_scale": 2.0, "nz_delta_scale": 2.0, "phi_max": np.pi / 3.0
        }
        control = action_to_control(state, actions[index], cfg)
        phi, nz = control.phi, control.nz
        balance = float(nz * np.cos(phi) - np.cos(state.theta))
        theta_dot = float(9.81 / state.v * balance)
        row = {
            "step": next_step, "aircraft": index, "altitude": state.altitude,
            "theta": state.theta, "phi": phi, "nz": nz, "a1": a1, "a2": a2,
            "vertical_balance": balance, "theta_dot": theta_dot,
        }
        rows.append((a0, a1, a2, phi, nz, state, balance, theta_dot))
        if rings is not None:
            rings[index].append(row)
    if not rows:
        return
    arrays = {
        "abs_a0": np.abs([r[0] for r in rows]), "a1": [r[1] for r in rows],
        "abs_a2": np.abs([r[2] for r in rows]), "abs_phi": np.abs([r[3] for r in rows]),
        "nz": [r[4] for r in rows], "theta": [r[5].theta for r in rows],
        "altitude": [r[5].altitude for r in rows], "speed": [r[5].v for r in rows],
        "vertical_balance": [r[6] for r in rows], "theta_dot": [r[7] for r in rows],
    }
    for name, values in arrays.items():
        stream_add(summary["stream"], name, np.asarray(values))
    balance = np.asarray(arrays["vertical_balance"])
    summary["balance_count"] += len(balance)
    summary["balance_negative"] += int(np.count_nonzero(balance < 0.0))
    summary["balance_below_minus_0_1"] += int(np.count_nonzero(balance < -0.1))
    summary["balance_positive"] += int(np.count_nonzero(balance > 0.0))
    for degrees, key in ((30.0, "bank_over_30"), (45.0, "bank_over_45")):
        mask = np.asarray(arrays["abs_phi"]) > np.deg2rad(degrees)
        conditional = summary[key]
        conditional["count"] += int(mask.sum())
        conditional["nz_sum"] += float(np.asarray(arrays["nz"])[mask].sum())
        conditional["balance_sum"] += float(balance[mask].sum())
        conditional["negative"] += int(np.count_nonzero(balance[mask] < 0.0))


def merge_controls(target: dict, source: dict) -> None:
    stream_merge(target["stream"], source["stream"])
    for key in ("balance_count", "balance_negative", "balance_below_minus_0_1", "balance_positive"):
        target[key] += source[key]
    for bank_key in ("bank_over_30", "bank_over_45"):
        for key in ("count", "nz_sum", "balance_sum", "negative"):
            target[bank_key][key] += source[bank_key][key]


def finalize_controls(summary: dict) -> dict:
    count = max(summary["balance_count"], 1)
    result = stream_finalize(summary["stream"])
    result["balance_fractions"] = {
        "below_zero": summary["balance_negative"] / count,
        "below_minus_0_1": summary["balance_below_minus_0_1"] / count,
        "above_zero": summary["balance_positive"] / count,
    }
    for bank_key in ("bank_over_30", "bank_over_45"):
        entry = summary[bank_key]
        bank_count = max(entry["count"], 1)
        result[bank_key] = {
            "count": entry["count"], "mean_nz": entry["nz_sum"] / bank_count,
            "mean_vertical_balance": entry["balance_sum"] / bank_count,
            "fraction_balance_negative": entry["negative"] / bank_count,
        }
    return result


def empty_rollout() -> dict:
    return {
        "episodes": 0, "episode_lengths": [], "red": empty_control_summary(),
        "blue": empty_control_summary(), "sums": {key: 0 for key in BOUNDARY_KEYS},
        "red_attack_kills": 0, "blue_attack_kills": 0, "red_win": 0,
        "blue_win": 0, "draw": 0, "attackable_episodes": 0,
        "lock_episodes": 0, "kill_episodes": 0, "first_attackable": [],
        "first_lock": [], "first_kill": [], "death_windows": [], "seed_records": [],
    }


def actor_actions(actor, observation: np.ndarray, stochastic: bool) -> np.ndarray:
    tensor = torch.as_tensor(observation, dtype=torch.float32)
    with torch.no_grad():
        actions = actor.sample(tensor)[0] if stochastic else actor.deterministic(tensor)
    return actions.cpu().numpy()


def rollout_worker(task: tuple[str, str, int | None, int, int, bool]) -> dict:
    config_path, mode, actor_seed, seed_start, episodes, collect_windows = task
    torch.set_num_threads(1)
    actor = fresh_actor(actor_seed) if actor_seed is not None else None
    rng = np.random.default_rng(seed_start + 90_000_000)
    result = empty_rollout()
    for episode_offset in range(episodes):
        env_seed = seed_start + episode_offset
        env = MultiUAVCombatEnv(config_path)
        observation, _ = env.reset(env_seed)
        red_rings = [deque(maxlen=101) for _ in range(4)]
        blue_rings = [deque(maxlen=101) for _ in range(4)]
        while True:
            base_red = env.fixed_policy.team_actions(env.red, env.blue)
            base_blue = env.fixed_policy.team_actions(env.blue, env.red)
            if mode == "current_rule":
                red_actions, blue_actions = base_red, base_blue
            elif mode == "bank_rule":
                red_actions = bank_compensated_actions(env.red, base_red)
                blue_actions = bank_compensated_actions(env.blue, base_blue)
            elif mode == "current_vs_straight":
                red_actions, blue_actions = base_red, np.zeros((4, 3), dtype=np.float32)
            elif mode == "bank_vs_straight":
                red_actions = bank_compensated_actions(env.red, base_red)
                blue_actions = np.zeros((4, 3), dtype=np.float32)
            elif mode in ("fresh_deterministic", "fresh_stochastic"):
                red_actions = actor_actions(actor, observation, mode == "fresh_stochastic")
                blue_actions = base_blue
            elif mode == "uniform_random":
                red_actions = rng.uniform(-1.0, 1.0, size=(4, 3)).astype(np.float32)
                blue_actions = base_blue
            else:
                raise ValueError(mode)
            before_red = env.red_alive_mask.copy()
            before_blue = env.blue_alive_mask.copy()
            add_controls(
                result["red"], env.red, red_actions, red_rings, env.steps + 1,
                env.config["action"],
            )
            add_controls(
                result["blue"], env.blue, blue_actions, blue_rings, env.steps + 1,
                env.config["action"],
            )
            observation, _, terminated, truncated, info = env.step(red_actions, blue_actions)
            if collect_windows and len(result["death_windows"]) < 80:
                for team_name, before, after, causes, rings in (
                    ("red", before_red, env.red_alive_mask, env.red_altitude_causes, red_rings),
                    ("blue", before_blue, env.blue_alive_mask, env.blue_altitude_causes, blue_rings),
                ):
                    for index in range(4):
                        if before[index] and not after[index] and causes[index] == "altitude_low":
                            death_step = env.steps
                            window = []
                            for row in rings[index]:
                                aligned = dict(row)
                                aligned["relative_step"] = row["step"] - death_step
                                aligned["team"] = team_name
                                aligned["env_seed"] = env_seed
                                window.append(aligned)
                            result["death_windows"].append(window)
            if terminated or truncated:
                result["episodes"] += 1
                result["episode_lengths"].append(info["episode_length"])
                for key in BOUNDARY_KEYS:
                    result["sums"][key] += info[key]
                for key in ("red_attack_kills", "blue_attack_kills", "red_win", "blue_win", "draw"):
                    result[key] += int(info[key])
                for first_key, list_key, count_key in (
                    ("first_attackable_step", "first_attackable", "attackable_episodes"),
                    ("first_lock_step", "first_lock", "lock_episodes"),
                    ("first_kill_step", "first_kill", "kill_episodes"),
                ):
                    if info[first_key] is not None:
                        result[list_key].append(info[first_key])
                        result[count_key] += 1
                result["seed_records"].append({
                    "actor_seed": actor_seed, "environment_seed": env_seed,
                    "termination_reason": info["termination_reason"],
                })
                break
    return result


def merge_rollouts(results: list[dict]) -> dict:
    merged = empty_rollout()
    for source in results:
        merged["episodes"] += source["episodes"]
        merge_controls(merged["red"], source["red"])
        merge_controls(merged["blue"], source["blue"])
        for key in ("episode_lengths", "first_attackable", "first_lock", "first_kill", "death_windows", "seed_records"):
            merged[key].extend(source[key])
        for key in ("red_attack_kills", "blue_attack_kills", "red_win", "blue_win", "draw", "attackable_episodes", "lock_episodes", "kill_episodes"):
            merged[key] += source[key]
        for key in BOUNDARY_KEYS:
            merged["sums"][key] += source["sums"][key]
    return merged


def quantiles(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "min": None, "p10": None, "p50": None, "p90": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {
        "count": len(values), "min": float(array.min()), "p10": float(np.percentile(array, 10)),
        "p50": float(np.percentile(array, 50)), "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
    }


def finalize_rollout(result: dict) -> dict:
    episodes = max(result["episodes"], 1)
    return {
        "episodes": result["episodes"],
        "episode_length": quantiles(result["episode_lengths"]),
        "episode_length_mean": float(np.mean(result["episode_lengths"])),
        "win_rate": result["red_win"] / episodes, "loss_rate": result["blue_win"] / episodes,
        "draw_rate": result["draw"] / episodes,
        "red_attack_kills_per_episode": result["red_attack_kills"] / episodes,
        "blue_attack_kills_per_episode": result["blue_attack_kills"] / episodes,
        **{f"{key}_per_episode": result["sums"][key] / episodes for key in BOUNDARY_KEYS},
        "attackable_episode_rate": result["attackable_episodes"] / episodes,
        "completed_lock_episode_rate": result["lock_episodes"] / episodes,
        "kill_episode_rate": result["kill_episodes"] / episodes,
        "first_attackable_step": quantiles(result["first_attackable"]),
        "first_lock_step": quantiles(result["first_lock"]),
        "first_kill_step": quantiles(result["first_kill"]),
        "red_control": finalize_controls(result["red"]),
        "blue_control": finalize_controls(result["blue"]),
        "actor_environment_seeds": result["seed_records"],
    }


def action_space_map() -> dict:
    a1_values = np.linspace(-1.0, 1.0, 41)
    a2_values = np.array([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
    summaries = []
    tolerance = 1e-3
    for theta_deg in (-30, -15, 0, 15, 30):
        theta = np.deg2rad(theta_deg)
        for speed in (225.0, 260.0):
            grid_a1, grid_a2 = np.meshgrid(a1_values, a2_values)
            balance = vertical_balance(theta, grid_a1, grid_a2)
            theta_dot = 9.81 / speed * balance
            summaries.append({
                "theta_deg": theta_deg, "speed": speed, "samples": int(balance.size),
                "fraction_negative": float(np.mean(balance < -tolerance)),
                "fraction_positive": float(np.mean(balance > tolerance)),
                "fraction_near_zero": float(np.mean(np.abs(balance) <= tolerance)),
                "theta_dot_min": float(theta_dot.min()), "theta_dot_max": float(theta_dot.max()),
                "a1_trim_by_a2": [
                    {"a2": float(a2), "phi_deg": float(60.0 * a2),
                     "nz_trim": float(trim_normal_load(theta, np.pi / 3.0 * a2)),
                     "a1_trim": float(trim_a1(theta, np.pi / 3.0 * a2))}
                    for a2 in a2_values
                ],
            })
    level_special = []
    for phi_deg in (0, 15, 30, 45, 60):
        nz = float(trim_normal_load(0.0, np.deg2rad(phi_deg)))
        level_special.append({"phi_deg": phi_deg, "nz_trim": nz, "a1_trim": (nz - 1.0) / 4.0})
    return {"balance_tolerance": tolerance, "grid_summaries": summaries, "level_flight_trim": level_special}


def exact_stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()), "std": float(values.std()), "p01": float(np.percentile(values, 1)),
        "p10": float(np.percentile(values, 10)), "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)), "p99": float(np.percentile(values, 99)),
        "min": float(values.min()), "max": float(values.max()),
    }


def actor_initialization_diagnosis(config_path: Path) -> dict:
    observations, theta = [], []
    env = MultiUAVCombatEnv(config_path)
    for seed in range(1000):
        observation, _ = env.reset(20_000_000 + seed)
        observations.append(observation)
        theta.extend(state.theta for state in env.red)
    tensor = torch.as_tensor(np.concatenate(observations), dtype=torch.float32)
    theta = np.asarray(theta)
    result = {}
    for seed in ACTOR_SEEDS:
        actor = fresh_actor(seed)
        with torch.no_grad():
            distribution = actor.distribution(tensor)
            raw_mean = distribution.mean.cpu().numpy()
            std = distribution.stddev.cpu().numpy()
            log_std = np.log(std)
            actions = torch.tanh(distribution.mean).cpu().numpy()
        balance = vertical_balance(theta, actions[:, 1], actions[:, 2])
        result[str(seed)] = {
            "samples": len(actions),
            "raw_mean": [exact_stats(raw_mean[:, i]) for i in range(3)],
            "log_std": [exact_stats(log_std[:, i]) for i in range(3)],
            "std": [exact_stats(std[:, i]) for i in range(3)],
            "tanh_action": [exact_stats(actions[:, i]) for i in range(3)],
            "corr_a1_abs_a2": float(np.corrcoef(actions[:, 1], np.abs(actions[:, 2]))[0, 1]),
            "initial_balance_fractions": {
                "below_zero": float(np.mean(balance < 0.0)),
                "below_minus_0_1": float(np.mean(balance < -0.1)),
                "above_zero": float(np.mean(balance > 0.0)),
            },
        }
    return result


def empty_short() -> dict:
    return {key: [] for key in ("altitude_change", "min_altitude", "max_altitude", "theta_change", "max_abs_theta")}


def short_worker(task: tuple[str, str, int | None, int, int]) -> dict:
    config_path, mode, actor_seed, seed_start, episodes = task
    torch.set_num_threads(1)
    actor = fresh_actor(actor_seed) if actor_seed is not None else None
    rng = np.random.default_rng(seed_start + 80_000_000)
    result = empty_short()
    for offset in range(episodes):
        env = MultiUAVCombatEnv(config_path)
        observation, _ = env.reset(seed_start + offset)
        initial_altitude = np.array([state.altitude for state in env.red])
        initial_theta = np.array([state.theta for state in env.red])
        minimum, maximum = initial_altitude.copy(), initial_altitude.copy()
        max_abs_theta = np.abs(initial_theta)
        for _ in range(100):
            if mode == "fresh_stochastic":
                red_actions = actor_actions(actor, observation, True)
            else:
                red_actions = rng.uniform(-1.0, 1.0, size=(4, 3)).astype(np.float32)
            observation, _, terminated, truncated, _ = env.step(red_actions)
            altitudes = np.array([state.altitude for state in env.red])
            thetas = np.array([state.theta for state in env.red])
            minimum = np.minimum(minimum, altitudes)
            maximum = np.maximum(maximum, altitudes)
            max_abs_theta = np.maximum(max_abs_theta, np.abs(thetas))
            if terminated or truncated:
                break
        final_altitude = np.array([state.altitude for state in env.red])
        final_theta = np.array([state.theta for state in env.red])
        result["altitude_change"].extend(final_altitude - initial_altitude)
        result["min_altitude"].extend(minimum)
        result["max_altitude"].extend(maximum)
        result["theta_change"].extend(final_theta - initial_theta)
        result["max_abs_theta"].extend(max_abs_theta)
    return result


def finalize_short(results: list[dict]) -> dict:
    merged = empty_short()
    for result in results:
        for key in merged:
            merged[key].extend(result[key])
    minimum = np.asarray(merged["min_altitude"])
    maximum = np.asarray(merged["max_altitude"])
    return {
        **{key: exact_stats(np.asarray(values)) for key, values in merged.items()},
        "low_altitude_reach_fraction": {str(level): float(np.mean(minimum < level)) for level in (2000, 1500, 1000, 600)},
        "high_altitude_reach_fraction": {str(level): float(np.mean(maximum > level)) for level in (4000, 4500, 5000, 5400)},
    }


def death_alignment(windows: list[list[dict]], minimum_events: int = 50) -> dict:
    selected = windows[:max(minimum_events, 50)]
    result = {"events": len(selected), "relative_steps": {}}
    for relative in (-100, -75, -50, -25, -10, -5, -1, 0):
        rows = [row for window in selected for row in window if row["relative_step"] == relative]
        result["relative_steps"][str(relative)] = {
            "samples": len(rows),
            **{key: float(np.mean([row[key] for row in rows])) if rows else None for key in (
                "altitude", "theta", "phi", "nz", "a1", "a2", "vertical_balance", "theta_dot"
            )},
        }
    return result


def save_death_csvs(directory: Path, windows: list[list[dict]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for number, window in enumerate(windows[:5], 1):
        path = directory / f"rule_low_altitude_death_{number:02d}.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(window[0]))
            writer.writeheader(); writer.writerows(window)


def trim_relative_offline(config_path: Path) -> dict:
    """Category-B-only comparison on the same real stochastic state/action samples."""
    observations, theta = [], []
    env = MultiUAVCombatEnv(config_path)
    for reset_seed in range(1000):
        observation, _ = env.reset(40_000_000 + reset_seed)
        observations.append(observation)
        theta.extend(state.theta for state in env.red)
    tensor = torch.as_tensor(np.concatenate(observations), dtype=torch.float32)
    theta = np.asarray(theta)
    comparisons = {}
    for seed in ACTOR_SEEDS:
        actor = fresh_actor(seed)
        torch.manual_seed(seed + 1_000_000)
        with torch.no_grad():
            actions = actor.sample(tensor)[0].cpu().numpy()
        a1, a2 = actions[:, 1], actions[:, 2]
        current = vertical_balance(theta, a1, a2)
        phi = np.pi / 3.0 * a2
        alternative = 2.0 * a1 * np.cos(phi)
        comparisons[str(seed)] = {
            "samples": len(actions),
            "current": {"mean": float(current.mean()), "std": float(current.std()),
                        "below_zero": float(np.mean(current < 0)), "below_minus_0_1": float(np.mean(current < -0.1))},
            "trim_relative_k2": {"mean": float(alternative.mean()), "std": float(alternative.std()),
                                 "below_zero": float(np.mean(alternative < 0)), "below_minus_0_1": float(np.mean(alternative < -0.1))},
        }
    return {"same_state_action_sample_comparison": comparisons,
            "note": "Offline diagnostic only; canonical action mapping was not changed."}


def _short_result(initial_altitude, initial_theta, env, minimum, maximum, max_abs_theta):
    final_altitude = np.array([state.altitude for state in env.red])
    final_theta = np.array([state.theta for state in env.red])
    return {
        "altitude_change": list(final_altitude - initial_altitude),
        "min_altitude": list(minimum), "max_altitude": list(maximum),
        "theta_change": list(final_theta - initial_theta), "max_abs_theta": list(max_abs_theta),
    }


def trim_relative_short_rollout(config_path: Path, episodes: int = 100) -> dict:
    """Replay identical fresh-stochastic action sequences under two diagnostic semantics."""
    current_results, alternative_results = [], []
    for episode in range(episodes):
        actor_seed = ACTOR_SEEDS[episode % len(ACTOR_SEEDS)]
        env_seed = 41_000_000 + episode
        actor = fresh_actor(actor_seed)
        torch.manual_seed(actor_seed * 100_000 + episode)
        current_env = MultiUAVCombatEnv(config_path)
        observation, _ = current_env.reset(env_seed)
        raw_actions = []
        initial_altitude = np.array([state.altitude for state in current_env.red])
        initial_theta = np.array([state.theta for state in current_env.red])
        minimum, maximum = initial_altitude.copy(), initial_altitude.copy()
        max_abs_theta = np.abs(initial_theta)
        for _ in range(100):
            actions = actor_actions(actor, observation, True)
            raw_actions.append(actions.copy())
            observation, _, terminated, truncated, _ = current_env.step(actions)
            altitudes = np.array([state.altitude for state in current_env.red])
            thetas = np.array([state.theta for state in current_env.red])
            minimum, maximum = np.minimum(minimum, altitudes), np.maximum(maximum, altitudes)
            max_abs_theta = np.maximum(max_abs_theta, np.abs(thetas))
            if terminated or truncated:
                break
        current_results.append(_short_result(
            initial_altitude, initial_theta, current_env, minimum, maximum, max_abs_theta
        ))

        alternative_env = MultiUAVCombatEnv(config_path)
        alternative_env.reset(env_seed)
        initial_altitude = np.array([state.altitude for state in alternative_env.red])
        initial_theta = np.array([state.theta for state in alternative_env.red])
        minimum, maximum = initial_altitude.copy(), initial_altitude.copy()
        max_abs_theta = np.abs(initial_theta)
        for actions in raw_actions:
            equivalent = actions.copy()
            for index, state in enumerate(alternative_env.red):
                if state.alive:
                    phi = np.pi / 3.0 * equivalent[index, 2]
                    nz = float(trim_normal_load(state.theta, phi)) + 2.0 * equivalent[index, 1]
                    equivalent[index, 1] = np.clip((nz - 1.0) / 4.0, -1.0, 1.0)
            _, _, terminated, truncated, _ = alternative_env.step(equivalent)
            altitudes = np.array([state.altitude for state in alternative_env.red])
            thetas = np.array([state.theta for state in alternative_env.red])
            minimum, maximum = np.minimum(minimum, altitudes), np.maximum(maximum, altitudes)
            max_abs_theta = np.maximum(max_abs_theta, np.abs(thetas))
            if terminated or truncated:
                break
        alternative_results.append(_short_result(
            initial_altitude, initial_theta, alternative_env, minimum, maximum, max_abs_theta
        ))
    return {
        "episodes": episodes,
        "same_raw_action_sequences": True,
        "current_mapping": finalize_short(current_results),
        "trim_relative_k2": finalize_short(alternative_results),
    }


def comparison_table(modes: dict) -> list[dict]:
    names = [
        "current_rule", "fresh_deterministic_actor", "fresh_stochastic_actor",
        "uniform_random_actor", "bank_compensated_probe",
    ]
    rows = []
    for name in names:
        mode, control = modes[name], modes[name]["red_control"]
        rows.append({
            "mode": name,
            "red_low_altitude_loss_per_episode": mode["red_low_altitude_losses_per_episode"],
            "red_high_altitude_loss_per_episode": mode["red_high_altitude_losses_per_episode"],
            "red_attack_kill_per_episode": mode["red_attack_kills_per_episode"],
            "episode_length_mean": mode["episode_length_mean"],
            "mean_abs_phi": control["abs_phi"]["mean"], "mean_nz": control["nz"]["mean"],
            "mean_vertical_balance": control["vertical_balance"]["mean"],
            "fraction_balance_below_zero": control["balance_fractions"]["below_zero"],
            "fraction_balance_below_minus_0_1": control["balance_fractions"]["below_minus_0_1"],
        })
    return rows


def run_tasks(executor, tasks):
    return list(executor.map(rollout_worker, tasks, chunksize=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(5, os.cpu_count() or 1))
    parser.add_argument("--quick", action="store_true", help="diagnostic plumbing check only")
    parser.add_argument("--output", default="outputs/action_stability_diagnosis.json")
    parser.add_argument("--timeseries-dir", default="outputs/action_stability_timeseries")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = root / "configs/combat_environment.yaml"
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    controller_episodes = 1 if args.quick else 50
    actor_episodes = 1 if args.quick else 40
    short_episodes = 2 if args.quick else 100

    modes: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        rule_tasks = [(str(config), "current_rule", None, 30_000_000 + i * 50, controller_episodes, True) for i in range(4)]
        bank_tasks = [(str(config), "bank_rule", None, 30_000_000 + i * 50, controller_episodes, False) for i in range(4)]
        current_straight_tasks = [(str(config), "current_vs_straight", None, 31_000_000 + i * 50, controller_episodes, False) for i in range(4)]
        bank_straight_tasks = [(str(config), "bank_vs_straight", None, 31_000_000 + i * 50, controller_episodes, False) for i in range(4)]
        mode_raw = {
            "current_rule": merge_rollouts(run_tasks(executor, rule_tasks)),
            "bank_compensated_probe": merge_rollouts(run_tasks(executor, bank_tasks)),
            "current_pursuit_vs_straight": merge_rollouts(run_tasks(executor, current_straight_tasks)),
            "bank_compensated_vs_straight": merge_rollouts(run_tasks(executor, bank_straight_tasks)),
        }
        det_tasks = [(str(config), "fresh_deterministic", seed, 32_000_000 + i * 40, actor_episodes, False) for i, seed in enumerate(ACTOR_SEEDS)]
        stoch_tasks = [(str(config), "fresh_stochastic", seed, 33_000_000 + i * 40, actor_episodes, False) for i, seed in enumerate(ACTOR_SEEDS)]
        uniform_tasks = [(str(config), "uniform_random", None, 34_000_000 + i * 40, actor_episodes, False) for i in range(5)]
        det_results = run_tasks(executor, det_tasks)
        stoch_results = run_tasks(executor, stoch_tasks)
        uniform_results = run_tasks(executor, uniform_tasks)
        mode_raw["fresh_deterministic_actor"] = merge_rollouts(det_results)
        mode_raw["fresh_stochastic_actor"] = merge_rollouts(stoch_results)
        mode_raw["uniform_random_actor"] = merge_rollouts(uniform_results)
        for name, raw in mode_raw.items():
            modes[name] = finalize_rollout(raw)

        stochastic_short_tasks = [(str(config), "fresh_stochastic", seed, 35_000_000 + i * 100, short_episodes) for i, seed in enumerate(ACTOR_SEEDS)]
        uniform_short_tasks = [(str(config), "uniform", None, 36_000_000 + i * 100, short_episodes) for i in range(5)]
        short_stochastic = finalize_short(list(executor.map(short_worker, stochastic_short_tasks)))
        short_uniform = finalize_short(list(executor.map(short_worker, uniform_short_tasks)))

    initialization = actor_initialization_diagnosis(config)
    rule_raw = mode_raw["current_rule"]
    save_death_csvs(root / args.timeseries_dir, rule_raw["death_windows"])
    death_evidence = death_alignment(rule_raw["death_windows"])

    rule_low = modes["current_rule"]["red_low_altitude_losses_per_episode"] + modes["current_rule"]["blue_low_altitude_losses_per_episode"]
    bank_low = modes["bank_compensated_probe"]["red_low_altitude_losses_per_episode"] + modes["bank_compensated_probe"]["blue_low_altitude_losses_per_episode"]
    stochastic_low = modes["fresh_stochastic_actor"]["red_low_altitude_losses_per_episode"]
    if rule_low > 1.0 and bank_low < 0.5 * rule_low and stochastic_low > 1.0:
        category = "B"
        trim_probe = trim_relative_offline(config)
        trim_probe["same_action_sequence_100_step_rollout"] = trim_relative_short_rollout(
            config, episodes=10 if args.quick else 100
        )
    elif rule_low > 1.0 and bank_low < 0.5 * rule_low and stochastic_low <= 1.0:
        category = "A"
        trim_probe = None
    else:
        category = "C"
        trim_probe = None

    output = {
        "frozen_environment": {
            "dt": 0.1, "action_mapping": ["nx=2*a0", "nz=1+4*a1", "phi=pi/3*a2"],
            "training_performed": False, "checkpoint_loaded": False,
        },
        "action_space_stability_map": action_space_map(),
        "actor_initialization": initialization,
        "rollout_modes": modes,
        "actor_seed_breakdown": {
            "fresh_deterministic": {
                str(seed): finalize_rollout(result) for seed, result in zip(ACTOR_SEEDS, det_results)
            },
            "fresh_stochastic": {
                str(seed): finalize_rollout(result) for seed, result in zip(ACTOR_SEEDS, stoch_results)
            },
        },
        "mode_comparison_table": comparison_table(modes),
        "rule_low_altitude_death_alignment": death_evidence,
        "fresh_stochastic_100_step_stability": short_stochastic,
        "uniform_random_100_step_stability": short_uniform,
        "causal_classification": {
            "category": category, "rule_total_low_losses_per_episode": rule_low,
            "bank_compensated_total_low_losses_per_episode": bank_low,
            "fresh_stochastic_red_low_losses_per_episode": stochastic_low,
        },
        "trim_relative_diagnostic": trim_probe,
    }
    path = root / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(path), "category": category,
        "rule_low_per_episode": rule_low, "bank_low_per_episode": bank_low,
        "stochastic_red_low_per_episode": stochastic_low,
    }, indent=2))


if __name__ == "__main__":
    main()
