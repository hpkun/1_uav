"""Read-only V1.4 500k policy-trajectory and reward-landscape diagnosis."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import yaml

from uav_combat.environment.control import action_to_control
from uav_combat.environment.env import MultiUAVCombatEnv
from uav_combat.environment.geometry import engagement_geometry
from uav_combat.environment.reward import tactical_potentials
from uav_combat.madsac.actor import SharedSquashedGaussianActor


EVALUATION_SEEDS = list(range(10_000_000, 10_000_020))
TRAJECTORY_SEEDS = set(EVALUATION_SEEDS[:5])
TIME_WINDOWS = ((0, 100), (100, 200), (200, 400), (400, 600), (600, 1000))
TIME_WINDOW_NAMES = ("0-10s", "10-20s", "20-40s", "40-60s", "60-100s")
DISTANCE_THRESHOLDS = (8000.0, 6000.0, 4000.0, 3000.0, 2000.0, 1500.0)
COUNTERFACTUAL_HORIZONS = (1, 10, 50, 100)


def distribution(values: list[float] | np.ndarray) -> dict:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {
            "count": 0, "mean": None, "std": None, "p10": None,
            "median": None, "p90": None, "min": None, "max": None,
        }
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def fraction(values: list[float] | np.ndarray, predicate: Callable[[np.ndarray], np.ndarray]) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.mean(predicate(array))) if array.size else 0.0


def score_components(attacker, target, distance_scale: float) -> dict[str, float]:
    geometry = engagement_geometry(attacker, target)
    range_score = float(np.clip(1.0 - geometry.distance / distance_scale, 0.0, 1.0))
    attack_score = float((1.0 + np.cos(geometry.attack_angle)) / 2.0)
    escape_score = float((1.0 + np.cos(geometry.escape_angle)) / 2.0)
    return {
        "distance": geometry.distance,
        "attack_angle": geometry.attack_angle,
        "escape_angle": geometry.escape_angle,
        "range_score": range_score,
        "attack_score": attack_score,
        "escape_score": escape_score,
        "product": range_score * attack_score * escape_score,
    }


def closing_speed(attacker, target) -> float:
    displacement = np.array([
        target.x - attacker.x, target.y - attacker.y, target.z - attacker.z
    ], dtype=float)
    distance = float(np.linalg.norm(displacement))
    if distance <= 1e-12:
        return 0.0
    relative_velocity = target.velocity_vector() - attacker.velocity_vector()
    distance_rate = float(np.dot(displacement, relative_velocity) / distance)
    return -distance_rate


def nearest_target(own, targets):
    alive = [target for target in targets if target.alive]
    return min(
        alive,
        key=lambda target: (
            (target.x - own.x) ** 2
            + (target.y - own.y) ** 2
            + (target.z - own.z) ** 2
        ),
        default=None,
    )


def best_components(attacker, targets, distance_scale: float) -> dict[str, float]:
    candidates = [
        score_components(attacker, target, distance_scale)
        for target in targets if target.alive
    ]
    return max(candidates, key=lambda row: row["product"], default={
        "distance": 0.0, "attack_angle": 0.0, "escape_angle": 0.0,
        "range_score": 0.0, "attack_score": 0.0, "escape_score": 0.0,
        "product": 0.0,
    })


def team_snapshot(env: MultiUAVCombatEnv) -> dict:
    red = [state for state in env.red if state.alive]
    blue = [state for state in env.blue if state.alive]
    pairs = [(r, b) for r in red for b in blue]
    red_center = np.mean([[s.x, s.y, s.z] for s in red], axis=0) if red else np.zeros(3)
    blue_center = np.mean([[s.x, s.y, s.z] for s in blue], axis=0) if blue else np.zeros(3)
    nearest_distances = []
    nearest_closing = []
    for own in red:
        target = nearest_target(own, blue)
        if target is not None:
            nearest_distances.append(engagement_geometry(own, target).distance)
            nearest_closing.append(closing_speed(own, target))
    alive = red + blue
    horizontal_pairs = [
        float(np.hypot(a.x - b.x, a.y - b.y))
        for index, a in enumerate(alive) for b in alive[index + 1:]
    ]
    return {
        "centroid_separation": float(np.linalg.norm(blue_center - red_center)),
        "minimum_red_blue_distance": min(
            (engagement_geometry(r, b).distance for r, b in pairs), default=0.0
        ),
        "mean_nearest_enemy_distance": float(np.mean(nearest_distances)) if nearest_distances else 0.0,
        "maximum_nearest_enemy_distance": max(nearest_distances, default=0.0),
        "mean_nearest_closing_speed": float(np.mean(nearest_closing)) if nearest_closing else 0.0,
        "max_all_pair_horizontal_spread": max(horizontal_pairs, default=0.0),
        "red_mean_speed": float(np.mean([s.v for s in red])) if red else 0.0,
        "blue_mean_speed": float(np.mean([s.v for s in blue])) if blue else 0.0,
    }


def actor_from_checkpoint(checkpoint: Path) -> SharedSquashedGaussianActor:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hidden = int(state["actor"]["backbone.0.weight"].shape[0])
    actor = SharedSquashedGaussianActor(hidden_dim=hidden, activation="relu")
    actor.load_state_dict(state["actor"])
    return actor.eval()


def fresh_actor(seed: int = 2023) -> SharedSquashedGaussianActor:
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return SharedSquashedGaussianActor(hidden_dim=256, activation="relu").eval()


@torch.no_grad()
def actor_actions(actor: SharedSquashedGaussianActor, observation: np.ndarray, mask: np.ndarray) -> np.ndarray:
    tensor = torch.as_tensor(observation, dtype=torch.float32)
    actions = actor.deterministic(tensor).cpu().numpy()
    return actions * mask[:, None]


class Accumulator:
    def __init__(self) -> None:
        self.values: dict[str, list[float]] = defaultdict(list)
        self.geometry = {
            side: defaultdict(int) for side in ("red", "blue")
        }
        self.closing_windows: dict[str, list[float]] = {
            name: [] for name in TIME_WINDOW_NAMES
        }
        self.plateau_windows = {
            name: {key: [0, 0] for key in ("phi_001", "phi_005", "delta_0001", "delta_001")}
            for name in TIME_WINDOW_NAMES
        }
        self.speed_by_step: list[list[float]] = [[] for _ in range(1000)]
        self.blue_speed_by_step: list[list[float]] = [[] for _ in range(1000)]
        self.nearest_distance_by_step: list[list[float]] = [[] for _ in range(1000)]
        self.episode_summaries: list[dict] = []
        self.pursuit_snapshots: dict[int, list[dict]] = {
            step: [] for step in (10, 50, 100, 200)
        }

    @staticmethod
    def window_index(step: int) -> int:
        for index, (start, end) in enumerate(TIME_WINDOWS):
            if start <= step < end:
                return index
        return len(TIME_WINDOWS) - 1

    def add_geometry(self, side: str, components: dict[str, float], weapon) -> None:
        counter = self.geometry[side]
        distance_ok = components["distance"] <= weapon.attack_distance_max
        attack_ok = components["attack_angle"] <= weapon.attack_angle_max
        escape_ok = components["escape_angle"] <= weapon.escape_angle_max
        counter["total"] += 1
        counter["distance"] += int(distance_ok)
        counter["attack"] += int(attack_ok)
        counter["escape"] += int(escape_ok)
        counter["distance_attack"] += int(distance_ok and attack_ok)
        counter["distance_escape"] += int(distance_ok and escape_ok)
        counter["attack_escape"] += int(attack_ok and escape_ok)
        counter["all"] += int(distance_ok and attack_ok and escape_ok)
        if distance_ok:
            counter["in_range"] += 1
            counter["in_range_attack_failure"] += int(not attack_ok)
            counter["in_range_escape_failure"] += int(not escape_ok)
            counter["in_range_both_failure"] += int(not attack_ok and not escape_ok)

    def add_plateau(
        self, step: int, phi: np.ndarray, delta: np.ndarray, alive_mask: np.ndarray
    ) -> None:
        name = TIME_WINDOW_NAMES[self.window_index(step)]
        for value in np.abs(phi[np.asarray(alive_mask, dtype=bool)]):
            self.plateau_windows[name]["phi_001"][0] += int(value < 0.01)
            self.plateau_windows[name]["phi_005"][0] += int(value < 0.05)
            self.plateau_windows[name]["phi_001"][1] += 1
            self.plateau_windows[name]["phi_005"][1] += 1
        for value in np.abs(delta[np.asarray(alive_mask, dtype=bool)]):
            self.plateau_windows[name]["delta_0001"][0] += int(value < 1e-4)
            self.plateau_windows[name]["delta_001"][0] += int(value < 1e-3)
            self.plateau_windows[name]["delta_0001"][1] += 1
            self.plateau_windows[name]["delta_001"][1] += 1


TRAJECTORY_FIELDS = [
    "policy", "seed", "step", "time_s", "red_id", "a0", "a1", "a2",
    "speed", "theta", "psi", "altitude", "nx", "nz", "phi_control",
    "nearest_blue_distance", "nearest_blue_closing_speed", "nearest_attack_angle_deg",
    "nearest_escape_angle_deg", "red_phi", "delta_phi", "shaping_reward",
    "event_reward", "centroid_separation", "minimum_red_blue_distance",
    "mean_nearest_enemy_distance", "maximum_nearest_enemy_distance",
    "max_all_pair_horizontal_spread", "blue_mean_speed",
]


def summarize_geometry(counter: dict[str, int]) -> dict:
    total = max(counter["total"], 1)
    in_range = counter["in_range"]
    result = {
        f"{key}_fraction": counter[key] / total
        for key in (
            "distance", "attack", "escape", "distance_attack", "distance_escape",
            "attack_escape", "all",
        )
    }
    result.update({
        "pair_step_samples": counter["total"],
        "in_range_samples": in_range,
        "in_range_attack_failure_fraction": (
            counter["in_range_attack_failure"] / in_range if in_range else None
        ),
        "in_range_escape_failure_fraction": (
            counter["in_range_escape_failure"] / in_range if in_range else None
        ),
        "in_range_both_failure_fraction": (
            counter["in_range_both_failure"] / in_range if in_range else None
        ),
    })
    return result


def summarize_action(values: dict[str, list[float]], key: str) -> dict:
    array = np.asarray(values[key], dtype=float)
    result = distribution(array)
    result.update({
        "absolute_mean": float(np.mean(np.abs(array))),
        "fraction_abs_gt_0_9": float(np.mean(np.abs(array) > 0.9)),
        "fraction_abs_gt_0_99": float(np.mean(np.abs(array) > 0.99)),
    })
    return result


def summarize_accumulator(name: str, acc: Accumulator) -> dict:
    values = acc.values
    episode_mean = lambda key: float(np.mean([row[key] for row in acc.episode_summaries]))
    time_series = []
    for step in range(1000):
        speeds = np.asarray(acc.speed_by_step[step], dtype=float)
        time_series.append({
            "step": step,
            "time_s": step * 0.1,
            "red_speed_mean": float(speeds.mean()) if speeds.size else None,
            "red_speed_p10": float(np.percentile(speeds, 10)) if speeds.size else None,
            "red_speed_p50": float(np.percentile(speeds, 50)) if speeds.size else None,
            "red_speed_p90": float(np.percentile(speeds, 90)) if speeds.size else None,
            "blue_speed_mean": float(np.mean(acc.blue_speed_by_step[step])) if acc.blue_speed_by_step[step] else None,
            "red_nearest_distance_mean": float(np.mean(acc.nearest_distance_by_step[step])) if acc.nearest_distance_by_step[step] else None,
        })
    plateau = {}
    for window, counters in acc.plateau_windows.items():
        plateau[window] = {
            "fraction_abs_phi_lt_0_01": counters["phi_001"][0] / max(counters["phi_001"][1], 1),
            "fraction_abs_phi_lt_0_05": counters["phi_005"][0] / max(counters["phi_005"][1], 1),
            "fraction_abs_delta_phi_lt_1e_4": counters["delta_0001"][0] / max(counters["delta_0001"][1], 1),
            "fraction_abs_delta_phi_lt_1e_3": counters["delta_001"][0] / max(counters["delta_001"][1], 1),
        }
    return {
        "policy": name,
        "episodes": len(acc.episode_summaries),
        "actions": {f"a{index}": summarize_action(values, f"a{index}") for index in range(3)},
        "states": {
            key: distribution(values[key]) for key in ("speed", "theta", "psi", "altitude")
        },
        "controls": {
            key: distribution(values[key]) for key in ("nx", "nz", "phi_control")
        },
        "speed_thresholds": {
            "fraction_red_speed_gt_280": fraction(values["speed"], lambda x: x > 280.0),
            "fraction_red_speed_gt_295": fraction(values["speed"], lambda x: x > 295.0),
            "red_mean_speed": float(np.mean(values["speed"])),
            "blue_mean_speed": float(np.mean(values["blue_speed"])),
        },
        "closing_speed": {
            "initial": distribution(values["initial_closing"]),
            "windows": {
                window: distribution(acc.closing_windows[window]) for window in TIME_WINDOW_NAMES
            },
            "fraction_positive": fraction(values["closing"], lambda x: x > 0.0),
            "fraction_negative": fraction(values["closing"], lambda x: x < 0.0),
        },
        "nearest_enemy_distance": {
            "distribution": distribution(values["nearest_distance"]),
            "step_fraction_at_or_below": {
                str(int(threshold)): fraction(
                    values["nearest_distance"], lambda x, t=threshold: x <= t
                ) for threshold in DISTANCE_THRESHOLDS
            },
        },
        "team_geometry": {
            "centroid_separation": distribution(values["centroid_separation"]),
            "minimum_red_blue_distance": distribution(values["minimum_distance"]),
            "mean_nearest_enemy_distance": distribution(values["mean_nearest_distance"]),
            "maximum_nearest_enemy_distance": distribution(values["max_nearest_distance"]),
            "max_all_pair_horizontal_spread": distribution(values["horizontal_spread"]),
        },
        "red_attack_geometry": summarize_geometry(acc.geometry["red"]),
        "blue_attack_geometry": summarize_geometry(acc.geometry["blue"]),
        "tactical_score": {
            key: distribution(values[key]) for key in (
                "best_attack_range", "best_attack_attack", "best_attack_escape",
                "best_attack_product", "best_threat_range", "best_threat_attack",
                "best_threat_escape", "best_threat_product", "phi", "delta_phi",
            )
        },
        "reward": {
            "undiscounted_shaping_team_mean": episode_mean("undiscounted_shaping"),
            "discounted_shaping_team_mean": episode_mean("discounted_shaping"),
            "event_reward_team_mean": episode_mean("event_reward"),
            "telescope_max_abs_error": max(
                row["telescope_max_abs_error"] for row in acc.episode_summaries
            ),
        },
        "plateau_by_time": plateau,
        "outcomes": {
            "red_win_rate": episode_mean("red_win"),
            "blue_win_rate": episode_mean("blue_win"),
            "draw_rate": episode_mean("draw"),
            "timeout_rate": episode_mean("timeout"),
            "average_red_loss": episode_mean("red_losses"),
            "average_blue_loss": episode_mean("blue_losses"),
            "average_red_low_altitude_loss": episode_mean("red_low_altitude_losses"),
            "average_episode_length": episode_mean("episode_length"),
        },
        "comparison": {
            "initial_10s_mean_red_speed": float(np.mean(values["speed_0_10"])),
            "initial_10s_mean_closing_speed": float(np.mean(values["closing_0_10"])),
            "minimum_distance": min(values["minimum_distance"]),
            "time_to_4km_mean_s_reached_only": distribution(values["time_to_4km"])["mean"],
            "time_to_4km_reach_rate": len(values["time_to_4km"]) / len(acc.episode_summaries),
            "time_to_1_5km_mean_s_reached_only": distribution(values["time_to_1_5km"])["mean"],
            "time_to_1_5km_reach_rate": len(values["time_to_1_5km"]) / len(acc.episode_summaries),
            "red_attack_angle_valid_fraction": summarize_geometry(acc.geometry["red"])["attack_fraction"],
            "red_escape_angle_valid_fraction": summarize_geometry(acc.geometry["red"])["escape_fraction"],
            "red_attackable_fraction": summarize_geometry(acc.geometry["red"])["all_fraction"],
            "mean_abs_phi": float(np.mean(np.abs(values["phi"]))),
            "mean_abs_delta_phi": float(np.mean(np.abs(values["delta_phi"]))),
            "max_pair_spread": max(values["horizontal_spread"]),
            "final_centroid_separation": episode_mean("final_centroid_separation"),
        },
        "pure_pursuit_snapshots": {
            str(step): {
                key: float(np.mean([row[key] for row in rows]))
                for key in rows[0]
            } if rows else {}
            for step, rows in acc.pursuit_snapshots.items()
        },
        "speed_time_series": time_series,
    }


def run_policy(
    name: str,
    config: dict,
    action_function: Callable[[MultiUAVCombatEnv, np.ndarray], np.ndarray],
    output_dir: Path,
) -> dict:
    acc = Accumulator()
    distance_scale = float(config["reward"]["engagement_distance_scale"])
    gamma = 0.99
    for seed in EVALUATION_SEEDS:
        env = MultiUAVCombatEnv(config)
        observation, _ = env.reset(seed)
        initial_phi = tactical_potentials(env.red, env.blue, distance_scale).astype(float)
        phi_sum = np.zeros(4, dtype=float)
        discounted_phi_sum = np.zeros(4, dtype=float)
        event_sum = np.zeros(4, dtype=float)
        first_4km = None
        first_15km = None
        detail_stream = None
        writer = None
        if seed in TRAJECTORY_SEEDS:
            detail_stream = (output_dir / f"seed_trajectory_{name}_{seed}.csv").open(
                "w", newline="", encoding="utf-8"
            )
            writer = csv.DictWriter(detail_stream, fieldnames=TRAJECTORY_FIELDS)
            writer.writeheader()
        final_info = None
        try:
            while True:
                step = env.steps
                snapshot = team_snapshot(env)
                phi_before = tactical_potentials(env.red, env.blue, distance_scale).astype(float)
                alive_before = env.red_alive_mask.astype(bool)
                actions = np.asarray(action_function(env, observation), dtype=np.float32)
                pre_states = [state.copy() for state in env.red]
                per_agent = []
                for red_id, own in enumerate(env.red):
                    if not own.alive:
                        per_agent.append(None)
                        continue
                    target = nearest_target(own, env.blue)
                    nearest = score_components(own, target, distance_scale)
                    close = closing_speed(own, target)
                    control = action_to_control(own, actions[red_id], config["action"])
                    attack = best_components(own, env.blue, distance_scale)
                    threats = [
                        score_components(blue, own, distance_scale)
                        for blue in env.blue if blue.alive
                    ]
                    threat = max(threats, key=lambda row: row["product"])
                    per_agent.append((nearest, close, control, attack, threat))
                    for index in range(3):
                        acc.values[f"a{index}"].append(float(actions[red_id, index]))
                    for key, value in (
                        ("speed", own.v), ("theta", own.theta), ("psi", own.psi),
                        ("altitude", own.altitude), ("nx", control.nx),
                        ("nz", control.nz), ("phi_control", control.phi),
                        ("nearest_distance", nearest["distance"]), ("closing", close),
                        ("best_attack_range", attack["range_score"]),
                        ("best_attack_attack", attack["attack_score"]),
                        ("best_attack_escape", attack["escape_score"]),
                        ("best_attack_product", attack["product"]),
                        ("best_threat_range", threat["range_score"]),
                        ("best_threat_attack", threat["attack_score"]),
                        ("best_threat_escape", threat["escape_score"]),
                        ("best_threat_product", threat["product"]),
                        ("phi", phi_before[red_id]),
                    ):
                        acc.values[key].append(float(value))
                    acc.speed_by_step[step].append(own.v)
                    acc.nearest_distance_by_step[step].append(nearest["distance"])
                    window = TIME_WINDOW_NAMES[acc.window_index(step)]
                    acc.closing_windows[window].append(close)
                    if step == 0:
                        acc.values["initial_closing"].append(close)
                    if step < 100:
                        acc.values["speed_0_10"].append(own.v)
                        acc.values["closing_0_10"].append(close)
                for state in env.blue:
                    if state.alive:
                        acc.values["blue_speed"].append(state.v)
                        acc.blue_speed_by_step[step].append(state.v)
                for key, value in (
                    ("centroid_separation", snapshot["centroid_separation"]),
                    ("minimum_distance", snapshot["minimum_red_blue_distance"]),
                    ("mean_nearest_distance", snapshot["mean_nearest_enemy_distance"]),
                    ("max_nearest_distance", snapshot["maximum_nearest_enemy_distance"]),
                    ("horizontal_spread", snapshot["max_all_pair_horizontal_spread"]),
                ):
                    acc.values[key].append(value)
                if first_4km is None and snapshot["minimum_red_blue_distance"] <= 4000.0:
                    first_4km = step * env.dt
                if first_15km is None and snapshot["minimum_red_blue_distance"] <= 1500.0:
                    first_15km = step * env.dt
                for red in env.red:
                    if red.alive:
                        for blue in env.blue:
                            if blue.alive:
                                acc.add_geometry(
                                    "red", score_components(red, blue, distance_scale), env.weapon
                                )
                                acc.add_geometry(
                                    "blue", score_components(blue, red, distance_scale), env.weapon
                                )
                observation, _, terminated, truncated, info = env.step(actions)
                delta_phi = np.asarray(info["tactical_potential"], dtype=float) - phi_before
                phi_sum += delta_phi
                discounted_phi_sum += (gamma ** step) * delta_phi
                event_sum += np.asarray(info["event_rewards"], dtype=float)
                acc.values["delta_phi"].extend(delta_phi[alive_before].tolist())
                acc.add_plateau(step, phi_before, delta_phi, alive_before)
                if writer is not None:
                    for red_id, row in enumerate(per_agent):
                        if row is None:
                            continue
                        nearest, close, control, _, _ = row
                        own = pre_states[red_id]
                        writer.writerow({
                            "policy": name, "seed": seed, "step": step,
                            "time_s": step * env.dt, "red_id": red_id,
                            "a0": actions[red_id, 0], "a1": actions[red_id, 1],
                            "a2": actions[red_id, 2], "speed": own.v,
                            "theta": own.theta, "psi": own.psi,
                            "altitude": own.altitude, "nx": control.nx,
                            "nz": control.nz, "phi_control": control.phi,
                            "nearest_blue_distance": nearest["distance"],
                            "nearest_blue_closing_speed": close,
                            "nearest_attack_angle_deg": np.degrees(nearest["attack_angle"]),
                            "nearest_escape_angle_deg": np.degrees(nearest["escape_angle"]),
                            "red_phi": phi_before[red_id], "delta_phi": delta_phi[red_id],
                            "shaping_reward": info["shaping_rewards"][red_id],
                            "event_reward": info["event_rewards"][red_id],
                            "centroid_separation": snapshot["centroid_separation"],
                            "minimum_red_blue_distance": snapshot["minimum_red_blue_distance"],
                            "mean_nearest_enemy_distance": snapshot["mean_nearest_enemy_distance"],
                            "maximum_nearest_enemy_distance": snapshot["maximum_nearest_enemy_distance"],
                            "max_all_pair_horizontal_spread": snapshot["max_all_pair_horizontal_spread"],
                            "blue_mean_speed": snapshot["blue_mean_speed"],
                        })
                if name == "scripted_pursuit" and env.steps in acc.pursuit_snapshots:
                    after = team_snapshot(env)
                    after_phi = tactical_potentials(env.red, env.blue, distance_scale)
                    acc.pursuit_snapshots[env.steps].append({
                        "mean_phi": float(np.mean(after_phi)),
                        "mean_delta_phi_from_reset": float(np.mean(after_phi - initial_phi)),
                        "minimum_distance": after["minimum_red_blue_distance"],
                        "mean_closing_speed": after["mean_nearest_closing_speed"],
                        "centroid_separation": after["centroid_separation"],
                    })
                final_info = info
                if terminated or truncated:
                    break
        finally:
            if detail_stream is not None:
                detail_stream.close()
        final_phi = tactical_potentials(env.red, env.blue, distance_scale).astype(float)
        final_snapshot = team_snapshot(env)
        if first_4km is not None:
            acc.values["time_to_4km"].append(first_4km)
        if first_15km is not None:
            acc.values["time_to_1_5km"].append(first_15km)
        acc.episode_summaries.append({
            "undiscounted_shaping": float(phi_sum.sum()),
            "discounted_shaping": float(discounted_phi_sum.sum()),
            "event_reward": float(event_sum.sum()),
            "telescope_max_abs_error": float(np.max(np.abs(phi_sum - (final_phi - initial_phi)))),
            "red_win": float(final_info["red_win"]),
            "blue_win": float(final_info["blue_win"]),
            "draw": float(final_info["draw"]),
            "timeout": float(final_info["termination_reason"] == "draw_timeout"),
            "red_losses": final_info["red_losses"],
            "blue_losses": final_info["blue_losses"],
            "red_low_altitude_losses": final_info["red_low_altitude_losses"],
            "episode_length": final_info["episode_length"],
            "final_centroid_separation": final_snapshot["centroid_separation"],
        })
    return summarize_accumulator(name, acc)


def canonical_reset_summary(config: dict) -> dict:
    scale = float(config["reward"]["engagement_distance_scale"])
    values: dict[str, list[float]] = defaultdict(list)
    for seed in EVALUATION_SEEDS:
        env = MultiUAVCombatEnv(config)
        env.reset(seed)
        red_phi = tactical_potentials(env.red, env.blue, scale)
        blue_phi = tactical_potentials(env.blue, env.red, scale)
        values["red_phi"].extend(red_phi.tolist())
        values["blue_phi"].extend(blue_phi.tolist())
        for side, attackers, targets in (
            ("red", env.red, env.blue), ("blue", env.blue, env.red)
        ):
            for attacker in attackers:
                best = best_components(attacker, targets, scale)
                for key in (
                    "distance", "attack_angle", "escape_angle", "range_score",
                    "attack_score", "escape_score", "product",
                ):
                    value = best[key]
                    if "angle" in key:
                        value = np.degrees(value)
                    values[f"{side}_{key}"].append(float(value))
    result = {key: distribution(value) for key, value in values.items()}
    result["interpretation"] = (
        "Canonical head-on geometry has near-180-degree escape angle; the escape "
        "component therefore suppresses the product despite positive range and attack scores."
    )
    return result


def counterfactual_families(final_actor) -> dict[str, Callable]:
    constants = {
        "zero": [0, 0, 0], "accelerate_straight": [1, 0, 0],
        "decelerate_straight": [-1, 0, 0], "left_bank": [0, 0, -0.5],
        "right_bank": [0, 0, 0.5], "hard_left": [0, 0, -1],
        "hard_right": [0, 0, 1], "climb": [0, 0.5, 0],
        "descend": [0, -0.5, 0],
    }
    families = {
        name: (lambda env, obs, action=np.asarray(action, dtype=np.float32):
               np.tile(action, (4, 1)) * env.red_alive_mask[:, None])
        for name, action in constants.items()
    }
    families["scripted_pursuit"] = lambda env, obs: env.fixed_policy.team_actions(env.red, env.blue)
    families["final_actor"] = lambda env, obs: actor_actions(final_actor, obs, env.red_alive_mask)
    for dimension in range(3):
        for delta in (-0.25, -0.1, 0.1, 0.25):
            name = f"final_actor_a{dimension}_{delta:+.2f}"
            def perturbed(env, obs, dim=dimension, amount=delta):
                actions = actor_actions(final_actor, obs, env.red_alive_mask)
                actions[:, dim] = np.clip(actions[:, dim] + amount, -1.0, 1.0)
                return actions
            families[name] = perturbed
    return families


def nearest_red_geometry(env: MultiUAVCombatEnv) -> tuple[float, float]:
    attack, escape = [], []
    for own in env.red:
        if own.alive:
            target = nearest_target(own, env.blue)
            if target is not None:
                geometry = engagement_geometry(own, target)
                attack.append(geometry.attack_angle)
                escape.append(geometry.escape_angle)
    return float(np.mean(attack)), float(np.mean(escape))


def run_counterfactuals(config: dict, final_actor, output: Path) -> list[dict]:
    scale = float(config["reward"]["engagement_distance_scale"])
    gamma = 0.99
    rows = []
    for seed in EVALUATION_SEEDS:
        for family_name, action_function in counterfactual_families(final_actor).items():
            env = MultiUAVCombatEnv(config)
            observation, _ = env.reset(seed)
            phi_start = tactical_potentials(env.red, env.blue, scale).astype(float)
            cumulative = np.zeros(4, dtype=float)
            discounted = np.zeros(4, dtype=float)
            initial = team_snapshot(env)
            minimum_distance = initial["minimum_red_blue_distance"]
            reached = {8000: False, 4000: False, 1500: False}
            for step in range(100):
                phi_before = tactical_potentials(env.red, env.blue, scale).astype(float)
                actions = action_function(env, observation)
                observation, _, terminated, truncated, info = env.step(actions)
                delta = np.asarray(info["tactical_potential"], dtype=float) - phi_before
                cumulative += delta
                discounted += gamma ** step * delta
                snapshot = team_snapshot(env)
                minimum_distance = min(minimum_distance, snapshot["minimum_red_blue_distance"])
                for threshold in reached:
                    reached[threshold] |= snapshot["minimum_red_blue_distance"] <= threshold
                if env.steps in COUNTERFACTUAL_HORIZONS:
                    phi_end = tactical_potentials(env.red, env.blue, scale).astype(float)
                    attack, escape = nearest_red_geometry(env)
                    rows.append({
                        "seed": seed, "maneuver": family_name, "horizon_steps": env.steps,
                        "horizon_seconds": env.steps * env.dt,
                        "mean_phi_start": float(np.mean(phi_start)),
                        "mean_phi_end": float(np.mean(phi_end)),
                        "mean_phi_change": float(np.mean(phi_end - phi_start)),
                        "sum_mean_delta_phi": float(np.mean(cumulative)),
                        "discounted_mean_delta_phi": float(np.mean(discounted)),
                        "team_phi_change": float(np.sum(phi_end - phi_start)),
                        "minimum_enemy_distance": minimum_distance,
                        "initial_mean_closing_speed": initial["mean_nearest_closing_speed"],
                        "final_mean_closing_speed": snapshot["mean_nearest_closing_speed"],
                        "final_mean_attack_angle_deg": float(np.degrees(attack)),
                        "final_mean_escape_angle_deg": float(np.degrees(escape)),
                        "entered_8km": reached[8000], "entered_4km": reached[4000],
                        "entered_1_5km": reached[1500],
                    })
                if terminated or truncated:
                    break
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def landscape_summary(rows: list[dict]) -> dict:
    result = {}
    for maneuver in sorted({row["maneuver"] for row in rows}):
        result[maneuver] = {}
        for horizon in COUNTERFACTUAL_HORIZONS:
            subset = [
                row for row in rows
                if row["maneuver"] == maneuver and row["horizon_steps"] == horizon
            ]
            result[maneuver][str(horizon)] = {
                key: float(np.mean([row[key] for row in subset]))
                for key in (
                    "mean_phi_change", "sum_mean_delta_phi", "discounted_mean_delta_phi",
                    "minimum_enemy_distance", "initial_mean_closing_speed",
                    "final_mean_closing_speed", "final_mean_attack_angle_deg",
                    "final_mean_escape_angle_deg", "entered_8km", "entered_4km",
                    "entered_1_5km",
                )
            }
    return result


def entropy_summary(metrics_path: Path, alpha: float) -> dict:
    values = []
    if metrics_path.exists():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            value = json.loads(line).get("entropy")
            if value is not None:
                values.append(float(value))
    result = distribution(values)
    result["alpha"] = alpha
    result["alpha_times_mean_entropy"] = alpha * result["mean"] if values else None
    result["source"] = "training_metrics.jsonl actor entropy diagnostic"
    return result


def observation_learnability() -> dict:
    return {
        "information_missing": False,
        "information_present_but_not_explicitly_encoded": True,
        "relative_slot": "[r_forward,r_right,r_up,vrel_forward,vrel_right,vrel_up,alive]",
        "distance": "||r||; frame rotation preserves Euclidean norm",
        "closing_rate": "dot(r,v_rel)/||r||; closing_speed is its negative",
        "attack_angle": "acos(r_forward/||r||), because own forward axis is own velocity direction",
        "target_velocity": "v_target = v_own + v_rel; own speed and theta plus the flight-path frame recover own velocity in local coordinates",
        "escape_angle": "acos(dot(v_target,r)/(||v_target||*||r||))",
        "qualification": (
            "All combat quantities are derivable for each visible alive slot, but the network "
            "must learn norms, dot products, target selection, inverse cosine-like boundaries, "
            "and permutation-sensitive slot comparisons from samples."
        ),
    }


def write_policy_comparison(summaries: dict[str, dict], output: Path) -> None:
    rows = [summary["comparison"] | {"policy": name} for name, summary in summaries.items()]
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["policy", *summaries[next(iter(summaries))]["comparison"]])
        writer.writeheader()
        writer.writerows(rows)


def finite_tree(value) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return not isinstance(value, float) or np.isfinite(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="outputs/madsac_v1_4_pilot_500k_parallel/run_seed_2023/latest.pt",
    )
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    checkpoint = (root / args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else checkpoint.parent / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(
        (root / "configs/combat_environment.yaml").read_text(encoding="utf-8")
    )
    algorithm = yaml.safe_load(
        (root / "configs/madsac.yaml").read_text(encoding="utf-8")
    )
    final_actor = actor_from_checkpoint(checkpoint)
    initial_actor = fresh_actor(2023)
    policies = {
        "final_500k": lambda env, obs: actor_actions(final_actor, obs, env.red_alive_mask),
        "fresh_untrained": lambda env, obs: actor_actions(initial_actor, obs, env.red_alive_mask),
        "scripted_pursuit": lambda env, obs: env.fixed_policy.team_actions(env.red, env.blue),
    }
    summaries = {}
    for name, function in policies.items():
        print(f"[TRAJECTORY] policy={name}", flush=True)
        summaries[name] = run_policy(name, config, function, output_dir)
    print("[LANDSCAPE] counterfactual rollouts", flush=True)
    landscape_rows = run_counterfactuals(
        config, final_actor, output_dir / "reward_landscape.csv"
    )
    write_policy_comparison(summaries, output_dir / "policy_comparison.csv")
    alpha = float(algorithm["training"]["alpha"])
    trajectory_summary = {
        "checkpoint": str(checkpoint),
        "evaluation_seeds": EVALUATION_SEEDS,
        "full_trajectory_seeds": sorted(TRAJECTORY_SEEDS),
        "fresh_actor_seed": 2023,
        "policies": summaries,
        "canonical_reset": canonical_reset_summary(config),
        "training_entropy": entropy_summary(
            checkpoint.parent / "training_metrics.jsonl", alpha
        ),
        "observation_learnability": observation_learnability(),
    }
    diagnosis_report = {
        "policy_comparison": {
            name: summary["comparison"] for name, summary in summaries.items()
        },
        "reward_landscape_summary": landscape_summary(landscape_rows),
        "pure_pursuit_reward_baseline": summaries["scripted_pursuit"]["pure_pursuit_snapshots"],
        "telescope_max_abs_error": max(
            summary["reward"]["telescope_max_abs_error"] for summary in summaries.values()
        ),
        "all_outputs_finite": finite_tree(trajectory_summary),
        "observation_learnability": observation_learnability(),
        "active_code_modified": False,
        "training_executed": False,
    }
    (output_dir / "trajectory_summary.json").write_text(
        json.dumps(trajectory_summary, indent=2), encoding="utf-8"
    )
    (output_dir / "diagnosis_report.json").write_text(
        json.dumps(diagnosis_report, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "output_dir": str(output_dir),
        "telescope_max_abs_error": diagnosis_report["telescope_max_abs_error"],
        "all_outputs_finite": diagnosis_report["all_outputs_finite"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
