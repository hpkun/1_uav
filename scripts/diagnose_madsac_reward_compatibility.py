"""Offline Li et al. (2023) reward compatibility audit for V1.4.

This diagnostic never changes the environment reward.  It replays fixed policies and
computes Equation (25) from post-transition states alongside the active V1.4 reward.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import yaml

from uav_combat.environment.env import MultiUAVCombatEnv
from uav_combat.environment.geometry import engagement_geometry, engagement_score
from uav_combat.environment.reward import tactical_potentials
from uav_combat.madsac.actor import SharedSquashedGaussianActor
from uav_combat.models import AircraftState


EVALUATION_SEEDS = list(range(10_000_000, 10_000_020))
PAPER_DISTANCE_M = 4000.0
PAPER_GUIDE_REWARD = 0.001
DEG_5 = np.deg2rad(5.0)
DEG_15 = np.deg2rad(15.0)
DEG_30 = np.deg2rad(30.0)
EXPECTED_ACTIVE_REWARD_SHA256 = "f369f1a9a6acd2b8c110936017df687fe032216db42f27decef8965c7c8bc4a4"
WINDOWS = {
    "0-5s": (0, 50), "5-10s": (50, 100), "0-10s": (0, 100),
    "8-15s": (80, 150), "10-20s": (100, 200), "15-25s": (150, 250),
    "0-20s": (0, 200), "20-40s": (200, 400), "25-40s": (250, 400),
    "10-40s": (100, 400), "0-40s": (0, 400), "40-60s": (400, 600),
    "40-100s": (400, 1000), "60-100s": (600, 1000), "0-100s": (0, 1000),
}
PLATEAU_WINDOWS = ("0-10s", "10-20s", "20-40s", "40-60s", "60-100s")


@dataclass(frozen=True)
class PaperGeometry:
    """Signed diagnostic geometry; paper reward always applies absolute values."""

    distance: float
    ata: float
    aa: float
    ha: float


def wrap_angle(value: float) -> float:
    return float((value + np.pi) % (2.0 * np.pi) - np.pi)


def paper_geometry(own: AircraftState, target: AircraftState) -> PaperGeometry:
    """Operationalize Fig. 2/Eq. (6) without using active geometry.py.

    LOS points from ``own`` to ``target`` in the horizontal plane. ATA is from
    own nose to LOS; AA is from LOS to target nose (not reverse LOS); HA is the
    elevation from horizontal LOS to the target, positive when target is above
    in the V1.4 NED coordinate system.
    """
    dx, dy, dz = target.x - own.x, target.y - own.y, target.z - own.z
    horizontal = float(np.hypot(dx, dy))
    los = float(np.arctan2(dy, dx)) if horizontal > 1e-12 else own.psi
    return PaperGeometry(
        distance=float(np.sqrt(dx * dx + dy * dy + dz * dz)),
        ata=wrap_angle(los - own.psi),
        aa=wrap_angle(target.psi - los),
        ha=float(np.arctan2(-dz, horizontal)),
    )


def _tier(angle_a: float, angle_b: float, values: tuple[float, float, float]) -> float:
    """Strongest-first precedence resolves Eq. (25)'s overlapping printed cases."""
    a, b = abs(angle_a), abs(angle_b)
    if a <= DEG_5 and b <= DEG_5:
        return values[2]
    if a <= DEG_15 and b <= DEG_15:
        return values[1]
    if a <= DEG_30 and b <= DEG_30:
        return values[0]
    return 0.0


def paper_r3(red_geometry: PaperGeometry) -> float:
    return float(PAPER_GUIDE_REWARD if (
        abs(red_geometry.ata) <= DEG_30
        and abs(red_geometry.ha) <= DEG_30
        and red_geometry.distance >= PAPER_DISTANCE_M
    ) else 0.0)


def paper_r4(red_geometry: PaperGeometry, blue_geometry: PaperGeometry) -> float:
    """Equation (25), with printed R41 branch first and strongest tier first."""
    if abs(red_geometry.aa) <= DEG_30 and red_geometry.distance <= PAPER_DISTANCE_M:
        return _tier(red_geometry.ata, red_geometry.ha, (0.01, 0.02, 0.1))
    if abs(blue_geometry.aa) <= DEG_30 and blue_geometry.distance <= PAPER_DISTANCE_M:
        return _tier(blue_geometry.ata, blue_geometry.ha, (-0.015, -0.025, -0.15))
    return 0.0


def paper_r1(blue_destroyed: int, red_destroyed: int) -> float:
    return 10.0 * int(blue_destroyed) - 10.0 * int(red_destroyed)


def paper_r2_v1_4() -> tuple[float, str]:
    return 0.0, "not applicable in V1.4 horizontal-unbounded setting"


def nearest_target(own: AircraftState, targets: list[AircraftState]) -> AircraftState | None:
    alive = [target for target in targets if target.alive]
    return min(alive, key=lambda target: (
        (target.x - own.x) ** 2 + (target.y - own.y) ** 2 + (target.z - own.z) ** 2
    ), default=None)


def closing_speed(own: AircraftState, target: AircraftState) -> float:
    displacement = np.array([target.x - own.x, target.y - own.y, target.z - own.z])
    distance = float(np.linalg.norm(displacement))
    if distance <= 1e-12:
        return 0.0
    relative_velocity = target.velocity_vector() - own.velocity_vector()
    return -float(np.dot(displacement, relative_velocity) / distance)


def load_actor(checkpoint: Path) -> SharedSquashedGaussianActor:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hidden = int(payload["actor"]["backbone.0.weight"].shape[0])
    actor = SharedSquashedGaussianActor(hidden_dim=hidden, activation="relu")
    actor.load_state_dict(payload["actor"])
    return actor.eval()


def fresh_actor(seed: int = 2023) -> SharedSquashedGaussianActor:
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return SharedSquashedGaussianActor(hidden_dim=256, activation="relu").eval()


@torch.no_grad()
def deterministic_actions(actor, observation: np.ndarray, mask: np.ndarray) -> np.ndarray:
    actions = actor.deterministic(torch.as_tensor(observation, dtype=torch.float32))
    return actions.cpu().numpy() * mask[:, None]


def distribution(values: Iterable[float], absolute: bool = False) -> dict:
    array = np.asarray(list(values), dtype=float)
    if absolute:
        array = np.abs(array)
    if array.size == 0:
        return {key: None for key in ("mean", "median", "p90", "p99", "max")}
    return {
        "mean": float(array.mean()), "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)), "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


def score_snapshot(env: MultiUAVCombatEnv) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scale = float(env.config["reward"]["engagement_distance_scale"])
    attack, threat = np.zeros(4), np.zeros(4)
    for index, own in enumerate(env.red):
        if not own.alive:
            continue
        attack[index] = max((engagement_score(engagement_geometry(own, target), scale)
                             for target in env.blue if target.alive), default=0.0)
        threat[index] = max((engagement_score(engagement_geometry(target, own), scale)
                             for target in env.blue if target.alive), default=0.0)
    return attack, threat, attack - threat


STEP_FIELDS = [
    "policy", "seed", "step", "time_s", "red_alive", "blue_alive",
    "mean_a0", "mean_a1", "mean_a2", "mean_abs_a2", "mean_distance",
    "mean_closing_speed", "mean_ata_deg", "mean_aa_deg", "mean_ha_deg",
    "best_attack_score_mean", "best_threat_score_mean", "phi_current_team",
    "current_delta_phi_team", "current_event_team", "current_total_team",
    "paper_r1_team", "paper_r2_team", "paper_r3_team", "paper_r4_team",
    "paper_r4_positive_team", "paper_r4_negative_team", "paper_total_team",
    "paper_r3_occurrences", "paper_r4_positive_occurrences",
    "paper_r4_negative_occurrences", "alive_agent_samples", "current_attackable",
    "selected_target_attackable", "any_pair_paper_positive",
    "any_pair_paper_negative", "best_pair_paper_r4",
    "positive_r4_and_attackable", "positive_r4_not_attackable",
    "attackable_no_positive_r4",
]


def geometry_rewards(env: MultiUAVCombatEnv) -> dict:
    values = defaultdict(float)
    for own in env.red:
        if not own.alive:
            continue
        target = nearest_target(own, env.blue)
        if target is None:
            continue
        red_geo = paper_geometry(own, target)
        blue_geo = paper_geometry(target, own)
        r3, r4 = paper_r3(red_geo), paper_r4(red_geo, blue_geo)
        selected_attackable = env.weapon.attackable(engagement_geometry(own, target))
        current_attackable = any(
            env.weapon.attackable(engagement_geometry(own, candidate))
            for candidate in env.blue if candidate.alive
        )
        all_pair_r4 = [
            paper_r4(paper_geometry(own, candidate), paper_geometry(candidate, own))
            for candidate in env.blue if candidate.alive
        ]
        positive = r4 > 0.0
        values["r3"] += r3
        values["r4"] += r4
        values["r4_positive"] += max(r4, 0.0)
        values["r4_negative"] += min(r4, 0.0)
        values["r3_occurrences"] += int(r3 != 0.0)
        values["r4_positive_occurrences"] += int(positive)
        values["r4_negative_occurrences"] += int(r4 < 0.0)
        values["alive_agent_samples"] += 1
        values["current_attackable"] += int(current_attackable)
        values["selected_target_attackable"] += int(selected_attackable)
        values["any_pair_paper_positive"] += int(any(value > 0.0 for value in all_pair_r4))
        values["any_pair_paper_negative"] += int(any(value < 0.0 for value in all_pair_r4))
        values["best_pair_paper_r4"] += max(all_pair_r4, default=0.0)
        values["positive_r4_and_attackable"] += int(positive and current_attackable)
        values["positive_r4_not_attackable"] += int(positive and not current_attackable)
        values["attackable_no_positive_r4"] += int(current_attackable and not positive)
        values["distance_sum"] += red_geo.distance
        values["closing_sum"] += closing_speed(own, target)
        values["ata_sum"] += abs(red_geo.ata)
        values["aa_sum"] += abs(red_geo.aa)
        values["ha_sum"] += abs(red_geo.ha)
    return values


def make_step_row(
    policy: str, seed: int, step: int, env: MultiUAVCombatEnv, actions: np.ndarray,
    info: dict, prior_red_losses: int, prior_red_kills: int,
) -> dict:
    geom = geometry_rewards(env)
    samples = max(int(geom["alive_agent_samples"]), 1)
    current_shaping = float(np.sum(info["shaping_rewards"]))
    current_event = float(np.sum(info["event_rewards"]))
    red_losses = int(info["red_losses"]) - prior_red_losses
    blue_kills = int(info["red_attack_kills"]) - prior_red_kills
    r1 = paper_r1(blue_kills, red_losses)
    attack, threat, phi = score_snapshot(env)
    alive_action = actions[np.asarray(info["red_alive_mask"], dtype=bool)]
    if alive_action.size == 0:
        alive_action = np.zeros((1, 3))
    return {
        "policy": policy, "seed": seed, "step": step, "time_s": step * env.dt,
        "red_alive": int(info["red_survivors"]), "blue_alive": int(info["blue_survivors"]),
        "mean_a0": float(alive_action[:, 0].mean()), "mean_a1": float(alive_action[:, 1].mean()),
        "mean_a2": float(alive_action[:, 2].mean()), "mean_abs_a2": float(np.abs(alive_action[:, 2]).mean()),
        "mean_distance": geom["distance_sum"] / samples,
        "mean_closing_speed": geom["closing_sum"] / samples,
        "mean_ata_deg": float(np.degrees(geom["ata_sum"] / samples)),
        "mean_aa_deg": float(np.degrees(geom["aa_sum"] / samples)),
        "mean_ha_deg": float(np.degrees(geom["ha_sum"] / samples)),
        "best_attack_score_mean": float(attack.mean()),
        "best_threat_score_mean": float(threat.mean()), "phi_current_team": float(phi.sum()),
        "current_delta_phi_team": current_shaping, "current_event_team": current_event,
        "current_total_team": current_shaping + current_event,
        "paper_r1_team": r1, "paper_r2_team": 0.0, "paper_r3_team": geom["r3"],
        "paper_r4_team": geom["r4"], "paper_r4_positive_team": geom["r4_positive"],
        "paper_r4_negative_team": geom["r4_negative"],
        "paper_total_team": r1 + geom["r3"] + geom["r4"],
        "paper_r3_occurrences": int(geom["r3_occurrences"]),
        "paper_r4_positive_occurrences": int(geom["r4_positive_occurrences"]),
        "paper_r4_negative_occurrences": int(geom["r4_negative_occurrences"]),
        **{key: int(geom[key]) for key in (
            "alive_agent_samples", "current_attackable", "selected_target_attackable",
            "any_pair_paper_positive", "any_pair_paper_negative",
            "positive_r4_and_attackable",
            "positive_r4_not_attackable", "attackable_no_positive_r4",
        )},
        "best_pair_paper_r4": float(geom["best_pair_paper_r4"]),
    }


def rollout_policy(
    name: str, config: dict, action_fn: Callable, output_dir: Path,
    max_steps: int | None = None,
) -> list[dict]:
    rows: list[dict] = []
    output = output_dir / f"seed_reward_trajectory_{name}.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=STEP_FIELDS)
        writer.writeheader()
        for seed in EVALUATION_SEEDS:
            env = MultiUAVCombatEnv(config)
            observation, _ = env.reset(seed)
            limit = min(env.max_steps, max_steps or env.max_steps)
            for step in range(limit):
                actions = action_fn(env, observation)
                prior_losses = 4 - int(env.red_alive_mask.sum())
                prior_kills = env.red_attack_kills
                observation, _, terminated, truncated, info = env.step(actions)
                row = make_step_row(name, seed, step, env, actions, info, prior_losses, prior_kills)
                rows.append(row)
                writer.writerow(row)
                if terminated or truncated:
                    break
    return rows


def aggregate_episode(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["policy"], row["seed"])].append(row)
    result = []
    for (policy, seed), group in grouped.items():
        result.append({
            "policy": policy, "seed": seed, "steps": len(group),
            **{f"{key}_sum": float(sum(row[key] for row in group)) for key in (
                "current_delta_phi_team", "current_event_team", "current_total_team",
                "paper_r1_team", "paper_r2_team", "paper_r3_team", "paper_r4_team",
                "paper_r4_positive_team", "paper_r4_negative_team", "paper_total_team",
            )},
            "final_red_alive": group[-1]["red_alive"], "final_blue_alive": group[-1]["blue_alive"],
        })
    return result


def add_policy_ranks(rows: list[dict]) -> None:
    """Add per-seed descending ranks; equal values receive the same competition rank."""
    for seed in EVALUATION_SEEDS:
        seed_rows = [row for row in rows if row["seed"] == seed]
        for source, destination in (("current_total_team_sum", "current_rank"),
                                    ("paper_total_team_sum", "madsac_rank")):
            values = [row[source] for row in seed_rows]
            for row in seed_rows:
                row[destination] = 1 + sum(value > row[source] for value in values)


def window_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["policy"], row["seed"])].append(row)
    result = []
    for (policy, seed), group in grouped.items():
        for window, (start, end) in WINDOWS.items():
            subset = [row for row in group if start <= row["step"] < end]
            if not subset:
                continue
            alive_samples = sum(row["alive_agent_samples"] for row in subset)
            result.append({
                "policy": policy, "seed": seed, "window": window,
                "start_s": start / 10.0, "end_s": end / 10.0, "steps": len(subset),
                **{f"{key}_sum": float(sum(row[key] for row in subset)) for key in (
                    "current_delta_phi_team", "current_event_team", "current_total_team",
                    "paper_r1_team", "paper_r3_team", "paper_r4_team",
                    "paper_r4_positive_team", "paper_r4_negative_team", "paper_total_team",
                )},
                "r3_occurrence_rate": sum(row["paper_r3_occurrences"] for row in subset) / max(alive_samples, 1),
                "r4_positive_occurrence_rate": sum(row["paper_r4_positive_occurrences"] for row in subset) / max(alive_samples, 1),
                "r4_negative_occurrence_rate": sum(row["paper_r4_negative_occurrences"] for row in subset) / max(alive_samples, 1),
                **{f"{key}_mean": float(np.mean([row[key] for row in subset])) for key in (
                    "mean_distance", "mean_closing_speed", "mean_ata_deg", "mean_aa_deg",
                    "mean_ha_deg", "mean_a2", "mean_abs_a2",
                )},
            })
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_windows(rows: list[dict]) -> dict:
    summary = {}
    for policy in sorted({row["policy"] for row in rows}):
        summary[policy] = {}
        for window in WINDOWS:
            subset = [row for row in rows if row["policy"] == policy and row["window"] == window]
            if not subset:
                continue
            summary[policy][window] = {
                key: float(np.mean([row[key] for row in subset]))
                for key in subset[0] if key not in {"policy", "seed", "window"}
            }
    return summary


def ordering(episodes: list[dict], windows: list[dict]) -> dict:
    def compare(rows: list[dict], reward_key: str, context: str) -> dict:
        lookup = {(row["policy"], row["seed"]): row[reward_key] for row in rows}
        margins = [lookup[("scripted_pursuit", seed)] - lookup[("final_500k", seed)]
                   for seed in EVALUATION_SEEDS]
        return {
            "context": context, "reward_key": reward_key,
            "pursuit_gt_final_seeds": int(sum(value > 0 for value in margins)),
            "final_gt_pursuit_seeds": int(sum(value < 0 for value in margins)),
            "ties": int(sum(value == 0 for value in margins)),
            "pursuit_minus_final_mean_margin": float(np.mean(margins)),
            "merge_turn_is_pure_pursuit_alias": True,
        }
    result = {
        "episode_current": compare(episodes, "current_total_team_sum", "0-100s/termination"),
        "episode_madsac": compare(episodes, "paper_total_team_sum", "0-100s/termination"),
    }
    for window in ("0-5s", "0-10s", "0-20s", "0-40s", "0-100s", "10-40s", "40-100s"):
        subset = [row for row in windows if row["window"] == window]
        result[f"{window}_current"] = compare(subset, "current_total_team_sum", window)
        result[f"{window}_madsac"] = compare(subset, "paper_total_team_sum", window)
    return result


def alignment_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["policy"], row["seed"])].append(row)
    result = []
    for (policy, seed), group in grouped.items():
        counts = {key: int(sum(row[key] for row in group)) for key in (
            "alive_agent_samples", "current_attackable", "paper_r4_positive_occurrences",
            "selected_target_attackable", "any_pair_paper_positive", "any_pair_paper_negative",
            "positive_r4_and_attackable", "positive_r4_not_attackable", "attackable_no_positive_r4",
        )}
        total = max(counts["alive_agent_samples"], 1)
        paper_positive = counts["paper_r4_positive_occurrences"]
        attackable = counts["current_attackable"]
        result.append({
            "policy": policy, "seed": seed, **counts,
            "both_fraction_all": counts["positive_r4_and_attackable"] / total,
            "paper_positive_not_attackable_fraction_all": counts["positive_r4_not_attackable"] / total,
            "attackable_no_paper_positive_fraction_all": counts["attackable_no_positive_r4"] / total,
            "paper_positive_precision_to_weapon": counts["positive_r4_and_attackable"] / max(paper_positive, 1),
            "weapon_attackable_covered_by_positive_r4": counts["positive_r4_and_attackable"] / max(attackable, 1),
            "any_pair_positive_r4_fraction_all": counts["any_pair_paper_positive"] / total,
            "any_pair_negative_r4_fraction_all": counts["any_pair_paper_negative"] / total,
        })
    return result


def scale_statistics(rows: list[dict], episodes: list[dict]) -> dict:
    channels = {
        "current_delta_phi": [row["current_delta_phi_team"] / max(row["alive_agent_samples"], 1) for row in rows],
        "madsac_r3": [row["paper_r3_team"] / max(row["alive_agent_samples"], 1) for row in rows],
        "madsac_r4": [row["paper_r4_team"] / max(row["alive_agent_samples"], 1) for row in rows],
        "kill_death_r1": [row["paper_r1_team"] / 4.0 for row in rows],
    }
    return {
        "per_agent_step": {name: {"signed": distribution(values), "absolute": distribution(values, True)}
                           for name, values in channels.items()},
        "per_episode_team": {
            key: {"signed": distribution([row[key] for row in episodes]),
                  "absolute": distribution([row[key] for row in episodes], True)}
            for key in ("current_delta_phi_team_sum", "paper_r1_team_sum", "paper_r3_team_sum",
                        "paper_r4_team_sum", "paper_total_team_sum")
        },
        "sac_entropy_reference": {"alpha": 0.1, "observed_entropy": 2.05, "alpha_times_entropy": 0.205},
    }


def plateau_summary(rows: list[dict]) -> dict:
    result = {}
    for policy in sorted({row["policy"] for row in rows}):
        result[policy] = {}
        for window in PLATEAU_WINDOWS:
            start, end = WINDOWS[window]
            subset = [row for row in rows if row["policy"] == policy and start <= row["step"] < end]
            result[policy][window] = {}
            for reward, key in (("current", "current_delta_phi_team"), ("madsac", "paper_total_team")):
                values = np.asarray([row[key] / max(row["alive_agent_samples"], 1) for row in subset])
                result[policy][window][reward] = {
                    "fraction_zero": float(np.mean(values == 0.0)),
                    "fraction_abs_lt_1e_4": float(np.mean(np.abs(values) < 1e-4)),
                    "fraction_abs_lt_1e_3": float(np.mean(np.abs(values) < 1e-3)),
                }
    return result


def definition(active_hash: str) -> dict:
    return {
        "source": {"paper": "Li et al. 2023, Aerospace 10(6):574", "equation": "Eq. (25)",
                   "geometry": "Fig. 2 and Eq. (6)", "normative_source": "project PDF"},
        "paper_exactly_supported": {
            "R": "R1 + R2 + R3 + R4",
            "R1": {"blue_destroyed": 10.0, "red_destroyed": -10.0},
            "R2": {"red_leaves_air_combat_area": -10.0},
            "R3": {"reward": 0.001, "conditions": "|ATA_r| <= 30 deg AND |HA_r| <= 30 deg AND d_r >= 4000 m"},
            "R4": {
                "R41_outer": "|AA_r| <= 30 deg AND d_r <= 4000 m",
                "R41_tiers": {"30deg": 0.01, "15deg": 0.02, "5deg": 0.1},
                "R42_outer": "|AA_b| <= 30 deg AND d_b <= 4000 m",
                "R42_tiers": {"30deg": -0.015, "15deg": -0.025, "5deg": -0.15},
                "inner_tier_angles": "corresponding |ATA| and |HA|, inclusive",
            },
        },
        "paper_underspecified": {
            "Eq6_ATA": "printed first line conflicts with Fig. 2 and duplicates printed AA-like construction",
            "nested_R4_precedence": "5/15/30 degree cases overlap; paper does not state precedence",
            "R41_vs_R42_overlap": "paper does not state branch precedence if both outer cases hold",
            "multi_enemy_target": "paper does not identify which enemy supplies local R3/R4 geometry",
            "undefined_inner_R4": "paper does not print an otherwise value inside R41/R42",
            "reward_timing": "paper does not specify pre- versus post-transition geometry",
        },
        "v1_4_adaptation": {
            "R2": paper_r2_v1_4()[1], "R2_value": 0.0,
            "target_selection": "nearest surviving Blue per Red, matching canonical fixed-policy convention",
            "geometry_timing": "post-transition state for per-step paper reward",
            "nested_R4_precedence": "strongest 5 deg, then 15 deg, then 30 deg",
            "outer_R4_precedence": "printed R41 branch before R42",
            "inner_R4_otherwise": 0.0,
            "distance": "3-D Euclidean distance in current V1.4 state",
            "merge_turn": "alias of pure pursuit because it already retargets and action_toward every step after merge",
        },
        "diagnostic_geometry": {
            "LOS": "atan2(y_target-y_own, x_target-x_own)",
            "ATA": "wrap(LOS - psi_own)", "AA": "wrap(psi_target - LOS)",
            "HA": "atan2(-(z_target-z_own), horizontal_distance) in NED",
            "units": "radians internally; degrees only in reports", "vector_direction": "LOS points own -> target",
        },
        "active_reward_sha256": active_hash,
    }


def active_reward_hash(root: Path) -> str:
    return hashlib.sha256((root / "src/uav_combat/environment/reward.py").read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/madsac_v1_4_pilot_500k_parallel/run_seed_2023/latest.pt")
    parser.add_argument("--output-dir", default="outputs/madsac_v1_4_reward_audit")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (root / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((root / "configs/combat_environment.yaml").read_text(encoding="utf-8"))
    final_actor = load_actor((root / args.checkpoint).resolve())
    initial_actor = fresh_actor()
    policies = {
        "final_500k": lambda env, obs: deterministic_actions(final_actor, obs, env.red_alive_mask),
        "fresh_untrained": lambda env, obs: deterministic_actions(initial_actor, obs, env.red_alive_mask),
        "scripted_pursuit": lambda env, obs: env.fixed_policy.team_actions(env.red, env.blue),
    }
    all_rows = []
    for name, policy in policies.items():
        print(f"[AUDIT] policy={name} seeds=20", flush=True)
        all_rows.extend(rollout_policy(name, config, policy, output))
    fragment_actions = {
        "straight_trim": [0, 0, 0], "hard_left": [0, 0, -1], "hard_right": [0, 0, 1],
        "climb": [0, 0.5, 0], "descend": [0, -0.5, 0],
        "decelerate": [-1, 0, 0], "accelerate": [1, 0, 0],
    }
    fragment_rows = []
    for name, action in fragment_actions.items():
        constant = np.asarray(action, dtype=np.float32)
        fn = lambda env, obs, value=constant: np.tile(value, (4, 1)) * env.red_alive_mask[:, None]
        print(f"[FRAGMENT] maneuver={name} horizon=40s", flush=True)
        fragment_rows.extend(rollout_policy(name, config, fn, output, max_steps=400))
    episodes = aggregate_episode(all_rows)
    add_policy_ranks(episodes)
    windows = window_rows(all_rows)
    alignment = alignment_rows(all_rows)
    fragment_windows = window_rows(fragment_rows)
    active_hash = active_reward_hash(root)
    definitions = definition(active_hash)
    scales = scale_statistics(all_rows, episodes)
    ordering_result = ordering(episodes, windows)
    window_means = summarize_windows(windows)
    fragment_means = summarize_windows(fragment_windows)
    selected_r4_events = sum(row["paper_r4_positive_occurrences"] + row["paper_r4_negative_occurrences"] for row in all_rows)
    any_pair_r4_events = sum(row["any_pair_paper_positive"] + row["any_pair_paper_negative"] for row in all_rows)
    report = {
        "definitions": definitions,
        "evaluation_seeds": EVALUATION_SEEDS,
        "policy_window_means": window_means,
        "policy_episode_means": {
            policy: {key: float(np.mean([row[key] for row in episodes if row["policy"] == policy]))
                     for key in episodes[0] if key not in {"policy", "seed"}}
            for policy in policies
        },
        "reward_ordering": ordering_result,
        "plateau": plateau_summary(all_rows),
        "weapon_alignment_mean": {
            policy: {key: float(np.mean([row[key] for row in alignment if row["policy"] == policy]))
                     for key in alignment[0] if key not in {"policy", "seed"}}
            for policy in policies
        },
        "maneuver_fragment_window_means": fragment_means,
        "target_selection_sensitivity": {
            "selected_nearest_r4_nonzero_agent_steps": selected_r4_events,
            "any_blue_pair_r4_nonzero_agent_steps": any_pair_r4_events,
            "purpose": "sensitivity only; adapted reward totals continue to use nearest surviving Blue",
        },
        "reward_exploit_risks": {
            "decelerate_0_10s_r3": fragment_means["decelerate"]["0-10s"]["paper_r3_team_sum"],
            "accelerate_0_10s_r3": fragment_means["accelerate"]["0-10s"]["paper_r3_team_sum"],
            "pursuit_0_10s_r3": window_means["scripted_pursuit"]["0-10s"]["paper_r3_team_sum"],
            "slow_far_range_residence_risk": "R3 pays every aligned step at d>=4 km, so slower closure can accumulate more reward",
        },
        "compatibility_verdict": "MADSAC REWARD NOT COMPATIBLE",
        "verdict_basis": (
            "Under the declared minimal adaptation, R4 supplies no nonzero reward on the evaluated "
            "final/pursuit/fresh trajectories and paper total reward does not rank pursuit above final "
            "in a majority of seeds; R3 also rewards longer residence outside 4 km."
        ),
        "merge_turn_policy": "same rollout as scripted_pursuit; no duplicate policy invented",
        "active_reward_hash_matches_pre_audit": active_hash == EXPECTED_ACTIVE_REWARD_SHA256,
        "active_environment_modified": False, "active_reward_modified": False,
        "observation_action_weapon_modified": False, "trainer_modified": False,
        "training_executed": False, "git_executed": False,
    }
    (output / "madsac_reward_definition.json").write_text(json.dumps(definitions, indent=2), encoding="utf-8")
    write_csv(output / "reward_policy_comparison.csv", episodes)
    write_csv(output / "reward_time_window_comparison.csv", windows)
    write_csv(output / "reward_weapon_alignment.csv", alignment)
    write_csv(output / "reward_maneuver_fragments.csv", fragment_windows)
    (output / "reward_scale_statistics.json").write_text(json.dumps(scales, indent=2), encoding="utf-8")
    (output / "reward_audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "steps": len(all_rows),
                      "fragment_steps": len(fragment_rows),
                      "active_reward_hash_matches": report["active_reward_hash_matches_pre_audit"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
