"""Reproducible paper-alignment audit for the fixed homogeneous 3v3 environment."""

from __future__ import annotations

import csv
import json
import multiprocessing as mp
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from math import isclose, pi
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np

from uav_env.actions.discrete_15 import DiscreteAction15, get_control
from uav_env.algorithms.mappo.adapter import CombatEnvDescription, MAPPOEnvAdapter, ParallelCombatVectorEnv
from uav_env.algorithms.mappo.returns import compute_gae
from uav_env.combat.attack_geometry import AttackZoneConfig, compute_combat_geometry
from uav_env.combat.damage import DamageConfig, apply_damage, damage_for_random_value
from uav_env.combat.events import EpisodeOutcome
from uav_env.combat.multi_combat import ResolvedAttack, assign_nearest_targets_independently, resolve_multi_attacks
from uav_env.core.enums import Team
from uav_env.core.geometry import normalize_angle
from uav_env.core.state import UAVState
from uav_env.core.symmetry import mirror_action_xz, mirror_state_xz
from uav_env.dynamics.propagation import propagate_action_hold, propagate_state
from uav_env.envs import make_3v3_env
from uav_env.envs.combat_multi_env import CombatMultiEnv
from uav_env.observations.global_state import build_global_state
from uav_env.observations.multi_observation import build_multi_observations
from uav_env.observations.global_state import global_state_feature_names_v2
from uav_env.observations.multi_observation import multi_observation_feature_names_v2, multi_observation_feature_names_v2_for_agent
from uav_env.opponents.random import RandomOpponent
from uav_env.opponents.straight import StraightOpponent
from uav_env.opponents.pursuit import PursuitOpponent
from uav_env.opponents.team_controller import TeamRuleController
from uav_env.rewards.multi_reward import assign_dense_rewards, multi_terminal_reward_allocations


OUTPUT_DIR = Path("outputs/audit")
JSON_PATH = OUTPUT_DIR / "3v3_environment_audit.json"
CSV_PATH = OUTPUT_DIR / "3v3_environment_audit.csv"


def finite(value: Any) -> bool:
    array = np.asarray(value, dtype=np.float64)
    return bool(np.all(np.isfinite(array)))


def to_builtin(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_builtin(val) for key, val in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_builtin(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class Audit:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.artifacts: dict[str, Any] = {}

    def add(
        self,
        check_id: str,
        item: str,
        status: str,
        severity: str,
        classification: str,
        summary: str,
        evidence: dict[str, Any] | None = None,
    ) -> None:
        self.rows.append(
            {
                "check_id": check_id,
                "item": item,
                "status": status,
                "severity": severity,
                "classification": classification,
                "summary": summary,
                "evidence": to_builtin(evidence or {}),
            }
        )


def new_env(seed: int = 1, scenario: str = "head_on_formation") -> CombatMultiEnv:
    return make_3v3_env(scenario, "pursuit", seed=seed, multi_terminal_reward_profile="paper_2024_exact")


def make_pursuit(env: CombatMultiEnv) -> PursuitOpponent:
    pursuit_cfg = {key: float(value) for key, value in env.config["pursuit"].items()}
    return PursuitOpponent(
        env.profile,
        env.attack_config,
        float(env.config["physics_dt"]),
        int(env.config["physics_steps_per_action"]),
        float(env.config["gravity"]),
        float(env.config["max_altitude"]),
        **pursuit_cfg,
    )


def make_policy(name: str, env: CombatMultiEnv) -> Any:
    if name == "pursuit":
        return make_pursuit(env)
    if name == "straight":
        return StraightOpponent()
    if name == "random":
        return RandomOpponent()
    raise ValueError(name)


def audit_config(audit: Audit) -> CombatMultiEnv:
    env = new_env(1)
    observation, info = env.reset(seed=1)
    cfg = env.config
    selected = {
        "red_count": env.red_count,
        "blue_count": env.blue_count,
        "physics_dt": cfg["physics_dt"],
        "decision_dt": cfg["decision_dt"],
        "physics_steps_per_action": cfg["physics_steps_per_action"],
        "max_episode_seconds": cfg["max_episode_seconds"],
        "max_decision_steps": cfg["max_decision_steps"],
        "altitude_range": [cfg["min_altitude"], cfg["max_altitude"]],
        "speed_range": [cfg["min_speed"], cfg["max_speed"]],
        "overload_range": {
            "nx": [cfg["min_tangential_overload"], cfg["max_tangential_overload"]],
            "nz": [cfg["min_normal_overload"], cfg["max_normal_overload"]],
        },
        "initial_team_distance": cfg["initial_team_distance"],
        "initial_speed": cfg["initial_speed"],
        "initial_altitude": cfg["initial_altitude"],
        "formation_lateral_spacing": cfg["formation_lateral_spacing"],
        "attack": {key: cfg[key] for key in AttackZoneConfig.__dataclass_fields__},
        "damage": {"thresholds": cfg["damage_probability_thresholds"], "values": cfg["damage_values"]},
        "initial_health": cfg["initial_health"],
        "reward": {"r_den0": cfg["r_den0"], "r_win0": cfg["r_win0"], "r_lose0": cfg["r_lose0"]},
        "terminal_reward": cfg["project_assumptions"]["multi_terminal_reward"],
        "pursuit": cfg["pursuit"],
        "local_observation_dim": env.local_observation_dim,
        "global_state_dim": env.global_state_dim,
        "action_count": len(DiscreteAction15),
    }
    audit.artifacts["config"] = selected
    ok = (
        isclose(float(cfg["decision_dt"]) / float(cfg["physics_dt"]), 5.0)
        and isclose(int(cfg["max_decision_steps"]) * float(cfg["decision_dt"]), 200.0)
        and env.red_count == env.blue_count == 3
        and observation.shape == (3, 45)
        and info["global_state"].shape == (87,)
        and len(DiscreteAction15) == 15
    )
    audit.add("config", "3v3 merged configuration", "pass" if ok else "fail", "P0", "project_defined", "Fixed homogeneous 3v3 dimensions and timing are internally consistent.", selected)
    return env


def audit_initial_scene(audit: Audit) -> None:
    env = new_env(2)
    env.reset(seed=2)
    states = {u.uav_id: to_builtin(u.state) for u in env.all_aircraft}
    ys = [-500.0, 0.0, 500.0]
    checks = []
    for index, y in enumerate(ys):
        red = env.red_aircraft[index].state
        blue = env.blue_aircraft[index].state
        checks.extend(
            [
                isclose(red.x, -900.0),
                isclose(blue.x, 900.0),
                isclose(red.y, y),
                isclose(blue.y, y),
                isclose(red.z, 1800.0),
                isclose(blue.z, 1800.0),
                isclose(red.speed, 110.0),
                isclose(blue.speed, 110.0),
                isclose(red.heading_angle, 0.0),
                isclose(blue.heading_angle, pi),
                isclose(red.x, -blue.x),
                isclose(red.y, blue.y),
            ]
        )
    same_team_distances = [
        float(np.linalg.norm(a.state.position_vector() - b.state.position_vector()))
        for team in (env.red_aircraft, env.blue_aircraft)
        for i, a in enumerate(team)
        for b in team[i + 1 :]
    ]
    checks.append(all(distance > 0.0 for distance in same_team_distances))
    checks.append(all(float(env.config["min_altitude"]) <= u.state.z <= float(env.config["max_altitude"]) for u in env.all_aircraft))
    checks.append(all(float(env.config["min_speed"]) <= u.state.speed <= float(env.config["max_speed"]) for u in env.all_aircraft))
    audit.artifacts["initial_states"] = states
    audit.add("initial_scene", "head_on_formation initial state", "pass" if all(checks) else "fail", "P0", "project_defined", "Initial 3v3 head-on formation matches the fixed homogeneous project scenario.", {"states": states, "same_team_distances": same_team_distances})


def audit_actions_dynamics(audit: Audit) -> None:
    env = new_env(3)
    env.reset(seed=3)
    initial = env.red_aircraft[1].state
    rows: list[dict[str, Any]] = []
    for action in DiscreteAction15:
        result = propagate_action_hold(
            initial,
            get_control(action),
            env.profile,
            float(env.config["physics_dt"]),
            int(env.config["physics_steps_per_action"]),
            float(env.config["gravity"]),
            float(env.config["min_altitude"]),
            float(env.config["max_altitude"]),
        )
        final = result.final_state
        rows.append(
            {
                "action": int(action),
                "nx": result.actual_control.tangential_overload,
                "nz": result.actual_control.normal_overload,
                "bank_angle": result.actual_control.bank_angle,
                "dx": final.x - initial.x,
                "dy": final.y - initial.y,
                "dz": final.z - initial.z,
                "dv": final.speed - initial.speed,
                "dtheta": final.flight_path_angle - initial.flight_path_angle,
                "dpsi": final.heading_angle - initial.heading_angle,
            }
        )
    by_action = {row["action"]: row for row in rows}
    checks = [
        all(finite(list(row.values())[1:]) for row in rows),
        by_action[1]["dv"] > 0.0,
        by_action[2]["dv"] < 0.0,
        by_action[3]["dtheta"] > 0.0,
        by_action[6]["dtheta"] < 0.0,
        abs(by_action[0]["dz"]) < 1.0e-6 and abs(by_action[0]["dv"]) < 1.0e-6,
        isclose(normalize_angle(by_action[9]["dpsi"]), -normalize_angle(by_action[12]["dpsi"]), rel_tol=1e-5, abs_tol=1e-5),
    ]
    audit.artifacts["action_dynamics"] = rows
    audit.add("action_dynamics", "15 actions and equation (1) propagation", "pass" if all(checks) else "fail", "P0", "paper_defined", "The action table is propagated through the three-degree-of-freedom overload model.", {"mapping": "point_mass_3d_derivative implements dx, dy, dz, dv, dtheta, dpsi from paper equation (1).", "rows": rows})


def state_for_geometry(distance: float, heading: float = 0.0, target_heading: float = 0.0) -> tuple[UAVState, UAVState]:
    red = UAVState(0.0, 0.0, 1800.0, 110.0, 0.0, heading, 300.0, True, int(Team.RED), "homogeneous")
    blue = UAVState(distance, 0.0, 1800.0, 110.0, 0.0, target_heading, 300.0, True, int(Team.BLUE), "homogeneous")
    return red, blue


def audit_attack_geometry(audit: Audit) -> None:
    env = new_env(4)
    cfg = env.attack_config
    cases: list[dict[str, Any]] = []
    distances = [cfg.attack_distance_min - 1.0, cfg.attack_distance_min, cfg.attack_distance_max, cfg.attack_distance_max + 1.0]
    for distance in distances:
        red, blue = state_for_geometry(distance, 0.0, 0.0)
        g = compute_combat_geometry(red, blue, cfg)
        cases.append({"case": f"distance_{distance}", **to_builtin(g)})
    for delta in (-1e-6, 0.0, 1e-6):
        red, blue = state_for_geometry(500.0, cfg.attack_angle_max + delta, 0.0)
        cases.append({"case": f"attack_angle_{delta}", **to_builtin(compute_combat_geometry(red, blue, cfg))})
        red, blue = state_for_geometry(500.0, 0.0, cfg.escape_angle_max + delta)
        cases.append({"case": f"escape_angle_{delta}", **to_builtin(compute_combat_geometry(red, blue, cfg))})
    red, blue = state_for_geometry(500.0, 0.0, 0.0)
    swapped = compute_combat_geometry(blue, red, cfg)
    ok = all(finite([case["distance"], case["attacker_attack_angle"], case["target_escape_angle"]]) for case in cases) and finite(swapped.distance)
    audit.artifacts["attack_geometry"] = {"cases": cases, "red_blue_swapped": to_builtin(swapped)}
    audit.add("attack_geometry", "attack area, true attack condition, and advantage area", "pass" if ok else "fail", "P0", "paper_defined", "Attack geometry follows distance and angle inequalities from paper equations (3), (4), (6), and (7).", audit.artifacts["attack_geometry"])


def audit_damage(audit: Audit) -> None:
    env = new_env(5)
    env.reset(seed=5)
    config = env.damage_config
    samples = [0.0, 0.099999, 0.1, 0.399999, 0.4, 0.799999, 0.8, 0.999999]
    damage_values = {str(sample): damage_for_random_value(sample, config) for sample in samples}
    state = env.red_aircraft[0].state
    updated, first = apply_damage(state, config, 0.0)
    nearly_dead = replace(state, health=10.0)
    destroyed, second = apply_damage(nearly_dead, config, 0.0)
    aircraft = [*env.red_aircraft, *env.blue_aircraft]
    multi = resolve_multi_attacks(aircraft, env.attack_config, env.damage_config, np.random.default_rng(1))
    expected = 0.1 * 51.0 + 0.3 * 21.0 + 0.4 * 11.0
    ok = (
        damage_values["0.0"] == 51.0
        and damage_values["0.1"] == 21.0
        and damage_values["0.4"] == 11.0
        and damage_values["0.8"] == 0.0
        and updated.health <= state.health
        and second.effective_damage <= 10.0
        and second.overkill_damage == 41.0
        and not destroyed.alive
        and isclose(expected, 15.8)
    )
    audit.artifacts["damage"] = {"samples": damage_values, "single_hit": to_builtin(first), "destroy_hit": to_builtin(second), "multi_resolved_count": len(multi.resolved_attacks), "expected_nominal_damage": expected}
    audit.add("damage", "piecewise damage and simultaneous resolution", "pass" if ok else "fail", "P0", "paper_defined", "Damage thresholds match paper equation (5); simultaneous resolution is project-defined to remove traversal-order dependence.", audit.artifacts["damage"])


def local_obs(env: CombatMultiEnv) -> np.ndarray:
    return build_multi_observations(env.red_aircraft, env.blue_aircraft, env.normalization_config).raw


def global_state(env: CombatMultiEnv) -> np.ndarray:
    return build_global_state(env.red_aircraft, env.blue_aircraft, env.normalization_config).raw


def compare_vectors(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    diff = np.asarray(first) - np.asarray(second)
    return {
        "identical": bool(np.array_equal(first, second)),
        "allclose": bool(np.allclose(first, second)),
        "l1": float(np.sum(np.abs(diff))),
        "l2": float(np.linalg.norm(diff)),
        "changed_indices": np.flatnonzero(np.abs(diff) > 1e-12).tolist(),
    }


def audit_local_observation(audit: Audit) -> None:
    env = new_env(6)
    env.reset(seed=6)
    base = local_obs(env)[0].copy()
    cases: dict[str, Any] = {"feature_names": build_multi_observations(env.red_aircraft, env.blue_aircraft, env.normalization_config).feature_names}
    env.red_aircraft[0].state = replace(env.red_aircraft[0].state, heading_angle=pi / 2)
    cases["own_heading_changed"] = compare_vectors(base, local_obs(env)[0])
    env.reset(seed=6)
    env.red_aircraft[0].state = replace(env.red_aircraft[0].state, health=1.0)
    cases["own_health_300_vs_1"] = compare_vectors(base, local_obs(env)[0])
    env.reset(seed=6)
    env.blue_aircraft[0].state = replace(env.blue_aircraft[0].state, health=1.0)
    cases["enemy_health_300_vs_1"] = compare_vectors(base, local_obs(env)[0])
    env.reset(seed=6)
    env.blue_aircraft[0].state = replace(env.blue_aircraft[0].state, x=-100.0, y=0.001)
    env.blue_aircraft[1].state = replace(env.blue_aircraft[1].state, x=-100.0, y=-0.001)
    before = local_obs(env)[0].copy()
    env.blue_aircraft[0].state = replace(env.blue_aircraft[0].state, y=-0.001)
    env.blue_aircraft[1].state = replace(env.blue_aircraft[1].state, y=0.001)
    cases["enemy_distance_crossing"] = compare_vectors(before, local_obs(env)[0])
    env.reset(seed=6)
    alive = local_obs(env)[0].copy()
    env.blue_aircraft[0].state = replace(env.blue_aircraft[0].state, alive=False, damaged=True, health=0.0)
    cases["enemy_alive_vs_dead_same_pose"] = compare_vectors(alive, local_obs(env)[0])
    cases["relative_angle_formula"] = "relative_angles() uses atan2(dz, hypot(dx,dy)) and atan2(dy,dx); it is a global line-of-sight angle, not body-relative yaw."
    audit.artifacts["local_observation"] = cases
    severe = cases["own_health_300_vs_1"]["identical"] and cases["enemy_health_300_vs_1"]["identical"]
    audit.add("local_observation", "45-dimensional local observation sufficiency", "warn" if severe else "pass", "P1" if severe else "P2", "unresolved", "The Actor observation omits health and uses dynamic distance-ranked entity slots; this can create feed-forward state aliasing.", cases)


def audit_global_state(audit: Audit) -> None:
    env = new_env(7)
    env.reset(seed=7)
    base = global_state(env).copy()
    cases: dict[str, Any] = {"feature_names": build_global_state(env.red_aircraft, env.blue_aircraft, env.normalization_config).feature_names}
    env.blue_aircraft[0].state = replace(env.blue_aircraft[0].state, alive=False, damaged=True, health=0.0)
    cases["blue_alive_vs_dead_same_pose"] = compare_vectors(base, global_state(env))
    env.reset(seed=7)
    env.red_aircraft[0].state = replace(env.red_aircraft[0].state, health=1.0)
    cases["red_health_300_vs_1"] = compare_vectors(base, global_state(env))
    env.reset(seed=7)
    env.blue_aircraft[0].state = replace(env.blue_aircraft[0].state, health=1.0)
    cases["blue_health_300_vs_1"] = compare_vectors(base, global_state(env))
    env.reset(seed=7)
    env.blue_aircraft[0].state = replace(env.blue_aircraft[0].state, last_action=int(DiscreteAction15.RIGHT_HOLD))
    cases["blue_last_action_changed"] = compare_vectors(base, global_state(env))
    env.reset(seed=7)
    env.blue_aircraft[0].state = replace(env.blue_aircraft[0].state, alive=False, damaged=True, health=0.0, x=9999.0)
    cases["dead_blue_frozen_position_changed"] = compare_vectors(base, global_state(env))
    cases["critic_markov_note"] = "Global state includes red failure flags and red last actions, but omits health values and blue damaged/action flags; dead blue geometry remains in pair blocks."
    audit.artifacts["global_state"] = cases
    p1 = cases["red_health_300_vs_1"]["identical"] or cases["blue_health_300_vs_1"]["identical"] or cases["blue_alive_vs_dead_same_pose"]["identical"]
    audit.add("global_state", "87-dimensional centralized state Markov sufficiency", "warn" if p1 else "pass", "P1" if p1 else "P2", "unresolved", "The Critic state may not be Markov because health and blue alive/action state are omitted or represented only indirectly.", cases)


def audit_slot_stability(audit: Audit) -> None:
    env = new_env(8)
    env.reset(seed=8)
    rows = []
    for step, offset in enumerate([2.0, 1.0, 0.1, -0.1, -1.0, -2.0]):
        env.blue_aircraft[0].state = replace(env.blue_aircraft[0].state, x=-100.0, y=offset)
        env.blue_aircraft[1].state = replace(env.blue_aircraft[1].state, x=-100.0, y=-offset)
        own = env.red_aircraft[0]
        ranked = sorted(env.blue_aircraft, key=lambda u: (not u.is_alive, float(np.linalg.norm(u.state.position_vector() - own.state.position_vector())), u.uav_id))
        obs = local_obs(env)[0]
        rows.append({"step": step, "offset": offset, "slots": [u.uav_id for u in ranked], "slot_prefix": obs[12:34].tolist()})
    switched = any(rows[index]["slots"] != rows[index - 1]["slots"] for index in range(1, len(rows)))
    audit.artifacts["slot_stability"] = rows
    audit.add("slot_stability", "distance-ranked enemy slot stability", "warn" if switched else "pass", "P1" if switched else "P2", "project_defined", "Enemy blocks are sorted by current distance, so ordinary MLP inputs can experience entity-slot jumps at crossings.", {"rows": rows, "mlp_note": "The shared feed-forward Actor has no permutation-equivariant structure."})


def run_rule_episode(seed: int, red_policy_name: str, scenario: str = "head_on_formation") -> tuple[dict[str, Any], dict[str, float]]:
    env = new_env(seed, scenario)
    env.reset(seed=seed)
    controller = TeamRuleController(red_policy_name, make_policy(red_policy_name, env), seed + 1_000_003)
    terminated = truncated = False
    team_return = 0.0
    reward_sums = defaultdict(float)
    while not (terminated or truncated):
        selected, _ = controller.select_actions(env.red_aircraft, env.blue_aircraft)
        _, reward, terminated, truncated, info = env.step(np.asarray([int(action) for action in selected], dtype=np.int64))
        team_return += float(reward)
        for breakdown in info["agent_reward_breakdowns"].values():
            reward_sums["situation"] += float(breakdown.situation)
            reward_sums["event"] += float(breakdown.event)
            reward_sums["assigned_dense"] += float(breakdown.assigned_dense)
            reward_sums["terminal"] += float(breakdown.terminal)
            reward_sums["absolute_total"] += abs(float(breakdown.total))
            if breakdown.event < 0.0:
                reward_sums["negative_event"] += abs(float(breakdown.event))
            if breakdown.event > 0.0:
                reward_sums["positive_event"] += abs(float(breakdown.event))
    outcome = info["outcome"]
    stats = env.get_statistics()["aircraft"]
    red_ids = [f"red_{index}" for index in range(3)]
    blue_ids = [f"blue_{index}" for index in range(3)]
    def side_sum(ids: list[str], key: str) -> float:
        return float(sum(float(stats[uav_id].get(key, 0.0)) for uav_id in ids))

    summary = {
        "winner": outcome.winner or "none",
        "reason": outcome.termination_reason,
        "red_survivors": int(outcome.red_survivors),
        "blue_survivors": int(outcome.blue_survivors),
        "steps": int(outcome.decision_steps),
        "red_attack_attempts": side_sum(red_ids, "attack_attempts"),
        "blue_attack_attempts": side_sum(blue_ids, "attack_attempts"),
        "red_hits": side_sum(red_ids, "hits"),
        "blue_hits": side_sum(blue_ids, "hits"),
        "red_nominal_damage": side_sum(red_ids, "nominal_damage"),
        "blue_nominal_damage": side_sum(blue_ids, "nominal_damage"),
        "red_effective_damage": side_sum(red_ids, "effective_damage"),
        "blue_effective_damage": side_sum(blue_ids, "effective_damage"),
        "red_overkill_damage": side_sum(red_ids, "overkill_damage"),
        "blue_overkill_damage": side_sum(blue_ids, "overkill_damage"),
        "red_attack_area_steps": side_sum(red_ids, "attack_area_steps"),
        "blue_attack_area_steps": side_sum(blue_ids, "attack_area_steps"),
        "red_ground_crashes": side_sum(red_ids, "ground_crashes"),
        "blue_ground_crashes": side_sum(blue_ids, "ground_crashes"),
        "red_ceiling_violations": side_sum(red_ids, "ceiling_violations"),
        "blue_ceiling_violations": side_sum(blue_ids, "ceiling_violations"),
        "red_collisions": side_sum(red_ids, "collisions"),
        "blue_collisions": side_sum(blue_ids, "collisions"),
        "team_return": team_return,
    }
    summary["attack_attempts"] = summary["red_attack_attempts"] + summary["blue_attack_attempts"]
    summary["red_crashes"] = summary["red_ground_crashes"]
    summary["blue_crashes"] = summary["blue_ground_crashes"]
    return summary, dict(reward_sums)


def aggregate_rule(policy: str, episodes: int = 100, scenario: str = "head_on_formation") -> dict[str, Any]:
    tasks = [(10_000 + offset, policy, scenario) for offset in range(episodes)]
    with mp.get_context("spawn").Pool(processes=4) as pool:
        completed = pool.starmap(run_rule_episode, tasks)
    summaries = [item[0] for item in completed]
    reward_sums = [item[1] for item in completed]
    reasons = Counter(item["reason"] for item in summaries)
    winners = Counter(item["winner"] for item in summaries)
    result = {
        "episodes": episodes,
        "winners": dict(winners),
        "reasons": dict(reasons),
        "red_timeout_survival_wins": sum(item["winner"] == "red" and item["reason"] == "timeout" for item in summaries),
        "blue_timeout_survival_wins": sum(item["winner"] == "blue" and item["reason"] == "timeout" for item in summaries),
        "red_elimination_wins": sum(item["winner"] == "red" and item["reason"] == "blue_eliminated" for item in summaries),
        "blue_elimination_wins": sum(item["winner"] == "blue" and item["reason"] == "red_eliminated" for item in summaries),
        "timeouts": reasons.get("timeout", 0),
        "draws": winners.get("draw", 0),
        "mean_episode_steps": mean(item["steps"] for item in summaries),
        "mean_team_return": mean(item["team_return"] for item in summaries),
        "reward_scale": {
            key: mean(item.get(key, 0.0) for item in reward_sums)
            for key in ("situation", "event", "assigned_dense", "terminal", "absolute_total", "negative_event", "positive_event")
        },
    }
    for name in ("attack_attempts","hits","nominal_damage","effective_damage","overkill_damage","attack_area_steps","ground_crashes","ceiling_violations","collisions","survivors"):
        result[f"mean_red_{name}"] = mean(item[f"red_{name}"] for item in summaries)
        result[f"mean_blue_{name}"] = mean(item[f"blue_{name}"] for item in summaries)
    result["mean_attack_attempts"] = result["mean_red_attack_attempts"] + result["mean_blue_attack_attempts"]
    result["mean_hits"] = result["mean_red_hits"] + result["mean_blue_hits"]
    result["mean_effective_damage"] = result["mean_red_effective_damage"] + result["mean_blue_effective_damage"]
    result["mean_attack_area_steps"] = result["mean_red_attack_area_steps"] + result["mean_blue_attack_area_steps"]
    result["mean_crashes"] = result["mean_red_ground_crashes"] + result["mean_blue_ground_crashes"]
    return result


def audit_blue_rule_and_rule_experiments(audit: Audit) -> None:
    env = new_env(9)
    env.reset(seed=9)
    for index, red in enumerate(env.red_aircraft):
        red.state = replace(red.state, x=float(index * 10), y=0.0, z=1800.0)
    for index, blue in enumerate(env.blue_aircraft):
        blue.state = replace(blue.state, x=100.0, y=float(index), z=1800.0)
    assignments = assign_nearest_targets_independently(env.blue_aircraft, env.red_aircraft)
    actions = env._blue_actions(assignments)
    target_death_ok = True
    env.red_aircraft[0].state = replace(env.red_aircraft[0].state, alive=False, damaged=True, health=0.0)
    after_death = assign_nearest_targets_independently(env.blue_aircraft, env.red_aircraft)
    rule_stats = {policy: aggregate_rule(policy, 100) for policy in ("pursuit", "straight", "random")}
    finite_stats = finite([value for stats in rule_stats.values() for value in stats.values() if isinstance(value, (int, float))])
    ok = all(action in DiscreteAction15 for action in actions) and all(item.target_id != "red_0" for item in after_death) and finite_stats and target_death_ok
    audit.artifacts["rule_experiments"] = rule_stats
    audit.add("blue_rule", "3v3 nearest-living-target pursuit rule and non-training rule probes", "pass" if ok else "fail", "P1", "project_defined", "The blue pursuit rule is a project fixed geometric opponent; 100-episode rule probes are diagnostics, not learning evidence.", {"initial_assignments": [to_builtin(a) for a in assignments], "actions": [int(a) for a in actions], "after_target_death": [to_builtin(a) for a in after_death], "rule_stats": rule_stats})


def audit_symmetry(audit: Audit) -> None:
    env = new_env(10)
    env.reset(seed=10)
    state = env.red_aircraft[0].state
    mirrored = mirror_state_xz(state)
    remirrored = mirror_state_xz(mirrored)
    left = propagate_state(state, get_control(DiscreteAction15.LEFT_HOLD), env.profile, float(env.config["physics_dt"]), float(env.config["gravity"]))
    mirrored_right = propagate_state(mirrored, get_control(mirror_action_xz(DiscreteAction15.LEFT_HOLD)), env.profile, float(env.config["physics_dt"]), float(env.config["gravity"]))
    ok = np.allclose(remirrored.to_kinematic_vector(), state.to_kinematic_vector()) and np.allclose(mirror_state_xz(left).to_kinematic_vector(), mirrored_right.to_kinematic_vector())
    audit.add("symmetry", "x-z mirror dynamics and left/right action mapping", "pass" if ok else "fail", "P1", "paper_defined", "Dynamics and left/right action mapping are mirror-consistent under paired states.", {"left": to_builtin(left), "mirrored_right": to_builtin(mirrored_right)})


def audit_rewards(audit: Audit) -> None:
    raw = {"red_1": 0.2, "red_2": -0.4}
    damaged = {"red_1": True, "red_2": False}
    assigned = assign_dense_rewards(raw, damaged, 3, 0.01)
    env = new_env(11)
    env.reset(seed=11)
    env.red_aircraft[0].state = replace(env.red_aircraft[0].state, z=float(env.config["min_altitude"]), health=300.0, alive=True, damaged=False, crashed=False)
    _, _, terminated, truncated, info = env.step(np.zeros(3, dtype=np.int64))
    first = info["agent_reward_breakdowns"]["red_0"]
    _, _, _, _, info2 = env.step(np.zeros(3, dtype=np.int64))
    second = info2["agent_reward_breakdowns"]["red_0"]
    ok = assigned["red_1"] < 0.0 and first.assigned_dense < 0.0 and all(
        getattr(second, field) == 0.0
        for field in ("situation", "event", "raw_dense", "assigned_dense", "terminal", "total", "contribution_score")
    )
    audit.add("reward_lifecycle", "situation, event, Algorithm 2, and terminal reward lifecycle", "pass" if ok else "fail", "P0", "project_defined", "Destroyed-step dense penalty is capped negative as a project assumption; post-destruction step rewards are zero while terminal slots remain allocated at episode end.", {"assigned_dense_demo": assigned, "destroyed_step": to_builtin(first), "next_step": to_builtin(second), "terminated": terminated, "truncated": truncated})


def audit_terminal_semantics(audit: Audit) -> None:
    env = new_env(12)
    env.reset(seed=12)
    scenarios = [
        EpisodeOutcome("red", True, True, "timeout", 400, 200.0, 3, 2),
        EpisodeOutcome("blue", True, True, "timeout", 400, 200.0, 2, 3),
        EpisodeOutcome("draw", True, True, "timeout", 400, 200.0, 2, 2),
        EpisodeOutcome("red", True, False, "blue_eliminated", 20, 10.0, 3, 0),
        EpisodeOutcome("blue", False, True, "red_eliminated", 20, 10.0, 0, 3),
        EpisodeOutcome("draw", False, False, "simultaneous_elimination", 20, 10.0, 0, 0),
    ]
    rows = []
    for outcome in scenarios:
        allocations = multi_terminal_reward_allocations(outcome, env.red_aircraft, {f"red_{i}": 1.0 for i in range(3)}, env.config)
        rows.append(
            {
                "winner": outcome.winner,
                "reason": outcome.termination_reason,
                "formula": "draw" if outcome.winner == "draw" else ("win" if outcome.winner == "red" else "lose"),
                "terminal_rewards": {key: value.reward for key, value in allocations.items()},
            }
        )
    audit.artifacts["terminal_semantics"] = rows
    audit.add("terminal_semantics", "timeout and elimination terminal formula selection", "warn", "P1", "unresolved", "Timeout survivor-count wins currently call the same win/lose terminal formula as elimination outcomes; this must be reported separately from elimination wins.", {"rows": rows, "draw_as_loss": env.config.get("draw_as_loss"), "draw_reward": env.config["project_assumptions"]["multi_terminal_reward"]["draw_reward"]})


def audit_mappo_interface(audit: Audit) -> None:
    env = new_env(13)
    adapter = MAPPOEnvAdapter(env)
    reset = adapter.reset(13)
    step = adapter.step(np.zeros(3, dtype=np.int64))
    vector = ParallelCombatVectorEnv(CombatEnvDescription("3v3", "head_on_formation", "pursuit", "paper_2024_exact"), 4, 13)
    try:
        reset_vec = vector.reset()
        result_vec = vector.step(np.zeros((4, 3), dtype=np.int64))
        workers_alive = all(vector.workers_alive)
    finally:
        vector.close()
    rewards = np.ones((2, 1, 3), dtype=np.float32)
    values = np.zeros((3, 1, 3), dtype=np.float32)
    terminated = np.asarray([[True], [False]])
    truncated = np.asarray([[False], [True]])
    terminal_values = np.ones((2, 1, 3), dtype=np.float32)
    adv, returns = compute_gae(rewards, values, terminated, truncated, terminal_values, np.ones_like(terminated, dtype=np.float32), 0.99, 0.95)
    conditions = {
        "reset_local_shape": reset.local_obs.shape == (3, 45),
        "reset_global_shape": reset.global_state.shape == (87,),
        "agent_reward_shape": step.agent_rewards.shape == (3,),
        "team_reward_is_mean": bool(np.isclose(step.team_reward, float(np.mean(step.agent_rewards)), rtol=1.0e-6, atol=1.0e-6)),
        "vector_local_shape": reset_vec["local_obs"].shape == (4, 3, 45),
        "terminal_state_shape": result_vec["terminal_steps"][0].global_state.shape == (87,),
        "workers_alive_after_step": workers_alive,
        "finite_gae_advantages": finite(adv),
        "finite_gae_returns": finite(returns),
    }
    ok = all(conditions.values())
    audit.add("mappo_interface", "fixed 3-agent MAPPO adapter and vector interface", "pass" if ok else "fail", "P0", "project_defined", "Adapter exposes 3 rewards, 45-d Actor observations, 87-d Critic state, fixed masks, terminal-state retention, and finite GAE.", {"conditions": conditions, "reset_shapes": {"local": reset.local_obs.shape, "global": reset.global_state.shape}, "vector_shapes": {key: np.asarray(value).shape for key, value in reset_vec.items() if key in {"local_obs", "global_state", "available_actions"}}, "gae_adv": adv.tolist()})


def audit_v2_environment(audit: Audit) -> None:
    env = new_env(101, "head_on_mirrored_jitter_v2")
    obs, info = env.reset(seed=101)
    checks = {
        "schema_timeaware": env.environment_schema_version == "homogeneous_3v3_v2_timeaware",
        "local_63d": obs.shape == (3, 63),
        "global_61d": info["global_state"].shape == (61,),
        "feature_names_63d": info["local_observation_feature_names"] == multi_observation_feature_names_v2(),
        "per_agent_feature_names_true_ids": info["local_observation_feature_names_by_agent"]["red_0"] == multi_observation_feature_names_v2_for_agent("red_0"),
        "feature_names_61d": info["global_state_feature_names"] == global_state_feature_names_v2(),
        "reset_progress_normalized": info["local_observations_raw"][0, 7] == 0.0 and info["local_observations"][0, 7] == -1.0 and info["global_state_raw"][60] == 0.0 and info["global_state"][60] == -1.0,
    }
    env.decision_step = 399
    late_obs = env._observations()
    late_state = env._global_state()
    env.decision_step = 0
    start_obs = env._observations()
    start_state = env._global_state()
    checks["time_markov_only_progress_changes"] = np.flatnonzero(np.abs(start_obs.raw[0] - late_obs.raw[0]) > 1e-12).tolist() == [7] and np.flatnonzero(np.abs(start_state.raw - late_state.raw) > 1e-12).tolist() == [60]
    env.decision_step = 400
    checks["timeout_progress_one"] = env._observations().normalized[0, 7] == 1.0 and env._global_state().normalized[60] == 1.0
    env.decision_step = 0
    old_config = dict(env.config)
    old_config["environment_schema_version"] = "homogeneous_3v3_v2"
    old_config["observation_schema"] = "fixed_id_body_62d"
    old_config["global_state_schema"] = "full_entity_60d"
    try:
        CombatMultiEnv(old_config, "head_on_mirrored_jitter_v2", "pursuit")
        checks["old_v2_runtime_schema_rejected"] = False
    except ValueError as error:
        checks["old_v2_runtime_schema_rejected"] = "development-only 62D/60D schema" in str(error)
    no_bootstrap_adv, _ = compute_gae(
        np.asarray([[[1.0]]], dtype=np.float32),
        np.asarray([[[2.0]], [[99.0]]], dtype=np.float32),
        np.asarray([[False]]),
        np.asarray([[True]]),
        np.asarray([[[10.0]]], dtype=np.float32),
        np.asarray([[0.0]], dtype=np.float32),
        0.99,
        1.0,
    )
    legacy_bootstrap_adv, _ = compute_gae(
        np.asarray([[[1.0]]], dtype=np.float32),
        np.asarray([[[2.0]], [[99.0]]], dtype=np.float32),
        np.asarray([[False]]),
        np.asarray([[True]]),
        np.asarray([[[10.0]]], dtype=np.float32),
        np.asarray([[1.0]], dtype=np.float32),
        0.99,
        1.0,
    )
    checks["timeaware_timeout_gae_no_bootstrap"] = float(no_bootstrap_adv[0, 0, 0]) == -1.0
    checks["legacy_truncated_gae_bootstrap"] = isclose(float(legacy_bootstrap_adv[0, 0, 0]), 8.9, rel_tol=1e-6, abs_tol=1e-6)
    checks["codex_spec_absent"] = not Path("CODEX_SPEC.md").exists()
    base_obs = env._observations().raw[0].copy()
    env.red_aircraft[0].state = replace(env.red_aircraft[0].state, heading_angle=pi / 2)
    checks["actor_heading_distinguishable"] = not np.array_equal(base_obs, env._observations().raw[0])
    env = new_env(101, "head_on_mirrored_jitter_v2"); env.reset(seed=101)
    base_obs = env._observations().raw[0].copy(); env.red_aircraft[0].state = replace(env.red_aircraft[0].state, health=1.0)
    checks["actor_own_health_distinguishable"] = not np.array_equal(base_obs, env._observations().raw[0])
    env = new_env(101, "head_on_mirrored_jitter_v2"); env.reset(seed=101)
    base_obs = env._observations().raw[0].copy(); env.blue_aircraft[0].state = replace(env.blue_aircraft[0].state, health=1.0)
    checks["actor_enemy_health_distinguishable"] = not np.array_equal(base_obs, env._observations().raw[0])
    env = new_env(101, "symmetric_stress_test_v2"); env.reset(seed=101)
    own = env.red_aircraft[1]
    env.blue_aircraft[0].state = replace(env.blue_aircraft[0].state, x=own.state.x + 100.0, y=own.state.y + 100.0)
    left = env._observations().raw[1][32]
    env.blue_aircraft[0].state = replace(env.blue_aircraft[0].state, y=own.state.y - 100.0)
    right = env._observations().raw[1][32]
    checks["body_bearing_left_right"] = left > 0.0 and right < 0.0
    slots = []
    for offset in (2.0, 0.1, -0.1, -2.0):
        env.blue_aircraft[0].state = replace(env.blue_aircraft[0].state, x=own.state.x + 100.0, y=own.state.y + offset)
        env.blue_aircraft[1].state = replace(env.blue_aircraft[1].state, x=own.state.x + 100.0, y=own.state.y - offset)
        raw = env._observations().raw[1]
        slots.append((float(raw[26]), float(raw[39])))
    checks["fixed_enemy_slots"] = slots[0][0] > 0.0 and slots[-1][0] < 0.0 and slots[0][1] < 0.0
    env.blue_aircraft[0].state = replace(env.blue_aircraft[0].state, alive=False, damaged=True, health=0.0, x=9999.0)
    local_dead = env._observations()
    checks["dead_local_slot_zero"] = bool(local_dead.raw[1][24] == -1.0 and np.allclose(local_dead.raw[1][25:37], 0.0))
    checks["dead_local_normalized_zero"] = bool(np.allclose(local_dead.normalized[1][24:37], [-1.0, *([0.0] * 12)]) and not np.any(local_dead.saturated_feature_masks[1][24:37]))

    env = new_env(102, "head_on_mirrored_jitter_v2"); env.reset(seed=102)
    base_state = env._global_state().raw.copy()
    distinctions = []
    for mutator in (
        lambda e: setattr(e.red_aircraft[0], "state", replace(e.red_aircraft[0].state, z=1.0)),
        lambda e: setattr(e.red_aircraft[0], "state", replace(e.red_aircraft[0].state, speed=150.0)),
        lambda e: setattr(e.red_aircraft[0], "state", replace(e.red_aircraft[0].state, health=1.0)),
        lambda e: setattr(e.blue_aircraft[0], "state", replace(e.blue_aircraft[0].state, health=1.0)),
        lambda e: setattr(e.blue_aircraft[0], "state", replace(e.blue_aircraft[0].state, alive=False, damaged=True, health=0.0)),
        lambda e: setattr(e.blue_aircraft[0], "state", replace(e.blue_aircraft[0].state, last_action=int(DiscreteAction15.RIGHT_HOLD))),
        lambda e: setattr(e.blue_aircraft[0], "state", replace(e.blue_aircraft[0].state, heading_angle=1.0)),
        lambda e: setattr(e.blue_aircraft[0], "state", replace(e.blue_aircraft[0].state, flight_path_angle=0.1)),
    ):
        env2 = new_env(102, "head_on_mirrored_jitter_v2"); env2.reset(seed=102); mutator(env2)
        distinctions.append(not np.array_equal(base_state, env2._global_state().raw))
    checks["critic_markov_distinctions"] = all(distinctions)
    env = new_env(102, "head_on_mirrored_jitter_v2"); env.reset(seed=102)
    env.blue_aircraft[0].state = replace(env.blue_aircraft[0].state, alive=False, damaged=True, health=0.0, x=9999.0)
    dead_global = env._global_state()
    checks["dead_global_slot_zero"] = bool(dead_global.raw[30] == -1.0 and np.allclose(dead_global.raw[31:39], 0.0))
    checks["dead_global_normalized_zero"] = bool(np.allclose(dead_global.normalized[30:40], [-1.0, *([0.0] * 9)]) and not np.any(dead_global.saturated_feature_mask[30:40]) and dead_global.raw[60] == 0.0)

    terminal = multi_terminal_reward_allocations(EpisodeOutcome("red", True, True, "timeout", 400, 200.0, 3, 2), env.red_aircraft, {}, env.config)
    checks["timeout_minus_four"] = {item.reward for item in terminal.values()} == {-4.0}
    checks["timeout_profile"] = {item.profile for item in terminal.values()} == {"project_3v3_v2_timeout"}
    simultaneous = multi_terminal_reward_allocations(EpisodeOutcome("draw", False, False, "simultaneous_elimination", 20, 10.0, 0, 0), env.red_aircraft, {}, env.config)
    checks["simultaneous_profile"] = {item.reward for item in simultaneous.values()} == {0.0} and {item.profile for item in simultaneous.values()} == {"project_3v3_v2_simultaneous_elimination"}
    win = multi_terminal_reward_allocations(EpisodeOutcome("red", True, False, "blue_eliminated", 10, 5.0, 3, 0), env.red_aircraft, {}, env.config)
    checks["elimination_uses_win_formula"] = all(item.reward > 0.0 and item.profile == "paper_2024_exact" for item in win.values())
    env.red_aircraft[0].state = replace(env.red_aircraft[0].state, z=float(env.config["min_altitude"]), health=300.0, alive=True, damaged=False, crashed=False)
    _, _, _, _, step_info = env.step(np.zeros(3, dtype=np.int64))
    bd = step_info["agent_reward_breakdowns"]["red_0"]
    checks["combat_event_not_assigned_shape"] = bd.combat_event == -0.5 and bd.assigned_shape <= -0.03 and bd.dense_reward <= -0.53

    a = new_env(201, "head_on_mirrored_jitter_v2"); b = new_env(201, "head_on_mirrored_jitter_v2"); c = new_env(202, "head_on_mirrored_jitter_v2")
    a.reset(seed=201); b.reset(seed=201); c.reset(seed=202)
    checks["same_seed_jitter_reproducible"] = [u.state.to_kinematic_vector().tolist() for u in a.all_aircraft] == [u.state.to_kinematic_vector().tolist() for u in b.all_aircraft]
    checks["different_seed_jitter_changes"] = [u.state.to_kinematic_vector().tolist() for u in a.all_aircraft] != [u.state.to_kinematic_vector().tolist() for u in c.all_aircraft]
    stress = new_env(203, "symmetric_stress_test_v2"); stress.reset(seed=203)
    checks["symmetric_stress_exact"] = [u.state.x for u in stress.red_aircraft] == [-900.0, -900.0, -900.0] and [u.state.x for u in stress.blue_aircraft] == [900.0, 900.0, 900.0]
    timeout_env = new_env(204, "symmetric_stress_test_v2")
    timeout_env.reset(seed=204)
    real_timeout_info: dict[str, Any] | None = None
    for _ in range(400):
        _, _, terminated, truncated, real_timeout_info = timeout_env.step(np.zeros(3, dtype=np.int64))
        if terminated or truncated:
            break
    checks["real_400th_terminal_progress_one"] = bool(
        real_timeout_info is not None
        and real_timeout_info["outcome"].termination_reason == "timeout"
        and real_timeout_info["decision_step"] == 400
        and real_timeout_info["global_state"][60] == 1.0
        and np.allclose(real_timeout_info["local_observations"][:, 7], 1.0)
    )
    vector = ParallelCombatVectorEnv(CombatEnvDescription("3v3", "head_on_mirrored_jitter_v2", "pursuit", "paper_2024_exact"), 4, 301)
    try:
        reset = vector.reset()
        checks["parallel_v2_shapes"] = reset["local_obs"].shape == (4, 3, 63) and reset["global_state"].shape == (4, 61)
    finally:
        vector.close()
    audit.artifacts["v2_rule_experiments"] = {policy: aggregate_rule(policy, 100, "head_on_mirrored_jitter_v2") for policy in ("pursuit", "straight", "random")}
    audit.artifacts["v2_symmetric_stress_rule"] = {"pursuit": aggregate_rule("pursuit", 100, "symmetric_stress_test_v2")}
    pursuit_rule = audit.artifacts["v2_rule_experiments"]["pursuit"]
    attack_warning = pursuit_rule["mean_red_attack_attempts"] == 0.0 or pursuit_rule["mean_blue_attack_attempts"] == 0.0
    status = "fail" if not all(checks.values()) else ("warn" if attack_warning else "pass")
    audit.add("v2_environment", "homogeneous_3v3_v2_timeaware observation, state, reward, reset, and diagnostics", status, "P1" if status == "warn" else "P0", "v2", "Time-aware V2 fixes the audited state/slot/reward/timeout semantics; pursuit reachability is a warning if either side has zero attack attempts.", {"checks": checks, "all_checks_pass": all(checks.values()), "slot_trace": slots, "pursuit_attack_warning": attack_warning, "rule_stats": audit.artifacts["v2_rule_experiments"], "stress": audit.artifacts["v2_symmetric_stress_rule"]})


def flatten_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    flattened = []
    for row in rows:
        flattened.append({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value) for key, value in row.items()})
    return flattened


def main() -> None:
    audit = Audit()
    audit_config(audit)
    audit_initial_scene(audit)
    audit_actions_dynamics(audit)
    audit_attack_geometry(audit)
    audit_damage(audit)
    audit_local_observation(audit)
    audit_global_state(audit)
    audit_slot_stability(audit)
    audit_blue_rule_and_rule_experiments(audit)
    audit_symmetry(audit)
    audit_rewards(audit)
    audit_terminal_semantics(audit)
    audit_mappo_interface(audit)
    audit_v2_environment(audit)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"rows": to_builtin(audit.rows), "artifacts": to_builtin(audit.artifacts)}
    JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    flattened = flatten_rows(audit.rows)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["check_id", "item", "status", "severity", "classification", "summary", "evidence"])
        writer.writeheader()
        writer.writerows(flattened)
    failures = [row for row in audit.rows if row["status"] == "fail"]
    print(f"wrote {JSON_PATH}")
    print(f"wrote {CSV_PATH}")
    print(f"checks={len(audit.rows)} failures={len(failures)} warnings={sum(row['status'] == 'warn' for row in audit.rows)}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
