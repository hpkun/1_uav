"""2024-style multi-agent situation, allocation, contribution, and terminal rewards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from uav_env.combat.attack_geometry import AttackZoneConfig, compute_combat_geometry
from uav_env.combat.events import EpisodeOutcome
from uav_env.core.state import UAVState
from uav_env.entities.uav import UAV
from uav_env.rewards.components import angle_reward, paper_distance_reward, paper_height_reward, speed_reward


@dataclass(frozen=True)
class MultiAgentRewardBreakdown:
    """Per-red-agent multi-combat reward accounting."""

    situation: float
    event: float
    raw_dense: float
    assigned_dense: float
    terminal: float
    total: float
    contribution_score: float


def pair_situation_reward(previous_red: UAVState, previous_blue: UAVState, red: UAVState, blue: UAVState, config: dict[str, Any]) -> float:
    """Return the published four-component red-blue situation reward."""

    attack_config = AttackZoneConfig.from_config(config)
    previous_geometry = compute_combat_geometry(previous_red, previous_blue, attack_config)
    geometry = compute_combat_geometry(red, blue, attack_config)
    angle = angle_reward(geometry.attacker_attack_angle, geometry.target_escape_angle)
    distance = paper_distance_reward(
        previous_geometry.distance, geometry.distance, float(config["attack_distance_min"]),
        float(config["attack_distance_max"]), float(config["advantage_distance_max"]), float(config["desired_distance_max"]),
    )[0]
    height_cfg = config["height_reward"]
    height = paper_height_reward(
        red.z - blue.z, float(config["attack_distance_max"]), float(height_cfg["H_max"]),
        float(height_cfg["H_adv"]), float(height_cfg["H_att"]), float(height_cfg["H_min"]),
    )
    speed = speed_reward(red.speed, blue.speed)
    return 0.15 * angle + 0.60 * distance + 0.10 * speed + 0.15 * height


def individual_situation_reward(red: UAV, living_blues: Sequence[UAV], previous_states: Mapping[str, UAVState], config: dict[str, Any]) -> float:
    """Use the maximum pair situation reward, or zero when no blue survives."""

    values = [pair_situation_reward(previous_states[red.uav_id], previous_states[blue.uav_id], red.state, blue.state, config) for blue in living_blues if blue.is_alive]
    return max(values) if values else 0.0


def assign_dense_rewards(raw_dense: Mapping[str, float], alive: Mapping[str, bool], r_den0: float = 0.01) -> dict[str, float]:
    """Implement published Algorithm 2 branches plus the documented negative-alive convention."""

    count = len(raw_dense)
    if count == 0 or set(raw_dense) != set(alive):
        raise ValueError("raw_dense and alive must contain the same nonempty agents")
    minimum = min(raw_dense.values())
    positive = {key: value for key, value in raw_dense.items() if alive[key] and value > 0.01}
    alpha = sum(positive.values())
    result: dict[str, float] = {}
    for agent_id, value in raw_dense.items():
        if not alive[agent_id]:
            result[agent_id] = -r_den0 * count - minimum
        elif value > 0.01:
            factor = r_den0 * count + 0.003 * len(positive) / count + 0.007 * alpha / count
            result[agent_id] = factor * value / alpha if alpha > 0.0 else 0.0
        elif value > -0.01:
            result[agent_id] = 0.0
        else:
            result[agent_id] = value
    return result


def multi_terminal_rewards(
    outcome: EpisodeOutcome,
    red_aircraft: Sequence[UAV],
    contribution_scores: Mapping[str, float],
    config: dict[str, Any],
) -> dict[str, float]:
    """Apply zero-safe project weights to the published multi-terminal structure."""

    if outcome.termination_reason == "ongoing":
        return {u.uav_id: 0.0 for u in red_aircraft}
    assumptions = config["project_assumptions"]["multi_terminal_reward"]
    if outcome.winner == "draw":
        return {u.uav_id: float(assumptions["draw_reward"]) for u in red_aircraft}
    weights = assumptions["win_weights"] if outcome.winner == "red" else assumptions["lose_weights"]
    base = float(config["r_win0"] if outcome.winner == "red" else config["r_lose0"])
    total_health = sum(max(0.0, u.state.health) for u in red_aircraft)
    total_beta = sum(max(0.0, contribution_scores.get(u.uav_id, 0.0)) for u in red_aircraft)
    team_health_ratio = total_health / (len(red_aircraft) * float(config["initial_health"]))
    rewards: dict[str, float] = {}
    for aircraft in red_aircraft:
        own_health_ratio = max(0.0, aircraft.state.health) / float(config["initial_health"])
        beta_share = max(0.0, contribution_scores.get(aircraft.uav_id, 0.0)) / total_beta if total_beta > 0.0 else 1.0 / len(red_aircraft)
        if outcome.winner == "red":
            factor = float(weights[0]) * team_health_ratio + float(weights[1]) * own_health_ratio + float(weights[2]) * beta_share
        else:
            factor = float(weights[0]) * (1.0 - team_health_ratio) + float(weights[1]) * (1.0 - own_health_ratio) + float(weights[2]) * (1.0 - beta_share)
        rewards[aircraft.uav_id] = base * factor
    return rewards


def multi_reward(team: Sequence[UAVState], opponents: Sequence[UAVState]) -> float:
    """Legacy scalar helper intentionally replaced by per-agent environment rewards."""

    del team, opponents
    raise ValueError("Use MultiAgentRewardBreakdown from CombatMultiEnv")
