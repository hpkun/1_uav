"""2024-style multi-agent situation, allocation, contribution, and terminal rewards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

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
    terminal_profile: str = "none"
    terminal_team_base: float = 0.0
    terminal_allocation_factor: float = 0.0
    terminal_base_share_component: float = 0.0
    terminal_survival_component: float = 0.0
    terminal_contribution_component: float = 0.0
    terminal_health_component: float = 0.0
    terminal_alive_count: int = 0
    terminal_contribution_denominator: float = 0.0
    terminal_health_denominator: float = 0.0


@dataclass(frozen=True)
class TerminalRewardAllocation:
    reward: float
    profile: str
    team_base: float
    allocation_factor: float
    base_share_component: float = 0.0
    survival_component: float = 0.0
    contribution_component: float = 0.0
    health_component: float = 0.0
    alive_count: int = 0
    contribution_denominator: float = 0.0
    health_denominator: float = 0.0


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
    """Return terminal rewards using the explicitly selected profile."""

    return {key: value.reward for key, value in multi_terminal_reward_allocations(outcome, red_aircraft, contribution_scores, config).items()}


def multi_terminal_reward_allocations(
    outcome: EpisodeOutcome, red_aircraft: Sequence[UAV], contribution_scores: Mapping[str, float], config: dict[str, Any],
) -> dict[str, TerminalRewardAllocation]:
    """Return terminal rewards and inspectable formula components."""

    profile = str(config.get("multi_terminal_reward_profile", "project_balanced"))
    if outcome.termination_reason == "ongoing":
        return {u.uav_id: TerminalRewardAllocation(0.0, profile, 0.0, 0.0) for u in red_aircraft}
    assumptions = config["project_assumptions"]["multi_terminal_reward"]
    if outcome.winner == "draw":
        value=float(assumptions["draw_reward"])
        return {u.uav_id: TerminalRewardAllocation(value, profile, value, 1.0, 1.0) for u in red_aircraft}
    weights = assumptions["win_weights"] if outcome.winner == "red" else assumptions["lose_weights"]
    if len(weights) != 3 or any(not np.isfinite(float(w)) or float(w) < 0.0 for w in weights):
        raise ValueError("Terminal weights must be three finite nonnegative values")
    if profile == "project_balanced":
        if abs(sum(float(w) for w in weights) - 1.0) > 1.0e-8:
            raise ValueError("project_balanced terminal weights must sum to one")
        return _project_balanced_allocations(outcome, red_aircraft, contribution_scores, config)
    if profile != "paper_2024_exact": raise ValueError(f"Unknown terminal reward profile: {profile}")
    n=len(red_aircraft); max_steps=int(config["max_decision_steps"]); remaining=(max_steps-outcome.decision_steps)/max_steps
    won=outcome.winner=="red"
    # Zheng, Wei and Duan (2024), equations (21) and (23).
    time_factor=(0.75+0.25*remaining) if won else (0.80+0.20*remaining)
    base=float(config["r_win0"] if won else config["r_lose0"])*n*time_factor
    health=[max(0.0,u.state.health) for u in red_aircraft]; b0=float(config["initial_health"]); alive_count=sum(u.is_alive for u in red_aircraft)
    beta=[max(0.0,float(contribution_scores.get(u.uav_id,0.0))) for u in red_aircraft]
    if won:
        # Equation (22): beta is set to one when the sum is zero.  B_r,sum
        # sums surviving red UAV health; a zero guard only handles impossible
        # or synthetic edge cases without changing the published expression.
        beta_sum=sum(beta)
        total_health=sum(value for value,u in zip(health,red_aircraft) if u.is_alive)
        contribution=[float(weights[1])*(value/beta_sum if beta_sum > 0.0 else 1.0/n) for value in beta]
        health_component=[float(weights[2])*(value/total_health)*(value/b0) if total_health>0.0 else 0.0 for value in health]
        base_share=[float(weights[0])/n for _ in red_aircraft]
        survival_component=[0.03*alive_count for _ in red_aircraft]
        contribution_denominator=beta_sum
        health_denominator=total_health
    else:
        # Equations (24)-(25): the contribution denominator is max(beta'),
        # not sum(beta'), and reverse health is divided directly by B0.
        beta_prime=[max(beta)-value+1.0 for value in beta]
        beta_prime_max=max(beta_prime)
        reverse_health=[b0-value+10.0 for value in health]
        contribution=[float(weights[1])*value/beta_prime_max for value in beta_prime]
        health_component=[float(weights[2])*value/b0 for value in reverse_health]
        base_share=[float(weights[0])/n for _ in red_aircraft]
        survival_component=[-0.02*alive_count for _ in red_aircraft]
        contribution_denominator=beta_prime_max
        health_denominator=b0
    result={}
    for index,u in enumerate(red_aircraft):
        factor=base_share[index]+survival_component[index]+contribution[index]+health_component[index]
        result[u.uav_id]=TerminalRewardAllocation(
            reward=base*factor, profile=profile, team_base=base, allocation_factor=factor,
            base_share_component=base_share[index], survival_component=survival_component[index],
            contribution_component=contribution[index], health_component=health_component[index],
            alive_count=alive_count, contribution_denominator=contribution_denominator,
            health_denominator=health_denominator,
        )
    return result


def _project_balanced_allocations(outcome: EpisodeOutcome, red_aircraft: Sequence[UAV], contribution_scores: Mapping[str,float], config: dict[str,Any]) -> dict[str,TerminalRewardAllocation]:
    assumptions=config["project_assumptions"]["multi_terminal_reward"]
    weights=assumptions["win_weights"] if outcome.winner=="red" else assumptions["lose_weights"]
    base = float(config["r_win0"] if outcome.winner == "red" else config["r_lose0"])
    total_health = sum(max(0.0, u.state.health) for u in red_aircraft)
    total_beta = sum(max(0.0, contribution_scores.get(u.uav_id, 0.0)) for u in red_aircraft)
    team_health_ratio = total_health / (len(red_aircraft) * float(config["initial_health"]))
    rewards: dict[str, TerminalRewardAllocation] = {}
    for aircraft in red_aircraft:
        own_health_ratio = max(0.0, aircraft.state.health) / float(config["initial_health"])
        beta_share = max(0.0, contribution_scores.get(aircraft.uav_id, 0.0)) / total_beta if total_beta > 0.0 else 1.0 / len(red_aircraft)
        if outcome.winner == "red":
            base_component = float(weights[0]) * team_health_ratio
            health_value = float(weights[1]) * own_health_ratio
            contribution_value = float(weights[2]) * beta_share
        else:
            base_component = float(weights[0]) * (1.0 - team_health_ratio)
            health_value = float(weights[1]) * (1.0 - own_health_ratio)
            contribution_value = float(weights[2]) * (1.0 - beta_share)
        factor = base_component + health_value + contribution_value
        rewards[aircraft.uav_id] = TerminalRewardAllocation(
            reward=base*factor, profile="project_balanced", team_base=base, allocation_factor=factor,
            base_share_component=base_component,
            contribution_component=contribution_value,
            health_component=health_value,
            alive_count=sum(u.is_alive for u in red_aircraft),
            contribution_denominator=total_beta,
            health_denominator=len(red_aircraft)*float(config["initial_health"]),
        )
    return rewards


def multi_reward(team: Sequence[UAVState], opponents: Sequence[UAVState]) -> float:
    """Legacy scalar helper intentionally replaced by per-agent environment rewards."""

    del team, opponents
    raise ValueError("Use MultiAgentRewardBreakdown from CombatMultiEnv")
