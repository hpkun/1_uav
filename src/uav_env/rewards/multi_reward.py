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
    terminal_profile: str = "none"
    terminal_team_base: float = 0.0
    terminal_allocation_factor: float = 0.0
    terminal_health_component: float = 0.0
    terminal_contribution_component: float = 0.0
    terminal_survival_component: float = 0.0


@dataclass(frozen=True)
class TerminalRewardAllocation:
    reward: float
    profile: str
    team_base: float
    allocation_factor: float
    health_component: float
    contribution_component: float
    survival_component: float


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

    profile = str(config.get("multi_terminal_reward_profile", "paper_2024_exact"))
    if outcome.termination_reason == "ongoing":
        return {u.uav_id: TerminalRewardAllocation(0.0, profile, 0.0, 0.0, 0.0, 0.0, 0.0) for u in red_aircraft}
    assumptions = config["project_assumptions"]["multi_terminal_reward"]
    if outcome.winner == "draw":
        value=float(assumptions["draw_reward"])
        return {u.uav_id: TerminalRewardAllocation(value, profile, value, 1.0, 0.0, 0.0, 0.0) for u in red_aircraft}
    weights = assumptions["win_weights"] if outcome.winner == "red" else assumptions["lose_weights"]
    if len(weights)!=3 or any(float(w)<0 for w in weights) or abs(sum(float(w) for w in weights)-1.0)>1e-8:
        raise ValueError("Terminal weights must be three nonnegative values summing to one")
    if profile == "project_balanced":
        legacy=dict(config); legacy["multi_terminal_reward_profile"]="_legacy"
        return _project_balanced_allocations(outcome,red_aircraft,contribution_scores,legacy)
    if profile != "paper_2024_exact": raise ValueError(f"Unknown terminal reward profile: {profile}")
    n=len(red_aircraft); max_steps=int(config["max_decision_steps"]); remaining=(max_steps-outcome.decision_steps)/max_steps
    won=outcome.winner=="red"; base=float(config["r_win0"] if won else config["r_lose0"])*n*((1.0+remaining) if won else (0.8+0.2*remaining))
    health=[max(0.0,u.state.health) for u in red_aircraft]; b0=float(config["initial_health"]); alive_count=sum(u.is_alive for u in red_aircraft)
    beta=[max(0.0,float(contribution_scores.get(u.uav_id,0.0))) for u in red_aircraft]
    if won:
        beta_sum=sum(beta); beta_share=[v/beta_sum if beta_sum>0 else 1/n for v in beta]
        total_health=sum(health); health_share=[(v/total_health if total_health>0 else 1/n)*(total_health/(n*b0) if total_health>0 else 0.0) for v in health]
        survival=[float(u.is_alive)/max(alive_count,1) for u in red_aircraft]
    else:
        beta_prime=[max(beta)-v+1.0 for v in beta]; beta_sum=sum(beta_prime); beta_share=[v/beta_sum for v in beta_prime]
        reverse_health=[b0-v+10.0 for v in health]; total_reverse=sum(reverse_health); health_share=[v/total_reverse for v in reverse_health]
        failed=max(n-alive_count,1); survival=[float(not u.is_alive)/failed for u in red_aircraft]
    result={}
    for index,u in enumerate(red_aircraft):
        survival_component=float(weights[0])*survival[index]; contribution_component=float(weights[1])*beta_share[index]; health_component=float(weights[2])*health_share[index]
        factor=survival_component+contribution_component+health_component
        result[u.uav_id]=TerminalRewardAllocation(base*factor,profile,base,factor,health_component,contribution_component,survival_component)
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
            factor = float(weights[0]) * team_health_ratio + float(weights[1]) * own_health_ratio + float(weights[2]) * beta_share
        else:
            factor = float(weights[0]) * (1.0 - team_health_ratio) + float(weights[1]) * (1.0 - own_health_ratio) + float(weights[2]) * (1.0 - beta_share)
        rewards[aircraft.uav_id] = TerminalRewardAllocation(base*factor,"project_balanced",base,factor,float(weights[1])*own_health_ratio,float(weights[2])*beta_share,float(weights[0])*team_health_ratio)
    return rewards


def multi_reward(team: Sequence[UAVState], opponents: Sequence[UAVState]) -> float:
    """Legacy scalar helper intentionally replaced by per-agent environment rewards."""

    del team, opponents
    raise ValueError("Use MultiAgentRewardBreakdown from CombatMultiEnv")
