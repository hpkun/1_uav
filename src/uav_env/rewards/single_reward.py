"""Explainable red-agent reward assembly for homogeneous 1v1 combat."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from uav_env.combat.attack_geometry import CombatGeometry
from uav_env.combat.damage import DamageResult
from uav_env.combat.events import EpisodeOutcome
from uav_env.core.state import UAVState
from uav_env.rewards.components import (
    advantage_reward,
    angle_reward,
    dense_reward,
    distance_reward,
    height_reward,
    speed_reward,
)


@dataclass(frozen=True)
class RewardBreakdown:
    """Primitive diagnostics and additive reward contributions."""

    angle: float = 0.0
    distance: float = 0.0
    height: float = 0.0
    speed: float = 0.0
    dense: float = 0.0
    advantage: float = 0.0
    attack_area: float = 0.0
    hit: float = 0.0
    destroy: float = 0.0
    being_hit: float = 0.0
    being_destroyed: float = 0.0
    boundary: float = 0.0
    terminal: float = 0.0
    total: float = 0.0

    def additive_total(self) -> float:
        """Sum fields that contribute directly to the scalar reward."""

        return (
            self.dense
            + self.advantage
            + self.attack_area
            + self.hit
            + self.destroy
            + self.being_hit
            + self.being_destroyed
            + self.boundary
            + self.terminal
        )


def terminal_reward(
    outcome: EpisodeOutcome | None,
    current_step: int,
    max_steps: int,
    remaining_health: float,
    initial_health: float,
    draw_as_loss: bool,
) -> float:
    """Return the 2023 terminal reward from the red-agent viewpoint."""

    if outcome is None or outcome.termination_reason == "ongoing":
        return 0.0
    if outcome.winner == "red":
        return 5.0 + 3.0 * (max_steps - current_step) / max_steps + 6.0 * remaining_health / initial_health
    if outcome.winner == "draw" and not draw_as_loss:
        return 0.0
    return -15.0


def compute_reward_breakdown(
    previous_red_to_blue: CombatGeometry,
    red_to_blue: CombatGeometry,
    previous_blue_to_red: CombatGeometry,
    blue_to_red: CombatGeometry,
    red_state: UAVState,
    blue_state: UAVState,
    damage_to_blue: DamageResult,
    damage_to_red: DamageResult,
    boundary_violation: bool,
    outcome: EpisodeOutcome | None,
    current_step: int,
    config: dict[str, Any],
) -> RewardBreakdown:
    """Compute the scalar reward and preserve every interpretable component."""

    angle = angle_reward(red_to_blue.attacker_attack_angle, red_to_blue.target_escape_angle)
    distance = distance_reward(
        previous_red_to_blue.distance,
        red_to_blue.distance,
        float(config["attack_distance_min"]),
        float(config["attack_distance_max"]),
        float(config["advantage_distance_max"]),
        float(config["desired_distance_max"]),
    )
    h_config = config["height_reward"]
    height = height_reward(
        red_state.z - blue_state.z,
        float(h_config["H_max"]),
        float(h_config["H_adv"]),
        float(h_config["H_att"]),
        float(h_config["H_min"]),
    )
    speed = speed_reward(red_state.speed, blue_state.speed)
    dense = dense_reward(angle, distance, height, speed)
    advantage = 0.0
    if red_to_blue.in_advantage_area:
        advantage += advantage_reward(
            red_to_blue.distance,
            red_to_blue.target_escape_angle,
            float(config["advantage_distance_min"]),
            float(config["advantage_distance_max"]),
        )
    if blue_to_red.in_advantage_area:
        advantage -= 1.0
    mode = str(config.get("event_reward_mode", "per_step"))
    if mode not in {"per_step", "on_enter"}:
        raise ValueError("event_reward_mode must be 'per_step' or 'on_enter'")
    own_attack = red_to_blue.in_attack_area and (mode == "per_step" or not previous_red_to_blue.in_attack_area)
    enemy_attack = blue_to_red.in_attack_area and (mode == "per_step" or not previous_blue_to_red.in_attack_area)
    attack_area = (0.3 if own_attack else 0.0) - (0.3 if enemy_attack else 0.0)
    hit = 0.8 if damage_to_blue.damage > 0.0 else 0.0
    destroy = 1.5 if damage_to_blue.destroyed else 0.0
    being_hit = -0.9 if damage_to_red.damage > 0.0 else 0.0
    being_destroyed = -1.6 if damage_to_red.destroyed else 0.0
    boundary = -0.5 if boundary_violation else 0.0
    terminal = terminal_reward(
        outcome,
        current_step,
        int(config["max_decision_steps"]),
        red_state.health,
        float(config["initial_health"]),
        bool(config.get("draw_as_loss", True)),
    )
    result = RewardBreakdown(
        angle=angle,
        distance=distance,
        height=height,
        speed=speed,
        dense=dense,
        advantage=advantage,
        attack_area=attack_area,
        hit=hit,
        destroy=destroy,
        being_hit=being_hit,
        being_destroyed=being_destroyed,
        boundary=boundary,
        terminal=terminal,
    )
    return RewardBreakdown(**{**result.__dict__, "total": result.additive_total()})


def single_reward(*args: Any, **kwargs: Any) -> RewardBreakdown:
    """Backward-compatible explicit wrapper around reward assembly."""

    return compute_reward_breakdown(*args, **kwargs)
