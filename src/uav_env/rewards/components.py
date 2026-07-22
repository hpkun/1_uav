"""Independently testable 1v1 reward components."""

from __future__ import annotations

from math import pi

import numpy as np


def angle_reward(attack_angle: float, escape_angle: float) -> float:
    """Return the 2023 multiplicative angle reward."""

    attack = float(np.clip(attack_angle, 0.0, pi))
    escape = float(np.clip(escape_angle, 0.0, pi))
    return ((pi - attack) / pi) * ((pi - escape) / pi)


def approach_reward(previous_distance: float, distance: float, scale: float) -> float:
    """Return a bounded signed closing term used by the project convention."""

    if scale <= 0.0:
        raise ValueError("Approach scale must be positive")
    return float(np.clip((previous_distance - distance) / scale, -1.0, 1.0))


def piecewise_distance_reward(
    distance: float,
    attack_distance_min: float,
    attack_distance_max: float,
    advantage_distance_max: float,
    desired_distance_max: float,
) -> float:
    """Continuous proximity score over attack, advantage, and desired ranges."""

    if not 0.0 < attack_distance_min < attack_distance_max < advantage_distance_max < desired_distance_max:
        raise ValueError("Distance reward breakpoints must be strictly increasing")
    if distance <= 0.0:
        return 0.0
    if distance < attack_distance_min:
        return distance / attack_distance_min
    if distance <= attack_distance_max:
        return 1.0
    if distance <= advantage_distance_max:
        fraction = (distance - attack_distance_max) / (advantage_distance_max - attack_distance_max)
        return 1.0 - 0.5 * fraction
    if distance <= desired_distance_max:
        fraction = (distance - advantage_distance_max) / (desired_distance_max - advantage_distance_max)
        return 0.5 * (1.0 - fraction)
    return 0.0


def distance_reward(
    previous_distance: float,
    distance: float,
    attack_distance_min: float,
    attack_distance_max: float,
    advantage_distance_max: float,
    desired_distance_max: float,
) -> float:
    """Combine continuous range quality with a small approach incentive."""

    proximity = piecewise_distance_reward(
        distance,
        attack_distance_min,
        attack_distance_max,
        advantage_distance_max,
        desired_distance_max,
    )
    closing = approach_reward(previous_distance, distance, desired_distance_max)
    return float(np.clip(0.9 * proximity + 0.1 * closing, -1.0, 1.0))


def height_reward(
    altitude_advantage: float,
    h_max: float = 500.0,
    h_adv: float = 300.0,
    h_att: float = 100.0,
    h_min: float = -300.0,
) -> float:
    """Continuous trapezoid favoring a moderate positive height advantage."""

    if not h_min < h_att < h_adv < h_max:
        raise ValueError("Height reward breakpoints must be strictly increasing")
    if altitude_advantage <= h_min or altitude_advantage >= h_max:
        return 0.0
    if altitude_advantage < h_att:
        return (altitude_advantage - h_min) / (h_att - h_min)
    if altitude_advantage <= h_adv:
        return 1.0
    return (h_max - altitude_advantage) / (h_max - h_adv)


def speed_reward(speed_self: float, speed_enemy: float) -> float:
    """Return the specified piecewise reward for the speed ratio."""

    if speed_enemy <= 0.0:
        raise ValueError("Enemy speed must be positive")
    ratio = speed_self / speed_enemy
    if ratio > 1.5:
        return 0.1
    if ratio >= 1.0:
        return 1.0
    if ratio >= 0.8:
        return 5.0 * ratio - 4.0
    return 0.0


def dense_reward(angle: float, distance: float, height: float, speed: float) -> float:
    """Combine the four 2023 dense terms with their specified weights."""

    return (0.15 * angle + 0.60 * distance + 0.15 * height + 0.10 * speed - 1.0) * 0.05


def advantage_reward(
    distance: float,
    enemy_escape_angle: float,
    advantage_distance_min: float,
    advantage_distance_max: float,
) -> float:
    """Return the specified positive score while occupying the advantage area."""

    if advantage_distance_max <= advantage_distance_min:
        raise ValueError("Advantage-distance interval is invalid")
    distance_term = (advantage_distance_max - distance) / (advantage_distance_max - advantage_distance_min)
    angle_term = (pi - float(np.clip(enemy_escape_angle, 0.0, pi))) / pi
    return 0.6 * float(np.clip(distance_term, 0.0, 1.0)) + 0.4 * angle_term
