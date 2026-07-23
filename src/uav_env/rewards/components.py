"""Paper-calibrated and explicitly labeled debug reward components."""

from __future__ import annotations

from math import pi

import numpy as np


def angle_reward(attack_angle: float, escape_angle: float) -> float:
    """Return the 2023 multiplicative angle reward."""

    attack = float(np.clip(attack_angle, 0.0, pi))
    escape = float(np.clip(escape_angle, 0.0, pi))
    return ((pi - attack) / pi) * ((pi - escape) / pi)


def paper_distance_approach_reward(previous_distance: float, current_distance: float, d_mid: float) -> float:
    """Return 0.25 only while range decreases outside the attack midpoint."""

    distance_decrease = previous_distance - current_distance
    return 0.25 if distance_decrease > 0.0 and current_distance > d_mid else 0.0


def paper_piecewise_distance_reward(
    distance: float,
    attack_distance_min: float,
    attack_distance_max: float,
    advantage_distance_max: float,
    desired_distance_max: float,
) -> float:
    """Evaluate the exact published 2023 piecewise distance expression."""

    if not 0.0 < attack_distance_min < attack_distance_max < advantage_distance_max < desired_distance_max:
        raise ValueError("Distance reward breakpoints must be strictly increasing")
    d_mid = (attack_distance_min + attack_distance_max) / 2.0
    a1 = -1.0 / (desired_distance_max - advantage_distance_max) ** 2
    a2 = -1.0 / (advantage_distance_max - attack_distance_max) ** 2
    a3 = 1.0 / ((d_mid - attack_distance_min) * (d_mid - attack_distance_max))
    if advantage_distance_max < distance <= desired_distance_max:
        return 0.25 * (a1 * (distance - advantage_distance_max) ** 2 + 1.0)
    if attack_distance_max < distance <= advantage_distance_max:
        return 0.25 + 0.25 * (a2 * (distance - attack_distance_max) ** 2 + 1.0)
    if attack_distance_min < distance <= attack_distance_max:
        return 0.50 + 0.25 * a3 * (distance - attack_distance_min) * (distance - attack_distance_max)
    return 0.0


def paper_distance_reward(
    previous_distance: float,
    current_distance: float,
    attack_distance_min: float,
    attack_distance_max: float,
    advantage_distance_max: float,
    desired_distance_max: float,
) -> tuple[float, float, float]:
    """Return total, approach, and piecewise distance rewards."""

    d_mid = (attack_distance_min + attack_distance_max) / 2.0
    approach = paper_distance_approach_reward(previous_distance, current_distance, d_mid)
    piecewise = paper_piecewise_distance_reward(
        current_distance, attack_distance_min, attack_distance_max, advantage_distance_max, desired_distance_max
    )
    return approach + piecewise, approach, piecewise


def debug_linear_distance_reward(
    previous_distance: float,
    distance: float,
    attack_distance_min: float,
    attack_distance_max: float,
    advantage_distance_max: float,
    desired_distance_max: float,
) -> tuple[float, float, float]:
    """Retain the former continuous linear approximation for debugging only."""

    if distance <= 0.0:
        piecewise = 0.0
    elif distance < attack_distance_min:
        piecewise = distance / attack_distance_min
    elif distance <= attack_distance_max:
        piecewise = 1.0
    elif distance <= advantage_distance_max:
        piecewise = 1.0 - 0.5 * (distance - attack_distance_max) / (advantage_distance_max - attack_distance_max)
    elif distance <= desired_distance_max:
        piecewise = 0.5 * (1.0 - (distance - advantage_distance_max) / (desired_distance_max - advantage_distance_max))
    else:
        piecewise = 0.0
    approach = float(np.clip((previous_distance - distance) / desired_distance_max, -1.0, 1.0))
    total = float(np.clip(0.9 * piecewise + 0.1 * approach, -1.0, 1.0))
    return total, approach, piecewise


def paper_height_reward(
    delta_h: float,
    attack_distance_max: float,
    h_max: float = 500.0,
    h_adv: float = 300.0,
    h_att: float = 100.0,
    h_min: float = -300.0,
) -> float:
    """Evaluate the exact published 2023 height reward."""

    if not h_min < h_att < h_adv < h_max <= attack_distance_max:
        raise ValueError("Height reward breakpoints are invalid")
    h1 = -0.9 / (h_max - h_adv) ** 2
    h2 = -1.0 / (h_min - h_att) ** 2
    if h_max < delta_h <= attack_distance_max:
        return 0.1
    if h_adv < delta_h <= h_max:
        return h1 * (delta_h - h_adv) ** 2 + 1.0
    if h_att < delta_h <= h_adv:
        return 1.0
    if h_min < delta_h <= h_att:
        return h2 * (delta_h - h_att) ** 2 + 1.0
    return 0.0


def debug_linear_height_reward(
    delta_h: float,
    h_max: float = 500.0,
    h_adv: float = 300.0,
    h_att: float = 100.0,
    h_min: float = -300.0,
) -> float:
    """Retain the former continuous trapezoid for debugging only."""

    if delta_h <= h_min or delta_h >= h_max:
        return 0.0
    if delta_h < h_att:
        return (delta_h - h_min) / (h_att - h_min)
    if delta_h <= h_adv:
        return 1.0
    return (h_max - delta_h) / (h_max - h_adv)


def speed_reward(speed_self: float, speed_enemy: float) -> float:
    """Return the published piecewise speed-ratio reward."""

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
    """Combine the four 2023 terms with published weights."""

    return (0.15 * angle + 0.60 * distance + 0.15 * height + 0.10 * speed - 1.0) * 0.05


def advantage_reward(distance: float, enemy_escape_angle: float, minimum: float, maximum: float) -> float:
    """Return the published positive advantage-area score."""

    if maximum <= minimum:
        raise ValueError("Advantage-distance interval is invalid")
    distance_term = (maximum - distance) / (maximum - minimum)
    angle_term = (pi - float(np.clip(enemy_escape_angle, 0.0, pi))) / pi
    return 0.6 * float(np.clip(distance_term, 0.0, 1.0)) + 0.4 * angle_term


def piecewise_distance_reward(distance: float, attack_distance_min: float, attack_distance_max: float, advantage_distance_max: float, desired_distance_max: float) -> float:
    """Backward-compatible alias for the debug linear piecewise component."""

    return debug_linear_distance_reward(distance, distance, attack_distance_min, attack_distance_max, advantage_distance_max, desired_distance_max)[2]


def height_reward(
    delta_h: float,
    h_max: float = 500.0,
    h_adv: float = 300.0,
    h_att: float = 100.0,
    h_min: float = -300.0,
) -> float:
    """Backward-compatible wrapper for the debug linear height term."""

    return debug_linear_height_reward(delta_h, h_max, h_adv, h_att, h_min)
