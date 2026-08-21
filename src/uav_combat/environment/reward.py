"""Stage-based rewards serving only directed combat progress."""
from __future__ import annotations

import numpy as np

from ..models import AircraftState
from .geometry import engagement_geometry
from .weapon import WeaponEnvelope


def relation_score(attacker: AircraftState, target: AircraftState, config: dict) -> float:
    """Bounded continuous angular advantage inside tactical range."""
    geometry = engagement_geometry(attacker, target)
    tactical_range = float(config["tactical_range"])
    range_floor = float(config["range_floor"])
    range_score = float(np.clip(
        (tactical_range - geometry.distance) / (tactical_range - range_floor),
        0.0,
        1.0,
    ))
    attack_score = 0.5 * (1.0 + np.cos(geometry.attack_angle))
    aspect_score = 0.5 * (1.0 + np.cos(geometry.target_aspect))
    return float(range_score * (0.6 * attack_score + 0.4 * aspect_score))


def _best_relation(
    attacker: AircraftState, targets: list[AircraftState], config: dict
) -> float:
    return max(
        (relation_score(attacker, target, config) for target in targets if target.alive),
        default=0.0,
    )


def combat_reward_components(
    current_team: list[AircraftState],
    current_opponents: list[AircraftState],
    next_team: list[AircraftState],
    next_opponents: list[AircraftState],
    weapon: WeaponEnvelope,
    config: dict,
    dt: float,
) -> dict[str, np.ndarray]:
    """Return progress, tactical and fire-opportunity vectors."""
    progress = np.zeros(len(current_team), dtype=np.float32)
    tactical = np.zeros(len(current_team), dtype=np.float32)
    fire = np.zeros(len(current_team), dtype=np.float32)
    for index, (own, next_own) in enumerate(zip(current_team, next_team)):
        if not own.alive or not next_own.alive:
            continue

        candidates = [
            (engagement_geometry(own, target).distance, target_index)
            for target_index, target in enumerate(current_opponents) if target.alive
        ]
        if candidates:
            distance, target_index = min(candidates)
            next_target = next_opponents[target_index]
            if next_target.alive and distance > weapon.range_max:
                next_distance = engagement_geometry(next_own, next_target).distance
                normalized_closure = (distance - next_distance) / (
                    float(config["closing_speed_scale"]) * dt
                )
                progress[index] = float(config["progress_weight"]) * np.clip(
                    normalized_closure, -1.0, 1.0
                )

        attack = _best_relation(next_own, next_opponents, config)
        threat = max(
            (relation_score(target, next_own, config)
             for target in next_opponents if target.alive),
            default=0.0,
        )
        tactical[index] = float(config["tactical_weight"]) * (attack - threat)

        own_window = any(
            target.alive and weapon.in_fire_window(
                engagement_geometry(next_own, target)
            ) for target in next_opponents
        )
        threat_window = any(
            target.alive and weapon.in_fire_window(
                engagement_geometry(target, next_own)
            ) for target in next_opponents
        )
        fire[index] = (
            float(config["fire_opportunity_reward"]) * own_window
            - float(config["threat_opportunity_penalty"]) * threat_window
        )
    return {"progress": progress, "tactical": tactical, "fire": fire}


__all__ = ["combat_reward_components", "relation_score"]
