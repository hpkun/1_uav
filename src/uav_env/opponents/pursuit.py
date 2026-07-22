"""One-step geometric pursuit used for environment learnability checks."""

from __future__ import annotations

from dataclasses import replace
from math import pi

import numpy as np

from uav_env.actions.discrete_15 import DiscreteAction15, get_control
from uav_env.combat.attack_geometry import AttackZoneConfig, compute_combat_geometry
from uav_env.core.state import UAVState
from uav_env.dynamics.propagation import propagate_state
from uav_env.entities.type_profiles import UAVTypeProfile
from uav_env.opponents.base import RuleOpponent


class PursuitOpponent(RuleOpponent):
    """Explainable geometric rule; it is not the papers' predictive threat rule."""

    ANGLE_WEIGHT = 1.0
    DISTANCE_WEIGHT = 0.6
    ALTITUDE_WEIGHT = 0.2

    def __init__(
        self,
        profile: UAVTypeProfile,
        attack_config: AttackZoneConfig,
        physics_dt: float = 0.1,
        physics_steps: int = 5,
        gravity: float = 9.81,
        altitude_reference: float = 5000.0,
    ) -> None:
        self.profile = profile
        self.attack_config = attack_config
        self.physics_dt = physics_dt
        self.physics_steps = physics_steps
        self.gravity = gravity
        self.altitude_reference = altitude_reference

    def _predicted_target(self, target: UAVState) -> UAVState:
        duration = self.physics_dt * self.physics_steps
        position = target.position_vector() + duration * target.velocity_vector()
        return replace(target, x=float(position[0]), y=float(position[1]), z=float(position[2]))

    def _score(self, candidate: UAVState, predicted_target: UAVState) -> float:
        geometry = compute_combat_geometry(candidate, predicted_target, self.attack_config)
        if geometry.distance < self.attack_config.attack_distance_min:
            distance_error = self.attack_config.attack_distance_min - geometry.distance
        elif geometry.distance > self.attack_config.attack_distance_max:
            distance_error = geometry.distance - self.attack_config.attack_distance_max
        else:
            distance_error = 0.0
        distance_scale = max(self.attack_config.advantage_distance_max, 1.0)
        return (
            self.ANGLE_WEIGHT * geometry.attacker_attack_angle / pi
            + self.DISTANCE_WEIGHT * distance_error / distance_scale
            + self.ALTITUDE_WEIGHT * abs(candidate.z - predicted_target.z) / self.altitude_reference
        )

    def select_action(
        self,
        ownship: UAVState,
        opponent: UAVState,
        rng: np.random.Generator | None = None,
    ) -> DiscreteAction15:
        """Predict all 15 half-second actions and choose the minimum geometry cost."""

        del rng
        target = self._predicted_target(opponent)
        best_action = DiscreteAction15.LEVEL_HOLD
        best_score = float("inf")
        for action in DiscreteAction15:
            candidate = ownship.copy()
            for _ in range(self.physics_steps):
                candidate = propagate_state(candidate, get_control(action), self.profile, self.physics_dt, self.gravity)
            score = self._score(candidate, target)
            if score < best_score:
                best_action = action
                best_score = score
        return best_action
