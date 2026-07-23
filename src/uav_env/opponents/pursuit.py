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

    def __init__(
        self,
        profile: UAVTypeProfile,
        attack_config: AttackZoneConfig,
        physics_dt: float = 0.1,
        physics_steps: int = 5,
        gravity: float = 9.81,
        altitude_reference: float = 5000.0,
        angle_weight: float = 1.0,
        distance_weight: float = 0.6,
        altitude_weight: float = 0.2,
        boundary_penalty: float = 100.0,
        minimum_safe_altitude: float = 300.0,
        ceiling_margin: float = 300.0,
        unsafe_flight_path_penalty: float = 2.0,
    ) -> None:
        self.profile = profile
        self.attack_config = attack_config
        self.physics_dt = physics_dt
        self.physics_steps = physics_steps
        self.gravity = gravity
        self.altitude_reference = altitude_reference
        self.angle_weight = angle_weight
        self.distance_weight = distance_weight
        self.altitude_weight = altitude_weight
        self.boundary_penalty = boundary_penalty
        self.minimum_safe_altitude = minimum_safe_altitude
        self.ceiling_margin = ceiling_margin
        self.unsafe_flight_path_penalty = unsafe_flight_path_penalty

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
            self.angle_weight * geometry.attacker_attack_angle / pi
            + self.distance_weight * distance_error / distance_scale
            + self.altitude_weight * abs(candidate.z - predicted_target.z) / self.altitude_reference
        )

    def _predict_candidate(self, ownship: UAVState, action: DiscreteAction15) -> tuple[UAVState, bool, float]:
        candidate = ownship.copy()
        unsafe = False
        penalty = 0.0
        for _ in range(self.physics_steps):
            candidate = propagate_state(candidate, get_control(action), self.profile, self.physics_dt, self.gravity)
            if candidate.z < self.minimum_safe_altitude or candidate.z > self.altitude_reference - self.ceiling_margin:
                unsafe = True
                penalty += self.boundary_penalty
        if ownship.z <= self.minimum_safe_altitude and candidate.flight_path_angle < 0.0:
            penalty += self.unsafe_flight_path_penalty
        if ownship.z >= self.altitude_reference - self.ceiling_margin and candidate.flight_path_angle > 0.0:
            penalty += self.unsafe_flight_path_penalty
        return candidate, unsafe, penalty

    def _predict_all(self, ownship: UAVState) -> list[tuple[UAVState, bool, float]]:
        """Vectorize all 15 RK4 predictions to keep matrix evaluation practical."""

        vectors = np.repeat(ownship.to_kinematic_vector()[None, :], len(DiscreteAction15), axis=0)
        controls = np.asarray([get_control(action).to_vector() for action in DiscreteAction15], dtype=np.float64)
        unsafe = np.zeros(len(DiscreteAction15), dtype=bool)
        penalties = np.zeros(len(DiscreteAction15), dtype=np.float64)

        def derivative(states: np.ndarray) -> np.ndarray:
            speed, theta, heading = states[:, 3], states[:, 4], states[:, 5]
            cos_theta = np.cos(theta)
            safe_speed = np.where(np.abs(speed) >= 1.0e-8, speed, np.where(speed < 0.0, -1.0e-8, 1.0e-8))
            safe_cos = np.where(np.abs(cos_theta) >= 1.0e-8, cos_theta, np.where(cos_theta < 0.0, -1.0e-8, 1.0e-8))
            nx, nz, bank = controls[:, 0], controls[:, 1], controls[:, 2]
            return np.column_stack([
                speed * cos_theta * np.cos(heading), speed * cos_theta * np.sin(heading), speed * np.sin(theta),
                self.gravity * (nx - np.sin(theta)), self.gravity / safe_speed * (nz * np.cos(bank) - cos_theta),
                self.gravity * nz * np.sin(bank) / (safe_speed * safe_cos),
            ])

        dt = self.physics_dt
        for _ in range(self.physics_steps):
            k1 = derivative(vectors)
            k2 = derivative(vectors + dt * k1 / 2.0)
            k3 = derivative(vectors + dt * k2 / 2.0)
            k4 = derivative(vectors + dt * k3)
            vectors += dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            vectors[:, 3] = np.clip(vectors[:, 3], self.profile.min_speed, self.profile.max_speed)
            vectors[:, 4] = np.clip(vectors[:, 4], self.profile.min_flight_path_angle, self.profile.max_flight_path_angle)
            vectors[:, 5] %= 2.0 * pi
            violation = (vectors[:, 2] < self.minimum_safe_altitude) | (vectors[:, 2] > self.altitude_reference - self.ceiling_margin)
            unsafe |= violation
            penalties += violation * self.boundary_penalty
        penalties += ((ownship.z <= self.minimum_safe_altitude) & (vectors[:, 4] < 0.0)) * self.unsafe_flight_path_penalty
        penalties += ((ownship.z >= self.altitude_reference - self.ceiling_margin) & (vectors[:, 4] > 0.0)) * self.unsafe_flight_path_penalty
        return [(ownship.with_kinematic_vector(vectors[index]), bool(unsafe[index]), float(penalties[index])) for index in range(len(DiscreteAction15))]

    def select_action(
        self,
        ownship: UAVState,
        opponent: UAVState,
        rng: np.random.Generator | None = None,
    ) -> DiscreteAction15:
        """Predict all 15 half-second actions and choose the minimum geometry cost."""

        del rng
        target = self._predicted_target(opponent)
        if ownship.z <= self.minimum_safe_altitude:
            return DiscreteAction15.CLIMB_DECELERATE
        if ownship.z >= self.altitude_reference - self.ceiling_margin:
            return DiscreteAction15.DIVE_DECELERATE
        scored: list[tuple[float, DiscreteAction15, UAVState, bool]] = []
        for action, (candidate, unsafe, penalty) in zip(DiscreteAction15, self._predict_all(ownship)):
            scored.append((self._score(candidate, target) + penalty, action, candidate, unsafe))
        relative = target.position_vector()[:2] - ownship.position_vector()[:2]
        velocity = ownship.velocity_vector()[:2]
        scale = max(float(np.linalg.norm(relative) * np.linalg.norm(velocity)), 1.0)
        collinear = abs(float(velocity[0] * relative[1] - velocity[1] * relative[0])) / scale <= 1.0e-12
        safe = [entry for entry in scored if not entry[3]]
        if collinear:
            # At exact left/right degeneracy there is no reflection-equivariant
            # choice between a chiral pair. Prefer a non-turning maneuver.
            nonturning = [entry for entry in safe if int(entry[1]) <= int(DiscreteAction15.DIVE_DECELERATE)]
            if nonturning:
                safe = nonturning
        if safe:
            return min(safe, key=lambda entry: (entry[0], int(entry[1])))[1]
        if min(entry[2].z for entry in scored) < self.minimum_safe_altitude:
            recovery = [entry for entry in scored if entry[1] in {DiscreteAction15.CLIMB_HOLD, DiscreteAction15.CLIMB_ACCELERATE, DiscreteAction15.CLIMB_DECELERATE}]
        elif max(entry[2].z for entry in scored) > self.altitude_reference - self.ceiling_margin:
            recovery = [entry for entry in scored if entry[1] in {DiscreteAction15.DIVE_HOLD, DiscreteAction15.DIVE_ACCELERATE, DiscreteAction15.DIVE_DECELERATE}]
        else:
            recovery = []
        candidates = recovery or scored
        return min(candidates, key=lambda entry: (entry[0], int(entry[1])))[1]
