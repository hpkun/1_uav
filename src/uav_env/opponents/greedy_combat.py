"""Project-defined one-step greedy combat opponent for fixed 3v3 blue UAVs."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi
from typing import Any

import numpy as np

from uav_env.actions.discrete_15 import DiscreteAction15, get_control
from uav_env.combat.attack_geometry import AttackZoneConfig, CombatGeometry, compute_combat_geometry
from uav_env.core.state import UAVState
from uav_env.dynamics.propagation import propagate_state
from uav_env.entities.type_profiles import UAVTypeProfile
from uav_env.opponents.base import RuleOpponent


NON_TURNING_ACTIONS = {
    DiscreteAction15.LEVEL_HOLD,
    DiscreteAction15.LEVEL_ACCELERATE,
    DiscreteAction15.LEVEL_DECELERATE,
    DiscreteAction15.CLIMB_HOLD,
    DiscreteAction15.CLIMB_ACCELERATE,
    DiscreteAction15.CLIMB_DECELERATE,
    DiscreteAction15.DIVE_HOLD,
    DiscreteAction15.DIVE_ACCELERATE,
    DiscreteAction15.DIVE_DECELERATE,
}

CLIMB_ACTIONS = {
    DiscreteAction15.CLIMB_HOLD,
    DiscreteAction15.CLIMB_ACCELERATE,
    DiscreteAction15.CLIMB_DECELERATE,
}

DIVE_ACTIONS = {
    DiscreteAction15.DIVE_HOLD,
    DiscreteAction15.DIVE_ACCELERATE,
    DiscreteAction15.DIVE_DECELERATE,
}


@dataclass(frozen=True)
class GreedyCombatConfig:
    """Small set of deterministic one-step greedy scoring weights."""

    offense_weight: float = 1.0
    defense_weight: float = 0.7
    angle_score_weight: float = 0.6
    distance_score_weight: float = 0.4
    attack_area_bonus: float = 0.5
    advantage_area_bonus: float = 0.25
    can_attack_bonus: float = 2.0
    incoming_attack_area_penalty: float = 0.75
    incoming_advantage_area_penalty: float = 0.25
    incoming_can_attack_penalty: float = 2.0
    minimum_safe_altitude: float = 300.0
    ceiling_margin: float = 300.0

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "GreedyCombatConfig":
        """Construct config from a YAML mapping and reject unknown knobs."""

        expected = set(cls.__dataclass_fields__)
        extra = set(values) - expected
        missing = expected - set(values)
        if extra or missing:
            raise ValueError(f"Invalid greedy_combat config keys: missing={sorted(missing)}, extra={sorted(extra)}")
        return cls(**{key: float(values[key]) for key in expected})


@dataclass(frozen=True)
class CandidateEvaluation:
    """One candidate blue action after fixed-horizon prediction."""

    action: DiscreteAction15
    predicted_blue: UAVState
    offensive_score: float
    incoming_threat: float
    safety_penalty: float
    score: float
    unsafe: bool


class GreedyCombatOpponent(RuleOpponent):
    """Project-defined one-step greedy combat rule.

    It is not an exact reproduction of the paper's predictive-threat
    Algorithm 1. It enumerates blue's 15 discrete actions for one decision
    period, predicts the assigned red target by holding its last action, and
    chooses a deterministic attack/defense geometry score.
    """

    def __init__(
        self,
        profile: UAVTypeProfile,
        attack_config: AttackZoneConfig,
        physics_dt: float = 0.1,
        physics_steps: int = 5,
        gravity: float = 9.81,
        min_altitude: float = 0.0,
        max_altitude: float = 5000.0,
        config: GreedyCombatConfig | dict[str, Any] | None = None,
    ) -> None:
        self.profile = profile
        self.attack_config = attack_config
        self.physics_dt = float(physics_dt)
        self.physics_steps = int(physics_steps)
        self.gravity = float(gravity)
        self.min_altitude = float(min_altitude)
        self.max_altitude = float(max_altitude)
        if self.physics_dt <= 0.0 or self.physics_steps <= 0:
            raise ValueError("GreedyCombatOpponent requires positive physics timing")
        if not self.min_altitude < self.max_altitude:
            raise ValueError("Altitude bounds are invalid")
        self.greedy_config = (
            GreedyCombatConfig()
            if config is None
            else config
            if isinstance(config, GreedyCombatConfig)
            else GreedyCombatConfig.from_mapping(config)
        )
        self._validate_config()

    def _validate_config(self) -> None:
        values = self.greedy_config.__dict__
        if not all(isfinite(float(value)) for value in values.values()):
            raise ValueError("greedy_combat values must be finite")
        weight_keys = tuple(key for key in values if key not in {"minimum_safe_altitude", "ceiling_margin"})
        if any(float(values[key]) < 0.0 for key in weight_keys):
            raise ValueError("greedy_combat weights must be nonnegative")
        if self.greedy_config.minimum_safe_altitude < self.min_altitude:
            raise ValueError("minimum_safe_altitude must be >= min_altitude")
        if self.greedy_config.ceiling_margin < 0.0:
            raise ValueError("ceiling_margin must be nonnegative")
        if not self.greedy_config.minimum_safe_altitude < self.max_altitude - self.greedy_config.ceiling_margin:
            raise ValueError("minimum_safe_altitude must be below max_altitude - ceiling_margin")

    def _coerce_last_action(self, state: UAVState) -> DiscreteAction15:
        try:
            return DiscreteAction15(state.last_action)
        except (TypeError, ValueError):
            return DiscreteAction15.LEVEL_HOLD

    def _predict_state(self, state: UAVState, action: DiscreteAction15) -> UAVState:
        """Propagate exactly one decision period using the environment dynamics."""

        predicted = state.copy()
        for _ in range(self.physics_steps):
            predicted = propagate_state(predicted, get_control(action), self.profile, self.physics_dt, self.gravity)
        return predicted

    def _predict_target(self, target: UAVState) -> UAVState:
        """Predict the target by holding its last valid discrete action."""

        return self._predict_state(target, self._coerce_last_action(target))

    def _distance_quality(self, distance: float) -> float:
        if self.attack_config.attack_distance_min <= distance <= self.attack_config.attack_distance_max:
            return 1.0
        if distance < self.attack_config.attack_distance_min:
            error = self.attack_config.attack_distance_min - distance
        else:
            error = distance - self.attack_config.attack_distance_max
        scale = max(float(self.attack_config.advantage_distance_max), 1.0)
        return float(np.clip(1.0 - error / scale, 0.0, 1.0))

    def _angle_quality(self, attack_angle: float) -> float:
        return float(np.clip(1.0 - attack_angle / pi, 0.0, 1.0))

    def _offensive_score(self, geometry: CombatGeometry) -> float:
        cfg = self.greedy_config
        value = (
            cfg.angle_score_weight * self._angle_quality(geometry.attacker_attack_angle)
            + cfg.distance_score_weight * self._distance_quality(geometry.distance)
            + cfg.attack_area_bonus * int(geometry.in_attack_area)
            + cfg.advantage_area_bonus * int(geometry.in_advantage_area)
            + cfg.can_attack_bonus * int(geometry.can_attack)
        )
        if not isfinite(value):
            raise ValueError("offensive score must be finite")
        return float(value)

    def _incoming_threat(self, geometry: CombatGeometry) -> float:
        cfg = self.greedy_config
        value = (
            cfg.angle_score_weight * self._angle_quality(geometry.attacker_attack_angle)
            + cfg.distance_score_weight * self._distance_quality(geometry.distance)
            + cfg.incoming_attack_area_penalty * int(geometry.in_attack_area)
            + cfg.incoming_advantage_area_penalty * int(geometry.in_advantage_area)
            + cfg.incoming_can_attack_penalty * int(geometry.can_attack)
        )
        if not isfinite(value):
            raise ValueError("incoming threat must be finite")
        return float(value)

    def _is_unsafe(self, candidate: UAVState) -> bool:
        vector = candidate.to_kinematic_vector()
        return (
            not np.isfinite(vector).all()
            or candidate.z < self.greedy_config.minimum_safe_altitude
            or candidate.z > self.max_altitude - self.greedy_config.ceiling_margin
        )

    def _evaluate_action(
        self,
        ownship: UAVState,
        predicted_target: UAVState,
        action: DiscreteAction15,
    ) -> CandidateEvaluation:
        predicted_blue = self._predict_state(ownship, action)
        unsafe = self._is_unsafe(predicted_blue)
        offensive_geometry = compute_combat_geometry(predicted_blue, predicted_target, self.attack_config)
        incoming_geometry = compute_combat_geometry(predicted_target, predicted_blue, self.attack_config)
        offensive = self._offensive_score(offensive_geometry)
        incoming = self._incoming_threat(incoming_geometry)
        safety_penalty = 0.0 if not unsafe else float("inf")
        score = self.greedy_config.offense_weight * offensive - self.greedy_config.defense_weight * incoming - safety_penalty
        return CandidateEvaluation(action, predicted_blue, offensive, incoming, safety_penalty, float(score), unsafe)

    def _evaluate_all(self, ownship: UAVState, predicted_target: UAVState) -> list[CandidateEvaluation]:
        return [self._evaluate_action(ownship, predicted_target, action) for action in DiscreteAction15]

    def _tie_key(self, item: CandidateEvaluation) -> tuple[float, int, int]:
        non_turning_priority = 1 if item.action in NON_TURNING_ACTIONS else 0
        return (item.score, non_turning_priority, -int(item.action))

    def _fallback_action(self, ownship: UAVState, evaluations: list[CandidateEvaluation]) -> DiscreteAction15:
        finite = [item for item in evaluations if np.isfinite(item.predicted_blue.to_kinematic_vector()).all()]
        if ownship.z <= self.greedy_config.minimum_safe_altitude:
            climbs = [item for item in finite if item.action in CLIMB_ACTIONS]
            if climbs:
                return max(climbs, key=lambda item: (item.predicted_blue.z, -int(item.action))).action
        if ownship.z >= self.max_altitude - self.greedy_config.ceiling_margin:
            dives = [item for item in finite if item.action in DIVE_ACTIONS]
            if dives:
                return min(dives, key=lambda item: (item.predicted_blue.z, int(item.action))).action
        return DiscreteAction15.LEVEL_HOLD

    def select_action(
        self,
        ownship: UAVState,
        opponent: UAVState,
        rng: np.random.Generator | None = None,
    ) -> DiscreteAction15:
        """Return the deterministic one-step greedy action for the assigned target."""

        del rng
        predicted_target = self._predict_target(opponent)
        evaluations = self._evaluate_all(ownship, predicted_target)
        safe = [item for item in evaluations if not item.unsafe]
        if safe:
            return max(safe, key=self._tie_key).action
        return self._fallback_action(ownship, evaluations)
