"""Runnable homogeneous UAV 1v1 Gymnasium environment."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from math import pi
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from uav_env.actions.discrete_15 import DiscreteAction15, get_control
from uav_env.combat.attack_geometry import AttackZoneConfig, CombatGeometry, compute_combat_geometry
from uav_env.combat.collision import has_collision
from uav_env.combat.damage import DamageConfig, DamageResult, sample_damage
from uav_env.combat.events import CombatEvent, EpisodeOutcome
from uav_env.core.enums import CombatEventType, Team
from uav_env.core.state import UAVState
from uav_env.dynamics.propagation import clip_control, propagate_state
from uav_env.entities.type_profiles import UAVTypeProfile, profile_from_config
from uav_env.entities.uav import UAV
from uav_env.envs.base_env import BaseUAVEnv
from uav_env.observations.normalization import NormalizationConfig
from uav_env.observations.single_observation import (
    actor_observation_raw_1v1,
    build_actor_observation_1v1,
    build_critic_state_1v1,
)
from uav_env.opponents.base import RuleOpponent
from uav_env.opponents.pursuit import PursuitOpponent
from uav_env.opponents.random import RandomOpponent
from uav_env.opponents.straight import StraightOpponent
from uav_env.rewards.single_reward import RewardBreakdown, compute_reward_breakdown
from uav_env.utils.config import validate_experiment_config


class Combat1v1Env(BaseUAVEnv):
    """Red external agent versus one reproducible blue rule policy."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: dict[str, Any],
        scenario_name: str | None = None,
        opponent: str | RuleOpponent = "straight",
        seed: int | None = None,
    ) -> None:
        validate_experiment_config(config)
        self.config = deepcopy(config)
        self.scenario_name = scenario_name or str(config["scenario_name"])
        self.profile: UAVTypeProfile = profile_from_config(config)
        self.attack_config = AttackZoneConfig.from_config(config)
        self.damage_config = DamageConfig.from_config(config)
        self.normalization_config = NormalizationConfig.from_config(config)
        self.action_space = spaces.Discrete(15)
        if self.normalization_config.clip_observation:
            self.observation_space = spaces.Box(-1.0, 1.0, shape=(11,), dtype=np.float64)
        else:
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(11,), dtype=np.float64)
        self._initial_seed = seed
        self._current_seed: int | None = None
        self._has_reset = False
        self.rng = np.random.default_rng(seed)
        self.opponent_policy = self._build_opponent(opponent)
        self.red: UAV
        self.blue: UAV
        self.decision_step = 0
        self.simulation_time = 0.0
        self._trajectory: list[dict[str, Any]] = []
        self._statistics: dict[str, float | int] = {}
        self._previous_red_geometry: CombatGeometry | None = None
        self._previous_blue_geometry: CombatGeometry | None = None

    def _build_opponent(self, opponent: str | RuleOpponent) -> RuleOpponent:
        if isinstance(opponent, RuleOpponent):
            return opponent
        if opponent == "straight":
            return StraightOpponent()
        if opponent == "random":
            return RandomOpponent()
        if opponent == "pursuit":
            return PursuitOpponent(
                self.profile,
                self.attack_config,
                float(self.config["physics_dt"]),
                int(self.config["physics_steps_per_action"]),
                float(self.config["gravity"]),
                float(self.config["max_altitude"]),
            )
        raise ValueError(f"Unknown opponent policy: {opponent!r}")

    def _new_state(
        self,
        uav_id: str,
        team: Team,
        x: float,
        y: float,
        z: float,
        speed: float,
        heading: float,
    ) -> UAVState:
        del uav_id
        return UAVState(
            x=x,
            y=y,
            z=z,
            speed=speed,
            flight_path_angle=0.0,
            heading_angle=heading % (2.0 * pi),
            health=self.profile.initial_health,
            alive=True,
            team_id=int(team),
            type_id=self.profile.type_id,
        )

    def _initialize_scenario(self, scenario: str) -> tuple[UAVState, UAVState]:
        if scenario == "tail_chase":
            distance = float(self.rng.uniform(self.config["initial_distance_min"], self.config["initial_distance_max"]))
            altitude = float(self.config["initial_altitude"])
            speed = float(self.config["initial_speed"])
            red = self._new_state("red", Team.RED, 0.0, 0.0, altitude, speed, 0.0)
            blue = self._new_state("blue", Team.BLUE, distance, 0.0, altitude, speed, 0.0)
            return red, blue
        if scenario == "head_on":
            distance = float(self.config["initial_distance"])
            altitude = float(self.config["initial_altitude"])
            speed = float(self.config["initial_speed"])
            red = self._new_state("red", Team.RED, -distance / 2.0, 0.0, altitude, speed, 0.0)
            blue = self._new_state("blue", Team.BLUE, distance / 2.0, 0.0, altitude, speed, pi)
            return red, blue
        if scenario == "balanced_random":
            extent = float(self.config["initial_xy_extent"])
            min_separation = float(self.config["project_assumptions"]["minimum_initial_separation"])
            for _ in range(10_000):
                red_xy = self.rng.uniform(-extent, extent, size=2)
                blue_xy = self.rng.uniform(-extent, extent, size=2)
                red_z = float(self.rng.uniform(self.config["initial_altitude_min"], self.config["initial_altitude_max"]))
                blue_z = float(self.rng.uniform(self.config["initial_altitude_min"], self.config["initial_altitude_max"]))
                separation = float(np.linalg.norm(np.asarray([*(red_xy - blue_xy), red_z - blue_z])))
                if separation >= min_separation:
                    speed = float(self.rng.uniform(self.config["initial_speed_min"], self.config["initial_speed_max"]))
                    red = self._new_state("red", Team.RED, float(red_xy[0]), float(red_xy[1]), red_z, speed, float(self.rng.uniform(0.0, 2.0 * pi)))
                    blue = self._new_state("blue", Team.BLUE, float(blue_xy[0]), float(blue_xy[1]), blue_z, speed, float(self.rng.uniform(0.0, 2.0 * pi)))
                    return red, blue
            raise RuntimeError("Could not sample a balanced scenario with sufficient separation")
        raise ValueError(f"Unknown scenario: {scenario!r}")

    def _observation(self) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        raw = actor_observation_raw_1v1(self.red.state, self.blue.state)
        observation = build_actor_observation_1v1(self.red.state, self.blue.state, self.normalization_config)
        unclipped_config = replace(self.normalization_config, clip_observation=False)
        unbounded = build_actor_observation_1v1(self.red.state, self.blue.state, unclipped_config)
        preclip_max_abs = float(np.max(np.abs(unbounded)))
        return observation, raw, preclip_max_abs

    def _outcome(self, timed_out: bool = False, reason_override: str | None = None) -> EpisodeOutcome:
        red_alive = self.red.state.alive
        blue_alive = self.blue.state.alive
        if not red_alive and not blue_alive:
            winner, reason = "draw", "simultaneous_destroyed"
        elif not blue_alive:
            winner, reason = "red", "blue_destroyed"
        elif not red_alive:
            winner, reason = "blue", "red_destroyed"
        elif timed_out:
            winner, reason = "draw", "timeout"
        else:
            winner, reason = None, "ongoing"
        return EpisodeOutcome(winner, red_alive, blue_alive, reason_override or reason, self.decision_step, self.simulation_time)

    def _event(
        self,
        event_type: CombatEventType,
        source_id: str | None = None,
        target_id: str | None = None,
        value: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CombatEvent:
        return CombatEvent(
            event_type,
            self.decision_step,
            self.simulation_time,
            source_id,
            target_id,
            value,
            metadata or {},
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        """Reset a configured scenario with all randomness controlled by one RNG."""

        effective_seed = seed
        if effective_seed is None and not self._has_reset:
            effective_seed = self._initial_seed
        gym.Env.reset(self, seed=effective_seed)
        self.rng = self.np_random
        if effective_seed is not None:
            self.action_space.seed(effective_seed)
            self._current_seed = effective_seed
        elif not self._has_reset:
            self._current_seed = None
        self._has_reset = True
        options = options or {}
        scenario = str(options.get("scenario", self.scenario_name))
        if "red_state" in options and "blue_state" in options:
            red_state = options["red_state"].copy()
            blue_state = options["blue_state"].copy()
        else:
            red_state, blue_state = self._initialize_scenario(scenario)
        self.scenario_name = scenario
        self.red = UAV("red", int(Team.RED), red_state, self.profile)
        self.blue = UAV("blue", int(Team.BLUE), blue_state, self.profile)
        self.decision_step = 0
        self.simulation_time = 0.0
        self._statistics = {
            "red_hits": 0,
            "blue_hits": 0,
            "red_damage": 0.0,
            "blue_damage": 0.0,
            "red_attack_area_steps": 0,
            "blue_attack_area_steps": 0,
            "red_attack_area_entries": 0,
            "blue_attack_area_entries": 0,
            "red_advantage_steps": 0,
            "blue_advantage_steps": 0,
            "red_advantage_entries": 0,
            "blue_advantage_entries": 0,
        }
        self._previous_red_geometry = compute_combat_geometry(self.red.state, self.blue.state, self.attack_config)
        self._previous_blue_geometry = compute_combat_geometry(self.blue.state, self.red.state, self.attack_config)
        self._trajectory = [
            {
                "decision_step": 0,
                "simulation_time": 0.0,
                "red_state": self.red.state.copy(),
                "blue_state": self.blue.state.copy(),
                "red_action": None,
                "blue_action": None,
                "damage_to_red": None,
                "damage_to_blue": None,
                "reward": 0.0,
                "events": [],
            }
        ]
        observation, raw, preclip = self._observation()
        info = {
            "red_state": self.red.state.copy(),
            "blue_state": self.blue.state.copy(),
            "actor_observation_raw": raw,
            "actor_observation_preclip_max_abs": preclip,
            "critic_state": build_critic_state_1v1(self.red.state, self.blue.state, self.normalization_config),
            "scenario_name": self.scenario_name,
            "seed": self._current_seed,
            "statistics": deepcopy(self._statistics),
        }
        return observation, info

    def _propagate_both(
        self,
        red_action: DiscreteAction15,
        blue_action: DiscreteAction15,
    ) -> tuple[dict[str, bool], Any, Any, list[dict[str, UAVState]], int]:
        red_control = clip_control(get_control(red_action), self.profile)
        blue_control = clip_control(get_control(blue_action), self.profile)
        self.red.state = replace(self.red.state, last_action=int(red_action))
        self.blue.state = replace(self.blue.state, last_action=int(blue_action))
        flags = {"red_ground": False, "red_ceiling": False, "blue_ground": False, "blue_ceiling": False}
        substeps: list[dict[str, UAVState]] = []
        executed = 0
        for _ in range(int(self.config["physics_steps_per_action"])):
            red_before = self.red.state
            blue_before = self.blue.state
            red_candidate = propagate_state(red_before, red_control, self.profile, float(self.config["physics_dt"]), float(self.config["gravity"]))
            blue_candidate = propagate_state(blue_before, blue_control, self.profile, float(self.config["physics_dt"]), float(self.config["gravity"]))
            flags["red_ground"] = red_candidate.z <= float(self.config["min_altitude"])
            flags["red_ceiling"] = red_candidate.z > float(self.config["max_altitude"])
            flags["blue_ground"] = blue_candidate.z <= float(self.config["min_altitude"])
            flags["blue_ceiling"] = blue_candidate.z > float(self.config["max_altitude"])
            if flags["red_ground"] or flags["red_ceiling"]:
                red_candidate = replace(red_candidate, health=0.0, alive=False, damaged=True, crashed=True)
            if flags["blue_ground"] or flags["blue_ceiling"]:
                blue_candidate = replace(blue_candidate, health=0.0, alive=False, damaged=True, crashed=True)
            self.red.state, self.blue.state = red_candidate, blue_candidate
            executed += 1
            substeps.append({"red_state": red_candidate.copy(), "blue_state": blue_candidate.copy()})
            if any(flags.values()):
                break
        return flags, red_control, blue_control, substeps, executed

    def _geometry_events(
        self,
        red_geometry: CombatGeometry,
        blue_geometry: CombatGeometry,
    ) -> list[CombatEvent]:
        assert self._previous_red_geometry is not None and self._previous_blue_geometry is not None
        events: list[CombatEvent] = []
        pairs = (
            ("red", "blue", red_geometry, self._previous_red_geometry),
            ("blue", "red", blue_geometry, self._previous_blue_geometry),
        )
        for source, target, current, previous in pairs:
            prefix = source
            if current.in_attack_area:
                self._statistics[f"{prefix}_attack_area_steps"] = int(self._statistics[f"{prefix}_attack_area_steps"]) + 1
            if current.in_attack_area and not previous.in_attack_area:
                self._statistics[f"{prefix}_attack_area_entries"] = int(self._statistics[f"{prefix}_attack_area_entries"]) + 1
                events.append(self._event(CombatEventType.ENTER_ATTACK_AREA, source, target))
            if current.in_advantage_area:
                self._statistics[f"{prefix}_advantage_steps"] = int(self._statistics[f"{prefix}_advantage_steps"]) + 1
            if current.in_advantage_area and not previous.in_advantage_area:
                self._statistics[f"{prefix}_advantage_entries"] = int(self._statistics[f"{prefix}_advantage_entries"]) + 1
                events.append(self._event(CombatEventType.ENTER_ADVANTAGE_AREA, source, target))
        return events

    def _resolve_attacks(
        self,
        red_geometry: CombatGeometry,
        blue_geometry: CombatGeometry,
        events: list[CombatEvent],
    ) -> tuple[DamageResult, DamageResult]:
        red_attempt = self.red.state.alive and self.blue.state.alive and red_geometry.can_attack
        blue_attempt = self.blue.state.alive and self.red.state.alive and blue_geometry.can_attack
        blue_updated, damage_to_blue = sample_damage(self.blue.state, self.damage_config, self.rng, red_attempt)
        red_updated, damage_to_red = sample_damage(self.red.state, self.damage_config, self.rng, blue_attempt)
        self.red.state, self.blue.state = red_updated, blue_updated
        for source, target, result in (
            ("red", "blue", damage_to_blue),
            ("blue", "red", damage_to_red),
        ):
            if result.attempted:
                events.append(self._event(CombatEventType.ATTACK_TRIGGERED, source, target, result.random_value))
                events.append(self._event(CombatEventType.HIT if result.damage > 0.0 else CombatEventType.MISS, source, target, result.damage))
                if result.destroyed:
                    events.append(self._event(CombatEventType.DESTROYED, source, target, result.damage))
        if damage_to_blue.damage > 0.0:
            self._statistics["red_hits"] = int(self._statistics["red_hits"]) + 1
        if damage_to_red.damage > 0.0:
            self._statistics["blue_hits"] = int(self._statistics["blue_hits"]) + 1
        self._statistics["red_damage"] = float(self._statistics["red_damage"]) + damage_to_blue.damage
        self._statistics["blue_damage"] = float(self._statistics["blue_damage"]) + damage_to_red.damage
        return damage_to_red, damage_to_blue

    def step(
        self,
        action: Any,
    ) -> tuple[NDArray[np.float64], float, bool, bool, dict[str, Any]]:
        """Advance one synchronized decision step and return the Gymnasium tuple."""

        if not hasattr(self, "red"):
            raise RuntimeError("reset() must be called before step()")
        if isinstance(action, bool) or not self.action_space.contains(action):
            raise ValueError(f"Invalid red action: {action!r}")
        red_action = DiscreteAction15(int(action))
        blue_action = self.opponent_policy.select_action(self.blue.state.copy(), self.red.state.copy(), self.rng)
        assert self._previous_red_geometry is not None and self._previous_blue_geometry is not None
        previous_red_geometry = self._previous_red_geometry
        previous_blue_geometry = self._previous_blue_geometry
        flags, red_control, blue_control, substeps, executed = self._propagate_both(red_action, blue_action)
        self.decision_step += 1
        self.simulation_time += executed * float(self.config["physics_dt"])
        events: list[CombatEvent] = []
        if flags["red_ground"]:
            events.append(self._event(CombatEventType.GROUND_CRASH, "red"))
        if flags["red_ceiling"]:
            events.append(self._event(CombatEventType.CEILING_VIOLATION, "red"))
        if flags["blue_ground"]:
            events.append(self._event(CombatEventType.GROUND_CRASH, "blue"))
        if flags["blue_ceiling"]:
            events.append(self._event(CombatEventType.CEILING_VIOLATION, "blue"))
        collision_distance = float(self.config["project_assumptions"].get("collision_distance", 0.0))
        collided = self.red.state.alive and self.blue.state.alive and has_collision(self.red.state, self.blue.state, collision_distance)
        if collided:
            self.red.state = replace(self.red.state, health=0.0, alive=False, damaged=True, crashed=True)
            self.blue.state = replace(self.blue.state, health=0.0, alive=False, damaged=True, crashed=True)
            events.append(self._event(CombatEventType.COLLISION, "red", "blue"))
        red_geometry = compute_combat_geometry(self.red.state, self.blue.state, self.attack_config)
        blue_geometry = compute_combat_geometry(self.blue.state, self.red.state, self.attack_config)
        events.extend(self._geometry_events(red_geometry, blue_geometry))
        if collided or any(flags.values()):
            _, damage_to_red = sample_damage(self.red.state, self.damage_config, self.rng, False)
            _, damage_to_blue = sample_damage(self.blue.state, self.damage_config, self.rng, False)
        else:
            damage_to_red, damage_to_blue = self._resolve_attacks(red_geometry, blue_geometry, events)
        max_steps_reached = self.decision_step >= int(self.config["max_decision_steps"])
        max_time_reached = self.simulation_time >= float(self.config["max_episode_seconds"]) - 1.0e-12
        terminated = not self.red.state.alive or not self.blue.state.alive
        truncated = not terminated and (max_steps_reached or max_time_reached)
        reason_override = None
        if collided:
            reason_override = "collision"
        elif flags["red_ground"]:
            reason_override = "red_ground_crash"
        elif flags["red_ceiling"]:
            reason_override = "red_ceiling_violation"
        elif flags["blue_ground"]:
            reason_override = "blue_ground_crash"
        elif flags["blue_ceiling"]:
            reason_override = "blue_ceiling_violation"
        outcome = self._outcome(timed_out=truncated, reason_override=reason_override)
        if truncated:
            events.append(self._event(CombatEventType.TIMEOUT))
        if terminated or truncated:
            event_type = CombatEventType.DRAW if outcome.winner == "draw" else (CombatEventType.WIN if outcome.winner == "red" else CombatEventType.LOSS)
            events.append(self._event(event_type, outcome.winner))
        reward_breakdown = compute_reward_breakdown(
            previous_red_geometry,
            red_geometry,
            previous_blue_geometry,
            blue_geometry,
            self.red.state,
            self.blue.state,
            damage_to_blue,
            damage_to_red,
            flags["red_ground"] or flags["red_ceiling"],
            outcome,
            self.decision_step,
            self.config,
        )
        self._previous_red_geometry = red_geometry
        self._previous_blue_geometry = blue_geometry
        observation, raw, preclip = self._observation()
        critic_state = build_critic_state_1v1(self.red.state, self.blue.state, self.normalization_config)
        self._trajectory.append(
            {
                "decision_step": self.decision_step,
                "simulation_time": self.simulation_time,
                "red_state": self.red.state.copy(),
                "blue_state": self.blue.state.copy(),
                "red_action": int(red_action),
                "blue_action": int(blue_action),
                "red_control": red_control,
                "blue_control": blue_control,
                "substeps": substeps,
                "damage_to_red": damage_to_red,
                "damage_to_blue": damage_to_blue,
                "reward": reward_breakdown.total,
                "reward_breakdown": reward_breakdown,
                "events": list(events),
            }
        )
        info = {
            "red_action": int(red_action),
            "blue_action": int(blue_action),
            "red_control": red_control,
            "blue_control": blue_control,
            "red_state": self.red.state.copy(),
            "blue_state": self.blue.state.copy(),
            "red_to_blue_geometry": red_geometry,
            "blue_to_red_geometry": blue_geometry,
            "damage_to_red": damage_to_red,
            "damage_to_blue": damage_to_blue,
            "events": events,
            "reward_breakdown": reward_breakdown,
            "actor_observation_raw": raw,
            "actor_observation_preclip_max_abs": preclip,
            "critic_state": critic_state,
            "simulation_time": self.simulation_time,
            "decision_step": self.decision_step,
            "outcome": outcome,
            "statistics": deepcopy(self._statistics),
        }
        return observation, reward_breakdown.total, terminated, truncated, info

    def get_trajectory(self) -> list[dict[str, Any]]:
        """Return a defensive copy of all decision and physical-step records."""

        return deepcopy(self._trajectory)

    def get_statistics(self) -> dict[str, float | int]:
        """Return a defensive copy of cumulative combat-area and damage counts."""

        return deepcopy(self._statistics)
