"""Runnable fixed homogeneous 2v2/3v3 Gymnasium combat environment."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from itertools import combinations
from math import pi
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from uav_env.actions.discrete_15 import DiscreteAction15, get_control
from uav_env.combat.attack_geometry import AttackZoneConfig, compute_combat_geometry
from uav_env.combat.collision import has_collision
from uav_env.combat.damage import DamageConfig
from uav_env.combat.events import CombatEvent, EpisodeOutcome
from uav_env.combat.multi_combat import AttackAttempt, ResolvedAttack, TargetAssignment, assign_nearest_targets_independently, assign_targets, resolve_multi_attacks
from uav_env.core.enums import CombatEventType, Team
from uav_env.core.state import UAVState
from uav_env.dynamics.propagation import propagate_state
from uav_env.entities.type_profiles import UAVTypeProfile, profile_from_config
from uav_env.entities.uav import UAV
from uav_env.envs.base_env import BaseUAVEnv
from uav_env.observations.global_state import GlobalStateResult, build_global_state, build_global_state_v2
from uav_env.observations.multi_observation import MultiObservationResult, build_multi_observations, build_multi_observations_v2
from uav_env.observations.normalization import NormalizationConfig
from uav_env.opponents.base import RuleOpponent
from uav_env.opponents.pursuit import PursuitOpponent
from uav_env.opponents.random import RandomOpponent
from uav_env.opponents.straight import StraightOpponent
from uav_env.rewards.components import advantage_reward
from uav_env.rewards.multi_reward import MultiAgentRewardBreakdown, assign_dense_rewards, individual_situation_reward, multi_terminal_reward_allocations
from uav_env.utils.config import validate_experiment_config


class CombatMultiEnv(BaseUAVEnv):
    """Fixed homogeneous equal-team experiment for team sizes two or three."""

    metadata = {"render_modes": []}

    def __init__(self, config: dict[str, Any], scenario_name: str | None = None, opponent: str = "straight", seed: int | None = None) -> None:
        validate_experiment_config(config)
        self.red_count = int(config["red_count"])
        self.blue_count = int(config["blue_count"])
        if self.red_count != self.blue_count or self.red_count not in {2, 3}:
            raise ValueError("CombatMultiEnv supports equal homogeneous team sizes of 2 or 3")
        self.config = deepcopy(config)
        self.environment_schema_version = str(config.get("environment_schema_version", "legacy"))
        self.is_v2 = self.environment_schema_version == "homogeneous_3v3_v2"
        self.scenario_name = scenario_name or str(config["scenario_name"])
        if self.red_count == 3 and self.scenario_name not in {"head_on_formation", "head_on_mirrored_jitter_v2", "symmetric_stress_test_v2"}:
            raise ValueError("Fixed 3v3 currently supports only head-on legacy or V2 scenarios")
        self.num_red_agents = self.red_count
        self.local_observation_dim = 62 if self.is_v2 else (self.red_count - 1) * 6 + self.blue_count * 11
        self.global_state_dim = 60 if self.is_v2 else self.red_count + self.red_count * self.blue_count * 9 + self.red_count
        self.profile: UAVTypeProfile = profile_from_config(config)
        self.attack_config = AttackZoneConfig.from_config(config)
        self.damage_config = DamageConfig.from_config(config)
        self.normalization_config = NormalizationConfig.from_config(config)
        self.action_space = spaces.MultiDiscrete([15] * self.red_count)
        bounds = (-1.0, 1.0) if self.normalization_config.mode == "symmetric_training" else (-np.inf, np.inf)
        self.observation_space = spaces.Box(bounds[0], bounds[1], shape=(self.red_count, self.local_observation_dim), dtype=np.float64)
        self._initial_seed = seed
        self._has_reset = False
        self._current_seed: int | None = None
        self.rng = np.random.default_rng(seed)
        self.blue_rule_rng = np.random.default_rng(None if seed is None else seed + 2_000_003)
        self.opponent_name = opponent
        self.opponent_policy = self._build_policy(opponent)
        self.red_aircraft: list[UAV] = []
        self.blue_aircraft: list[UAV] = []
        self.decision_step = 0
        self.simulation_time = 0.0
        self._trajectory: list[dict[str, Any]] = []
        self._statistics: dict[str, Any] = {}
        self._previous_states: dict[str, UAVState] = {}
        self.damage_sample_team_order: tuple[int, ...] | None = None

    @property
    def all_aircraft(self) -> list[UAV]:
        """Return stable red-major then blue-major aircraft order."""

        return [*self.red_aircraft, *self.blue_aircraft]

    def _build_policy(self, name: str) -> RuleOpponent:
        if name == "straight":
            return StraightOpponent()
        if name == "random":
            return RandomOpponent()
        if name == "pursuit":
            pursuit = self.config["pursuit"]
            return PursuitOpponent(
                self.profile, self.attack_config, float(self.config["physics_dt"]), int(self.config["physics_steps_per_action"]),
                float(self.config["gravity"]), float(self.config["max_altitude"]),
                float(pursuit["angle_weight"]), float(pursuit["distance_weight"]), float(pursuit["altitude_weight"]),
                float(pursuit["boundary_penalty"]), float(pursuit["minimum_safe_altitude"]),
                float(pursuit["ceiling_margin"]), float(pursuit["unsafe_flight_path_penalty"]),
            )
        raise ValueError(f"Unknown blue policy: {name!r}")

    def _state(self, team: Team, x: float, y: float, z: float, speed: float, heading: float) -> UAVState:
        return UAVState(
            x, y, z, speed, 0.0, heading % (2.0 * pi), self.profile.initial_health, True, int(team), self.profile.type_id,
            last_action=int(DiscreteAction15.LEVEL_HOLD),
        )

    def _initialize_scenario(self, scenario: str) -> tuple[list[UAVState], list[UAVState]]:
        if scenario in {"head_on_mirrored_jitter_v2", "symmetric_stress_test_v2"}:
            if self.red_count != 3:
                raise ValueError("V2 scenarios are fixed homogeneous 3v3 only")
            distance = float(self.config["initial_team_distance"])
            spacing = float(self.config["formation_lateral_spacing"])
            altitude = float(self.config["initial_altitude"])
            speed = float(self.config["initial_speed"])
            bases = [(index - 1.0) * spacing for index in range(3)]
            jitter_enabled = bool(self.config.get("mirror_jitter_enabled", scenario == "head_on_mirrored_jitter_v2"))
            reds: list[UAVState] = []
            blues: list[UAVState] = []
            for base_y in bases:
                if jitter_enabled:
                    longitudinal = float(self.rng.uniform(-float(self.config["longitudinal_jitter"]), float(self.config["longitudinal_jitter"])))
                    lateral = float(self.rng.uniform(-float(self.config["lateral_jitter"]), float(self.config["lateral_jitter"])))
                    z_jitter = float(self.rng.uniform(-float(self.config["altitude_jitter"]), float(self.config["altitude_jitter"])))
                    speed_jitter = float(self.rng.uniform(-float(self.config["speed_jitter"]), float(self.config["speed_jitter"])))
                    heading_jitter = float(self.rng.uniform(-float(self.config["heading_jitter"]), float(self.config["heading_jitter"])))
                else:
                    longitudinal = lateral = z_jitter = speed_jitter = heading_jitter = 0.0
                reds.append(self._state(Team.RED, -distance / 2.0 + longitudinal, base_y + lateral, altitude + z_jitter, speed + speed_jitter, heading_jitter))
                blues.append(self._state(Team.BLUE, distance / 2.0 - longitudinal, base_y - lateral, altitude - z_jitter, speed + speed_jitter, pi + heading_jitter))
            return reds, blues
        if scenario in {"head_on_formation", "offset_formation"}:
            distance = float(self.config["initial_team_distance"])
            spacing = float(self.config["formation_lateral_spacing"])
            altitude = float(self.config["initial_altitude"])
            speed = float(self.config["initial_speed"])
            offset_y = float(self.config.get("blue_lateral_offset", 0.0))
            offset_z = float(self.config.get("blue_altitude_offset", 0.0))
            lateral_positions = [(index - (self.red_count - 1) / 2.0) * spacing for index in range(self.red_count)]
            reds = [self._state(Team.RED, -distance / 2.0, y, altitude, speed, 0.0) for y in lateral_positions]
            blues = [self._state(Team.BLUE, distance / 2.0, y + offset_y, altitude + offset_z, speed, pi) for y in lateral_positions]
            return reds, blues
        if scenario == "balanced_random":
            if self.red_count != 2:
                raise ValueError("balanced_random is retained only for 2v2")
            extent = float(self.config["initial_xy_extent"])
            same_min = float(self.config["same_team_minimum_separation"])
            opposing_min = max(float(self.config["opposing_team_minimum_separation"]), float(self.config["attack_distance_max"]) + 1.0)
            for _ in range(10_000):
                positions = [np.asarray([*self.rng.uniform(-extent, extent, 2), self.rng.uniform(self.config["initial_altitude_min"], self.config["initial_altitude_max"])]) for _ in range(4)]
                same_ok = np.linalg.norm(positions[0] - positions[1]) >= same_min and np.linalg.norm(positions[2] - positions[3]) >= same_min
                opposing_ok = all(np.linalg.norm(positions[r] - positions[b]) >= opposing_min for r in (0, 1) for b in (2, 3))
                if same_ok and opposing_ok:
                    states = [
                        self._state(Team.RED if index < 2 else Team.BLUE, float(p[0]), float(p[1]), float(p[2]),
                                    float(self.rng.uniform(self.config["initial_speed_min"], self.config["initial_speed_max"])), float(self.rng.uniform(0.0, 2.0 * pi)))
                        for index, p in enumerate(positions)
                    ]
                    return states[:2], states[2:]
            raise RuntimeError("Could not sample a safe balanced 2v2 initialization")
        raise ValueError(f"Unknown 2v2 scenario: {scenario!r}")

    def _observations(self) -> MultiObservationResult:
        if self.is_v2:
            return build_multi_observations_v2(self.red_aircraft, self.blue_aircraft, self.config, self.attack_config)
        return build_multi_observations(self.red_aircraft, self.blue_aircraft, self.normalization_config)

    def _global_state(self) -> GlobalStateResult:
        if self.is_v2:
            return build_global_state_v2(self.red_aircraft, self.blue_aircraft, self.config)
        return build_global_state(self.red_aircraft, self.blue_aircraft, self.normalization_config)

    def _available_action_mask(self) -> NDArray[np.int8]:
        mask = np.ones((self.red_count, 15), dtype=np.int8)
        for index, aircraft in enumerate(self.red_aircraft):
            if not aircraft.is_alive:
                mask[index] = 0
                mask[index, int(DiscreteAction15.LEVEL_HOLD)] = 1
        return mask

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[NDArray[np.float64], dict[str, Any]]:
        """Reset a reproducible fixed-size multi-aircraft scenario."""

        effective = seed if seed is not None else (self._initial_seed if not self._has_reset else None)
        gym.Env.reset(self, seed=effective)
        self.rng = self.np_random
        if effective is not None:
            self.action_space.seed(effective)
            self._current_seed = effective
            self.blue_rule_rng = np.random.default_rng(effective + 2_000_003)
        self._has_reset = True
        options = options or {}
        scenario = str(options.get("scenario", self.scenario_name))
        if "red_states" in options and "blue_states" in options:
            red_states = [state.copy() for state in options["red_states"]]
            blue_states = [state.copy() for state in options["blue_states"]]
        else:
            red_states, blue_states = self._initialize_scenario(scenario)
        self.scenario_name = scenario
        self.red_aircraft = [UAV(f"red_{i}", int(Team.RED), state, self.profile) for i, state in enumerate(red_states)]
        self.blue_aircraft = [UAV(f"blue_{i}", int(Team.BLUE), state, self.profile) for i, state in enumerate(blue_states)]
        self.decision_step = 0
        self.simulation_time = 0.0
        per_aircraft = {
            u.uav_id: {"attack_attempts": 0, "nominal_damage": 0.0, "effective_damage": 0.0, "overkill_damage": 0.0, "hits": 0, "attack_area_steps": 0, "contribution_score": 0.0, "cumulative_reward": 0.0, "ground_crashes": 0, "ceiling_violations": 0}
            for u in self.all_aircraft
        }
        self._statistics = {"aircraft": per_aircraft, "collisions": 0, "timeouts": 0}
        self._previous_states = {u.uav_id: u.state.copy() for u in self.all_aircraft}
        self._trajectory = [{"decision_step": 0, "simulation_time": 0.0, "states": deepcopy(self._previous_states), "red_actions": None, "blue_actions": None, "events": []}]
        return self._build_step_output_info(reset=True)

    def _build_step_output_info(self, reset: bool = False, **extra: Any) -> tuple[NDArray[np.float64], dict[str, Any]]:
        local = self._observations()
        global_state = self._global_state()
        info = {
            "red_states": [u.state.copy() for u in self.red_aircraft], "blue_states": [u.state.copy() for u in self.blue_aircraft],
            "local_observations_raw": local.raw, "local_observations": local.normalized,
            "local_observation_feature_names": local.feature_names, "ally_alive_masks": local.ally_alive_masks,
            "local_observation_feature_names_by_agent": local.feature_names_by_agent,
            "local_observation_saturation_count": local.saturation_count,
            "local_observation_saturation_ratio": local.saturation_ratio,
            "local_observation_saturated_feature_names": local.saturated_feature_names,
            "enemy_alive_masks": local.enemy_alive_masks, "red_agent_alive_mask": local.own_alive_mask,
            "blue_alive_mask": np.asarray([int(u.is_alive) for u in self.blue_aircraft], dtype=np.int8),
            "available_action_mask": self._available_action_mask(), "global_state_raw": global_state.raw,
            "global_state": global_state.normalized, "global_state_feature_names": global_state.feature_names,
            "global_state_saturation_count": global_state.saturation_count,
            "global_state_saturation_ratio": global_state.saturation_ratio,
            "global_state_saturated_feature_names": global_state.saturated_feature_names,
            "statistics": self.get_statistics(), "outcome": self._outcome(False), "simulation_time": self.simulation_time,
            "decision_step": self.decision_step, "scenario_name": self.scenario_name, "seed": self._current_seed,
        }
        if not reset:
            info.update(extra)
        return local.normalized, info

    def _blue_actions(self, assignments: Sequence[TargetAssignment]) -> list[DiscreteAction15]:
        assignment_map = {a.attacker_id: a.target_id for a in assignments}
        red_map = {u.uav_id: u for u in self.red_aircraft}
        actions: list[DiscreteAction15] = []
        for blue in sorted(self.blue_aircraft, key=lambda u: u.uav_id):
            if not blue.is_alive or blue.uav_id not in assignment_map:
                actions.append(DiscreteAction15.LEVEL_HOLD)
            else:
                actions.append(self.opponent_policy.select_action(blue.state.copy(), red_map[assignment_map[blue.uav_id]].state.copy(), self.blue_rule_rng))
        return actions

    def _propagate_all(self, action_map: dict[str, DiscreteAction15]) -> tuple[list[dict[str, UAVState]], dict[str, str], int]:
        substeps: list[dict[str, UAVState]] = []
        boundary: dict[str, str] = {}
        executed = 0
        for _ in range(int(self.config["physics_steps_per_action"])):
            candidates: dict[str, UAVState] = {}
            for aircraft in self.all_aircraft:
                if aircraft.is_alive:
                    candidate = propagate_state(aircraft.state, get_control(action_map[aircraft.uav_id]), self.profile, float(self.config["physics_dt"]), float(self.config["gravity"]))
                    if candidate.z <= float(self.config["min_altitude"]):
                        candidate = replace(candidate, health=0.0, alive=False, damaged=True, crashed=True)
                        boundary[aircraft.uav_id] = "ground"
                    elif candidate.z > float(self.config["max_altitude"]):
                        candidate = replace(candidate, health=0.0, alive=False, damaged=True)
                        boundary[aircraft.uav_id] = "ceiling"
                    candidates[aircraft.uav_id] = candidate
                else:
                    candidates[aircraft.uav_id] = aircraft.state.copy()
            for aircraft in self.all_aircraft:
                aircraft.state = candidates[aircraft.uav_id]
            executed += 1
            substeps.append({key: state.copy() for key, state in candidates.items()})
            if not any(u.is_alive for u in self.red_aircraft) or not any(u.is_alive for u in self.blue_aircraft):
                break
        return substeps, boundary, executed

    def _resolve_collisions(self) -> tuple[list[tuple[str, str]], set[str]]:
        distance = float(self.config["project_assumptions"].get("collision_distance", 0.0))
        pairs: list[tuple[str, str]] = []
        affected: set[str] = set()
        if distance <= 0.0:
            return pairs, affected
        for first, second in combinations(self.all_aircraft, 2):
            if first.is_alive and second.is_alive and has_collision(first.state, second.state, distance):
                pairs.append((first.uav_id, second.uav_id))
                affected.update((first.uav_id, second.uav_id))
        for aircraft in self.all_aircraft:
            if aircraft.uav_id in affected:
                aircraft.state = replace(aircraft.state, health=0.0, alive=False, damaged=True)
        return pairs, affected

    def _outcome(self, timed_out: bool) -> EpisodeOutcome:
        red_survivors = sum(u.is_alive for u in self.red_aircraft)
        blue_survivors = sum(u.is_alive for u in self.blue_aircraft)
        if red_survivors == 0 and blue_survivors == 0:
            winner, reason = "draw", "simultaneous_elimination"
        elif blue_survivors == 0:
            winner, reason = "red", "blue_eliminated"
        elif red_survivors == 0:
            winner, reason = "blue", "red_eliminated"
        elif timed_out:
            winner = "red" if red_survivors > blue_survivors else "blue" if blue_survivors > red_survivors else "draw"
            reason = "timeout"
        else:
            winner, reason = None, "ongoing"
        return EpisodeOutcome(winner, red_survivors > 0, blue_survivors > 0, reason, self.decision_step, self.simulation_time, red_survivors, blue_survivors)

    def _event_reward(self, red: UAV, resolved: Sequence[ResolvedAttack], boundary: dict[str, str], collision_ids: set[str]) -> tuple[float, float]:
        geometry, contribution = self._geometry_event_reward(red)
        combat, combat_contribution, _ = self._combat_event_reward(red, resolved, boundary, collision_ids)
        return geometry + combat, contribution + combat_contribution

    def _geometry_event_reward(self, red: UAV) -> tuple[float, float]:
        event, contribution = 0.0, 0.0
        if red.is_alive:
            for blue in self.blue_aircraft:
                if not blue.is_alive:
                    continue
                own = compute_combat_geometry(red.state, blue.state, self.attack_config)
                enemy = compute_combat_geometry(blue.state, red.state, self.attack_config)
                if own.in_advantage_area:
                    event += advantage_reward(own.distance, own.target_escape_angle, float(self.config["advantage_distance_min"]), float(self.config["advantage_distance_max"]))
                    contribution += 1.0
                if enemy.in_advantage_area:
                    event -= 1.0
                if own.in_attack_area:
                    event += 0.3
                if enemy.in_attack_area:
                    event -= 0.3
        return event, contribution

    def _combat_event_reward(self, red: UAV, resolved: Sequence[ResolvedAttack], boundary: dict[str, str], collision_ids: set[str]) -> tuple[float, float, dict[str, float]]:
        event, contribution = 0.0, 0.0
        components = {"hit": 0.0, "destroy": 0.0, "attacked": 0.0, "destroyed": 0.0, "boundary_collision": 0.0}
        for attack in resolved:
            if attack.attacker_id == red.uav_id and attack.hit:
                event += 0.8; components["hit"] += 0.8
                contribution += 2.0
                if attack.destroy_credit:
                    event += 1.5; components["destroy"] += 1.5
                    contribution += 5.0
            if attack.target_id == red.uav_id and attack.hit:
                event -= 0.9; components["attacked"] -= 0.9
        # Destruction is a target-level event.  Multiple simultaneous hits on
        # the same aircraft must not multiply the destroyed penalty.
        if any(attack.target_id == red.uav_id and attack.destroy_credit for attack in resolved):
            event -= 1.6; components["destroyed"] -= 1.6
        if red.uav_id in boundary or red.uav_id in collision_ids:
            event -= 0.5; components["boundary_collision"] -= 0.5
        return event, contribution, components

    def step(self, action: Any) -> tuple[NDArray[np.float64], float, bool, bool, dict[str, Any]]:
        """Advance one synchronized multi-aircraft decision step."""

        if len(self.red_aircraft) != self.red_count:
            raise RuntimeError("reset() must be called before step()")
        actions_array = np.asarray(action)
        if actions_array.shape != (self.red_count,) or not self.action_space.contains(actions_array):
            raise ValueError(f"Red action must be a valid length-{self.red_count} MultiDiscrete action")
        red_actions = [DiscreteAction15(int(value)) if aircraft.is_alive else DiscreteAction15.LEVEL_HOLD for value, aircraft in zip(actions_array, self.red_aircraft)]
        assignments = (
            assign_nearest_targets_independently(self.blue_aircraft, self.red_aircraft)
            if self.blue_count == 3
            else assign_targets(self.blue_aircraft, self.red_aircraft)
        )
        blue_actions = self._blue_actions(assignments)
        action_map = {u.uav_id: action for u, action in zip(self.red_aircraft, red_actions)} | {u.uav_id: action for u, action in zip(self.blue_aircraft, blue_actions)}
        for aircraft in self.all_aircraft:
            aircraft.state = replace(aircraft.state, last_action=int(action_map[aircraft.uav_id]))
        previous_states = {u.uav_id: u.state.copy() for u in self.all_aircraft}
        red_alive_at_step_start = {red.uav_id: bool(red.is_alive) for red in self.red_aircraft}
        substeps, boundary, executed = self._propagate_all(action_map)
        collision_pairs, collision_ids = self._resolve_collisions()
        combat_result = resolve_multi_attacks(
            self.all_aircraft, self.attack_config, self.damage_config, self.rng,
            self.damage_sample_team_order,
        )
        for aircraft in self.all_aircraft:
            aircraft.state = combat_result.updated_states[aircraft.uav_id]
        red_damaged_this_step = {
            red.uav_id: red_alive_at_step_start[red.uav_id] and not red.is_alive
            for red in self.red_aircraft
        }
        step_active_ids = {red_id for red_id, was_alive in red_alive_at_step_start.items() if was_alive}
        self.decision_step += 1
        self.simulation_time += executed * float(self.config["physics_dt"])
        for aircraft_id, kind in boundary.items():
            self._statistics["aircraft"][aircraft_id][f"{kind}_crashes" if kind == "ground" else "ceiling_violations"] += 1
        self._statistics["collisions"] += len(collision_pairs)
        for attempt in combat_result.attack_attempts:
            self._statistics["aircraft"][attempt.attacker_id]["attack_attempts"] += 1
        for attack in combat_result.resolved_attacks:
            stats = self._statistics["aircraft"][attack.attacker_id]
            stats["nominal_damage"] += attack.nominal_damage
            stats["effective_damage"] += attack.effective_damage
            stats["overkill_damage"] += attack.overkill_damage
            stats["hits"] += int(attack.hit)
        for attacker in self.all_aircraft:
            opponents = self.blue_aircraft if attacker.team == int(Team.RED) else self.red_aircraft
            if attacker.is_alive and any(
                target.is_alive and compute_combat_geometry(attacker.state, target.state, self.attack_config).in_attack_area
                for target in opponents
            ):
                self._statistics["aircraft"][attacker.uav_id]["attack_area_steps"] += 1
        maxed = self.decision_step >= int(self.config["max_decision_steps"]) or self.simulation_time >= float(self.config["max_episode_seconds"]) - 1.0e-12
        terminated = not any(u.is_alive for u in self.red_aircraft) or not any(u.is_alive for u in self.blue_aircraft)
        truncated = not terminated and maxed
        if truncated:
            self._statistics["timeouts"] += 1
        outcome = self._outcome(truncated)
        raw_dense: dict[str, float] = {}
        event_values: dict[str, float] = {}
        step_contributions: dict[str, float] = {}
        geometry_values: dict[str, float] = {}
        combat_values: dict[str, float] = {}
        combat_components: dict[str, dict[str, float]] = {}
        for red in self.red_aircraft:
            if red.uav_id not in step_active_ids:
                event_values[red.uav_id] = 0.0
                step_contributions[red.uav_id] = 0.0
                geometry_values[red.uav_id] = 0.0
                combat_values[red.uav_id] = 0.0
                combat_components[red.uav_id] = {"hit": 0.0, "destroy": 0.0, "attacked": 0.0, "destroyed": 0.0, "boundary_collision": 0.0}
                continue
            situation = individual_situation_reward(red, self.blue_aircraft, previous_states, self.config) if red.is_alive else 0.0
            if self.is_v2:
                geometry_event, geometry_contribution = self._geometry_event_reward(red)
                combat_event, combat_contribution, components = self._combat_event_reward(red, combat_result.resolved_attacks, boundary, collision_ids)
                event = geometry_event + combat_event
                contribution = geometry_contribution + combat_contribution
                geometry_values[red.uav_id] = geometry_event
                combat_values[red.uav_id] = combat_event
                combat_components[red.uav_id] = components
                raw_dense[red.uav_id] = situation + geometry_event
            else:
                event, contribution = self._event_reward(red, combat_result.resolved_attacks, boundary, collision_ids)
                geometry_values[red.uav_id] = event
                combat_values[red.uav_id] = 0.0
                combat_components[red.uav_id] = {"hit": 0.0, "destroy": 0.0, "attacked": 0.0, "destroyed": 0.0, "boundary_collision": 0.0}
                raw_dense[red.uav_id] = situation + event
            event_values[red.uav_id] = event
            step_contributions[red.uav_id] = contribution
            self._statistics["aircraft"][red.uav_id]["contribution_score"] += contribution
        assigned = (
            assign_dense_rewards(
                raw_dense,
                {red_id: red_damaged_this_step[red_id] for red_id in raw_dense},
                self.red_count,
                float(self.config["r_den0"]),
            )
            if raw_dense
            else {}
        )
        terminal = multi_terminal_reward_allocations(outcome, self.red_aircraft, {u.uav_id: self._statistics["aircraft"][u.uav_id]["contribution_score"] for u in self.red_aircraft}, self.config)
        breakdowns: dict[str, MultiAgentRewardBreakdown] = {}
        for red in self.red_aircraft:
            situation = raw_dense.get(red.uav_id, 0.0) - geometry_values[red.uav_id] if self.is_v2 else raw_dense.get(red.uav_id, 0.0) - event_values[red.uav_id]
            allocation=terminal[red.uav_id]
            dense_reward = assigned.get(red.uav_id, 0.0) + (combat_values[red.uav_id] if self.is_v2 else 0.0)
            total = dense_reward + allocation.reward
            components = combat_components[red.uav_id]
            breakdowns[red.uav_id] = MultiAgentRewardBreakdown(
                situation=situation, event=event_values[red.uav_id], raw_dense=raw_dense.get(red.uav_id, 0.0),
                assigned_dense=dense_reward, terminal=allocation.reward, total=total,
                contribution_score=step_contributions[red.uav_id], terminal_profile=allocation.profile,
                terminal_team_base=allocation.team_base, terminal_allocation_factor=allocation.allocation_factor,
                terminal_base_share_component=allocation.base_share_component,
                terminal_survival_component=allocation.survival_component,
                terminal_contribution_component=allocation.contribution_component,
                terminal_health_component=allocation.health_component,
                terminal_alive_count=allocation.alive_count,
                terminal_contribution_denominator=allocation.contribution_denominator,
                terminal_health_denominator=allocation.health_denominator,
                geometry_event=geometry_values[red.uav_id],
                combat_event=combat_values[red.uav_id],
                raw_shape=raw_dense.get(red.uav_id, 0.0),
                assigned_shape=assigned.get(red.uav_id, 0.0),
                dense_reward=dense_reward,
                hit_event_reward=components["hit"],
                destroy_event_reward=components["destroy"],
                attacked_event_penalty=components["attacked"],
                destroyed_event_penalty=components["destroyed"],
                boundary_collision_penalty=components["boundary_collision"],
            )
            self._statistics["aircraft"][red.uav_id]["cumulative_reward"] += total
        agent_rewards = {key: value.total for key, value in breakdowns.items()}
        team_reward = float(np.mean(list(agent_rewards.values())))
        events: list[CombatEvent] = []
        for aircraft_id, kind in boundary.items():
            events.append(CombatEvent(CombatEventType.GROUND_CRASH if kind == "ground" else CombatEventType.CEILING_VIOLATION, self.decision_step, self.simulation_time, aircraft_id))
        for first, second in collision_pairs:
            events.append(CombatEvent(CombatEventType.COLLISION, self.decision_step, self.simulation_time, first, second))
        for attack in combat_result.resolved_attacks:
            events.append(CombatEvent(CombatEventType.HIT if attack.hit else CombatEventType.MISS, self.decision_step, self.simulation_time, attack.attacker_id, attack.target_id, attack.effective_damage))
            if attack.destroy_credit:
                events.append(CombatEvent(CombatEventType.DESTROYED, self.decision_step, self.simulation_time, attack.attacker_id, attack.target_id, attack.effective_damage))
        if truncated:
            events.append(CombatEvent(CombatEventType.TIMEOUT, self.decision_step, self.simulation_time))
        self._previous_states = previous_states
        self._trajectory.append({
            "decision_step": self.decision_step, "simulation_time": self.simulation_time,
            "states": {u.uav_id: u.state.copy() for u in self.all_aircraft}, "red_actions": [int(a) for a in red_actions],
            "blue_actions": [int(a) for a in blue_actions], "substeps": substeps, "attack_attempts": combat_result.attack_attempts,
            "resolved_attacks": combat_result.resolved_attacks, "agent_rewards": agent_rewards, "team_reward": team_reward, "events": events,
        })
        observation, info = self._build_step_output_info(
            red_actions=[int(a) for a in red_actions], blue_actions=[int(a) for a in blue_actions],
            blue_target_assignments=list(assignments), attack_attempts=combat_result.attack_attempts,
            resolved_attacks=combat_result.resolved_attacks, events=events, agent_reward_breakdowns=breakdowns,
            agent_rewards=agent_rewards, team_reward=team_reward, outcome=outcome,
        )
        return observation, team_reward, terminated, truncated, info

    def get_trajectory(self) -> list[dict[str, Any]]:
        """Return the complete decision and physical-substep trajectory."""

        return deepcopy(self._trajectory)

    def get_statistics(self) -> dict[str, Any]:
        """Return cumulative per-aircraft and episode statistics."""

        return deepcopy(self._statistics)

    def get_global_state(self) -> NDArray[np.float64]:
        """Return the current normalized centralized state."""

        return self._global_state().normalized.copy()

    def get_agent_masks(self) -> dict[str, NDArray[np.int8]]:
        """Return fixed agent-alive and available-action masks."""

        return {"agent_alive_mask": np.asarray([int(u.is_alive) for u in self.red_aircraft], dtype=np.int8), "available_action_mask": self._available_action_mask()}

    def __getstate__(self) -> dict[str, Any]:
        """Keep checkpoint state compact while preserving current dynamics state."""

        state = self.__dict__.copy()
        state["_trajectory"] = self._trajectory[-1:] if self._trajectory else []
        return state
