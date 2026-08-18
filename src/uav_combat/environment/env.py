"""Independent public-parameter 4v4 low-fidelity air-combat environment."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np

from ..config import aircraft_spec, load_config
from ..dynamics import PointMassDynamics
from ..integrator import RK4Integrator
from ..models import AircraftState
from .control import action_to_control
from .fixed_policy import NearestTargetPursuitPolicy
from .geometry import engagement_geometry, engagement_score
from .observation import OBSERVATION_DIM, build_team_observations
from .reward import team_potentials
from .scenario import random_line_abreast_states
from .weapon import LockState, WeaponEnvelope


class MultiUAVCombatEnv:
    """Four learned red UAVs versus a deterministic blue pursuit policy."""

    team_size, observation_dim, action_dim = 4, OBSERVATION_DIM, 3

    def __init__(self, config: str | Path | dict[str, Any] = "configs/combat_environment.yaml") -> None:
        self.config = load_config(config) if not isinstance(config, dict) else config
        self.spec = aircraft_spec(self.config)
        self.dt = float(self.config["simulation"]["dt"])
        self.max_steps = int(self.config["simulation"]["max_steps"])
        self.radius = float(self.config["battlefield"]["horizontal_radius"])
        self.altitude_min = float(self.config["battlefield"]["altitude_min"])
        self.altitude_max = float(self.config["battlefield"]["altitude_max"])
        self.dynamics = PointMassDynamics()
        self.integrator = RK4Integrator(self.dt)
        self.fixed_policy = NearestTargetPursuitPolicy(
            self.config["blue_policy"], self.config["battlefield"]
        )
        self.weapon = WeaponEnvelope(**self.config["weapon"])
        self.rng = np.random.default_rng()
        self.red: list[AircraftState] = []
        self.blue: list[AircraftState] = []
        self.red_locks = [LockState() for _ in range(4)]
        self.blue_locks = [LockState() for _ in range(4)]
        self.red_boundary_dead = np.zeros(4, dtype=bool)
        self.blue_boundary_dead = np.zeros(4, dtype=bool)
        self.red_boundary_causes = np.full(4, None, dtype=object)
        self.blue_boundary_causes = np.full(4, None, dtype=object)
        self.steps = 0
        self.red_attack_kills = 0
        self.blue_attack_kills = 0
        self.first_attackable_step: int | None = None
        self.first_lock_step: int | None = None
        self.first_kill_step: int | None = None

    @property
    def red_alive_mask(self) -> np.ndarray:
        return np.asarray([state.alive for state in self.red], dtype=np.float32)

    @property
    def blue_alive_mask(self) -> np.ndarray:
        return np.asarray([state.alive for state in self.blue], dtype=np.float32)

    @property
    def red_boundary_losses(self) -> int:
        return int(self.red_boundary_dead.sum())

    @property
    def blue_boundary_losses(self) -> int:
        return int(self.blue_boundary_dead.sum())

    def _boundary_cause_count(self, causes: np.ndarray, cause: str) -> int:
        return int(np.count_nonzero(causes == cause))

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        self.rng = np.random.default_rng(seed)
        self.red, self.blue, radial_angle = random_line_abreast_states(
            self.rng, **self.config["scenario"]
        )
        self.red_locks = [LockState() for _ in range(4)]
        self.blue_locks = [LockState() for _ in range(4)]
        self.red_boundary_dead.fill(False)
        self.blue_boundary_dead.fill(False)
        self.red_boundary_causes.fill(None)
        self.blue_boundary_causes.fill(None)
        self.steps = self.red_attack_kills = self.blue_attack_kills = 0
        self.first_attackable_step = self.first_lock_step = self.first_kill_step = None
        return self._observations(), {
            "radial_angle": radial_angle,
            "red_alive_mask": self.red_alive_mask,
            "blue_alive_mask": self.blue_alive_mask,
        }

    def _observations(self) -> np.ndarray:
        return build_team_observations(self.red, self.blue, self.config["observation"])

    @staticmethod
    def _snapshot(states: list[AircraftState]) -> list[AircraftState]:
        return [state.copy() for state in states]

    def _advance(self, states: list[AircraftState], actions: np.ndarray) -> None:
        for index, state in enumerate(states):
            if state.alive:
                control = action_to_control(actions[index], self.config["action"])
                states[index] = self.integrator.step(state, control, self.dynamics, self.spec)

    def _outside(self, state: AircraftState) -> bool:
        return bool(
            np.hypot(state.x, state.y) > self.radius
            or state.altitude < self.altitude_min
            or state.altitude > self.altitude_max
        )

    def _boundary_cause(self, state: AircraftState) -> str | None:
        if np.hypot(state.x, state.y) > self.radius:
            return "horizontal"
        if state.altitude < self.altitude_min:
            return "altitude_low"
        if state.altitude > self.altitude_max:
            return "altitude_high"
        return None

    def _resolve_boundaries(self) -> tuple[list[int], list[int]]:
        losses: tuple[list[int], list[int]] = ([], [])
        for team_index, (states, flags, causes, locks) in enumerate((
            (self.red, self.red_boundary_dead, self.red_boundary_causes, self.red_locks),
            (self.blue, self.blue_boundary_dead, self.blue_boundary_causes, self.blue_locks),
        )):
            for index, state in enumerate(states):
                cause = self._boundary_cause(state) if state.alive else None
                if cause is not None:
                    state.alive = False
                    flags[index] = True
                    causes[index] = cause
                    locks[index].reset()
                    losses[team_index].append(index)
        return losses

    def _attackable(self, attacker: AircraftState, target: AircraftState) -> bool:
        return attacker.alive and target.alive and self.weapon.attackable(
            engagement_geometry(attacker, target)
        )

    def _select_target(self, attacker: AircraftState, targets: list[AircraftState]) -> int | None:
        candidates = []
        for target_index, target in enumerate(targets):
            if self._attackable(attacker, target):
                geometry = engagement_geometry(attacker, target)
                candidates.append((
                    -engagement_score(
                        geometry, float(self.config["reward"]["engagement_distance_scale"])
                    ),
                    geometry.distance,
                    target_index,
                ))
        return min(candidates)[2] if candidates else None

    def _lock_proposals(
        self,
        attackers: list[AircraftState],
        targets: list[AircraftState],
        locks: list[LockState],
    ) -> list[tuple[int, int]]:
        proposals = []
        for attacker_index, attacker in enumerate(attackers):
            lock = locks[attacker_index]
            if not attacker.alive:
                lock.reset()
                continue
            target_index = lock.current_lock_target
            valid_existing = 0 <= target_index < len(targets) and self._attackable(
                attacker, targets[target_index]
            )
            if valid_existing:
                lock.lock_steps += 1
            else:
                lock.reset()
                target_index = self._select_target(attacker, targets)
                if target_index is not None:
                    lock.current_lock_target = target_index
                    lock.lock_steps = 1
                    if self.first_attackable_step is None:
                        self.first_attackable_step = self.steps
            if lock.current_lock_target >= 0 and lock.lock_steps >= self.weapon.lock_steps_required:
                proposals.append((attacker_index, lock.current_lock_target))
                if self.first_lock_step is None:
                    self.first_lock_step = self.steps
        return proposals

    def _resolve_combat(
        self,
        red_proposals: list[tuple[int, int]],
        blue_proposals: list[tuple[int, int]],
    ) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
        credited: tuple[dict[int, list[int]], dict[int, list[int]]] = ({}, {})
        for team_index, (targets, proposals) in enumerate((
            (self.blue, red_proposals), (self.red, blue_proposals)
        )):
            by_target: dict[int, list[int]] = {}
            for attacker_index, target_index in proposals:
                by_target.setdefault(target_index, []).append(attacker_index)
            for target_index, attacker_indices in sorted(by_target.items()):
                if targets[target_index].alive:
                    targets[target_index].alive = False
                    credited[team_index][target_index] = sorted(set(attacker_indices))
        self.red_attack_kills += len(credited[0])
        self.blue_attack_kills += len(credited[1])
        if (credited[0] or credited[1]) and self.first_kill_step is None:
            self.first_kill_step = self.steps
        return credited

    def _event_rewards(
        self,
        red_boundary_losses: list[int],
        blue_credited: dict[int, list[int]],
        red_credited: dict[int, list[int]],
    ) -> np.ndarray:
        reward_cfg = self.config["reward"]
        rewards = np.zeros(4, dtype=np.float32)
        for target_index, attackers in red_credited.items():
            share = float(reward_cfg["kill_reward"]) / len(attackers)
            rewards[attackers] += share
        for red_index in set(red_boundary_losses) | set(blue_credited):
            rewards[red_index] += float(reward_cfg["death_penalty"])
        return rewards

    def _outcome(self, truncated: bool) -> tuple[bool, bool, bool, str]:
        red_survivors = int(self.red_alive_mask.sum())
        blue_survivors = int(self.blue_alive_mask.sum())
        if red_survivors == 0 and blue_survivors == 0:
            return False, False, True, "draw_mutual_destruction"
        if blue_survivors == 0:
            return True, False, False, "red_win"
        if red_survivors == 0:
            return False, True, False, "blue_win"
        if truncated:
            return False, False, True, "draw_timeout"
        return False, False, False, "ongoing"

    def _info(
        self,
        rewards: np.ndarray,
        shaping_rewards: np.ndarray,
        event_rewards: np.ndarray,
        actions: np.ndarray,
        truncated: bool,
    ) -> dict[str, Any]:
        red_win, blue_win, draw, reason = self._outcome(truncated)
        red_survivors = int(self.red_alive_mask.sum())
        blue_survivors = int(self.blue_alive_mask.sum())
        return {
            "red_success": red_win,
            "red_win": red_win,
            "blue_win": blue_win,
            "draw": draw,
            "termination_reason": reason,
            "red_attack_kills": self.red_attack_kills,
            "blue_attack_kills": self.blue_attack_kills,
            "red_boundary_losses": self.red_boundary_losses,
            "blue_boundary_losses": self.blue_boundary_losses,
            "red_horizontal_boundary_losses": self._boundary_cause_count(
                self.red_boundary_causes, "horizontal"
            ),
            "blue_horizontal_boundary_losses": self._boundary_cause_count(
                self.blue_boundary_causes, "horizontal"
            ),
            "red_low_altitude_losses": self._boundary_cause_count(
                self.red_boundary_causes, "altitude_low"
            ),
            "blue_low_altitude_losses": self._boundary_cause_count(
                self.blue_boundary_causes, "altitude_low"
            ),
            "red_high_altitude_losses": self._boundary_cause_count(
                self.red_boundary_causes, "altitude_high"
            ),
            "blue_high_altitude_losses": self._boundary_cause_count(
                self.blue_boundary_causes, "altitude_high"
            ),
            "red_losses": 4 - red_survivors,
            "blue_losses": 4 - blue_survivors,
            "red_survivors": red_survivors,
            "blue_survivors": blue_survivors,
            "episode_length": self.steps,
            "first_attackable_step": self.first_attackable_step,
            "first_lock_step": self.first_lock_step,
            "first_kill_step": self.first_kill_step,
            "local_rewards": rewards.copy(),
            "shaping_rewards": shaping_rewards.copy(),
            "event_rewards": event_rewards.copy(),
            "executed_red_actions": actions.copy(),
            "red_alive_mask": self.red_alive_mask,
            "blue_alive_mask": self.blue_alive_mask,
        }

    def step(
        self, red_actions: np.ndarray, blue_actions: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:
        red_actions = np.asarray(red_actions, dtype=np.float32)
        if red_actions.shape != (4, 3) or not np.all(np.isfinite(red_actions)):
            raise ValueError("red_actions must be finite with shape (4, 3)")
        if blue_actions is None:
            blue_actions = self.fixed_policy.team_actions(self.blue, self.red)
        blue_actions = np.asarray(blue_actions, dtype=np.float32)
        if blue_actions.shape != (4, 3) or not np.all(np.isfinite(blue_actions)):
            raise ValueError("blue_actions must be finite with shape (4, 3)")

        distance_scale = float(self.config["reward"]["engagement_distance_scale"])
        current_potential = team_potentials(self.red, self.blue, distance_scale)
        executed_red = np.clip(red_actions, -1.0, 1.0) * self.red_alive_mask[:, None]
        executed_blue = np.clip(blue_actions, -1.0, 1.0) * self.blue_alive_mask[:, None]
        self._advance(self.red, executed_red)
        self._advance(self.blue, executed_blue)
        self.steps += 1
        red_boundary, _ = self._resolve_boundaries()

        red_snapshot, blue_snapshot = self._snapshot(self.red), self._snapshot(self.blue)
        red_proposals = self._lock_proposals(red_snapshot, blue_snapshot, self.red_locks)
        blue_proposals = self._lock_proposals(blue_snapshot, red_snapshot, self.blue_locks)
        red_credited, blue_credited = self._resolve_combat(red_proposals, blue_proposals)

        event_rewards = self._event_rewards(red_boundary, blue_credited, red_credited)
        next_potential = team_potentials(self.red, self.blue, distance_scale)
        shaping_rewards = float(self.config["reward"]["shaping_lambda"]) * (
            next_potential - current_potential
        )
        rewards = event_rewards + shaping_rewards
        red_survivors, blue_survivors = self.red_alive_mask.sum(), self.blue_alive_mask.sum()
        terminated = bool(red_survivors == 0 or blue_survivors == 0)
        truncated = bool(not terminated and self.steps >= self.max_steps)
        observations = self._observations()
        return observations, rewards.astype(np.float32), terminated, truncated, self._info(
            rewards, shaping_rewards, event_rewards, executed_red, truncated
        )


__all__ = ["MultiUAVCombatEnv"]
