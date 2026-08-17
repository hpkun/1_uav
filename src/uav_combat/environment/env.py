"""Minimal synchronous 4-red-vs-4-blue environment from Li et al. (2023)."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np

from ..config import aircraft_spec, load_config
from ..controller import TargetStateController
from ..dynamics import PointMassDynamics
from ..integrator import RK4Integrator
from ..models import AircraftState
from .fixed_policy import NearestTargetPursuitPolicy
from .geometry import compute_paper_geometry
from .observation import OBSERVATION_DIM, build_team_observations
from .reward import equation25_event_reward, equation25_geometric_reward
from .scenario import random_diameter_states
from .sensor import SensorModel
from .weapon import WeaponModel


class PaperUAVCombatEnv:
    """Paper scenario: learned red team against nearest-target fixed blue."""

    team_size, observation_dim, action_dim = 4, OBSERVATION_DIM, 3

    def __init__(self, config: str | Path | dict[str, Any] = "configs/paper_environment.yaml", sensor_noise: bool | None = None) -> None:
        self.config = load_config(config) if not isinstance(config, dict) else config
        self.spec = aircraft_spec(self.config)
        self.dt = float(self.config["simulation"]["dt"])
        self.max_steps = int(self.config["simulation"]["max_steps"])
        self.radius = float(self.config["battlefield"]["diameter"]) / 2.0
        assumptions = self.config["reproduction_assumptions"]
        sensor_cfg = dict(assumptions["sensor"])
        configured_noise = bool(sensor_cfg.pop("enabled"))
        self.sensor = SensorModel(**sensor_cfg, enabled=configured_noise if sensor_noise is None else sensor_noise)
        self.weapon = WeaponModel(**(self.config["weapon"] | assumptions["weapon"]))
        action_cfg = self.config["action"]
        self.controller = TargetStateController(
            delta_yaw_max=max(map(abs, action_cfg["delta_psi"])),
            delta_pitch_max=max(map(abs, action_cfg["delta_theta"])),
            delta_speed_max=max(map(abs, action_cfg["delta_v"])),
        )
        self.dynamics = PointMassDynamics()
        self.integrator = RK4Integrator(self.dt)
        self.fixed_policy = NearestTargetPursuitPolicy(
            delta_psi_max=self.controller.delta_yaw_max,
            delta_theta_max=self.controller.delta_pitch_max,
            delta_v_max=self.controller.delta_speed_max,
            desired_speed=assumptions["fixed_policy_desired_speed"],
        )
        self.scenario_cfg = assumptions["formation"]
        self.obs_cfg = assumptions["observation_normalization"]
        self.rng = np.random.default_rng()
        self.red: list[AircraftState] = []
        self.blue: list[AircraftState] = []
        self.red_phi = np.zeros(4)
        self.blue_phi = np.zeros(4)
        self.red_boundary_dead = np.zeros(4, dtype=bool)
        self.blue_boundary_dead = np.zeros(4, dtype=bool)
        self.steps = 0
        self.red_attack_kills = 0
        self.blue_attack_kills = 0

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

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        self.rng = np.random.default_rng(seed)
        self.red, self.blue, diameter_angle = random_diameter_states(self.rng, **self.scenario_cfg)
        self.red_phi.fill(0.0)
        self.blue_phi.fill(0.0)
        self.red_boundary_dead.fill(False)
        self.blue_boundary_dead.fill(False)
        self.steps = self.red_attack_kills = self.blue_attack_kills = 0
        observations = self._observations()
        return observations, {
            "diameter_angle": diameter_angle,
            "red_alive_mask": self.red_alive_mask,
            "blue_alive_mask": self.blue_alive_mask,
        }

    def _observations(self) -> np.ndarray:
        red_observed = [self.sensor.observe(s, self.red_phi[i], self.rng) for i, s in enumerate(self.red)]
        blue_observed = [self.sensor.observe(s, self.blue_phi[i], self.rng) for i, s in enumerate(self.blue)]
        return build_team_observations(
            red_observed, blue_observed,
            [s.alive for s in self.red], [s.alive for s in self.blue],
            **self.obs_cfg,
        )

    def _advance(self, states: list[AircraftState], phis: np.ndarray, actions: np.ndarray) -> None:
        for i, state in enumerate(states):
            if state.alive:
                _, control = self.controller.control_from_action(state, actions[i], self.spec)
                states[i] = self.integrator.step(state, control, self.dynamics, self.spec)
                phis[i] = control.phi

    @staticmethod
    def _kill(states: list[AircraftState], index: int, boundary_flags: np.ndarray, boundary: bool) -> bool:
        if not states[index].alive:
            return False
        states[index].alive = False
        if boundary:
            boundary_flags[index] = True
        return True

    def _resolve_boundaries(self) -> tuple[list[int], list[int]]:
        losses: tuple[list[int], list[int]] = ([], [])
        for team_index, (states, flags) in enumerate(((self.red, self.red_boundary_dead), (self.blue, self.blue_boundary_dead))):
            for i, state in enumerate(states):
                if state.alive and np.hypot(state.x, state.y) > self.radius:
                    if self._kill(states, i, flags, boundary=True):
                        losses[team_index].append(i)
        return losses

    @staticmethod
    def _snapshot(states: list[AircraftState]) -> list[AircraftState]:
        return [state.copy() for state in states]

    def _geometric_rewards(self, red: list[AircraftState], blue: list[AircraftState]) -> tuple[np.ndarray, list[int | None]]:
        rewards = np.zeros(4, dtype=np.float32)
        target_indices: list[int | None] = []
        for i, red_state in enumerate(red):
            target_index = self.fixed_policy.nearest_target_index(red_state, blue) if red_state.alive else None
            target_indices.append(target_index)
            if target_index is not None:
                blue_state = blue[target_index]
                rewards[i] = equation25_geometric_reward(
                    compute_paper_geometry(red_state, blue_state),
                    compute_paper_geometry(blue_state, red_state),
                )
        return rewards, target_indices

    def _hit_proposals(self, red: list[AircraftState], blue: list[AircraftState]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        proposals: tuple[list[tuple[int, int]], list[tuple[int, int]]] = ([], [])
        for team_index, (attackers, targets) in enumerate(((red, blue), (blue, red))):
            for attacker_index, attacker in enumerate(attackers):
                target_index = self.fixed_policy.nearest_target_index(attacker, targets) if attacker.alive else None
                if target_index is None:
                    continue
                geometry = compute_paper_geometry(attacker, targets[target_index])
                if self.weapon.can_fire(geometry) and self.weapon.sample_hit(geometry, self.rng):
                    proposals[team_index].append((attacker_index, target_index))
        return proposals

    def _apply_simultaneous_hits(self, red_hit_proposals: list[tuple[int, int]], blue_hit_proposals: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """Apply successful proposals from one shared pre-attack snapshot."""
        credited: tuple[list[tuple[int, int]], list[tuple[int, int]]] = ([], [])
        for team_index, (targets, flags, proposals) in enumerate((
            (self.blue, self.blue_boundary_dead, red_hit_proposals),
            (self.red, self.red_boundary_dead, blue_hit_proposals),
        )):
            by_target: dict[int, list[int]] = {}
            for attacker, target in proposals:
                by_target.setdefault(target, []).append(attacker)
            for target, attackers in sorted(by_target.items()):
                if self._kill(targets, target, flags, boundary=False):
                    credited[team_index].append((min(attackers), target))
        self.red_attack_kills += len(credited[0])
        self.blue_attack_kills += len(credited[1])
        return credited

    def _termination_reason(self, truncated: bool) -> str:
        if not any(state.alive for state in self.blue):
            return "all_blue_destroyed"
        if not any(state.alive for state in self.red):
            return "all_red_destroyed"
        return "timeout" if truncated else "ongoing"

    def _info(self, local_rewards: np.ndarray, executed_actions: np.ndarray, reward_targets: list[int | None], truncated: bool) -> dict[str, Any]:
        red_survivors = int(self.red_alive_mask.sum())
        blue_survivors = int(self.blue_alive_mask.sum())
        red_success = blue_survivors == 0
        episode_done = red_success or red_survivors == 0 or truncated
        return {
            "red_success": red_success,
            "red_win": red_success,
            "blue_win": bool(episode_done and not red_success),
            "termination_reason": self._termination_reason(truncated),
            "red_attack_kills": self.red_attack_kills,
            "blue_attack_kills": self.blue_attack_kills,
            "red_boundary_losses": self.red_boundary_losses,
            "blue_boundary_losses": self.blue_boundary_losses,
            "red_losses": 4 - red_survivors,
            "red_survivors": red_survivors,
            "blue_survivors": blue_survivors,
            "episode_length": self.steps,
            "local_rewards": local_rewards,
            "executed_red_actions": executed_actions,
            "reward_target_indices": reward_targets,
            "red_alive_mask": self.red_alive_mask,
            "blue_alive_mask": self.blue_alive_mask,
        }

    def step(self, red_actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:
        red_actions = np.asarray(red_actions, dtype=np.float32)
        if red_actions.shape != (4, 3) or not np.all(np.isfinite(red_actions)):
            raise ValueError("red_actions must be finite with shape (4, 3)")
        blue_actions = np.stack([self.fixed_policy.action(state, self.red)[0] for state in self.blue])
        executed_red = np.clip(red_actions, -1.0, 1.0) * self.red_alive_mask[:, None]
        executed_blue = np.clip(blue_actions, -1.0, 1.0) * self.blue_alive_mask[:, None]
        self._advance(self.red, self.red_phi, executed_red)
        self._advance(self.blue, self.blue_phi, executed_blue)
        red_boundary, _ = self._resolve_boundaries()
        red_snapshot, blue_snapshot = self._snapshot(self.red), self._snapshot(self.blue)
        local_rewards, reward_targets = self._geometric_rewards(red_snapshot, blue_snapshot)
        red_proposals, blue_proposals = self._hit_proposals(red_snapshot, blue_snapshot)
        red_hits, blue_hits = self._apply_simultaneous_hits(red_proposals, blue_proposals)
        for i in range(4):
            local_rewards[i] += equation25_event_reward(
                destroyed_blue=sum(attacker == i for attacker, _ in red_hits),
                red_attack_death=any(target == i for _, target in blue_hits),
                red_boundary_death=i in red_boundary,
            )
        rewards = np.full(4, float(local_rewards.sum()), dtype=np.float32)
        self.steps += 1
        terminated = not any(s.alive for s in self.red) or not any(s.alive for s in self.blue)
        truncated = self.steps >= self.max_steps and not terminated
        return self._observations(), rewards, terminated, truncated, self._info(local_rewards, executed_red, reward_targets, truncated)
