"""Single authoritative 4-red-vs-4-blue paper reproduction environment."""
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
from .observation import OBSERVATION_DIM, build_observation
from .reward import equation25_reward
from .scenario import random_diameter_states
from .sensor import SensorModel
from .weapon import WeaponModel


class PaperUAVCombatEnv:
    team_size = 4
    observation_dim = OBSERVATION_DIM
    action_dim = 3

    def __init__(self, config: str | Path | dict[str, Any] = "configs/paper_environment.yaml", sensor_noise: bool | None = None) -> None:
        self.config = load_config(config) if not isinstance(config, dict) else config
        self.spec = aircraft_spec(self.config)
        self.dt = float(self.config["simulation"]["dt"])
        self.max_steps = int(self.config["simulation"]["max_steps"])
        self.radius = float(self.config["battlefield"]["diameter"]) / 2.0
        assumptions = self.config["reproduction_assumptions"]
        sensor_cfg = assumptions["sensor"]
        enabled = sensor_cfg["enabled"] if sensor_noise is None else sensor_noise
        self.sensor = SensorModel(**{**sensor_cfg, "enabled": enabled})
        weapon = self.config["weapon"] | assumptions["weapon"]
        self.weapon = WeaponModel(**weapon)
        self.controller = TargetStateController()
        self.dynamics = PointMassDynamics()
        self.integrator = RK4Integrator(self.dt)
        self.blue_policy = NearestTargetPursuitPolicy(desired_speed=assumptions["fixed_policy_desired_speed"])
        self.scenario_cfg = assumptions["formation"]
        self.obs_cfg = assumptions["observation_normalization"]
        self.rng = np.random.default_rng()
        self.red: list[AircraftState] = []
        self.blue: list[AircraftState] = []
        self.red_phi = np.zeros(4)
        self.blue_phi = np.zeros(4)
        self.steps = 0
        self.red_attack_kills = 0
        self.blue_attack_kills = 0
        self.red_boundary_losses = 0
        self.blue_boundary_losses = 0

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        self.rng = np.random.default_rng(seed)
        self.red, self.blue, diameter_angle = random_diameter_states(self.rng, **self.scenario_cfg)
        self.red_phi[:] = 0.0; self.blue_phi[:] = 0.0
        self.steps = self.red_attack_kills = self.blue_attack_kills = 0
        self.red_boundary_losses = self.blue_boundary_losses = 0
        return self._observations(), {"diameter_angle": diameter_angle, "red_ids": [f"red_{i}" for i in range(4)], "blue_ids": [f"blue_{i}" for i in range(4)]}

    def _observed(self) -> tuple[list, list]:
        return (
            [self.sensor.observe(s, self.red_phi[i], self.rng) for i, s in enumerate(self.red)],
            [self.sensor.observe(s, self.blue_phi[i], self.rng) for i, s in enumerate(self.blue)],
        )

    def _observations(self) -> np.ndarray:
        red_o, blue_o = self._observed()
        ra, ba = [s.alive for s in self.red], [s.alive for s in self.blue]
        return np.stack([build_observation(i, red_o, blue_o, ra, ba, **self.obs_cfg) for i in range(4)])

    def _advance(self, states: list[AircraftState], phis: np.ndarray, actions: np.ndarray) -> None:
        for i, state in enumerate(states):
            if not state.alive:
                continue
            _, control = self.controller.control_from_action(state, actions[i], self.spec)
            states[i] = self.integrator.step(state, control, self.dynamics, self.spec)
            phis[i] = control.phi

    def _boundary(self, states: list[AircraftState]) -> list[int]:
        losses = []
        for i, state in enumerate(states):
            if state.alive and np.hypot(state.x, state.y) > self.radius:
                state.alive = False; losses.append(i)
        return losses

    def _attacks(self) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """Resolve simultaneous automatic fire; target choice is an assumption."""
        proposals: list[tuple[str, int, int]] = []
        for team, attackers, targets in (("red", self.red, self.blue), ("blue", self.blue, self.red)):
            for i, attacker in enumerate(attackers):
                target_i = self.blue_policy.nearest_target_index(attacker, targets) if attacker.alive else None
                if target_i is None:
                    continue
                g = compute_paper_geometry(attacker, targets[target_i])
                if self.weapon.can_fire(g) and self.weapon.sample_hit(g, self.rng):
                    proposals.append((team, i, target_i))
        red_hits, blue_hits = [], []
        for team, attacker_i, target_i in proposals:
            target = self.blue[target_i] if team == "red" else self.red[target_i]
            if target.alive:
                target.alive = False
                (red_hits if team == "red" else blue_hits).append((attacker_i, target_i))
        return red_hits, blue_hits

    def step(self, red_actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:
        red_actions = np.asarray(red_actions, dtype=np.float32)
        if red_actions.shape != (4, 3) or not np.all(np.isfinite(red_actions)):
            raise ValueError("red_actions must be finite with shape (4, 3)")
        blue_actions = np.stack([self.blue_policy.action(s, self.red)[0] for s in self.blue])
        self._advance(self.red, self.red_phi, np.clip(red_actions, -1.0, 1.0))
        self._advance(self.blue, self.blue_phi, blue_actions)
        red_boundary = self._boundary(self.red); blue_boundary = self._boundary(self.blue)
        self.red_boundary_losses += len(red_boundary); self.blue_boundary_losses += len(blue_boundary)
        red_hits, blue_hits = self._attacks()
        self.red_attack_kills += len(red_hits); self.blue_attack_kills += len(blue_hits)
        local = np.zeros(4, dtype=np.float32)
        for i, state in enumerate(self.red):
            target_i = self.blue_policy.nearest_target_index(state, self.blue) if state.alive else None
            red_g = compute_paper_geometry(state, self.blue[target_i]) if target_i is not None else None
            threat_candidates = [(j, compute_paper_geometry(b, state)) for j, b in enumerate(self.blue) if b.alive and state.alive]
            blue_g = min(threat_candidates, key=lambda x: x[1].distance)[1] if threat_candidates else None
            local[i] = equation25_reward(
                red_g, blue_g,
                destroyed_blue=sum(a == i for a, _ in red_hits),
                red_destroyed=(i in [t for _, t in blue_hits]),
                red_boundary_loss=(i in red_boundary),
            )
        team_reward = float(local.sum())
        rewards = np.full(4, team_reward, dtype=np.float32)
        self.steps += 1
        no_blue = not any(s.alive for s in self.blue)
        no_red = not any(s.alive for s in self.red)
        success = self.red_attack_kills == 4
        terminated = no_blue or no_red
        truncated = self.steps >= self.max_steps and not terminated
        info = {
            "win": success,
            "environment_outcome": "red_win" if success else ("mission_failure" if terminated or truncated else "ongoing"),
            "attack_kills": self.red_attack_kills,
            "red_attack_kills": self.red_attack_kills,
            "blue_attack_kills": self.blue_attack_kills,
            "boundary_losses": self.red_boundary_losses + self.blue_boundary_losses,
            "red_boundary_losses": self.red_boundary_losses,
            "blue_boundary_losses": self.blue_boundary_losses,
            "red_survivors": sum(s.alive for s in self.red),
            "blue_survivors": sum(s.alive for s in self.blue),
            "episode_length": self.steps,
            "local_rewards": local,
            "blue_target_indices": [self.blue_policy.nearest_target_index(s, self.red) for s in self.blue],
        }
        return self._observations(), rewards, terminated, truncated, info
