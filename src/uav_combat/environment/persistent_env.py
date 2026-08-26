"""Minimal persistent-wave variant of the frozen V2.3 combat environment."""
from __future__ import annotations

from typing import Any
import numpy as np

from ..math_utils import wrap_angle
from ..models import AircraftState
from .env import MultiUAVCombatEnv
from .weapon import FireState


PERSISTENT_WAVE_VARIANT = "persistent_wave_v1"


class PersistentWaveCombatEnv(MultiUAVCombatEnv):
    """Keep Red state and immediately replace each defeated Blue wave."""

    environment_variant = PERSISTENT_WAVE_VARIANT

    def __init__(self, config: Any = "configs/persistent_wave_environment.yaml") -> None:
        super().__init__(config)
        if self.config.get("environment_variant") != self.environment_variant:
            raise ValueError(
                f"environment_variant must be {self.environment_variant!r}"
            )
        wave_config = self.config.get("persistent_waves")
        if not isinstance(wave_config, dict):
            raise ValueError("persistent_waves configuration is required")
        self.total_waves = int(wave_config["total_waves"])
        self.spawn_radius = float(wave_config["spawn_radius"])
        self.spawn_direction_count = int(wave_config["spawn_direction_count"])
        if self.total_waves < 1:
            raise ValueError("total_waves must be positive")
        if not 0.0 < self.spawn_radius < self.arena_radius:
            raise ValueError("spawn_radius must be inside the arena")
        if self.spawn_direction_count < 1:
            raise ValueError("spawn_direction_count must be positive")
        maximum_offset = max(map(abs, self.config["scenario"]["formation_offsets"]))
        if np.hypot(self.spawn_radius, maximum_offset) >= self.arena_radius:
            raise ValueError("configured Blue formation does not fit inside arena")
        self.wave_index = 1
        self.waves_cleared = 0
        self.last_spawn_candidate_index: int | None = None
        self.last_minimum_spawn_distance: float | None = None
        self.wave_records: list[dict[str, Any]] = []
        self._wave_start_step = 0
        self._wave_start_red_survivors = self.team_size
        self._wave_start_counts: dict[str, dict[str, int]] = {}
        self._wave_start_rewards: dict[str, float] = {}
        self._wave_record_open = False

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = super().reset(seed)
        self.wave_index = 1
        self.waves_cleared = 0
        self.last_spawn_candidate_index = None
        self.last_minimum_spawn_distance = None
        self.wave_records = []
        self._begin_wave_record()
        info.update(self._wave_info(False, False, None))
        return observation, info

    def _begin_wave_record(self) -> None:
        self._wave_record_open = True
        self._wave_start_step = self.steps
        self._wave_start_red_survivors = int(self.red_alive_mask.sum())
        self._wave_start_counts = {
            side: dict(counts) for side, counts in self.combat_counts.items()
        }
        self._wave_start_rewards = {
            name: float(values.sum())
            for name, values in self.episode_reward_components.items()
        }

    def _finish_wave_record(
        self, wave_cleared: bool, termination_reason: str
    ) -> None:
        if not self._wave_record_open:
            return
        record: dict[str, Any] = {
            "wave_index": self.wave_index,
            "start_step": self._wave_start_step,
            "end_step": self.steps,
            "duration_steps": self.steps - self._wave_start_step,
            "red_survivors_start": self._wave_start_red_survivors,
            "red_survivors_end": int(self.red_alive_mask.sum()),
            "blue_survivors_start": self.team_size,
            "blue_survivors_end": int(self.blue_alive_mask.sum()),
            "wave_completed": True,
            "wave_cleared": bool(wave_cleared),
            "termination_reason": termination_reason,
        }
        for side in ("red", "blue"):
            for event in (
                "fire_attempts", "weapon_hits", "attack_kills",
                "boundary_exits", "ground_losses",
            ):
                record[f"{side}_{event}"] = (
                    self.combat_counts[side][event]
                    - self._wave_start_counts[side][event]
                )
        team_return = 0.0
        for name, values in self.episode_reward_components.items():
            value = float(values.sum()) - self._wave_start_rewards[name]
            record[f"{name}_total"] = value
            team_return += value
        record["team_return"] = team_return
        self.wave_records.append(record)
        self._wave_record_open = False

    def _candidate_blue_wave(self, radial_angle: float) -> list[AircraftState]:
        scenario = self.config["scenario"]
        radial = np.array([np.cos(radial_angle), np.sin(radial_angle)])
        lateral = np.array([-radial[1], radial[0]])
        center = self.spawn_radius * radial
        alive_red = [state for state in self.red if state.alive]
        red_center = np.mean([[state.x, state.y] for state in alive_red], axis=0)
        nominal_heading = float(np.arctan2(
            red_center[1] - center[1], red_center[0] - center[0]
        ))
        states = []
        for offset in scenario["formation_offsets"]:
            position = center + float(offset) * lateral
            states.append(AircraftState(
                x=float(position[0]),
                y=float(position[1]),
                z=-float(scenario["altitude_center"] + self.rng.uniform(
                    -scenario["altitude_perturbation_max"],
                    scenario["altitude_perturbation_max"],
                )),
                v=float(scenario["speed_center"] + self.rng.uniform(
                    -scenario["speed_perturbation_max"],
                    scenario["speed_perturbation_max"],
                )),
                theta=0.0,
                psi=float(wrap_angle(nominal_heading + self.rng.uniform(
                    -scenario["heading_perturbation_max"],
                    scenario["heading_perturbation_max"],
                ))),
            ))
        return states

    def _blue_wave_inside_arena(self, states: list[AircraftState]) -> bool:
        return bool(
            len(states) == self.team_size
            and all(np.hypot(state.x, state.y) < self.arena_radius for state in states)
        )

    def _minimum_red_blue_distance(self, states: list[AircraftState]) -> float:
        alive_red = [state for state in self.red if state.alive]
        if len(states) != self.team_size or not alive_red:
            raise RuntimeError("minimum spawn distance requires surviving Red")
        return min(
            float(np.linalg.norm(np.array([
                blue.x - red.x, blue.y - red.y, blue.z - red.z,
            ])))
            for blue in states for red in alive_red
        )

    def _spawn_next_wave(self) -> float:
        alive_red = [state for state in self.red if state.alive]
        if not alive_red:
            raise RuntimeError("cannot spawn a wave after Red elimination")
        best_index = 0
        best_angle = -np.pi
        best_distance = -np.inf
        best_candidate: list[AircraftState] | None = None
        for index, radial_angle in enumerate(np.linspace(
            -np.pi, np.pi, self.spawn_direction_count, endpoint=False
        )):
            radial_angle = float(radial_angle)
            candidate = self._candidate_blue_wave(radial_angle)
            distance = self._minimum_red_blue_distance(candidate)
            if distance > best_distance:
                best_index = index
                best_angle = radial_angle
                best_distance = distance
                best_candidate = candidate
        # __init__ proves every enumerated formation fits inside the arena.
        assert best_candidate is not None and self._blue_wave_inside_arena(best_candidate)
        self.blue = best_candidate
        self.red_fire_states = [FireState() for _ in range(self.team_size)]
        self.blue_fire_states = [FireState() for _ in range(self.team_size)]
        self.blue_last_executed_phi.fill(0.0)
        self.last_spawn_candidate_index = best_index
        self.last_minimum_spawn_distance = best_distance
        return best_angle

    def _wave_info(
        self,
        wave_cleared: bool,
        spawned_next_wave: bool,
        spawn_radial_angle: float | None,
    ) -> dict[str, Any]:
        return {
            "environment_variant": self.environment_variant,
            "wave_index": self.wave_index,
            "total_waves": self.total_waves,
            "waves_cleared": self.waves_cleared,
            "wave_cleared_this_step": wave_cleared,
            "spawned_next_wave": spawned_next_wave,
            "wave_spawn_radial_angle": spawn_radial_angle,
            "wave_spawn_candidate_index": (
                self.last_spawn_candidate_index if spawned_next_wave else None
            ),
            "minimum_spawn_distance": (
                self.last_minimum_spawn_distance if spawned_next_wave else None
            ),
            "per_wave_metrics": [dict(record) for record in self.wave_records],
        }

    def step(
        self, red_actions: np.ndarray, blue_actions: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = super().step(
            red_actions, blue_actions
        )
        red_survivors = int(self.red_alive_mask.sum())
        wave_cleared = bool(red_survivors > 0 and self.blue_alive_mask.sum() == 0)
        spawned_next_wave = False
        spawn_radial_angle = None

        if wave_cleared:
            if self.wave_index < self.total_waves and self.steps >= self.max_steps:
                clear_reason = "red_failure_timeout"
            elif self.wave_index == self.total_waves:
                clear_reason = str(info["termination_reason"])
            else:
                clear_reason = "wave_cleared"
            self._finish_wave_record(True, clear_reason)
            self.waves_cleared += 1
            if self.wave_index < self.total_waves:
                if self.steps >= self.max_steps:
                    terminated = False
                    truncated = True
                    info.update({
                        "red_success": False,
                        "red_win": False,
                        "blue_win": False,
                        "draw": False,
                        "termination_reason": "red_failure_timeout",
                    })
                else:
                    spawn_radial_angle = self._spawn_next_wave()
                    self.wave_index += 1
                    self._begin_wave_record()
                    spawned_next_wave = True
                    terminated = False
                    truncated = False
                    observation = self._observations()
                    info.update({
                        "red_success": False,
                        "red_win": False,
                        "blue_win": False,
                        "draw": False,
                        "termination_reason": "ongoing",
                    })

        if (terminated or truncated) and self._wave_record_open:
            self._finish_wave_record(False, str(info["termination_reason"]))

        current_blue_losses = self.team_size - int(self.blue_alive_mask.sum())
        info["blue_losses"] = (
            self.waves_cleared * self.team_size
            if wave_cleared
            else self.waves_cleared * self.team_size + current_blue_losses
        )
        # super().step() built info before an intermediate replacement. Always
        # overwrite state-derived fields so returned masks match observation.
        info.update({
            "red_survivors": int(self.red_alive_mask.sum()),
            "blue_survivors": int(self.blue_alive_mask.sum()),
            "red_alive_mask": self.red_alive_mask,
            "blue_alive_mask": self.blue_alive_mask,
        })
        info.update(self._wave_info(
            wave_cleared, spawned_next_wave, spawn_radial_angle
        ))
        return observation, reward, terminated, truncated, info


__all__ = ["PERSISTENT_WAVE_VARIANT", "PersistentWaveCombatEnv"]
