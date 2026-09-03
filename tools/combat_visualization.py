"""Read-only helpers for recording and rendering one combat episode."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np

from env.models import AircraftState
from env.persistent_env import PersistentWaveCombatEnv

TRACE_SCHEMA_VERSION = 1
FEATURE_NAMES = ["x", "y", "z", "v", "theta", "psi"]
RESERVED_SEED_RANGES = ((29_000_000, 29_000_019), (30_000_000, 30_000_199))


def state_array(state: AircraftState) -> np.ndarray:
    return np.asarray(state.as_array(), dtype=np.float32)


def states_array(states: list[AircraftState], count: int = 4) -> np.ndarray:
    result = np.full((count, 6), np.nan, dtype=np.float32)
    for index, state in enumerate(states[:count]):
        result[index] = state_array(state)
    return result


class RecordingPersistentWaveCombatEnv(PersistentWaveCombatEnv):
    """Environment-semantics-neutral observer for pre-respawn Blue states."""

    def reset(self, seed: int | None = None):
        self.visualization_spawn_snapshot = None
        return super().reset(seed)

    def _spawn_next_wave(self) -> float:
        self.visualization_spawn_snapshot = {
            "old_wave_index": int(self.wave_index),
            "step": int(self.steps),
            "old_blue": [state.copy() for state in self.blue],
            "red": [state.copy() for state in self.red],
        }
        return super()._spawn_next_wave()


def assert_episode_seed_allowed(seed: int) -> None:
    value = int(seed)
    if any(start <= value <= end for start, end in RESERVED_SEED_RANGES):
        raise ValueError(f"episode seed {value} is reserved for formal protocol")


def ensure_fresh_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory is non-empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def checkpoint_sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_frame(frames: dict[str, list[Any]], env: RecordingPersistentWaveCombatEnv,
                 active_wave: int, waves_cleared: int, time_s: float,
                 info: dict[str, Any] | None = None) -> None:
    red = states_array(env.red)
    blue = np.full((env.total_waves, 4, 6), np.nan, dtype=np.float32)
    blue_alive = np.zeros((env.total_waves, 4), dtype=bool)
    wave = max(1, min(env.total_waves, int(active_wave)))
    blue[wave - 1] = states_array(env.blue)
    blue_alive[wave - 1] = [state.alive for state in env.blue]
    snapshot = getattr(env, "visualization_spawn_snapshot", None)
    if snapshot is not None and int(snapshot["step"]) == int(env.steps):
        old = int(snapshot["old_wave_index"]) - 1
        blue[old] = states_array(snapshot["old_blue"])
        blue_alive[old] = [state.alive for state in snapshot["old_blue"]]
    frames["red_kinematics"].append(red)
    frames["red_alive"].append(np.asarray([s.alive for s in env.red], dtype=bool))
    frames["blue_kinematics"].append(blue)
    frames["blue_alive"].append(blue_alive)
    frames["steps"].append(int(env.steps))
    frames["time_s"].append(float(time_s))
    frames["active_wave"].append(int(wave))
    frames["waves_cleared"].append(int(waves_cleared))


def frame_arrays(frames: dict[str, list[Any]]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value) for key, value in frames.items()}


def write_trace(path: Path, frames: dict[str, list[Any]], transitions: dict[str, list[Any]]) -> dict[str, tuple[int, ...]]:
    arrays = frame_arrays(frames)
    arrays.update({key: np.asarray(value) for key, value in transitions.items()})
    np.savez_compressed(path, **arrays)
    return {key: tuple(value.shape) for key, value in arrays.items()}


def read_trace(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        result = {key: source[key].copy() for key in source.files}
    required = {"red_kinematics", "red_alive", "blue_kinematics", "blue_alive", "steps", "time_s", "active_wave", "waves_cleared"}
    missing = required - result.keys()
    if missing or result["red_kinematics"].ndim != 3 or result["blue_kinematics"].ndim != 4:
        raise ValueError(f"invalid trace schema; missing={sorted(missing)}")
    if result["red_kinematics"].shape[0] != result["steps"].shape[0]:
        raise ValueError("trace frame count mismatch")
    return result


def dump_metadata(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


__all__ = ["TRACE_SCHEMA_VERSION", "FEATURE_NAMES", "RecordingPersistentWaveCombatEnv", "assert_episode_seed_allowed", "ensure_fresh_output", "state_array", "states_array", "append_frame", "write_trace", "read_trace", "checkpoint_sha256", "dump_metadata"]
