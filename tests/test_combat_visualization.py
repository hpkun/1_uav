from pathlib import Path
import json
import numpy as np
import pytest

from env.models import AircraftState
from env.persistent_env import PersistentWaveCombatEnv
from tools.combat_visualization import (TRACE_SCHEMA_VERSION, RecordingPersistentWaveCombatEnv,
    append_frame, assert_episode_seed_allowed, ensure_fresh_output, read_trace, states_array, write_trace)

ROOT = Path(__file__).resolve().parents[1]


def same_state(left, right):
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same_state(left[key], right[key]) for key in left)
    if isinstance(left, np.ndarray):
        return np.array_equal(left, right)
    return left == right


def test_aircraft_state_and_trace_shapes():
    state = AircraftState(1, 2, -3, 4, 5, 6, True)
    result = states_array([state])
    assert np.array_equal(result[0], [1, 2, -3, 4, 5, 6])
    assert np.all(np.isnan(result[1:]))
    env = RecordingPersistentWaveCombatEnv(ROOT / "configs/persistent_wave_v2_environment.yaml")
    env.reset(40000000)
    frames = {key: [] for key in ("red_kinematics", "red_alive", "blue_kinematics", "blue_alive", "steps", "time_s", "active_wave", "waves_cleared")}
    append_frame(frames, env, 1, 0, 0.0)
    arrays = write_trace(ROOT / "tests" / "_temporary_trace.npz", frames, {})
    try:
        trace = read_trace(ROOT / "tests" / "_temporary_trace.npz")
        assert trace["red_kinematics"].shape == (1, 4, 6)
        assert trace["blue_kinematics"].shape == (1, 3, 4, 6)
        assert np.all(np.isnan(trace["blue_kinematics"][0, 1:]))
        assert np.all(~trace["blue_alive"][0, 1:])
    finally:
        (ROOT / "tests" / "_temporary_trace.npz").unlink(missing_ok=True)


def test_reserved_seed_guard_and_fresh_output(tmp_path):
    for seed in (29000000, 30000199):
        with pytest.raises(ValueError): assert_episode_seed_allowed(seed)
    assert_episode_seed_allowed(40000000)
    ensure_fresh_output(tmp_path / "new")
    (tmp_path / "new" / "x").write_text("x")
    with pytest.raises(FileExistsError): ensure_fresh_output(tmp_path / "new")


def test_visualization_subclass_captures_pre_spawn_terminal_state():
    env = RecordingPersistentWaveCombatEnv(ROOT / "configs/persistent_wave_v2_environment.yaml")
    env.reset(40000000)
    old = [state.copy() for state in env.blue]
    env._spawn_next_wave()
    snapshot = env.visualization_spawn_snapshot
    assert snapshot["step"] == env.steps and snapshot["old_wave_index"] == 1
    assert np.array_equal(states_array(snapshot["old_blue"]), states_array(old))


def test_visualization_observer_is_spawn_semantics_neutral():
    base = PersistentWaveCombatEnv(ROOT / "configs/persistent_wave_v2_environment.yaml")
    observed = RecordingPersistentWaveCombatEnv(ROOT / "configs/persistent_wave_v2_environment.yaml")
    base.reset(40000000); observed.reset(40000000)
    assert np.array_equal(states_array(base.red), states_array(observed.red))
    assert np.array_equal(states_array(base.blue), states_array(observed.blue))
    base_angle = base._spawn_next_wave(); observed_angle = observed._spawn_next_wave()
    assert base_angle == observed_angle
    assert base.last_spawn_candidate_index == observed.last_spawn_candidate_index
    assert base.last_minimum_spawn_distance == observed.last_minimum_spawn_distance
    assert np.array_equal(states_array(base.red), states_array(observed.red))
    assert np.array_equal(states_array(base.blue), states_array(observed.blue))
    assert same_state(base.rng.bit_generator.state, observed.rng.bit_generator.state)


def test_altitude_rule_is_ned_negation():
    state = AircraftState(0, 0, -123, 1, 0, 0)
    assert -state.z == 123
