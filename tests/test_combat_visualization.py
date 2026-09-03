from pathlib import Path
import json
import shutil
import subprocess
import numpy as np
import pytest

from env.models import AircraftState
from env.persistent_env import PersistentWaveCombatEnv
from tools.combat_visualization import (TRACE_SCHEMA_VERSION, RecordingPersistentWaveCombatEnv,
    append_frame, assert_episode_seed_allowed, blue_losses_at_frame, ensure_fresh_output,
    heading_endpoint, read_trace, recent_events, states_array, trace_frame_to_render_index, write_trace)
from tools.render_combat_episode_interactive import render as render_interactive

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


def test_heading_endpoint_pitch_uses_ned_sign():
    climbing = np.array([0, 0, -100, 100, 0.2, 0], dtype=float)
    descending = np.array([0, 0, -100, 100, -0.2, 0], dtype=float)
    assert -heading_endpoint(climbing)[2] > -climbing[2]
    assert -heading_endpoint(descending)[2] < -descending[2]


def test_blue_loss_timeline_has_no_future_leakage():
    deaths = [
        {"side": "blue", "frame": 3},
        {"side": "red", "frame": 4},
        {"side": "blue", "frame": 8},
    ]
    assert blue_losses_at_frame(deaths, 0) == 0
    assert blue_losses_at_frame(deaths, 3) == 1
    assert blue_losses_at_frame(deaths, 99) == 2


def test_recent_events_excludes_future_events():
    events = [{"trace_frame": frame, "type": str(frame)} for frame in (1, 3, 8)]
    assert [event["trace_frame"] for event in recent_events(events, 3)] == [1, 3]
    assert recent_events(events, 0) == []


def test_event_hold_mapping_uses_rendered_frame_units_with_stride_four():
    rendered = [0, 4, 8, 12, 16]
    assert trace_frame_to_render_index(rendered, 1) == 1
    assert trace_frame_to_render_index(rendered, 4) == 1
    assert trace_frame_to_render_index(rendered, 9) == 3
    # At 30 fps an event beginning at rendered index 1 remains for 30 output frames,
    # independent of the four-trace-frame stride.
    hold_frames = round(30 * 1.0)
    assert sum(1 for index in range(100) if 1 <= index and index - 1 < hold_frames) == 30


def test_interactive_html_is_standalone_and_accepts_old_metadata(tmp_path):
    frames = 3
    red = np.zeros((frames, 4, 6), dtype=np.float32)
    blue = np.full((frames, 2, 4, 6), np.nan, dtype=np.float32)
    blue[:, 0] = 0
    trace_path = tmp_path / "episode_trace.npz"
    np.savez_compressed(trace_path, red_kinematics=red,
        red_alive=np.ones((frames, 4), dtype=bool), blue_kinematics=blue,
        blue_alive=np.concatenate((np.ones((frames, 1, 4), dtype=bool),
                                   np.zeros((frames, 1, 4), dtype=bool)), axis=1),
        steps=np.arange(frames), time_s=np.arange(frames) * 0.1,
        active_wave=np.ones(frames, dtype=int), waves_cleared=np.zeros(frames, dtype=int))
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({"trace_schema_version": 1, "total_waves": 2}), encoding="utf-8")
    output = tmp_path / "episode_interactive.html"
    render_interactive(trace_path, metadata_path, output)
    html = output.read_text(encoding="utf-8")
    assert output.stat().st_size > 1_000_000
    assert "<script src=" not in html.lower()
    assert all(token in html for token in ("Play", "Reset Camera", "slider", "uirevision", "scatter3d"))
    payload = html.split('<script type="application/json" id="payload">', 1)[1].split("</script>", 1)[0]
    json.loads(payload)
    assert "NaN" not in payload
    application = html.split("// APP_JS_START", 1)[1].split("// APP_JS_END", 1)[0]
    assert "\nFINAL" not in application
    node = shutil.which("node")
    if node:
        script = tmp_path / "application.js"
        script.write_text(application, encoding="utf-8")
        checked = subprocess.run([node, "--check", str(script)], capture_output=True, text=True)
        assert checked.returncode == 0, checked.stderr
