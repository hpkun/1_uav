from dataclasses import asdict
import shutil
import subprocess

import numpy as np
import pytest

from algorithm.interception import episode, trace_row
from algorithm.common.serialization import write_json
from env.interception import InterceptionConfig, InterceptionEnv
from env.interception.config import VERSION
from tools.interception_replay import render, render_video, video_times
from tools.interception_timeline import Timeline


def save_trace(path, config, rows):
    write_json(path, dict(version=VERSION, config=asdict(config), trace_version=2,
                          result={"outcome": rows[-1]["outcome"]}, frames=rows))


def fixture(path, kind, *, record=True):
    common = dict(evader_mode="constant_velocity", constant_direction=(1, 0, 0))
    p = [[-30, -12, 15, 1.2, 0, 0], [30, 12, 15, 1.2, 0, 0],
         [-15, -20, 15, 0, 0, 0], [15, 20, 15, 0, 0, 0], [0, 25, 15, 0, 0, 0]]
    e = [[-26.8, -12, 15, .6, 0, 0], [33.6, 12, 15, .6, 0, 0]]
    u = np.full((5, 3), .5)
    if kind == "success":
        config = InterceptionConfig(scale="5v2", evader_speed_fraction=.1, **common)
        u[:2, 0] += 1.2/16
    elif kind == "collision":
        config = InterceptionConfig(dt=1., **common)
        p = [[-2, 20, 15, 8, 0, 0], [2, 20, 15, -8, 0, 0], [0, 30, 15, 0, 0, 0]]
        e = [[50, 0, 15, 0, 0, 0]]
        u[:2, 0] = (1., 0.)
    elif kind in ("pursuer_boundary_violation", "evader_escape"):
        config = InterceptionConfig(scale="1v1", dt=3., **common)
        p = [[79, 20, 15, 4, 0, 0]] if kind == "pursuer_boundary_violation" else [[60, 20, 15, 0, 0, 0]]
        e = [[40, -20, 15, 6, 0, 0]] if kind == "pursuer_boundary_violation" else [[79, -20, 15, 6, 0, 0]]
        u[0, 0] = 0.
    else:
        config = InterceptionConfig(scale="1v1", dt=3., max_steps=1, **common)
        p = [[15, 20, 15, 0, 0, 0]]
        e = [[-8 if kind == "breach" else 40, 0, 15, 6, 0, 0]]
    env = InterceptionEnv(config)
    _, info = env.reset(12, dict(scenario="custom", pursuers=p, evaders=e))
    rows = [trace_row(info)] if record else []
    values = []
    while True:
        _, rewards, terminated, truncated, info = env.step(u)
        values.append((info, rewards))
        if record:
            rows.append(trace_row(info, u))
        if terminated or truncated:
            break
    if record:
        save_trace(path, config, rows)
    return values


@pytest.mark.parametrize("scale", ["1v1", "3v1"])
def test_reference_capture_uses_only_target_point(tmp_path, scale):
    config = InterceptionConfig(scale=scale)
    row, frames = episode(config, 1000, {"scenario": "fixed"}, "reference", trace=True)
    path = tmp_path / "capture.json"
    save_trace(path, config, frames)
    timeline = Timeline(path)
    event, marker = timeline.events[0], timeline.display_events[0]
    assert row["outcome"] == "success" and row["time"] == event["time"]
    assert marker["entity_ids"] == event["entities"] and marker["entity_ids"][1] == 5
    assert marker["position"] == event["positions"][1] != event["positions"][0]
    assert (marker["plotly_marker"], marker["mpl_marker"], marker["color"]) == ("x", "x", "#dc1824")
    assert not timeline.at(row["time"])["entities"][-1]["active"]


def test_immediate_boundary_exit_has_a_playable_zero_time(tmp_path):
    config = InterceptionConfig(scale="1v1", evader_mode="constant_velocity")
    env = InterceptionEnv(config)
    _, first = env.reset(12, dict(scenario="custom", pursuers=[[79.7,20,15,4,0,0]],
                                  evaders=[[40,-20,15,6,0,0]]))
    action = np.full((5,3), .5)
    _, _, terminated, _, last = env.step(action)
    assert terminated and last["time"] == 0.
    path = tmp_path/"zero.json"
    save_trace(path, config, [trace_row(first), trace_row(last, action)])
    timeline = Timeline(path)
    assert timeline.key_times == [0.] and timeline.at(0.)["outcome"] == "pursuer_boundary_violation"
    assert timeline.display_events[0]["position"] == first["pursuers"][0, :3].tolist()


@pytest.mark.parametrize("kind", ["success", "collision", "breach", "timeout",
                                 "pursuer_boundary_violation", "evader_escape"])
def test_real_events_and_boundary_states(tmp_path, kind):
    path = tmp_path / "trace.json"
    fixture(path, kind)
    timeline = Timeline(path)
    assert timeline.at(timeline.duration)["outcome"] == kind
    if kind in ("pursuer_boundary_violation", "evader_escape"):
        marker, = timeline.display_events
        assert marker["time"] == timeline.duration
        assert marker["position"][0] == pytest.approx(79.7, abs=1e-6)
        assert marker["mpl_marker"] == ("+" if kind == "pursuer_boundary_violation" else "o")
    for row in timeline.frames:
        sample = timeline.at(row["time"])
        assert sample["step"] == row["step"]
        for entity, (key, mask, _, _, i, _) in zip(sample["entities"], timeline.entities):
            assert entity["position"] == row[key][i][:3]
            assert entity["active"] == row[mask][i]
            if entity["active"]:
                assert entity["velocity"] == row[key][i][3:]
    if kind == "success":
        first, second = timeline.display_events
        assert first["entity_ids"] == [0, 5] and second["entity_ids"] == [1, 6]
        assert first["time"] == pytest.approx(5/6) and second["time"] == pytest.approx(1.5)
        sample = timeline.at((first["time"]+second["time"])/2)
        assert sample["entities"][-2]["position"] == first["position"]
        assert sample["entities"][-1]["active"]
        assert sample["nearest_pair"]["evader"] == "E1"
        assert sample["nearest_pair"]["relative_speed"] == pytest.approx(.6)
    elif kind == "collision":
        marker, = timeline.display_events
        assert marker["entity_ids"] == [0, 1] and marker["mpl_marker"] == "D"
        assert marker["time"] == pytest.approx(.2125)
        assert marker["position"] == np.mean(marker["positions"], axis=0).tolist()
    elif kind == "breach":
        marker, = timeline.display_events
        assert marker["entity_ids"] == [5, 5] and marker["mpl_marker"] == "s"
        assert marker["time"] == pytest.approx(.5)
        assert marker["position"] == timeline.events[0]["positions"][0]
        np.testing.assert_allclose(marker["position"], [-5., 0., 15.], atol=1e-8)
    elif kind == "timeout":
        assert timeline.duration == 3. and timeline.display_events == []


def test_timeline_rejects_outside_and_nonfinite(tmp_path):
    path = tmp_path / "trace.json"
    fixture(path, "collision")
    timeline = Timeline(path)
    for value in (-1e-12, timeline.duration+1e-12, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="replay time"):
            timeline.at(value)


@pytest.mark.parametrize(("duration", "fps", "count"), [(5.444331752283873,15,82),
                          (.001,15,2), (0.,15,1), (1.,10,10)])
def test_video_grid_is_uniform_and_preserves_terminal(duration, fps, count):
    times = video_times(duration, fps)
    assert len(times) == count
    assert times[0] == 0. and times[-1] == duration
    if count > 1:
        np.testing.assert_allclose(np.diff(times), duration/(count-1), atol=1e-14, rtol=0)
        assert all(b > a for a, b in zip(times, times[1:]))
    if duration >= 1/fps:
        assert abs(count/fps-duration) < 1/fps


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg absent")
def test_decoded_capture_markers_and_passive_export(tmp_path, monkeypatch):
    source, video = tmp_path/"partial.json", tmp_path/"partial.mp4"
    without = fixture(source, "success", record=False)
    with_trace = fixture(source, "success")
    np.testing.assert_equal(without, with_trace)
    original = source.read_bytes()

    def unexpected_step(*args, **kwargs):
        raise AssertionError("exporter advanced the environment")

    monkeypatch.setattr(InterceptionEnv, "step", unexpected_step)
    render(source, tmp_path/"partial.html")
    render_video(source, video, fps=15, width=640, height=360)
    assert source.read_bytes() == original
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(video),
                                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    frames = np.frombuffer(raw, np.uint8).reshape(-1, 360, 640, 3)
    trace = Timeline(source)
    times = video_times(trace.duration, 15)
    assert len(times) == len(frames)
    first = np.searchsorted(times, trace.events[0]["time"])
    red = (frames[..., 0] > 140) & (frames[..., 1] < 75) & (frames[..., 2] < 85)
    assert not red[:first].any()
    assert red[first:, :, :320].sum(axis=(1, 2)).min() > 20
    assert not red[:-1, :, 320:].any()
    assert red[-1, :, 320:].sum() > 20
    original_video = video.read_bytes()
    with pytest.raises(FileExistsError):
        render_video(source, video, fps=15, width=640, height=360)
    assert video.read_bytes() == original_video
