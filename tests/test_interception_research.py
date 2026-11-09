import json
import shutil
import subprocess

import numpy as np
import pytest
import torch

import algorithm.interception as cli
from algorithm.common.interception_checkpoint import read
from algorithm.common.research import EvaluationProtocol, commit_header, journal, score, summarize
from algorithm.common.serialization import write_json
from algorithm.mappo.interception import TrainConfig
from env.interception import InterceptionConfig
from env.interception.config import VERSION
from env.interception.motion import command
from tools.interception_replay import render, render_video, video_times
from tools.interception_report import generate
from tools.interception_timeline import Timeline


def test_capture_and_protection_rates_are_independent():
    outcomes = ["success", "timeout", "evader_escape", "pursuer_boundary_violation", "breach",
                "friendly_collision", "simulation_failure"]
    rows = [dict(outcome=outcome, time=1., minimum_separation=2., episode_return=0.,
                 control_effort=0., reward_parts={}, target_ids=[0,1],
                 captured_ids=[0,1] if outcome == "success" else [0] if outcome == "timeout" else [])
            for outcome in outcomes]
    stats = summarize(rows)
    assert stats["success_rate"] == stats["evader_escape_rate"] == stats["partial_capture_rate"] == 1/7
    assert stats["protection_success_rate"] == 2/7
    assert sum(stats["outcomes"].values()) == 7
    assert stats["numerical_failure_count"] == 1


def exact(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            exact(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            exact(a, b)
    elif hasattr(left, "__dict__"):
        exact(vars(left), vars(right))
    else:
        assert left == right


def test_trace_is_passive_and_keeps_physical_time(tmp_path):
    config = InterceptionConfig(scale="5v2")
    options = {"scenario": "fixed"}
    a, frames = cli.episode(config, 1000, options, "predictive", trace=True)
    b, _ = cli.episode(config, 1000, options, "predictive")
    exact(a, b)
    assert a["outcome"] == "success"
    assert a["capture_count"] == 2
    assert a["time"] < frames[-1]["step"] * config.dt
    assert frames[0]["commands"] is None and frames[0]["normalized_actions"] is None
    row = frames[1]
    np.testing.assert_allclose(command(row["normalized_actions"], config.pursuer_vmax), row["requested_velocities"][:5])
    assert not np.allclose(row["pursuers"][:, 3:], row["commands"][:5])
    assert a["episode_return"] == pytest.approx(sum(a["reward_parts"].values()), abs=1e-5)
    cli.evaluate(tmp_path/"run", config, options, "predictive", [1000], compressed=True)
    timeline = Timeline(tmp_path/"run/trajectory_000.json.gz")
    last = timeline.at(timeline.duration)
    assert not any(x["active"] for x in last["entities"] if x["name"].startswith("E"))
    for event in timeline.events:
        if event["kind"] != "interception_contact":
            continue
        capture = timeline.at(event["time"])
        target = next(e for e in capture["entities"] if e["name"] == f'E{event["entities"][1]-5}')
        assert target["position"] == event["positions"][1]
        assert not target["active"] and target["speed"] is None
    destination = tmp_path/"replay.html"
    render(tmp_path/"run/trajectory_000.json.gz", destination)
    html = destination.read_text(encoding="utf-8")
    payload, _ = json.JSONDecoder().raw_decode(html.split("const DATA = ", 1)[1])
    assert [payload["frames"][i]["time"] for i in payload["keys"]] == timeline.key_times
    assert payload["frames"][-1]["dwell"] is None
    assert len(payload["events"]) == 2
    with pytest.raises(FileExistsError):
        render(tmp_path/"run/trajectory_000.json.gz", destination)


def collision_trace(path):
    config = InterceptionConfig(dt=1., evader_mode="constant_velocity")
    from dataclasses import asdict
    from env.interception import InterceptionEnv
    env = InterceptionEnv(config)
    _, first = env.reset(12, dict(scenario="custom",
        pursuers=[[-2, 20, 15, 8, 0, 0], [2, 20, 15, -8, 0, 0], [0, 30, 15, 0, 0, 0]],
        evaders=[[50, 0, 15, 0, 0, 0]]))
    actions = np.full((5, 3), .5)
    actions[:2, 0] = (.5+8/24, .5-8/24)
    _, _, terminated, _, second = env.step(actions)
    assert terminated and second["outcome"] == "friendly_collision"
    write_json(path, dict(version=VERSION, config=asdict(config), result={"outcome": "friendly_collision"},
                          frames=[cli.trace_row(first), cli.trace_row(second, actions)]))


def test_legacy_detection_and_analytic_midstep(tmp_path):
    path = tmp_path/"legacy.json"
    collision_trace(path)
    data = json.loads(path.read_text())
    for frame in data["frames"]:
        for name in ("distances", "normalized_actions", "requested_velocities", "initial_accelerations"):
            frame.pop(name)
    data["version"] = "uav-interception-v3.0.0"
    write_json(path, data)
    trace = Timeline(path)
    assert trace.duration == pytest.approx(.1875, abs=1e-7)
    midway = trace.at(.1)
    assert midway["entities"][0]["position"][0] == pytest.approx(-1.2)
    assert midway["dwell"] is None
    assert trace.at(trace.duration)["entities"][0]["position"] == trace.events[-1]["positions"][0]


def test_partial_capture_stays_visible_without_motion(tmp_path):
    from dataclasses import asdict
    from env.interception import InterceptionEnv
    config = InterceptionConfig(scale="5v2", evader_mode="constant_velocity",
                                evader_speed_fraction=.6/13, constant_direction=(1, 0, 0))
    env = InterceptionEnv(config)
    _, info = env.reset(12, dict(scenario="custom",
        pursuers=[[-30,-12,15,1.2,0,0], [30,12,15,1.2,0,0], [-15,-20,15,0,0,0],
                  [15,20,15,0,0,0], [0,25,15,0,0,0]],
        evaders=[[-28.8,-12,15,.6,0,0], [32.1,12,15,.6,0,0]]))
    rows = [cli.trace_row(info)]
    actions = np.full((5, 3), .5)
    actions[:2, 0] += 1.2 / 24
    for _ in range(20):
        _, _, terminal, _, info = env.step(actions)
        rows.append(cli.trace_row(info, actions))
        if terminal:
            break
    path = tmp_path/"partial.json"
    write_json(path, dict(version=VERSION, trace_version=2, config=asdict(config),
                          result={"outcome": info["outcome"]}, frames=rows))
    timeline = Timeline(path)
    first, second = timeline.events
    assert first["time"] < second["time"]
    for time in (first["time"], (first["time"]+second["time"])/2):
        left, right = timeline.at(time)["entities"][-2:]
        assert not left["active"] and left["position"] == first["positions"][1]
        assert right["active"] and right["velocity"] is not None


def test_mp4_missing_ffmpeg(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="Install"):
        render_video("unused", tmp_path/"video.mp4")
    assert not (tmp_path/"video.mp4").exists()


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg tools absent")
def test_real_mp4_has_expected_duration_and_frames(tmp_path):
    source = tmp_path/"trace.json"
    output = tmp_path/"video.mp4"
    collision_trace(source)
    render_video(source, output, fps=20, width=320, height=240)
    probe = subprocess.check_output(["ffprobe", "-v", "error", "-show_streams",
                                     "-of", "json", str(output)], text=True)
    stream = json.loads(probe)["streams"][0]
    duration = Timeline(source).duration
    assert int(stream["nb_frames"]) == len(video_times(duration, 20)) == 4
    assert abs(float(stream["duration"])-duration) < 1/20
    assert (stream["width"], stream["height"]) == (320, 240)


def test_best_lexicographic_order_and_empty_window():
    def entry(update, success, breach, collision, time):
        return dict(update=update, summary=dict(success_rate=success, breach_rate=breach,
                    collision_rate=collision, mean_capture_time_s=time))
    assert min([entry(1, .4, .1, 0, 1), entry(2, .5, .2, .1, 2)], key=score)["update"] == 2
    assert min([entry(1, .4, .2, 0, 1), entry(2, .4, .1, .1, 2)], key=score)["update"] == 2
    assert min([entry(1, .4, .2, .2, 1), entry(2, .4, .2, .1, 2)], key=score)["update"] == 2
    assert min([entry(1, .4, .2, .1, 3), entry(2, .4, .2, .1, 2)], key=score)["update"] == 2
    assert min([entry(2, 0, 1, 0, None), entry(1, 0, 1, 0, None)], key=score)["update"] == 1
    assert summarize([])["success_rate"] is None


def test_cuda_training_evaluation_resume_and_reconciliation(tmp_path, monkeypatch, capsys):
    config = InterceptionConfig(scale="1v1", dt=1.)
    training = TrainConfig(workers=1, horizon=5, epochs=1)
    protocol = EvaluationProtocol(1, 2, 310, "fixed", False)
    plain, continuous, segmented = [tmp_path/key for key in ("plain", "continuous", "segmented")]
    cli.train(plain, config, training, updates=2, log_interval=3)
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("[Interception START]") and "Sampling 1 worker x 5 steps" in lines[0]
    assert [line.split(" | ")[0] for line in lines if "[Interception TRAIN]" in line] == [
        "[Interception TRAIN] Update 2/2 (100.0%)"]
    assert "Train window n=1/100" in lines[1]
    assert lines[-1].startswith("[Interception DONE]")
    cli.train(continuous, config, training, updates=2, evaluation=protocol)
    logs = capsys.readouterr().out
    assert "Train window n=0/100, no completed episodes yet" in logs
    assert logs.count("[Interception EVAL]") == 2
    assert "Scenario fixed, actions=Beta sample" in logs
    assert "Best update 1" in logs
    original = cli.atomic_json

    def interrupt_update(path, data):
        if path.name == "update_000001.json":
            raise OSError("simulated interruption after checkpoint")
        original(path, data)

    with monkeypatch.context() as patch:
        patch.setattr(cli, "atomic_json", interrupt_update)
        with pytest.raises(OSError, match="simulated interruption"):
            cli.train(segmented, config, training, updates=1, evaluation=protocol)
    with monkeypatch.context() as patch:
        def interrupt_best(*args):
            if args[2] == 1:
                raise OSError("simulated interruption after evaluation")
        patch.setattr(cli, "publish_best", interrupt_best)
        with pytest.raises(OSError, match="after evaluation"):
            cli.train(segmented, resume=True)
    evaluation_before = (segmented/"evaluation_000001.json").read_bytes()
    assert "[Interception EVAL]" not in capsys.readouterr().out
    cli.train(segmented, resume=True)
    logs = capsys.readouterr().out
    assert "[Interception RESUME]" in logs and "From update 1 / 5 joint steps" in logs
    assert logs.count("[Interception EVAL]") == 1 and "[Interception EVAL] Update 2" in logs
    assert (segmented/"evaluation_000001.json").read_bytes() == evaluation_before
    variants = [read(folder/"latest.pt") for folder in (plain, continuous, segmented)]
    for candidate in variants[1:]:
        for key in ("actor", "critic", "actor_optimizer", "critic_optimizer", "steps", "episodes",
                    "torch_rng", "cuda_rng", "numpy_rng", "python_rng", "vector"):
            exact(variants[0][key], candidate[key])
    first = json.loads((segmented/"evaluation_000001.json").read_text())
    other = json.loads((continuous/"evaluation_000001.json").read_text())
    exact(first["runs"], other["runs"])
    assert first["seeds"] == [310, 311]
    best = read(segmented/"best.pt")
    assert best["updates"] == json.loads((segmented/"best.json").read_text())["best_update"]
    journal_records = journal(segmented, commit_header(segmented/"latest.pt"))
    entries = [json.loads(row) for row in (segmented/"episode_metrics.jsonl").read_text().splitlines()]
    assert len(entries) == variants[2]["episodes"]
    assert len({(row["worker_id"], row["worker_episode_index"]) for row in entries}) == len(entries)
    assert [row["update"] for row in journal_records] == [1, 2]
    for row in entries:
        assert row["episode_return"] == pytest.approx(sum(row["reward_components"].values()), abs=1e-5)
    generate([segmented], tmp_path/"report")
    assert json.loads((tmp_path/"report/summary.json").read_text())["runs"][0]["completed_updates"] == 2
    checkpoint = str(segmented/"latest.pt")
    human, machine = (tmp_path/name for name in ("human_eval", "json_eval"))
    cli.main(["evaluate", checkpoint, "--out", str(human)])
    assert "[Interception EVAL]" in capsys.readouterr().out
    cli.main(["evaluate", checkpoint, "--out", str(machine), "--json"])
    summary = json.loads(capsys.readouterr().out)
    assert summary == {key: json.loads((machine/"summary.json").read_text())[key] for key in summary}
    for name in ("summary.json", "trajectory_000.json"):
        assert (human/name).read_bytes() == (machine/name).read_bytes()
    # Attempt an old-generation resume with later immutable records present.
    if best["updates"] == 1:
        shutil.copyfile(segmented/"best.pt", segmented/"latest.pt")
    else:
        (segmented/"update_000003.json").write_text("{}")
    with pytest.raises(ValueError, match="newer records"):
        cli.train(segmented, resume=True)
