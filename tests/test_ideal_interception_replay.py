from dataclasses import replace
import json
import shutil
import subprocess

import numpy as np
import pytest

from env.ideal_interception import InterceptionConfig
from tools.ideal_interception_trace import record, write_trace
from tools.ideal_interception_timeline import Timeline


def test_trace_round_trip_and_continuous_contact(tmp_path):
    base=InterceptionConfig()
    config=replace(base,dt=1.,max_steps=10,
        defender_starts=((-5.,0.,60.),)+base.defender_starts[1:],
        intruder_starts=((5.,0.,60.),)+base.intruder_starts[1:],
        intruder_ends=((-50.,0.,60.),)+base.intruder_ends[1:],
        protected_center=(-220.,0.,0.))
    data=record(config,7)
    assert data["trace_version"]=="uav-ideal-trace-v1"
    assert [entity["role"] for entity in data["entities"]]==["defender","intruder"]
    assert data["result"]["outcome"]=="all_intercepted"
    contact=next(e for e in data["result"]["events"] if e["kind"]=="interception_contact")
    assert 0<contact["time"]<config.dt
    paths=[]
    for suffix in ("json","json.gz"):
        path=tmp_path/f"trace.{suffix}"
        write_trace(path,data)
        timeline=Timeline(path)
        paths.append(path)
        assert contact["time"] in timeline.key_times
        before=timeline.at(contact["time"]*.5)
        defender=next(e for e in before["entities"] if e["name"]=="D0")
        assert defender["active"]
        np.testing.assert_allclose(defender["position"],
            np.asarray(data["frames"][0]["defenders"][0][:3])+np.array([20.,0.,0.])*before["time"])
        at=timeline.at(contact["time"])
        assert not any(e["active"] for e in at["entities"])
        assert at["time"]==pytest.approx(data["frames"][-1]["time"])
    assert json.loads(paths[0].read_text())["frames"]==data["frames"]


def test_record_is_deterministic_and_reference_tamper_is_rejected(tmp_path):
    config=InterceptionConfig(scale="3v1",intruder_mode="bezier",initialization="random",max_steps=2)
    a,b=record(config,31),record(config,31)
    assert a==b
    assert a["scene_diagnostics"]["step_ratio"]>0
    assert a["scene_diagnostics"]["paths"][0]["duration_fraction"]>0
    a["references"][0]["points"][0][0]+=1
    path=tmp_path/"tampered.json"
    write_trace(path,a)
    with pytest.raises(ValueError,match="hash mismatch"):
        Timeline(path)


def test_multitarget_member_exit_is_visible_inside_control_step(tmp_path):
    base=InterceptionConfig(scale="5v2",dt=1.,max_steps=2)
    config=replace(base,defender_starts=((-5.,0.,60.),)+base.defender_starts[1:],
                   intruder_starts=((5.,0.,60.),base.intruder_starts[1]),
                   intruder_ends=((-50.,0.,60.),base.intruder_ends[1]))
    path=tmp_path/"multi.json"
    write_trace(path,record(config,7))
    timeline=Timeline(path)
    hit=next(e for e in timeline.events if e["kind"]=="interception_contact")
    assert 0<hit["time"]<1
    entities={e["name"]:e for e in timeline.at(hit["time"])["entities"]}
    assert not entities["D0"]["active"] and not entities["I0"]["active"]
    assert entities["I1"]["active"]
    assert timeline.at(hit["time"])["outcome"]=="ongoing"


@pytest.mark.parametrize("scale,mode",[("1v1","straight"),("3v1","bezier"),("5v2","bezier")])
def test_replay_scales_and_vertical_motion(tmp_path,scale,mode):
    base=InterceptionConfig(scale=scale,intruder_mode=mode,max_steps=2)
    starts=((200.,-10.,40.),base.intruder_starts[1])
    ends=((-260.,-10.,80.),base.intruder_ends[1])
    config=replace(base,intruder_starts=starts,intruder_ends=ends)
    path=tmp_path/f"{scale}.json.gz"
    write_trace(path,record(config,17))
    timeline=Timeline(path)
    initial=timeline.at(0.)
    final=timeline.at(timeline.duration)
    assert final["entities"][-config.intruder_count]["position"][2] > initial["entities"][-config.intruder_count]["position"][2]
    assert timeline.data["scene_diagnostics"]["paths"][0]["within_bounds"]


def test_ideal_timeline_rejects_v4_trace(tmp_path):
    path=tmp_path/"v4.json"
    path.write_text(json.dumps(dict(version="uav-interception-v4.0.0",frames=[])))
    with pytest.raises(ValueError,match="protocol"):
        Timeline(path)
    ideal=tmp_path/"ideal.json"
    write_trace(ideal,record(InterceptionConfig(max_steps=2),7))
    from tools.interception_timeline import Timeline as V4Timeline
    with pytest.raises(ValueError,match="unsupported interception trajectory"):
        V4Timeline(ideal)


def test_html_uses_recorded_trace_without_simulation(tmp_path):
    pytest.importorskip("plotly")
    from tools.ideal_interception_replay import render_html
    data=record(InterceptionConfig(max_steps=2),7)
    source=tmp_path/"trace.json"
    write_trace(source,data)
    target=tmp_path/"view.html"
    render_html(source,target,"top")
    content=target.read_text()
    assert "Ideal interception" in content and data["scenario_id"][:12] in content
    assert "reference" in content and "Geometric events" in content
    render_html(source,tmp_path/"view3d.html","3d")
    assert (tmp_path/"view3d.html").stat().st_size>1000


@pytest.mark.skipif(shutil.which("ffmpeg") is None,reason="FFmpeg unavailable")
@pytest.mark.parametrize("view",["3d","top","altitude"])
def test_short_mp4_uses_the_same_trace(tmp_path,view):
    from tools.ideal_interception_replay import render_mp4
    source=tmp_path/"trace.json"
    write_trace(source,record(InterceptionConfig(max_steps=2),7))
    target=tmp_path/f"{view}.mp4"
    render_mp4(source,target,view,fps=30)
    assert target.stat().st_size>1000
    if shutil.which("ffprobe"):
        output=subprocess.check_output(["ffprobe","-v","error","-show_entries",
            "format=duration:stream=width,height,nb_frames","-of","json",str(target)],text=True)
        metadata=json.loads(output)
        assert abs(float(metadata["format"]["duration"])-.2)<1/30+1e-3
        assert metadata["streams"][0]["width"]==1280
        assert metadata["streams"][0]["height"]==720


def test_html_clock_and_existing_outputs(tmp_path,monkeypatch):
    pytest.importorskip("plotly")
    from tools import ideal_interception_replay as replay
    source=tmp_path/"trace.json"
    write_trace(source,record(InterceptionConfig(max_steps=2),7))
    target=tmp_path/"view.html"
    replay.render_html(source,target,"altitude")
    html=target.read_text()
    assert "performance.now()" in html and "(now-started)*rate/1000" in html
    assert "XZ Side View" in html
    with pytest.raises(FileExistsError):
        replay.render_html(source,target,"altitude")
    original=target.read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(replay,"_publish",lambda *args: (_ for _ in ()).throw(OSError("injected")))
        with pytest.raises(OSError,match="injected"):
            replay.render_html(source,target,"altitude",overwrite=True)
    assert target.read_bytes()==original


def test_mp4_encoder_failure_preserves_existing_file(tmp_path,monkeypatch):
    from tools import ideal_interception_replay as replay
    target=tmp_path/"existing.mp4"
    target.write_bytes(b"historical")
    with pytest.raises(FileExistsError):
        replay.render_mp4("unused",target)
    monkeypatch.setattr(replay,"_render_mp4_to_path",lambda *args: (_ for _ in ()).throw(RuntimeError("encoder failed")))
    with pytest.raises(RuntimeError,match="encoder failed"):
        replay.render_mp4("unused",target,overwrite=True)
    assert target.read_bytes()==b"historical"
