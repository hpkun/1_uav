from copy import deepcopy
from dataclasses import replace

import pytest

from env.ideal_interception import InterceptionConfig
from tools.ideal_interception_trace import record,write_trace
from tools.ideal_interception_timeline import Timeline


def fixture():
    base=InterceptionConfig()
    return record(replace(base,dt=1.,max_steps=2,
        defender_starts=((-5.,0.,60.),)+base.defender_starts[1:],
        intruder_starts=((5.,0.,60.),)+base.intruder_starts[1:],
        intruder_ends=((-50.,0.,60.),)+base.intruder_ends[1:]),7)


@pytest.mark.parametrize("change,match",[
    (lambda d:d["references"][0]["specification"].__setitem__("speed",21.),"reference 0"),
    (lambda d:d["references"][0]["specification"].__setitem__("duration",1.),"reference 0"),
    (lambda d:d["frames"][-1]["defenders"][0].__setitem__(0,99.),"reconstructed state"),
    (lambda d:d["frames"][-1]["executed_velocities"][0].__setitem__(0,0.),"executed speed"),
    (lambda d:d["frames"][-1]["events"][0].__setitem__("time",2.),"event outside"),
    (lambda d:d["frames"][-1]["events"][0].__setitem__("members",[99,5]),"event members"),
    (lambda d:d["frames"][-1]["events"].append(deepcopy(d["frames"][-1]["events"][1])),"inactive member"),
    (lambda d:d.pop("config"),"required field"),
])
def test_tampered_trace_is_rejected(tmp_path,change,match):
    data=deepcopy(fixture())
    change(data)
    path=tmp_path/"bad.json"
    write_trace(path,data)
    with pytest.raises(ValueError,match=match):
        Timeline(path)


def test_valid_trace_reconstructs_step_internal_exit(tmp_path):
    data=fixture()
    path=tmp_path/"good.json.gz"
    write_trace(path,data)
    timeline=Timeline(path)
    contact=next(e for e in timeline.events if e["kind"]=="interception_contact")
    assert timeline.at(contact["time"])["entities"][0]["active"] is False
    assert timeline.at(timeline.duration)["outcome"]==data["result"]["outcome"]
