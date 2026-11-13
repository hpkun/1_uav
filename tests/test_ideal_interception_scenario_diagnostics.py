from dataclasses import replace
from dataclasses import asdict
import json

import pytest

from env.ideal_interception import InterceptionConfig,InterceptionEnv
from env.ideal_interception.scenario import validate_scenario
from tools.ideal_interception_trace import main as trace_main


def test_full_path_diagnostics_and_seed_identity():
    base=InterceptionConfig(scale="5v2",initialization="random",intruder_mode="bezier")
    config=replace(base,defender_speeds=(24.,)*5)
    first=InterceptionEnv(config).reset(23)[1]
    second=InterceptionEnv(config).reset(23)[1]
    assert first["scenario_id"]==second["scenario_id"]
    report=first["scene_diagnostics"]
    assert report==second["scene_diagnostics"]
    assert report["maximum_step_ratio"]==report["defender_step_ratio"]
    assert report["defender_step_ratio"]>report["intruder_step_ratio"]
    assert all(row["within_bounds"] for row in report["paths"])
    assert all(row["minimum_boundary_margin"]>=0 for row in report["paths"])


def test_scenario_declaration_rejects_bypass_and_late_entry():
    base=InterceptionConfig()
    bypass=replace(base,intruder_starts=((200.,100.,60.),base.intruder_starts[1]),
                   intruder_ends=((-260.,100.,60.),base.intruder_ends[1]))
    report=InterceptionEnv(bypass).reset()[1]["scene_diagnostics"]
    assert report["paths"][0]["path_class"]=="bypass"
    validate_scenario(report,"geometry")
    with pytest.raises(ValueError,match="region_crossing"):
        validate_scenario(report,"region_crossing")
    late=replace(base,max_steps=2)
    report=InterceptionEnv(late).reset()[1]["scene_diagnostics"]
    assert report["paths"][0]["path_class"]=="region_crossing"
    assert not report["paths"][0]["entry_within_episode"]
    with pytest.raises(ValueError,match="region_crossing"):
        validate_scenario(report,"region_crossing")


def test_model_free_cli_uses_full_config_and_rejects_conflicts(tmp_path):
    config=InterceptionConfig(max_steps=2,defender_speeds=(24.,)*5)
    source=tmp_path/"scene.json"
    source.write_text(json.dumps(asdict(config)))
    a,b=tmp_path/"a.json",tmp_path/"b.json"
    trace_main([str(a),"--config",str(source),"--seed","31"])
    trace_main([str(b),"--config",str(source),"--seed","31"])
    assert json.loads(a.read_text())==json.loads(b.read_text())
    assert json.loads(a.read_text())["config"]["defender_speeds"]==[24.]*5
    with pytest.raises(SystemExit):
        trace_main([str(tmp_path/"bad.json"),"--config",str(source),"--scale","1v1"])
    source.write_text('{"unknown_parameter": 1}')
    with pytest.raises(TypeError,match="unknown_parameter"):
        trace_main([str(tmp_path/"bad.json"),"--config",str(source)])
