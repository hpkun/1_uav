from dataclasses import replace
import gzip
import json

import numpy as np
import pytest

from env.ideal_interception import InterceptionConfig,InterceptionEnv
from tools.ideal_blue_observations import generate,main


@pytest.mark.parametrize("mode",["straight","bezier"])
def test_complete_observations_match_environment_reference(mode):
    config=InterceptionConfig(scale="5v2",intruder_mode=mode,initialization="random")
    env=InterceptionEnv(config)
    env.reset(17)
    data=generate(config,17)
    assert data["target_count"]==2
    assert data["scenario_id"]==env.scenario_id
    for j,target in enumerate(data["targets"]):
        assert target["number_of_regular_samples"]==len(target["observations"])
        assert target["last_regular_sample_time"]<=target["reference_duration"]
        for row in target["observations"]:
            p,v=env.paths[j].sample(row["time"])
            np.testing.assert_allclose(row["position"],p,atol=1e-10)
            np.testing.assert_allclose(row["velocity"],v,atol=1e-10)
            assert row["active"]


def test_blue_path_outlives_red_terminal_event():
    base=InterceptionConfig()
    config=replace(base,defender_starts=((294.,-50.,60.),)+base.defender_starts[1:])
    env=InterceptionEnv(config)
    _,info=env.reset(17)
    action=np.full((5,3),.5)
    action[0]=[1.,.5,.5]
    for _ in range(10):
        _,_,done,_,info=env.step(action)
        if done:
            break
    assert done and info["outcome"]=="defender_boundary"
    data=generate(config,17)
    target=data["targets"][0]
    assert target["last_regular_sample_time"]>info["time"]+10
    assert target["reference_duration"]>target["last_regular_sample_time"]-.1


def test_non_grid_endpoint_is_separate_and_reproducible(tmp_path):
    config=replace(InterceptionConfig(),intruder_speeds=(19.,20.))
    data=generate(config,23)
    assert data==generate(config,23)
    target=data["targets"][0]
    assert not target["endpoint"]["regular"]
    assert target["last_regular_sample_time"]<target["endpoint"]["time"]
    assert target["number_of_regular_samples"]==len(target["observations"])
    path=tmp_path/"observations.json.gz"
    main(["--out",str(path),"--config",str(_config_file(tmp_path,config)),"--seed","23"])
    with gzip.open(path,"rt",encoding="utf-8") as stream:
        assert json.load(stream)==data


def test_targets_keep_independent_sampling_ranges():
    config=replace(InterceptionConfig(scale="5v2",intruder_mode="bezier"),
                   intruder_speeds=(19.,23.))
    data=generate(config,17)
    first,second=data["targets"]
    assert first["reference_duration"]!=second["reference_duration"]
    assert first["number_of_regular_samples"]!=second["number_of_regular_samples"]
    for target in data["targets"]:
        assert target["last_regular_sample_time"]<=target["reference_duration"]
        assert target["reference_duration"]-target["last_regular_sample_time"]<config.dt


def _config_file(folder,config):
    from dataclasses import asdict
    path=folder/"scene.json"
    path.write_text(json.dumps(asdict(config)))
    return path
