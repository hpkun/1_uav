"""V3.5 nominal geometry, fixed matching, protocol isolation and shape checks."""
from pathlib import Path
from zipfile import ZipFile
import json
import math
import re

import numpy as np
import pytest
import torch

from algorithm.ideal_interception import InterceptionMAPPO,PROTOCOL as MODEL_PROTOCOL,vector_env
from algorithm.common.interception_checkpoint import read
from algorithm.mappo.interception import TrainConfig
from algorithm.ideal_multi_predictive_guidance import run_episode
from env.ideal_interception import InterceptionEnv
from env.ideal_interception.config import (PROTOCOL,SCALES,DEFENDER_SLOTS,INTRUDER_SLOTS,
    OBS_DIM,STATE_DIM,FIFTEEN_DEFENDER_CIRCLE,FIFTEEN_INTRUDER_WEDGE,
    SIX_INTRUDER_WEDGE,formal_6v6_level2_config,formal_15v15_config,
    formal_15v15_level2_config)
from env.ideal_interception.scenario import initialize,diagnostics
from tools.ideal_interception_timeline import Timeline,SLOT_SCHEMAS
from tools.ideal_interception_trace import record,write_trace
from tools.ideal_interception_replay import render_html


def test_15v15_geometry_assignment_and_observation():
    assert (PROTOCOL,DEFENDER_SLOTS,INTRUDER_SLOTS,OBS_DIM,STATE_DIM)==(
        "uav-ideal-interception-v3.5.0",15,15,265,444)
    assert SCALES["15v15"]==(15,15)
    config=formal_15v15_level2_config()
    assert config.dt==.05 and config.max_steps==600
    assert config.lower==(-300.,-300.,0.) and config.upper==(300.,300.,120.)
    assert config.protected_center==(-220.,0.,0.)
    assert config.protected_radius==20. and config.contact_distance==.1
    assert config.defender_speeds==(20.,)*15 and config.intruder_speeds==(20.,)*15
    assert FIFTEEN_INTRUDER_WEDGE[:6]==SIX_INTRUDER_WEDGE
    assert len(FIFTEEN_INTRUDER_WEDGE)==15
    circle=np.asarray(FIFTEEN_DEFENDER_CIRCLE)
    np.testing.assert_allclose(circle[0],(-60.,0.,60.),atol=1e-12)
    np.testing.assert_allclose(circle.mean(axis=0),(-120.,0.,60.),atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(circle[:,:2]-(-120.,0.),axis=1),60.,atol=1e-12)
    nearest=min(np.linalg.norm(circle[i]-circle[j]) for i in range(15) for j in range(i+1,15))
    assert nearest==pytest.approx(120*math.sin(math.pi/15),abs=1e-10) and nearest>8
    first=initialize(config,17)
    second=initialize(config,17)
    assert first[-1]==second[-1] and first[3]==second[3]
    assert len(first[3])==len(set(first[3].values()))==15
    cost=sum(np.linalg.norm(first[0][i,:3]-first[1][j,:3]) for i,j in first[3].items())
    assert math.isfinite(cost) and cost>0
    assert len(first[2])==15 and len({p.specification()["polyline_sha256"] for p in first[2]})==15
    assert diagnostics(config,first[2])["scenario_validity"]["region_crossing"]
    env=InterceptionEnv(config)
    obs,info=env.reset(17)
    assert obs.shape==(15,265) and info["centralized_state"].shape==(444,)
    assert info["defender_mask"].all() and info["intruder_mask"].all()
    assert info["assignment_total_cost_m"]==pytest.approx(cost)
    assignment_start=6+7*(DEFENDER_SLOTS+INTRUDER_SLOTS)
    state_assignment_start=7*(DEFENDER_SLOTS+INTRUDER_SLOTS)
    assigned=info["assignment"][14]
    np.testing.assert_array_equal(obs[14,assignment_start:assignment_start+15],np.eye(15)[assigned])
    np.testing.assert_array_equal(info["centralized_state"][state_assignment_start+14*15:state_assignment_start+15*15],np.eye(15)[assigned])
    assert np.isfinite(obs).all() and np.isfinite(info["centralized_state"]).all()
    action=np.full((DEFENDER_SLOTS,3),.5)
    _,reward,_,_,next_info=env.step(action)
    assert reward.shape==(DEFENDER_SLOTS,)
    assert next_info["original_assignment"]==info["original_assignment"]


def test_former_six_slot_constructor_pads_without_activating_members():
    old=formal_6v6_level2_config()
    six_slot=type(old)(scale="6v6",defender_starts=old.defender_starts[:6],
        defender_speeds=old.defender_speeds[:6],
        defender_directions=old.defender_directions[:6],
        intruder_starts=old.intruder_starts[:6],intruder_ends=old.intruder_ends[:6],
        intruder_speeds=old.intruder_speeds[:6],
        assignment_policy="hungarian_fixed",intruder_motion_level="level2_maneuvering")
    assert six_slot.defender_count==six_slot.intruder_count==6
    assert initialize(six_slot,17)[-1]==initialize(old,17)[-1]
    with pytest.raises(ValueError,match="invalid avoidance stress profile"):
        formal_15v15_level2_config(scenario_profile_version="avoidance_stress_v1",
                                   defender_circle_radius_m=30.)


def test_15v15_member_resolution_and_inactive_motion():
    env=InterceptionEnv(formal_15v15_level2_config())
    env.reset(17)
    for j in range(1,15):
        env.intruder_mask[j]=False
        env.intruder_status[j]="intercepted"
        env.intruders[j,3:]=0.
    env.interception_count=14
    owner=next(i for i,j in env.original_assignment.items() if j==0)
    for i in range(15):
        if i!=owner:
            env.defender_mask[i]=False
            env.defender_status[i]="contact_exit"
            env.defenders[i,3:]=0.
    env.assignment={owner:0}
    target_direction=env.intruders[0,3:]/20.
    env.defenders[owner,:3]=env.intruders[0,:3]+target_direction
    env.defenders[owner,3:]=-20.*target_direction
    action=np.full((DEFENDER_SLOTS,3),.5)
    action[owner]=.5-.5*target_direction
    _,_,done,_,info=env.step(action)
    assert done and info["outcome"]=="all_intercepted"
    assert info["intercepted_count"]==15 and info["mission_success"]
    assert not info["defender_mask"][owner] and not info["intruder_mask"][0]
    assert not np.asarray(info["defenders"])[~info["defender_mask"],3:].any()


def test_historical_trace_slot_schemas_and_current_replay(tmp_path):
    root=Path(__file__).resolve().parents[1]/"outputs/ideal_guidance"
    paths={
        "uav-ideal-interception-v3.4.0":root/"v34_level2_three_point_seed17/episode.json",
        "uav-ideal-interception-v3.3.0":root/"multi6_v33_seed17_zuo_tp_verified/episode.json",
        "uav-ideal-interception-v3.2.0":root/"v32_zuo_seed17/episode.json",
        "uav-ideal-interception-v3.1.0":root/"formal_pip_seed17/episode.json",
        "uav-ideal-interception-v3.0.0":root/"three_point_pip_canonical_bezier_seed17/episode.json",
    }
    assert SLOT_SCHEMAS["uav-ideal-interception-v2.0.0"]==(5,2)
    for protocol,path in paths.items():
        timeline=Timeline(path)
        assert timeline.data["environment_protocol"]==protocol
        assert (timeline.defender_slots,timeline.intruder_slots)==SLOT_SCHEMAS[protocol]
        assert all(e["id"]<sum(SLOT_SCHEMAS[protocol]) for e in timeline.at(0)["entities"])
    _,trace=run_episode(formal_15v15_level2_config(),"three_point_pip",17)
    path=tmp_path/"v35.json"
    write_trace(path,trace)
    timeline=Timeline(path)
    assert (timeline.defender_slots,timeline.intruder_slots)==(15,15)
    assert len(timeline.at(0)["entities"])==30


def test_15v15_html_roles_and_member_labels(tmp_path):
    fixture=formal_15v15_config(intruder_mode="straight",max_steps=2)
    path=tmp_path/"fixture.json"
    write_trace(path,record(fixture,17))
    html=tmp_path/"fixture.html"
    render_html(path,html)
    markup=html.read_text(encoding="utf-8")
    assert "D14" in markup and "I14" in markup
    assert "Red Defender" in markup and "Blue Intruder" in markup
    assert "15v15" in markup and "avoid=0" in markup
    samples=json.loads(re.search(r"const times=(\[[^;]+\]), keys=",markup).group(1))
    assert samples==Timeline(path).key_times


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required by repository policy")
def test_15v15_mappo_shapes_and_old_checkpoint_rejection(tmp_path):
    training=TrainConfig(workers=1,horizon=1)
    model=InterceptionMAPPO(training)
    assert model.actor[0].in_features==OBS_DIM
    assert model.critic[0].in_features==STATE_DIM
    assert model.critic[-1].out_features==DEFENDER_SLOTS
    with vector_env(formal_15v15_level2_config(),training) as vector:
        assert vector.observations.shape==(1,15,265)
        assert vector.states.shape==(1,444)
        assert vector.masks.shape==(1,15)
        actions,_=model.act(vector.observations,vector.masks,deterministic=True)
        assert actions.shape==(1,15,3)
        transition=vector.step(actions.cpu().numpy())
        assert transition.rewards.shape==(1,15)
        assert transition.transition_states.shape==(1,444)
    for old,observation,state,defenders,intruders in (
            ("3.4.0",112,128,6,6),("3.3.0",112,128,6,6),
            ("3.2.0",68,66,5,2)):
        header=tmp_path/f"v{old}.pt"
        with ZipFile(header,"w") as archive:
            archive.writestr("archive/interception_protocol.json",json.dumps(
                MODEL_PROTOCOL|dict(version=f"uav-ideal-interception-v{old}",
                    observation=observation,state=state,defenders=defenders,intruders=intruders)))
        with pytest.raises(ValueError,match="incompatible checkpoint protocol"):
            read(header,protocol=MODEL_PROTOCOL)
