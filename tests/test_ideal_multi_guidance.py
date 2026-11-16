"""V3.4 six-on-six geometry, fixed matching, events, and causal guidance."""
from dataclasses import replace
from itertools import permutations
import json
import shutil
import zipfile

import numpy as np
import pytest
import torch

from algorithm.ideal_interception import InterceptionMAPPO,vector_env,PROTOCOL as LEARNING_PROTOCOL
from algorithm.common.interception_checkpoint import read
from algorithm.ideal_multi_predictive_guidance import METHODS,run_episode
from tools.ideal_multi_guidance_benchmark import benchmark
from algorithm.mappo.interception import TrainConfig
from env.ideal_interception import InterceptionEnv
from env.ideal_interception.assignment import minimum_cost_assignment
from env.ideal_interception.config import (PROTOCOL,OBS_DIM,STATE_DIM,
    DEFENDER_SLOTS,INTRUDER_SLOTS,SIX_DEFENDER_CIRCLE,SIX_INTRUDER_WEDGE,
    formal_6v6_config)
from tools.ideal_interception_timeline import Timeline
from tools.ideal_interception_trace import record,write_trace
from tools.ideal_interception_replay import render_html,render_mp4


def test_v33_slots_circle_wedge_and_seeded_paths():
    assert (PROTOCOL,DEFENDER_SLOTS,INTRUDER_SLOTS,OBS_DIM,STATE_DIM)==(
        "uav-ideal-interception-v3.4.0",6,6,112,128)
    config=formal_6v6_config()
    assert config.defender_starts==SIX_DEFENDER_CIRCLE
    assert config.intruder_starts==SIX_INTRUDER_WEDGE
    circle=np.asarray(SIX_DEFENDER_CIRCLE)
    np.testing.assert_allclose(circle.mean(axis=0),(-120,0,60),atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(circle[:,:2]-(-120,0),axis=1),50,atol=1e-10)
    assert [row[0] for row in SIX_INTRUDER_WEDGE]==[190,210,210,230,230,230]
    assert [row[1] for row in SIX_INTRUDER_WEDGE]==[0,-30,30,-60,0,60]
    env=InterceptionEnv(config)
    obs,a=env.reset(17)
    with pytest.raises(ValueError,match="incompatible"):
        env.restore(env.snapshot()|{"protocol":"uav-ideal-interception-v3.2.0"})
    b=InterceptionEnv(config).reset(17)[1]
    c=InterceptionEnv(config).reset(18)[1]
    assert obs.shape==(6,112) and a["centralized_state"].shape==(128,)
    assert a["scenario_id"]==b["scenario_id"]!=c["scenario_id"]
    assert a["intruder_paths"]==b["intruder_paths"]!=c["intruder_paths"]
    starts=np.asarray(a["intruders"][:6,:3])
    assert np.all(np.abs(starts-np.asarray(SIX_INTRUDER_WEDGE))<=5)
    assert len({row["polyline_sha256"] for row in a["intruder_paths"]})==6
    assert all(row["within_bounds"] and row["entry_within_episode"]
               for row in a["scene_diagnostics"]["paths"])
    assert all(np.all(path.lengths>0) for path in env.paths)


def test_hungarian_optimum_tie_and_frozen_assignment():
    matrix=np.array([[9,2,7,8,4,6],[6,4,3,7,5,8],[5,8,1,8,3,4],
                     [7,6,9,4,2,3],[3,5,7,6,8,2],[8,3,4,5,6,1]],float)
    assignment=minimum_cost_assignment(matrix)
    assert set(assignment)==set(range(6)) and set(assignment.values())==set(range(6))
    cost=sum(matrix[i,j] for i,j in assignment.items())
    brute=min(sum(matrix[i,j] for i,j in enumerate(p)) for p in permutations(range(6)))
    assert cost==pytest.approx(brute)
    assert minimum_cost_assignment(np.ones((6,6)))=={i:i for i in range(6)}
    config=formal_6v6_config()
    for seed in (17,18):
        env=InterceptionEnv(config)
        _,info=env.reset(seed)
        initial=info["original_assignment"]
        positions=info["defenders"][:6,:3],info["intruders"][:6,:3]
        cost_matrix=np.linalg.norm(positions[0][:,None]-positions[1][None],axis=2)
        assert sum(cost_matrix[i,j] for i,j in initial.items())==pytest.approx(
            min(sum(cost_matrix[i,j] for i,j in enumerate(p)) for p in permutations(range(6))))
        env.step(np.full((6,3),.5))
        assert env.original_assignment==initial
        assert env.assignment==initial


def _fixture(**changes):
    return formal_6v6_config(intruder_mode="straight",random_range=0.,
                             intruder_start_jitter_m=0.,**changes)


def test_single_member_events_continue_and_breached_pair_never_reassigns():
    base=_fixture()
    intruders=((-197.,0.,60.),)+base.intruder_starts[1:]
    breach=InterceptionEnv(replace(base,dt=.2,intruder_starts=intruders,
        intruder_ends=((-260.,0.,60.),)+base.intruder_ends[1:]))
    _,initial=breach.reset(17)
    owner=next(i for i,j in initial["original_assignment"].items() if j==0)
    _,_,done,_,info=breach.step(np.full((6,3),.5))
    assert not done and info["outcome"]=="ongoing" and info["breached_count"]==1
    assert info["intruder_status"][0]=="breached"
    assert info["defender_status"][owner]=="target_breached_no_reassignment"
    assert owner not in info["assignment"]
    assert info["original_assignment"]==initial["original_assignment"]
    breach.step(np.full((6,3),.5))
    assert breach.original_assignment==initial["original_assignment"]
    assert owner not in breach.assignment

    starts=((299.,0.,60.),)+base.defender_starts[1:]
    boundary=InterceptionEnv(replace(base,defender_starts=starts))
    boundary.reset(17)
    _,_,done,_,info=boundary.step(np.full((6,3),.5))
    assert not done and info["defender_boundary_count"]==1
    assert info["defender_status"][0]=="boundary_exit"

    starts=((188.,0.,60.),)+base.defender_starts[1:]
    contact=InterceptionEnv(replace(base,defender_starts=starts))
    contact.reset(17)
    _,_,done,_,info=contact.step(np.full((6,3),.5))
    assert not done and info["intercepted_count"]==1 and info["remaining_intruder_count"]==5

    starts=((-70.,-1.,60.),(-70.,1.,60.))+base.defender_starts[2:]
    friendly=InterceptionEnv(replace(base,defender_starts=starts))
    friendly.reset(17)
    action=np.full((6,3),.5)
    action[0]=(.5,1.,.5)
    action[1]=(.5,0.,.5)
    _,_,done,_,info=friendly.step(action)
    assert not done and info["friendly_collision_count"]==1
    assert not info["defender_mask"][:2].any()

    intruder_starts=((0.,-1.,60.),(0.,1.,60.))+base.intruder_starts[2:]
    intruder_ends=((0.,10.,60.),(0.,-10.,60.))+base.intruder_ends[2:]
    intruder_collision=InterceptionEnv(replace(base,intruder_starts=intruder_starts,
                                               intruder_ends=intruder_ends))
    intruder_collision.reset(17)
    _,_,done,_,info=intruder_collision.step(np.full((6,3),.5))
    assert not done and info["intruder_collision_count"]==1
    assert not info["intruder_mask"][:2].any()
    assert info["remaining_intruder_count"]==4


@pytest.mark.parametrize("prior_breaches,expected_outcome",[(0,"all_intercepted"),
                                                       (1,"mixed_resolution")])
def test_all_intruders_must_resolve_before_finalization(prior_breaches,expected_outcome):
    env=InterceptionEnv(_fixture())
    env.reset(17)
    owner=next(i for i,j in env.original_assignment.items() if j==0)
    for j in range(1,6):
        env.intruder_mask[j]=False
        env.intruder_status[j]="breached" if j<=prior_breaches else "intercepted"
        env.intruders[j,3:]=0.
    for i in range(6):
        if i!=owner:
            env.defender_mask[i]=False
            env.defender_status[i]="contact_exit"
            env.defenders[i,3:]=0.
    env.interception_count=5-prior_breaches
    env.breach_count=prior_breaches
    env.assignment={owner:0}
    env.defenders[owner,:3]=env.intruders[0,:3]-np.array([2.,0.,0.])
    env.defenders[owner,3:]=[20.,0.,0.]
    _,_,done,_,info=env.step(np.full((6,3),.5))
    assert done and info["outcome"]==expected_outcome
    assert info["intercepted_count"]==6-prior_breaches
    assert info["breached_count"]==prior_breaches
    assert info["remaining_intruder_count"]==0
    assert info["mission_success"]==(prior_breaches==0)


def test_two_methods_share_scene_and_member_resolution(tmp_path):
    results={method:run_episode(formal_6v6_config(),method,17) for method in METHODS}
    summaries={method:row[0] for method,row in results.items()}
    assert len({row["scenario_id"] for row in summaries.values()})==1
    assert len({tuple(row["path_hashes"]) for row in summaries.values()})==1
    assert all(row["original_assignment"]==summaries["two_point_pip"]["original_assignment"]
               for row in summaries.values())
    assert all(row["defender_starts"]==summaries["two_point_pip"]["defender_starts"] and
               row["intruder_starts"]==summaries["two_point_pip"]["intruder_starts"]
               for row in summaries.values())
    for method,(summary,trace) in results.items():
        assert summary["warmup_steps"]>0 and summary["predictive_steps_total"]>0
        assert summary["intercepted_count"]+summary["breached_count"]+summary["other_resolved_count"]==6
        assert all(np.shape(frame["control"]["action"])==(6,3) for frame in trace["frames"][1:])
        for frame in trace["frames"][1:]:
            for pair in frame["control"]["pairs"].values():
                assert np.linalg.norm(pair["guidance_direction_unit"])==pytest.approx(1.)
        path=tmp_path/f"{method}.json"
        write_trace(path,trace)
        assert len(Timeline(path).at(0)["entities"])==12
    assert all(summaries[method]["mission_success"] for method in METHODS)
    assert all(summaries[method]["intercepted_count"]==6 for method in METHODS)


def test_six_member_html_and_mp4_replay(tmp_path):
    source=tmp_path/"episode.json"
    write_trace(source,record(_fixture(max_steps=2),17))
    html=tmp_path/"episode.html"
    render_html(source,html)
    markup=html.read_text(encoding="utf-8")
    assert "D5" in markup and "I5" in markup
    assert "Fixed assignment: D0" in markup
    if shutil.which("ffmpeg"):
        mp4=tmp_path/"episode.mp4"
        render_mp4(source,mp4,fps=2)
        assert mp4.stat().st_size>0


def test_paired_benchmark_aggregates_member_counts():
    result=benchmark(formal_6v6_config(),range(17,19))
    assert result["paired_scenarios"] and result["seed_count"]==2
    for trial in result["trials"]:
        assert len(trial["path_hashes"])==6
        assert len(trial["original_assignment"])==6
    for method in METHODS:
        aggregate=result["aggregates"][method]
        rows=[trial["methods"][method] for trial in result["trials"]]
        assert aggregate["total_intercepted"]==sum(row["intercepted_count"] for row in rows)
        assert aggregate["total_breached"]==sum(row["breached_count"] for row in rows)
        assert aggregate["mission_success_trials"]==sum(row["mission_success"] for row in rows)
        assert aggregate["prediction_horizon_p95_s"] is not None
        assert 0<=aggregate["future_truth_coverage"]<=1


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required by repository policy")
def test_six_member_mappo_vector_interface_smoke(tmp_path):
    config=formal_6v6_config()
    training=TrainConfig(workers=1,horizon=1)
    model=InterceptionMAPPO(training)
    assert model.actor[0].in_features==112
    assert model.critic[0].in_features==128
    assert model.critic[-1].out_features==6
    with vector_env(config,training) as vector:
        assert vector.observations.shape==(1,6,112)
        assert vector.states.shape==(1,128)
        assert vector.masks.shape==(1,6)
        action,_=model.act(vector.observations,vector.masks,deterministic=True)
        assert action.shape==(1,6,3)
        result=vector.step(action.cpu().numpy())
        assert result.rewards.shape==(1,6)
        assert result.transition_states.shape==(1,128)
    old=tmp_path/"v32_header.pt"
    with zipfile.ZipFile(old,"w") as archive:
        archive.writestr("archive/interception_protocol.json",json.dumps(
            LEARNING_PROTOCOL|{"version":"uav-ideal-interception-v3.2.0",
                               "observation":68,"state":66,"defenders":5,"intruders":2}))
    with pytest.raises(ValueError,match="incompatible checkpoint protocol"):
        read(old,protocol=LEARNING_PROTOCOL)
    prior=tmp_path/"v33_header.pt"
    with zipfile.ZipFile(prior,"w") as archive:
        archive.writestr("archive/interception_protocol.json",json.dumps(
            LEARNING_PROTOCOL|{"version":"uav-ideal-interception-v3.3.0"}))
    with pytest.raises(ValueError,match="incompatible checkpoint protocol"):
        read(prior,protocol=LEARNING_PROTOCOL)
