from dataclasses import replace
import json
import sys
import zipfile

import numpy as np
import pytest
import torch

from algorithm.common.interception_checkpoint import read, restore, save
from algorithm.common.vector_env import ParallelVectorEnv
from algorithm.mappo.interception import MAPPO, TrainConfig
from algorithm.navigation import NavigationAccumulator, PROTOCOL, read_checkpoint, validate_discount
from env.heading_potential import HeadingConfig, direction_quality, heading_transition
from env.navigation import NavigationConfig, NavigationEnv


@pytest.mark.parametrize(("v","point","expected","angle"),[
    ([1,0,0],[10,0,0],.5,0),
    ([-1,0,0],[10,0,0],-.5,180),
    ([0,1,0],[10,0,0],0,90),
    ([0,0,0],[10,0,0],0,None),
    ([1,0,0],[0,0,0],0,None),
    ([1e-300,0,0],[10,0,0],1e-300,0),
    ([1,0,0],[1e-300,0,0],1e-301,0),
    ([1,2,2],[2,4,4],.75,0),
])
def test_bounded_3d_quality(v,point,expected,angle):
    result=direction_quality([[0,0,0,*v]],point,[True],HeadingConfig())
    assert np.isfinite(result["quality"]).all()
    assert result["quality"][0] == pytest.approx(expected,rel=1e-12,abs=0.)
    if angle is None:
        assert result["angle_deg"][0] is None
    else:
        assert result["angle_deg"][0] == pytest.approx(angle,abs=1e-6)


def test_potential_sign_constant_direction_weight_and_mask_normalization():
    base=HeadingConfig(heading_enabled=True)
    bad=np.array([[0,0,0,-1,0,0],[0,0,0,1,0,0]],dtype=float)
    good=bad.copy()
    good[0,3]=1.
    w=[10,0,0]
    active=[True,True]
    def transition(a,b,c=base,next_mask=active):
        return heading_transition(a,b,w,active,next_mask,2,False,c)
    up=transition(bad,good)["reward"]
    assert up > 0 and transition(good,bad)["reward"] < 0
    assert transition(good,good)["reward"] < 0
    # A fixed negative potential produces the small opposite-sign discount term.
    assert transition(-good,-good)["reward"] == pytest.approx(.0015)
    assert transition(good,good,replace(base,heading_discount=1.))["reward"] == 0
    assert transition(bad,good,replace(base,heading_weight=.1))["reward"] == pytest.approx(up/3)
    assert transition(bad,good,replace(base,heading_weight=1.))["reward"] == pytest.approx(up/.3)
    dropped=transition(good,good,next_mask=[False,True])
    assert dropped["next"]["slots"] == [1]
    assert dropped["next_potential"] == pytest.approx(dropped["potential"]/2)
    assert transition(bad,good,replace(base,heading_enabled=False))["reward"] == 0


def action(vx=1.,vy=0.):
    u=np.full((5,3),.5)
    u[:,:2] += [vx/8,vy/8]
    return u


def make_env(**overrides):
    settings=dict(heading_enabled=True,start_jitter=(0,0,0),initial_velocity=(1.,0,0))
    return NavigationEnv(NavigationConfig(**(settings|overrides)))


def test_one_step_arrival_and_timeout_have_distinct_successor_potentials():
    env=make_env(waypoint=(1.55,0,9))
    env.reset(22)
    obs,rewards,terminated,truncated,info=env.step(action())
    assert terminated and not truncated and info["time"]==pytest.approx(.05)
    assert info["heading"]["next_potential"] == 0.
    assert not info["active_mask"].any() and not obs.any() and rewards[0] != 0
    env=make_env(max_steps=1)
    env.reset(22)
    _,_,terminated,truncated,info=env.step(action())
    assert not terminated and truncated and info["active_mask"][0]
    assert info["heading"]["next_potential"] > 0.
    with pytest.raises(RuntimeError):
        env.step(action())
    env.reset(23)
    _,_,_,_,next_info=env.step(action())
    assert info["heading"]==next_info["heading"]


@pytest.mark.parametrize("max_steps",[3,180])
def test_discounted_telescoping_on_complete_and_truncated_routes(max_steps):
    env=make_env(max_steps=max_steps)
    _,info=env.reset(19)
    tracker=NavigationAccumulator()
    tracker.reset(info)
    while True:
        _,rewards,terminated,truncated,info=env.step(action(3.))
        tracker.add(info,rewards)
        assert np.isfinite(rewards).all()
        if terminated or truncated:
            break
    summary=tracker.summary(info)
    assert abs(summary["shaping_identity_error"]) < 1e-14
    if max_steps==180:
        assert terminated and summary["discounted_shaping"] == pytest.approx(-tracker.initial_phi)
    else:
        assert truncated and tracker.next_phi != 0


def test_partial_arrival_keeps_active_team_and_uses_pre_step_reward_mask():
    env=make_env(members=2)
    env.reset(7)
    env.state.pursuers[0,0]=7.45
    _,rewards,terminated,truncated,info=env.step(action())
    assert not terminated and not truncated and info["arrivals"]==1
    assert info["heading"]["active_count"]==2 and info["heading"]["next"]["slots"]==[1]
    assert rewards[0]==rewards[1]
    frozen=info["states"][0].copy()
    _,rewards,_,_,info=env.step(action())
    assert info["heading"]["current"]["slots"]==[1] and rewards[0]==0
    np.testing.assert_array_equal(info["states"][0],frozen)


def test_boundary_is_a_true_terminal_with_zero_potential():
    env=make_env(start=(27.95,0,9))
    env.reset(4)
    _,_,terminated,truncated,info=env.step(action())
    assert terminated and not truncated and info["outcome"]=="boundary"
    assert info["time"]==pytest.approx(.05,abs=1e-7)
    assert info["heading"]["next_potential"]==0 and info["reward_parts"]["safety"]==-2


def test_heading_off_preserves_base_rewards_and_states():
    assert not HeadingConfig().heading_enabled
    off=make_env(heading_enabled=False)
    on=make_env()
    off.reset(12);on.reset(12)
    for _ in range(5):
        oa,ra,ta,xa,ia=off.step(action())
        ob,rb,tb,xb,ib=on.step(action())
        np.testing.assert_array_equal(oa,ob)
        assert (ta,xa)==(tb,xb)
        assert ia["reward_parts"]=={k:v for k,v in ib["reward_parts"].items() if k!="heading"}
        assert ia["heading"]["reward"]==0
        assert rb[0]-ra[0]==pytest.approx(ib["heading"]["reward"],abs=1e-7)
    with pytest.raises(ValueError,match="gamma"):
        validate_discount(on.config,TrainConfig(gamma=.98))


def test_historical_fixed_interception_reward_fingerprint(check_interception_history):
    check_interception_history("default", 87, "1b66ce8760c6f1d569117885ee155546e566a6bcd3a85a4183f18a9fde21c85e")


def test_cuda_navigation_checkpoint_identity_and_exact_rollout(tmp_path):
    c=NavigationConfig(heading_enabled=True,max_steps=3)
    settings=TrainConfig(workers=1,horizon=5,epochs=1)
    model=MAPPO(settings)
    with ParallelVectorEnv(c,1,settings.seed,environment_factory=NavigationEnv,
                           accumulator_factory=NavigationAccumulator) as vector:
        rollout,completed=model.collect(vector)
        assert len(completed)==1 and completed[0]["outcome"]=="timeout"
        assert rollout.observations.shape[-1]==52 and rollout.states.shape[-1]==54
        assert model.update(rollout)["valid_samples"]==5
        path=tmp_path/"nav.pt"
        save(path,model,c,vector,protocol=PROTOCOL)
        with pytest.raises(ValueError,match="protocol"):
            read(path)
        payload=read(path,protocol=PROTOCOL)
        expected,_=model.collect(vector)
        with pytest.raises(ValueError,match="configuration"):
            restore(payload,model,replace(c,heading_enabled=False),vector,protocol=PROTOCOL)
        restore(payload,model,c,vector,protocol=PROTOCOL)
        actual,_=model.collect(vector)
        for name,value in vars(expected).items():
            torch.testing.assert_close(value,getattr(actual,name),rtol=0,atol=0)


def test_navigation_checkpoint_reads_original_cli_tracker(tmp_path, monkeypatch):
    path = tmp_path/"legacy_cli.pt"
    with monkeypatch.context() as patch:
        patch.setattr(sys.modules["__main__"],"NavigationAccumulator",NavigationAccumulator,raising=False)
        patch.setattr(NavigationAccumulator,"__module__","__main__")
        torch.save(dict(protocol=PROTOCOL,tracker=NavigationAccumulator()),path)
    with zipfile.ZipFile(path,"a") as archive:
        root=archive.namelist()[0].split("/")[0]
        archive.writestr(root+"/interception_protocol.json",json.dumps(PROTOCOL))
    assert isinstance(read_checkpoint(path)["tracker"],NavigationAccumulator)
