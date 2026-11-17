"""Causal PG observation, reward and state restoration contracts."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from env.ideal_interception import InterceptionEnv
from env.ideal_interception.config import (DEFENDER_SLOTS,OBS_DIM,STATE_DIM,PROTOCOL,
    formal_6v6_config,formal_6v6_level2_config,formal_15v15_level2_config)
from env.ideal_interception.learning import (PredictionGuidedInterceptionEnv,
    PG_OBS_DIM,build_pg_observation,usable_pip,guidance_reward,safety_reward,
    minimum_friendly_distances,event_rewards)
from prediction.intercept_prediction import estimate_three_point_pip_from_history


@pytest.mark.parametrize("factory,active_count",[(formal_6v6_level2_config,6),
                                                   (formal_15v15_level2_config,15)])
def test_base_protocol_and_pg_shape_blocks(factory,active_count):
    config=factory()
    base_observation,base_info=InterceptionEnv(config).reset(17)
    assert PROTOCOL=="uav-ideal-interception-v3.5.0"
    assert base_observation.shape==(DEFENDER_SLOTS,OBS_DIM)==(15,265)
    assert base_info["centralized_state"].shape==(STATE_DIM,)==(444,)
    env=PredictionGuidedInterceptionEnv(config)
    observation,info=env.reset(17)
    assert observation.shape==(15,PG_OBS_DIM)==(15,48)
    assert info["centralized_state"].shape==(444,)
    assert np.all(observation[active_count:]==0)
    own=info["defenders"][0]
    target=info["intruders"][info["assignment"][0],:3]
    relative=target-own[:3]
    np.testing.assert_allclose(observation[0,:3],
        (own[:3]-config.protected_center)/config.position_scale,rtol=1e-6)
    np.testing.assert_allclose(observation[0,3:6],own[3:]/config.speed_scale)
    np.testing.assert_allclose(observation[0,6:9],relative/np.linalg.norm(relative))
    assert observation[0,9]==pytest.approx(np.linalg.norm(relative)/config.position_scale)
    assert np.all(observation[0,10:19]==0)
    assert observation[0,47]==1.
    distances=sorted((np.linalg.norm(info["defenders"][k,:3]-own[:3]),k)
        for k in range(1,active_count))[:4]
    for rank,(_,k) in enumerate(distances):
        start=19+7*rank
        np.testing.assert_allclose(observation[0,start:start+3],
            (info["defenders"][k,:3]-own[:3])/config.position_scale,rtol=1e-6)
        assert observation[0,start+6]==1.


def test_three_point_history_velocity_and_true_velocity_isolation():
    config=formal_6v6_level2_config()
    env=PredictionGuidedInterceptionEnv(config)
    observation,info=env.reset(17)
    action=np.full((DEFENDER_SLOTS,3),.5)
    for index in range(2):
        observation,_,_,_,info=env.step(action)
        if index==0:
            assert np.all(observation[0,10:19]==0)
    j=info["assignment"][0]
    prediction=estimate_three_point_pip_from_history(env.history,j,
        info["defenders"][0,:3],config.defender_speeds[0])
    assert prediction.status=="READY"
    assert observation[0,13]==1. and observation[0,18]==1.
    np.testing.assert_allclose(observation[0,10:13],
        np.asarray(prediction.estimated_target_velocity_mps)/config.speed_scale,rtol=1e-6)
    delta=np.asarray(prediction.predicted_target_position_m)-info["defenders"][0,:3]
    np.testing.assert_allclose(observation[0,14:17],
        delta/np.linalg.norm(delta),rtol=1e-6)
    assert np.linalg.norm(observation[0,14:17])==pytest.approx(1.,abs=1e-6)
    assert observation[0,17]==pytest.approx(prediction.horizon_s/(config.dt*config.max_steps))
    modified=deepcopy(info)
    modified["intruders"][:,3:]=np.nan
    same,_=build_pg_observation(modified,env.history,config)
    np.testing.assert_array_equal(same,observation)


@pytest.mark.parametrize("delta,expected",[
    ((.1,0.,0.),(1.,0.,0.)),
    ((100.,0.,0.),(1.,0.,0.)),
    ((1.,2.,2.),(1/3,2/3,2/3)),
])
def test_pip_direction_is_unit_length_at_near_and_far_ranges(monkeypatch,delta,expected):
    config=formal_6v6_level2_config()
    env=PredictionGuidedInterceptionEnv(config)
    _,info=env.reset(17)
    own=info["defenders"][0,:3]
    point=tuple(own+np.asarray(delta))
    prediction=SimpleNamespace(status="READY",horizon_s=1.,
        predicted_target_position_m=point,estimated_target_velocity_mps=(1.,0.,0.))
    import env.ideal_interception.learning as learning
    monkeypatch.setattr(learning,"estimate_three_point_pip_from_history",lambda *args:prediction)
    observation,priors=build_pg_observation(info,env.history,config)
    np.testing.assert_allclose(observation[0,14:17],expected,rtol=1e-6,atol=1e-6)
    assert np.linalg.norm(observation[0,14:17])==pytest.approx(1.,abs=1e-6)
    assert observation[0,17]==pytest.approx(1./(config.dt*config.max_steps))
    assert observation[0,18]==1. and priors[0]["pip_valid"]


def test_coincident_pip_is_invalid_and_guidance_falls_back(monkeypatch):
    config=formal_6v6_level2_config()
    env=PredictionGuidedInterceptionEnv(config)
    _,info=env.reset(17)
    prediction=SimpleNamespace(status="READY",horizon_s=1.,
        predicted_target_position_m=tuple(info["defenders"][0,:3]),
        estimated_target_velocity_mps=(1.,0.,0.))
    import env.ideal_interception.learning as learning
    monkeypatch.setattr(learning,"estimate_three_point_pip_from_history",lambda *args:prediction)
    observation,priors=build_pg_observation(info,env.history,config)
    assert np.array_equal(observation[0,14:19],np.zeros(5))
    assert np.isfinite(observation).all()
    assert not priors[0]["pip_valid"] and priors[0]["pip_point"] is None
    np.testing.assert_array_equal(priors[0]["target_position"],
        info["intruders"][info["assignment"][0],:3])
    guidance_points=[]
    original_guidance=learning.guidance_reward
    def record_guidance(before,after,point,speed,dt):
        guidance_points.append(np.asarray(point).copy())
        return original_guidance(before,after,point,speed,dt)
    monkeypatch.setattr(learning,"guidance_reward",record_guidance)
    env.step(np.full((DEFENDER_SLOTS,3),.5))
    np.testing.assert_array_equal(guidance_points[0],priors[0]["target_position"])


def test_invalid_pip_zero_block_and_current_target_fallback(monkeypatch):
    config=formal_6v6_level2_config()
    env=PredictionGuidedInterceptionEnv(config)
    _,info=env.reset(17)
    j=info["assignment"][0]
    target=info["intruders"][j,:3]
    huge=SimpleNamespace(status="READY",horizon_s=100.,
        predicted_target_position_m=tuple(target),estimated_target_velocity_mps=(1.,0.,0.))
    assert not usable_pip(huge,0.,config)
    outside=SimpleNamespace(status="READY",horizon_s=1.,
        predicted_target_position_m=(999.,0.,60.),estimated_target_velocity_mps=(1.,0.,0.))
    assert not usable_pip(outside,0.,config)
    import env.ideal_interception.learning as learning
    monkeypatch.setattr(learning,"estimate_three_point_pip_from_history",lambda *args:huge)
    observation,priors=build_pg_observation(info,env.history,config)
    assert np.all(observation[0,14:19]==0)
    assert observation[0,13]==1.  # Velocity remains valid when PIP is invalid.
    assert not priors[0]["pip_valid"]
    assert np.isfinite(observation).all()
    before=info["defenders"][0,:3]
    after=before+np.array([1.,0.,0.])
    assert guidance_reward(before,after,priors[0]["target_position"],20.,.05)>0


def test_guidance_safety_and_continuous_pair_geometry():
    before=np.array((0.,0.,0.))
    point=np.array((100.,0.,0.))
    assert guidance_reward(before,(1.,0.,0.),point,20.,.05)==pytest.approx(.01)
    assert guidance_reward(before,(-1.,0.,0.),point,20.,.05)==pytest.approx(-.01)
    assert abs(guidance_reward(before,(0.,1.,0.),point,20.,.05))<.0001
    assert safety_reward(8.)==0.
    assert safety_reward(5.)==pytest.approx(-.00072)
    assert safety_reward(3.)==pytest.approx(-.002)
    assert safety_reward(0.)==pytest.approx(-.020)
    assert safety_reward(2.999999)<safety_reward(3.)
    first=dict(time=0.,defender_mask=np.array([True,True]),
        defenders=np.array([[-5.,0.,0.,20.,0.,0.],[5.,0.,0.,-20.,0.,0.]]))
    last=dict(time=.5,executed_velocities=np.array([[20.,0.,0.],[-20.,0.,0.]]),events=[])
    distances=minimum_friendly_distances(first,last)
    assert distances=={0:0.,1:0.}
    assert safety_reward(distances[0])<0
    last["events"]=[dict(kind="member_inactive",time=.025,members=[0])]
    distances=minimum_friendly_distances(first,last)
    assert distances[0]==pytest.approx(9.)
    assert safety_reward(distances[0])==0.


def test_event_reward_ownership_and_mask():
    assignment={0:0,1:1,2:2}
    mask=np.zeros(DEFENDER_SLOTS,bool);mask[:3]=True
    def reward(kind,members):
        return event_rewards([dict(kind=kind,members=members)],assignment,mask)
    assert reward("interception_contact",[0,DEFENDER_SLOTS])[0]==10.
    assert reward("interception_contact",[0,DEFENDER_SLOTS+1])[0]==-15.
    assert reward("region_breach",[DEFENDER_SLOTS+2])[2]==-12.
    assert reward("friendly_collision",[0,1])[:2].tolist()==[-15.,-15.]
    assert reward("boundary_exit",[1])[1]==-15.
    assert not reward("intruder_collision",[DEFENDER_SLOTS,DEFENDER_SLOTS+1]).any()
    assert not reward("member_inactive",[0]).any()
    mask[2]=False
    assert not reward("region_breach",[DEFENDER_SLOTS+2]).any()


def test_learning_reward_mask_and_snapshot_restore():
    config=formal_6v6_level2_config()
    env=PredictionGuidedInterceptionEnv(config)
    env.reset(17)
    neutral=np.full((DEFENDER_SLOTS,3),.5)
    env.step(neutral)
    snapshot=env.snapshot()
    restored=PredictionGuidedInterceptionEnv(config)
    restored.restore(snapshot)
    a=env.step(neutral)
    b=restored.step(neutral)
    np.testing.assert_array_equal(a[0],b[0])
    np.testing.assert_array_equal(a[1],b[1])
    assert a[4]["events"]==b[4]["events"]
    assert a[4]["learning_reward_parts"]==b[4]["learning_reward_parts"]
    assert not a[1][6:].any()
    assert a[1].dtype==np.float32 and a[1].shape==(DEFENDER_SLOTS,)
    assert a[4]["learning_reward_sum"]==pytest.approx(float(a[1].sum()))
    assert set(a[4]["learning_reward_parts"])=={"guide","safe","event","time"}
    assert a[4]["learning_reward_sum"]!=a[4]["team_reward"]


def test_step_start_active_defender_receives_own_contact_reward():
    config=formal_6v6_config(intruder_mode="straight",random_range=0.,
                             intruder_start_jitter_m=0.)
    env=PredictionGuidedInterceptionEnv(config)
    _,info=env.reset(17)
    owner=next(i for i,j in info["original_assignment"].items() if j==0)
    velocity=env.base.intruders[0,3:].copy()
    heading=velocity/np.linalg.norm(velocity)
    env.base.defenders[owner,:3]=env.base.intruders[0,:3]+heading
    env.base.defenders[owner,3:]=-velocity
    action=np.full((DEFENDER_SLOTS,3),.5)
    action[owner]=.5-.5*heading
    _,rewards,_,_,after=env.step(action)
    assert any(e["kind"]=="interception_contact" and e["members"][0]==owner
               for e in after["events"])
    assert not after["defender_mask"][owner]
    assert rewards[owner]>9.
    assert after["learning_reward_parts"]["event"]>=10.
