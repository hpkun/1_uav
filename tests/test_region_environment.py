import copy
import json
import math
import numpy as np
import pytest
from env.region_competition import RegionConfig, RegionCompetitionEnv
from env.region_competition.regions import residence_interval, score_transition, centers_at
from env.region_competition.reward import compute_reward
from algorithm.common.serialization import serializable

TRIM=np.array([.5,1/3,1/8])


def canonical(value):
    return json.dumps(serializable(value),sort_keys=True,allow_nan=False)


@pytest.mark.parametrize('scale,n',[('3v3',3),('5v5',5)])
def test_both_teams_analytic_flight_and_dimensions(scale,n):
    cfg=RegionConfig(scale=scale,blue_policy='cruise')
    env=RegionCompetitionEnv(cfg)
    red=np.tile([0.,0.,3000.,200.,0.,0.],(n,1)); red[:,1]=np.arange(n)*300
    blue=red.copy(); blue[:,0]=5000; blue[:,5]=np.pi
    obs,info=env.reset(5,dict(red_states=red,blue_states=blue))
    assert obs.shape==(n,12) and info['centralized_state'].shape==(60,)
    if n==3: assert not info['centralized_state'][36:].any()
    obs,r,term,trunc,info=env.step(np.tile(TRIM,(n,1)))
    np.testing.assert_allclose(info['red']['states'][:,:3]-red[:,:3],np.tile([50,0,0],(n,1)),atol=1e-10)
    np.testing.assert_allclose(info['blue']['states'][:,:3]-blue[:,:3],np.tile([-50,0,0],(n,1)),atol=1e-10)
    for t in ('red','blue'):
        np.testing.assert_allclose(info[t]['states'][:,3:5],np.tile([200,0],(n,1)))
        np.testing.assert_allclose(sum(info[t]['reward_components'].values()),info[t]['rewards'])
    assert not term and not trunc


@pytest.mark.parametrize('mode',['cruise','region','region_safe'])
def test_seed_reproducibility_moving_regions_and_blue(mode):
    cfg=RegionConfig(blue_policy=mode,max_steps=6)
    a,b=RegionCompetitionEnv(cfg),RegionCompetitionEnv(cfg)
    _,ia=a.reset(28); _,ib=b.reset(28)
    assert canonical(ia)==canonical(ib)
    initial=copy.deepcopy(ia)
    for _ in range(6):
        aa=a.policy_actions('red','region_safe'); bb=b.policy_actions('red','region_safe')
        xa=a.step(aa); xb=b.step(bb); assert canonical(xa)==canonical(xb)
    ia=xa[-1]
    assert xa[3] and not xa[2] and ia['bootstrap_terminal']
    assert ia['remaining_steps']==0 and not xa[0][:,11].any()
    assert not np.array_equal(initial['blue']['states'],ia['blue']['states'])
    assert not np.array_equal(initial['regions'],ia['regions'])
    with pytest.raises(RuntimeError): a.step(aa)


def test_mirrored_initial_layout_and_shared_region_motion():
    env=RegionCompetitionEnv(); _,info=env.reset(31)
    red,blue=info['red']['states'],info['blue']['states']
    np.testing.assert_allclose(red[:,0],-blue[:,0])
    np.testing.assert_allclose(red[:,1:5],blue[:,1:5])
    np.testing.assert_allclose(np.sin(red[:,5]),np.sin(blue[:,5]))
    np.testing.assert_allclose(np.cos(red[:,5]),-np.cos(blue[:,5]))
    for time in (0,.25,62.5):
        c=centers_at(env.anchors,env.phase,time,env.config)
        assert not c[:,0].any()
        np.testing.assert_allclose(c[1]-c[0],env.anchors[1]-env.anchors[0])


@pytest.mark.parametrize('control',[(0.,0.,2.),(.3,0.,1/math.cos(.3))])
def test_both_teams_climb_or_turn_through_same_dynamics(control):
    env=RegionCompetitionEnv(); _,info=env.reset(4)
    options={t+'_states':info[t]['states'].copy() for t in ('red','blue')}
    for state in options.values(): state[:,4]=0
    env.reset(4,options)
    normalized=(np.array(control)+[np.pi,1,0])/[2*np.pi,3,8]
    before=copy.deepcopy(env.states)
    info=env.step_teams(np.tile(normalized,(3,1)),np.tile(normalized,(3,1)))[-1]
    for t in ('red','blue'):
        if control[0]==0:
            assert (info[t]['states'][:,4]>0).all() and (info[t]['states'][:,2]>before[t][:,2]).all()
        else:
            from env.dynamics import wrap_angle
            assert (wrap_angle(info[t]['states'][:,5]-before[t][:,5])>0).all()
            np.testing.assert_allclose(info[t]['states'][:,4],0,atol=1e-12)


def test_both_teams_have_separate_clip_counts_and_constraint_penalties():
    env=RegionCompetitionEnv(); _,info=env.reset(4)
    options={t+'_states':info[t]['states'].copy() for t in ('red','blue')}
    for state in options.values(): state[:,3]=100; state[:,4]=env.config.gamma_limit
    env.reset(4,options)
    info=env.step_teams(np.tile([.5,1/3,1],(3,1)),np.tile([.5,1/3,1],(3,1)))[-1]
    for t in ('red','blue'):
        assert info[t]['numerical_counters']['gamma_clips']==3
        assert info[t]['numerical_counters']['speed_clips']==3
        np.testing.assert_allclose(info[t]['reward_components']['numerical'],-.2)


def test_swept_residence_known_answers_and_no_stack_scoring():
    assert residence_interval([-2,0,0],[2,0,0],1)==pytest.approx((.25,.75))
    assert residence_interval([0,1,0],[0,1,0],1)==(0,1)
    assert residence_interval([-2,1,0],[2,1,0],1) is None
    cfg=RegionConfig(num_regions=1,region_radius=1)
    before={t:np.zeros((3,6)) for t in ('red','blue')}; after=copy.deepcopy(before)
    before['red'][:,:3]=[-2,0,0]; after['red'][:,:3]=[2,0,0]
    before['blue'][:,:3]=after['blue'][:,:3]=[20,0,0]
    masks={t:np.ones(3,dtype=bool) for t in before}; center=np.zeros((1,3))
    scoring=score_transition(before,after,center,center,masks,cfg,1)
    assert scoring['credits']['red'].sum()==pytest.approx(.125)
    np.testing.assert_allclose(scoring['credits']['red'],.125/3)
    np.testing.assert_allclose(scoring['dwell']['red'],.125)
    after['blue'][:,:3]=[2,0,0]; before['blue'][:,:3]=[-2,0,0]
    contested=score_transition(before,after,center,center,masks,cfg,1)
    assert not contested['credits']['red'].any() and not contested['credits']['blue'].any()


def test_tracking_cannot_credit_region_motion_and_terminal_once():
    cfg=RegionConfig(); state=np.tile([0,0,3000,200,0,0],(3,1)).astype(float)
    zero=np.zeros(3); events={k:zero.copy() for k in ('gamma_clips','speed_clips','nonfinite_states')}
    r,p=compute_reward(state,state,np.array([[100,0,3000]]),np.zeros(3,dtype=int),np.ones(3,bool),np.ones(3,bool),zero,events,1,False,cfg)
    assert not p['tracking'].any() and not p['result'].any()
    env=RegionCompetitionEnv(RegionConfig(max_steps=1)); env.reset(1)
    result=env.step(np.tile(TRIM,(3,1)))
    assert not result[2] and result[3] and result[-1]['finished']
    with pytest.raises(RuntimeError): env.step(np.tile(TRIM,(3,1)))


def test_invalid_slots_atomic_validation_true_terminal_and_snapshot():
    env=RegionCompetitionEnv(); env.reset(3,dict(red_active=[True,False,True]))
    initial=env.snapshot(); a=np.tile(TRIM,(3,1)); a[1]=np.nan
    old=env.states['red'][1].copy(); first=env.step(a)
    assert not first[0][1].any(); np.testing.assert_array_equal(old,env.states['red'][1])
    env.restore(initial); again=env.step(a); assert canonical(first)==canonical(again)
    snapshot=env.snapshot()
    with pytest.raises(ValueError): env.step_teams(np.tile(TRIM,(3,1)),np.full((3,3),2))
    assert canonical(env.snapshot())==canonical(snapshot)
    env.reset(3,dict(blue_active=[False,False,False]))
    _,_,term,trunc,info=env.step(np.tile(TRIM,(3,1)))
    assert term and not trunc and info['end_reason']=='no_active_team'


@pytest.mark.parametrize('option',[dict(num_regions=0),dict(max_steps=1500),dict(dt=.1),dict(region_period=0),dict(gamma_limit=math.pi/2)])
def test_invalid_configuration(option):
    with pytest.raises(ValueError): RegionConfig(**option)
