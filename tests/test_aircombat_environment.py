"""Hand event/reward cases and complete seeded bilateral episodes."""
import copy
import numpy as np
import pytest
from env.aircombat import AirCombatEnv,AirCombatConfig
from env.aircombat.scenarios import generate
from env.aircombat.combat import resolve_attacks
from env.aircombat.geometry import pair_advantages
from env.aircombat.observations import observe
from env.aircombat.rewards import allied_penalty,compute_rewards,COMPONENTS

C = AirCombatConfig()
TRIM = np.array([.5,1/3,1/8])


def state():
    return generate(C,np.random.default_rng(9))


def test_two_full_steps_exit_reset_and_unique_victim_event():
    s = state()
    ra,ba = np.zeros((3,3)),np.zeros((3,3))
    ra[:2,0]=5
    assert resolve_attacks(s,ra,ba,2)==[]
    assert s.adversarial_alive.all()
    resolve_attacks(s,ra*0,ba,2)
    assert not s.friendly_dwell.any()
    assert resolve_attacks(s,ra,ba,2)==[]
    events = resolve_attacks(s,ra,ba,2)
    assert len(events)==1 and events[0]['attacker_id']==0 and events[0]['eligible_attackers']==[0,1]
    assert not s.adversarial_alive[0]
    assert resolve_attacks(s,ra,ba,2)==[]
    assert len(s.events)==1


@pytest.mark.parametrize('side',['Red','Blue'])
def test_both_sides_can_kill(side):
    s = state()
    ra,ba = np.zeros((3,3)),np.zeros((3,3))
    (ra if side=='Red' else ba)[1,2]=5
    resolve_attacks(s,ra,ba,2)
    events = resolve_attacks(s,ra,ba,2)
    assert events[0]['attacker_side']==side and events[0]['target_id']==2
    assert not (s.adversarial_alive if side=='Red' else s.friendly_alive)[2]


def test_simultaneous_elimination_applied_after_both_event_sets():
    s = state()
    ra=ba=np.eye(3)*5
    resolve_attacks(s,ra,ba,2)
    events = resolve_attacks(s,ra,ba,2)
    assert len(events)==6 and not s.friendly_alive.any() and not s.adversarial_alive.any()
    s.step_count=50
    rewards,parts = compute_rewards(s,np.ones(3,dtype=bool),np.arange(3),ra,ba,np.zeros((3,3)),events,C)
    np.testing.assert_array_equal(parts['terminal'],[-2]*3)
    np.testing.assert_array_equal(parts['shootdown'],[10]*3)
    np.testing.assert_array_equal(rewards,[8]*3)


def test_allied_penalty_shared_threats_and_live_filter_hand_answers():
    red=np.array([[0,0,0,200,0,0],[50,0,0,200,0,0],[500,0,0,200,0,0]],dtype=float)
    ba=np.array([[5,5,0],[0,5,5]],dtype=float)
    # For UAV0: (.5+1 + 0+0)/2; UAV1: (.5+1 + 0+1)/2.
    np.testing.assert_allclose(allied_penalty(red,np.ones(3,bool),np.ones(2,bool),ba,C),[-.75,-1.25,-.5])
    np.testing.assert_allclose(allied_penalty(red,np.array([1,1,0],bool),np.array([0,1],bool),ba,C),[-.5,-.5,0])
    assert not allied_penalty(red,np.array([1,0,0],bool),np.ones(2,bool),ba,C).any()


def test_five_rewards_independent_hand_values_and_no_hidden_terms():
    s=state()
    s.friendly_states[:,:3]=[[0,0,6000],[500,0,6000],[1000,0,6000]]
    s.step_count=51
    s.adversarial_alive[:]=False
    s.friendly_alive[:]=[1,1,0]
    ra=np.diag([5.,0.,5.]); ba=np.diag([0.,5.,0.]); posture=np.diag([2.,-.01,.5])
    events=[dict(attacker_side='Red',attacker_id=0,target_id=0)]
    current=np.array([1,1,0],bool)
    reward,parts=compute_rewards(s,current,np.arange(3),ra,ba,posture,events,C)
    assert tuple(parts)==COMPONENTS
    np.testing.assert_allclose(parts['attack'],[5,-5,0])
    np.testing.assert_allclose(parts['posture'],[2,-.01,0])
    np.testing.assert_allclose(parts['allied'],[0,0,0])
    np.testing.assert_allclose(parts['shootdown'],[10,0,0])
    np.testing.assert_allclose(parts['terminal'],[38,38,0])
    np.testing.assert_allclose(reward,[55,32.99,0])
    np.testing.assert_array_equal(reward,sum(parts.values()))
    s.adversarial_alive[2]=True
    _,parts=compute_rewards(s,current,np.arange(3),ra,ba,posture,[],C)
    assert not parts['terminal'].any() and not parts['shootdown'].any()


def test_raw_observations_padding_and_live_unassigned_are_distinct():
    s=state()
    s.assignments[:]=[1,-1,-1]
    s.friendly_alive[2]=False
    obs,central=observe(s,C)
    g,_,_,_=pair_advantages(s.friendly_states,s.adversarial_states,C)
    r,b=s.friendly_states[0],s.adversarial_states[1]
    np.testing.assert_allclose(obs[0],[1,*r[3:6],g.alpha_i[0,1],r[2],*b[3:6],g.alpha_j[0,1],b[2],g.distance[0,1]])
    assert obs[1,0]==1 and obs[1,1]==s.friendly_states[1,3]
    assert not obs[1,[4,6,7,8,9,10,11]].any()
    assert not obs[2].any() and not central[24:].any()
    np.testing.assert_array_equal(central[:24],obs[:2].flatten())


def chase_options():
    # One live pair; other IDs retained. Blue's steady flight has highest score.
    red=np.array([[0,0,6000,200,0,0],[0,8000,6000,200,0,0],[0,9000,6000,200,0,0]],float)
    blue=red.copy(); blue[0,0]=1000; blue[1:,0]=10000
    return dict(friendly_states=red,adversarial_states=blue,friendly_alive=[1,0,0],adversarial_alive=[1,0,0])


def test_environment_death_current_reward_next_mask_and_frozen_dead_slots(monkeypatch):
    import env.aircombat.environment as module
    # Fix opponent choice only to isolate physical event integration.
    monkeypatch.setattr(module,'choose_blue_controls',lambda s,c,r:(np.tile([0,0,1.],(3,1)),np.zeros(3,int)))
    e=AirCombatEnv(); obs,info=e.reset(1,chase_options())
    dead_red=e.state.friendly_states[1:].copy(); dead_blue=e.state.adversarial_states[1:].copy()
    actions=np.tile(TRIM,(3,1)); actions[1:]=np.nan
    _,r,term,trunc,info=e.step(actions)
    assert not term and not trunc and not info['events']
    np.testing.assert_allclose(r,[7,0,0])
    _,r,term,trunc,info=e.step(actions)
    assert term and not trunc and info['win'] and len(info['events'])==1
    assert r[0]==pytest.approx(37) # attack5+posture2+r0(10)+terminal20
    np.testing.assert_array_equal(e.state.friendly_states[1:],dead_red)
    np.testing.assert_array_equal(e.state.adversarial_states[1:],dead_blue)
    assert info['current_active_mask'][0] and info['next_active_mask'][0]
    np.testing.assert_array_equal(info['assignments'][1:],[-1,-1])
    with pytest.raises(RuntimeError): e.step(actions)


def test_killed_red_current_transition_and_future_inactive(monkeypatch):
    import env.aircombat.environment as module
    monkeypatch.setattr(module,'choose_blue_controls',lambda s,c,r:(np.tile([0,0,1.],(3,1)),np.zeros(3,int)))
    o=chase_options(); o['friendly_states'][0,0]=1000; o['adversarial_states'][0,0]=0
    o['friendly_alive']=[1,1,0]; o['adversarial_alive']=[1,1,0]
    e=AirCombatEnv(); e.reset(1,o)
    e.step(np.tile(TRIM,(3,1)))
    _,reward,term,_,info=e.step(np.tile(TRIM,(3,1)))
    assert not term and info['current_active_mask'][0] and not info['next_active_mask'][0]
    assert reward[0]!=0
    frozen=e.state.friendly_states[0].copy()
    for _ in range(2):
        actions=np.tile(TRIM,(3,1)); actions[0]=np.nan
        obs,reward,_,_,info=e.step(actions)
        assert not obs[0].any() and reward[0]==0 and info['assignments'][0]==-1
        np.testing.assert_array_equal(e.state.friendly_states[0],frozen)


def test_live_unassigned_executes_actor_control_without_idle_trim(monkeypatch):
    import env.aircombat.environment as module
    monkeypatch.setattr(module,'choose_blue_controls',lambda s,c,r:(np.tile([0,0,1.],(3,1)),np.zeros(3,int)))
    o=chase_options();o['friendly_alive']=[1,1,1]
    e=AirCombatEnv();obs,info=e.reset(1,o)
    np.testing.assert_array_equal(info['assignments'],[0,-1,-1])
    assert info['active_mask'].all() and obs[1,0]==1
    actions=np.tile([.75,1.,.25],(3,1))
    _,_,_,_,info=e.step(actions)
    np.testing.assert_allclose(info['red_controls'],np.tile([np.pi/2,2,2],(3,1)))


def test_environment_simultaneous_both_sides_and_current_death_reward(monkeypatch):
    import env.aircombat.environment as module
    monkeypatch.setattr(module,'choose_blue_controls',lambda s,c,r:(np.tile([0,0,1.],(3,1)),np.zeros(3,int)))
    o=chase_options()
    o['friendly_states'][1,:2]=[1000,6000]
    o['adversarial_states'][1,:2]=[0,6000]
    o['friendly_alive']=o['adversarial_alive']=[1,1,0]
    e=AirCombatEnv();e.reset(1,o)
    e.step(np.tile(TRIM,(3,1)))
    _,r,term,trunc,info=e.step(np.tile(TRIM,(3,1)))
    assert not term and not trunc
    assert {event['attacker_side'] for event in info['events']}=={'Red','Blue'}
    assert info['blue_shot_down']==1 and info['red_survivors']==1
    assert info['current_active_mask'][1] and not info['next_active_mask'][1]
    assert r[1]!=0 and info['reward_components']['shootdown'][0]==C.shootdown_r0
    frozen=e.state.adversarial_states[0].copy()
    _,_,_,_,info=e.step(np.tile(TRIM,(3,1)))
    np.testing.assert_array_equal(e.state.adversarial_states[0],frozen)
    assert not info['reward_components']['shootdown'].any()


def test_physical_mutual_elimination_hexagon_contract(monkeypatch):
    import env.aircombat.environment as module
    monkeypatch.setattr(module,'choose_blue_controls',lambda s,c,r:(np.tile([0,0,1.],(3,1)),np.zeros(3,int)))
    angle=np.arange(6)*np.pi/3
    positions=np.column_stack((1000*np.cos(angle),1000*np.sin(angle),np.full(6,6000.)))
    headings=np.arctan2(np.roll(positions,-1,axis=0)[:,1]-positions[:,1],
                        np.roll(positions,-1,axis=0)[:,0]-positions[:,0])
    planes=np.column_stack((positions,np.full(6,200.),np.zeros(6),headings))
    e=AirCombatEnv();e.reset(1,dict(friendly_states=planes[::2],adversarial_states=planes[1::2]))
    e.step(np.tile(TRIM,(3,1)))
    obs,r,term,trunc,info=e.step(np.tile(TRIM,(3,1)))
    assert term and not trunc and info['outcome']=='mutual_elimination' and info['win']
    assert info['blue_shot_down']==3 and info['red_survivors']==0
    assert len(info['events'])==6 and info['current_active_mask'].all()
    assert not info['next_active_mask'].any() and not obs.any()
    assert not info['reward_components']['terminal'].any() # floor(2/25)=0, zero survivors


@pytest.mark.parametrize('angle,target_heading',[(np.pi/3,0),(0,np.pi)])
def test_no_kill_for_bad_cone_or_target_direction(monkeypatch,angle,target_heading):
    import env.aircombat.environment as module
    monkeypatch.setattr(module,'choose_blue_controls',lambda s,c,r:(np.tile([0,0,1.],(3,1)),np.zeros(3,int)))
    o=chase_options()
    o['adversarial_states'][0,:3]=[1000*np.cos(angle),1000*np.sin(angle),6000]
    o['adversarial_states'][0,5]=target_heading
    e=AirCombatEnv();e.reset(1,o)
    for _ in range(3):
        _,_,_,_,info=e.step(np.tile(TRIM,(3,1)))
        assert not info['events'] and e.state.adversarial_alive[0]


def test_vertical_flight_continues_without_fabricating_failure_or_shootdown():
    e=AirCombatEnv()
    o=chase_options();o['friendly_states'][0,4]=np.pi/2
    e.reset(2,o)
    before=copy.deepcopy(e.state)
    _,reward,term,trunc,info=e.step(np.tile(TRIM,(3,1)))
    assert not term and not trunc and not info['numerical_failure'] and not info['win']
    assert info['blue_shot_down']==0
    assert not info['events'] and np.isfinite(reward).all()
    np.testing.assert_array_equal(e.state.friendly_alive,before.friendly_alive)
    assert not np.array_equal(e.state.friendly_states,before.friendly_states)
    assert info['numerical_diagnostics']['continuation_substeps']>0


@pytest.mark.parametrize('scale',['3v3','5v5','3v5'])
def test_full_episode_reproducible_and_timeout(scale):
    c=AirCombatConfig(scale=scale)
    a,b=AirCombatEnv(c),AirCombatEnv(c)
    oa,ia=a.reset(987);ob,ib=b.reset(987)
    np.testing.assert_array_equal(oa,ob)
    for t in range(250):
        action=np.tile(TRIM,(c.num_uavs,1))
        ra,rb=a.step(action),b.step(action)
        for k in range(4):
            np.testing.assert_array_equal(ra[k],rb[k])
        assert ra[4]['events']==rb[4]['events']
        assert not ra[4]['numerical_failure']
        assert np.isfinite(ra[0]).all() and np.isfinite(ra[1]).all()
        np.testing.assert_array_equal(ra[1],sum(ra[4]['reward_components'].values()))
        if ra[2] or ra[3]:
            assert t<250
            break
    else:
        pytest.fail('episode did not finish')
    if ra[3]: assert t==249 and not ra[2]


def test_deterministic_scenario_and_environment_snapshot_exact():
    a=AirCombatEnv();a.reset(5,{'scenario':'head_on_3v3'})
    for _ in range(3): a.step(np.tile(TRIM,(3,1)))
    snapshot=a.snapshot()
    b=AirCombatEnv();b.restore(snapshot)
    for _ in range(3):
        x,y=a.step(np.tile(TRIM,(3,1))),b.step(np.tile(TRIM,(3,1)))
        np.testing.assert_array_equal(x[0],y[0]);np.testing.assert_array_equal(x[1],y[1])
        np.testing.assert_array_equal(x[4]['blue_action_indices'],y[4]['blue_action_indices'])
    assert a.rng.bit_generator.state==b.rng.bit_generator.state


def test_exact_250_step_timeout():
    o=chase_options()
    o['adversarial_states'][:,0]=1e6
    e=AirCombatEnv();e.reset(1,o)
    for t in range(250):
        _,_,term,trunc,info=e.step(np.tile(TRIM,(3,1)))
        assert not term
        assert trunc==(t==249)
    assert not info['win'] and info['outcome']=='timeout'
