"""Experiment A: deterministic real combat, no policy/Blue/weapon mocks."""
import json
from dataclasses import replace
import numpy as np
import pytest
from algorithm.common.runtime import require_cuda
from env.aircombat import AirCombatConfig,AirCombatEnv
from env.aircombat.geometry import pair_advantages,relative_geometry
from env.aircombat.assignment import assignment_matrices
from env.aircombat.combat import resolve_attacks
from env.aircombat.rewards import compute_rewards

TRIM=np.tile([.5,1/3,1/8],(3,1))
C=AirCombatConfig()


@pytest.fixture(scope='module',autouse=True)
def runtime():
    require_cuda('cuda')


def options(case):
    red=np.array([[0,0,6000,200,0,0],[20000,10000,6000,200,0,0],[20000,-10000,6000,200,0,0]],float)
    blue=np.array([[1000,0,6000,200,0,0],[30000,10000,6000,200,0,0],[30000,-10000,6000,200,0,0]],float)
    mask=[1,0,0]
    if case=='A2':blue[1]=[-1000,0,6000,200,0,0];mask=[1,1,0]
    if case=='A3':blue[:2,:3]=[[1000,-100,6000],[1000,100,6000]];mask=[1,1,0]
    return dict(friendly_states=red,adversarial_states=blue,friendly_alive=[1,0,0],adversarial_alive=mask)


def actual_step(env):
    s=env.state
    before=dict(red_lock=s.friendly_dwell.copy(),blue_lock=s.adversarial_dwell.copy(),
                red_alive=s.friendly_alive.copy(),blue_alive=s.adversarial_alive.copy(),aph=s.assignments.copy())
    obs,reward,term,trunc,info=env.step(TRIM)
    assert obs.shape==(3,12) and info['centralized_state'].shape==(60,)
    assert s.friendly_states.shape==s.adversarial_states.shape==(3,6)
    np.testing.assert_array_equal(info['red_controls'][0],[0.,0.,1.])
    _,ra,ba,p=pair_advantages(s.friendly_states,s.adversarial_states,env.config)
    live=before['red_alive'][:,None]&before['blue_alive'][None,:]
    # This is an explicitly labelled reconstruction of the transient threshold
    # value, NOT a recorded post-return counter. The real function immediately
    # zeros counters involving dead slots after emitting an event.
    red_at_resolution=np.where(live&(ra>0),before['red_lock']+1,0)
    blue_at_resolution=np.where(live.T&(ba>0),before['blue_lock']+1,0)
    for event in info['events']:
        counters=red_at_resolution if event['attacker_side']=='Red' else blue_at_resolution
        assert counters[event['attacker_id'],event['target_id']]>=env.config.attack_dwell_steps
    np.testing.assert_array_equal(reward,sum(info['reward_components'].values()))
    assert np.isfinite(obs).all() and np.isfinite(info['centralized_state']).all() and np.isfinite(reward).all()
    assert np.isfinite(s.friendly_states).all() and np.isfinite(s.adversarial_states).all()
    assert not info['numerical_failure'] and not info['failures'] and env.counters['integration_failures']==0
    for k,v in info['reward_components'].items():np.testing.assert_array_equal(v[1:],0)
    j=int(before['aph'][0])
    assert info['reward_components']['attack'][0]==ra[0,j]-ba[j,0]
    facts=dict(step=s.step_count,time_s=s.step_count*env.config.dt,aph_before=before['aph'].tolist(),
        aph_after=info['assignments'].tolist(),red_attack=ra[0].tolist(),blue_attack_on_red0=ba[:,0].tolist(),
        red_lock_before=before['red_lock'][0].tolist(),blue_lock_before=before['blue_lock'][:,0].tolist(),
        red_lock_at_resolution_reconstructed=red_at_resolution[0].tolist(),
        blue_lock_at_resolution_reconstructed=blue_at_resolution[:,0].tolist(),
        red_lock_after=s.friendly_dwell[0].tolist(),blue_lock_after=s.adversarial_dwell[:,0].tolist(),
        blue_action_indices=info['blue_action_indices'].tolist(),blue_controls=info['blue_controls'].tolist(),
        red0_state=s.friendly_states[0].tolist(),blue_states=s.adversarial_states.tolist(),
        parts={k:float(v[0]) for k,v in info['reward_components'].items()},reward=float(reward[0]),
        events=info['events'],terminated=term,truncated=trunc,outcome=info['outcome'])
    geom=relative_geometry(s.friendly_states[:1],s.adversarial_states)
    reverse=relative_geometry(s.adversarial_states,s.friendly_states[:1])
    facts['red_geometry']=[dict(distance=float(geom.distance[0,k]),alpha_i_deg=float(np.degrees(geom.alpha_i[0,k])),
        alpha_j_deg=float(np.degrees(geom.alpha_j[0,k]))) for k in range(3)]
    facts['blue_geometry']=[dict(distance=float(reverse.distance[k,0]),alpha_i_deg=float(np.degrees(reverse.alpha_i[k,0])),
        alpha_j_deg=float(np.degrees(reverse.alpha_j[k,0]))) for k in range(3)]
    return facts


def emit(case,mode,seed,rows):
    print('EXPERIMENT_A_JSON '+json.dumps(dict(case=case,reward_mode=mode,seed=seed,rows=rows),allow_nan=False))


@pytest.mark.parametrize('mode',['paper','approach_gated'])
@pytest.mark.parametrize('seed',[7,100,1000])
def test_a1_real_blue_two_step_kill_and_rewards(mode,seed):
    e=AirCombatEnv(replace(C,posture_reward_mode=mode));e.reset(seed,options('A1'))
    g,ra,ba,_=pair_advantages(e.state.friendly_states,e.state.adversarial_states,e.config)
    assert ra[0,0]==C.attack_eta and ba[0,0]==0 and g.distance[0,0]==1000
    assert e.state.assignments.tolist()==[0,-1,-1]
    assert np.array_equal(e.state.friendly_states[0],[0,0,6000,200,0,0])
    first=actual_step(e);assert first['red_lock_after']==[1,0,0] and not first['events'] and not first['terminated']
    second=actual_step(e)
    np.testing.assert_array_equal(first['red0_state'],[50,0,6000,200,0,0])
    np.testing.assert_array_equal(second['red0_state'],[100,0,6000,200,0,0])
    assert second['red_lock_at_resolution_reconstructed'][0]==2 and second['red_lock_after']==[0,0,0]
    assert second['events']==[dict(attacker_side='Red',attacker_id=0,target_id=0,step=2,eligible_attackers=[0])]
    assert second['parts']['shootdown']==C.shootdown_r0
    assert second['parts']['terminal']==C.terminal_e1-np.floor(2/C.terminal_e2)==20
    assert second['parts']['attack']==5 and second['terminated'] and not second['truncated'] and second['outcome']=='win'
    assert not e.state.adversarial_alive.any() and e.state.friendly_alive[0]
    emit('A1',mode,seed,[first,second])


@pytest.mark.parametrize('mode',['paper','approach_gated'])
def test_a2_unassigned_blue_threat_is_real_but_absent_from_attack_reward(mode):
    e=AirCombatEnv(replace(C,posture_reward_mode=mode));e.reset(100,options('A2'))
    _,ra,ba,_=pair_advantages(e.state.friendly_states,e.state.adversarial_states,C)
    assert e.state.assignments[0]==0 and ba[0,0]==0 and ba[1,0]==5
    first=actual_step(e);second=actual_step(e)
    for row in (first,second):
        assert row['aph_before'][0]==0 and row['blue_attack_on_red0'][:2]==[0,5]
        assert row['parts']['attack']==5 # assigned B0 only, B1's -5 is absent
    assert first['blue_lock_after']==[0,1,0] and not first['events']
    assert second['blue_lock_at_resolution_reconstructed']==[0,2,0]
    assert second['events']==[
        dict(attacker_side='Red',attacker_id=0,target_id=0,step=2,eligible_attackers=[0]),
        dict(attacker_side='Blue',attacker_id=1,target_id=0,step=2,eligible_attackers=[1])]
    assert not e.state.friendly_alive.any() and e.state.adversarial_alive.tolist()==[False,True,False]
    assert second['outcome']=='red_eliminated' and second['terminated']
    assert second['parts']['shootdown']==10 and second['parts']['terminal']==0
    emit('A2',mode,100,[first,second])


@pytest.mark.parametrize('mode',['paper','approach_gated'])
def test_a3_one_red_really_kills_two_blue_with_single_aph_assignment(mode):
    e=AirCombatEnv(replace(C,posture_reward_mode=mode));e.reset(100,options('A3'))
    assert e.state.assignments[0]==0 and e.f[0].sum()==1
    _,ra,ba,_=pair_advantages(e.state.friendly_states,e.state.adversarial_states,C)
    np.testing.assert_array_equal(ra[0,:2],[5,5])
    first=actual_step(e);second=actual_step(e)
    assert first['red_lock_after']==[1,1,0] and not first['events']
    assert second['red_lock_at_resolution_reconstructed']==[2,2,0]
    assert second['events']==[
        dict(attacker_side='Red',attacker_id=0,target_id=0,step=2,eligible_attackers=[0]),
        dict(attacker_side='Red',attacker_id=0,target_id=1,step=2,eligible_attackers=[0])]
    assert second['parts']['shootdown']==2*C.shootdown_r0==20
    assert second['parts']['terminal']==20 and second['parts']['attack']==5
    assert second['outcome']=='win' and second['terminated']
    emit('A3',mode,100,[first,second])


def lower_step(s,step):
    """Explicit prescribed geometry snapshots, not a flying environment trace."""
    s.step_count=step
    s.assignments,ha,hb,f=assignment_matrices(s,C)
    _,ra,ba,posture=pair_advantages(s.friendly_states,s.adversarial_states,C)
    before=s.friendly_dwell.copy();assignment=s.assignments.copy()
    current=s.friendly_alive.copy()
    live=s.friendly_alive[:,None]&s.adversarial_alive[None,:]
    at_resolution=np.where(live&(ra>0),before+1,0)
    events=resolve_attacks(s,ra,ba,C.attack_dwell_steps)
    reward,parts=compute_rewards(s,current,assignment,ra,ba,posture,events,C)
    np.testing.assert_array_equal(reward,sum(parts.values()));assert np.isfinite(reward).all()
    return dict(step=step,aph_before=assignment.tolist(),red_attack=ra[0].tolist(),
        lock_before=before[0].tolist(),lock_at_resolution_reconstructed=at_resolution[0].tolist(),
        lock_after=s.friendly_dwell[0].tolist(),events=events,
        parts={k:float(v[0]) for k,v in parts.items()},reward=float(reward[0]))


def test_a4_interruption_resets_lock_real_geometry_and_real_resolver():
    e=AirCombatEnv();e.reset(100,options('A1'));s=e.state;rows=[]
    for step,y in enumerate((0,3000,0,0),1):
        s.adversarial_states[0,1]=y
        rows.append(lower_step(s,step))
    assert [row['red_attack'][0] for row in rows]==[5,0,5,5]
    assert [row['lock_after'][0] for row in rows]==[1,0,1,0]
    assert not any(r['events'] for r in rows[:3])
    assert rows[3]['events']==[dict(attacker_side='Red',attacker_id=0,target_id=0,step=4,eligible_attackers=[0])]
    assert [r['parts']['attack'] for r in rows]==[5,0,5,5]
    assert [r['parts']['shootdown'] for r in rows]==[0,0,0,10]
    emit('A4_interrupt','paper',100,rows)


def test_a4_aph_target_switch_does_not_reset_old_target_dwell():
    e=AirCombatEnv();e.reset(100,options('A3'));s=e.state
    # B0 has the better real posture initially, both remain in the attack cone.
    s.adversarial_states[:2,1]=[0,100]
    first=lower_step(s,1)
    # Only prescribed LOSs change; no direct assignments or count writes.
    # APH now genuinely prefers B1, while B0 remains eligible continuously.
    s.adversarial_states[:2,1]=[100,0]
    second=lower_step(s,2)
    assert first['aph_before'][0]==0 and second['aph_before'][0]==1
    assert first['lock_after']==[1,1,0] and second['lock_before']==[1,1,0]
    assert second['lock_at_resolution_reconstructed']==[2,2,0]
    assert {(e['attacker_id'],e['target_id']) for e in second['events']}=={(0,0),(0,1)}
    assert all(e['attacker_side']=='Red' and e['step']==2 for e in second['events'])
    assert second['parts']['attack']==5 and second['parts']['shootdown']==20
    emit('A4_switch','paper',100,[first,second])
