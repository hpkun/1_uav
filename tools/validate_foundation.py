"""Bounded physical/Blue/weapon acceptance, no learning or checkpoint loading."""
import argparse
import copy
import json
import math
from pathlib import Path
import pickle
import numpy as np
from algorithm.common.runtime import require_cuda
from env.aircombat import AirCombatConfig,AirCombatEnv
from env.aircombat.framed_dynamics import advance,initial_frame,frame_error,SimulationFault
from env.aircombat.opponent import BLUE_CONTROLS,candidate_scores,choose_blue_controls
from env.aircombat.geometry import relative_geometry,pair_advantages
from env.dynamics import physical_actions,velocity_vector,wrap_angle
from tests.test_foundation_blue import EXPECTED,independent_pair_score
from tests.test_experiment_a import options,TRIM

C=AirCombatConfig()


def reference_rhs(t,s,u):
    """Direct scalar Eqs.(1)-(2), independent verification reference only."""
    v,g,p=s[3:];mu,zeta,n=u;gravity=C.gravity
    return np.array([v*np.cos(g)*np.cos(p),v*np.cos(g)*np.sin(p),v*np.sin(g),
            gravity*(zeta-np.sin(g)),gravity/v*(n*np.cos(mu)-np.cos(g)),
            gravity*n*np.sin(mu)/(v*np.cos(g))])


def flight(name,control,steps=20,state=None):
    y=np.array([0,0,6000,200,0,0],float) if state is None else np.array(state,float)
    start=y.copy();f=initial_frame(y);states=[y.copy()];clips=0;orth=0.
    for _ in range(steps):
        y,f,d=advance(y,control,f,C);states.append(y.copy());clips+=d['speed_clips'];orth=max(orth,frame_error(f))
        np.testing.assert_allclose(velocity_vector(y)/y[3],f[:,0],atol=2e-13,rtol=0)
    duration=steps*C.dt
    # An independent, small-step angular RK4 oracle, NOT a simulator setting.
    # Runtime dt remains .25 and every actual transition uses existing advance.
    ref=start.copy();reference=[ref.copy()];h=C.dt/32
    for _ in range(steps):
        for _ in range(32):
            k1=reference_rhs(0,ref,control);k2=reference_rhs(0,ref+h*k1/2,control)
            k3=reference_rhs(0,ref+h*k2/2,control);k4=reference_rhs(0,ref+h*k3,control)
            ref=ref+h*(k1+2*k2+2*k3+k4)/6
        reference.append(ref.copy())
    observed=np.array(states);delta=observed-np.array(reference);delta[:,5]=wrap_angle(delta[:,5])
    errors=dict(position_m=float(np.linalg.norm(delta[:,:3],axis=1).max()),speed_m_s=float(abs(delta[:,3]).max()),
                gamma_rad=float(abs(delta[:,4]).max()),psi_rad=float(abs(delta[:,5]).max()))
    assert errors['position_m']<.05 and errors['speed_m_s']<.002 and max(errors['gamma_rad'],errors['psi_rad'])<1e-4
    assert orth<5e-14 and not clips
    return dict(name=name,control=list(control),steps=steps,duration_s=duration,initial=start.tolist(),final=y.tolist(),
        final_angles_deg=np.degrees(y[4:6]).tolist(),reference_errors=errors,frame_error=orth,speed_clips=clips,states=observed.tolist())


def flight_acceptance():
    mapped=physical_actions(np.array([[0,0,0],[.5,.5,.5],[1,1,1],[.5,1/3,1/8]]))
    np.testing.assert_array_equal(mapped,[[-np.pi,-1,0],[0,.5,4],[np.pi,2,8],[0,0,1]])
    cases=[flight('trim',[0,0,1]),flight('accelerate',[0,2,1]),flight('decelerate',[0,-1,1]),
        flight('turn_plus_30',[np.pi/6,0,1/np.cos(np.pi/6)]),
        flight('turn_minus_30',[-np.pi/6,0,1/np.cos(np.pi/6)]),
        flight('turn_plus_60',[np.pi/3,0,2]),flight('climb',[0,0,2]),flight('ballistic_dive',[0,0,0]),
        flight('large_roll',[np.radians(170),0,8],steps=4),
        flight('large_gamma',[.6,1,2],steps=4,state=[0,0,6000,200,np.pi/3,.4])]
    for case in cases:
        name=case['name'];t=case['duration_s'];expected=None
        if name in ('trim','accelerate','decelerate'):
            zeta=case['control'][1];expected=[200*t+.5*C.gravity*zeta*t*t,0,6000,200+C.gravity*zeta*t,0,0]
        elif name.startswith('turn_'):
            omega=C.gravity*np.tan(case['control'][0])/200
            expected=[200*np.sin(omega*t)/omega,200*(1-np.cos(omega*t))/omega,6000,200,0,omega*t]
        elif name=='ballistic_dive':expected=[200*t,0,6000-.5*C.gravity*t*t,np.hypot(200,C.gravity*t),-np.arctan2(C.gravity*t,200),0]
        if expected is not None:
            error=np.array(case['final'])-expected
            assert np.linalg.norm(error[:3])<.001 and max(abs(error[3:]))<1e-5
            case['analytic_expected']=expected;case['analytic_position_error_m']=float(np.linalg.norm(error[:3]))
    plus,minus=np.array(cases[3]['final']),np.array(cases[4]['final'])
    np.testing.assert_allclose(plus[[0,2,3,4]],minus[[0,2,3,4]],atol=2e-12,rtol=0)
    np.testing.assert_allclose(plus[[1,5]],-minus[[1,5]],atol=2e-12,rtol=0)
    assert cases[6]['final'][2]>6000 and cases[6]['final'][4]>0
    clips=[]
    for speed,zeta,expected in ((399.,2.,400.),(100.5,-1.,100.)):
        y=np.array([0,0,6000,speed,0,0]);out,f,d=advance(y,[0,zeta,1],initial_frame(y),C)
        assert out[3]==expected and d['speed_clips']==1
        clips.append(dict(initial_speed=speed,control=[0,zeta,1],unclipped_speed=speed+C.dt*C.gravity*zeta,
                          final=out.tolist(),speed_clips=d['speed_clips']))
    return dict(action_map=mapped.tolist(),cases=cases,speed_boundaries=clips)


def complex_acceptance():
    cases=[]
    for sign in (-1,1):
        for mu in ((0.,.7) if sign==1 else (0.,.7,np.pi)):
            for zeta,n in ((-1.,0.),(2.,8.)):
                y=np.array([0,0,6000,200,sign*np.radians(89.9),.4]);f=initial_frame(y);start=y.copy()
                jumps=[];distances=[];orth=0.;entries=0;substeps=0;directions=0.;states=[y.copy()]
                for _ in range(16):
                    old=y.copy();y,f,d=advance(y,[mu,zeta,n],f,C)
                    jumps.append(float(np.linalg.norm(velocity_vector(y)-velocity_vector(old))))
                    distances.append(float(np.linalg.norm(y[:3]-old[:3])))
                    directions=max(directions,float(np.max(abs(velocity_vector(y)/y[3]-f[:,0]))))
                    orth=max(orth,frame_error(f));entries+=d['pole_entries'];substeps+=d['continuation_substeps'];states.append(y.copy())
                    assert np.isfinite(y).all() and C.v_min<=y[3]<=C.v_max and orth<5e-14
                    assert jumps[-1]<C.dt*C.gravity*(abs(zeta)+n+2)*1.1
                    assert distances[-1]<C.dt*(old[3]+C.dt*C.gravity*(abs(zeta)+1))
                assert directions<2e-13
                if n==8 and mu==0 and sign==1:assert max(s[4] for s in states)>np.pi/2
                if n==8 and mu==np.pi and sign==-1:assert min(s[4] for s in states)<-np.pi/2
                cases.append(dict(initial=start.tolist(),control=[mu,zeta,n],final=y.tolist(),max_velocity_jump_m_s=max(jumps),
                    max_step_distance_m=max(distances),frame_error=orth,direction_error=directions,pole_entries=entries,
                    continuation_substeps=substeps,states=np.array(states).tolist()))
    faults=[]
    y=np.array([0,0,6000,200,0,0.]);f=initial_frame(y)
    for name,u in (('bank_out_of_range',[np.pi+.01,0,1]),('negative_normal',[0,0,-.1]),('nan_control',[0,np.nan,1])):
        try:advance(y,u,f,C)
        except SimulationFault as exc:faults.append(dict(case=name,reason=str(exc)))
        else:raise AssertionError('illegal control accepted')
    e=AirCombatEnv(C);e.reset(71);e.state.friendly_frames[0,0,0]=np.nan
    state=copy.deepcopy(e.state);rng=copy.deepcopy(e.rng.bit_generator.state)
    try:e.step(TRIM)
    except SimulationFault:pass
    else:raise AssertionError('corrupted frame accepted')
    assert e.state.step_count==0 and not e.state.events and e.rng.bit_generator.state==rng
    np.testing.assert_array_equal(e.state.friendly_states,state.friendly_states)
    assert e.done and e.fault and e.state.friendly_alive.all() and e.state.adversarial_alive.all()
    return dict(polar_cases=cases,explicit_faults=faults,corrupt_frame_transaction_aborted=True)


def blue_acceptance():
    np.testing.assert_array_equal(BLUE_CONTROLS,EXPECTED)
    e=AirCombatEnv(C);e.reset(41)
    blue=np.array([0,0,6000,200,0,0.]);red=np.array([[1000,0,6000,200,0,0],[1500,0,6000,200,0,0.]])
    frame=initial_frame(blue);frozen=pickle.dumps(e.snapshot(),protocol=5)
    scores=candidate_scores(blue,red,C,frame);records=[];error=0.
    for k,u in enumerate(EXPECTED):
        predicted,pf,_=advance(blue,u,frame,C)
        pair=[independent_pair_score(predicted,r) for r in red];delta=abs(scores[k]-sum(pair));error=max(error,delta)
        assert delta<2e-14
        records.append(dict(paper_no=k+1,runtime_control=u.tolist(),virtual_state=predicted.tolist(),
                            independent_pair_scores=pair,total=float(scores[k])))
    assert pickle.dumps(e.snapshot(),protocol=5)==frozen
    order=np.argsort(-scores,kind='stable')
    assert scores[0]==2*(5+1/3)
    left,_,_=advance(blue,EXPECTED[3],frame,C);right,_,_=advance(blue,EXPECTED[6],frame,C)
    assert left[5]<0<right[5]
    return dict(table_I_exact=True,default_aggregation=C.opponent_aggregation,best_probability=C.opponent_best_probability,
        independent_max_score_error=error,ranking_paper_no=(order+1).tolist(),candidates=records,
        table_left_step=left.tolist(),table_right_step=right.tolist(),
        label_note='Table Left (mu<0) decreases psi; positive-y/CCW turn has mu>0. Numerical table and Eq.(2) retained.')


def hexagon():
    theta=np.arange(6)*np.pi/3;p=np.column_stack((1000*np.cos(theta),1000*np.sin(theta),np.full(6,6000)))
    v=np.roll(p,-1,axis=0)-p
    planes=np.column_stack((p,np.full(6,200.),np.zeros(6),np.arctan2(v[:,1],v[:,0])))
    return dict(friendly_states=planes[::2],adversarial_states=planes[1::2])


def scene(name,seed,opts,limit=250):
    e=AirCombatEnv(C);obs,info=e.reset(seed,opts);rows=[]
    for _ in range(limit):
        alive_r=e.state.friendly_alive.copy();alive_b=e.state.adversarial_alive.copy()
        counts=e.state.friendly_dwell.copy(),e.state.adversarial_dwell.copy()
        _,reward,term,trunc,info=e.step(TRIM)
        g,ra,ba,_=pair_advantages(e.state.friendly_states,e.state.adversarial_states,C)
        live=alive_r[:,None]&alive_b[None,:];qual=(ra>0)&live,(ba>0)&live.T
        thresholds=[np.where(q,old+1,0) for q,old in zip(qual,counts)]
        for event in info['events']:
            a,b=event['attacker_id'],event['target_id'];side=0 if event['attacker_side']=='Red' else 1
            assert thresholds[side][a,b]>=2
        np.testing.assert_array_equal(reward,sum(info['reward_components'].values()))
        assert not info['numerical_failure']
        rows.append(dict(step=e.state.step_count,red=info['friendly_states'].tolist(),blue=info['adversarial_states'].tolist(),
            red_alive=info['friendly_alive'].tolist(),blue_alive=info['adversarial_alive'].tolist(),
            red_control=info['red_controls'].tolist(),blue_control=info['blue_controls'].tolist(),
            blue_paper_actions=(info['blue_action_indices']+1).tolist(),assignments=info['transition_assignments'].tolist(),
            red_alpha_i_deg=np.degrees(g.alpha_i).tolist(),red_alpha_j_deg=np.degrees(g.alpha_j).tolist(),
            distance=g.distance.tolist(),red_eligible=qual[0].tolist(),blue_eligible=qual[1].tolist(),
            red_dwell_at_resolution=thresholds[0].tolist(),blue_dwell_at_resolution=thresholds[1].tolist(),
            events=info['events'],outcome=info['outcome'],terminated=term,truncated=trunc))
        if term or trunc:break
    return dict(name=name,seed=seed,steps=len(rows),outcome=info['outcome'],win=info['win'],
                blue_kills=info['blue_shot_down'],red_survivors=info['red_survivors'],rows=rows)


def environment_acceptance():
    tail=options('A1');far=options('A1');far['adversarial_states'][:,0]=1e6
    multi=options('A1');multi['friendly_alive']=[1,1,0];multi['friendly_states'][1]=[0,50,6000,200,0,0]
    requests=[('tail',100,tail,5),('head_on',77,{'scenario':'head_on_3v3'},250),
        ('multiple_attackers',100,multi,5),('mutual_hexagon',100,hexagon(),5),('timeout',77,far,250)]
    results=[]
    for name,seed,opts,limit in requests:
        a=scene(name,seed,opts,limit);b=scene(name,seed,opts,limit)
        assert json.dumps(a,sort_keys=True)==json.dumps(b,sort_keys=True)
        a['repeat_exact']=True;results.append(a)
    assert results[0]['steps']==2 and results[0]['blue_kills']==1 and results[0]['win']
    event=results[2]['rows'][-1]['events'];assert len(event)==1 and event[0]['eligible_attackers']==[0,1]
    assert results[3]['outcome']=='mutual_elimination' and results[3]['win'] and len(results[3]['rows'][-1]['events'])==6
    assert results[4]['outcome']=='timeout' and results[4]['steps']==250 and not results[4]['win']
    # Non-assigned threat, actual mutual kills on different pairs, then dead freeze.
    opts=options('A2');opts['friendly_alive']=[1,1,0]
    e=AirCombatEnv(C);e.reset(100,opts);e.step(TRIM);second=e.step(TRIM)
    assert {x['attacker_side'] for x in second[4]['events']}=={'Red','Blue'} and not second[2]
    frozen=e.state.friendly_states[0].copy(),e.state.adversarial_states[0].copy()
    actions=TRIM.copy();actions[0]=np.nan
    third=e.step(actions)
    np.testing.assert_array_equal(e.state.friendly_states[0],frozen[0]);np.testing.assert_array_equal(e.state.adversarial_states[0],frozen[1])
    assert not third[0][0].any() and not third[4]['red_controls'][0].any() and third[4]['blue_action_indices'][0]==-1
    assert not e.state.friendly_dwell[0].any() and not e.state.adversarial_dwell[0].any()
    return dict(scenes=results,actual_mutual_pair_events=second[4]['events'],dead_slots_frozen=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    require_cuda('cuda')  # Runtime policy; no network, optimizer or GPU workload.
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    result=dict(environment=C.to_dict(),learned_models_used=False,training_steps=0,
                blue=blue_acceptance(),flight=flight_acceptance(),complex=complex_acceptance(),environment_checks=environment_acceptance())
    (out/'acceptance.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(blue_score_error=result['blue']['independent_max_score_error'],
        flights=[{k:v for k,v in s.items() if k not in ('states',)} for s in result['flight']['cases']],
        scenes=[{k:v for k,v in s.items() if k!='rows'} for s in result['environment_checks']['scenes']]),indent=2))


if __name__=='__main__':main()
