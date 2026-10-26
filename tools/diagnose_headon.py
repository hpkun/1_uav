"""Ten specified Table VII episodes; read-only, no training or plot framework."""
import argparse
import hashlib
from pathlib import Path
import json
import numpy as np
from algorithm.experiments import load_model,write_json
from env.aircombat import AirCombatEnv
from env.aircombat.geometry import relative_geometry,pair_advantages
from env.dynamics import velocity_vector,wrap_angle


def stats(values):
    a=np.asarray(values,float)
    return dict(n=len(a),min=a.min(0).tolist(),mean=a.mean(0).tolist(),max=a.max(0).tolist()) if len(a) else dict(n=0)


def heading(v):return np.arctan2(v[1],v[0])


def run(checkpoint,output):
    path=Path(checkpoint);digest=hashlib.sha256(path.read_bytes()).hexdigest()
    trainer,c,payload=load_model(path)
    if (payload['training_episode_count']!=2500 or c.scale!='3v3' or c.posture_reward_mode!='paper'
            or c.observation_encoding!='unsigned' or trainer.config.observation_normalization!='fixed_scale_v1'):
        raise ValueError('diagnosis requires the completed 2500-episode unsigned normalized paper 3v3 model')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    episodes=[]
    with (output/'steps.jsonl').open('w',encoding='utf-8') as log:
        for seed in range(10010000,10010010):
            env=AirCombatEnv(c);obs,info=env.reset(seed,{'scenario':'head_on_3v3'})
            first={side:dict(range=None,attack=None,kill=None) for side in ('Red','Blue')}
            counts={side:dict(range_pairs=0,own_cone_pairs=0,attack_pairs=0,two_step_pairs=0,
                interrupted_locks=0,pole_entries=0,continuation_substeps=0) for side in first}
            turn=dict(toward=0,away=0,neutral=0,skipped=0)
            by_side={k:[] for k in ('left','right','aligned')}
            controls=[[] for _ in range(3)];angles=[[] for _ in range(3)]
            switches=[];events=[];unassigned_blue_threats=0;assigned_red_attack_pairs=0
            nose_changes=[];large_gamma={side:0 for side in first}
            while True:
                s=env.state;pre_r=s.friendly_states.copy();pre_b=s.adversarial_states.copy()
                live_r=s.friendly_alive.copy();live_b=s.adversarial_alive.copy();assigned=s.assignments.copy()
                dwell=(s.friendly_dwell.copy(),s.adversarial_dwell.copy())
                actions=trainer.sample_actions(obs,info['active_mask'],deterministic=True)[0].cpu().numpy()
                obs,reward,term,trunc,info=env.step(actions)
                r,b=s.friendly_states,s.adversarial_states;step=s.step_count
                g=relative_geometry(r,b,c.geometry_epsilon);gb=relative_geometry(b,r,c.geometry_epsilon)
                _,ra,ba,_=pair_advantages(r,b,c)
                pair_live=live_r[:,None]&live_b[None,:]
                reconstructed=[];qual=[];expected_events=[]
                for side,geom,alive,prev,adv in (
                    ('Red',g,pair_live,dwell[0],ra),('Blue',gb,pair_live.T,dwell[1],ba)):
                    distance=(geom.distance>c.geometry_epsilon)&(geom.distance<=c.tau_max)&alive
                    cone=distance&(geom.alpha_i<=c.alpha_max)
                    q=cone&(geom.alpha_j<np.pi/2)
                    np.testing.assert_array_equal(q,(adv>0)&alive)
                    streak=np.where(q,prev+1,0);ready=streak>=c.attack_dwell_steps
                    for key,mask in (('range_pairs',distance),('own_cone_pairs',cone),('attack_pairs',q),('two_step_pairs',ready)):
                        counts[side][key]+=int(mask.sum())
                    counts[side]['interrupted_locks']+=int(((prev>0)&alive&~q).sum())
                    for key,mask in (('range',distance),('attack',q)):
                        if first[side][key] is None and mask.any():first[side][key]=step
                    # Each victim has one owner, even if several pairs reach two.
                    for victim in np.flatnonzero(ready.any(0)):
                        expected_events.append((side,int(np.flatnonzero(ready[:,victim])[0]),int(victim)))
                    reconstructed.append(streak);qual.append(q)
                assert expected_events==[(e['attacker_side'],e['attacker_id'],e['target_id']) for e in info['events']]
                post_live=s.friendly_alive[:,None]&s.adversarial_alive[None,:]
                np.testing.assert_array_equal(s.friendly_dwell,reconstructed[0]*post_live)
                np.testing.assert_array_equal(s.adversarial_dwell,reconstructed[1]*post_live.T)
                for e in info['events']:
                    side=e['attacker_side'];i,j=e['attacker_id'],e['target_id']
                    if first[side]['kill'] is None:first[side]['kill']=step
                    own,target,geom=(r,b,g) if side=='Red' else (b,r,gb)
                    events.append(dict(**e,time_s=step*c.dt,attacker_position=own[i,:3].tolist(),
                        target_position=target[j,:3].tolist(),distance=float(geom.distance[i,j]),
                        alpha_i_deg=float(np.degrees(geom.alpha_i[i,j])),alpha_j_deg=float(np.degrees(geom.alpha_j[i,j]))))
                for item in info['numerical_step']:
                    for key in ('pole_entries','continuation_substeps'):counts[item['side']][key]+=int(item[key])
                for side,states,alive in (('Red',r,live_r),('Blue',b,live_b)):
                    large_gamma[side]+=int((abs(states[alive,4])>np.pi/3).sum())
                for i in np.flatnonzero(live_r):
                    controls[i].append(info['red_controls'][i].copy())
                    angles[i].append(np.degrees(r[i,4:6]).tolist())
                    j=assigned[i]
                    if j<0:continue
                    assigned_red_attack_pairs+=int(qual[0][i,j])
                    unassigned_blue_threats+=int(qual[1][:,i].sum()-qual[1][j,i])
                    los=pre_b[j,:3]-pre_r[i,:3]
                    v0,v1=velocity_vector(pre_r[i]),velocity_vector(r[i])
                    a0=relative_geometry(pre_r[i:i+1],pre_b[j:j+1],c.geometry_epsilon).alpha_i[0,0]
                    # Fixed pre-action LOS isolates own rotation from target motion.
                    a1=np.arctan2(np.linalg.norm(np.cross(v1,los)),np.dot(v1,los))
                    nose_changes.append(float(np.degrees(a1-a0)))
                    if (np.linalg.norm(los[:2])<=c.geometry_epsilon or
                            min(np.linalg.norm(v0[:2])/pre_r[i,3],np.linalg.norm(v1[:2])/r[i,3])<c.pole_cap_ratio):
                        turn['skipped']+=1;continue
                    error=np.arctan2(v0[0]*los[1]-v0[1]*los[0],np.dot(v0[:2],los[:2]))
                    dyaw=float(wrap_angle(heading(v1)-heading(v0)))
                    label='aligned' if abs(error)<np.radians(1) else 'left' if error>0 else 'right'
                    by_side[label].append([float(np.degrees(info['red_controls'][i,0])),float(np.degrees(dyaw)),float(np.degrees(error))])
                    if label=='aligned' or abs(dyaw)<1e-8:turn['neutral']+=1
                    else:turn['toward' if error*dyaw>0 else 'away']+=1
                for i in np.flatnonzero(assigned!=s.assignments):
                    switches.append(dict(step=step,red=int(i),before=int(assigned[i]),after=int(s.assignments[i]),
                        reason='death/unassigned' if not live_r[i] or not s.friendly_alive[i] or assigned[i]<0 or s.assignments[i]<0 or not s.adversarial_alive[assigned[i]] else 'live_target_switch'))
                row=dict(seed=seed,step=step,pre_assignments=assigned.tolist(),post_assignments=s.assignments.tolist(),
                    pre_red=pre_r.tolist(),pre_blue=pre_b.tolist(),post_red=r.tolist(),post_blue=b.tolist(),
                    pre_red_alive=live_r.tolist(),pre_blue_alive=live_b.tolist(),red_actions=actions.tolist(),
                    red_controls=info['red_controls'].tolist(),blue_controls=info['blue_controls'].tolist(),
                    red_distance=g.distance.tolist(),red_alpha_i_deg=np.degrees(g.alpha_i).tolist(),
                    red_alpha_j_deg=np.degrees(g.alpha_j).tolist(),red_eligible=qual[0].tolist(),blue_eligible=qual[1].tolist(),
                    red_dwell_at_resolution=reconstructed[0].tolist(),blue_dwell_at_resolution=reconstructed[1].tolist(),
                    numerical_step=info['numerical_step'],events=info['events'])
                log.write(json.dumps(row,default=lambda x:x.tolist(),allow_nan=False)+'\n')
                if term or trunc:break
            episodes.append(dict(seed=seed,steps=step,outcome=info['outcome'],red_survivors=info['red_survivors'],
                blue_survivors=int(s.adversarial_alive.sum()),first_steps=first,counts=counts,events=events,
                assignment_changes=switches,live_target_switches=sum(e['reason']=='live_target_switch' for e in switches),
                assigned_red_attack_pairs=assigned_red_attack_pairs,unassigned_blue_threat_pairs=unassigned_blue_threats,
                turn_counts=turn,bearing_groups={k:stats(v) for k,v in by_side.items()},
                own_rotation_alpha_change_deg=stats(nose_changes),large_gamma_over_60_uav_steps=large_gamma,
                red_controls_mu_rad_zeta_normal=[stats(v) for v in controls],red_gamma_psi_deg=[stats(v) for v in angles],
                numerical_diagnostics=info['numerical_diagnostics'],numerical_failure=info['numerical_failure']))
    # Controlled left/right ambiguity probe, separate environments, no rollout mutation.
    mirror=[]
    for y in (600.,-600.):
        r=np.tile([0,0,6000,200,0,0],(3,1)).astype(float)
        b=np.tile([1500,y,6000,200,0,0],(3,1)).astype(float)
        probe=AirCombatEnv(c);o,inf=probe.reset(123,dict(friendly_states=r,adversarial_states=b,
            friendly_alive=[1,0,0],adversarial_alive=[1,0,0]))
        action=trainer.sample_actions(o,inf['active_mask'],deterministic=True)[0].cpu().numpy()
        mirror.append(dict(y=y,observation=o[0].tolist(),physical_control=(action[0]*[2*np.pi,3,8]+[-np.pi,-1,0]).tolist()))
    assert mirror[0]['observation']==mirror[1]['observation'] and mirror[0]['physical_control']==mirror[1]['physical_control']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    result=dict(checkpoint=str(path.resolve()),sha256=digest,model_episodes=payload['training_episode_count'],
        scenario='head_on_3v3',table='VII',policy='deterministic Beta mean',dt=c.dt,episodes=episodes,mirror_probe=mirror,
        mean_steps=float(np.mean([e['steps'] for e in episodes])),wins=sum(e['blue_survivors']==0 for e in episodes),
        counting='post-motion/pre-death live pairs; reconstructed threshold counts verified against actual events and surviving dwell; same-step events simultaneous',
        turn_definition='sign(pre-action horizontal LOS bearing)*actual horizontal yaw increment; skip polar cap; not a causal optimality test')
    write_json(output/'diagnosis.json',result)
    for e in episodes:
        print(json.dumps({k:e[k] for k in ('seed','steps','red_survivors','blue_survivors','first_steps','counts','turn_counts','live_target_switches')},ensure_ascii=False))
    print(json.dumps(dict(mean_steps=result['mean_steps'],wins=result['wins'],mirror_probe=mirror),ensure_ascii=False))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('checkpoint');p.add_argument('--output',required=True)
    a=p.parse_args();run(a.checkpoint,a.output)
