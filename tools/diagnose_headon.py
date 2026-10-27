"""Ten specified Table VII episodes; read-only, no training or plot framework."""
import argparse
import hashlib
from pathlib import Path
import json
import numpy as np
from algorithm.experiments import load_model,write_json,configurations
from env.aircombat import AirCombatEnv
from env.aircombat.geometry import relative_geometry,pair_advantages
from env.aircombat.framed_dynamics import initial_frame
from env.dynamics import velocity_vector,wrap_angle


def stats(values):
    a=np.asarray(values,float)
    return dict(n=len(a),min=a.min(0).tolist(),mean=a.mean(0).tolist(),max=a.max(0).tolist()) if len(a) else dict(n=0)


def heading(v):return np.arctan2(v[1],v[0])


def mirror_check(probes, encoding):
    """Check representation, measure actions; never require signed actions to differ."""
    a,b=np.array(probes[0]['observation']),np.array(probes[1]['observation'])
    if encoding=='signed_alpha_v1':
        assert a[4]>0 and b[4]<0 and a[4]==-b[4]
        a[4]=abs(a[4]);b[4]=abs(b[4])
    else:
        assert encoding=='unsigned'
        np.testing.assert_array_equal(probes[0]['physical_control'],probes[1]['physical_control'])
    np.testing.assert_array_equal(a,b)
    return (np.array(probes[0]['physical_control'])-probes[1]['physical_control']).tolist()


def transition_samples(row, config):
    """Detached telemetry from actual endpoints and returned reward, including death endpoints."""
    r0,b0,r1,b1=[np.asarray(row[k],float) for k in ('pre_red','pre_blue','post_red','post_blue')]
    g=relative_geometry(r1,b1,config.geometry_epsilon)
    flight=[];posture=[] if 'reward_components' in row else None
    for i in np.flatnonzero(row['pre_red_alive']):
        mu,zeta,n=row['red_controls'][i]
        frame=np.array(row['pre_red_frames'][i]) if 'pre_red_frames' in row else initial_frame(r0[i])
        h=float(np.linalg.norm(frame[:2,0]));projection=np.cos(mu)*frame[2,1]+np.sin(mu)*frame[2,2]
        # Exact frame gravity projection: in the ordinary aligned chart this is
        # n*cos(mu)-cos(gamma), not n-1. No gamma derivative at the vertical pole.
        margin=float(n*projection/h-h) if h>1e-9 else None
        v0,v1=velocity_vector(r0[i]),velocity_vector(r1[i])
        physical_gamma0=np.arctan2(v0[2],np.linalg.norm(v0[:2]))
        physical_gamma1=np.arctan2(v1[2],np.linalg.norm(v1[:2]))
        item=dict(seed=row['seed'],step=row['step'],red_id=int(i),mu_deg=float(np.degrees(mu)),zeta=zeta,normal=n,
            gamma_deg=float(np.degrees(r1[i,4])),physical_gamma_deg=float(np.degrees(physical_gamma1)),
            gamma_dot_deg_s=float(np.degrees(r1[i,4]-r0[i,4])/config.dt),
            physical_gamma_dot_deg_s=float(np.degrees(physical_gamma1-physical_gamma0)/config.dt),
            gamma_rhs_deg_s=float(np.degrees(config.gravity/r0[i,3]*margin)) if margin is not None else None,
            normal_required_constant_gamma=float(h*h/projection) if projection>1e-9 else None,
            vertical_margin=margin,exact_frame='pre_red_frames' in row,
            altitude=float(r1[i,2]),dz=float(r1[i,2]-r0[i,2]),target=row['pre_assignments'][i],turn=None)
        j=item['target']
        if j>=0 and row['pre_blue_alive'][j]:
            los=b0[j,:3]-r0[i,:3];d0=float(np.linalg.norm(los));d1=float(np.linalg.norm(b1[j,:3]-r1[i,:3]))
            if min(np.linalg.norm(v0[:2])/r0[i,3],np.linalg.norm(v1[:2])/r1[i,3])>=config.pole_cap_ratio and np.linalg.norm(los[:2])>config.geometry_epsilon:
                error=np.arctan2(v0[0]*los[1]-v0[1]*los[0],np.dot(v0[:2],los[:2]))
                dyaw=float(wrap_angle(heading(v1)-heading(v0)))
                item['turn']='neutral' if abs(error)<np.radians(1) or abs(dyaw)<1e-8 else 'toward' if error*dyaw>0 else 'away'
            if posture is not None:
                own=float((r1[i,:3]-r0[i,:3])@(los/d0)/config.dt) if d0>0 else 0.
                blue=float((b1[j,:3]-b0[j,:3])@(los/d0)/config.dt) if d0>0 else 0.
                rel=(d0-d1)/config.dt
                # This is the REAL returned paper component. Never reconstruct P
                # with an alternative formula or feed a counterfactual to policy.
                p=float(row['reward_components']['posture'][i])
                band='close' if d1<config.tau_min else 'engagement' if d1<=config.tau_max else 'far'
                gate=float(np.clip(min(own,rel)/config.posture_gate_speed,0,1)) if d0>0 else 0.
                posture.append(dict(seed=row['seed'],step=row['step'],red_id=int(i),target=int(j),d0=d0,d1=d1,
                    c_rel=rel,c_self=own,c_blue_outward=blue,posture=p,alpha_i_deg=float(np.degrees(g.alpha_i[i,j])),
                    alpha_j_deg=float(np.degrees(g.alpha_j[i,j])),band=band,
                    direction='approaching' if rel>0 else 'receding' if rel<0 else 'maintaining',
                    counterfactual_gated_posture=min(p,0)+gate*max(p,0) if band=='far' else p))
        flight.append(item)
    return flight,posture


def longest_run(samples, predicate):
    """Consecutive physical steps for the SAME UAV/target; switches break runs."""
    active={};best=None
    for s in samples:
        key=(s['seed'],s['red_id']);prev=active.get(key)
        if not predicate(s):active.pop(key,None);continue
        if prev is None or prev['target']!=s['target'] or prev['end']!=s['step']-1:
            prev=dict(seed=s['seed'],red_id=s['red_id'],target=s['target'],start=s['step'],end=s['step']-1,
                      samples=0,posture_sum=0.,d0=s.get('d0'),d1=None,c_self_sum=0.,self_receding=0)
        prev['end']=s['step'];prev['samples']+=1;prev['posture_sum']+=s.get('posture',0.)
        prev['d1']=s.get('d1');prev['c_self_sum']+=s.get('c_self',0.);prev['self_receding']+=int(s.get('c_self',0)<0)
        active[key]=prev
        if best is None or prev['samples']>best['samples']:best=dict(prev)
    return best


def reward_group(samples, total):
    n=len(samples);p=np.array([s['posture'] for s in samples]);positive=p[p>0]
    return dict(n=n,fraction=n/total if total else None,mean=float(p.mean()) if n else None,
                sum=float(p.sum()),positive_n=len(positive),positive_fraction=len(positive)/n if n else None,
                positive_sum=float(positive.sum()))


def summarize_rows(rows, config):
    flights=[];postures=[];range_alpha=[];entry_alpha=[];previous_range={};initial_z={}
    for row in rows:
        if row['seed'] not in initial_z:initial_z[row['seed']]=np.array(row['pre_red'])[:,2]
        f,p=transition_samples(row,config)
        for s in f:s['height_change']=s['altitude']-initial_z[s['seed']][s['red_id']]
        flights.extend(f)
        if p is not None:postures.extend(p)
        live=np.array(row['pre_red_alive'])[:,None]&np.array(row['pre_blue_alive'])[None,:]
        distance=np.array(row['red_distance']);inside=live&(distance>config.geometry_epsilon)&(distance<=config.tau_max)
        alpha=np.array(row['red_alpha_i_deg']);range_alpha.extend(alpha[inside].tolist())
        old=previous_range.get(row['seed'],np.zeros_like(inside))
        entry_alpha.extend(alpha[inside&~old].tolist());previous_range[row['seed']]=inside
    def flight_summary(samples):
        n=len(samples);valid=[s for s in samples if s['vertical_margin'] is not None]
        turns={k:sum(s['turn']==k for s in samples) for k in ('toward','away','neutral')}
        moving=turns['toward']+turns['away']
        result={k:stats([s[k] for s in samples]) for k in ('mu_deg','zeta','normal','gamma_deg','physical_gamma_deg',
            'gamma_dot_deg_s','physical_gamma_dot_deg_s','altitude','height_change')}
        result.update(turn_counts=turns,turn_toward_fraction=turns['toward']/moving if moving else None,
            normal_above_equilibrium_fraction=sum(s['vertical_margin']>0 for s in valid)/len(valid) if valid else None,
            gamma_rising_fraction=sum(s['physical_gamma_dot_deg_s']>0 for s in samples)/n if n else None,
            gamma_rhs_deg_s=stats([s['gamma_rhs_deg_s'] for s in valid]),
            frame_equilibrium_source='exact saved frames' if all(s['exact_frame'] for s in samples) else 'nominal angle-chart frames; archive has no frames',
            longest_above_equilibrium=longest_run([dict(s,target=-1) for s in samples],lambda s:s['vertical_margin'] is not None and s['vertical_margin']>0))
        return result
    result=dict(flight=flight_summary(flights),first40_flight=flight_summary([s for s in flights if s['step']<=40]),
                in_range_alpha_i_deg=stats(range_alpha),range_entry_alpha_i_deg=stats(entry_alpha))
    if not postures:
        result['posture']=None  # Old archive did not save actual reward; do not manufacture it.
        return result
    far=[s for s in postures if s['band']=='far'];receding=[s for s in far if s['direction']=='receding']
    all_positive=sum(max(0,s['posture']) for s in far)
    groups={key:reward_group([s for s in far if s['direction']==key],len(far)) for key in ('approaching','receding','maintaining')}
    split={key:reward_group([s for s in receding if predicate(s)],len(receding)) for key,predicate in (
        ('self_receding',lambda s:s['c_self']<0),('self_stationary',lambda s:s['c_self']==0),
        ('self_closing_blue_faster',lambda s:s['c_self']>0 and s['c_blue_outward']>s['c_self']),
        ('self_closing_transverse_growth',lambda s:s['c_self']>0 and s['c_blue_outward']<=s['c_self']))}
    result['posture']=dict(valid_samples=len(postures),bands={k:reward_group([s for s in postures if s['band']==k],len(postures)) for k in ('close','engagement','far')},
        far=reward_group(far,len(postures)),far_directions=groups,receding_causes=split,
        receding_positive_fraction_of_far=groups['receding']['positive_n']/len(far) if far else None,
        receding_positive_share_of_far_positive=groups['receding']['positive_sum']/all_positive if all_positive else None,
        counterfactual_far_gated_sum=sum(s['counterfactual_gated_posture'] for s in far),
        longest_receding_positive=longest_run(postures,lambda s:s['band']=='far' and s['direction']=='receding' and s['posture']>0),
        longest_self_receding_positive=longest_run(postures,lambda s:s['band']=='far' and s['direction']=='receding' and s['c_self']<0 and s['posture']>0))
    return result


def run(checkpoint,output,compare_steps=None):
    path=Path(checkpoint);digest=hashlib.sha256(path.read_bytes()).hexdigest()
    trainer,c,payload=load_model(path)
    if (payload['training_episode_count']!=2500 or c.scale!='3v3' or c.posture_reward_mode!='paper'
            or c.observation_encoding not in ('unsigned','signed_alpha_v1') or trainer.config.observation_normalization!='fixed_scale_v1'):
        raise ValueError('diagnosis requires the completed 2500-episode normalized paper 3v3 model')
    experiment='headon_xu_norm_signed_3v3' if c.observation_encoding=='signed_alpha_v1' else 'headon_xu_norm_3v3'
    expected_env,expected_train=configurations(experiment,trainer.config.training_seed_index,episodes=2500)
    if c!=expected_env or trainer.config!=expected_train:raise ValueError('formal model configuration mismatch')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    episodes=[]
    with (output/'steps.jsonl').open('w',encoding='utf-8') as log:
        for seed in range(10010000,10010010):
            env=AirCombatEnv(c);obs,info=env.reset(seed,{'scenario':'head_on_3v3'})
            first={side:dict(range=None,cone=None,attack=None,kill=None) for side in ('Red','Blue')}
            for side,own,target in (('Red',env.state.friendly_states,env.state.adversarial_states),
                                    ('Blue',env.state.adversarial_states,env.state.friendly_states)):
                first[side]['angle']=0 if (relative_geometry(own,target,c.geometry_epsilon).alpha_i<=c.alpha_max).any() else None
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
                pre_frames=s.friendly_frames.copy()
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
                    for key,mask in (('range',distance),('angle',alive&(geom.alpha_i<=c.alpha_max)),('cone',cone),('attack',q)):
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
                    pre_red_frames=pre_frames.tolist(),reward_components={k:v.tolist() for k,v in info['reward_components'].items()},
                    post_red_alive=s.friendly_alive.tolist(),post_blue_alive=s.adversarial_alive.tolist(),
                    red_controls=info['red_controls'].tolist(),blue_controls=info['blue_controls'].tolist(),
                    red_distance=g.distance.tolist(),red_alpha_i_deg=np.degrees(g.alpha_i).tolist(),
                    red_alpha_j_deg=np.degrees(g.alpha_j).tolist(),red_eligible=qual[0].tolist(),blue_eligible=qual[1].tolist(),
                    red_dwell_at_resolution=reconstructed[0].tolist(),blue_dwell_at_resolution=reconstructed[1].tolist(),
                    numerical_step=info['numerical_step'],events=info['events'])
                _,samples=transition_samples(row,c)
                row['posture_samples']=samples
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
    action_difference=mirror_check(mirror,c.observation_encoding)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    result=dict(checkpoint=str(path.resolve()),sha256=digest,model_episodes=payload['training_episode_count'],
        scenario='head_on_3v3',table='VII',policy='deterministic Beta mean',dt=c.dt,episodes=episodes,mirror_probe=mirror,
        observation_encoding=c.observation_encoding,mirror_control_difference=action_difference,
        mean_steps=float(np.mean([e['steps'] for e in episodes])),wins=sum(e['blue_survivors']==0 for e in episodes),
        counting='post-motion/pre-death live pairs; reconstructed threshold counts verified against actual events and surviving dwell; same-step events simultaneous',
        turn_definition='sign(pre-action horizontal LOS bearing)*actual horizontal yaw increment; skip polar cap; not a causal optimality test')
    rows=[json.loads(line) for line in (output/'steps.jsonl').read_text().splitlines()]
    result['analysis']=summarize_rows(rows,c)
    if compare_steps:
        old=[json.loads(line) for line in Path(compare_steps).read_text().splitlines()]
        result['old_archive_comparison']=dict(source=str(Path(compare_steps).resolve()),analysis=summarize_rows(old,c))
    write_json(output/'diagnosis.json',result)
    for e in episodes:
        print(json.dumps({k:e[k] for k in ('seed','steps','red_survivors','blue_survivors','first_steps','counts','turn_counts','live_target_switches')},ensure_ascii=False))
    print(json.dumps(dict(mean_steps=result['mean_steps'],wins=result['wins'],mirror_probe=mirror),ensure_ascii=False))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('checkpoint');p.add_argument('--output',required=True)
    p.add_argument('--compare-steps',help='read an existing old-model steps.jsonl; never rerun its episodes')
    a=p.parse_args();run(a.checkpoint,a.output,a.compare_steps)
