"""Small matched-seed reachability and untrained Actor diagnostics. No learning."""
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from env.config import EnvConfig,SCALES
from env.factory import make_environment
from algorithm.common.runtime import require_cuda
from algorithm.common.serialization import write_json,serializable
from algorithm.mappo.networks import Actor
from .navigation_baseline import NavigationController,NavigationConfig,reachable_scenario


def nearest_distance(info):
    active=np.asarray(info['active_mask'],dtype=bool)
    remaining=~np.asarray(info['completed'],dtype=bool)
    if not remaining.any(): return 0.0
    if not active.any(): return None
    d=np.linalg.norm(np.asarray(info['states'])[active,None,:3]-np.asarray(info['waypoints'])[None,remaining],axis=-1)
    return float(d.min(axis=1).mean())


def run_episode(config,seed,policy,*,options=None,actor=None,deterministic=False,trace_path=None,nav_config=None):
    env=make_environment(config); obs,info=env.reset(seed,options=options)
    initial=nearest_distance(info); rng=np.random.default_rng(seed+1000000)
    nav=NavigationController(nav_config)
    frames=[]; per_uav=np.zeros(config.num_uavs,dtype=int); assigned_before=[set() for _ in per_uav]
    later_assignments=[]; reassignment_completions=[]
    progress=travel=abs_gamma=speed=0.; active_steps=gamma_clips=speed_clips=switches=0
    valid=np.flatnonzero(info['active_mask'])
    initial_pairs=info['pairwise_distances'][np.ix_(valid,valid)].copy()
    np.fill_diagonal(initial_pairs,np.inf)
    minimum=float(initial_pairs.min()) if len(valid)>1 else np.inf
    action_rows=[]; physical_rows=[]
    from .recording import TrajectoryRecorder
    recorder=TrajectoryRecorder(config,seed,info) if trace_path else None
    if recorder: recorder.data['provenance']=dict(policy=policy,deterministic=deterministic,scenario='manual_reachable' if options else 'standard')
    for i,j in enumerate(info['assignments']):
        if j>=0: assigned_before[i].add(int(j))
    while True:
        diag={}
        if policy=='trim':
            gamma=info['states'][:,4]
            actions=np.column_stack((np.full(config.num_uavs,.5),(np.sin(gamma)+1)/3,np.cos(gamma)/8))
        elif policy=='random': actions=rng.uniform(size=(config.num_uavs,3))
        elif policy=='navigation': actions,diag=nav.actions(info,config)
        elif policy=='actor':
            with torch.no_grad(): actions=actor.act(torch.as_tensor(np.where(info['active_mask'][:,None],obs,0),dtype=torch.float32,device=next(actor.parameters()).device),deterministic)[0].cpu().numpy()
        else: raise ValueError('unknown audit policy')
        current=info['active_mask'].copy()
        obs,reward,term,trunc,info=env.step(actions)
        ids=np.flatnonzero(current); active_steps+=len(ids)
        action_rows.extend(actions[ids]); physical_rows.extend(info['physical_actions'][ids])
        progress+=float(info['reward_components']['progress'].sum())
        travel+=float(np.linalg.norm(info['states'][ids,:3]-info['previous_states'][ids,:3],axis=1).sum())
        abs_gamma+=float(abs(info['states'][ids,4]).sum()); speed+=float(info['states'][ids,3].sum())
        gamma_clips+=info['numerical_delta']['gamma_clips']; speed_clips+=info['numerical_delta']['speed_clips']
        active=np.flatnonzero(info['next_active_mask'])
        if len(active)>1:
            d=info['pairwise_distances'][np.ix_(active,active)].copy(); np.fill_diagonal(d,np.inf); minimum=min(minimum,float(d.min()))
        changed=(info['assignments']!=info['transition_assignments']) & info['next_active_mask']
        switches+=int(changed.sum())
        for i in np.flatnonzero(changed):
            j=int(info['assignments'][i])
            if j>=0 and j not in assigned_before[i]:
                if assigned_before[i]: later_assignments.append(dict(step=info['episode_length'],uav_id=int(i),waypoint_id=j,order=len(assigned_before[i])+1))
                assigned_before[i].add(j)
        for event in info['events']:
            i=event['uav_id']; per_uav[i]+=1
            if any(x['uav_id']==i and x['waypoint_id']==event['waypoint_id'] for x in later_assignments): reassignment_completions.append(event)
        frames.append(serializable(dict(step=info['episode_length'],normalized_actions=actions,physical_actions=info['physical_actions'],
            active_uav_steps=len(ids),numerical_delta=info['numerical_delta'],numerical_events=info['numerical_events'],navigation=diag)))
        if recorder: recorder.append(info)
        if term or trunc: break
    if recorder: recorder.save(trace_path)
    action_rows=np.asarray(action_rows); physical_rows=np.asarray(physical_rows)
    return dict(seed=int(seed),policy=policy,success=info['success'],completion_rate=info['completion_rate'],completed_waypoints=info['completed_waypoints'],
        episode_length=info['episode_length'],initial_nearest_waypoint_distance=initial,final_nearest_waypoint_distance=nearest_distance(info),
        episode_progress=progress,total_travel_distance=travel,active_uav_steps=active_steps,
        gamma_clip_rate=gamma_clips/max(1,active_steps),speed_clip_rate=speed_clips/max(1,active_steps),gamma_clips=gamma_clips,speed_clips=speed_clips,
        average_abs_gamma=abs_gamma/max(1,active_steps),average_speed=speed/max(1,active_steps),minimum_inter_uav_distance=minimum if np.isfinite(minimum) else None,
        waypoint_assignment_switches=switches,per_uav_completed_waypoints=per_uav.tolist(),later_assignments=later_assignments,
        reassignment_completions=reassignment_completions,success_after_reassignment=bool(info['success'] and later_assignments),
        normalized_action_mean=action_rows.mean(0).tolist(),normalized_action_std=action_rows.std(0).tolist(),
        physical_action_mean=physical_rows.mean(0).tolist(),physical_action_std=physical_rows.std(0).tolist(),
        final_states=info['states'].tolist(),final_assignments=info['assignments'].tolist(),
        remaining_waypoints=info['waypoints'][~info['completed']].tolist(),
        navigation_normal_saturations=sum(int(np.sum(f['navigation'].get('normal_load_saturation',[]))) for f in frames),
        control_steps=frames)


def aggregate(rows):
    keys=('completion_rate','completed_waypoints','episode_length','initial_nearest_waypoint_distance','final_nearest_waypoint_distance','episode_progress',
          'total_travel_distance','average_abs_gamma','average_speed','minimum_inter_uav_distance','waypoint_assignment_switches')
    result={k:float(np.mean([r[k] for r in rows if r[k] is not None])) if any(r[k] is not None for r in rows) else None for k in keys}
    result.update(success_rate=float(np.mean([r['success'] for r in rows])),episodes=len(rows))
    den=sum(r['active_uav_steps'] for r in rows)
    result.update(active_uav_steps=den,gamma_clip_rate=sum(r['gamma_clips'] for r in rows)/max(1,den),speed_clip_rate=sum(r['speed_clips'] for r in rows)/max(1,den))
    return result


def audit(output,seeds=(101,102,103,104,105),nav_config=None):
    require_cuda(); out=Path(output)
    if out.exists() and any(out.iterdir()): raise FileExistsError('audit output must be empty')
    out.mkdir(parents=True,exist_ok=True); rows=[]
    nc=nav_config or NavigationConfig()
    for scale in SCALES:
        for random in (False,True):
            scene='random' if random else 'fixed'
            # A fixed scenario is seed-invariant; run it once, random scenarios use five matched seeds.
            for seed in seeds if random else seeds[:1]:
                for policy in ('trim','random','navigation'):
                    r=run_episode(EnvConfig(scale=scale,random_scenario=random),seed,policy,nav_config=nc,
                        trace_path=out/f'{scale}_{scene}_{policy}_{seed}.json' if seed==seeds[0] and policy=='navigation' else None)
                    r.update(scale=scale,scenario=scene); rows.append(r)
        print(f'navigation audit complete: {scale}',flush=True)
    manual=run_episode(EnvConfig(scale='3U5P',random_scenario=False),1,'navigation',options=reachable_scenario(),nav_config=nc,trace_path=out/'manual_reachable.json')
    if not manual['success']: raise AssertionError('manual reachable navigation scenario failed')
    comparison=[]
    # Exactly the same environment seeds and matched Torch seeds across modes.
    for scale in SCALES:
        for mode in ('legacy','trim_centered'):
            torch.manual_seed(17); actor=Actor(mode).cuda()
            for seed in seeds[:2]:
                for deterministic in (False,True):
                    torch.manual_seed(seed)
                    r=run_episode(EnvConfig(scale=scale),seed,'actor',actor=actor,deterministic=deterministic)
                    r.update(scale=scale,initialization_mode=mode,deterministic=deterministic); comparison.append(r)
    summaries=[]
    for scale in SCALES:
        for scene in ('fixed','random'):
            for policy in ('trim','random','navigation'):
                values=[r for r in rows if r['scale']==scale and r['scenario']==scene and r['policy']==policy]
                summaries.append(dict(scale=scale,scenario=scene,policy=policy,**aggregate(values)))
    write_json(out/'navigation_results.json',dict(seeds=seeds,navigation_config=asdict(nc),summaries=summaries,episodes=rows,manual_reachable=manual))
    write_json(out/'initialization_comparison.json',comparison)
    from .render_replay import render_replay
    render_replay(out/'manual_reachable.json',out/'manual_reachable.html')
    return summaries,comparison


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[101,102,103,104,105]); p.add_argument('--navigation-config')
    a=p.parse_args()
    if not 1<=len(a.seeds)<=5: raise ValueError('audit accepts one to five seeds')
    if a.navigation_config:
        import yaml
        nc=NavigationConfig(**yaml.safe_load(Path(a.navigation_config).read_text(encoding='utf-8')))
    else: nc=NavigationConfig()
    audit(a.output,tuple(a.seeds),nc)


if __name__=='__main__': main()
