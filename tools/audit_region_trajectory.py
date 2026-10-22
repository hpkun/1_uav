"""Independent read-only score/reward reconstruction. No env, model or simulation imports."""
import argparse
import json
import math
from pathlib import Path
import numpy as np


def independent_credit(before,after,c0,c1,masks,radius,dt,fraction_epsilon):
    """Solve every entry/exit time independently, then integrate active sets."""
    n=len(before['red']); credits={t:np.zeros(n) for t in before}; dwell={t:np.zeros((n,len(c0))) for t in before}
    for j in range(len(c0)):
        intervals={}; breaks=[0.,1.]
        for t in before:
            for i in np.flatnonzero(masks[t]):
                a=before[t][i,:3]-c0[j]; b=after[t][i,:3]-c1[j]; v=b-a
                speed2=float(v@v)
                if speed2<1e-20:
                    roots=(0.,1.) if float(a@a)<=radius**2 else None
                else:
                    closest=-float(a@v)/speed2
                    residual=radius**2-float((a+closest*v)@(a+closest*v))
                    if residual<=0: roots=None
                    else:
                        half=math.sqrt(residual/speed2); lo=max(0.,closest-half); hi=min(1.,closest+half)
                        roots=(lo,hi) if hi>lo else None
                if roots:
                    intervals[t,int(i)]=roots; breaks.extend(roots); dwell[t][i,j]=(roots[1]-roots[0])*dt
        breaks=sorted(set(breaks))
        for lo,hi in zip(breaks[:-1],breaks[1:]):
            if hi-lo<=fraction_epsilon: continue
            middle=(lo+hi)/2
            occupants={t:[i for (team,i),(a,b) in intervals.items() if team==t and a<middle<b] for t in before}
            for t,other in [('red','blue'),('blue','red')]:
                if occupants[t] and not occupants[other]: credits[t][occupants[t]]+=(hi-lo)*dt/len(occupants[t])
    return credits,dwell


def audit(path):
    def invalid_constant(value):
        raise ValueError('nonfinite JSON value: '+value)
    data=json.loads(Path(path).read_text(encoding='utf-8'),parse_constant=invalid_constant); cfg=data['config']; frames=data['frames']
    if not frames or len(frames)<2: raise ValueError('trajectory needs at least one transition')
    expected_scores=dict(red=0.,blue=0.); max_score=max_reward=max_dwell=max_event=max_action=0.
    summaries={t:dict(path=np.zeros(len(frames[0][t]['states'])),active_steps=np.zeros(len(frames[0][t]['states']),dtype=int)) for t in expected_scores}
    for previous,current in zip(frames[:-1],frames[1:]):
        if current['step']!=previous['step']+1: raise ValueError('nonconsecutive trajectory')
        before={t:np.asarray(previous[t]['states'],dtype=float) for t in expected_scores}
        after={t:np.asarray(current[t]['states'],dtype=float) for t in expected_scores}
        if not all(np.isfinite(x).all() for x in [*before.values(),*after.values()]): raise ValueError('nonfinite trajectory states')
        masks={t:np.asarray(current[t]['current_active_mask'],dtype=bool)&np.asarray(current[t]['active_mask'],dtype=bool) for t in expected_scores}
        c0=np.asarray(previous['regions']); c1=np.asarray(current['regions'])
        credits,dwell=independent_credit(before,after,c0,c1,masks,cfg['region_radius'],cfg['dt'],cfg['residence_fraction_epsilon'])
        ledger_dwell={t:np.zeros_like(dwell[t]) for t in expected_scores}
        last_end={}; event_keys=set()
        for event in current['events']:
            region=event['region_id']; lo=event['start_fraction']; hi=event['end_fraction']
            if not 0<=region<len(c0) or not 0<=lo<hi<=1 or event['step']!=current['step']:
                raise ValueError('invalid region event interval')
            key=(region,lo,hi)
            if key in event_keys or lo<last_end.get(region,0)-1e-12: raise ValueError('duplicated/overlapping region event')
            event_keys.add(key); last_end[region]=hi
            duration=(hi-lo)*cfg['dt']; max_event=max(max_event,abs(duration-event['duration']))
            midpoint=(lo+hi)/2; center=c0[region]+midpoint*(c1[region]-c0[region])
            for t in expected_scores:
                position=before[t][:,:3]+midpoint*(after[t][:,:3]-before[t][:,:3])
                expected_ids=np.flatnonzero(masks[t] & (np.linalg.norm(position-center,axis=1)<=cfg['region_radius'])).tolist()
                if event[t+'_ids']!=expected_ids: raise ValueError('event occupants disagree with recorded geometry')
                ledger_dwell[t][expected_ids,region]+=duration
        for t in expected_scores: expected_scores[t]+=float(credits[t].sum())
        delta=expected_scores['red']-expected_scores['blue']; outcome=0 if abs(delta)<1e-10 else (1 if delta>0 else -1)
        for t,sign in [('red',1),('blue',-1)]:
            row=current[t]; n=len(after[t]); active=np.asarray(row['active_mask'],dtype=bool)
            valid=np.asarray(row['current_active_mask'],dtype=bool)
            candidate=np.asarray(row['actions'])*[2*np.pi,3,8]+[-np.pi,-1,0]
            if valid.any(): max_action=max(max_action,float(np.max(abs(candidate[valid]-np.asarray(row['physical_actions'])[valid]))))
            max_event=max(max_event,float(np.max(abs(ledger_dwell[t]-dwell[t]))))
            sums={k:np.zeros(n) for k in ('control','tracking','separation','numerical','result')}
            for i in np.flatnonzero(valid):
                center=c0[current['region_ids'][i]]
                sums['tracking'][i]=cfg['tracking_weight']*(np.linalg.norm(before[t][i,:3]-center)-np.linalg.norm(after[t][i,:3]-center))/cfg['distance_scale']
                sums['control'][i]=cfg['control_weight']*credits[t][i]
                sums['numerical'][i]=-cfg['numerical_weight']*sum(e[i] for e in row['numerical_events'].values())
                if current['finished']: sums['result'][i]=cfg['result_weight']*sign*outcome
            for i in np.flatnonzero(active):
                others=np.flatnonzero(active & (np.arange(n)!=i))
                if len(others): sums['separation'][i]=-cfg['separation_weight']*np.maximum(0,1-np.linalg.norm(after[t][others,:3]-after[t][i,:3],axis=1)/cfg['safe_distance']).mean()
            for k in sums: max_reward=max(max_reward,float(np.max(abs(sums[k]-np.asarray(row['reward_components'][k])))))
            max_reward=max(max_reward,float(np.max(abs(sum(sums.values())-np.asarray(row['rewards'])))))
            max_score=max(max_score,abs(expected_scores[t]-current['scores'][t]),float(np.max(abs(credits[t]-np.asarray(row['control_credit'])))))
            max_dwell=max(max_dwell,float(np.max(abs(dwell[t]-np.asarray(row['dwell'])))))
            summaries[t]['path']+=np.linalg.norm(after[t][:,:3]-before[t][:,:3],axis=1)*valid
            summaries[t]['active_steps']+=valid
    return dict(source=str(Path(path).resolve()),seed=data['seed'],provenance=data.get('provenance'),steps=frames[-1]['step'],
                recomputed_scores=expected_scores,max_score_error=max_score,max_reward_error=max_reward,max_dwell_error=max_dwell,
                max_event_error=max_event,max_action_mapping_error=max_action,
                passed=max(max_score,max_reward,max_dwell,max_event,max_action)<1e-8,
                flight={t:{k:v.tolist() for k,v in s.items()} for t,s in summaries.items()},
                assumptions='residence is exact for recorded piecewise-linear relative segments; this tool does not resimulate RK4 or infer missing policies')


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('trajectory',type=Path); p.add_argument('--output',type=Path); args=p.parse_args()
    result=audit(args.trajectory); text=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)
    if args.output:
        if args.output.resolve()==args.trajectory.resolve(): raise ValueError('cannot overwrite source')
        with args.output.open('x',encoding='utf-8') as f: f.write(text+'\n')
    else: print(text)
    if not result['passed']: raise SystemExit(1)


if __name__=='__main__': main()
