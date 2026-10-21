"""Read-only training report. No model loading, simulation, torch or CUDA dependency."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from algorithm.common.progress import finite_number,number,percent,loss_values,format_model,episode_count


def dig(row,*keys):
    for key in keys:
        if not isinstance(row,dict): return None
        row=row.get(key)
    return row


def nonfinite_paths(value,path=''):
    if isinstance(value,dict):
        return [p for k,v in value.items() for p in nonfinite_paths(v,path+'/'+str(k))]
    if isinstance(value,list):
        return [p for i,v in enumerate(value) for p in nonfinite_paths(v,path+f'/{i}')]
    bad=isinstance(value,float) and not math.isfinite(value)
    bad=bad or isinstance(value,str) and value.strip().lower() in ('nan','inf','-inf','+inf','infinity','-infinity','+infinity')
    return [path] if bad else []


def load_run(run_dir):
    root=Path(run_dir); warnings=[]
    def read_object(name):
        try:
            value=json.loads((root/name).read_text(encoding='utf-8-sig'))
            if not isinstance(value,dict): raise ValueError('expected object')
            return value
        except (OSError,ValueError) as error:
            warnings.append(f'{name}: unavailable or malformed; missing values shown as N/A ({type(error).__name__})')
            return {}
    config=read_object('run_config.json'); summary=read_object('run_summary.json')
    records=[]; history=[]
    try:
        with (root/'training_metrics.jsonl').open(encoding='utf-8-sig') as f:
            for i,line in enumerate(f,1):
                if not line.strip(): continue
                try:
                    row=json.loads(line)
                    if not isinstance(row,dict): raise ValueError('expected object')
                    records.append(row)
                except ValueError: warnings.append(f'training_metrics.jsonl row {i}: malformed record ignored in report only')
    except OSError: warnings.append('training_metrics.jsonl: unavailable; metrics shown as N/A')
    try:
        with (root/'evaluation_history.csv').open(encoding='utf-8-sig',newline='') as f:
            history=list(csv.DictReader(f,strict=True))
    except (OSError,csv.Error): warnings.append('evaluation_history.csv: unavailable or malformed; history shown as N/A')
    data=dict(config=config,summary=summary,records=records,history=history,warnings=warnings)
    if nonfinite_paths({k:v for k,v in data.items() if k!='warnings'}):
        warnings.append('Nonfinite numeric values found; they are excluded from means and displayed as N/A')
    if any(loss_values(r.get('optimization'))[1] for r in records):
        warnings.append('Historical v2 actor_loss means POLICY LOSS; actor_total_loss was not recorded (N/A). Original files remain unchanged.')
    recorded_steps=[finite_number(r.get('sampled_steps')) for r in records]
    recorded_steps=[s for s in recorded_steps if s is not None]
    summary_steps=finite_number(summary.get('sampled_steps'))
    if any(b<=a for a,b in zip(recorded_steps,recorded_steps[1:])):
        warnings.append('Training log contains duplicate/out-of-order sampled_steps; inspect resume provenance before appending.')
    if summary_steps is not None and recorded_steps and recorded_steps[-1]!=summary_steps:
        warnings.append('Last training log step differs from run_summary.json; inspect resume/log consistency.')
    return data


def average(values):
    values=[n for value in values if (n:=finite_number(value)) is not None]
    return mean(values) if values else None


def diagnostics(data,window):
    rows=data['records'][-window:]; hints=[]
    evaluations=[finite_number(r.get('success_rate')) for r in data['history']]
    evaluations=[x for x in evaluations if x is not None][-window:]
    if len(evaluations)>=3 and all(x==0 for x in evaluations):
        hints.append(f'Independent evaluation success stayed at 0% for {len(evaluations)} reports; inspect task progress and evaluation n.')
    completion=[finite_number(r.get('completion_rate')) for r in data['history']]
    completion=[x for x in completion if x is not None][-window:]
    if len(completion)>=3 and max(completion)<=completion[0]+1e-6:
        hints.append('Independent completion has not increased over the recent evaluation window.')
    distance=[finite_number(dig(r,'rollout','mean_target_distance')) for r in rows]
    distance=[x for x in distance if x is not None]
    if len(distance)>=3 and all(b>a for a,b in zip(distance,distance[1:])):
        hints.append('Mean assigned-target distance increased at every recent update; reassignment can also affect this metric.')
    for section,key,limit,label in (('rollout','gamma_clip_rate',.1,'gamma clipping'),('rollout','speed_clip_rate',.1,'speed clipping'),
            ('optimization','approx_kl',.03,'PPO KL'),('optimization','gradient_norm',5.,'pre-clipping gradient norm')):
        values=[finite_number(dig(r,section,key)) for r in rows]
        values=[x for x in values if x is not None]
        if len(values)>=3 and sum(x>limit for x in values)/len(values)>=.8:
            hints.append(f'Persistently elevated {label}: >{limit:g} in at least 80% of the recent window.')
    return hints


def render_report(data,run_dir,last_updates=10):
    if type(last_updates) is not int or last_updates<1: raise ValueError('last_updates must be positive')
    config,summary,rows,history=(data[k] for k in ('config','summary','records','history'))
    total=dig(config,'environment','scale'); scale=total if isinstance(total,str) else 'N/A'
    point_counts={'3U3P':3,'3U5P':5,'5U5P':5}; points=point_counts.get(scale)
    last=rows[-1] if rows else {}
    lines=[f'[REPORT] {Path(run_dir).resolve()}',
        f"Scale {scale} | Environment {config.get('environment_version',summary.get('environment_version','N/A'))}"
        f" | Algorithm {config.get('algorithm_version',summary.get('algorithm_version','N/A'))}",
        f"Sampled steps {number(summary.get('sampled_steps',last.get('sampled_steps')),0,group=True)}"
        f" | PPO updates {number(summary.get('updates',last.get('update')),0)}"
        f" | Training episodes {number(summary.get('joint_episode_count',last.get('joint_episode_count')),0)}",
        '', 'Independent evaluation history (recorded rows; n is actual evaluated episodes when available):',
        '  Steps       Update  Model     n    Success  Completion  Mean Return  Completed points']
    by_step={finite_number(r.get('sampled_steps')):r.get('independent_evaluation') for r in rows
             if finite_number(r.get('sampled_steps')) is not None and isinstance(r.get('independent_evaluation'),dict)}
    for h in history:
        step=finite_number(h.get('sampled_steps')); report=by_step.get(step,{})
        lines.append(f"  {number(step,0,group=True):>10} {number(h.get('update'),0):>7} {str(h.get('checkpoint_role') or 'N/A'):>8}"
            f" {number(episode_count(report),0):>4} {percent(h.get('success_rate')):>9} {percent(h.get('completion_rate')):>11}"
            f" {number(h.get('mean_return'),3):>12} {number(report.get('mean_completed_waypoints'),1)}/{number(points,0)}")
    if not history: lines.append('  N/A')
    lines+=['','Best / latest (independent evaluations):','  '+format_model('BEST',summary.get('best_evaluation'),points),
        '  '+format_model('LATEST',summary.get('latest_evaluation'),points),
        f"  Recorded best path: {summary.get('best_checkpoint') or 'N/A'}",f"  Recorded latest path: {summary.get('latest_checkpoint') or 'N/A'}",'']
    first,tail=rows[:10],rows[-last_updates:]
    lines+= [f'Optimization means (equal weight per update): first {len(first)} / last {len(tail)}',
             '  Metric                 First           Last']
    fields=('policy_loss','actor_total_loss','critic_loss','entropy','approx_kl','clip_fraction','gradient_norm')
    for key in fields:
        a=average(loss_values(r.get('optimization'))[0].get(key) for r in first)
        b=average(loss_values(r.get('optimization'))[0].get(key) for r in tail)
        show=(lambda v:percent(v,2)) if key=='clip_fraction' else (lambda v:number(v,5))
        lines.append(f'  {key:<21} {show(a):>12} {show(b):>14}')
    if len(rows)<10+last_updates: lines.append('  Note: first/last windows overlap when history is short.')
    lines+=['','Task / completed training-episode trends (first-window mean -> last-window mean):']
    for section,key,label,formatter in (
        ('rollout','mean_target_distance','Target distance',lambda v:number(v,1)+' m'),
        ('rollout','target_distance_reduction','Distance reduction per rollout',lambda v:number(v,1,signed=True)+' m'),
        ('training','success_rate','Training success',percent),('training','completion_rate','Training completion',percent),
        ('training','mean_episode_return','Training return',lambda v:number(v,3)),
        ('rollout','speed_clip_rate','Speed clipping',percent),('rollout','gamma_clip_rate','Gamma clipping',percent)):
        a=average(dig(r,section,key) for r in first); b=average(dig(r,section,key) for r in tail)
        lines.append(f'  {label}: {formatter(a)} -> {formatter(b)}')
    lines+=['','Beta actions (last recorded rollout, dimensions mu / tangential / normal):']
    for key in ('beta_action_mean','beta_action_std'):
        value=dig(last,'rollout',key)
        entries=[value[i] if isinstance(value,list) and i<len(value) else None for i in range(3)]
        lines.append(f"  {key}: ["+', '.join(number(x,4) for x in entries)+']')
    lines+=['','Five reward component means (per-valid-UAV-step metrics, equal update weight):']
    for key in ('progress','completion','separation','efficiency','terminal'):
        a=average(dig(r,'rollout','reward_components_mean',key) for r in first)
        b=average(dig(r,'rollout','reward_components_mean',key) for r in tail)
        lines.append(f'  {key}: {number(a,5)} -> {number(b,5)}')
    lines+=['','Data notes / diagnostic hints (not an automatic failure or convergence verdict):']
    hints=data['warnings']+diagnostics(data,last_updates)
    lines+=['  - '+message for message in hints] if hints else ['  No configured heuristic triggered; missing information can limit diagnostics.']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',required=True); p.add_argument('--last-updates',type=int,default=10)
    a=p.parse_args()
    if a.last_updates<1: p.error('--last-updates must be positive')
    print(render_report(load_run(a.run_dir),a.run_dir,a.last_updates))


if __name__=='__main__': main()
