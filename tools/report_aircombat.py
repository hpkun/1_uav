"""Static Section IV figures/tables from actual archived runs; never paper values."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from algorithm.experiments import FACTORS,write_json


def mean_std(values):
    a=np.asarray(values,dtype=float)
    if not len(a):return dict(mean=None,std=None,n=0)
    if not np.isfinite(a).all():raise ValueError('statistics require finite actual results')
    return dict(mean=float(a.mean()),std=float(a.std(ddof=1)) if len(a)>1 else None,n=len(a))


def moving_average(values,window=20):
    if window<1:raise ValueError('window must be positive')
    a=np.asarray(values,dtype=float)
    return np.convolve(a,np.ones(window)/window,mode='valid') if len(a)>=window else np.array([])


def read_runs(root):
    runs=[];identities=set()
    for path in sorted(Path(root).glob('*/seed_*/experiment.json')):
        meta=json.loads(path.read_text(encoding='utf-8'))
        identity=(meta['experiment'],meta['training_seed'])
        if identity in identities:raise ValueError('duplicate experiment/seed result')
        identities.add(identity)
        log=path.parent/'train.jsonl';evaluation=path.parent/'final_evaluation.json'
        records=[json.loads(line) for line in log.read_text(encoding='utf-8').splitlines() if line.strip()] if log.exists() else []
        if any(b['updates']<=a['updates'] for a,b in zip(records,records[1:])):
            raise ValueError('training log updates must be strictly increasing')
        result=json.loads(evaluation.read_text(encoding='utf-8')) if evaluation.exists() else None
        if result and result.get('training_seed')!=meta['training_seed']:raise ValueError('evaluation training seed mismatch')
        runs.append(dict(meta=meta,records=records,evaluation=result,path=str(path.parent.resolve())))
    return runs


def training_episodes(records):
    result=[]
    for row in records:
        episodes=row.get('episodes',[])
        start=row['completed_episodes']-len(episodes)
        for k,episode in enumerate(episodes,1):result.append(dict(episode=start+k,**episode))
    return result


def final_stats(runs,name,metric):
    selected=[r for r in runs if r['meta']['experiment']==name and r['evaluation'] is not None]
    result=mean_std([r['evaluation'][metric] for r in selected])
    expected=1 if name=='sensitivity_baseline' or name.startswith(('eta_','e1_','e2_')) else 5
    result.update(expected_seeds=expected,status='formal_complete' if len(selected)==expected and all(
        not r['meta']['is_smoke'] and r['evaluation']['completed_training_episodes']==5000 and
        r['evaluation']['completed_episodes']==200 for r in selected) else 'partial_or_smoke',
        training_seeds=[r['meta']['training_seed'] for r in selected])
    return result


def write_csv(path,rows,fields):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)


def save_figure(fig,path):
    fig.tight_layout();fig.savefig(str(path)+'.png',dpi=160);fig.savefig(str(path)+'.svg');plt.close(fig)


def plot_curves(runs,output,metric,label,figure,window=1):
    fig,axes=plt.subplots(1,3,figsize=(12,3.5),squeeze=False)
    exported=[]
    for ax,scale in zip(axes[0],('3v3','5v5','3v5')):
        selected=[r for r in runs if r['meta']['experiment']=='main_'+scale]
        series=[]
        for run in selected:
            if metric=='episode_return':
                rows=training_episodes(run['records']);x=np.array([r['episode'] for r in rows]);y=np.array([r[metric] for r in rows])
            else:
                rows=run['records'];x=np.array([r['updates'] for r in rows]);y=np.array([r['ppo'][metric] for r in rows])
            if window>1:x=x[window-1:];y=moving_average(y,window)
            if len(x):series.append((x,y))
        if series:
            common=sorted(set.intersection(*(set(x.tolist()) for x,_ in series)))
            if common:
                values=np.array([[dict(zip(x,y))[i] for i in common] for x,y in series])
                mean=values.mean(0);std=values.std(0,ddof=1) if len(series)>1 else None
                ax.plot(common,mean,lw=1,label=f'n={len(series)} seed(s)')
                if std is not None:ax.fill_between(common,mean-std,mean+std,alpha=.2)
                for k,i in enumerate(common):exported.append(dict(scale=scale,x=i,mean=mean[k],std=std[k] if std is not None else None,n=len(series)))
                ax.legend(fontsize=8)
        if not ax.lines:ax.text(.5,.5,'No data' if window==1 else 'Requires 20 recorded updates',ha='center',transform=ax.transAxes,fontsize=9)
        ax.set(title=scale,xlabel='Completed episode' if metric=='episode_return' else 'PPO update',ylabel=label)
        ax.grid(alpha=.2)
    fig.suptitle(f'Fig {figure} — LOCAL results (sample std; incomplete runs labelled in summary)')
    save_figure(fig,output/f'fig{figure}')
    write_csv(output/f'fig{figure}_data.csv',exported,['scale','x','mean','std','n'])


def report(root,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    runs=read_runs(root)
    summary=dict(source='actual local train.jsonl and final_evaluation.json; no paper performance numbers',runs=len(runs))
    summary['fig5']={scale:final_stats(runs,'main_'+scale,'win_rate') for scale in ('3v3','5v5','3v5')}
    fig,ax=plt.subplots(figsize=(6,3.5))
    for k,(scale,s) in enumerate(summary['fig5'].items()):
        if s['mean'] is not None:
            ax.bar(k,100*s['mean'],width=.6,label=f"{scale}: n={s['n']}, {s['status']}")
            if s['std'] is not None:ax.errorbar(k,100*s['mean'],yerr=100*s['std'],fmt='none',color='black',capsize=4)
            ax.text(k,min(98,100*s['mean']+1),f"{100*s['mean']:.1f}%",ha='center',fontsize=9)
    ax.set(xticks=range(3),xticklabels=list(summary['fig5']),ylabel='Win rate (%)',ylim=(0,100),title='Fig 5 — LOCAL final evaluations')
    if ax.patches:ax.legend(fontsize=7)
    save_figure(fig,output/'fig5')
    table4=[]
    for name in ('main_3v3','without_aph','without_allied','without_masking'):
        row=dict(method=name)
        for metric in ('win_rate','mean_red_survivors','proximity_risk','shared_exposure_risk'):
            s=final_stats(runs,name,metric)
            label='win_rate_percent' if metric=='win_rate' else metric
            scale=100 if metric=='win_rate' else 1
            row.update({label+'_mean':None if s['mean'] is None else scale*s['mean'],
                        label+'_std':None if s['std'] is None else scale*s['std']})
        row.update(n=s['n'],status=s['status']);table4.append(row)
    summary['table_IV']=table4
    write_csv(output/'table_IV.csv',table4,['method','n','status',*[x+s for x in ('win_rate_percent','mean_red_survivors','proximity_risk','shared_exposure_risk') for s in ('_mean','_std')]])
    table5=[];fig,axes=plt.subplots(1,3,figsize=(10,3.5))
    for ax,(factor,(_,values,baseline)) in zip(axes,FACTORS.items()):
        points=[]
        for value in values:
            name='sensitivity_baseline' if value==baseline else f'{factor}_{value}'
            s=final_stats(runs,name,'win_rate')
            table5.append(dict(factor=factor,value=value,experiment=name,win_rate_percent=None if s['mean'] is None else 100*s['mean'],n=s['n'],status=s['status']))
            if s['mean'] is not None:points.append((value,100*s['mean']))
        if points:ax.plot(*np.array(points).T,'o-')
        else:ax.text(.5,.5,'No local evaluation',ha='center',transform=ax.transAxes)
        ax.set(xlabel=factor,ylabel='Win rate (%)',xticks=values,ylim=(0,100));ax.grid(alpha=.2)
    fig.suptitle('Fig 9 — LOCAL one-factor results; one training seed')
    save_figure(fig,output/'fig9')
    summary['table_V']=table5
    write_csv(output/'table_V.csv',table5,['factor','value','experiment','win_rate_percent','n','status'])
    plot_curves(runs,output,'episode_return','Training episodic reward',10)
    plot_curves(runs,output,'critic_loss','Critic value loss (20-update MA)',11,20)
    plot_curves(runs,output,'entropy','Policy entropy (20-update MA)',12,20)
    summary['blue_probability']=[];aggregate={}
    for path in sorted(Path(root).glob('blue_*/*.json')):
        item=json.loads(path.read_text(encoding='utf-8'))
        if item.get('kind')=='probability':
            rates=[]
            for seed in item['evaluation_protocol']['seeds']:
                group=[e['win'] for e in item['episodes'] if e['seed']==seed]
                rates.append(float(np.mean(group)))
            summary['blue_probability'].append(dict(probability=item['variant'],**mean_std(rates),
                std_unit='evaluation seed groups, each 200 episodes in formal protocol',is_smoke=item['is_smoke']))
        elif item.get('kind')=='aggregation':
            aggregate.setdefault(item['variant'],[]).append(item)
    summary['blue_aggregation']={mode:dict(**mean_std([x['win_rate'] for x in items]),
        std_unit='independent training checkpoints',is_smoke=any(x['is_smoke'] for x in items)) for mode,items in aggregate.items()}
    write_csv(output/'blue_probability.csv',summary['blue_probability'],['probability','mean','std','n','is_smoke','std_unit'])
    write_csv(output/'blue_aggregation.csv',[dict(mode=k,**v) for k,v in summary['blue_aggregation'].items()],['mode','mean','std','n','is_smoke','std_unit'])
    write_json(output/'summary.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='outputs/xu_iv');p.add_argument('--output',default='outputs/xu_iv_report')
    a=p.parse_args();print(json.dumps(report(a.root,a.output),ensure_ascii=False))


if __name__=='__main__':main()
