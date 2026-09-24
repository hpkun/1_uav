"""Offline-only preregistered analysis of six matched team-credit branches."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SEEDS=(5301,5302,5303);TARGET=1_805_280;SOURCE=1_505_280

def rows_jsonl(path):return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def endpoint(path):
 with path.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
 hit=[r for r in rows if int(r['sampled_steps'])==TARGET]
 if len(hit)!=1:raise RuntimeError(f'{path}: exact 1805280 evaluation missing/duplicated')
 r=hit[0];f=lambda k:float(r[k])
 w1,w2,w3=f('clear_wave_1_probability'),f('clear_wave_2_probability'),f('clear_wave_3_probability')
 return {'W1':w1,'W2':w2,'W3':w3,'Q2':None if w1==0 else w2/w1,'Q3':None if w2==0 else w3/w2,'AverageWaves':f('average_waves_cleared'),'Return':f('average_return'),'RedLoss':f('average_red_loss'),'BlueLoss':f('average_blue_loss'),'Boundary':f('average_red_boundary_exits'),'Ground':f('average_red_ground_losses'),'EpisodeLength':f('average_episode_length'),'evaluation_episodes':int(float(r['evaluation_episodes'])),'evaluation_seed_base':int(float(r['evaluation_seed_base'])),'evaluation_seed_end':int(float(r['evaluation_seed_end']))}
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output-dir',default='outputs/team_credit_300k_analysis');args=parser.parse_args();out=ROOT/args.output_dir
 records=[];mechanism=[]
 for seed in SEEDS:
  pair={}
  for branch in ('control','teammean'):
   d=ROOT/f'outputs/dev_team_credit_{branch}_seed{seed}_300k'
   required=[d/x for x in ('run_summary.json','run_config.json','branch_from.json','evaluation_history.csv','optimization_metrics.jsonl','training_metrics.jsonl','latest.pt','final.pt')]
   if any(not p.exists() for p in required):raise FileNotFoundError(f'incomplete branch {d}')
   summary=json.loads((d/'run_summary.json').read_text());run=json.loads((d/'run_config.json').read_text());prov=json.loads((d/'branch_from.json').read_text())
   if int(summary['sampled_steps'])!=TARGET or int(run['total_sampled_steps'])!=TARGET:raise RuntimeError(f'{d}: target mismatch')
   if int(prov['parent_sampled_steps'])!=SOURCE or int(prov['source_training_seed'])!=seed or not prov.get('source_checkpoint_unchanged'):raise RuntimeError(f'{d}: source provenance mismatch')
   ep=endpoint(d/'evaluation_history.csv')
   if ep['evaluation_episodes']!=50 or (ep['evaluation_seed_base'],ep['evaluation_seed_end'])!=(44_000_000,44_000_049):raise RuntimeError(f'{d}: evaluation protocol mismatch')
   text='\n'.join((d/x).read_text(errors='ignore') for x in ('train.log','optimization_metrics.jsonl','training_metrics.jsonl'))
   if any(x in text.lower() for x in ('traceback','out of memory','nan','inf')):raise RuntimeError(f'{d}: failure/non-finite marker')
   opt=[r for r in rows_jsonl(d/'optimization_metrics.jsonl') if SOURCE<int(r['sampled_steps'])<=TARGET]
   def total(k):return sum(float(r.get(k,0)) for r in opt)
   entries={w:{'count':total(f'natural_entry_count_wave{w}'),'survivors':total(f'natural_entry_survivor_sum_wave{w}')} for w in (2,3)}
   for w in entries:entries[w]['mean']=None if entries[w]['count']==0 else entries[w]['survivors']/entries[w]['count']
   episodes=[r for r in rows_jsonl(d/'training_metrics.jsonl') if SOURCE<int(r['sampled_steps'])<=TARGET]
   means={k:(sum(float(r[k]) for r in episodes)/len(episodes) if episodes else None) for k in ('red_losses','red_boundary_exits','red_ground_losses','waves_cleared')}
   pair[branch]={'endpoint':ep,'entries':entries,'training_episode_means':means,'source_sha256':prov['parent_checkpoint_sha256'],'run':run,'provenance':prov}
  if pair['control']['source_sha256']!=pair['teammean']['source_sha256']:raise RuntimeError(f'seed {seed}: unmatched source SHA')
  metrics={k:pair['teammean']['endpoint'][k]-pair['control']['endpoint'][k] for k in ('W1','W2','W3','AverageWaves','Return','RedLoss','BlueLoss','Boundary','Ground','EpisodeLength')}
  metrics.update({'seed':seed,'control_source_sha256':pair['control']['source_sha256']})
  for w in (2,3):
   c,t=pair['control']['entries'][w],pair['teammean']['entries'][w];metrics[f'W{w}_entry_count_control']=c['count'];metrics[f'W{w}_entry_count_treatment']=t['count'];metrics[f'W{w}_entry_survivor_mean_control']=c['mean'];metrics[f'W{w}_entry_survivor_mean_treatment']=t['mean'];metrics[f'delta_W{w}_entry_survivor_mean']=None if c['mean'] is None or t['mean'] is None else t['mean']-c['mean']
  records.append(metrics);mechanism.append({'seed':seed,**pair})
 aw_wins=sum(r['AverageWaves']>0 for r in records);red_wins=sum(r['RedLoss']<0 for r in records);entry_valid=[r for r in records if r['delta_W2_entry_survivor_mean'] is not None];entry_wins=sum(r['delta_W2_entry_survivor_mean']>0 for r in entry_valid)
 mean=lambda k:sum(r[k] for r in records)/len(records)
 A=aw_wins>=2 and mean('AverageWaves')>0;B=red_wins>=2 and mean('RedLoss')<0;C=len(entry_valid)==3 and entry_wins>=2;D=all(r['AverageWaves']>-.5 and r['W1']>-.25 for r in records)
 decision='SAFETY_FAIL' if not D else 'NOT_SUPPORTED' if not A else 'MECHANISM_INCONCLUSIVE' if not (B and C) else 'PROMISING'
 result={'TEAM_CREDIT_300K_SCREEN':decision,'primary_endpoint':TARGET,'training_seed_replication_n':3,'paired_deltas':records,'mechanism_records':mechanism,'gates':{'A_performance':A,'B_survival':B,'C_entry':C,'D_catastrophic_guard':D,'average_waves_wins':aw_wins,'red_loss_wins':red_wins,'W2_entry_wins':entry_wins,'W2_entry_valid_seeds':len(entry_valid),'mean_delta_AverageWaves':mean('AverageWaves'),'mean_delta_RedLoss':mean('RedLoss')},'interpretation':'TEAM_LEVEL_CREDIT_ASSIGNMENT_BENEFIT only if positive; teammate-loss-only causation is not identified'}
 out.mkdir(parents=True,exist_ok=False);(out/'analysis.json').write_text(json.dumps(result,indent=2));
 with (out/'paired_endpoint.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=sorted({k for r in records for k in r}));w.writeheader();w.writerows(records)
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
