import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tools.report_training import load_run,render_report


def historical_run(root):
    (root/'run_config.json').write_text(json.dumps(dict(environment_version='coop3d-v2.0.0',algorithm_version='mappo-coop3d-v2.0.0',environment={'scale':'3U3P'})))
    report=dict(success_rate=0.,completion_rate=0.,mean_completed_waypoints=0.,mean_reward=-2.,mean_episode_length=250.,episodes=[{'success':False}]*5)
    (root/'run_summary.json').write_text(json.dumps(dict(sampled_steps=3000,updates=3,joint_episode_count=30,best_evaluation=report,latest_evaluation=report,
        best_checkpoint='/original/best.pt',latest_checkpoint='/original/latest.pt')))
    rows=[]
    for i in range(3):
        rows.append(dict(sampled_steps=1000*(i+1),update=i+1,joint_episode_count=10*(i+1),independent_evaluation=report,
            optimization=dict(actor_loss=-.01*(i+1),critic_loss=.1,entropy=-2.,approx_kl=.05,clip_fraction=.03,gradient_norm=6.),
            training=dict(success_rate=0.,completion_rate=0.,mean_episode_return=-2.),
            rollout=dict(mean_target_distance=1000+i*100,target_distance_reduction=-100.,speed_clip_rate=.2,gamma_clip_rate=.3,
                beta_action_mean=[.5,1/3,.125],beta_action_std=[.1,.1,.1],reward_components_mean=dict(progress=-.2,completion=0.,separation=0.,efficiency=-.01,terminal=0.))))
    (root/'training_metrics.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    with (root/'evaluation_history.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['sampled_steps','update','checkpoint_role','success_rate','completion_rate','mean_return'])
        for r in rows: w.writerow([r['sampled_steps'],r['update'],'current',0,0,-2])


def test_legacy_report_read_only_and_diagnostics(tmp_path):
    historical_run(tmp_path)
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}
    text=render_report(load_run(tmp_path),tmp_path,10)
    for word in ('3U3P','3,000','Historical v2 actor_loss means POLICY LOSS','actor_total_loss','N/A','n=5',
        'Target distance','Beta actions','Five reward component','speed clipping','gamma clipping','PPO KL','gradient norm',
        'stayed at 0%','not increased','distance increased'): assert word in text
    assert before=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}


def test_missing_and_invalid_fields_do_not_crash(tmp_path):
    text=render_report(load_run(tmp_path),tmp_path)
    assert 'N/A' in text
    historical_run(tmp_path)
    with (tmp_path/'training_metrics.jsonl').open('a') as f:
        f.write('{"sampled_steps": 4000, "optimization": {"actor_loss": NaN}}\nBROKEN\n')
    text=render_report(load_run(tmp_path),tmp_path)
    assert 'Nonfinite numeric values found' in text and 'malformed record' in text
    with (tmp_path/'training_metrics.jsonl').open('a') as f:
        f.write('{"sampled_steps": null, "optimization": 7}\n')
    assert 'N/A' in render_report(load_run(tmp_path),tmp_path)


def test_offline_cli_does_not_import_models_or_cuda(tmp_path):
    historical_run(tmp_path)
    script="""import builtins,runpy,sys
original=builtins.__import__
def guarded(name,*args,**kwargs):
 if name=='torch' or name=='env' or name.startswith(('env.','algorithm.mappo')): raise AssertionError('forbidden offline import '+name)
 return original(name,*args,**kwargs)
builtins.__import__=guarded
sys.argv=['report','--run-dir',sys.argv[1],'--last-updates','10']
runpy.run_module('tools.report_training',run_name='__main__')
"""
    result=subprocess.run([sys.executable,'-c',script,str(tmp_path)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert '[REPORT]' in result.stdout and 'POLICY LOSS' in result.stdout
