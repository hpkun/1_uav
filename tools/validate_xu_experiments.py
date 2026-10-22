"""Bounded Section IV CUDA acceptance; actual records clearly labelled smoke."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import torch
from algorithm.experiments import (CATALOG,configurations,manifest,write_json,export_plan,run_experiment,
                                    run_directory,final_evaluation,blue_sensitivity)
from algorithm.mappo.trainer import ProtocolMAPPOTrainer
from algorithm.common.aircombat_vector import AirCombatVectorEnv,combat_metrics
from algorithm.common.checkpoint import save_protocol_checkpoint,read_protocol_checkpoint
from algorithm.common.runtime import require_cuda
from env.aircombat.config import ENVIRONMENT_VERSION
from tools.report_aircombat import report
from tools.aircombat_trajectories import record


def sample_run(name,seed,root):
    env,cfg=configurations(name,seed,smoke=True)
    cfg=replace(cfg,num_envs=1,horizon=2,training_episodes=2,minibatch_size=32)
    t=ProtocolMAPPOTrainer(cfg);path=run_directory(root,name,seed);path.mkdir(parents=True)
    with AirCombatVectorEnv(env,1,training_seed_index=seed) as v:
        batch,episodes=t.collect(v);metric=t.update(batch)
        save_protocol_checkpoint(path/'latest.pt',t,env_config=env,vector=v)
    meta=manifest(name,seed,smoke=True);meta['training']=cfg.to_dict()
    meta['validation']='one real 2-step rollout and one update; training budget not completed'
    write_json(path/'experiment.json',meta)
    row=dict(completed_episodes=t.training_episode_count,sampled_steps=t.sampled_steps,updates=t.updates,
             episode_metrics=combat_metrics(episodes),rollout=t.rollout_metrics,ppo=metric,episodes=episodes)
    (path/'train.jsonl').write_text(json.dumps(row,allow_nan=False)+'\n',encoding='utf-8')
    final_evaluation(path/'latest.pt',path/'final_evaluation.json',episodes=2)
    return dict(experiment=name,seed=seed,valid_samples=metric['valid_samples'],updates=t.updates,
                completed_training_episodes=t.training_episode_count)


def validate(output):
    require_cuda('cuda');output=Path(output)
    if output.exists() and any(output.iterdir()):raise FileExistsError('use a fresh acceptance directory')
    output.mkdir(parents=True);root=output/'runs'
    export_plan(output/'formal_plan.json')
    # One actual two-episode training loop exercises archived episodic reward,
    # exact checkpoint callbacks, final evaluation and 20-update plot handling.
    trained=run_experiment('main_3v3',1,root,smoke=True)
    result=dict(device=torch.cuda.get_device_name(0),formal_training_executed=False,
                short_training=dict(episodes=trained.training_episode_count,sampled_steps=trained.sampled_steps,updates=trained.updates),
                single_rollout_checks=[])
    for name in CATALOG:
        if name=='main_3v3':continue
        result['single_rollout_checks'].append(sample_run(name,1,root))
        print(f'validated {name}: one real rollout/update',flush=True)
    for scale in ('3v3','5v5'):
        for seed in (2,3,4,5):result['single_rollout_checks'].append(sample_run('main_'+scale,seed,root))
    checkpoints=[run_directory(root,'main_5v5',s)/'latest.pt' for s in (1,2,3,4,5)]
    blue_sensitivity(checkpoints[:1],root/'blue_probability',kind='probability',smoke=True)
    blue_sensitivity(checkpoints,root/'blue_aggregation',kind='aggregation',smoke=True)
    result['blue_smoke']=dict(probability_episodes=3*5,aggregation_episodes=5*3*5)
    result['trajectories']={}
    for scale,scenarios in (('3v3',('random_3v3','head_on_3v3','rear_end_3v3')),
                            ('5v5',('representative_5v5',)),('3v5',('representative_3v5',))):
        checkpoint=run_directory(root,'main_'+scale,1)/'latest.pt'
        for scenario in scenarios:
            data=record(checkpoint,scenario,output/'trajectories'/scenario)
            result['trajectories'][scenario]=dict(steps=data['summary']['episode_length'],events=len(data['events']),outcome=data['summary']['outcome'])
    result['statistics']=report(root,output/'figures')
    final=run_directory(root,'main_3v3',1)
    targets=[json.loads(x) for x in (final/'checkpoints.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [r['target_episode'] for r in targets]==[1,2]
    for row in targets:
        payload=read_protocol_checkpoint(final/row['path'],environment_version=ENVIRONMENT_VERSION)
        assert payload['training_episode_count']==row['target_episode']
    result['exact_checkpoint_targets']=[r['target_episode'] for r in targets]
    write_json(output/'acceptance.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('statistics','single_rollout_checks')},ensure_ascii=False),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='outputs/xu_iv_acceptance')
    validate(p.parse_args().output)


if __name__=='__main__':main()
