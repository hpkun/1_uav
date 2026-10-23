"""Xu Section IV experiment catalogue and small CLI, no alternative algorithms.

List/export is read-only. Training runs only one explicitly selected seed.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from env.aircombat import AirCombatConfig
from env.aircombat.config import ENVIRONMENT_VERSION
from algorithm.mappo.config import ProtocolTrainConfig
from algorithm.mappo.trainer import ProtocolMAPPOTrainer
from algorithm.common.checkpoint import read_protocol_checkpoint,restore_protocol_checkpoint
from algorithm.aircombat import train,evaluate
from algorithm.common.aircombat_progress import format_evaluation

SEEDS=(1,2,3,4,5)  # Published count; concrete five indices are a disclosed choice.
CATALOG={
    'main_3v3':dict(family='main',scale='3v3'),
    'main_5v5':dict(family='main',scale='5v5'),
    'main_3v5':dict(family='main',scale='3v5'),
    'without_aph':dict(family='ablation',scale='3v3',env=dict(assignment_mode='nearest')),
    'without_allied':dict(family='ablation',scale='3v3',env=dict(allied_penalty_enabled=False)),
    'without_masking':dict(family='ablation',scale='3v3',training=dict(ppo_masking=False)),
    'sensitivity_baseline':dict(family='coefficient',scale='3v3'),
}
FACTORS={'eta':('attack_eta',(3,5,7),5),'e1':('terminal_e1',(10,20,30),20),
         'e2':('terminal_e2',(15,25,35),25)}
for factor,(field,values,baseline) in FACTORS.items():
    for value in values:
        if value!=baseline:
            CATALOG[f'{factor}_{value}']=dict(family='coefficient',scale='3v3',env={field:value})


def configurations(name,seed=1,*,smoke=False,episodes=None):
    if episodes is not None and (type(episodes) is not int or episodes not in (2500,5000)):
        raise ValueError('experiment episodes must be 2500 or 5000')
    if smoke and episodes is not None:raise ValueError('--episodes cannot be combined with --smoke')
    if name not in CATALOG:raise ValueError(f'unknown experiment {name}')
    entry=CATALOG[name]
    allowed=(1,) if entry['family']=='coefficient' else SEEDS
    if seed not in allowed:raise ValueError(f'{name} training seeds must be {allowed}')
    env=AirCombatConfig(scale=entry['scale'],**entry.get('env',{}))
    cfg=ProtocolTrainConfig(training_seed_index=seed,seed=1000*seed,**entry.get('training',{}))
    if episodes is not None:cfg=replace(cfg,training_episodes=episodes)
    if smoke:
        cfg=replace(cfg,training_episodes=2,num_envs=2,horizon=8,ppo_epochs=1,evaluation_episodes=2,
                    checkpoint_interval_episodes=1)
    return env,cfg


def budget_kind(episodes):
    return 'paper_5000' if episodes==5000 else 'half_budget_diagnostic' if episodes==2500 else 'short_validation'


def manifest(name,seed=1,*,smoke=False,episodes=None):
    env,cfg=configurations(name,seed,smoke=smoke,episodes=episodes)
    return dict(experiment=name,family=CATALOG[name]['family'],training_seed=seed,is_smoke=smoke,
                training_budget_kind=budget_kind(cfg.training_episodes),training_episode_target=cfg.training_episodes,
                environment_version=ENVIRONMENT_VERSION,environment=env.to_dict(),training=cfg.to_dict(),
                final_evaluation=dict(episodes=cfg.evaluation_episodes,seed_start=10000000+10000*seed),
                checkpoint_targets=list(range(cfg.checkpoint_interval_episodes,cfg.training_episodes+1,cfg.checkpoint_interval_episodes)))


def run_directory(root,name,seed):return Path(root)/name/f'seed_{seed}'


def export_plan(path):
    tasks=[]
    for name,entry in CATALOG.items():
        for seed in ((1,) if entry['family']=='coefficient' else SEEDS):tasks.append(manifest(name,seed))
    data=dict(training_runs=tasks,blue_evaluation=dict(
        probabilities=dict(checkpoint='one fixed main_5v5 seed_1/latest.pt',probabilities=[1.,.8,.6],
                           seeds=[100,101,102,103,104],episodes_per_seed=200,aggregation='aggregate-sum'),
        aggregation=dict(checkpoints=[f'main_5v5/seed_{s}/episodes_004950.pt' for s in SEEDS],
                         modes=['aggregate-sum','opportunistic-max','threat-aware-maximin'],
                         probability=.8,seeds=[100,101,102,103,104],episodes_per_seed=40)),
        sensitivity_baseline='single seed_1 baseline reused for all three factor centre points',
        assumptions=dict(seed_indices=list(SEEDS),coefficient_seed=1,r0='10, unpublished placeholder; unchanged',
                         checkpoint='exact completed-episode snapshot; current rollout retained; no forced PPO update'))
    write_json(path,data)
    return data


def write_json(path,data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def load_model(checkpoint):
    payload=read_protocol_checkpoint(checkpoint,environment_version=ENVIRONMENT_VERSION)
    env=AirCombatConfig(**payload['env_config']);cfg=ProtocolTrainConfig(**payload['train_config'])
    trainer=ProtocolMAPPOTrainer(cfg)
    restore_protocol_checkpoint(payload,trainer,env_config=env)
    return trainer,env,payload


def final_evaluation(checkpoint,output,*,episodes=None,console=True):
    trainer,env,payload=load_model(checkpoint)
    count=trainer.config.evaluation_episodes if episodes is None else episodes
    if type(count) is not int or count<1:raise ValueError('evaluation episodes must be positive')
    start=10000000+10000*trainer.config.training_seed_index
    if console:print(f'[EVAL] starting independent n={count} | Model {Path(checkpoint).resolve()}',flush=True)
    result=evaluate(trainer,env,list(range(start,start+count)))
    result.update(checkpoint=str(Path(checkpoint).resolve()),completed_training_episodes=payload['training_episode_count'],
                  training_seed=trainer.config.training_seed_index,environment=env.to_dict())
    result.update(training_episode_target=trainer.config.training_episodes,
                  training_budget_kind=budget_kind(trainer.config.training_episodes))
    write_json(output,result)
    if console:print(format_evaluation(result,Path(checkpoint).resolve()),flush=True)
    return result


def run_experiment(name,seed,root=None,*,smoke=False,resume=None,episodes=None,console=True,log_every_updates=5):
    env,cfg=configurations(name,seed,smoke=smoke,episodes=episodes)
    if root is None:root='outputs/xu_iv_v2_2500' if cfg.training_episodes==2500 else 'outputs/xu_iv_v2'
    output=run_directory(root,name,seed)
    if not resume and ((output/'train.jsonl').exists() or (output/'latest.pt').exists()):
        raise FileExistsError('existing training run; use --resume or a fresh root')
    if resume:
        payload=read_protocol_checkpoint(resume,environment_version=ENVIRONMENT_VERSION)
        if AirCombatConfig(**payload['env_config'])!=env or ProtocolTrainConfig(**payload['train_config'])!=cfg:
            raise ValueError('resume experiment configuration mismatch')
    write_json(output/'experiment.json',manifest(name,seed,smoke=smoke,episodes=episodes))
    # train() refuses accidental reuse, and resume verifies exact configurations.
    trainer=train(env,cfg,output,resume=resume,console=console,log_every_updates=log_every_updates)
    final_evaluation(output/'latest.pt',output/'final_evaluation.json',console=console)
    if console:print(f'[DONE] experiment={name} | budget={budget_kind(cfg.training_episodes)} | output={output}',flush=True)
    return trainer


def blue_sensitivity(checkpoints,output,*,kind,smoke=False):
    if kind not in ('probability','aggregation'):raise ValueError('unknown sensitivity kind')
    if len(checkpoints)!=(1 if kind=='probability' else 5):
        raise ValueError('probability needs one fixed checkpoint; aggregation needs five independent checkpoints')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if any(output.glob('*.json')):raise FileExistsError('use a fresh sensitivity directory')
    variants=(1.,.8,.6) if kind=='probability' else ('aggregate-sum','opportunistic-max','threat-aware-maximin')
    results=[];seen=set();models=[]
    for path in checkpoints:
        trainer,env,payload=load_model(path)
        seed=trainer.config.training_seed_index
        if seed in seen:raise ValueError('aggregation checkpoints must have distinct training seeds')
        seen.add(seed)
        if env.scale!='5v5' or env.opponent_aggregation!='aggregate-sum' or env.opponent_best_probability!=.8:
            raise ValueError('sensitivity requires an aggregate-sum p=.8 5v5 training checkpoint')
        if env.assignment_mode!='aph' or not env.allied_penalty_enabled or not trainer.config.ppo_masking:
            raise ValueError('sensitivity requires the main APH-MAPPO method')
        if not smoke and payload['training_episode_count']!=(5000 if kind=='probability' else 4950):
            raise ValueError('formal probability requires completed 5000; aggregation requires exact episode 4950')
        if not smoke:
            expected_env,expected_cfg=configurations('main_5v5',seed)
            if env!=expected_env or trainer.config!=expected_cfg:
                raise ValueError('formal sensitivity checkpoint must use the main Table III configuration')
        models.append((path,trainer,env,payload,seed))
    # Validate the whole bundle before writing any partial evaluation result.
    for path,trainer,env,payload,seed in models:
        for value in variants:
            changed=replace(env,**({'opponent_best_probability':value} if kind=='probability' else {'opponent_aggregation':value}))
            episodes_per_seed=1 if smoke else (200 if kind=='probability' else 40)
            result=evaluate(trainer,changed,list(range(100,105)),episodes_per_seed=episodes_per_seed)
            result.update(kind=kind,variant=value,training_seed=seed,is_smoke=smoke,checkpoint=str(Path(path).resolve()),
                          completed_training_episodes=payload['training_episode_count'],environment=changed.to_dict())
            write_json(output/f'seed_{seed}_{value}.json',result);results.append(result)
    return results


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    plan=sub.add_parser('plan');plan.add_argument('--output',default='configs/xu_iv/plan.json')
    t=sub.add_parser('train');t.add_argument('--experiment',choices=tuple(CATALOG),required=True)
    t.add_argument('--seed',type=int,default=1);t.add_argument('--root',help='default: outputs/xu_iv_v2_2500 for 2500, outputs/xu_iv_v2 for 5000')
    t.add_argument('--episodes',type=int,choices=(2500,5000),help='completed-episode target; default paper budget 5000')
    t.add_argument('--log-every-updates',type=int,default=5);t.add_argument('--quiet',action='store_true')
    t.add_argument('--smoke',action='store_true');t.add_argument('--resume')
    e=sub.add_parser('evaluate');e.add_argument('checkpoint');e.add_argument('--output',required=True)
    e.add_argument('--episodes',type=int)
    b=sub.add_parser('blue');b.add_argument('--kind',choices=('probability','aggregation'),required=True)
    b.add_argument('--checkpoints',nargs='+',required=True);b.add_argument('--output',required=True);b.add_argument('--smoke',action='store_true')
    args=p.parse_args(argv)
    if args.command=='plan':print(f"Exported {len(export_plan(args.output)['training_runs'])} formal single-seed runs (none started)")
    elif args.command=='train':run_experiment(args.experiment,args.seed,args.root,smoke=args.smoke,resume=args.resume,
                                            episodes=args.episodes,console=not args.quiet,log_every_updates=args.log_every_updates)
    elif args.command=='evaluate':final_evaluation(args.checkpoint,args.output,episodes=args.episodes)
    else:blue_sensitivity(args.checkpoints,args.output,kind=args.kind,smoke=args.smoke)


if __name__=='__main__':main()
