"""Xu air combat training/evaluation CLI; CUDA required, no inspection adapters.

PowerShell: python -m algorithm.aircombat train --scale 3v3 --episodes 5000
"""
import argparse
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from env.aircombat import AirCombatConfig, AirCombatEnv
from env.aircombat.config import ENVIRONMENT_VERSION
from algorithm.mappo.config import ProtocolTrainConfig
from algorithm.mappo.trainer import ProtocolMAPPOTrainer
from algorithm.common.runtime import require_cuda
from algorithm.common.aircombat_vector import AirCombatVectorEnv, CombatEpisodeAccumulator, combat_metrics
from algorithm.common.checkpoint import save_protocol_checkpoint,read_protocol_checkpoint,restore_protocol_checkpoint
from algorithm.common.aircombat_progress import TrainingConsole
from algorithm.common.checkpoint_policy import checkpoint_policy,checkpoint_targets


def evaluate(trainer, env_config, seeds, *, scenario=None, episodes_per_seed=1):
    require_cuda(str(trainer.device))
    seeds = tuple(seeds)
    if not seeds or any(type(s) is not int or s<0 for s in seeds):
        raise ValueError('evaluation requires nonnegative integer seeds')
    if set(seeds)&set(trainer.config.worker_seeds()):
        raise ValueError('evaluation seeds overlap training workers')
    if type(episodes_per_seed) is not int or episodes_per_seed<1:
        raise ValueError('episodes_per_seed must be positive')
    summaries = []
    for seed in seeds:
        env = AirCombatEnv(env_config)
        for episode_index in range(episodes_per_seed):
            obs,info = env.reset(seed if episode_index==0 else None,options={'scenario':scenario} if scenario else None)
            tracker = CombatEpisodeAccumulator();tracker.reset(info)
            while True:
                actions,_ = trainer.sample_actions(obs,info['active_mask'],deterministic=True)
                obs,reward,terminated,truncated,info = env.step(actions.cpu().numpy())
                tracker.add(info,reward)
                if terminated or truncated:
                    summary = tracker.summary(info)
                    summary.update(seed=seed,episode_index=episode_index)
                    summaries.append(summary)
                    break
    return dict(**combat_metrics(summaries),episodes=summaries,environment_version=ENVIRONMENT_VERSION,
        integrator=env_config.integrator,evaluation_protocol=dict(
        seeds=list(seeds),episodes_per_seed=episodes_per_seed,policy='deterministic Beta mean',
        reset='seed once per group; subsequent episodes continue independent Generator stream'))


def train(env_config, train_config, output, *, resume=None, console=True, log_every_updates=5):
    started_at=time.monotonic()
    if type(log_every_updates) is not int or log_every_updates<1:
        raise ValueError('log_every_updates must be a positive integer')
    require_cuda('cuda')
    if env_config.opponent_aggregation!='aggregate-sum' or env_config.opponent_best_probability!=.8:
        raise ValueError('Blue variants are evaluation-only; training requires aggregate-sum p=.8')
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    random.seed(train_config.seed)
    np.random.seed(train_config.seed)
    trainer = ProtocolMAPPOTrainer(train_config)
    trainer.checkpoint_policy = checkpoint_policy(train_config)
    log = output/'train.jsonl'
    history=[]
    with AirCombatVectorEnv(env_config,train_config.num_envs,train_config.training_seed_index) as vector:
        if resume:
            payload = read_protocol_checkpoint(resume,environment_version=ENVIRONMENT_VERSION)
            restore_protocol_checkpoint(payload,trainer,env_config=env_config,vector=vector)
            # An older snapshot must not overwrite models from its future.
            for path in output.glob('episodes_*.pt'):
                if int(path.stem.removeprefix('episodes_'))>trainer.training_episode_count:
                    raise FileExistsError('newer checkpoints exist; resume in a fresh directory')
            if (output/'latest.pt').exists():
                latest=read_protocol_checkpoint(output/'latest.pt',environment_version=ENVIRONMENT_VERSION)
                if latest['sampled_steps']>trainer.sampled_steps or latest['updates']>trainer.updates:
                    raise FileExistsError('newer latest.pt exists; resume in a fresh directory')
            # Remove stale rows beyond this resume boundary before appending.
            if log.exists():
                records = [json.loads(line) for line in log.read_text(encoding='utf-8').splitlines() if line.strip()]
                records = [r for r in records if r['updates']<=trainer.updates and r['sampled_steps']<=trainer.sampled_steps]
                log.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
                history=[episode for row in records for episode in row.get('episodes',[])]
            index=output/'checkpoints.jsonl'
            if index.exists():
                rows=[json.loads(x) for x in index.read_text(encoding='utf-8').splitlines() if x.strip()]
                rows=[r for r in rows if r['completed_episodes']<=trainer.training_episode_count and r['updates']<=trainer.updates]
                index.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
        elif log.exists() or any(output.glob('*.pt')):
            raise ValueError('output already contains training state; use --resume or a fresh directory')
        manifest = dict(environment_version=ENVIRONMENT_VERSION,environment=env_config.to_dict(),training=train_config.to_dict(),
                        reproduction_assumptions=env_config.reproduction_assumptions(),
                        training_assumptions=train_config.reproduction_assumptions(),
                        checkpoint_policy=trainer.checkpoint_policy,
                        checkpoint_targets=checkpoint_targets(train_config,trainer.checkpoint_policy),
                        cuda=torch.cuda.get_device_name(0))
        (output/'config.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        display=TrainingConsole(train_config.training_episodes,enabled=console,every=log_every_updates,
                                start_steps=trainer.sampled_steps,history=history,started_at=started_at)
        display.start(trainer,vector,env_config,output,resume=resume)
        if console:display.emit(f"[START] GPU name: {torch.cuda.get_device_name(0)}")
        targets=set(checkpoint_targets(train_config,trainer.checkpoint_policy))
        def target_checkpoint(t,v):
            if t.training_episode_count not in targets:
                return
            path=output/f'episodes_{t.training_episode_count:06d}.pt'
            if path.exists():
                raise FileExistsError(f'checkpoint already exists; resume in a fresh directory: {path}')
            save_protocol_checkpoint(path,t,env_config=env_config,vector=v)
            row=dict(target_episode=t.training_episode_count,completed_episodes=t.training_episode_count,
                     updates=t.updates,sampled_steps=t.sampled_steps,phase='collecting_rollout',path=path.name)
            with (output/'checkpoints.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(row)+'\n')
            display.checkpoint(t,path)
        while trainer.training_episode_count<train_config.training_episodes or getattr(trainer,'pending_collection',None) is not None:
            remaining = train_config.training_episodes-trainer.training_episode_count
            try:
                rollout,episodes = trainer.collect(vector,episode_limit=max(1,remaining),checkpoint_callback=target_checkpoint)
            except (FloatingPointError,RuntimeError,TimeoutError) as exc:
                # Never update PPO or replace a checkpoint after collection fails.
                with (output/'simulation_faults.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(dict(environment_version=ENVIRONMENT_VERSION,
                        integrator=env_config.integrator,completed_episodes=trainer.training_episode_count,
                        updates=trainer.updates,sampled_steps=trainer.sampled_steps,diagnostic=str(exc)),ensure_ascii=False)+'\n')
                display.emit(f'[FAULT] simulation collection stopped | PPO updates {trainer.updates} | details: simulation_faults.jsonl')
                raise
            update = trainer.update(rollout)
            record = dict(completed_episodes=trainer.training_episode_count,sampled_steps=trainer.sampled_steps,
                          updates=trainer.updates,episode_metrics=combat_metrics(episodes),
                          rollout=trainer.rollout_metrics,ppo=update,episodes=episodes)
            with log.open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
            display.update(record)
        save_protocol_checkpoint(output/'latest.pt',trainer,env_config=env_config,vector=vector)
        display.checkpoint(trainer,output/'latest.pt')
        display.done(trainer)
    return trainer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    t = sub.add_parser('train')
    t.add_argument('--scale',choices=('3v3','5v5','3v5'),default='3v3')
    t.add_argument('--episodes',type=int,default=5000)
    t.add_argument('--seed-index',type=int,default=1)
    t.add_argument('--num-envs',type=int,default=10)
    t.add_argument('--horizon',type=int,default=1500)
    t.add_argument('--ppo-epochs',type=int,default=10)
    t.add_argument('--minibatch-size',type=int,default=128)
    t.add_argument('--checkpoint-interval',type=int,default=50)
    t.add_argument('--r0',type=float,default=10.,help='PAPER UNREPORTED: experimental placeholder')
    t.add_argument('--output',default='outputs/xu_aircombat')
    t.add_argument('--resume')
    e = sub.add_parser('evaluate')
    e.add_argument('checkpoint')
    e.add_argument('--episodes',type=int,default=200)
    e.add_argument('--seed',type=int,default=10000000)
    e.add_argument('--scenario')
    e.add_argument('--output',default='outputs/xu_evaluation.json')
    args = parser.parse_args(argv)
    require_cuda('cuda')
    if args.command=='train':
        config = ProtocolTrainConfig(training_episodes=args.episodes,training_seed_index=args.seed_index,
            seed=1000*args.seed_index,num_envs=args.num_envs,horizon=args.horizon,ppo_epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,checkpoint_interval_episodes=args.checkpoint_interval)
        train(AirCombatConfig(scale=args.scale,shootdown_r0=args.r0),config,args.output,resume=args.resume)
    else:
        if args.episodes<1:
            parser.error('--episodes must be positive')
        payload = read_protocol_checkpoint(args.checkpoint,environment_version=ENVIRONMENT_VERSION)
        config = ProtocolTrainConfig(**payload['train_config'])
        env_config = AirCombatConfig(**payload['env_config'])
        trainer = ProtocolMAPPOTrainer(config)
        restore_protocol_checkpoint(payload,trainer,env_config=env_config)
        result = evaluate(trainer,env_config,list(range(args.seed,args.seed+args.episodes)),scenario=args.scenario)
        path = Path(args.output)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        print(json.dumps({k:v for k,v in result.items() if k!='episodes'},ensure_ascii=False))


if __name__=='__main__':
    main()
