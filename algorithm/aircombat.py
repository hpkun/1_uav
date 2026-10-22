"""Xu air combat training/evaluation CLI; CUDA required, no inspection adapters.

PowerShell: python -m algorithm.aircombat train --scale 3v3 --episodes 5000
"""
import argparse
import json
from pathlib import Path
import random
import numpy as np
import torch
from env.aircombat import AirCombatConfig, AirCombatEnv
from env.aircombat.config import ENVIRONMENT_VERSION
from algorithm.mappo.config import ProtocolTrainConfig
from algorithm.mappo.trainer import ProtocolMAPPOTrainer
from algorithm.common.runtime import require_cuda
from algorithm.common.aircombat_vector import AirCombatVectorEnv, CombatEpisodeAccumulator, combat_metrics
from algorithm.common.checkpoint import save_protocol_checkpoint,read_protocol_checkpoint,restore_protocol_checkpoint


def evaluate(trainer, env_config, seeds, *, scenario=None):
    require_cuda(str(trainer.device))
    seeds = tuple(seeds)
    if not seeds or any(type(s) is not int or s<0 for s in seeds):
        raise ValueError('evaluation requires nonnegative integer seeds')
    if set(seeds)&set(trainer.config.worker_seeds()):
        raise ValueError('evaluation seeds overlap training workers')
    summaries = []
    for seed in seeds:
        env = AirCombatEnv(env_config)
        obs,info = env.reset(seed,options={'scenario':scenario} if scenario else None)
        tracker = CombatEpisodeAccumulator()
        tracker.reset(info)
        while True:
            actions,_ = trainer.sample_actions(obs,info['active_mask'],deterministic=True)
            obs,reward,terminated,truncated,info = env.step(actions.cpu().numpy())
            tracker.add(info,reward)
            if terminated or truncated:
                summary = tracker.summary(info)
                summary['seed'] = seed
                summaries.append(summary)
                break
    return dict(**combat_metrics(summaries),episodes=summaries)


def train(env_config, train_config, output, *, resume=None):
    require_cuda('cuda')
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    random.seed(train_config.seed)
    np.random.seed(train_config.seed)
    trainer = ProtocolMAPPOTrainer(train_config)
    log = output/'train.jsonl'
    with AirCombatVectorEnv(env_config,train_config.num_envs,train_config.training_seed_index) as vector:
        if resume:
            payload = read_protocol_checkpoint(resume,environment_version=ENVIRONMENT_VERSION)
            restore_protocol_checkpoint(payload,trainer,env_config=env_config,vector=vector)
            # Remove stale rows beyond this resume boundary before appending.
            if log.exists():
                records = [json.loads(line) for line in log.read_text(encoding='utf-8').splitlines() if line.strip()]
                records = [r for r in records if r['updates']<=trainer.updates and r['sampled_steps']<=trainer.sampled_steps]
                log.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
        elif log.exists() or (output/'latest.pt').exists():
            raise ValueError('output already contains training state; use --resume or a fresh directory')
        manifest = dict(environment=env_config.to_dict(),training=train_config.to_dict(),
                        reproduction_assumptions=env_config.reproduction_assumptions(),
                        training_assumptions=train_config.reproduction_assumptions(),
                        cuda=torch.cuda.get_device_name(0))
        (output/'config.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        interval = train_config.checkpoint_interval_episodes
        last_bucket = trainer.training_episode_count//interval
        while trainer.training_episode_count<train_config.training_episodes:
            remaining = train_config.training_episodes-trainer.training_episode_count
            rollout,episodes = trainer.collect(vector,episode_limit=remaining)
            update = trainer.update(rollout)
            record = dict(completed_episodes=trainer.training_episode_count,sampled_steps=trainer.sampled_steps,
                          updates=trainer.updates,episode_metrics=combat_metrics(episodes),
                          rollout=trainer.rollout_metrics,ppo=update)
            with log.open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
            print(json.dumps({k:v for k,v in record.items() if k!='ppo'},ensure_ascii=False),flush=True)
            bucket = trainer.training_episode_count//interval
            if bucket>last_bucket:
                save_protocol_checkpoint(output/f'episodes_{trainer.training_episode_count:06d}.pt',trainer,
                                         env_config=env_config,vector=vector)
                last_bucket = bucket
        save_protocol_checkpoint(output/'latest.pt',trainer,env_config=env_config,vector=vector)
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
