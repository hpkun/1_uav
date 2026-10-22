"""Train/evaluate/record the independent 12/60/3 region baseline (CUDA mandatory)."""
import argparse
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    train=sub.add_parser('train')
    train.add_argument('--env-config',required=True); train.add_argument('--train-config',required=True)
    train.add_argument('--total-sampled-steps',type=int,required=True); train.add_argument('--resume')
    train.add_argument('--output',required=True); train.add_argument('--device',default='cuda')
    for name in ('evaluate','record'):
        q=sub.add_parser(name); q.add_argument('--checkpoint',required=True); q.add_argument('--output',required=True)
        q.add_argument('--seed-base',type=int,default=20000000); q.add_argument('--episodes',type=int,default=1 if name=='record' else 5)
        q.add_argument('--red-rule',choices=['cruise','region','region_safe']); q.add_argument('--device',default='cuda')
    args=p.parse_args()
    from algorithm.common.runtime import require_cuda
    from algorithm.common.serialization import write_json
    from env.region_competition import RegionConfig
    from .config import RegionTrainConfig
    require_cuda(args.device)
    if args.command=='train':
        from .runner import train as run_train
        run_train(RegionConfig.load(args.env_config),RegionTrainConfig.load(args.train_config),args.output,args.total_sampled_steps,resume=args.resume,device=args.device)
        return
    if args.episodes<1 or args.command=='record' and args.episodes!=1: p.error('record requires exactly one episode; evaluation count must be positive')
    from .checkpoint import read_checkpoint,restore_checkpoint
    from .trainer import RegionTrainer
    from .evaluation import evaluate,check_seeds
    from tools.region_replay import render_replay
    data=read_checkpoint(args.checkpoint); ec=RegionConfig(**data['env_config']); tc=RegionTrainConfig(**data['train_config'])
    seeds=list(range(args.seed_base,args.seed_base+args.episodes)); check_seeds(seeds,tc,data['vector'])
    out=Path(args.output)
    if out.exists() and any(out.iterdir()): raise FileExistsError('output must be empty')
    out.mkdir(parents=True,exist_ok=True)
    trainer=RegionTrainer(tc,device=args.device); restore_checkpoint(data,trainer,ec)
    report=evaluate(trainer,ec,seeds,trace_path=out/'trajectory.json',red_rule=args.red_rule,
                    provenance=dict(checkpoint=str(Path(args.checkpoint).resolve()),sampled_steps=trainer.sampled_steps,update=trainer.updates,
                                    red_policy=args.red_rule or 'mappo_mean',blue_policy=ec.blue_policy))
    write_json(out/'evaluation.json',report); render_replay(out/'trajectory.json',out/'replay.html')
    print('Red score',report['mean_red_score'],'Blue score',report['mean_blue_score'],'relative',report['mean_relative_score'])
    print(out/'replay.html')


if __name__=='__main__': main()
