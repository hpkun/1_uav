"""Evaluate an explicitly identified checkpoint without confusing best and latest."""


def main():
    import argparse
    from pathlib import Path
    from env.config import EnvConfig
    from algorithm.mappo.config import TrainConfig
    from algorithm.mappo.trainer import MAPPOTrainer
    from algorithm.mappo.evaluation import evaluate,ensure_independent
    from algorithm.common.checkpoint import read_checkpoint,restore_checkpoint
    from algorithm.common.runtime import require_cuda
    from algorithm.common.serialization import write_json
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True); p.add_argument('--device',default='cuda')
    p.add_argument('--seed-base',type=int,default=20000000); p.add_argument('--episodes',type=int,default=5)
    p.add_argument('--output',required=True)
    a=p.parse_args(); require_cuda(a.device)
    if a.episodes<1: raise ValueError('episodes must be positive')
    data=read_checkpoint(a.checkpoint)
    ec,tc=EnvConfig(**data['env_config']),TrainConfig(**data['train_config'])
    seeds=range(a.seed_base,a.seed_base+a.episodes)
    if data['vector']: ensure_independent(seeds,tc.seed,tc.num_envs,data['vector']['episodes'])
    trainer=MAPPOTrainer(tc,device=a.device); restore_checkpoint(data,trainer,ec)
    out=Path(a.output)
    if out.exists() and any(out.iterdir()): raise FileExistsError('evaluation output must be empty')
    out.mkdir(parents=True,exist_ok=True)
    provenance=dict(checkpoint=str(Path(a.checkpoint).resolve()),checkpoint_role=Path(a.checkpoint).stem,
                    sampled_steps=trainer.sampled_steps,update=trainer.updates)
    report=evaluate(trainer,ec,seeds,out/'trajectory.json',provenance=provenance)
    write_json(out/'evaluation.json',dict(provenance=provenance,**report))
    from tools.render_replay import render_replay
    render_replay(out/'trajectory.json',out/'replay.html')
    print(report)


if __name__=='__main__': main()
