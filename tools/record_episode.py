"""Record a specified checkpoint or deterministic navigation, on an explicit seed."""


def main():
    import argparse
    from pathlib import Path
    from env.config import EnvConfig
    from algorithm.common.runtime import require_cuda
    from algorithm.common.serialization import write_json
    from .render_replay import render_replay
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--checkpoint'); group.add_argument('--policy',choices=['navigation','trim'])
    p.add_argument('--env-config'); p.add_argument('--seed',type=int,default=20000000)
    p.add_argument('--device',default='cuda'); p.add_argument('--output',required=True)
    p.add_argument('--policy-diagnostics',action='store_true',help='record original policy action and Beta parameters in the new checkpoint trace')
    a=p.parse_args(); require_cuda(a.device)
    out=Path(a.output)
    if out.exists() and any(out.iterdir()): raise FileExistsError('recording output must be empty')
    out.mkdir(parents=True,exist_ok=True)
    if a.checkpoint:
        from algorithm.common.checkpoint import read_checkpoint,restore_checkpoint
        from algorithm.mappo.trainer import MAPPOTrainer
        from algorithm.mappo.config import TrainConfig
        from algorithm.mappo.evaluation import evaluate,ensure_independent
        saved=read_checkpoint(a.checkpoint); ec=EnvConfig(**saved['env_config']); tc=TrainConfig(**saved['train_config'])
        if a.env_config and EnvConfig.load(a.env_config).to_dict()!=ec.to_dict(): raise ValueError('recording environment does not match checkpoint')
        if saved['vector']: ensure_independent([a.seed],tc.seed,tc.num_envs,saved['vector']['episodes'])
        trainer=MAPPOTrainer(tc,device=a.device); restore_checkpoint(saved,trainer,ec)
        result=evaluate(trainer,ec,[a.seed],out/'trajectory.json',provenance=dict(checkpoint=str(Path(a.checkpoint).resolve()),checkpoint_role=Path(a.checkpoint).stem,update=trainer.updates,sampled_steps=trainer.sampled_steps),policy_diagnostics=a.policy_diagnostics)
    else:
        from .audit_navigation import run_episode
        ec=EnvConfig.load(a.env_config) if a.env_config else EnvConfig()
        result=run_episode(ec,a.seed,a.policy,trace_path=out/'trajectory.json')
    write_json(out/'recording_report.json',result)
    render_replay(out/'trajectory.json',out/'replay.html')
    print(out/'replay.html')


if __name__=='__main__': main()
