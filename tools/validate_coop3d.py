"""Bounded acceptance: two small CUDA smoke invocations including resume."""


def main():
    import argparse
    from pathlib import Path
    import torch
    from env import EnvConfig
    from env.factory import make_environment
    from algorithm.mappo.config import TrainConfig
    from algorithm.mappo.trainer import MAPPOTrainer
    from algorithm.mappo.runner import train
    from algorithm.common.checkpoint import read_checkpoint,restore_checkpoint
    from algorithm.common.runtime import require_cuda
    from algorithm.common.serialization import write_json
    from algorithm.mappo.evaluation import evaluate
    from tools.render_replay import render_replay
    require_cuda()
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',required=True); a=p.parse_args()
    out=Path(a.output)
    if out.exists() and any(out.iterdir()): raise FileExistsError('smoke output must be empty')
    ec=EnvConfig(max_steps=8,failure_events={4:[1]})
    tc=TrainConfig(num_envs=2,horizon=8,ppo_epochs=1,minibatch_size=32,evaluation_episodes=1,
        evaluation_interval_sampled_steps=16,checkpoint_interval_sampled_steps=16,console_interval_sampled_steps=16)
    initial=MAPPOTrainer(tc)
    before=[p.detach().clone() for p in initial.actor.parameters()]
    trainer,first=train(ec,tc,out,17)
    assert first['sampled_steps']==17 and trainer.updates==2
    assert any(not torch.equal(x,y) for x,y in zip(before,trainer.actor.parameters()))
    payload=read_checkpoint(out/'latest.pt'); clone=MAPPOTrainer(tc); restore_checkpoint(payload,clone,ec)
    for x,y in zip(trainer.actor.parameters(),clone.actor.parameters()): torch.testing.assert_close(x,y,rtol=0,atol=0)
    trainer,second=train(ec,tc,out,25,resume=out/'latest.pt')
    assert trainer.sampled_steps==25 and trainer.updates==3
    report=evaluate(trainer,ec,[20000000],out/'latest_trajectory.json',provenance=dict(checkpoint_role='latest',checkpoint=str((out/'latest.pt').resolve()),sampled_steps=25,update=3))
    render_replay(out/'latest_trajectory.json',out/'latest_replay.html')
    write_json(out/'smoke_results.json',dict(first=first,resumed=second,reloaded_parameters_exact=True,
        gradients_changed_actor=True,holdout=report,cuda=torch.cuda.get_device_name(),
        smoke_invocations=2,total_joint_transitions=25,ppo_updates=3))
    print('PASS: CUDA, exact 17/25 joint budgets, 2 workers, 3 short updates, checkpoint/resume/evaluation/replay',flush=True)


if __name__=='__main__': main()
