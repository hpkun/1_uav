import csv
import json
from pathlib import Path
import pytest
from algorithm.common.resume_logs import check_resume_logs,ResumeLogConflict


def setup_run(root,train_steps=(100,),eval_steps=(100,),*,checkpoint_steps=100):
    payload=dict(sampled_steps=checkpoint_steps,updates=checkpoint_steps//10,best_score=[0.,0.,1.],env_config={},train_config={})
    checkpoint=root/'latest.pt'; checkpoint.write_text(json.dumps(payload))
    if train_steps is not None:
        (root/'training_metrics.jsonl').write_text(''.join(json.dumps(dict(sampled_steps=s,update=s//10))+'\n' for s in train_steps))
    if eval_steps is not None:
        with (root/'evaluation_history.csv').open('w',newline='') as f:
            w=csv.writer(f); w.writerow(['sampled_steps','update','checkpoint_role','success_rate','completion_rate','mean_return'])
            for s in eval_steps: w.writerow([s,s//10,'current',0,0,1])
    (root/'best.pt').write_text(json.dumps(dict(payload,sampled_steps=80,updates=8)))
    return checkpoint,payload


def reader(path): return json.loads(Path(path).read_text())


def snapshot(root): return {p.name:p.read_bytes() for p in root.iterdir()}


@pytest.mark.parametrize('train,eval',[(None,None),((90,),(90,)),((100,),(100,))])
def test_missing_lagging_and_aligned_logs_are_read_only(tmp_path,train,eval):
    checkpoint,payload=setup_run(tmp_path,train,eval)
    before=snapshot(tmp_path)
    warnings=check_resume_logs(tmp_path,checkpoint,payload,reader)
    assert bool(warnings)==(train!=(100,))
    assert snapshot(tmp_path)==before


@pytest.mark.parametrize('train,eval',[((100,100),(100,)),((90,110),(100,)),((100,),(100,110)),((100,),(100,100)),((100,90),(100,))])
def test_duplicate_or_future_logs_refused_without_writing(tmp_path,train,eval):
    checkpoint,payload=setup_run(tmp_path,train,eval)
    before=snapshot(tmp_path)
    with pytest.raises(ResumeLogConflict): check_resume_logs(tmp_path,checkpoint,payload,reader)
    assert snapshot(tmp_path)==before


@pytest.mark.parametrize('name',['latest.pt','best.pt'])
def test_future_checkpoints_never_silently_overwritten(tmp_path,name):
    checkpoint,payload=setup_run(tmp_path)
    checkpoint=tmp_path/'resume_old.pt'; checkpoint.write_text(json.dumps(payload))
    (tmp_path/name).write_text(json.dumps(dict(payload,sampled_steps=120,updates=12)))
    before=snapshot(tmp_path)
    with pytest.raises(ResumeLogConflict,match='newer'): check_resume_logs(tmp_path,checkpoint,payload,reader)
    assert snapshot(tmp_path)==before


def test_newer_source_best_is_not_copied_to_fresh_branch(tmp_path):
    source=tmp_path/'source'; source.mkdir(); output=tmp_path/'branch'
    checkpoint,payload=setup_run(source)
    (source/'best.pt').write_text(json.dumps(dict(payload,sampled_steps=120,updates=12)))
    with pytest.raises(ResumeLogConflict,match='newer'): check_resume_logs(output,checkpoint,payload,reader)
    assert not output.exists()


def test_malformed_tail_and_future_summary_refused(tmp_path):
    checkpoint,payload=setup_run(tmp_path)
    (tmp_path/'run_summary.json').write_text(json.dumps(dict(sampled_steps=120,updates=12)))
    with pytest.raises(ResumeLogConflict,match='summary'): check_resume_logs(tmp_path,checkpoint,payload,reader)
    (tmp_path/'run_summary.json').write_text(json.dumps(dict(sampled_steps=100,updates=10)))
    with (tmp_path/'training_metrics.jsonl').open('a') as f: f.write('interrupted')
    with pytest.raises(ResumeLogConflict,match='malformed'): check_resume_logs(tmp_path,checkpoint,payload,reader)


def test_runner_preflight_happens_before_all_mutations(monkeypatch,tmp_path):
    from env import EnvConfig
    from algorithm.mappo.config import TrainConfig
    import algorithm.mappo.runner as runner
    checkpoint,payload=setup_run(tmp_path,train_steps=(110,))
    ec,tc=EnvConfig(),TrainConfig()
    payload.update(env_config=ec.to_dict(),train_config=tc.to_dict())
    monkeypatch.setattr(runner,'require_cuda',lambda *_:None)
    monkeypatch.setattr(runner,'read_checkpoint',lambda _:payload)
    def unexpected(*a,**k): raise AssertionError('trainer must not start before resume preflight')
    monkeypatch.setattr(runner,'MAPPOTrainer',unexpected)
    before=snapshot(tmp_path)
    with pytest.raises(ResumeLogConflict): runner.train(ec,tc,tmp_path,120,resume=checkpoint)
    assert snapshot(tmp_path)==before


def test_best_source_mismatch_refused(tmp_path):
    checkpoint,payload=setup_run(tmp_path)
    (tmp_path/'best.pt').write_text(json.dumps(dict(payload,sampled_steps=80,updates=8,best_score=[1,1,2])))
    with pytest.raises(ResumeLogConflict,match='provenance'): check_resume_logs(tmp_path,checkpoint,payload,reader)


@pytest.mark.parametrize('existing_steps',[(90,),None])
def test_resumed_runner_appends_only_new_steps_and_keeps_gap(monkeypatch,tmp_path,capsys,existing_steps):
    """Exercise the actual writer/scheduler with metadata fakes, without any rollout simulation."""
    from env import EnvConfig
    from algorithm.mappo.config import TrainConfig
    import algorithm.mappo.runner as runner
    checkpoint,payload=setup_run(tmp_path,existing_steps,existing_steps)
    ec,tc=EnvConfig(),TrainConfig(num_envs=1,horizon=10,evaluation_episodes=1)
    payload.update(env_config=ec.to_dict(),train_config=tc.to_dict(),best_score=[0.,0.,0.])
    checkpoint.write_text(json.dumps(payload))
    (tmp_path/'best.pt').write_text(json.dumps(dict(payload,sampled_steps=80,updates=8)))
    prefix=(tmp_path/'training_metrics.jsonl').read_bytes() if existing_steps else b''
    class Trainer:
        def __init__(self,config,device):
            self.config=config; self.device=device
            self.sampled_steps=self.updates=self.training_episode_count=0
        def collect(self,vector,budget):
            self.sampled_steps+=budget
            self.rollout_metrics={'sampled_steps':budget}
            return None,[]
        def update(self,_):
            self.updates+=1
            return dict(policy_loss=0.,actor_total_loss=0.,actor_loss=0.,critic_loss=0.,update=self.updates)
    class Vector:
        def __init__(self,*a,**k): pass
        def __enter__(self): return self
        def __exit__(self,*a): pass
    def restore(p,trainer,*a):
        trainer.sampled_steps=p['sampled_steps']; trainer.updates=p['updates']
    def save(path,trainer,*a):
        Path(path).write_text(json.dumps(dict(payload,sampled_steps=trainer.sampled_steps,updates=trainer.updates)))
    def typed_reader(path):
        # Real torch checkpoints preserve tuples; JSON metadata fakes do not.
        result=reader(path)
        result.update(env_config=ec.to_dict(),train_config=tc.to_dict())
        return result
    evaluation=dict(success_rate=0.,completion_rate=0.,mean_reward=0.,mean_completed_waypoints=0.,mean_episode_length=1.,episodes=[{'success':False}])
    monkeypatch.setattr(runner,'require_cuda',lambda *_:None)
    monkeypatch.setattr(runner.torch.cuda,'get_device_name',lambda:'unit metadata only')
    monkeypatch.setattr(runner,'MAPPOTrainer',Trainer)
    monkeypatch.setattr(runner,'ParallelVectorEnv',Vector)
    monkeypatch.setattr(runner,'read_checkpoint',typed_reader)
    monkeypatch.setattr(runner,'restore_checkpoint',restore)
    monkeypatch.setattr(runner,'save_checkpoint',save)
    monkeypatch.setattr(runner,'evaluate',lambda *a,**k:evaluation)
    _,summary=runner.train(ec,tc,tmp_path,110,resume=checkpoint)
    path=tmp_path/'training_metrics.jsonl'
    assert path.read_bytes().startswith(prefix)
    steps=[r['sampled_steps'] for r in map(json.loads,path.read_text().splitlines())]
    assert steps==([90,110] if existing_steps else [110])
    with (tmp_path/'evaluation_history.csv').open(newline='') as f:
        assert [int(r['sampled_steps']) for r in csv.DictReader(f)]==steps
    assert summary['sampled_steps']==110
    assert '[RESUME]' in capsys.readouterr().out
