"""Disk policy counts plus real CUDA collection, evaluation and exact resume."""
import json

import pytest

import algorithm.aircombat as runner
from algorithm.common.checkpoint import read_protocol_checkpoint
from algorithm.common.checkpoint_policy import checkpoint_policy,checkpoint_targets
from algorithm.experiments import manifest,load_model
from algorithm.mappo.config import ProtocolTrainConfig
from env.aircombat import AirCombatConfig
from env.aircombat.config import ENVIRONMENT_VERSION
from tests.test_aircombat_training import tensor_equal


@pytest.mark.parametrize('budget,count',[(2500,10),(5000,21)])
def test_formal_disk_files_index_and_plan(tmp_path,monkeypatch,budget,count):
    # Exercise the real runner callback at every original 50-episode boundary.
    # No simulation or PPO budget is spent on counting 5000 episode filenames.
    class Trainer:
        def __init__(self,c):
            self.training_episode_count=self.updates=self.sampled_steps=0
            self.pending_collection=None
        def collect(self,vector,**kwargs):
            self.training_episode_count+=50
            self.sampled_steps+=50
            kwargs['checkpoint_callback'](self,vector)
            self.rollout_metrics={}
            return None,[]
        def update(self,rollout):
            self.updates+=1
            return {}
    class Vector:
        def __init__(self,*args):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
    class Display:
        def __init__(self,*args,**kwargs):pass
        def __getattr__(self,name):return lambda *args,**kwargs:None
    monkeypatch.setattr(runner,'ProtocolMAPPOTrainer',Trainer)
    monkeypatch.setattr(runner,'AirCombatVectorEnv',Vector)
    monkeypatch.setattr(runner,'TrainingConsole',Display)
    monkeypatch.setattr(runner,'save_protocol_checkpoint',lambda path,*args,**kwargs:path.write_bytes(b'full serializer tested separately'))
    runner.train(AirCombatConfig(),ProtocolTrainConfig(training_episodes=budget),tmp_path,console=False)
    expected=manifest('main_3v3',episodes=budget)['checkpoint_targets']
    rows=[json.loads(line) for line in (tmp_path/'checkpoints.jsonl').read_text().splitlines()]
    assert [row['completed_episodes'] for row in rows]==expected
    assert len(rows)==count
    assert {p.name for p in tmp_path.glob('*.pt')}=={'latest.pt',*(f'episodes_{n:06d}.pt' for n in expected)}
    assert json.loads((tmp_path/'config.json').read_text(encoding='utf-8'))['checkpoint_targets']==expected
    assert (4950 in expected)==(budget==5000)


@pytest.mark.parametrize('interval',[1,7,100,250])
def test_custom_intervals_unchanged(interval):
    c=ProtocolTrainConfig(checkpoint_interval_episodes=interval)
    assert checkpoint_targets(c)==list(range(interval,5001,interval))
    assert manifest('main_5v5',smoke=True)['checkpoint_targets']==[1,2]


def test_sparse_vs_legacy_full_state_resume_and_independent_evaluation(tmp_path,monkeypatch):
    ec=AirCombatConfig(max_steps=1)
    c=ProtocolTrainConfig(num_envs=2,horizon=4,ppo_epochs=1,minibatch_size=64,training_episodes=250)
    sparse=runner.train(ec,c,tmp_path/'sparse',console=False)
    sparse_payload=read_protocol_checkpoint(tmp_path/'sparse'/'latest.pt',environment_version=ENVIRONMENT_VERSION)
    intermediate=read_protocol_checkpoint(tmp_path/'sparse'/'episodes_000250.pt',environment_version=ENVIRONMENT_VERSION)
    assert intermediate['phase']=='collecting_rollout' and intermediate['pending_collection'] is not None
    assert all(key in intermediate for key in ('actor','critic','actor_optimizer','critic_optimizer',
               'torch_rng','cuda_rng','numpy_rng','python_rng','vector'))
    original_policy=checkpoint_policy
    monkeypatch.setattr(runner,'checkpoint_policy',lambda cfg,**kwargs:original_policy(cfg,legacy=True))
    # Simulate checkpoints made before this change (no policy field).
    original_save=runner.save_protocol_checkpoint
    def old_save(path,t,**kwargs):
        policy=t.checkpoint_policy
        del t.checkpoint_policy
        try:original_save(path,t,**kwargs)
        finally:t.checkpoint_policy=policy
    monkeypatch.setattr(runner,'save_protocol_checkpoint',old_save)
    dense=runner.train(ec,c,tmp_path/'legacy',console=False)
    assert len(list((tmp_path/'sparse').glob('episodes_*.pt')))==1
    assert len(list((tmp_path/'legacy').glob('episodes_*.pt')))==5
    tensor_equal(sparse.actor.state_dict(),dense.actor.state_dict())
    tensor_equal(sparse.critic.state_dict(),dense.critic.state_dict())
    dense_payload=read_protocol_checkpoint(tmp_path/'legacy'/'latest.pt',environment_version=ENVIRONMENT_VERSION)
    for key in ('actor_optimizer','critic_optimizer','torch_rng','cuda_rng','numpy_rng','python_rng','vector','pending_collection','updates','sampled_steps'):
        tensor_equal(sparse_payload[key],dense_payload[key])
    monkeypatch.setattr(runner,'save_protocol_checkpoint',original_save)
    monkeypatch.setattr(runner,'checkpoint_policy',original_policy)
    for source,episode in [('sparse',250),('legacy',50)]:
        before={p.name:p.read_bytes() for p in (tmp_path/source).glob('*.pt')}
        resumed=runner.train(ec,c,tmp_path/f'resume_{source}',resume=tmp_path/source/f'episodes_{episode:06d}.pt',console=False)
        tensor_equal(sparse.actor.state_dict(),resumed.actor.state_dict())
        tensor_equal(sparse.critic.state_dict(),resumed.critic.state_dict())
        assert sparse.updates==resumed.updates and sparse.sampled_steps==resumed.sampled_steps
        assert before=={p.name:p.read_bytes() for p in (tmp_path/source).glob('*.pt')}
        policy=json.loads((tmp_path/f'resume_{source}'/'config.json').read_text(encoding='utf-8'))['checkpoint_policy']
        assert policy['interval_episodes']==(250 if source=='sparse' else 50)
    before={p.name:p.read_bytes() for p in (tmp_path/'legacy').iterdir() if p.is_file()}
    with pytest.raises(FileExistsError,match='newer'):
        runner.train(ec,c,tmp_path/'legacy',resume=tmp_path/'legacy'/'episodes_000050.pt',console=False)
    assert before=={p.name:p.read_bytes() for p in (tmp_path/'legacy').iterdir() if p.is_file()}
    model,env,_=load_model(tmp_path/'sparse'/'latest.pt')
    assert runner.evaluate(model,env,[10000000])['completed_episodes']==1


def test_legacy_experiment_manifest_is_preserved(tmp_path,monkeypatch):
    import algorithm.experiments as experiments
    env,cfg=experiments.configurations('main_3v3',smoke=True)
    output=experiments.run_directory(tmp_path,'main_3v3',1)
    output.mkdir(parents=True)
    old_plan=manifest('main_3v3',smoke=True)
    del old_plan['checkpoint_policy']
    experiments.write_json(output/'experiment.json',old_plan)
    before=(output/'experiment.json').read_bytes()
    monkeypatch.setattr(experiments,'read_protocol_checkpoint',lambda *args,**kwargs:
                        dict(env_config=env.to_dict(),train_config=cfg.to_dict()))
    monkeypatch.setattr(experiments,'train',lambda *args,**kwargs:None)
    monkeypatch.setattr(experiments,'final_evaluation',lambda *args,**kwargs:None)
    experiments.run_experiment('main_3v3',1,tmp_path,smoke=True,resume=output/'old.pt',console=False)
    assert before==(output/'experiment.json').read_bytes()
