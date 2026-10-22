from dataclasses import fields
import copy
import json
import random
import numpy as np
import pytest
import torch
from torch import nn
from env.region_competition import RegionConfig
from algorithm.region_mappo.config import RegionTrainConfig
from algorithm.region_mappo.trainer import RegionTrainer
from algorithm.region_mappo.vector_env import RegionVectorEnv
from algorithm.region_mappo.checkpoint import save_checkpoint, read_checkpoint, restore_checkpoint
from algorithm.region_mappo.evaluation import evaluate, check_seeds
from algorithm.mappo.trainer import Rollout
from algorithm.common.serialization import serializable
from tools.audit_region_trajectory import audit
from tools.region_replay import render_replay


def canonical(value):
    return json.dumps(serializable(value),sort_keys=True,allow_nan=False,default=lambda obj:serializable(vars(obj)))


def test_published_config_and_independent_protocol():
    cfg=RegionTrainConfig.load('configs/mappo_region.yaml')
    assert (cfg.horizon,cfg.num_envs,cfg.ppo_epochs,cfg.minibatch_size)==(1500,10,10,128)
    assert (cfg.gamma,cfg.gae_lambda,cfg.actor_lr,cfg.critic_lr,cfg.clip,cfg.entropy_coef)==(.954,.95,.0003,.0003,.2,.001)
    trainer=RegionTrainer(cfg)
    assert trainer.actor.network[0].in_features==12
    assert trainer.critic.network[0].in_features==60
    assert trainer.critic.network[-1].out_features==5
    obs=torch.rand((9,12),device=trainer.device)
    actions,logp=trainer.actor.act(obs)
    dist=trainer.actor.distribution(obs)
    assert ((actions>0)&(actions<1)).all()
    torch.testing.assert_close(logp,dist.log_prob(actions).sum(-1))
    torch.testing.assert_close(trainer.actor.evaluate(obs,actions)[0],logp)


@pytest.mark.parametrize('true_terminal',[False,True])
def test_finite_horizon_gae_hand_answers_no_reset_leakage(true_terminal):
    trainer=RegionTrainer(RegionTrainConfig(gamma=.9,gae_lambda=.8))
    d=trainer.device
    values=torch.tensor([[[1.,4.,1.]],[[2.,float('nan'),2.]],[[10.,10.,10.]]],device=d)
    next_values=torch.tensor([[[2.,99.,2.]],[[99.,99.,99.]],[[20.,20.,20.]]],device=d)
    rewards=torch.tensor([[[1.,5.,1.]],[[3.,float('nan'),3.]],[[100.,100.,100.]]],device=d)
    masks=torch.ones_like(values); masks[1,0,1]=0
    nxt=torch.ones_like(values); nxt[:2,0,1]=0
    term=torch.tensor([[False],[true_terminal],[False]],device=d)
    trunc=torch.tensor([[False],[not true_terminal],[False]],device=d)
    zeros=torch.zeros((3,1,3,3),device=d)
    batch=Rollout(torch.zeros((3,1,3,12),device=d),torch.zeros((3,1,60),device=d),zeros,
                  torch.zeros_like(values),values,next_values,rewards,masks,nxt,term,trunc)
    advantages,returns=trainer.estimate_advantages(batch)
    np.testing.assert_allclose(advantages[:,0,0].cpu(),[2.52,1,108],atol=1e-6)
    np.testing.assert_allclose(returns[:,0,0].cpu(),[3.52,3,118],atol=1e-6)
    assert advantages[0,0,1].item()==1 and returns[0,0,1].item()==5
    assert advantages[1,0,1].item()==0 and returns[1,0,1].item()==0


class ClockValue(nn.Module):
    def forward(self,states):
        return states[:,11,None].expand(-1,5)*10


def test_real_1500_rollout_crosses_six_250_step_episodes():
    trainer=RegionTrainer(RegionTrainConfig(num_envs=1,horizon=1500,ppo_epochs=1))
    trainer.critic=ClockValue().to(trainer.device)
    with RegionVectorEnv(RegionConfig(),1,123) as vector:
        batch,episodes=trainer.collect(vector)
        assert trainer.sampled_steps==1500 and trainer.rollout_metrics['valid_uav_samples']==4500
        assert len(episodes)==6 and vector.episodes.tolist()==[6]
        assert not batch.terminated.any()
        assert torch.nonzero(batch.truncated[:,0]).flatten().cpu().tolist()==[249,499,749,999,1249,1499]
        for end in (249,499,749,999,1249,1499):
            assert batch.next_values[end,0,0].item()==0  # real deadline state, never reset state
        assert batch.values[250,0,0].item()==10
        expected=np.array([1-(i%250)/250 for i in range(1500)],dtype=np.float32)
        np.testing.assert_allclose(batch.observations[:,0,0,11].cpu(),expected,rtol=0,atol=0)


def test_checkpoint_bitwise_resume_with_workers_optimizers_and_rng(tmp_path):
    cfg=RegionTrainConfig(num_envs=2,horizon=3,ppo_epochs=1,minibatch_size=8)
    ec=RegionConfig(max_steps=2)
    trainer=RegionTrainer(cfg)
    with RegionVectorEnv(ec,2,cfg.seed,[10000000]) as vector:
        assert len(set(vector.worker_pids))==2
        assert not np.array_equal(vector.observations[0],vector.observations[1])
        batch,episodes=trainer.collect(vector,5)
        assert trainer.sampled_steps==5 and trainer.rollout_metrics['valid_uav_samples']==15
        assert not batch.current_masks[-1,1].any()
        before=[p.detach().clone() for p in trainer.actor.parameters()]
        result=trainer.update(batch)
        assert result['valid_samples']==15 and result['optimizer_steps']==2
        assert any(not torch.equal(a,b) for a,b in zip(before,trainer.actor.parameters()))
        np.random.seed(123); random.seed(456)
        path=tmp_path/'region.pt'; save_checkpoint(path,trainer,ec,vector)
        payload=read_checkpoint(path)
        expected_np=np.random.rand(); expected_py=random.random()
        expected_batch,_=trainer.collect(vector,5); expected=trainer.update(expected_batch)
        expected_actor=copy.deepcopy(trainer.actor.state_dict()); expected_critic=copy.deepcopy(trainer.critic.state_dict())
        expected_adam=copy.deepcopy(trainer.actor_optimizer.state_dict()); expected_cadam=copy.deepcopy(trainer.critic_optimizer.state_dict())
        expected_workers=canonical(vector.snapshot())
    restored=RegionTrainer(cfg)
    with RegionVectorEnv(ec,2,cfg.seed,[10000000]) as vector:
        restore_checkpoint(payload,restored,ec,vector)
        assert np.random.rand()==expected_np and random.random()==expected_py
        batch,_=restored.collect(vector,5); actual=restored.update(batch)
        for field in fields(Rollout): torch.testing.assert_close(getattr(batch,field.name),getattr(expected_batch,field.name),rtol=0,atol=0)
        assert expected==actual
        for name,value in restored.actor.state_dict().items(): torch.testing.assert_close(value,expected_actor[name],rtol=0,atol=0)
        for name,value in restored.critic.state_dict().items(): torch.testing.assert_close(value,expected_critic[name],rtol=0,atol=0)
        for actual_opt,expected_opt in [(restored.actor_optimizer.state_dict(),expected_adam),(restored.critic_optimizer.state_dict(),expected_cadam)]:
            for key,state in actual_opt['state'].items():
                for name,value in state.items(): torch.testing.assert_close(value,expected_opt['state'][key][name],rtol=0,atol=0)
            assert actual_opt['param_groups']==expected_opt['param_groups']
        assert canonical(vector.snapshot())==expected_workers
        check_seeds([10000000],cfg,vector.snapshot())
        with pytest.raises(ValueError): check_seeds([cfg.seed],cfg,vector.snapshot())
    from algorithm.common.checkpoint import read_checkpoint as inspection_read
    with pytest.raises(ValueError,match='incompatible'): inspection_read(path)
    from algorithm.mappo.trainer import MAPPOTrainer
    from algorithm.common.checkpoint import save_checkpoint as inspection_save
    from env.config import EnvConfig
    old=tmp_path/'inspection.pt'; inspection_save(old,MAPPOTrainer(),EnvConfig())
    with pytest.raises(ValueError,match='incompatible'): read_checkpoint(old)


def test_five_agent_rollout_updates_all_five_value_heads():
    trainer=RegionTrainer(RegionTrainConfig(num_envs=1,horizon=2,ppo_epochs=1,minibatch_size=8))
    with RegionVectorEnv(RegionConfig(scale='5v5',max_steps=2),1,123) as vector:
        batch,episodes=trainer.collect(vector)
        assert batch.states.shape==(2,1,60) and batch.actions.shape==(2,1,5,3)
        before=trainer.critic.network[-1].weight.detach().clone()
        result=trainer.update(batch)
        assert result['valid_samples']==10 and result['optimizer_steps']==2
        assert all(not torch.equal(a,b) for a,b in zip(before,trainer.critic.network[-1].weight))


def test_independent_offline_score_reward_and_interactive_replay(tmp_path):
    trainer=RegionTrainer()
    trace=tmp_path/'trajectory.json'
    report=evaluate(trainer,RegionConfig(),[20000000,20000001],trace_path=trace,red_rule='region_safe')
    assert len(report['per_episode'])==2 and report['mean_episode_length']==250
    before=trace.read_bytes(); result=audit(trace)
    assert result['passed'] and result['max_reward_error']<1e-8 and trace.read_bytes()==before
    # Mirrored equal controllers occupy together: contested intervals earn neither team points.
    assert abs(result['recomputed_scores']['red'])<1e-8 and abs(result['recomputed_scores']['blue'])<1e-8
    assert report['red']['mean_dwell_seconds']>0 and report['blue']['mean_dwell_seconds']>0
    replay=render_replay(trace,tmp_path/'replay.html')
    html=replay.read_text(encoding='utf-8')
    assert 'red UAV 0' in html and 'blue UAV 0' in html and 'Region 0' in html and 'Plotly.addFrames' in html
    assert 'surface' in html and 'sliders' in html and 'red' in html and 'blue' in html
    with pytest.raises(FileExistsError): render_replay(trace,replay)
    damaged=json.loads(trace.read_text(encoding='utf-8')); damaged['frames'][10]['scores']['red']+=1
    other=tmp_path/'damaged.json'; other.write_text(json.dumps(damaged),encoding='utf-8')
    assert not audit(other)['passed']
    asymmetric=tmp_path/'asymmetric.json'
    evaluate(trainer,RegionConfig(blue_policy='cruise'),[20000000],trace_path=asymmetric,red_rule='region_safe')
    result=audit(asymmetric)
    assert result['passed'] and sum(result['recomputed_scores'].values())>0
