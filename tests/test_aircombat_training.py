"""Real CUDA rollout/update, final-state bootstrap and exact worker/RNG resume."""
import copy
import dataclasses
import json
import random
import numpy as np
import pytest
import torch
from env.aircombat import AirCombatConfig
from env.aircombat.config import ENVIRONMENT_VERSION
from algorithm.common.aircombat_vector import AirCombatVectorEnv,combat_metrics
from algorithm.common.checkpoint import (save_protocol_checkpoint,read_protocol_checkpoint,
                                         restore_protocol_checkpoint,read_checkpoint)
from algorithm.mappo.config import ProtocolTrainConfig
from algorithm.mappo.trainer import ProtocolMAPPOTrainer,gae,standardize_advantages
from algorithm.aircombat import evaluate,train


def tensor_equal(a,b):
    if isinstance(a,torch.Tensor): torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,np.ndarray): np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: tensor_equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b): tensor_equal(x,y)
    elif dataclasses.is_dataclass(a):
        for f in dataclasses.fields(a):tensor_equal(getattr(a,f.name),getattr(b,f.name))
    elif hasattr(a,'__dict__'):
        tensor_equal(vars(a),vars(b))
    else: assert a==b


def test_real_parallel_rollout_exact_beta_logp_masks_update_and_gradients():
    c=ProtocolTrainConfig(num_envs=2,horizon=4,ppo_epochs=1,minibatch_size=7)
    t=ProtocolMAPPOTrainer(c)
    before=[p.detach().clone() for p in t.actor.parameters()]
    with AirCombatVectorEnv(AirCombatConfig(),2) as v:
        rollout,episodes=t.collect(v)
        assert len(set(v.worker_pids))==2
        assert rollout.observations.shape==(4,2,3,12) and rollout.states.shape==(4,2,60)
        with torch.no_grad():
            logp,_=t.actor.evaluate(rollout.observations,rollout.actions)
        torch.testing.assert_close(logp,rollout.log_probs,rtol=0,atol=0)
        assert t.sampled_steps==8
        result=t.update(rollout)
    assert result['valid_samples']==24 and result['optimizer_steps']>0 and t.updates==1
    assert any(not torch.equal(p,q) for p,q in zip(t.actor.parameters(),before))
    assert all(np.isfinite(result[k]) for k in ('policy_loss','critic_loss','entropy','approx_kl','gradient_norm'))
    assert all(p.grad is None or torch.isfinite(p.grad).all() for m in (t.actor,t.critic) for p in m.parameters())


def test_vector_auto_reset_preserves_real_final_state_and_timeout_bootstrap():
    ec=AirCombatConfig(max_steps=1)
    c=ProtocolTrainConfig(num_envs=1,horizon=2,ppo_epochs=1)
    t=ProtocolMAPPOTrainer(c)
    with AirCombatVectorEnv(ec,1) as v:
        rollout,episodes=t.collect(v)
        assert len(episodes)==2 and t.training_episode_count==2
        assert rollout.truncated.all() and not rollout.terminated.any()
        assert rollout.next_masks.all()
        # Capture final state through the vector API separately; reset state differs.
        tr=v.step(np.full((1,3,3),.5))
        assert not np.array_equal(tr.states,tr.transition_states)
        assert not np.array_equal(tr.observations,tr.transition_observations)
        assert tr.next_masks.all() and tr.current_masks.all()
        final_value=t.critic(t.tensor(tr.transition_states))[...,:3]
        assert torch.isfinite(final_value).all()
    adv,ret=gae(rollout.rewards,rollout.values,rollout.next_values,rollout.current_masks,
                rollout.next_masks,rollout.terminated,rollout.truncated,c.gamma,c.gae_lambda)
    torch.testing.assert_close(ret,rollout.rewards+c.gamma*rollout.next_values)


def test_collect_next_values_uses_final_states_not_reset(monkeypatch):
    c=ProtocolTrainConfig(num_envs=1,horizon=1,ppo_epochs=1)
    t=ProtocolMAPPOTrainer(c)
    with AirCombatVectorEnv(AirCombatConfig(max_steps=1),1) as v:
        original=v.step
        captured={}
        def step(*args,**kwargs):
            tr=original(*args,**kwargs)
            captured['tr']=tr
            return tr
        monkeypatch.setattr(v,'step',step)
        r,_=t.collect(v)
        with torch.no_grad():
            expected=t.critic(t.tensor(captured['tr'].transition_states))[...,:3]
        torch.testing.assert_close(r.next_values[0],expected,rtol=0,atol=0)


def test_gae_keeps_death_reward_and_excludes_later_dead_and_reset_carry():
    r=torch.tensor([[[7.,2.]],[[float('nan'),3.]]],device='cuda')
    mask=torch.tensor([[[1.,1.]],[[0.,1.]]],device='cuda')
    nxt=torch.tensor([[[0.,1.]],[[0.,1.]]],device='cuda')
    values=torch.ones_like(r);nv=torch.full_like(r,5.)
    terminal=torch.tensor([[False],[False]],device='cuda');timeout=torch.tensor([[True],[False]],device='cuda')
    adv,ret=gae(r,values,nv,mask,nxt,terminal,timeout,.954,.95)
    assert ret[0,0,0]==7 and not ret[1,0,0] and not adv[1,0,0]
    assert ret[0,0,1].item()==pytest.approx(2+.954*5)


def test_singleton_keeps_policy_signal_and_empty_update_skips():
    a=torch.tensor([3.,1.,3.],device='cuda')
    slots=torch.tensor([0,1,1],device='cuda')
    out=standardize_advantages(a,slots,per_agent=True)
    torch.testing.assert_close(out,torch.tensor([3.,-1.,1.],device='cuda'))
    t=ProtocolMAPPOTrainer(ProtocolTrainConfig(num_envs=1,horizon=1,ppo_epochs=1,entropy_coef=0))
    with AirCombatVectorEnv(AirCombatConfig(),1) as v:
        r,_=t.collect(v)
    # Keep exactly one true environment sample, inspect singleton with no entropy.
    r.current_masks[...,1:]=0;r.next_masks[...,1:]=0;r.observations[...,1:,:]=0
    r.states[...,12:]=0
    r.terminated[:]=True
    r.rewards[...,0]=10
    before=[p.detach().clone() for p in t.actor.parameters()]
    m=t.update(r)
    assert m['valid_samples']==1 and m['policy_loss']!=0
    assert any(not torch.equal(p,q) for p,q in zip(t.actor.parameters(),before))
    r.current_masks.zero_();r.next_masks.zero_();r.observations.zero_();r.states.zero_()
    weights=copy.deepcopy(t.actor.state_dict());rng=torch.cuda.get_rng_state().clone()
    result=t.update(r)
    assert result['optimizer_steps']==0 and t.updates==1
    tensor_equal(weights,t.actor.state_dict());torch.testing.assert_close(rng,torch.cuda.get_rng_state(),rtol=0,atol=0)


def test_checkpoint_full_worker_state_and_resume_exact_following_rollout_update(tmp_path):
    ec=AirCombatConfig(max_steps=3)
    c=ProtocolTrainConfig(num_envs=2,horizon=2,ppo_epochs=1,minibatch_size=7)
    t=ProtocolMAPPOTrainer(c)
    with AirCombatVectorEnv(ec,2) as v:
        r,_=t.collect(v);t.update(r)
        path=tmp_path/'combat.pt'
        save_protocol_checkpoint(path,t,env_config=ec,vector=v)
        payload=read_protocol_checkpoint(path,environment_version=ENVIRONMENT_VERSION)
        for worker in payload['vector']['workers']:
            assert 'task' not in worker and worker['state'].friendly_states.shape==(3,6)
            assert worker['state'].friendly_dwell.shape==(3,3)
        r1,e1=t.collect(v);m1=t.update(r1)
        end=v.snapshot()
        actor=copy.deepcopy(t.actor.state_dict());critic=copy.deepcopy(t.critic.state_dict())
        opt=copy.deepcopy(t.actor_optimizer.state_dict());optc=copy.deepcopy(t.critic_optimizer.state_dict())
        next_random=(random.random(),np.random.random(),torch.rand(3,device='cuda'))
        clone=ProtocolMAPPOTrainer(c)
        restore_protocol_checkpoint(payload,clone,env_config=ec,vector=v)
        r2,e2=clone.collect(v);m2=clone.update(r2)
        tensor_equal(r1,r2);tensor_equal(e1,e2);tensor_equal(m1,m2)
        tensor_equal(actor,clone.actor.state_dict());tensor_equal(critic,clone.critic.state_dict())
        tensor_equal(opt,clone.actor_optimizer.state_dict());tensor_equal(optc,clone.critic_optimizer.state_dict())
        tensor_equal(end,v.snapshot())
        tensor_equal(next_random,(random.random(),np.random.random(),torch.rand(3,device='cuda')))
    with pytest.raises(ValueError):read_protocol_checkpoint(path)
    with pytest.raises(ValueError):read_checkpoint(path)
    old=tmp_path/'old.pt';torch.save({'environment_version':'coop3d-v3.0.0'},old)
    with pytest.raises(ValueError):read_protocol_checkpoint(old,environment_version=ENVIRONMENT_VERSION)


def test_evaluation_independent_rng_stream_win_rate_and_default():
    c=ProtocolTrainConfig(num_envs=1,horizon=1,ppo_epochs=1)
    assert c.evaluation_episodes==200 and c.training_episodes==5000
    t=ProtocolMAPPOTrainer(c)
    state=(copy.deepcopy(np.random.get_state()),torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone())
    a=evaluate(t,AirCombatConfig(max_steps=2),[10000000,10000001])
    b=evaluate(t,AirCombatConfig(max_steps=2),[10000000,10000001])
    assert a==b and a['win_rate']==0
    tensor_equal(state,(np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state()))
    metrics=combat_metrics([dict(episode_return=1,episode_length=2,win=False,blue_shot_down=2,
        red_survivors=3,numerical_failures=0,reward_components_mean=dict.fromkeys(('attack','posture','allied','shootdown','terminal'),0))])
    assert metrics['mean_blue_shot_down']==2 and metrics['win_rate']==0
    with pytest.raises(ValueError):evaluate(t,AirCombatConfig(),[1000])


def test_episode_budget_checkpoint_schedule_cli_core_and_bounded_horizon(tmp_path):
    c=ProtocolTrainConfig(num_envs=2,horizon=2,ppo_epochs=1,minibatch_size=8,
                          training_episodes=3,checkpoint_interval_episodes=1)
    trainer=train(AirCombatConfig(max_steps=2),c,tmp_path/'run')
    assert trainer.training_episode_count==3
    assert (tmp_path/'run'/'latest.pt').exists()
    lines=[json.loads(s) for s in (tmp_path/'run'/'train.jsonl').read_text(encoding='utf-8').splitlines()]
    assert lines[-1]['completed_episodes']==3 and all(r['ppo']['valid_samples']>0 for r in lines)
    with AirCombatVectorEnv(AirCombatConfig(max_steps=10),2) as v:
        r,episodes=trainer.collect(v,episode_limit=1)
        assert len(r.rewards)<=c.horizon and not episodes
