import dataclasses
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from env import EnvConfig
from algorithm.mappo.config import TrainConfig
from algorithm.mappo.networks import Actor,Critic,TRIM_ACTION
from algorithm.mappo.trainer import gae,MAPPOTrainer,Rollout,clean_global
from algorithm.common.vector_env import ParallelVectorEnv
from algorithm.common.checkpoint import read_checkpoint,save_checkpoint,restore_checkpoint
from algorithm.mappo.evaluation import ensure_independent


def test_trim_centered_beta_parameters_and_exploration():
    torch.manual_seed(1); a=Actor()
    obs=torch.randn(32,12)
    dist=a.distribution(obs)
    torch.testing.assert_close(dist.mean,torch.tensor(TRIM_ACTION).expand(32,-1),atol=.001,rtol=0)
    torch.testing.assert_close(dist.concentration1.mean(0),torch.tensor([8.,16/3,2.]),atol=.01,rtol=0)
    sample=dist.sample((1000,))
    assert ((sample>0)&(sample<1)).all() and (sample.std(0)>.05).all()
    logp,entropy=a.evaluate(obs,dist.mean)
    torch.testing.assert_close(entropy,dist.entropy().sum(-1))
    assert torch.isfinite(logp).all()


def test_individual_value_fitting(record_property):
    from algorithm.common.runtime import require_cuda
    require_cuda(); torch.manual_seed(8); torch.set_num_threads(1)
    state=torch.randn(1,90,device='cuda'); target=torch.tensor([[-2.,0.,1.,3.,6.]],device='cuda')
    critic=Critic().cuda()
    scalar=torch.nn.Sequential(torch.nn.Linear(90,128),torch.nn.Tanh(),torch.nn.Linear(128,128),torch.nn.Tanh(),torch.nn.Linear(128,1)).cuda()
    for model in (critic,scalar):
        opt=torch.optim.Adam(model.parameters(),lr=.003)
        for _ in range(160):
            loss=(model(state)-target).square().mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        new_loss=float((critic(state)-target).square().mean()); old_loss=float((scalar(state)-target).square().mean())
    record_property('individual_value_mse',new_loss); record_property('scalar_broadcast_mse',old_loss)
    assert new_loss<1e-5 and old_loss>5
    torch.testing.assert_close(critic(state),target,atol=.01,rtol=0)


def test_nan_inactive_gae_and_terminal_successor():
    r=torch.tensor([[[1.,2.],[3.,4.]],[[float('nan'),5.],[float('nan'),6.]]])
    cur=torch.tensor([[[1.,1.],[1.,1.]],[[0.,1.],[0.,1.]]])
    nxt=torch.tensor([[[0.,1.],[0.,1.]],[[0.,1.],[0.,1.]]])
    values=torch.where(cur.bool(),torch.ones_like(r),torch.full_like(r,float('nan')))
    next_values=torch.full_like(r,float('nan')); next_values[0,:,1]=2
    term=torch.tensor([[False,False],[True,True]]); trunc=torch.tensor([[True,True],[False,False]])
    a,ret=gae(r,values,next_values,cur,nxt,term,trunc,.9,.95)
    assert torch.isfinite(a).all() and torch.isfinite(ret).all()
    assert a[0,0,0]==0 and a[0,0,1]==pytest.approx(2+.9*2-1)
    assert a[0,1,0]==2
    assert not a[1,:,0].any()


@pytest.mark.parametrize('field',['reward','value','next_value'])
def test_valid_nonfinite_gae_fails(field):
    data=[torch.ones(1,1,1) for _ in range(3)]
    data[{'reward':0,'value':1,'next_value':2}[field]][0]=float('nan')
    with pytest.raises(FloatingPointError,match='valid samples'):
        gae(*data,torch.ones(1,1,1),torch.ones(1,1,1),torch.zeros(1,1,dtype=torch.bool),torch.zeros(1,1,dtype=torch.bool),.954,.95)


def test_partial_vector_budget_and_true_successor_value():
    cfg=TrainConfig(num_envs=2,horizon=8,ppo_epochs=1,evaluation_episodes=1)
    ec=EnvConfig(max_steps=2,failure_events={1:[1]})
    trainer=MAPPOTrainer(cfg)
    with ParallelVectorEnv(ec,2,1000) as vector:
        batch,episodes=trainer.collect(vector,joint_steps=5)
        assert trainer.sampled_steps==5
        assert [p['step'] for p in vector.snapshot()['workers']]==[1,0]
        assert not batch.current_masks[-1,1].any()  # non-stepped worker is padding
        assert len(episodes)==2
        assert batch.truncated[1].all()
        # Restore an initial vector and replay the identical saved actions to
        # verify next_values use real successors, not auto-reset states.
    with ParallelVectorEnv(ec,2,1000) as vector:
        for t in range(2):
            transition=vector.step(batch.actions[t].cpu().numpy())
        with torch.no_grad():
            expected=trainer.critic(clean_global(trainer.tensor(transition.transition_states),trainer.tensor(transition.next_masks)))[...,:3]
            reset=trainer.critic(clean_global(trainer.tensor(transition.states),trainer.tensor(vector.masks)))[...,:3]
        torch.testing.assert_close(batch.next_values[1],expected)
        assert not torch.allclose(expected,reset)


def test_nan_inactive_losses_and_unused_heads():
    c=TrainConfig(num_envs=1,horizon=4,ppo_epochs=1,minibatch_size=32)
    trainer=MAPPOTrainer(c)
    with ParallelVectorEnv(EnvConfig(failure_events={1:[1]}),1,c.seed) as vector:
        original,_=trainer.collect(vector)
    fields={f.name:getattr(original,f.name).clone() for f in dataclasses.fields(original)}
    invalid=~original.current_masks.bool()
    for key in ('rewards','values','next_values','actions','observations','log_probs'):
        fields[key][invalid]=float('nan')
    # Last next value of the failed agent can also be arbitrary on failure step.
    fields['next_values'][~original.next_masks.bool()]=float('nan')
    for t in range(1,4): fields['states'][t,0,12:24]=float('nan')
    a,b=MAPPOTrainer(c),MAPPOTrainer(c)
    torch.manual_seed(13); ma=a.update(original)
    torch.manual_seed(13); mb=b.update(Rollout(**fields))
    assert ma==mb
    for x,y in zip(a.critic.parameters(),b.critic.parameters()): torch.testing.assert_close(x,y,rtol=0,atol=0)
    assert not a.critic.network[-1].weight.grad[3:].any()


def test_seed_isolation_and_legacy_checkpoint_rejection(tmp_path):
    ensure_independent([10000000],1000,2,np.array([4,2]))
    with pytest.raises(ValueError,match='used for training'): ensure_independent([1004],1000,2,np.array([4,2]))
    old=tmp_path/'old.pt'; torch.save(dict(environment_version='coop3d-v1.0.0',critic_input=60),old)
    with pytest.raises(ValueError,match='incompatible checkpoint'): read_checkpoint(old)
    with pytest.raises(ValueError,match='overlaps'): ParallelVectorEnv(EnvConfig(),1,seed=7,forbidden_seeds=[7])


@pytest.mark.parametrize('kwargs',[{'num_envs':1.2},{'ppo_epochs':0},{'gamma':float('nan')},{'entropy_coef':float('inf')},
    {'beta_concentration':8},{'initialization_mode':'bad'},{'evaluation_interval_sampled_steps':0},{'evaluation_episodes':0},{'clip':2},
    {'evaluation_seeds':[1.5]},{'seed':-1}])
def test_algorithm_config_validation(kwargs):
    with pytest.raises(ValueError): TrainConfig(**kwargs)
