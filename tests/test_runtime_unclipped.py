import dataclasses
import math

import numpy as np
import pytest
import torch
from torch import nn

from env import EnvConfig
from env.factory import make_environment
from env.dynamics import derivatives
from algorithm.common.vector_env import ParallelVectorEnv
from algorithm.mappo.config import TrainConfig
from algorithm.mappo.trainer import MAPPOTrainer, gae


def test_real_environment_path_accepts_86_degrees_and_no_gamma_limit_config():
    assert 'gamma_limit' not in {f.name for f in dataclasses.fields(EnvConfig)}
    with pytest.raises(TypeError): EnvConfig(gamma_limit=math.radians(85))
    env=make_environment(EnvConfig(random_scenario=False))
    gamma=math.radians(86)
    states=np.array([[0,-300,3000,200,gamma,0],[0,0,3000,200,gamma,0],[0,300,3000,200,gamma,0]])
    obs,info=env.reset(3,options={'states':states})
    np.testing.assert_allclose(obs[:,2], gamma/np.pi, rtol=1e-7)
    action=np.tile([.5,(math.sin(gamma)+1)/3,math.cos(gamma)/8],(3,1))
    obs,reward,_,_,info=env.step(action)
    expected=states.copy(); expected[:,0]+=50*math.cos(gamma); expected[:,2]+=50*math.sin(gamma)
    np.testing.assert_allclose(info['states'],expected,rtol=0,atol=1e-12)
    assert info['numerical_counters']['gamma_clips']==0
    assert not info['failures'] and np.isfinite(obs).all() and np.isfinite(reward).all()


def test_negative_cosine_preserves_original_derivative_sign_and_idle_load_bounds():
    gamma=math.radians(100)
    d=derivatives([0,0,3000,200,gamma,0],[.3,0,1],EnvConfig())
    assert d[5]<0  # no abs(cos), positive floor, or gamma clamp
    env=make_environment(EnvConfig(random_scenario=False))
    states=np.tile([0,0,3000,200,gamma,0],(3,1)); states[:,1]=[-300,0,300]
    env.reset(3,options={'states':states})
    env.assignments[-1]=-1
    _,_,_,_,info=env.step(np.tile([.5,1/3,1/8],(3,1)))
    np.testing.assert_allclose(info['states'][-1,3:],[200,gamma,0],rtol=0,atol=1e-12)
    assert info['physical_actions'][-1,2]>=0 and info['physical_actions'][-1,0]==np.pi


def test_stage_singularity_failure_is_deterministic_recorded_and_finite():
    env=make_environment(EnvConfig(random_scenario=False))
    states=np.tile([0,0,3000,100,math.radians(89),0],(3,1)); states[:,1]=[-300,0,300]
    for _ in range(2):
        env.reset(3,options={'states':states})
        obs,reward,terminated,truncated,info=env.step(np.tile([.5,2/3,1.],(3,1)))
        np.testing.assert_array_equal(info['states'],states)
        assert terminated and not truncated and not info['active_mask'].any()
        assert info['numerical_counters']['integration_failures']==3
        assert len(info['failures'])==3
        assert all(f['reason']=='integration_failed' and 'singularity' in f['detail'] for f in info['failures'])
        assert np.isfinite(obs).all() and np.isfinite(reward).all() and np.isfinite(info['centralized_state']).all()


def test_unrepresentable_observation_fails_instead_of_entering_training_tensor():
    env=make_environment(EnvConfig(random_scenario=False))
    states=np.tile([0,0,3000,200,1e308,0],(3,1))
    with np.errstate(over='ignore',invalid='ignore'), pytest.raises(FloatingPointError,match='finite float32'):
        env.reset(3,options={'states':states})


class TrimSampler(nn.Module):
    def act(self, obs, deterministic=False):
        a=torch.tensor([.5,1/3,1/8],device=obs.device).expand(*obs.shape[:-1],3)
        return a,torch.zeros(obs.shape[:-1],device=obs.device)

    def evaluate(self,obs,actions):
        return torch.zeros(obs.shape[:-1],device=obs.device),torch.zeros(obs.shape[:-1],device=obs.device)


class PoleCrossingSampler(TrimSampler):
    def act(self,obs,deterministic=False):
        a=torch.tensor([.5,2/3,1.],device=obs.device).expand(*obs.shape[:-1],3)
        return a,torch.zeros(obs.shape[:-1],device=obs.device)


def test_failed_integrations_are_recorded_by_collector_without_nonfinite_tensors():
    t=MAPPOTrainer(TrainConfig(num_envs=1,horizon=1))
    t.actor=PoleCrossingSampler()
    config=EnvConfig(scenario={'initial_gamma':(math.radians(89),math.radians(89)), 'initial_speed':(100,100)})
    with ParallelVectorEnv(config,1,t.config.seed) as vector:
        batch,episodes=t.collect(vector)
    assert t.rollout_metrics['numerical_events']['integration_failures']==3
    assert len(episodes)==1 and batch.terminated.all()
    assert batch.current_masks.sum()==3 and not batch.next_masks.any()
    for field in dataclasses.fields(batch): assert torch.isfinite(getattr(batch,field.name)).all()


def test_actual_1500_rollout_crosses_six_250_step_episodes_and_gae_cuts_carry():
    # Collection only: one worker, trim actions, no optimization/long training.
    t=MAPPOTrainer(TrainConfig(num_envs=1,horizon=1500))
    t.actor=TrimSampler()
    with ParallelVectorEnv(EnvConfig(random_scenario=False),1,t.config.seed) as vector:
        batch,episodes=t.collect(vector)
        assert vector.snapshot()['workers'][0]['step']==0
    assert t.sampled_steps==1500 and t.updates==0
    assert len(episodes)==6 and all(e['episode_length']==250 for e in episodes)
    boundaries=torch.arange(249,1500,250,device=t.device)
    assert batch.truncated.sum()==6 and batch.truncated[boundaries].all() and not batch.terminated.any()
    assert batch.current_masks.sum()==4500
    # Independent hand answer for all 1500 entries: delta=1+.954*2-1,
    # reset states never leak across the 250-step time-limit boundaries.
    r=torch.ones_like(batch.rewards); v=r.clone(); nv=2*r
    a,_=gae(r,v,nv,batch.current_masks,batch.next_masks,batch.terminated,batch.truncated,.954,.95)
    torch.testing.assert_close(a[boundaries],torch.full_like(a[boundaries],1.908))
    q=.954*.95
    expected=1.908*(1-q**250)/(1-q)
    torch.testing.assert_close(a[torch.arange(0,1500,250,device=t.device)],torch.full((6,1,3),expected,device=t.device))
