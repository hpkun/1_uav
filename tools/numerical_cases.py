"""Controlled counterexamples/hand answers; no new task or training reward."""
import math

import numpy as np
import torch
from torch import nn

from env import EnvConfig
from env.factory import make_environment
from algorithm.common.metrics import EpisodeAccumulator
from algorithm.mappo.config import ProtocolTrainConfig
from algorithm.mappo.paper_protocol import pack_centralized_slots
from algorithm.mappo.trainer import ProtocolMAPPOTrainer, Rollout, gae, standardize_advantages
from algorithm.mappo.diagnostics import isolated_rng


def reward_boundaries():
    """Same initial geometry, original 250-step limit and unmodified reward."""
    gamma = math.radians(89)
    direction = np.array([math.cos(gamma), 0., math.sin(gamma)])
    states = np.tile([0., 0, 3000, 100, gamma, 0], (3, 1))
    states[:, 1] = [-10000, 0, 10000]
    points = states[:, :3] - 1000*direction
    trim = np.array([.5, (math.sin(gamma)+1)/3, math.cos(gamma)/8])
    rows = []
    for name, failed, injected in (
        ('normal_timeout', [], False), ('integration_one', [0], False),
        ('integration_all', [0,1,2], False), ('injected_one', [0], True), ('injected_all', [0,1,2], True)):
        config = EnvConfig(random_scenario=False, failure_events={1:failed} if injected else {})
        env = make_environment(config)
        obs, info = env.reset(7, options=dict(states=states, waypoints=points))
        tracker = EpisodeAccumulator(); tracker.reset(info)
        totals = {k:np.zeros(3) for k in ('progress','completion','separation','efficiency','terminal')}
        prefix = []
        while True:
            actions = np.tile(trim, (3,1))
            if failed and not injected and env.step_count == 0:
                actions[failed] = [.5, 2/3, 1.]
            obs, reward, terminal, truncated, info = env.step(actions)
            tracker.add(info, reward)
            for k in totals: totals[k] += info['reward_components'][k]
            assert np.isfinite(obs).all() and np.isfinite(reward).all()
            if len(prefix) < 2:
                r = torch.tensor(reward, dtype=torch.float64).reshape(1,1,3)
                mask = torch.tensor(info['current_active_mask']).reshape(1,1,3).double()
                nxt = torch.tensor(info['next_active_mask']).reshape(1,1,3).double()
                adv, ret = gae(r, torch.ones_like(r), torch.full_like(r,2), mask, nxt,
                               torch.tensor([[terminal]]), torch.tensor([[truncated]]), .954, .95)
                prefix.append(dict(step=info['episode_length'], current_mask=mask.flatten().tolist(),
                                   next_mask=nxt.flatten().tolist(), reward=reward.tolist(),
                                   components={k:v.tolist() for k,v in info['reward_components'].items()},
                                   failures=info['failures'], terminated=terminal,truncated=truncated,
                                   one_step_advantage_v1_next_v2=adv.flatten().tolist(),
                                   one_step_return_v1_next_v2=ret.flatten().tolist()))
            if terminal or truncated: break
        expected = np.full(3, -15.)
        if failed:
            expected[failed] = -.06 if injected else -.01
        rows.append(dict(name=name, steps=env.step_count, terminated=terminal, truncated=truncated,
                         per_uav_return=tracker.reward.tolist(), expected_analytic_return=expected.tolist(),
                         analytic_error=(tracker.reward-expected).tolist(),
                         expected_float32_return=[float(np.float32(-.06))*250 if i not in failed
                                                  else float(np.float32(-.06 if injected else -.01)) for i in range(3)],
                         component_totals={k:v.tolist() for k,v in totals.items()},
                         summary=tracker.summary(info), first_steps=prefix,
                         numerical_counters=info['numerical_counters']))
    return dict(source='controlled inspection reset fixtures, NOT a new training task',
                initial_states=states.tolist(), waypoints=points.tolist(), trim_action=trim.tolist(),
                construction='targets 1000 m opposite initial velocity; 10000 m lateral separation; no visits/proximity rewards',
                healthy_step_analytic=dict(progress=-.05,efficiency=-.01,total=-.06), cases=rows)


class HandActor(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.logs = nn.Parameter(torch.tensor([1.4,.7,1.1,1.,1.],device=device).log())

    def evaluate(self, obs, actions):
        ids = obs[:,1].long()-200
        return self.logs[ids], (ids+1).to(obs.dtype)*.1


class HandCritic(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.heads = nn.Parameter(torch.zeros(5,device=device))

    def forward(self, states):
        return self.heads.expand(len(states),-1)


def hand_batch(counts, device, poison=True):
    """Terminal toy transitions: V=0 so GAE A=R=r=i*10+t+1 exactly."""
    length, n = max(1,max(counts)), len(counts)
    mask = np.array([[t<c for c in counts] for t in range(length)], dtype=np.float32)[:,None,:]
    obs = np.zeros((length,1,n,12),dtype=np.float32); obs[...,0] = mask
    for i in range(n): obs[:counts[i],0,i,1] = 200+i
    states = pack_centralized_slots(obs,mask)
    tensor = lambda x:torch.as_tensor(x,device=device,dtype=torch.float32)
    rewards = np.zeros_like(mask)
    for i,c in enumerate(counts): rewards[:c,0,i] = np.arange(1,c+1)+10*i
    z = torch.zeros_like(tensor(mask))
    batch = Rollout(tensor(obs),tensor(states),torch.full((*mask.shape,3),.5,device=device),z.clone(),z.clone(),z.clone(),
                    tensor(rewards),tensor(mask),tensor(mask),torch.ones((length,1),device=device,dtype=torch.bool),
                    torch.zeros((length,1),device=device,dtype=torch.bool))
    if poison:
        invalid = ~batch.current_masks.bool()
        for field in ('observations','actions','log_probs','values','next_values','rewards'):
            getattr(batch,field)[invalid] = float('nan')
        slots=batch.states.reshape(length,1,5,12)
        full=torch.zeros((length,1,5),device=device,dtype=torch.bool); full[...,:n]=batch.current_masks.bool()
        slots[~full]=float('nan')
    return batch


def _reference(counts, minibatch, normalize, order, ratios, entropy_coef=.001):
    """Independent NumPy scalar objectives and gradients, not trainer helpers."""
    rewards, ids = [], []
    for t in range(max(1,max(counts))):
        for i,c in enumerate(counts):
            if t<c: rewards.append(10*i+t+1.); ids.append(i)
    r, slots = np.array(rewards),np.array(ids,dtype=int)
    a=r.copy()
    for i,c in enumerate(counts):
        if c and normalize:
            a[slots==i]=(a[slots==i]-a[slots==i].mean())/(a[slots==i].std()+1e-8)
    answers=[]
    participation=np.zeros(len(counts),dtype=int)
    for start in range(0,len(order),minibatch):
        ix=order[start:start+minibatch]; policy=entropy=critic=0.
        ag=np.zeros(5); cg=np.zeros(5)
        for i in np.unique(slots[ix]):
            sel=ix[slots[ix]==i]; rho=ratios[i]; clipped=np.clip(rho,.8,1.2)
            policy-=np.minimum(rho*a[sel],clipped*a[sel]).mean()
            entropy+=(i+1)*.1; critic+=np.square(r[sel]).mean()
            active=(rho*a[sel] <= clipped*a[sel]) | (rho==clipped)
            ag[i]=-np.where(active,rho*a[sel],0).mean(); cg[i]=-2*r[sel].mean()
            participation[i]+=1
        answers.append([policy,policy-entropy_coef*entropy,critic,np.linalg.norm(ag),np.linalg.norm(cg)])
    means=np.mean(answers,axis=0) if answers else np.zeros(5)
    return dict(zip(('policy_loss','actor_total_loss','critic_loss','actor_norm_mean','critic_norm_mean'),means.tolist())),a,participation.tolist(),answers


def ppo_boundaries():
    rows=[]
    with isolated_rng():
        for name,counts,mini,normalize in (
            ('singleton',(1,0,0,0,0),128,True), ('two_samples',(2,0,0,0,0),128,True),
            ('imbalanced',(31,2,1,0,0),128,True), ('single_uav_minibatches',(4,3,2,0,0),1,True),
            ('multi_uav_minibatch',(4,3,2,0,0),128,True), ('unnormalized_singleton',(1,0,0,0,0),128,False),
            ('all_invalid',(0,0,0,0,0),128,True)):
            config=ProtocolTrainConfig(ppo_epochs=1,minibatch_size=mini,normalize_advantages=normalize)
            t=ProtocolMAPPOTrainer(config)
            t.actor,t.critic=HandActor(t.device),HandCritic(t.device)
            ratios=np.array([1.4,.7,1.1,1.,1.])
            if name in ('singleton','unnormalized_singleton'):
                ratios[0]=1.  # No clipping confound in the normalization comparison.
                with torch.no_grad(): t.actor.logs[0]=0.
            t.actor_optimizer=torch.optim.SGD(t.actor.parameters(),lr=0)
            t.critic_optimizer=torch.optim.SGD(t.critic.parameters(),lr=0)
            batch=hand_batch(counts,t.device)
            seed=123
            generator=torch.Generator(device=t.device).manual_seed(seed)
            order=torch.randperm(sum(counts),device=t.device,generator=generator).cpu().numpy()
            expected,adv_after,participation,per_batch=_reference(counts,mini,normalize,order,ratios)
            advantages,returns=gae(batch.rewards,batch.values,batch.next_values,batch.current_masks,batch.next_masks,
                                  batch.terminated,batch.truncated,config.gamma,config.gae_lambda)
            valid=batch.current_masks.bool()
            slots=torch.arange(len(counts),device=t.device).expand_as(batch.current_masks)[valid]
            before=advantages[valid]
            after=standardize_advantages(before,slots,per_agent=True) if normalize and len(before) else before
            torch.manual_seed(seed)
            result=t.update(batch)
            diag=result['numerical_diagnostics']
            actual={k:result[k] for k in ('policy_loss','actor_total_loss','critic_loss')}
            actual.update(actor_norm_mean=diag['actor_gradient_norm']['mean'] or 0.,critic_norm_mean=diag['critic_gradient_norm']['mean'] or 0.)
            for model,key,column in ((t.actor,'actor_last_clipped_norm',3),(t.critic,'critic_last_clipped_norm',4)):
                actual[key]=math.sqrt(sum(float(p.grad.detach().double().square().sum()) for p in model.parameters() if p.grad is not None))
                norm=per_batch[-1][column] if per_batch else 0.
                expected[key]=norm*min(1.,config.max_grad_norm/(norm+1e-6))
            rows.append(dict(name=name,counts=list(counts),minibatch=mini,normalized=normalize,actual=actual,expected=expected,
                             absolute_error={k:abs(actual[k]-expected[k]) for k in expected},
                             expected_advantage_after=adv_after.tolist(), expected_participation=participation,
                             actual_advantage_before=before.tolist(), actual_advantage_after=after.tolist(),
                             actual_returns=returns[valid].tolist(), ratios=ratios.tolist(),
                             expected_per_minibatch=per_batch, metrics=result,
                             sampled_steps=t.sampled_steps, updates=t.updates,
                             gae_definition='terminal rewards i*10+t+1, V=0, A=R=rewards on valid slots'))
        # Real Beta: singleton policy advantage vanishes, entropy gradient need not.
        t=ProtocolMAPPOTrainer(ProtocolTrainConfig(ppo_epochs=1))
        t.actor_optimizer=torch.optim.SGD(t.actor.parameters(),lr=0)
        t.critic_optimizer=torch.optim.SGD(t.critic.parameters(),lr=0)
        b=hand_batch((1,0,0,0,0),t.device)
        with torch.no_grad():
            b.actions[0,0,0]=t.actor.distribution(b.observations[0,0,:1]).mean[0]
            b.log_probs[0,0,0]=t.actor.evaluate(b.observations[0,0,:1],b.actions[0,0,:1])[0][0]
        real=t.update(b)
    return dict(source='artificial terminal arithmetic cases; not environment samples; SGD lr=0 for hand answers',
                cases=rows,real_beta_singleton=real)
