"""Hand answers for independent 12/60 optimization, masking and identity."""
import dataclasses
import math

import numpy as np
import pytest
import torch
from torch import nn

from algorithm.common.checkpoint import (read_checkpoint, save_checkpoint, restore_checkpoint,
    read_protocol_checkpoint, save_protocol_checkpoint, restore_protocol_checkpoint)
from algorithm.mappo.config import TrainConfig, ProtocolTrainConfig
from algorithm.mappo.networks import Actor, ProtocolCritic
from algorithm.mappo.paper_protocol import pack_centralized_slots
from algorithm.mappo.trainer import MAPPOTrainer, ProtocolMAPPOTrainer, Rollout
from env.config import EnvConfig


def batch_for_hand_answer(device):
    # Agent 0 has TWO samples, agent 1 ONE; pooling must give a different answer.
    mask = torch.tensor([[[1., 1.]], [[1., 0.]]], device=device)
    obs = torch.zeros((2, 1, 2, 12), device=device)
    obs[..., 0] = mask
    obs[..., 1] = torch.tensor([[[200, 202]], [[201, 0]]], device=device)
    states = torch.as_tensor(pack_centralized_slots(obs.cpu().numpy(), mask.cpu().numpy()), device=device, dtype=torch.float32)
    values = torch.tensor([[[1., 2.]], [[1., float('nan')]]], device=device)
    rewards = torch.tensor([[[2., 0.]], [[4., float('nan')]]], device=device)
    actions = torch.full((2, 1, 2, 3), .5, device=device)
    return Rollout(obs, states, actions, torch.zeros_like(mask), values, torch.full_like(mask, float('nan')),
                   rewards, mask, mask, torch.ones((2, 1), device=device, dtype=torch.bool),
                   torch.zeros((2, 1), device=device, dtype=torch.bool))


class HandActor(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.logs = nn.Parameter(torch.tensor([1.4, .7, 1.1], device=device).log())

    def evaluate(self, obs, actions):
        ids = obs[:, 1].long()-200
        return self.logs[ids], torch.tensor([.1, .3, .9], device=obs.device)[ids]


class HandCritic(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.heads = nn.Parameter(torch.tensor([1., 2., 30., 40., 50.], device=device))

    def forward(self, state):
        return self.heads.expand(len(state), -1)


def test_per_individual_normalization_no_half_mse_and_gradient_hand_answers():
    trainer = ProtocolMAPPOTrainer(ProtocolTrainConfig(ppo_epochs=1, minibatch_size=3,
        normalize_advantages=False, entropy_coef=.1, max_grad_norm=100))
    trainer.actor, trainer.critic = HandActor(trainer.device), HandCritic(trainer.device)
    trainer.actor_optimizer = torch.optim.SGD(trainer.actor.parameters(), lr=0)
    trainer.critic_optimizer = torch.optim.SGD(trainer.critic.parameters(), lr=0)
    result = trainer.update(batch_for_hand_answer(trainer.device))
    # Agent0: mean(min(1.4*1,1.2*1), min(.7*3,.8*3)) = 1.65.
    # Agent1: min(1.1*(-2),1.1*(-2))=-2.2. Sum policy loss=+.55.
    assert result['policy_loss'] == pytest.approx(.55, abs=2e-7)
    assert result['entropy'] == pytest.approx(1.1, abs=2e-7)
    assert result['actor_total_loss'] == pytest.approx(.44, abs=2e-7)
    # Per-head MSE: mean(1,9)+4=9; historical pooled half-MSE is 7/3.
    assert result['critic_loss'] == pytest.approx(9)
    torch.testing.assert_close(trainer.critic.heads.grad, torch.tensor([-4.,4.,0.,0.,0.],device=trainer.device))
    torch.testing.assert_close(trainer.actor.logs.grad, torch.tensor([0.,-1.05,2.2],device=trainer.device))
    assert result['valid_samples'] == 3 and result['optimizer_steps'] == 1


def test_60d_packing_is_explicit_and_never_slices_90d():
    slots = np.zeros((2, 3, 12)); slots[..., 0] = 1; slots[..., 1] = 200
    states = pack_centralized_slots(slots, np.ones((2, 3)))
    np.testing.assert_array_equal(states[:, :36], slots.reshape(2, 36))
    assert states.shape == (2, 60) and not states[:, 36:].any()
    critic = ProtocolCritic()
    assert critic(torch.tensor(states, dtype=torch.float32)).shape == (2, 5)
    for bad in (torch.zeros(2, 90), torch.full((2, 60), float('nan'))):
        with pytest.raises(FloatingPointError): critic(bad)
    for bad in (np.zeros(90), np.zeros((6,12)), np.zeros((0,12))):
        with pytest.raises(ValueError): pack_centralized_slots(bad, np.zeros(bad.shape[:-1]))
    with pytest.raises(ValueError): pack_centralized_slots(slots, np.zeros((2,3)))


def test_beta_probability_of_exact_returned_action_and_masked_slots():
    t = ProtocolMAPPOTrainer()
    obs = np.zeros((3, 12),dtype=np.float32); obs[:2,0] = 1; obs[:2,1] = 200
    obs[2] = np.nan  # discarded slot must not enter a forward pass
    actions, logp = t.sample_actions(obs, [1,1,0])
    clean = t.tensor(obs[:2])
    dist = t.actor.distribution(clean)
    torch.testing.assert_close(logp[:2], dist.log_prob(actions[:2]).sum(-1))
    assert torch.all((actions[:2] > 0) & (actions[:2] < 1))
    assert not actions[2].any() and logp[2] == 0
    torch.testing.assert_close(t.actor.evaluate(clean, actions[:2])[1], dist.entropy().sum(-1))
    with pytest.raises(FloatingPointError): Actor().distribution(torch.zeros(1, 11))
    with pytest.raises(RuntimeError,match='no task/reward collector'): t.collect(object())


def test_external_actor_slots_must_agree_with_explicit_critic_slots():
    t = ProtocolMAPPOTrainer()
    b = batch_for_hand_answer(t.device)
    b.states[..., 1] += 1
    with pytest.raises(ValueError, match='disagree'): t.update(b)


def test_masked_data_cannot_change_either_optimizer_and_gradients_are_separate():
    c = ProtocolTrainConfig(ppo_epochs=1, minibatch_size=3, normalize_advantages=False)
    a,b = ProtocolMAPPOTrainer(c), ProtocolMAPPOTrainer(c)
    original = batch_for_hand_answer(a.device)
    modified = Rollout(**{f.name:getattr(original,f.name).clone() for f in dataclasses.fields(original)})
    invalid = ~modified.current_masks.bool()
    for name in ('observations','actions','log_probs','values','next_values','rewards'):
        getattr(modified,name)[invalid] = float('nan')
    modified.states[1,0,12:24] = float('nan')
    assert set(map(id,a.actor.parameters())).isdisjoint(set(map(id,a.critic.parameters())))
    torch.manual_seed(97); ma=a.update(original)
    torch.manual_seed(97); mb=b.update(modified)
    assert ma==mb
    for m1,m2 in ((a.actor,b.actor),(a.critic,b.critic)):
        for p1,p2 in zip(m1.parameters(),m2.parameters()): torch.testing.assert_close(p1,p2,rtol=0,atol=0)
    assert all(p.grad is None for head in a.critic.value_heads[2:] for p in head.parameters())


def test_head_with_existing_adam_momentum_is_skipped_when_agent_has_no_samples():
    t=ProtocolMAPPOTrainer(ProtocolTrainConfig(ppo_epochs=1,minibatch_size=3,normalize_advantages=False))

    def single_agent(slot):
        b=batch_for_hand_answer(t.device)
        data={f.name:getattr(b,f.name)[:1].clone() for f in dataclasses.fields(b)}
        data['current_masks'].zero_(); data['current_masks'][...,slot]=1
        data['next_masks']=data['current_masks'].clone()
        data['observations'][~data['current_masks'].bool()]=0
        data['states']=t.tensor(pack_centralized_slots(data['observations'].cpu().numpy(),data['current_masks'].cpu().numpy()))
        return Rollout(**data)

    t.update(single_agent(0))
    head=t.critic.value_heads[0]
    weights=[p.detach().clone() for p in head.parameters()]
    moments=[{k:v.clone() for k,v in t.critic_optimizer.state[p].items()} for p in head.parameters()]
    assert all(m['step']==1 for m in moments)
    t.update(single_agent(1))
    for p,w,m in zip(head.parameters(),weights,moments):
        assert p.grad is None
        torch.testing.assert_close(p,w,rtol=0,atol=0)
        for key in m: torch.testing.assert_close(t.critic_optimizer.state[p][key],m[key],rtol=0,atol=0)


def test_protocol_checkpoint_roundtrip_and_reject_cross_family_before_load(tmp_path, monkeypatch):
    c = ProtocolTrainConfig(ppo_epochs=1,minibatch_size=3,normalize_advantages=False)
    t = ProtocolMAPPOTrainer(c)
    batch = batch_for_hand_answer(t.device)
    t.update(batch)
    path=tmp_path/'array.pt'; save_protocol_checkpoint(path,t)
    payload=read_protocol_checkpoint(path)
    t.update(batch)
    clone=ProtocolMAPPOTrainer(c); restore_protocol_checkpoint(payload,clone); clone.update(batch)
    for m1,m2 in ((t.actor,clone.actor),(t.critic,clone.critic)):
        for p1,p2 in zip(m1.parameters(),m2.parameters()): torch.testing.assert_close(p1,p2,rtol=0,atol=0)
    for o1,o2 in ((t.actor_optimizer,clone.actor_optimizer),(t.critic_optimizer,clone.critic_optimizer)):
        for s1,s2 in zip(o1.state.values(),o2.state.values()):
            for key in s1: torch.testing.assert_close(s1[key],s2[key],rtol=0,atol=0)
    inspection=MAPPOTrainer(TrainConfig()); p90=tmp_path/'inspection.pt'
    save_checkpoint(p90,inspection,EnvConfig())
    with pytest.raises(ValueError,match='inspection trainer'): save_checkpoint(tmp_path/'wrong.pt',clone,EnvConfig())
    with pytest.raises(ValueError): restore_protocol_checkpoint(payload,inspection)
    old=tmp_path/'v2.pt'; torch.save(dict(environment_version='coop3d-v2.0.0',algorithm_version='mappo-coop3d-v2.0.0'),old)
    # Protocol/identity rejection must occur BEFORE pickle deserialization.
    def forbidden(*args,**kwargs): raise AssertionError('torch.load must not be reached')
    monkeypatch.setattr(torch,'load',forbidden)
    for p in (p90,old):
        with pytest.raises(ValueError,match='incompatible checkpoint'): read_protocol_checkpoint(p)
    for p in (path,old):
        with pytest.raises(ValueError,match='incompatible checkpoint'): read_checkpoint(p)


def test_published_seed_rule_and_independent_configuration():
    from algorithm.mappo.evaluation import evaluate
    c=ProtocolTrainConfig(training_seed_index=4)
    assert c.worker_seeds()==tuple(range(4000,4010))
    assert c.horizon==1500 and c.checkpoint_interval_episodes==50 and c.evaluation_episodes==200
    with pytest.raises(TypeError): ProtocolMAPPOTrainer(TrainConfig())
    with pytest.raises(TypeError): MAPPOTrainer(c)
    with pytest.raises(ValueError,match='no task adapter'):
        evaluate(ProtocolMAPPOTrainer(c),EnvConfig(),[10000000])
