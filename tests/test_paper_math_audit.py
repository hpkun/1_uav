"""Known-answer PPO objective and per-UAV value indexing, without environment steps."""
import math

import pytest
import torch
from torch import nn

from algorithm.mappo.config import TrainConfig
from algorithm.mappo.trainer import MAPPOTrainer, Rollout


class FixedLogActor(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.logs = nn.Parameter(torch.tensor([1.4,.7,1.1], device=device).log())

    def evaluate(self, obs, actions):
        ids = obs[:,0].long()
        return self.logs[ids], torch.tensor([.1,.2,.3],device=obs.device)[ids]


class FixedHeadCritic(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.heads = nn.Parameter(torch.tensor([1.,2.,3.,40.,50.],device=device))

    def forward(self, states):
        return self.heads.expand(len(states),-1)


@pytest.mark.parametrize('normalize', [False,True])
def test_actual_update_matches_hand_calculated_clipping_and_individual_values(normalize):
    trainer=MAPPOTrainer(TrainConfig(ppo_epochs=1,minibatch_size=3,normalize_advantages=normalize,entropy_coef=.1))
    device=trainer.device
    trainer.actor=FixedLogActor(device); trainer.critic=FixedHeadCritic(device)
    # Zero learning rate isolates the measured objective; still runs real backward/update.
    trainer.actor_optimizer=torch.optim.SGD(trainer.actor.parameters(),lr=0)
    trainer.critic_optimizer=torch.optim.SGD(trainer.critic.parameters(),lr=0)
    obs=torch.zeros((1,1,3,12),device=device); obs[0,0,:,0]=torch.arange(3,device=device)
    values=torch.tensor([[[1.,2.,3.]]],device=device)
    rewards=torch.tensor([[[-1.,3.,6.]]],device=device)
    masks=torch.ones_like(values)
    batch=Rollout(obs,torch.zeros((1,1,90),device=device),torch.full((1,1,3,3),.5,device=device),
                  torch.zeros_like(values),values,torch.zeros_like(values),rewards,masks,masks,
                  torch.ones((1,1),device=device,dtype=torch.bool),torch.zeros((1,1),device=device,dtype=torch.bool))
    result=trainer.update(batch)
    # A=(-2,1,3). Selected min branches: 1.4*(-2), .7*1, 1.1*3.
    # Population-normalized A=(-8,1,7)/sqrt(38); branch signs are unchanged.
    expected_policy=-( -11.2+.7+7.7 )/(3*(math.sqrt(38)+3e-8)) if normalize else -.4
    assert result['policy_loss']==pytest.approx(expected_policy,abs=2e-7)
    assert result['actor_total_loss']==pytest.approx(expected_policy-.02,abs=2e-7)
    assert result['critic_loss']==pytest.approx(7/3,abs=2e-7)
    assert result['clip_fraction']==pytest.approx(2/3)
    assert result['valid_samples']==3 and result['optimizer_steps']==1
    assert not trainer.critic.heads.grad[3:].any()
