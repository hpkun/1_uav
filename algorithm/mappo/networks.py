"""Shared 12D Beta actor; separate 90D inspection and 60D protocol critics."""
import torch
from torch import nn
from torch.distributions import Beta
from torch.nn import functional as F

TRIM_ACTION = (0.5, 1/3, 1/8)


def mlp(input_dim, output_dim):
    return nn.Sequential(nn.Linear(input_dim,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,output_dim))


class Actor(nn.Module):
    def __init__(self, initialization_mode='trim_centered', concentration=16.0, output_weight_gain=0.01):
        super().__init__()
        self.network = mlp(12,6)
        if initialization_mode == 'trim_centered':
            if concentration <= 8 or not torch.isfinite(torch.tensor(concentration)):
                raise ValueError('concentration must exceed 8')
            center = torch.tensor(TRIM_ACTION)
            parameters = torch.cat((concentration*center, concentration*(1-center)))
            # Inverse softplus(p-1), stable even at large concentration.
            inverse = (parameters-1) + torch.log(-torch.expm1(-(parameters-1)))
            nn.init.orthogonal_(self.network[-1].weight, gain=output_weight_gain)
            with torch.no_grad():
                self.network[-1].bias.copy_(inverse)
        elif initialization_mode != 'legacy':
            raise ValueError('unknown actor initialization')

    def distribution(self, observations):
        if observations.shape[-1] != 12 or not torch.isfinite(observations).all():
            raise FloatingPointError('actor requires finite 12D valid observations')
        parameters = F.softplus(self.network(observations)) + 1
        return Beta(parameters[...,:3],parameters[...,3:])

    def act(self, observations, deterministic=False):
        dist = self.distribution(observations)
        actions = dist.mean if deterministic else dist.sample()
        return actions, dist.log_prob(actions).sum(-1)

    def evaluate(self, observations, actions):
        dist = self.distribution(observations)
        return dist.log_prob(actions).sum(-1), dist.entropy().sum(-1)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = mlp(90,5)

    def forward(self, states):
        if states.shape[-1] != 90 or not torch.isfinite(states).all():
            raise FloatingPointError('critic requires finite 90D global state')
        return self.network(states)


class ProtocolCritic(nn.Module):
    """Table III 60->128 Tanh->128 Tanh, five individual value heads.

    Five heads/shared trunk are an engineering assumption, not published
    output-head topology. No waypoint features and no 90D slicing adapter.
    """
    def __init__(self):
        super().__init__()
        self.network = mlp(60, 5)[:-1]
        # Separate Parameters let Adam skip an absent head (grad=None),
        # including its momentum/step counter, rather than merely zeroing grad.
        self.value_heads = nn.ModuleList(nn.Linear(128, 1) for _ in range(5))

    def forward(self, states):
        if states.shape[-1] != 60 or not torch.isfinite(states).all():
            raise FloatingPointError('protocol critic requires finite 60D state')
        features = self.network(states)
        return torch.cat([head(features) for head in self.value_heads], dim=-1)
