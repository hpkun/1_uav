import torch
from torch import nn
from torch.distributions import Beta
from torch.nn import functional as F


def mlp(input_dim, output_dim):
    return nn.Sequential(nn.Linear(input_dim, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, output_dim))


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = mlp(12, 6)

    def distribution(self, observations):
        parameters = F.softplus(self.network(observations)) + 1.0
        return Beta(parameters[..., :3], parameters[..., 3:])

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
        self.network = mlp(60, 1)

    def forward(self, states):
        return self.network(states).squeeze(-1)
