"""Shared actor/critic inference facade."""

from __future__ import annotations

import torch
from torch import Tensor
from uav_env.algorithms.mappo.distributions import sample_actions
from uav_env.algorithms.mappo.networks import CentralizedCritic, SharedActor


class MAPPOPolicy:
    def __init__(self, actor: SharedActor, critic: CentralizedCritic) -> None: self.actor,self.critic=actor,critic

    @torch.no_grad()
    def act(self, observations: Tensor, states: Tensor, available: Tensor, deterministic: bool=False) -> tuple[Tensor,Tensor,Tensor]:
        actions,log_probs,_=sample_actions(self.actor(observations,available),available,deterministic)
        return actions,log_probs,self.critic(states)
