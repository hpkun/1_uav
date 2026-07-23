import torch
from uav_env.algorithms.mappo.networks import CentralizedCritic

def test_critic_agent_values_shape():
 critic=CentralizedCritic(40,2); values=critic(torch.zeros(4,40));assert values.shape==(4,2)
 assert not torch.equal(values[:,0],values[:,1])

