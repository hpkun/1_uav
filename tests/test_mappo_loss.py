import torch
from uav_env.algorithms.mappo.trainer import huber_loss

def test_huber_hand_calculation():
 e=torch.tensor([1.,3.]);assert torch.allclose(huber_loss(e,2),torch.tensor([.5,4.]))

