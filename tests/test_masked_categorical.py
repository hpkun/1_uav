import torch
from uav_env.algorithms.mappo.distributions import masked_categorical

def test_masked_probability_zero():
 mask=torch.tensor([[1,0,1]],dtype=torch.bool);d=masked_categorical(torch.zeros(1,3),mask);assert d.probs[0,1]==0

