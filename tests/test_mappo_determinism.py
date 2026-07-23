import torch
from uav_env.algorithms.mappo.networks import SharedActor

def test_same_seed_network_identical():
 torch.manual_seed(7);a=SharedActor(11);torch.manual_seed(7);b=SharedActor(11);assert all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters()))

