import torch
from uav_env.algorithms.mappo.value_normalizer import ValueNormalizer

def test_value_normalizer_roundtrip_and_state():
 n=ValueNormalizer();x=torch.tensor([1.,2.,3.]);n.update(x);assert torch.allclose(n.denormalize(n.normalize(x)),x,atol=1e-5)
 m=ValueNormalizer();m.load_state_dict(n.state_dict());assert torch.allclose(m.normalize(x),n.normalize(x))

