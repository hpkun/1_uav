import pytest
import torch
from uav_env.algorithms.mappo.value_normalizer import ValueNormalizer
from uav_env.algorithms.mappo.trainer import explained_variance,ppo_value_loss,value_loss_inputs

def test_value_normalizer_roundtrip_and_state():
 n=ValueNormalizer();x=torch.tensor([1.,2.,3.]);n.update(x);assert torch.allclose(n.denormalize(n.normalize(x)),x,atol=1e-5)
 m=ValueNormalizer();m.load_state_dict(n.state_dict());assert torch.allclose(m.normalize(x),n.normalize(x))


def test_value_loss_triplet_uses_physical_scale_when_disabled():
 n=ValueNormalizer();new=torch.tensor([10.]);old=torch.tensor([8.]);target=torch.tensor([12.])
 actual=value_loss_inputs(new,old,target,n,False)
 assert all(torch.equal(a,b) for a,b in zip(actual,(new,old,target)))


def test_normalizer_update_keeps_unchanged_new_old_equal_and_does_not_false_clip():
 n=ValueNormalizer();old_physical=torch.tensor([10.,20.]);target=torch.tensor([100.,200.]);n.update(target)
 new,old,normalized_target=value_loss_inputs(old_physical.clone(),old_physical,target,n,True)
 assert torch.equal(new,old)
 assert torch.equal(new,n.normalize(old_physical))
 assert torch.equal(old,n.normalize(old_physical))
 assert torch.equal(normalized_target,n.normalize(target))
 clipped=ppo_value_loss(new,old,normalized_target,torch.ones(2),.2,True,False,1.)
 unclipped=ppo_value_loss(new,old,normalized_target,torch.ones(2),.2,False,False,1.)
 assert clipped==pytest.approx(float(unclipped))
 n.update(torch.tensor([-500.,700.]))
 new2,old2,_=value_loss_inputs(old_physical.clone(),old_physical,target,n,True)
 assert torch.equal(new2,old2)


def test_explained_variance_is_computed_directly_in_physical_space():
 prediction=torch.tensor([10.,20.,30.]);target=torch.tensor([12.,22.,32.])
 assert explained_variance(prediction,target)==pytest.approx(1.0)
