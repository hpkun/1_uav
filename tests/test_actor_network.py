import pytest,torch
from uav_env.algorithms.mappo.networks import SharedActor

def test_actor_shape_mask_and_dead_action():
 actor=SharedActor(11); obs=torch.zeros(3,11); mask=torch.ones(3,15,dtype=torch.bool);mask[0]=False;mask[0,0]=True
 logits=actor(obs,mask);assert logits.shape==(3,15) and torch.argmax(logits[0])==0
 with pytest.raises(ValueError):actor(obs,torch.zeros_like(mask))

