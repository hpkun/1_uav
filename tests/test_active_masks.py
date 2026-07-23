import torch

def test_actor_mask_excludes_dead_sample():
 losses=torch.tensor([2.,100.]);mask=torch.tensor([1.,0.]);assert (losses*mask).sum()/mask.sum()==2

