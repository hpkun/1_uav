import numpy as np
import pytest
from uav_env.algorithms.mappo.returns import compute_gae
from uav_env.algorithms.mappo.rollout_buffer import RolloutBuffer

def test_truncation_uses_terminal_not_reset_value():
 r=np.zeros((1,1,1),np.float32);v=np.array([[[1]],[[99]]],np.float32);d=np.array([[False]]);t=np.array([[True]]);tv=np.array([[[3]]],np.float32);mask=np.array([[1]],np.float32);a,_=compute_gae(r,v,d,t,tv,mask,1,1);assert a.item()==2


def test_rollout_buffer_truncation_bootstrap_mask_controls_next_values():
 b=RolloutBuffer(1,2,1,1,1)
 b.set_initial(np.zeros((2,1,1),np.float32),np.zeros((2,1),np.float32),np.ones((2,1,15),bool))
 b.insert(
  np.zeros((2,1),np.int64),np.zeros((2,1),np.float32),np.array([[2.0],[2.0]],np.float32),
  np.ones((2,1),np.float32),np.array([False,False]),np.array([True,True]),
  np.ones((2,1),np.float32),np.ones((2,1),np.float32),
  np.zeros((2,1,1),np.float32),np.zeros((2,1),np.float32),np.ones((2,1,15),bool),
  np.array([[10.0],[10.0]],np.float32),np.array([0.0,1.0],np.float32),
 )
 b.finish(np.array([[99.0],[99.0]],np.float32),0.99,1.0)
 assert b.next_values[0,0,0]==0.0
 assert b.next_values[0,1,0]==10.0
 assert b.advantages[0,0,0]==-1.0
 assert b.advantages[0,1,0]==pytest.approx(8.9)
