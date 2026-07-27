import numpy as np
import pytest
from uav_env.algorithms.mappo.returns import compute_gae

def test_terminated_and_truncated_bootstrap():
 r=np.ones((1,2,1),np.float32);v=np.array([[[2.],[2.]],[[9.],[9.]]],np.float32);term=np.array([[True,False]]);trunc=np.array([[False,True]]);tv=np.array([[[0.],[4.]]],np.float32)
 mask=np.array([[0.,1.]],np.float32)
 a,ret=compute_gae(r,v,term,trunc,tv,mask,1,1);assert a[0,0,0]==-1 and a[0,1,0]==3 and ret[0,1,0]==5

def test_three_step_gamma_lambda_one():
 r=np.ones((3,1,1),np.float32);v=np.zeros((4,1,1),np.float32);done=np.zeros((3,1),bool);a,_=compute_gae(r,v,done,done,np.zeros_like(r),np.zeros((3,1),np.float32),1,1);assert np.array_equal(a[:,0,0],[3,2,1])


def test_gae_uses_physical_reward_and_value_scale():
 rewards=np.array([[[25.0]]],np.float32)
 values=np.array([[[100.0]],[[120.0]]],np.float32)
 boundaries=np.zeros((1,1),bool)
 advantages,returns=compute_gae(rewards,values,boundaries,boundaries,np.zeros_like(rewards),np.zeros((1,1),np.float32),.5,1.)
 assert advantages[0,0,0]==-15.0
 assert returns[0,0,0]==85.0


def test_timeaware_timeout_mask_blocks_bootstrap_and_legacy_mask_allows_it():
 rewards=np.array([[[1.0]]],np.float32)
 values=np.array([[[2.0]],[[99.0]]],np.float32)
 terminated=np.array([[False]])
 truncated=np.array([[True]])
 terminal_values=np.array([[[10.0]]],np.float32)
 no_bootstrap,_=compute_gae(rewards,values,terminated,truncated,terminal_values,np.array([[0.0]],np.float32),0.99,1.0)
 legacy_bootstrap,_=compute_gae(rewards,values,terminated,truncated,terminal_values,np.array([[1.0]],np.float32),0.99,1.0)
 assert no_bootstrap[0,0,0]==-1.0
 assert legacy_bootstrap[0,0,0]==pytest.approx(8.9)
