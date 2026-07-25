import numpy as np
from uav_env.algorithms.mappo.returns import compute_gae

def test_terminated_and_truncated_bootstrap():
 r=np.ones((1,2,1),np.float32);v=np.array([[[2.],[2.]],[[9.],[9.]]],np.float32);term=np.array([[True,False]]);trunc=np.array([[False,True]]);tv=np.array([[[0.],[4.]]],np.float32)
 a,ret=compute_gae(r,v,term,trunc,tv,1,1);assert a[0,0,0]==-1 and a[0,1,0]==3 and ret[0,1,0]==5

def test_three_step_gamma_lambda_one():
 r=np.ones((3,1,1),np.float32);v=np.zeros((4,1,1),np.float32);done=np.zeros((3,1),bool);a,_=compute_gae(r,v,done,done,np.zeros_like(r),1,1);assert np.array_equal(a[:,0,0],[3,2,1])


def test_gae_uses_physical_reward_and_value_scale():
 rewards=np.array([[[25.0]]],np.float32)
 values=np.array([[[100.0]],[[120.0]]],np.float32)
 boundaries=np.zeros((1,1),bool)
 advantages,returns=compute_gae(rewards,values,boundaries,boundaries,np.zeros_like(rewards),.5,1.)
 assert advantages[0,0,0]==-15.0
 assert returns[0,0,0]==85.0
