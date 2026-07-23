import numpy as np
from uav_env.algorithms.mappo.returns import compute_gae

def test_truncation_uses_terminal_not_reset_value():
 r=np.zeros((1,1,1),np.float32);v=np.array([[[1]],[[99]]],np.float32);d=np.array([[False]]);t=np.array([[True]]);tv=np.array([[[3]]],np.float32);a,_=compute_gae(r,v,d,t,tv,1,1);assert a.item()==2

