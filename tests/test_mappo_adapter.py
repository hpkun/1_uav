import numpy as np
from uav_env.algorithms.mappo.adapter import MAPPOEnvAdapter
from uav_env.envs import make_1v1_env,make_2v2_env

def test_adapter_shapes():
 a=MAPPOEnvAdapter(make_1v1_env(seed=1)); s=a.reset(1); assert s.local_obs.shape==(1,11) and s.global_state.shape==(10,) and s.available_action_mask.shape==(1,15)
 b=MAPPOEnvAdapter(make_2v2_env(seed=1)); t=b.reset(1); assert t.local_obs.shape==(2,28) and t.global_state.shape==(40,) and t.available_action_mask.shape==(2,15)

