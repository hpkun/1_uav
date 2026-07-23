import numpy as np
from uav_env.algorithms.mappo.adapter import MAPPOEnvAdapter,SyncCombatVectorEnv
from uav_env.envs import make_1v1_env

def test_vector_shapes_and_determinism():
 f=lambda:MAPPOEnvAdapter(make_1v1_env(seed=3)); a=SyncCombatVectorEnv([f,f],3); b=SyncCombatVectorEnv([f,f],3)
 x=a.reset();y=b.reset();assert np.array_equal(x["local_obs"],y["local_obs"])
 r=a.step(np.zeros((2,1),np.int64));assert r["next_local_obs"].shape==(2,1,11)

