import numpy as np
from uav_env.envs import make_2v2_env

def test_multi_saturation_diagnostics_shapes():
 env=make_2v2_env(seed=1);_,info=env.reset(seed=1)
 assert info["local_observation_saturation_count"].shape==(2,)
 assert info["local_observation_saturation_ratio"].shape==(2,)
 assert np.isfinite(info["global_state_saturation_ratio"])

