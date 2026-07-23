import numpy as np

from uav_env.envs import make_2v2_env


def rollout(seed: int):
    env=make_2v2_env("balanced_random","random",seed=seed); env.reset(seed=seed); result=[]
    for _ in range(4):
        _,reward,terminated,truncated,info=env.step(np.array([0,1])); result.append((reward,info["blue_actions"],info["global_state"].tolist()))
        if terminated or truncated: break
    return result


def test_multi_seed_determinism() -> None:
    assert rollout(9)==rollout(9)
    assert rollout(9)!=rollout(10)
