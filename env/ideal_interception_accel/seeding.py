"""Shared external scenario-seed convention for evaluation and baselines."""
import numpy as np

from .environment import InterceptionEnv


def scenario_seed(seed):
    if type(seed) is not int or seed < 0:
        raise ValueError("scenario seed must be a nonnegative integer")
    return np.random.SeedSequence(seed)


def reset_evaluation_scene(config, seed):
    env=InterceptionEnv(config)
    observation,info=env.reset(scenario_seed(seed))
    return env,observation,info
