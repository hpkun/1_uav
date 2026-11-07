import numpy as np
from env.interception import InterceptionEnv
from .vector_env import ParallelVectorEnv


class EpisodeAccumulator:
    def reset(self, info):
        self.total = 0.
        self.effort = 0.
        self.parts = {}

    def add(self, info, rewards):
        self.total += float(rewards[info["active_mask"]][0])
        self.effort += info["control_effort"]
        for name, value in info["reward_parts"].items():
            self.parts[name] = self.parts.get(name, 0.) + value

    def summary(self, info):
        return dict(outcome=info["outcome"], time=info["time"], steps=info["step"],
                    minimum_separation=info["minimum_separation"], episode_return=self.total,
                    control_effort=self.effort, reward_parts=self.parts.copy())


class InterceptionVectorEnv(ParallelVectorEnv):
    def __init__(self, config, num_envs, seed):
        super().__init__(config, num_envs, seed, environment_factory=InterceptionEnv,
                         accumulator_factory=EpisodeAccumulator)
