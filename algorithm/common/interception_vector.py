import numpy as np
from env.interception import InterceptionEnv
from .vector_env import ParallelVectorEnv


class EpisodeAccumulator:
    def reset(self, info):
        self.total = 0.
        self.effort = 0.
        self.parts = {}
        self.targets = np.flatnonzero(info["evader_mask"]).tolist()
        self.captured = []
        self.breaches = self.collisions = 0

    def add(self, info, rewards):
        self.total += float(rewards[info["active_mask"]][0])
        self.effort += info["control_effort"]
        for event in info["events"]:
            if event["kind"] == "capture":
                self.captured.append(event["entities"][1] - len(info["active_mask"]))
            elif event["kind"] == "breach":
                self.breaches += 1
            elif event["kind"] == "collision":
                self.collisions += 1
        for name, value in info["reward_parts"].items():
            self.parts[name] = self.parts.get(name, 0.) + value

    def summary(self, info):
        return dict(outcome=info["outcome"], time=info["time"], steps=info["step"],
                    minimum_separation=info["minimum_separation"], episode_return=self.total,
                    control_effort=self.effort, reward_parts=self.parts.copy(),
                    target_ids=self.targets.copy(), captured_ids=self.captured.copy(),
                    capture_count=len(self.captured), breach_count=self.breaches,
                    collision_count=self.collisions,
                    capture_time=info["time"] if info["outcome"] == "success" else None)


class InterceptionVectorEnv(ParallelVectorEnv):
    def __init__(self, config, num_envs, seed):
        super().__init__(config, num_envs, seed, environment_factory=InterceptionEnv,
                         accumulator_factory=EpisodeAccumulator)
