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
        self.initial = {key: info[key].tolist() for key in ("pursuers", "evaders", "active_mask", "evader_mask")}
        self.capture_times, self.breach_times, self.escape_times = {}, {}, {}
        self.collision_types = {"PP": 0, "PE": 0, "EE": 0}
        self.near_misses = []
        self.capture_owners = {}

    def add(self, info, rewards):
        self.total += float(rewards[info["active_mask"]][0])
        self.effort += info["control_effort"]
        for event in info["events"]:
            if event["kind"] == "capture":
                pursuer, target = event["entities"]
                target -= len(info["active_mask"])
                self.captured.append(target)
                self.capture_times[str(target)] = event["time"]
                self.capture_owners[str(target)] = pursuer
            elif event["kind"] == "breach":
                self.breaches += 1
                self.breach_times[str(event["entities"][0]-len(info["active_mask"]))] = event["time"]
            elif event["kind"] == "collision":
                self.collisions += 1
                key = "".join(t[0].upper() for t in event["entity_types"])
                self.collision_types[key] += 1
            elif event["kind"] == "evader_escape":
                self.escape_times[str(event["entities"][0]-len(info["active_mask"]))] = event["time"]
            elif event["kind"] == "near_miss":
                self.near_misses.append(dict(time=event["time"], pair_ids=event["entities"]))
        for name, value in info["reward_parts"].items():
            self.parts[name] = self.parts.get(name, 0.) + value

    def summary(self, info):
        return dict(outcome=info["outcome"], time=info["time"], steps=info["step"],
                    minimum_separation=info["minimum_separation"], episode_return=self.total,
                    control_effort=self.effort, reward_parts=self.parts.copy(),
                    target_ids=self.targets.copy(), captured_ids=self.captured.copy(),
                    capture_count=len(self.captured), breach_count=self.breaches,
                    collision_count=self.collisions, collision_types=self.collision_types.copy(),
                    minimum_center_distance=info["minimum_center_distance"],
                    minimum_surface_clearance=info["minimum_surface_clearance"],
                    capture_times=self.capture_times.copy(), breach_times=self.breach_times.copy(),
                    escape_times=self.escape_times.copy(), capture_owners=self.capture_owners.copy(),
                    participation_seconds=info["participation_seconds"].tolist(),
                    near_miss_count=len(self.near_misses), near_misses=self.near_misses.copy(),
                    initial=self.initial, task=info["task"], initialization=info["initialization"],
                    capture_time=info["time"] if info["outcome"] == "success" else None)


class InterceptionVectorEnv(ParallelVectorEnv):
    def __init__(self, config, num_envs, seed):
        super().__init__(config, num_envs, seed, environment_factory=InterceptionEnv,
                         accumulator_factory=EpisodeAccumulator)
