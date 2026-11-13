"""Bilateral abstract point-mass interception environment."""
from copy import deepcopy
from dataclasses import asdict
import numpy as np

from .config import InterceptionConfig, PROTOCOL
from .events import advance
from .observation import observe
from .scenario import initialize


class InterceptionEnv:
    def __init__(self, config: InterceptionConfig):
        self.config = config
        self.outcome = "uninitialized"

    def reset(self, seed=None):
        c = self.config
        (self.defenders,self.intruders,self.paths,self.assignment,
         self.seed,self.scenario_id) = initialize(c,seed)
        self.defender_mask = np.arange(5)<c.defender_count
        self.intruder_mask = np.arange(2)<c.intruder_count
        self.defender_status = ["active" if flag else "unused" for flag in self.defender_mask]
        self.intruder_status = ["active" if flag else "unused" for flag in self.intruder_mask]
        self.time,self.step_count,self.outcome = 0.,0,"ongoing"
        self.events = []
        self.minimum_separation = None
        self.interception_count = 0
        self.friendly_collision_count = self.intruder_collision_count = self.breach_count = 0
        return self.output(include_paths=True)

    def output(self,events=(),parts=None,team_reward=0.,include_paths=False):
        obs,central = observe(self)
        info = dict(active_mask=self.defender_mask.copy(),centralized_state=central,
            defender_mask=self.defender_mask.copy(),intruder_mask=self.intruder_mask.copy(),
            defenders=self.defenders.copy(),intruders=self.intruders.copy(),
            defender_status=self.defender_status.copy(),intruder_status=self.intruder_status.copy(),
            assignment=self.assignment.copy(),events=list(events),reward_parts=dict(parts or {}),
            team_reward=team_reward,time=self.time,step=self.step_count,outcome=self.outcome,
            scenario_id=self.scenario_id,seed=self.seed,scale=self.config.scale,
            interception_count=self.interception_count,breach_count=self.breach_count,
            friendly_collision_count=self.friendly_collision_count,
            intruder_collision_count=self.intruder_collision_count,
            minimum_separation=self.minimum_separation,
            action_semantics="defender_desired_motion_direction")
        if include_paths:
            info["intruder_paths"] = [path.specification() for path in self.paths]
        return obs,info

    step = advance

    def snapshot(self):
        state = {key:deepcopy(value) for key,value in self.__dict__.items() if key!="config"}
        return dict(protocol=PROTOCOL,configuration=asdict(self.config),state=state)

    def restore(self,snapshot):
        if snapshot.get("protocol")!=PROTOCOL or snapshot.get("configuration")!=asdict(self.config):
            raise ValueError("incompatible ideal interception snapshot")
        self.__dict__.update(deepcopy(snapshot["state"]))
