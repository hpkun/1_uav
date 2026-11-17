"""Acceleration-limited experimental clone of env.ideal_interception.

The original ideal environment is intentionally left unchanged.
"""
from copy import deepcopy
from dataclasses import asdict
import numpy as np

from .config import InterceptionConfig, PROTOCOL,DEFENDER_SLOTS,INTRUDER_SLOTS
from .events import advance
from .observation import observe
from .scenario import initialize, diagnostics


class InterceptionEnv:
    def __init__(self, config: InterceptionConfig):
        self.config = config
        self.outcome = "uninitialized"

    def reset(self, seed=None):
        c = self.config
        (self.defenders,self.intruders,self.paths,self.assignment,
         self.seed,self.scenario_id) = initialize(c,seed)
        self.scene_diagnostics = diagnostics(c,self.paths)
        self.defender_mask = np.arange(DEFENDER_SLOTS)<c.defender_count
        self.intruder_mask = np.arange(INTRUDER_SLOTS)<c.intruder_count
        self.original_assignment = self.assignment.copy()
        self.per_pair_initial_distance_m = {i:float(np.linalg.norm(
            self.defenders[i,:3]-self.intruders[j,:3])) for i,j in self.assignment.items()}
        self.assignment_total_cost_m = sum(self.per_pair_initial_distance_m.values())
        self.defender_status = ["active" if flag else "unused" for flag in self.defender_mask]
        self.intruder_status = ["active" if flag else "unused" for flag in self.intruder_mask]
        self.time,self.step_count,self.outcome = 0.,0,"ongoing"
        self.events = []
        self.minimum_separation = None
        self.interception_count = 0
        self.friendly_collision_count = self.intruder_collision_count = self.breach_count = 0
        self.defender_boundary_count = 0
        self.last_heading = np.zeros((DEFENDER_SLOTS,3))
        self.last_heading[self.defender_mask] = self.defenders[self.defender_mask,3:] / np.asarray(
            c.defender_speeds)[:c.defender_count,None]
        self.acceleration_limited_steps_total = 0
        self.acceleration_commands_total = 0
        self.maximum_commanded_acceleration_mps2 = 0.
        self.maximum_executed_acceleration_mps2 = 0.
        return self.output(include_paths=True)

    def output(self,events=(),parts=None,team_reward=0.,include_paths=False):
        obs,central = observe(self)
        info = dict(active_mask=self.defender_mask.copy(),centralized_state=central,
            defender_mask=self.defender_mask.copy(),intruder_mask=self.intruder_mask.copy(),
            defenders=self.defenders.copy(),intruders=self.intruders.copy(),
            defender_status=self.defender_status.copy(),intruder_status=self.intruder_status.copy(),
            assignment=self.assignment.copy(),events=list(events),reward_parts=dict(parts or {}),
            original_assignment=self.original_assignment.copy(),
            assignment_total_cost_m=self.assignment_total_cost_m,
            per_pair_initial_distance_m=self.per_pair_initial_distance_m.copy(),
            team_reward=team_reward,time=self.time,step=self.step_count,outcome=self.outcome,
            scenario_id=self.scenario_id,seed=self.seed,scale=self.config.scale,
            interception_count=self.interception_count,breach_count=self.breach_count,
            friendly_collision_count=self.friendly_collision_count,
            intruder_collision_count=self.intruder_collision_count,
            initial_intruder_count=self.config.intruder_count,
            intercepted_count=self.interception_count,breached_count=self.breach_count,
            other_resolved_count=sum(status not in ("active","unused","intercepted","breached")
                                     for status in self.intruder_status),
            remaining_intruder_count=int(self.intruder_mask.sum()),
            defender_boundary_count=self.defender_boundary_count,
            active_defenders_remaining=int(self.defender_mask.sum()),
            mission_success=self.outcome=="all_intercepted",
            interception_ratio=self.interception_count/self.config.intruder_count,
            breach_ratio=self.breach_count/self.config.intruder_count,
            minimum_separation=self.minimum_separation,
            defender_acceleration_max_mps2=self.config.defender_acceleration_max,
            acceleration_limited_steps_total=self.acceleration_limited_steps_total,
            acceleration_limited_fraction=(self.acceleration_limited_steps_total /
                self.acceleration_commands_total if self.acceleration_commands_total else 0.),
            maximum_commanded_acceleration_mps2=self.maximum_commanded_acceleration_mps2,
            maximum_executed_acceleration_mps2=self.maximum_executed_acceleration_mps2,
            action_semantics="defender_desired_motion_direction")
        if include_paths:
            info["intruder_paths"] = [path.specification() for path in self.paths]
            info["intruder_plans"] = [getattr(path,"plan_metadata",None) for path in self.paths]
            info["scene_diagnostics"] = self.scene_diagnostics
        return obs,info

    step = advance

    def snapshot(self):
        state = {key:deepcopy(value) for key,value in self.__dict__.items() if key!="config"}
        return dict(protocol=PROTOCOL,configuration=asdict(self.config),state=state)

    def restore(self,snapshot):
        if snapshot.get("protocol")!=PROTOCOL or snapshot.get("configuration")!=asdict(self.config):
            raise ValueError("incompatible ideal interception snapshot")
        self.__dict__.update(deepcopy(snapshot["state"]))
