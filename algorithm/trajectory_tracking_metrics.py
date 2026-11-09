"""Per-episode tracking statistics shared by workers and independent evaluation."""
from statistics import mean

import numpy as np


class TrackingAccumulator:
    def reset(self, info):
        self.initial = info["actual_states"].tolist()
        self.paths = info["reference_paths"]
        self.seed = info["seed"]
        self.rewards = {}
        self.total = 0.
        self.errors = []
        self.phase_error = self.phase_time = self.length_normalized = 0.
        self.elapsed = self.position = self.velocity = 0.
        self.members = [dict(reference_end_error=0. if path["duration"]==0 else None,
                             hold_end_error=None,last_endpoint_error=None,
                             actual_length=0.,completion_time=None,speed_projection=0,
                             vertical_limit=0,horizontal_limit=0) for path in self.paths]

    def add(self, info, rewards):
        self.total += info["team_reward"]
        for key,value in info["reward_parts"].items():
            self.rewards[key] = self.rewards.get(key,0.)+value
        for sample in info["samples"]:
            i,dt = sample["id"],sample["duration"]
            p,v = sample["position_error"],sample["velocity_error"]
            self.errors.append(p)
            self.elapsed += dt
            self.position += p*dt
            self.velocity += v*dt
            phase = sample["phase_duration"]
            pe = sample["reference_end_error"] if sample["reference_end_error"] is not None else p
            self.phase_error += pe*phase
            length = self.paths[i]["length"]
            self.phase_time += phase
            self.length_normalized += pe*phase/length if length>0 else 0.
            member = self.members[i]
            for key in ("speed_projection","vertical_limit","horizontal_limit"):
                member[key] += sample[key]
            member["actual_length"] += sample["distance"]
            member["last_endpoint_error"] = sample["endpoint_error"]
            if sample["reference_end_error"] is not None:
                member["reference_end_error"] = sample["reference_end_error"]
            if info["status"][i] in ("completed","missed"):
                member["hold_end_error"] = sample["endpoint_error"]
            if sample["completion_time"] is not None:
                member["completion_time"] = sample["completion_time"]

    def summary(self, info):
        def ratio(n,d):
            return n/d if d else None
        members = [member | dict(reference_length=path["length"],reference_duration=path["duration"],
                   status=info["status"][i]) for i,(member,path) in enumerate(zip(self.members,self.paths))]
        return dict(outcome=info["outcome"],time=info["time"],steps=info["step"],
            initial_states=self.initial,reset_seed=self.seed,reference_paths=self.paths,members=members,
            team_return=self.total,reward_parts=self.rewards.copy(),
            mean_position_error=ratio(self.position,self.elapsed),mean_velocity_error=ratio(self.velocity,self.elapsed),
            position_error_percentiles=(dict(zip(("p50","p90","p99"),np.percentile(self.errors,[50,90,99]).tolist()))
                                        if self.errors else None),
            phase_time_normalized_error=ratio(self.phase_error,self.phase_time),
            path_normalized_error=ratio(self.length_normalized,self.phase_time))


def summarize(rows):
    def average(values):
        values = [v for v in values if v is not None]
        return mean(values) if values else None
    keys = ("mean_position_error","mean_velocity_error","phase_time_normalized_error","path_normalized_error","team_return")
    members = [member for row in rows for member in row["members"]]
    n = len(rows)
    return {key:average(row[key] for row in rows) for key in keys} | dict(
        episodes=n,success_rate=sum(r["outcome"]=="success" for r in rows)/n if n else None,
        boundary_rate=sum(r["outcome"]=="boundary" for r in rows)/n if n else None,
        outcomes={name:sum(row["outcome"]==name for row in rows) for name in
                  ("success","precision_failure","boundary","time_limit","numerical_failure")},
        mean_episode_percentiles={p:average(r["position_error_percentiles"][p] for r in rows
                                            if r["position_error_percentiles"] is not None)
                                  for p in ("p50","p90","p99")},
        member_metrics={key:average(m[key] for m in members) for key in
                        ("reference_end_error","hold_end_error","actual_length","reference_length",
                         "reference_duration","completion_time")},
        mean_completion_time_reference_ratio=average(
            m["completion_time"]/m["reference_duration"] for m in members
            if m["completion_time"] is not None and m["reference_duration"]>0),
        constraint_counts={key:sum(m[key] for m in members) for key in
                           ("speed_projection","vertical_limit","horizontal_limit")},
        mean_reward_parts={key:average(r["reward_parts"].get(key,0.) for r in rows)
                           for key in sorted({k for r in rows for k in r["reward_parts"]})})
