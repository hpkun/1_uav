"""Shared trajectory clock and analytic sampling for both replay exporters."""
from bisect import bisect_right
from copy import deepcopy
import gzip
import json
from pathlib import Path

import numpy as np

from env.interception.config import VERSION, MAX_PURSUERS
from env.interception.motion import advance

GROUPS = (("pursuers", "active_mask", "P", "#2378bf", 0),
          ("evaders", "evader_mask", "E", "#df6a36", MAX_PURSUERS))


class Timeline:
    def __init__(self, source):
        opener = gzip.open if Path(source).suffix == ".gz" else open
        with opener(source, "rt", encoding="utf-8") as stream:
            data = json.load(stream)
        if data.get("version") != VERSION or data.get("trace_version", 1) not in (1, 2):
            raise ValueError("unsupported interception trajectory")
        self.data = data
        self.frames = data["frames"]
        self.times = [float(row["time"]) for row in self.frames]
        if not self.times or self.times[0] != 0 or any(not np.isfinite(t) for t in self.times):
            raise ValueError("invalid trajectory time")
        if any(b <= a for a, b in zip(self.times, self.times[1:])):
            raise ValueError("trajectory time must be strictly increasing")
        self.duration = self.times[-1]
        self.events = sorted((event for row in self.frames for event in row["events"]),
                             key=lambda event: event["time"])
        self.captures = {event["entities"][1]: event for event in self.events if event["kind"] == "capture"}
        self.key_times = sorted(set(self.times + [event["time"] for event in self.events]))
        if self.key_times[0] < 0 or self.key_times[-1] > self.duration:
            raise ValueError("event time outside trajectory")
        self.entities = [(key, mask, f"{prefix}{i}", color, i, offset+i)
                         for key, mask, prefix, color, offset in GROUPS
                         for i, active in enumerate(self.frames[0][mask]) if active]

    def at(self, timestamp):
        index = bisect_right(self.times, timestamp) - 1
        before = self.frames[index]
        exact = timestamp == self.times[index]
        after = before if exact else self.frames[index+1]
        commands = after.get("commands")
        entities = []
        for key, mask, name, color, i, global_id in self.entities:
            capture = self.captures.get(global_id)
            captured = capture is not None and capture["time"] <= timestamp
            state = before[key][i]
            if not exact and commands is not None and before[mask][i]:
                tau = self.data["config"]["pursuer_tau" if key == "pursuers" else "evader_tau"]
                end = capture["time"] if captured else timestamp
                state = advance(state, commands[global_id], tau, end-before["time"]).tolist()
            position = capture["positions"][1] if captured else state[:3]
            # Event coordinates take precedence over reconstructed coordinates.
            for event in after["events"]:
                if abs(event["time"]-timestamp) <= 1e-10 and global_id in event["entities"]:
                    position = event["positions"][event["entities"].index(global_id)]
            active = bool(before[mask][i] and not captured)
            velocity = state[3:] if active else None
            entities.append(dict(name=name, color=color, position=position, velocity=velocity, active=active,
                                 speed=float(np.linalg.norm(velocity)) if active else None))
        return dict(time=timestamp, step=after["step"], entities=entities, outcome=before["outcome"],
                    distance=nearest(before.get("distances")) if exact else None,
                    minimum=before.get("minimum_separation") if exact else None,
                    dwell=before.get("dwell_seconds") if exact else None,
                    commands=commands, rewards=before.get("reward_parts") or None if exact else None)

    def bounds(self, samples):
        config = self.data["config"]
        center, radius = np.array(config["center"]), config["radius"]
        points = np.array([entity["position"] for row in samples for entity in row["entities"]])
        low = np.minimum(points.min(0), center-radius)
        high = np.maximum(points.max(0), center+radius)
        return np.column_stack((low-3, high+3)).tolist()


def nearest(matrix):
    if matrix is None:
        return None
    return min((x for row in matrix for x in row if x is not None), default=None)
