"""Shared trajectory clock and analytic sampling for both replay exporters."""
from bisect import bisect_right
import gzip
import json
from pathlib import Path

import numpy as np

from env.interception.config import VERSION, MAX_PURSUERS
from env.interception.motion import advance

GROUPS = (("pursuers", "active_mask", "P", "#d64040", 0),
          ("evaders", "evader_mask", "E", "#2378bf", MAX_PURSUERS))

# Plotly Scatter3d and Matplotlib each use their native marker.
EVENT_STYLES = {
    "interception_contact": ("Interception", "#1aa575", "x", "x"),
    "friendly_collision": ("Friendly collision", "#ef7900", "diamond", "D"),
    "evader_collision": ("Evader collision", "#ef7900", "diamond", "D"),
    "capture": ("Capture", "#dc1824", "x", "x"),
    "collision": ("Collision", "#ef7900", "diamond", "D"),
    "breach": ("Breach", "#f0c400", "square", "s"),
    "pursuer_boundary_violation": ("Boundary violation", "#8c43ac", "cross", "+"),
    "evader_escape": ("Escape", "#168c85", "circle-open", "o"),
}


def display_event(event):
    kind = event["kind"]
    title, color, plotly_marker, mpl_marker = EVENT_STYLES[kind]
    names = list(dict.fromkeys(f"P{i}" if i < MAX_PURSUERS else f"E{i-MAX_PURSUERS}"
                               for i in event["entities"]))
    positions = event["positions"]
    position = (np.mean(positions, axis=0).tolist() if kind in ("collision", "friendly_collision", "evader_collision", "interception_contact")
                else positions[1 if kind == "capture" else 0])
    return dict(kind=kind, time=event["time"], entity_ids=event["entities"],
                positions=positions, position=position, color=color, plotly_marker=plotly_marker,
                mpl_marker=mpl_marker,
                label=f"{title} {' / '.join(names)} · {event['time']:.6f} s")


class Timeline:
    def __init__(self, source):
        opener = gzip.open if Path(source).suffix == ".gz" else open
        with opener(source, "rt", encoding="utf-8") as stream:
            data = json.load(stream)
        if data.get("version") not in (VERSION, "uav-interception-v3.0.0", "uav-interception-v2.0.0",
                                       "uav-interception-v1.0.0") or data.get("trace_version", 1) not in (1, 2):
            raise ValueError("unsupported interception trajectory")
        self.data = data
        self.frames = data["frames"]
        self.times = [float(row["time"]) for row in self.frames]
        if not self.times or self.times[0] != 0 or any(not np.isfinite(t) for t in self.times):
            raise ValueError("invalid trajectory time")
        # An entity may start at the legal envelope limit and move outward
        # immediately. Retain both the initial and zero-time terminal records.
        immediate_exit = self.times == [0., 0.] and self.frames[-1]["outcome"] in (
            "pursuer_boundary_violation", "evader_escape")
        if not immediate_exit and any(b <= a for a, b in zip(self.times, self.times[1:])):
            raise ValueError("trajectory time must be strictly increasing")
        self.duration = self.times[-1]
        self.events = sorted((event for row in self.frames for event in row["events"]),
                             key=lambda event: event["time"])
        self.display_events = [display_event(event) for event in self.events if event["kind"] not in ("near_miss", "friendly_warning")]
        self.events_at = {}
        for event in self.events:
            self.events_at.setdefault(event["time"], []).append(event)
        self.captures = {}
        for event in self.events:
            if event["kind"] == "interception_contact":
                self.captures.update((entity, (event, index)) for index, entity in enumerate(event["entities"]))
            elif event["kind"] == "capture":
                self.captures[event["entities"][1]] = (event, 1)
        self.key_times = sorted(set(self.times + [event["time"] for event in self.events]))
        if self.key_times[0] < 0 or self.key_times[-1] > self.duration:
            raise ValueError("event time outside trajectory")
        self.entities = [(key, mask, f"{prefix}{i}", color, i, offset+i)
                         for key, mask, prefix, color, offset in GROUPS
                         for i, active in enumerate(self.frames[0][mask]) if active]

    def at(self, timestamp):
        if not np.isfinite(timestamp) or not 0 <= timestamp <= self.duration:
            raise ValueError(f"replay time must be in [0, {self.duration}] s")
        index = bisect_right(self.times, timestamp) - 1
        before = self.frames[index]
        exact = timestamp == self.times[index]
        after = before if exact else self.frames[index+1]
        commands = after.get("commands")
        entities = []
        for key, mask, name, color, i, global_id in self.entities:
            capture, endpoint = self.captures.get(global_id, (None, None))
            captured = capture is not None and capture["time"] <= timestamp
            state = before[key][i]
            reconstruct = not exact and commands is not None
            if reconstruct and before[mask][i] and not captured:
                tau = self.data["config"]["pursuer_tau" if key == "pursuers" else "evader_tau"]
                state = advance(state, commands[global_id], tau, timestamp-before["time"]).tolist()
            position = capture["positions"][endpoint] if captured else state[:3]
            # Event coordinates take precedence over reconstructed coordinates.
            for event in self.events_at.get(timestamp, ()):
                if global_id in event["entities"]:
                    position = event["positions"][event["entities"].index(global_id)]
            active = bool(before[mask][i] and not captured)
            velocity = state[3:] if active and (exact or reconstruct) else None
            actions = after.get("normalized_actions")
            entities.append(dict(name=name, color=color, position=position, velocity=velocity, active=active,
                                 captured=captured,
                                 action=actions[i] if key == "pursuers" and actions is not None and active else None,
                                 command=commands[global_id] if commands is not None and active else None,
                                 requested=(after["requested_velocities"][global_id]
                                            if active and after.get("requested_velocities") is not None else None),
                                 initial_acceleration=(after["initial_accelerations"][global_id]
                                                       if active and after.get("initial_accelerations") is not None else None),
                                 speed=float(np.linalg.norm(velocity)) if velocity is not None else None))
        pursuers = [entity for entity in entities if entity["name"].startswith("P") and entity["active"]]
        evaders = [entity for entity in entities if entity["name"].startswith("E") and entity["active"]]
        pairs = []
        if exact or commands is not None:
            for p in pursuers:
                for e in evaders:
                    pairs.append(dict(pursuer=p["name"], evader=e["name"],
                                      distance=float(np.linalg.norm(np.subtract(p["position"], e["position"]))),
                                      relative_speed=float(np.linalg.norm(np.subtract(p["velocity"], e["velocity"])))))
        pair = min(pairs, key=lambda row: row["distance"], default=None)
        return dict(time=timestamp, step=after["step"], entities=entities, outcome=before["outcome"],
                    nearest_pair=pair, pairs=pairs, distance=pair["distance"] if pair else None,
                    state_basis="recorded" if exact else "analytic" if commands is not None else "held legacy position",
                    recorded_time=before["time"], minimum=before.get("minimum_separation"),
                    minimum_surface_clearance=before.get("minimum_surface_clearance"),
                    dwell=before.get("dwell_seconds") if exact else None,
                    last_recorded_dwell=before.get("dwell_seconds"), commands=commands,
                    rewards=before.get("reward_parts") or None)

    def bounds(self, samples):
        config = self.data["config"]
        center, radius = np.array(config["center"]), config["radius"]
        points = np.array([entity["position"] for row in samples for entity in row["entities"]])
        low = np.minimum(points.min(0), center-radius)
        high = np.maximum(points.max(0), center+radius)
        if config.get("protected_zone") == "ground_cylinder":
            low[2] = min(points[:, 2].min(), center[2])
            high[2] = max(points[:, 2].max(), center[2] + config["protected_height"])
        return np.column_stack((low-3, high+3)).tolist()
