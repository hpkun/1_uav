"""Potential-based direction shaping for static navigation only."""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class HeadingConfig:
    heading_enabled: bool = False
    heading_weight: float = .3
    heading_velocity_reference: float = 1.
    heading_distance_reference: float = 5.
    heading_discount: float = .99

    def __post_init__(self):
        if type(self.heading_enabled) is not bool:
            raise ValueError("heading_enabled must be boolean")
        for name in ("heading_weight", "heading_velocity_reference", "heading_distance_reference",
                     "heading_discount"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {name}")
        if min(self.heading_velocity_reference, self.heading_distance_reference,
               self.heading_discount) <= 0 or self.heading_discount > 1:
            raise ValueError("invalid heading references or discount")


def direction_quality(states, waypoint, mask, config):
    slots = np.flatnonzero(mask)
    states = np.asarray(states,dtype=float)[slots]
    v, d = states[:, 3:], np.asarray(waypoint)-states[:, :3]
    speed = np.hypot.reduce(v, axis=1)
    distance = np.hypot.reduce(d, axis=1)
    if not np.isfinite(speed).all() or not np.isfinite(distance).all():
        raise ValueError("nonfinite navigation state")
    uv = np.divide(v, speed[:,None], out=np.zeros_like(v), where=speed[:,None] != 0)
    ud = np.divide(d, distance[:,None], out=np.zeros_like(d), where=distance[:,None] != 0)
    cosine = np.clip(np.sum(uv*ud,axis=1),-1.,1.)
    sv, sr = speed/np.maximum(speed, config.heading_velocity_reference), (
        config.heading_velocity_reference/np.maximum(speed, config.heading_velocity_reference))
    speed_gate = sv/(sv+sr)
    distance_gate = np.minimum(distance,config.heading_distance_reference)/config.heading_distance_reference
    valid = (speed > 0) & (distance > 0)
    return dict(slots=slots.tolist(), quality=(speed_gate*distance_gate*cosine).tolist(),
                speed=speed.tolist(), distance=distance.tolist(),
                angle_deg=[float(np.degrees(np.arccos(c))) if ok else None for c, ok in zip(cosine, valid)])


def heading_transition(before, after, waypoint, mask, next_mask, initial_count, terminated, config):
    current = direction_quality(before, waypoint, mask, config)
    successor = direction_quality(after, waypoint, next_mask, config)
    weight = config.heading_weight/initial_count
    phi = weight*sum(current["quality"])
    next_phi = 0. if terminated else weight*sum(successor["quality"])
    delta = config.heading_discount*next_phi-phi
    return dict(current=current, next=successor, potential=phi, next_potential=next_phi,
                difference=delta, reward=delta if config.heading_enabled else 0.,
                active_count=len(current["slots"]))
