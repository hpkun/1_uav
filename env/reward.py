"""Pure, reconstructible five-component inspection rewards."""
import numpy as np
from .models import pairwise_distances

COMPONENTS = ('progress', 'completion', 'separation', 'efficiency', 'terminal')


def compute_rewards(previous, current, waypoints, assignments, current_mask, next_mask, events, step, success, config):
    n = len(previous)
    parts = {name: np.zeros(n, dtype=np.float64) for name in COMPONENTS}
    for i in np.flatnonzero(current_mask):
        j = assignments[i]
        if j >= 0:
            before = np.linalg.norm(waypoints[j] - previous[i, :3])
            after = np.linalg.norm(waypoints[j] - current[i, :3])
            parts['progress'][i] = 2 * (before - after) / 1000
        parts['efficiency'][i] = -0.01
    for event in events:
        parts['completion'][event['uav_id']] += 10
    distances = pairwise_distances(current)
    valid = np.flatnonzero(next_mask)
    for i in valid:
        others = valid[valid != i]
        if len(others):
            parts['separation'][i] = -np.maximum(0, 1 - distances[i, others] / config.safe_distance).sum() / len(others)
        if success:
            parts['terminal'][i] = 20 - step // 25
    return sum(parts.values()), parts
