"""12D local observations and exactly 60D padded centralized input."""
import numpy as np
from .dynamics import wrap_angle
from .models import pairwise_distances


def observe(task, assignments, config):
    obs = np.zeros((len(task.states), 12), dtype=np.float32)
    distances = pairwise_distances(task.states)
    for i in np.flatnonzero(task.active_mask):
        x, y, z, v, gamma, psi = task.states[i]
        obs[i, :5] = [1, (v - config.v_min) / (config.v_max - config.v_min), gamma / config.gamma_limit, psi / np.pi, z / config.altitude_scale]
        j = assignments[i]
        if j >= 0:
            relative = task.waypoints[j] - np.array([x, y, z])
            obs[i, 5:8] = relative / config.distance_scale
            obs[i, 8] = np.linalg.norm(relative) / config.distance_scale
            obs[i, 9] = wrap_angle(np.arctan2(relative[1], relative[0]) - psi) / np.pi
            obs[i, 10] = 1
        others = np.flatnonzero(task.active_mask & (np.arange(len(obs)) != i))
        obs[i, 11] = distances[i, others].min() / config.distance_scale if len(others) else 0
    return obs


def centralized(observations):
    obs = np.asarray(observations)
    if obs.shape[-1] != 12 or obs.shape[-2] > 5:
        raise ValueError('expected [..., N<=5,12] observations')
    result = np.zeros((*obs.shape[:-2], 5, 12), dtype=obs.dtype)
    result[..., :obs.shape[-2], :] = obs
    return result.reshape(*obs.shape[:-2], 60)
