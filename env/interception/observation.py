import numpy as np
from .config import MAX_EVADERS, OBS_DIM, SLOT


def observe(state, config):
    p, e = state.pursuers, state.evaders
    scales = np.array([config.position_scale] * 3 + [config.velocity_scale] * 3)
    remaining = max(0., 1. - state.time / (config.dt * config.max_steps))
    observations = np.zeros((len(p), OBS_DIM+(len(e)-MAX_EVADERS)*SLOT), dtype=np.float32)
    for i in np.flatnonzero(state.pursuer_mask):
        own = p[i]
        start = np.r_[own / scales, (np.asarray(config.center) - own[:3]) / config.position_scale, remaining]
        teammates = np.arange(len(p)) != i
        others = np.vstack((e, p[teammates]))
        valid = np.r_[state.evader_mask, state.pursuer_mask[teammates]]
        relative = np.zeros((len(others), SLOT))
        relative[valid, :-1] = (others[valid] - own) / scales
        relative[valid, -1] = 1.
        observations[i] = np.r_[start, relative.ravel()]
    entities = np.zeros((len(p) + len(e), SLOT))
    valid = np.r_[state.pursuer_mask, state.evader_mask]
    entities[valid, :-1] = np.vstack((p, e))[valid] / scales
    entities[valid, -1] = 1.
    central = np.r_[entities.ravel(), np.asarray(config.center) / config.position_scale,
                    config.radius / config.position_scale, remaining].astype(np.float32)
    if not np.isfinite(observations).all() or not np.isfinite(central).all():
        raise FloatingPointError("nonfinite policy inputs")
    return observations, central
