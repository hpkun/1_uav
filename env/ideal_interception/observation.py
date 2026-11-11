"""Masked local Actor observations and bilateral centralized Critic state."""
import numpy as np

from .config import OBS_DIM, STATE_DIM


def observe(env):
    c = env.config
    remaining = max(0.,1-env.time/(c.dt*c.max_steps))
    center = np.asarray(c.protected_center,float)
    observation = np.zeros((5,OBS_DIM),np.float32)
    for i in np.flatnonzero(env.defender_mask):
        own = env.defenders[i]
        peer_slots = [np.r_[(env.defenders[j,:3]-own[:3])/c.position_scale,
                            (env.defenders[j,3:]-own[3:])/c.speed_scale,1.]
                      if env.defender_mask[j] else np.zeros(7) for j in range(5)]
        target_slots = [np.r_[(env.intruders[j,:3]-own[:3])/c.position_scale,
                              (env.intruders[j,3:]-own[3:])/c.speed_scale,1.]
                        if env.intruder_mask[j] else np.zeros(7) for j in range(2)]
        assigned = np.zeros(2)
        assigned[env.assignment[i]] = 1.
        observation[i] = np.concatenate((own[:3]/c.position_scale,
             own[3:]/c.speed_scale,*peer_slots,*target_slots,assigned,
             (center-own[:3])/c.position_scale,[remaining],
             env.defender_mask.astype(float),env.intruder_mask.astype(float)))
    defenders = [np.r_[env.defenders[i,:3]/c.position_scale,
                       env.defenders[i,3:]/c.speed_scale,float(env.defender_mask[i])]
                 if env.defender_mask[i] else np.zeros(7) for i in range(5)]
    intruders = [np.r_[env.intruders[j,:3]/c.position_scale,
                       env.intruders[j,3:]/c.speed_scale,float(env.intruder_mask[j])]
                 if env.intruder_mask[j] else np.zeros(7) for j in range(2)]
    assignments = [np.eye(2)[env.assignment[i]] if env.defender_mask[i]
                   else np.zeros(2) for i in range(5)]
    task = np.eye(3)[("1v1","3v1","5v2").index(c.scale)]
    central = np.concatenate((*defenders,*intruders,*assignments,
                              center/c.position_scale,task,[remaining])).astype(np.float32)
    if (observation.shape != (5,OBS_DIM) or central.shape != (STATE_DIM,)
            or not np.isfinite(observation).all() or not np.isfinite(central).all()):
        raise FloatingPointError("invalid bilateral observation")
    return observation,central
