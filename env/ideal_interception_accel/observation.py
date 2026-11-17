"""Masked local Actor observations and bilateral centralized Critic state."""
import numpy as np

from .config import OBS_DIM, STATE_DIM, DEFENDER_SLOTS, INTRUDER_SLOTS, SCALES


def observe(env):
    c = env.config
    remaining = max(0.,1-env.time/(c.dt*c.max_steps))
    center = np.asarray(c.protected_center,float)
    observation = np.zeros((DEFENDER_SLOTS,OBS_DIM),np.float32)
    for i in np.flatnonzero(env.defender_mask):
        own = env.defenders[i]
        peer_slots = [np.r_[(env.defenders[j,:3]-own[:3])/c.position_scale,
                            (env.defenders[j,3:]-own[3:])/c.speed_scale,1.]
                      if env.defender_mask[j] else np.zeros(7) for j in range(DEFENDER_SLOTS)]
        target_slots = [np.r_[(env.intruders[j,:3]-own[:3])/c.position_scale,
                              (env.intruders[j,3:]-own[3:])/c.speed_scale,1.]
                        if env.intruder_mask[j] else np.zeros(7) for j in range(INTRUDER_SLOTS)]
        assigned = np.zeros(INTRUDER_SLOTS)
        if i in env.assignment:
            assigned[env.assignment[i]] = 1.
        observation[i] = np.concatenate((own[:3]/c.position_scale,
             own[3:]/c.speed_scale,*peer_slots,*target_slots,assigned,
             (center-own[:3])/c.position_scale,[remaining],
             env.defender_mask.astype(float),env.intruder_mask.astype(float)))
    defenders = [np.r_[env.defenders[i,:3]/c.position_scale,
                       env.defenders[i,3:]/c.speed_scale,float(env.defender_mask[i])]
                 if env.defender_mask[i] else np.zeros(7) for i in range(DEFENDER_SLOTS)]
    intruders = [np.r_[env.intruders[j,:3]/c.position_scale,
                       env.intruders[j,3:]/c.speed_scale,float(env.intruder_mask[j])]
                 if env.intruder_mask[j] else np.zeros(7) for j in range(INTRUDER_SLOTS)]
    assignments = [np.eye(INTRUDER_SLOTS)[env.assignment[i]]
                   if env.defender_mask[i] and i in env.assignment
                   else np.zeros(INTRUDER_SLOTS) for i in range(DEFENDER_SLOTS)]
    task = np.eye(len(SCALES))[list(SCALES).index(c.scale)]
    central = np.concatenate((*defenders,*intruders,*assignments,
                              center/c.position_scale,task,[remaining])).astype(np.float32)
    if (observation.shape != (DEFENDER_SLOTS,OBS_DIM) or central.shape != (STATE_DIM,)
            or not np.isfinite(observation).all() or not np.isfinite(central).all()):
        raise FloatingPointError("invalid bilateral observation")
    return observation,central
