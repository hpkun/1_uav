"""Eq.(25) raw features; five stable friendly slots for Table III's 60D."""
import numpy as np
from algorithm.mappo.paper_protocol import pack_centralized_slots
from .geometry import relative_geometry
from env.dynamics import velocity_vector


def direction_sign(own, target, epsilon=1e-9):
    """Left=+1, right=-1; deterministic +1 tie preserves the 3D magnitude.

    Cross comparison uses unit horizontal vectors (dimensionless 1e-12).
    Zero distance, horizontal LOS <=epsilon or horizontal speed <=epsilon
    have no reliable left/right direction and use the same positive branch.
    No change to APH/reward/weapon geometry, which remains unsigned.
    """
    v=velocity_vector(own)[:2]
    los=np.asarray(target)[:2]-np.asarray(own)[:2]
    speed,distance=np.linalg.norm(v),np.linalg.norm(los)
    if speed<=epsilon or distance<=epsilon:return 1.
    v,los=v/speed,los/distance
    cross=v[0]*los[1]-v[1]*los[0]
    return -1. if cross < -1e-12 else 1.


def observe(state, config):
    red, blue = state.friendly_states, state.adversarial_states
    g = relative_geometry(red,blue,config.geometry_epsilon)
    obs = np.zeros((len(red),12),dtype=np.float64)
    for i in np.flatnonzero(state.friendly_alive):
        obs[i,[0,1,2,3,5]] = [1,*red[i,3:6],red[i,2]]
        j = state.assignments[i]
        if j>=0:
            if not state.adversarial_alive[j]:
                raise ValueError('observation assignment references a dead target')
            obs[i,4] = g.alpha_i[i,j]
            if config.observation_encoding=='signed_alpha_v1':
                obs[i,4] *= direction_sign(red[i],blue[j],config.geometry_epsilon)
            obs[i,6:] = [*blue[j,3:6],g.alpha_j[i,j],blue[j,2],g.distance[i,j]]
    return obs, pack_centralized_slots(obs,state.friendly_alive)
