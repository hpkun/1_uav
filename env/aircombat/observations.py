"""Eq.(25) raw features; five stable friendly slots for Table III's 60D."""
import numpy as np
from algorithm.mappo.paper_protocol import pack_centralized_slots
from .geometry import relative_geometry


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
            obs[i,6:] = [*blue[j,3:6],g.alpha_j[i,j],blue[j,2],g.distance[i,j]]
    return obs, pack_centralized_slots(obs,state.friendly_alive)
