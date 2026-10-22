"""Algorithm 1, pure functions; reuse the existing mathematical Hungarian solver."""
import math
import numpy as np
from env.assignment import hungarian
from .geometry import pair_advantages


def aph_assign(ha, hb, friendly_alive, adversarial_alive, eta=5.):
    ha, hb = np.asarray(ha), np.asarray(hb)
    red, blue = np.asarray(friendly_alive,dtype=bool), np.asarray(adversarial_alive,dtype=bool)
    if ha.shape != (len(red),len(blue)) or hb.shape != ha.shape or not (np.isfinite(ha).all() and np.isfinite(hb).all()):
        raise ValueError('APH matrices/masks disagree')
    assignments = np.full(len(red),-1,dtype=np.int64)
    f = np.zeros(ha.shape,dtype=np.int64)
    for i in np.flatnonzero(red):
        candidates = np.flatnonzero(blue & (ha[i] == eta))
        if len(candidates):
            assignments[i] = candidates[np.argmax(hb[i,candidates])]
            f[i,assignments[i]] = 1
    rows = np.flatnonzero(red & (assignments<0))
    columns = np.flatnonzero(blue & ~f.any(axis=0))
    if len(rows) and len(columns):
        cost = -hb[np.ix_(rows,columns)]
        if len(rows)>len(columns):
            cost = np.tile(cost,(1,math.ceil(len(rows)/len(columns))))
        for r,c in hungarian(cost):
            assignments[rows[r]] = columns[c%len(columns)]
            f[rows[r],assignments[rows[r]]] = 1
    return assignments, f


def assignment_matrices(state, config):
    _, ha, _, hb = pair_advantages(state.friendly_states,state.adversarial_states,config)
    live = state.friendly_alive[:,None] & state.adversarial_alive[None,:]
    ha, hb = np.where(live,ha,0.), np.where(live,hb,0.)
    assignments, f = aph_assign(ha,hb,state.friendly_alive,state.adversarial_alive,config.attack_eta)
    return assignments, ha, hb, f
