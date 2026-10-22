"""Exactly five terms, Eq.(20); no inspection shaping or reward scaling."""
import numpy as np

COMPONENTS = ('attack','posture','allied','shootdown','terminal')


def allied_penalty(red_states, red_alive, blue_alive, blue_advantage, config):
    result = np.zeros(len(red_states))
    ids = np.flatnonzero(red_alive)
    if len(ids)<2:
        return result
    q = (blue_advantage>0) & blue_alive[:,None] & red_alive[None,:]
    for i in ids:
        others = ids[ids!=i]
        proximity = np.maximum(0,1-np.linalg.norm(red_states[others,:3]-red_states[i,:3],axis=-1)/config.tau_min)
        shared = np.any(q[:,i,None] & q[:,others],axis=0)
        result[i] = -np.sum(proximity+shared)/(len(ids)-1)
    return result


def compute_rewards(state, current_mask, transition_assignments, red_advantage,
                    blue_advantage, posture, events, config):
    parts = {k:np.zeros(len(current_mask)) for k in COMPONENTS}
    for i in np.flatnonzero(current_mask):
        j = transition_assignments[i]
        if j>=0:
            parts['attack'][i] = red_advantage[i,j]-blue_advantage[j,i]
            parts['posture'][i] = posture[i,j]
    parts['allied'] = allied_penalty(state.friendly_states,state.friendly_alive,
                                     state.adversarial_alive,blue_advantage,config)
    for event in events:
        if event['attacker_side']=='Red':
            parts['shootdown'][event['attacker_id']] += config.shootdown_r0
    if not state.adversarial_alive.any():
        parts['terminal'][current_mask] = (state.friendly_alive.sum()*config.terminal_e1
                                           -np.floor(state.step_count/config.terminal_e2))
    for value in parts.values():
        value[~current_mask] = 0
    return sum(parts.values()), parts
