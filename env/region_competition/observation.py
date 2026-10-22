import numpy as np


def observe(states,active,opponents,opponent_active,centers,region_ids,step,config):
    obs=np.zeros((len(states),12),dtype=np.float32)
    for i in np.flatnonzero(active):
        x,y,z,v,gamma,psi=states[i]
        friends=np.flatnonzero(active & (np.arange(len(states))!=i))
        others=opponents[opponent_active,:3]
        obs[i,:6]=[1,(v-config.v_min)/(config.v_max-config.v_min),gamma/config.gamma_limit,np.sin(psi),np.cos(psi),z/config.altitude_scale]
        obs[i,6:9]=(centers[region_ids[i]]-states[i,:3])/config.distance_scale
        # No neighbor is represented by -1, distinguishable from co-location.
        obs[i,9]=np.linalg.norm(states[friends,:3]-states[i,:3],axis=1).min()/config.distance_scale if len(friends) else -1
        obs[i,10]=np.linalg.norm(others-states[i,:3],axis=1).min()/config.distance_scale if len(others) else -1
        obs[i,11]=max(0,config.max_steps-step)/config.max_steps
    return obs


def centralized(obs):
    result=np.zeros((5,12),dtype=np.float32)
    result[:len(obs)]=obs
    return result.reshape(60)
