"""Deterministic flight controllers. No opponent selection or opponent-directed control."""
import numpy as np
from env.dynamics import wrap_angle


def rule_actions(states,active,centers,region_ids,config,mode,orbit_sign=1):
    if mode not in ('cruise','region','region_safe'): raise ValueError('unknown rule policy')
    controls=np.zeros((len(states),3))
    for i in np.flatnonzero(active):
        _,_,_,v,gamma,psi=states[i]
        if mode=='cruise':
            controls[i]=[0,np.sin(gamma),np.cos(gamma)]
            continue
        delta=centers[region_ids[i]]-states[i,:3]
        direction=delta[:2].copy(); horizontal=np.linalg.norm(direction)
        if horizontal < .7*config.region_radius:
            # A tangent orbit about the region, not pursuit of another UAV.
            direction=orbit_sign*np.array([-delta[1],delta[0]])
        if mode=='region_safe':
            for j in np.flatnonzero(active):
                if i==j: continue
                separation=states[i,:3]-states[j,:3]; d=np.linalg.norm(separation)
                away=separation[:2]; horizontal_separation=np.linalg.norm(away)
                if d < 2*config.safe_distance:
                    if horizontal_separation<1e-9: away=np.array([0.,1. if i>j else -1.]); horizontal_separation=1.
                    direction+=away/horizontal_separation*(2*config.safe_distance-d)*5
        desired_psi=psi if np.linalg.norm(direction)<1e-9 else np.arctan2(direction[1],direction[0])
        desired_gamma=np.clip(np.arctan2(delta[2],max(horizontal,config.region_radius*.3)),-.3,.3)
        gamma_rate=np.clip(.7*(desired_gamma-gamma),-.08,.08)
        yaw_rate=np.clip(.6*wrap_angle(desired_psi-psi),-.12,.12)
        vertical=np.cos(gamma)+v/config.gravity*gamma_rate
        lateral=v*np.cos(gamma)/config.gravity*yaw_rate
        mu=np.arctan2(lateral,vertical)
        normal=np.clip(np.hypot(vertical,lateral),0,8)
        zeta=np.clip(np.sin(gamma)+(225-v)/(4*config.gravity),-1,2)
        controls[i]=[mu,zeta,normal]
    return np.clip((controls+np.array([np.pi,1,0]))/np.array([2*np.pi,3,8]),0,1)
