"""Deterministic feedback controller; diagnostic only, never used for MAPPO training."""
from dataclasses import asdict,dataclass
import math
import numpy as np
from env.dynamics import wrap_angle


@dataclass
class NavigationConfig:
    k_psi: float = 0.8
    k_gamma: float = 0.6
    max_psi_rate: float = 0.06
    max_gamma_rate: float = 0.03
    max_desired_gamma: float = math.radians(20)
    desired_speed: float = 225.0
    tau_v: float = 6.0

    def __post_init__(self):
        for key,value in asdict(self).items():
            if not math.isfinite(value) or value<=0: raise ValueError(f'invalid navigation {key}')
        if self.max_desired_gamma>=math.pi/2: raise ValueError('max_desired_gamma must be below pi/2')


class NavigationController:
    def __init__(self,config=None):
        self.config=config or NavigationConfig()

    def actions(self,info,env_config):
        c=self.config; states=np.asarray(info['states']); n=len(states)
        if not env_config.v_min<=c.desired_speed<=env_config.v_max:
            raise ValueError('navigation desired_speed is outside environment speed range')
        controls=np.zeros((n,3)); saturation=np.zeros((n,3),dtype=bool)
        desired_rates=np.zeros((n,2)); achieved_rates=np.zeros((n,2))
        for i in np.flatnonzero(info['active_mask']):
            _,_,_,v,gamma,psi=states[i]
            j=info['assignments'][i]
            if j<0:
                roll,tangential,normal=0.,np.sin(gamma),np.cos(gamma)
            else:
                dx,dy,dz=np.asarray(info['waypoints'])[j]-states[i,:3]
                psi_des=np.arctan2(dy,dx)
                gamma_des=np.clip(np.arctan2(dz,np.hypot(dx,dy)),-c.max_desired_gamma,c.max_desired_gamma)
                psi_rate=np.clip(c.k_psi*wrap_angle(psi_des-psi),-c.max_psi_rate,c.max_psi_rate)
                gamma_rate=np.clip(c.k_gamma*(gamma_des-gamma),-c.max_gamma_rate,c.max_gamma_rate)
                A=np.cos(gamma)+v*gamma_rate/env_config.gravity
                B=v*np.cos(gamma)*psi_rate/env_config.gravity
                normal=np.hypot(A,B); roll=np.arctan2(B,A)
                tangential=np.sin(gamma)+(c.desired_speed-v)/(env_config.gravity*c.tau_v)
                desired_rates[i]=[psi_rate,gamma_rate]
            raw=np.array([roll,tangential,normal])
            clipped=np.clip(raw,[-np.pi,-1,0],[np.pi,2,8])
            saturation[i]=np.abs(clipped-raw)>1e-12
            controls[i]=clipped
            mu,zeta,nload=clipped
            achieved_rates[i]=[env_config.gravity*nload*np.sin(mu)/(v*np.cos(gamma)),
                               env_config.gravity*(nload*np.cos(mu)-np.cos(gamma))/v]
        normalized=(controls-np.array([-np.pi,-1,0]))/np.array([2*np.pi,3,8])
        normalized[~np.asarray(info['active_mask'],dtype=bool)]=[.5,1/3,1/8]
        return normalized,dict(saturation=saturation,normal_load_saturation=saturation[:,2],
                               desired_rates=desired_rates,achieved_rates=achieved_rates,physical_controls=controls)


def reachable_scenario():
    """An additional labeled interface test, never a replacement for standard scenarios."""
    return dict(states=[[0,-700,3000,225,0,0],[0,0,3000,225,0,0],[0,700,3000,225,0,0]],
                waypoints=[[2200,-700,3000],[2200,0,3000],[2200,700,3000],[4700,-700,3200],[4700,700,2800]])
