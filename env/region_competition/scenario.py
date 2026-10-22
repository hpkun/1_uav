"""Mirrored teams and task-region phases use only the environment's own RNG."""
import numpy as np
from env.dynamics import wrap_angle


def initialize(config,rng,options=None):
    n=config.num_uavs
    region_y=(np.arange(config.num_regions)-(config.num_regions-1)/2)*config.region_spacing
    anchors=np.column_stack((np.zeros(config.num_regions),region_y,np.full(config.num_regions,config.start_altitude)))
    ids=np.arange(n)%config.num_regions
    red=np.column_stack((np.full(n,-config.start_distance),region_y[ids]+np.arange(n)//config.num_regions*300.,
                         rng.uniform(config.start_altitude-150,config.start_altitude+150,n),rng.uniform(210,240,n),
                         rng.uniform(-.04,.04,n),rng.uniform(-.1,.1,n)))
    blue=red.copy(); blue[:,0]*=-1; blue[:,5]=wrap_angle(np.pi-red[:,5])
    result=dict(red_states=red,blue_states=blue,red_active=np.ones(n,dtype=bool),blue_active=np.ones(n,dtype=bool),
                anchors=anchors,phase=float(rng.uniform(-np.pi,np.pi)))
    if options:
        if set(options)-set(result): raise ValueError('unknown scenario option')
        result.update(options)
    for team in ('red','blue'):
        states=np.array(result[team+'_states'],dtype=float,copy=True)
        active=np.asarray(result[team+'_active'])
        if states.shape!=(n,6) or active.shape!=(n,) or not np.isfinite(states).all() or not np.isin(active,(0,1)).all():
            raise ValueError('invalid team initial state')
        if ((states[:,3]<config.v_min)|(states[:,3]>config.v_max)|(abs(states[:,4])>config.gamma_limit)).any():
            raise ValueError('initial flight state outside envelope')
        states[:,5]=wrap_angle(states[:,5]); result[team+'_states']=states
        result[team+'_active']=active.astype(bool,copy=True)
    result['anchors']=np.array(result['anchors'],dtype=float,copy=True)
    if result['anchors'].shape!=(config.num_regions,3) or not np.isfinite(result['anchors']).all() or not np.isfinite(result['phase']):
        raise ValueError('invalid region motion')
    if config.num_regions>1:
        distances=np.linalg.norm(result['anchors'][:,None]-result['anchors'][None],axis=-1)
        np.fill_diagonal(distances,np.inf)
        if distances.min()<=2*config.region_radius: raise ValueError('region anchors must keep spheres disjoint')
    return result
