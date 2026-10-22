"""Five independent region-competition terms; no Coop3D inspection rewards."""
import numpy as np

COMPONENTS=('control','tracking','separation','numerical','result')


def compute_reward(previous,current,centers_before,region_ids,current_mask,next_mask,credit,events,outcome,finished,config):
    parts={name:np.zeros(len(current),dtype=float) for name in COMPONENTS}
    for i in np.flatnonzero(current_mask):
        region=centers_before[region_ids[i]]
        # Freeze the region at its PRE-step position: translation cannot masquerade as flight progress.
        progress=np.linalg.norm(previous[i,:3]-region)-np.linalg.norm(current[i,:3]-region)
        parts['tracking'][i]=config.tracking_weight*progress/config.distance_scale
        parts['control'][i]=config.control_weight*credit[i]
        count=sum(events[k][i] for k in events)
        parts['numerical'][i]=-config.numerical_weight*count
        if finished: parts['result'][i]=config.result_weight*outcome
    for i in np.flatnonzero(next_mask):
        ids=np.flatnonzero(next_mask & (np.arange(len(current))!=i))
        if len(ids):
            distances=np.linalg.norm(current[ids,:3]-current[i,:3],axis=1)
            parts['separation'][i]=-config.separation_weight*np.maximum(0,1-distances/config.safe_distance).mean()
    return sum(parts.values()),parts
