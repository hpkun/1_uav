"""Deterministic nearest-target association for active point-mass members."""
import numpy as np


def nearest_distance_assignment(defender_positions, intruder_positions,
                                defender_mask, intruder_mask):
    """Map each active defender to its nearest active intruder in 3D.

    Multiple defenders may select one intruder. Exact ties choose the lower
    intruder index, without randomness or order-dependent containers.
    """
    defenders = np.asarray(defender_positions, dtype=np.float64)
    intruders = np.asarray(intruder_positions, dtype=np.float64)
    own = np.asarray(defender_mask)
    target = np.asarray(intruder_mask)
    if (defenders.ndim != 2 or defenders.shape[1] != 3 or
            intruders.ndim != 2 or intruders.shape[1] != 3 or
            own.shape != (len(defenders),) or target.shape != (len(intruders),) or
            own.dtype.kind != "b" or target.dtype.kind != "b" or
            not np.isfinite(defenders).all() or not np.isfinite(intruders).all()):
        raise ValueError("invalid assignment positions or active masks")
    active_targets = np.flatnonzero(target)
    if not len(active_targets):
        return {}
    result = {}
    for i in np.flatnonzero(own):
        distance = np.linalg.norm(intruders[active_targets]-defenders[i], axis=1)
        result[int(i)] = int(active_targets[np.argmin(distance)])
    return result


def minimum_cost_assignment(cost):
    """O(n^3) one-to-one Hungarian solver; ascending rows/columns break ties."""
    matrix=np.asarray(cost,dtype=np.float64)
    if (matrix.ndim!=2 or matrix.shape[0]==0 or matrix.shape[0]!=matrix.shape[1]
            or not np.isfinite(matrix).all() or np.any(matrix<0)):
        raise ValueError("expected a finite nonnegative square cost matrix")
    n=len(matrix)
    u=np.zeros(n+1)
    v=np.zeros(n+1)
    p=np.zeros(n+1,dtype=int)
    way=np.zeros(n+1,dtype=int)
    for i in range(1,n+1):
        p[0]=i
        j0=0
        minv=np.full(n+1,np.inf)
        used=np.zeros(n+1,dtype=bool)
        while True:
            used[j0]=True
            i0=p[j0]
            delta=np.inf
            j1=0
            for j in range(1,n+1):
                if used[j]:
                    continue
                cur=matrix[i0-1,j-1]-u[i0]-v[j]
                if cur<minv[j]:
                    minv[j]=cur
                    way[j]=j0
                if minv[j]<delta:
                    delta=minv[j]
                    j1=j
            for j in range(n+1):
                if used[j]:
                    u[p[j]]+=delta
                    v[j]-=delta
                else:
                    minv[j]-=delta
            j0=j1
            if p[j0]==0:
                break
        while True:
            j1=way[j0]
            p[j0]=p[j1]
            j0=j1
            if j0==0:
                break
    return {int(p[j]-1):int(j-1) for j in range(1,n+1)}


def hungarian_fixed_assignment(defender_positions,intruder_positions):
    defenders=np.asarray(defender_positions,dtype=np.float64)
    intruders=np.asarray(intruder_positions,dtype=np.float64)
    if (defenders.ndim!=2 or intruders.shape!=defenders.shape or defenders.shape[1]!=3
            or not np.isfinite(defenders).all() or not np.isfinite(intruders).all()):
        raise ValueError("expected equally sized finite 3D position arrays")
    return minimum_cost_assignment(np.linalg.norm(defenders[:,None]-intruders[None],axis=2))
