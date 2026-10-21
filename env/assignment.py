"""Deterministic rectangular Hungarian matching without target reuse."""
import numpy as np
from .dynamics import wrap_angle


def hungarian(cost):
    """Shortest augmenting path Hungarian algorithm, O(min(n,m)^2 max(n,m))."""
    a = np.asarray(cost, dtype=np.float64)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError('cost must be a finite matrix')
    transposed = a.shape[0] > a.shape[1]
    if transposed:
        a = a.T
    n, m = a.shape
    u, v = np.zeros(n + 1), np.zeros(m + 1)
    p, way = np.zeros(m + 1, dtype=int), np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minimum = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta, j1 = np.inf, 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = a[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minimum[j]:
                        minimum[j], way[j] = cur, j0
                    if minimum[j] < delta:
                        delta, j1 = minimum[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    pairs = [(p[j] - 1, j - 1) for j in range(1, m + 1) if p[j]]
    return sorted((j, i) if transposed else (i, j) for i, j in pairs)


def assign(task, mode='hungarian'):
    result = np.full(len(task.states), -1, dtype=np.int64)
    ids, points = np.flatnonzero(task.active_mask), np.flatnonzero(~task.completed)
    if not len(ids) or not len(points):
        return result
    delta = task.waypoints[points][None] - task.states[ids, :3][:, None]
    distance = np.linalg.norm(delta, axis=-1)
    heading = abs(wrap_angle(np.arctan2(delta[..., 1], delta[..., 0]) - task.states[ids, 5, None]))
    cost = distance / 10000 + 0.2 * heading / np.pi - 0.1 * task.priorities[points][None]
    if mode == 'hungarian':
        pairs = hungarian(cost)
    elif mode == 'nearest':
        # Greedy global closest pair, ties broken by UAV then waypoint ID.
        candidates = sorted((distance[i, j], i, j) for i in range(len(ids)) for j in range(len(points)))
        rows, columns, pairs = set(), set(), []
        for _, i, j in candidates:
            if i not in rows and j not in columns:
                pairs.append((i, j))
                rows.add(i)
                columns.add(j)
    else:
        raise ValueError('unknown assignment mode')
    for i, j in pairs:
        result[ids[i]] = points[j]
    return result
