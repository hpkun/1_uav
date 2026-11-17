"""Static front-arc 20v20 layout and nearest assignment audit, separate from V3.5.

This module does not construct an InterceptionEnv or change its 15-slot protocol.
"""
import numpy as np

from .assignment import nearest_distance_assignment
from .config import FIFTEEN_INTRUDER_WEDGE

TWENTY_COUNT = 20
VERSION = "20v20-front-arc-nearest-v2"
DEFENDER_CENTER = (-220., 0., 60.)
TWENTY_DEFENDER_RADIUS_M = 65.
ARC_START_DEG = -75.
ARC_END_DEG = 75.
NEW_INTRUDER_ANCHORS = ((285., -150., 60.), (285., -75., 60.),
                        (285., 0., 60.), (285., 75., 60.),
                        (285., 150., 60.))
PROTECTED_CENTER = (-220., 0., 0.)
PROTECTED_RADIUS_M = 20.
ARENA_LOWER = (-300., -300., 0.)
ARENA_UPPER = (300., 300., 120.)
REFERENCE_NOTE = ("Scene layout is project-specific and only borrows the front-defense "
    "/ wedge-intrusion large-scale layout idea from Zuo et al. 2026; it does not "
    "reproduce the paper's LB-Hungarian algorithm or numerical scene.")


def initial_positions():
    """Return deterministic nominal anchors, without sampling or trajectory generation."""
    angles = np.deg2rad(np.linspace(ARC_START_DEG, ARC_END_DEG, TWENTY_COUNT))
    defenders = np.column_stack((
        DEFENDER_CENTER[0]+TWENTY_DEFENDER_RADIUS_M*np.cos(angles),
        DEFENDER_CENTER[1]+TWENTY_DEFENDER_RADIUS_M*np.sin(angles),
        np.full(TWENTY_COUNT, DEFENDER_CENTER[2])))
    intruders = np.asarray(FIFTEEN_INTRUDER_WEDGE+NEW_INTRUDER_ANCHORS,
                           dtype=np.float64)
    for positions in (defenders, intruders):
        if (positions.shape != (TWENTY_COUNT, 3) or not np.isfinite(positions).all()
                or np.any(positions < ARENA_LOWER) or np.any(positions > ARENA_UPPER)):
            raise ValueError("invalid 20v20 nominal layout")
    adjacent = np.linalg.norm(np.diff(defenders, axis=0), axis=1)
    if np.any(defenders[:, 0] <= PROTECTED_CENTER[0]) or adjacent.min() <= 8.:
        raise ValueError("front-arc defenders are behind the asset or too close")
    return defenders, intruders


def _cross_2d(a, b):
    return float(a[0]*b[1]-a[1]*b[0])


def segments_cross_interior(a, b, c, d):
    """Count only proper nonparallel interior intersections, never endpoints."""
    a, b, c, d = (np.asarray(point, dtype=float)[:2] for point in (a, b, c, d))
    r, s = b-a, d-c
    denominator = _cross_2d(r, s)
    if abs(denominator) <= 1e-12:
        return False
    t = _cross_2d(c-a, s)/denominator
    u = _cross_2d(c-a, r)/denominator
    # Shared endpoints can round to 0.9999999999999999 on converging rays.
    interior_tolerance = 1e-9
    return (interior_tolerance < t < 1.-interior_tolerance and
            interior_tolerance < u < 1.-interior_tolerance)


def assignment_line_crossings_xy(defenders, intruders, assignment):
    return sum(segments_cross_interior(defenders[i], intruders[assignment[i]],
                                       defenders[k], intruders[assignment[k]])
               for i in range(TWENTY_COUNT) for k in range(i+1, TWENTY_COUNT))


def monotonic_y_inversions(defenders, intruders, assignment):
    order = sorted(range(TWENTY_COUNT), key=lambda i: (round(float(defenders[i, 1]), 12), i))
    assigned_y = [float(intruders[assignment[i], 1]) for i in order]
    return sum(assigned_y[a] > assigned_y[b]+1e-9
               for a in range(TWENTY_COUNT) for b in range(a+1, TWENTY_COUNT))


def build_audit():
    defenders, intruders = initial_positions()
    active = np.ones(TWENTY_COUNT, dtype=bool)
    assignment = nearest_distance_assignment(defenders, intruders, active, active)
    expected = set(range(TWENTY_COUNT))
    if set(assignment) != expected or not set(assignment.values()) <= expected:
        raise ValueError("invalid nearest-distance assignment")
    distances = np.linalg.norm(defenders-intruders[[assignment[i] for i in range(TWENTY_COUNT)]],
                               axis=1)
    if not np.isfinite(distances).all() or np.any(distances <= 0):
        raise ValueError("invalid assigned 3D distance")
    pairs = [dict(defender=i, intruder=assignment[i],
                  defender_position=defenders[i].tolist(),
                  intruder_position=intruders[assignment[i]].tolist(),
                  distance_m=float(distances[i])) for i in range(TWENTY_COUNT)]
    return dict(experiment="ideal_20v20_nearest_assignment", version=VERSION,
        assignment_method="per_defender_nearest_active_intruder_3d_distance",
        reference_note=REFERENCE_NOTE,
        defender_count=TWENTY_COUNT, intruder_count=TWENTY_COUNT,
        defender_center=list(DEFENDER_CENTER),
        defender_radius_m=TWENTY_DEFENDER_RADIUS_M,
        protected_center=list(PROTECTED_CENTER), protected_radius_m=PROTECTED_RADIUS_M,
        defender_positions=defenders.tolist(), intruder_positions=intruders.tolist(),
        assignment={str(i):assignment[i] for i in range(TWENTY_COUNT)},
        pairs=pairs, statistics=dict(total_distance_m=float(distances.sum()),
            mean_distance_m=float(distances.mean()), std_distance_m=float(distances.std()),
            minimum_distance_m=float(distances.min()),
            maximum_distance_m=float(distances.max()),
            assignment_line_crossings_xy=assignment_line_crossings_xy(defenders,intruders,assignment),
            monotonic_y_inversions=monotonic_y_inversions(defenders,intruders,assignment)))
