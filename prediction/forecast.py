"""Causal, assignment-scoped target forecasts on a shared time axis."""
from dataclasses import dataclass

import numpy as np

from .three_point import predict_at as tan_at
from .three_point import recommended_horizon
from .zuo_linear import predict_zuo_at

METHODS = {"zuo_linear": (2, predict_zuo_at),
           "tan_quadratic": (3, tan_at)}


@dataclass(frozen=True)
class TargetForecast:
    target_id: int
    method: str
    observation_time: float | None
    current_position: tuple[float, float, float] | None
    future_times: tuple[float, ...]
    future_positions: tuple[tuple[float, float, float], ...]
    history_count: int
    status: str
    scenario_id: str | None
    query_offsets: tuple[float, ...] = ()
    horizon_cap_s: float | None = None
    horizon_statuses: tuple[str, ...] = ()


def forecast_assigned(history, assignment, query_times, methods=tuple(METHODS), tan_limit=None):
    """Predict each distinct active assigned target once; no defender motion is inferred.

    query_times are positive offsets from each target's latest observation.
    """
    offsets = np.asarray(query_times, dtype=np.float64)
    if (offsets.ndim != 1 or not len(offsets) or not np.isfinite(offsets).all() or
            np.any(offsets <= 0) or np.any(np.diff(offsets) <= 0)):
        raise ValueError("query_times must be increasing positive offsets")
    if not isinstance(assignment, dict) or any(type(i) is not int or type(j) is not int
                                               or i < 0 or j < 0 or j >= history.slots
                                               for i, j in assignment.items()):
        raise ValueError("invalid assignment")
    if any(method not in METHODS for method in methods):
        raise ValueError("unknown prediction method")
    result = {}
    for target_id in sorted(set(assignment.values())):
        if history.status(target_id) == "INACTIVE":
            continue
        times, positions = history.observed(target_id)
        for method in methods:
            required, predictor = METHODS[method]
            ready = history.method_status(target_id, method) == "READY"
            cap = (recommended_horizon(positions[-3:], history.dt, **tan_limit)
                   if ready and method == "tan_quadratic" and tan_limit is not None else None)
            try:
                predicted = predictor(positions[-required:], history.dt, offsets) if ready else np.empty((0, 3))
                status = "READY" if ready else "INSUFFICIENT_HISTORY"
            except FloatingPointError:
                predicted = np.empty((0, 3))
                status = "INVALID_PREDICTION"
            horizon_statuses = tuple("HORIZON_LIMITED" if cap is not None and offset > cap else "READY"
                                     for offset in offsets) if status == "READY" else ()
            result[(target_id, method)] = TargetForecast(
                target_id=target_id, method=method,
                observation_time=float(times[-1]) if len(times) else None,
                current_position=tuple(float(x) for x in positions[-1]) if len(positions) else None,
                future_times=tuple(float(times[-1]+offset) for offset in offsets) if status == "READY" else (),
                future_positions=tuple(tuple(float(x) for x in row) for row in predicted),
                history_count=required if ready else len(positions),
                status=status, scenario_id=history.scenario_id,
                query_offsets=tuple(float(x) for x in offsets),horizon_cap_s=cap,
                horizon_statuses=horizon_statuses)
    return result
