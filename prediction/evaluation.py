"""Causal predictions and after-the-fact errors on recorded simulation states."""
import numpy as np

from .history import IntruderHistory


def _metrics(rows, threshold):
    result = {}
    for horizon in (1, 2):
        values = np.asarray([row["error_m"] for row in rows
                             if row["horizon_steps"] == horizon and row["valid"]], dtype=float)
        suffix = str(horizon)
        result[f"ValidSamples_{suffix}"] = int(len(values))
        for name, value in (("MAE", np.mean(values) if len(values) else None),
                            ("RMSE", np.sqrt(np.mean(values**2)) if len(values) else None),
                            ("MedianError", np.median(values) if len(values) else None),
                            ("P95Error", np.percentile(values,95) if len(values) else None),
                            ("MaxError", np.max(values) if len(values) else None)):
            result[f"{name}_{suffix}"] = float(value) if value is not None else None
        if threshold is not None:
            result[f"WithinThreshold_{suffix}"] = float(np.mean(values <= threshold)) if len(values) else None
    return result


def evaluate_trace(timeline, threshold=None):
    if threshold is not None and (not np.isfinite(threshold) or threshold < 0):
        raise ValueError("accuracy threshold must be finite and nonnegative")
    data = timeline.data
    dt = float(data["config"]["dt"])
    frames = data["frames"]
    complete = {int(frame["step"]): frame for frame in frames
                if np.isclose(frame["time"],frame["step"]*dt,rtol=1e-9,atol=1e-9)}
    history = IntruderHistory(dt, slots=2)
    history.reset(data["scenario_id"])
    rows = []
    for frame in frames:
        step = int(frame["step"])
        if step not in complete or complete[step] is not frame:
            continue
        positions = np.asarray(frame["intruders"], dtype=float)[:,:3]
        history.observe(float(frame["time"]),positions,frame["intruder_mask"],data["scenario_id"])
        for j in range(2):
            predicted = history.predict(j)
            if predicted is None:
                continue
            for horizon in (1,2):
                future = complete.get(step+horizon)
                valid = bool(future is not None and future["intruder_mask"][j])
                actual = np.asarray(future["intruders"][j][:3],float) if valid else None
                estimate = predicted[horizon-1]
                error = float(np.linalg.norm(estimate-actual)) if valid else None
                rows.append(dict(scenario_id=data["scenario_id"],seed=data["seed"],
                    target_index=j,observation_time=float(frame["time"]),
                    horizon_steps=horizon,predicted_time=float(frame["time"]+horizon*dt),
                    predicted_x=float(estimate[0]),predicted_y=float(estimate[1]),
                    predicted_z=float(estimate[2]),
                    actual_x=float(actual[0]) if valid else None,
                    actual_y=float(actual[1]) if valid else None,
                    actual_z=float(actual[2]) if valid else None,
                    error_m=error,valid=valid))
    targets={f"I{j}":_metrics([row for row in rows if row["target_index"]==j],threshold)
             for j,active in enumerate(frames[0]["intruder_mask"]) if active}
    return dict(method="three_point_quadratic",trace_version=data["trace_version"],
                environment_protocol=data["environment_protocol"],
                scenario_id=data["scenario_id"],seed=data["seed"],dt=dt,
                accuracy_threshold_m=threshold,predictions=rows,
                metrics=dict(overall=_metrics(rows,threshold),targets=targets))
