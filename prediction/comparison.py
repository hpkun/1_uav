"""Paired offline Zuo/Tan prediction on the same recorded intruder observations."""
import gzip
import json
from pathlib import Path

import numpy as np

from .evaluation import _metrics
from .three_point import predict_two_steps
from .zuo_linear import predict_zuo_two_steps
from .zuo_linear import predict_zuo_at
from .three_point import predict_at, recommended_horizon

PROTOCOL = "uav-ideal-blue-observations-v1"
METHODS = ("zuo_linear","tan_quadratic")


def read_observations(path):
    opener=gzip.open if Path(path).suffix==".gz" else open
    with opener(path,"rt",encoding="utf-8") as stream:
        data=json.load(stream)
    if data.get("protocol")!=PROTOCOL:
        raise ValueError("unsupported blue observation protocol")
    if data.get("target_count")!=len(data.get("targets",[])) or data["target_count"] not in (1,2):
        raise ValueError("invalid target count")
    dt=float(data["dt"])
    if not np.isfinite(dt) or dt<=0:
        raise ValueError("invalid observation dt")
    for j,target in enumerate(data["targets"]):
        observations=target["observations"]
        if target["target_index"]!=j or not observations:
            raise ValueError(f"invalid I{j} observation identity")
        if len(observations)!=target["number_of_regular_samples"]:
            raise ValueError(f"invalid I{j} observation count")
        for k,row in enumerate(observations):
            if (not row["active"] or not np.isclose(row["time"],k*dt,rtol=1e-9,atol=1e-9)
                    or row["time"]>target["reference_duration"]+1e-9
                    or np.shape(row["position"])!=(3,) or not np.isfinite(row["position"]).all()):
                raise ValueError(f"invalid I{j} regular observation {k}")
    return data


def compare(data):
    if data.get("protocol")!=PROTOCOL:
        raise ValueError("unsupported blue observation protocol")
    dt=float(data["dt"])
    rows=[]
    target_summaries={}
    for target in data["targets"]:
        j=target["target_index"]
        observations=target["observations"]
        # Neither predictor receives a future observation or the Reference object.
        for k in range(2,len(observations)):
            history=np.asarray([observations[n]["position"] for n in (k-2,k-1,k)],dtype=np.float64)
            forecasts={"zuo_linear":predict_zuo_two_steps(history[1:],dt),
                       "tan_quadratic":predict_two_steps(history,dt)}
            for horizon in (1,2):
                future=observations[k+horizon] if k+horizon<len(observations) else None
                actual=np.asarray(future["position"],float) if future is not None else None
                for method in METHODS:
                    estimate=forecasts[method][horizon-1]
                    rows.append(dict(scenario_id=data["scenario_id"],seed=data["seed"],
                        target_index=j,observation_time=observations[k]["time"],method=method,
                        horizon_steps=horizon,predicted_time=observations[k]["time"]+horizon*dt,
                        predicted_x=float(estimate[0]),predicted_y=float(estimate[1]),
                        predicted_z=float(estimate[2]),
                        actual_x=float(actual[0]) if actual is not None else None,
                        actual_y=float(actual[1]) if actual is not None else None,
                        actual_z=float(actual[2]) if actual is not None else None,
                        error_m=float(np.linalg.norm(estimate-actual)) if actual is not None else None,
                        valid=actual is not None,
                        predicted_displacement_m=float(np.linalg.norm(estimate-history[-1]))))
        method_metrics={method:_metrics([row for row in rows if row["target_index"]==j
                                         and row["method"]==method],None) for method in METHODS}
        for horizon in (1,2):
            counts={method:method_metrics[method][f"ValidSamples_{horizon}"] for method in METHODS}
            if len(set(counts.values()))!=1:
                raise AssertionError(f"unpaired I{j} horizon {horizon} samples")
        difference={}
        for horizon in (1,2):
            zuo=method_metrics["zuo_linear"][f"MAE_{horizon}"]
            tan=method_metrics["tan_quadratic"][f"MAE_{horizon}"]
            difference[f"MAE_difference_{horizon}_m"]=(zuo-tan) if zuo is not None else None
            difference[f"relative_improvement_{horizon}"]=(zuo-tan)/zuo if zuo is not None and zuo>1e-9 else None
        target_summaries[f"I{j}"]=dict(reference_duration=target["reference_duration"],
            first_observation_time=target["first_observation_time"],
            last_regular_sample_time=target["last_regular_sample_time"],
            number_of_regular_samples=target["number_of_regular_samples"],
            number_of_paired_predictions={str(h):method_metrics["zuo_linear"][f"ValidSamples_{h}"]
                                          for h in (1,2)},methods=method_metrics,difference=difference)
    return dict(protocol="uav-ideal-prediction-comparison-v1",source_protocol=data["protocol"],
        scenario_id=data["scenario_id"],seed=data["seed"],scale=data["scale"],
        intruder_mode=data["intruder_mode"],initialization=data["initialization"],
        dt=dt,targets=target_summaries,predictions=rows)


def compare_horizons(data, horizon_steps=(1, 2, 5, 10)):
    """Paired errors at arbitrary positive grid horizons; legacy compare stays intact."""
    if data.get("protocol") != PROTOCOL:
        raise ValueError("unsupported blue observation protocol")
    horizons = tuple(horizon_steps)
    if not horizons or any(type(h) is not int or h < 1 for h in horizons) or tuple(sorted(set(horizons))) != horizons:
        raise ValueError("horizon_steps must be unique increasing positive integers")
    dt = float(data["dt"])
    result = {}
    for target in data["targets"]:
        positions = np.asarray([row["position"] for row in target["observations"]], dtype=np.float64)
        per_method = {method: {} for method in METHODS}
        for horizon in horizons:
            errors = {method: [] for method in METHODS}
            for k in range(2, len(positions)-horizon):
                offset = horizon*dt
                predictions = {
                    "zuo_linear": predict_zuo_at(positions[k-1:k+1], dt, [offset])[0],
                    "tan_quadratic": predict_at(positions[k-2:k+1], dt, [offset])[0],
                }
                for method in METHODS:
                    errors[method].append(float(np.linalg.norm(predictions[method]-positions[k+horizon])))
            for method in METHODS:
                values = np.asarray(errors[method], dtype=float)
                per_method[method][str(horizon)] = dict(
                    ValidSamples=int(len(values)),
                    MAE=float(np.mean(values)) if len(values) else None,
                    RMSE=float(np.sqrt(np.mean(values**2))) if len(values) else None,
                    P95Error=float(np.percentile(values, 95)) if len(values) else None,
                    MaxError=float(np.max(values)) if len(values) else None)
        result[f"I{target['target_index']}"] = per_method
    return dict(protocol="uav-ideal-prediction-horizons-v1",source_protocol=data["protocol"],
                scenario_id=data["scenario_id"],seed=data["seed"],dt=dt,
                horizon_steps=horizons,targets=result)


def compare_positions(data, horizon_steps=(1, 5, 10, 20), tan_limit=None):
    """One row per shared truth time with both raw predictions and errors."""
    if data.get("protocol") != PROTOCOL:
        raise ValueError("unsupported blue observation protocol")
    horizons = tuple(horizon_steps)
    if not horizons or any(type(h) is not int or h < 1 for h in horizons) or tuple(sorted(set(horizons))) != horizons:
        raise ValueError("horizon_steps must be unique increasing positive integers")
    dt = float(data["dt"])
    rows, summary = [], {}
    for target in data["targets"]:
        j = target["target_index"]
        observations = target["observations"]
        positions = np.asarray([row["position"] for row in observations], dtype=np.float64)
        by_horizon = {}
        for h in horizons:
            selection = []
            for k in range(2, len(positions)-h):
                history = positions[k-2:k+1]
                offset = h*dt
                zuo = predict_zuo_at(history[-2:], dt, [offset])[0]
                tan = predict_at(history, dt, [offset])[0]
                truth = positions[k+h]
                cap = recommended_horizon(history,dt,**tan_limit) if tan_limit is not None else None
                row = dict(scenario_id=data["scenario_id"],seed=data["seed"],target_index=j,
                    observation_time=observations[k]["time"],prediction_time=observations[k+h]["time"],
                    horizon_steps=h,horizon_s=offset,
                    **{f"zuo_{axis}":float(value) for axis,value in zip("xyz",zuo)},
                    **{f"three_point_{axis}":float(value) for axis,value in zip("xyz",tan)},
                    **{f"true_{axis}":float(value) for axis,value in zip("xyz",truth)},
                    position_gap_m=float(np.linalg.norm(tan-zuo)),
                    zuo_error_m=float(np.linalg.norm(zuo-truth)),
                    three_point_error_m=float(np.linalg.norm(tan-truth)),
                    tan_horizon_cap_s=cap,
                    tan_horizon_status=("HORIZON_LIMITED" if cap is not None and offset>cap else "READY"))
                row["error_improvement_m"] = row["zuo_error_m"]-row["three_point_error_m"]
                rows.append(row)
                selection.append(row)
            def metrics(key):
                values = np.asarray([row[key] for row in selection])
                return dict(MAE=float(np.mean(values)) if len(values) else None,
                    RMSE=float(np.sqrt(np.mean(values**2))) if len(values) else None,
                    P95Error=float(np.percentile(values,95)) if len(values) else None,
                    MaxError=float(np.max(values)) if len(values) else None)
            by_horizon[str(h)] = dict(valid_samples=len(selection),
                unavailable_future_samples=min(h,len(positions)-2),
                tan_horizon_limited=sum(row["tan_horizon_status"]=="HORIZON_LIMITED" for row in selection),
                zuo=metrics("zuo_error_m"),three_point=metrics("three_point_error_m"),
                position_gap=metrics("position_gap_m"),
                mean_error_improvement_m=float(np.mean([row["error_improvement_m"] for row in selection])) if selection else None)
        summary[f"I{j}"] = by_horizon
    return dict(protocol="uav-ideal-position-comparison-v1",source_protocol=data["protocol"],
        scenario_id=data["scenario_id"],seed=data["seed"],dt=dt,
        horizon_steps=list(horizons),tan_limit=tan_limit,targets=summary,rows=rows)
