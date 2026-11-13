"""Paired offline Zuo/Tan prediction on the same recorded intruder observations."""
import gzip
import json
from pathlib import Path

import numpy as np

from .evaluation import _metrics
from .three_point import predict_two_steps
from .zuo_linear import predict_zuo_two_steps

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
