"""Deterministic 1v1 ideal predictive guidance, separate from MAPPO."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from env.ideal_interception import InterceptionConfig
from env.ideal_interception.geometry import closest
from env.ideal_interception.trajectory import Reference
from prediction.history import IntruderHistory
from prediction.intercept_prediction import (estimate_zuo_prediction_from_history,
    estimate_zuo_pip_time_from_history,estimate_three_point_pip_from_history)
from tools.ideal_interception_trace import record, write_trace
from .ideal_guidance_analysis import analyze_trace,compare_pip_estimators_on_trace


METHODS=("zuo_tp","zuo_tp_pip_time","three_point_pip")


def canonical_1v1_config(intruder_mode="straight"):
    """Dedicated deterministic guidance geometry; training defaults stay untouched."""
    base=InterceptionConfig(intruder_mode=intruder_mode)
    return replace(base,defender_starts=((-100.,-30.,50.),)+base.defender_starts[1:],
        intruder_starts=((100.,0.,70.),)+base.intruder_starts[1:],
        intruder_ends=((-220.,0.,70.),)+base.intruder_ends[1:])


def _minimum_pair_distance(trace):
    spec=trace["references"][0]["specification"]
    path=Reference(spec["start"],spec["end"],spec["speed"],spec["mode"],
                   control_points=spec["controls"] if spec["mode"]=="bezier" else None)
    minimum=float("inf")
    for before,after in zip(trace["frames"][:-1],trace["frames"][1:]):
        if not before["defender_mask"][0] or not before["intruder_mask"][0]:
            continue
        start,end=before["time"],after["time"]
        velocity=np.asarray(after["executed_velocities"][0],float)
        own=np.asarray(before["defenders"][0][:3],float)
        for left,right,target,target_velocity in path.segments(start,end):
            point=own+velocity*(left-start)
            minimum=min(minimum,closest(point,velocity,target,target_velocity,right-left))
    return minimum if np.isfinite(minimum) else None


def run_episode(config=None,method="zuo_tp",seed=17):
    config=canonical_1v1_config() if config is None else config
    if config.scale!="1v1" or method not in METHODS:
        raise ValueError("deterministic guidance supports 1v1 and three named methods only")
    history=IntruderHistory(config.dt,slots=2)
    controls=[]
    def decide(info):
        time=float(info["time"])
        history.observe(time,info["intruders"][:,:3],info["intruder_mask"],info["scenario_id"])
        target_index=info["assignment"].get(0)
        if target_index is None or not info["defender_mask"][0] or not info["intruder_mask"][target_index]:
            raise RuntimeError("1v1 active assignment missing")
        own=np.asarray(info["defenders"][0,:3],float)
        target=np.asarray(info["intruders"][target_index,:3],float)
        if method=="zuo_tp":
            prediction=estimate_zuo_prediction_from_history(history,target_index,own,
                config.defender_speeds[0])
            active_mode="ZUO_TP"
        elif method=="zuo_tp_pip_time":
            prediction=estimate_zuo_pip_time_from_history(history,target_index,own,
                config.defender_speeds[0])
            active_mode="ZUO_TP_PIP_TIME"
        else:
            prediction=estimate_three_point_pip_from_history(history,target_index,own,
                config.defender_speeds[0])
            active_mode="THREE_POINT_PIP"
        if prediction.status=="READY":
            point=np.asarray(prediction.predicted_target_position_m,float)
            mode=active_mode
            reason=None
        elif prediction.status=="INSUFFICIENT_HISTORY":
            point=target
            mode="WARMUP_PP"
            reason=None
        else:
            point=target
            mode="FALLBACK_PP"
            reason=prediction.status
        delta=point-own
        length=float(np.linalg.norm(delta))
        if not np.isfinite(delta).all() or not np.isfinite(length) or length<=1e-12:
            # The newest observed target position is the only fallback point.
            delta=target-own
            length=float(np.linalg.norm(delta))
            mode="FALLBACK_PP"
            reason="INVALID_DIRECTION"
        if not np.isfinite(length) or length<=1e-12:
            raise FloatingPointError("no finite 1v1 guidance direction")
        direction=delta/length
        action=np.full((5,3),.5)
        action[0]=.5+.5*direction
        control=dict(decision_time_s=time,method=method,guidance_mode=mode,
            assigned_target=target_index,prediction_status=prediction.status,
            prediction_horizon_s=prediction.horizon_s,
            predicted_target_position_m=prediction.predicted_target_position_m,
            estimated_target_velocity_mps=prediction.estimated_target_velocity_mps,
            guidance_direction_unit=direction.tolist(),action=action.tolist(),
            fallback_reason=reason,pip_equation_degree=prediction.pip_equation_degree,
            pip_discriminant=prediction.pip_discriminant)
        controls.append(control)
        return action,control
    trace=record(config,seed,actions=decide)
    minimum_separation=_minimum_pair_distance(trace)
    horizons=[row["prediction_horizon_s"] for row in controls
              if row["guidance_mode"] in ("ZUO_TP","ZUO_TP_PIP_TIME","THREE_POINT_PIP")]
    summary=dict(method=method,seed=seed,intruder_policy=config.intruder_policy,
        intruder_mode=config.intruder_mode,
        intruder_path_sha256=trace["references"][0]["specification"]["polyline_sha256"],
        outcome=trace["result"]["outcome"],time_s=trace["result"]["time"],
        steps=len(controls),interception_count=sum(event["kind"]=="interception_contact"
            for event in trace["result"]["events"]),
        minimum_separation_m=minimum_separation,
        minimum_center_separation_m=minimum_separation,
        predictive_steps=sum(row["guidance_mode"] in ("ZUO_TP","ZUO_TP_PIP_TIME","THREE_POINT_PIP") for row in controls),
        warmup_steps=sum(row["guidance_mode"]=="WARMUP_PP" for row in controls),
        fallback_steps=sum(row["guidance_mode"]=="FALLBACK_PP" for row in controls),
        mean_prediction_horizon_s=float(np.mean(horizons)) if horizons else None,
        min_prediction_horizon_s=min(horizons) if horizons else None,
        max_prediction_horizon_s=max(horizons) if horizons else None,
        no_positive_pip_solution_steps=sum(row["prediction_status"]=="NO_POSITIVE_PIP_SOLUTION"
                                           for row in controls))
    summary.update(analyze_trace(trace,config))
    if method=="zuo_tp_pip_time":
        summary.update(compare_pip_estimators_on_trace(trace,config))
    return summary,trace


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method",choices=METHODS,required=True)
    parser.add_argument("--intruder-policy",choices=("fixture","zuo_bezier"),default="fixture")
    parser.add_argument("--intruder-mode",choices=("straight","bezier"),default=None,
                        help="fixture reference only; omit for zuo_bezier")
    parser.add_argument("--seed",type=int,default=17)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args(argv)
    if args.intruder_policy=="zuo_bezier" and args.intruder_mode is not None:
        parser.error("--intruder-mode is only valid for fixture references")
    config=(InterceptionConfig(scale="1v1") if args.intruder_policy=="zuo_bezier"
            else canonical_1v1_config(args.intruder_mode or "straight"))
    summary,trace=run_episode(config,
                              args.method,args.seed)
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/"summary.json").open("x",encoding="utf-8") as stream:
        json.dump(summary,stream,indent=2,allow_nan=False)
    write_trace(args.out/"episode.json",trace)
    print(f"method={summary['method']} outcome={summary['outcome']} "
          f"time={summary['time_s']:.3f}s steps={summary['steps']} "
          f"min_separation={summary['minimum_separation_m']:.4f}m")


if __name__=="__main__":
    main()
