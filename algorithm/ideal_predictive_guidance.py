"""Deterministic 1v1 ideal predictive guidance, separate from MAPPO."""
import argparse
import json
from pathlib import Path

import numpy as np

from env.ideal_interception import InterceptionConfig
from env.ideal_interception.geometry import closest
from env.ideal_interception.trajectory import Reference
from prediction.history import IntruderHistory
from prediction.intercept_prediction import (estimate_zuo_prediction_from_history,
    estimate_three_point_adaptive_from_history)
from tools.ideal_interception_trace import record, write_trace


METHODS=("zuo_tp","three_point_adaptive")


def _minimum_pair_distance(trace):
    spec=trace["references"][0]["specification"]
    path=Reference(spec["start"],spec["end"],spec["speed"],spec["mode"])
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


def run_episode(config=None,method="zuo_tp",seed=17,*,eta=.1,
                minimum_closing_speed=1.,maximum_horizon_s=1.,
                epsilon_acceleration=1e-6):
    config=InterceptionConfig() if config is None else config
    if config.scale!="1v1" or method not in METHODS:
        raise ValueError("deterministic guidance supports 1v1 and two named methods only")
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
        else:
            prediction=estimate_three_point_adaptive_from_history(history,target_index,own,
                config.defender_speeds[0],config.contact_distance,eta=eta,
                minimum_closing_speed=minimum_closing_speed,
                maximum_horizon_s=maximum_horizon_s,
                epsilon_acceleration=epsilon_acceleration)
            active_mode="THREE_POINT_ADAPTIVE"
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
            guidance_direction_unit=direction.tolist(),action=action.tolist(),
            fallback_reason=reason,range_horizon_s=prediction.range_horizon_s,
            maneuver_horizon_s=(prediction.maneuver_horizon_s if prediction.maneuver_horizon_s is None or
                                np.isfinite(prediction.maneuver_horizon_s) else "inf"),
            effective_closing_speed_mps=prediction.effective_closing_speed_mps,
            horizon_limited_by=prediction.horizon_limited_by)
        controls.append(control)
        return action,control
    trace=record(config,seed,actions=decide)
    horizons=[row["prediction_horizon_s"] for row in controls
              if row["guidance_mode"] in ("ZUO_TP","THREE_POINT_ADAPTIVE")]
    counts={name:sum(row["horizon_limited_by"]==name for row in controls)
            for name in ("range","maneuver","minimum","maximum")}
    summary=dict(method=method,seed=seed,intruder_mode=config.intruder_mode,
        outcome=trace["result"]["outcome"],time_s=trace["result"]["time"],
        steps=len(controls),interception_count=sum(event["kind"]=="interception_contact"
            for event in trace["result"]["events"]),
        minimum_separation_m=_minimum_pair_distance(trace),
        predictive_steps=sum(row["guidance_mode"] in ("ZUO_TP","THREE_POINT_ADAPTIVE") for row in controls),
        warmup_steps=sum(row["guidance_mode"]=="WARMUP_PP" for row in controls),
        fallback_steps=sum(row["guidance_mode"]=="FALLBACK_PP" for row in controls),
        mean_prediction_horizon_s=float(np.mean(horizons)) if horizons else None,
        min_prediction_horizon_s=min(horizons) if horizons else None,
        max_prediction_horizon_s=max(horizons) if horizons else None,
        **{f"{name}_limited_steps":value for name,value in counts.items()})
    return summary,trace


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method",choices=METHODS,required=True)
    parser.add_argument("--intruder-mode",choices=("straight","bezier"),default="straight")
    parser.add_argument("--seed",type=int,default=17)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args(argv)
    summary,trace=run_episode(InterceptionConfig(scale="1v1",intruder_mode=args.intruder_mode),
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
