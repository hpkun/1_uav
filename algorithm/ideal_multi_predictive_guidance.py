"""Model-free, fixed-assignment 6v6 guidance and post-episode diagnostics."""
import argparse
import json
from pathlib import Path

import numpy as np

from algorithm.common.serialization import serializable
from env.ideal_interception.config import (DEFENDER_SLOTS,INTRUDER_SLOTS,
    formal_6v6_config)
from env.ideal_interception.geometry import closest
from env.ideal_interception.trajectory import Reference
from prediction.history import IntruderHistory
from prediction.intercept_prediction import (estimate_zuo_prediction_from_history,
    estimate_zuo_pip_time_from_history,estimate_three_point_pip_from_history)
from tools.ideal_interception_trace import record,write_trace

METHODS=("zuo_tp","zuo_tp_pip_time","three_point_pip")
PREDICTORS=dict(zuo_tp=estimate_zuo_prediction_from_history,
    zuo_tp_pip_time=estimate_zuo_pip_time_from_history,
    three_point_pip=estimate_three_point_pip_from_history)
MODES=dict(zuo_tp="ZUO_TP",zuo_tp_pip_time="ZUO_TP_PIP_TIME",
    three_point_pip="THREE_POINT_PIP")


def _stats(values):
    return (dict(min=float(min(values)),mean=float(np.mean(values)),
                 p95=float(np.percentile(values,95)),max=float(max(values)))
            if values else dict(min=None,mean=None,p95=None,max=None))


def _analyze(trace,config,method):
    frames=trace["frames"]
    paths=[]
    for reference in trace["references"]:
        spec=reference["specification"]
        paths.append(Reference(spec["start"],spec["end"],spec["speed"],spec["mode"],
            control_points=spec["controls"] if spec["mode"]=="bezier" else None))
    horizons,errors=[],[]
    predictive=warmup=fallback=no_root=beyond=0
    lengths=np.zeros(DEFENDER_SLOTS)
    separation=None
    diagnostics={i:dict(predictive_steps=0,forward=0,backward=0,in_bounds=0,
                        below_floor=0) for i in range(config.defender_count)}
    for before,after in zip(frames[:-1],frames[1:]):
        t0,t1=before["time"],after["time"]
        exits={event["members"][0]:event["time"] for event in after["events"]
               if event["kind"]=="member_inactive"}
        active=np.flatnonzero(before["defender_mask"])
        velocity=np.asarray(after["executed_velocities"])
        for a,i in enumerate(active):
            duration=exits.get(int(i),t1)-t0
            lengths[i]+=float(np.linalg.norm(velocity[i]))*duration
            for j in active[a+1:]:
                common=min(exits.get(int(i),t1),exits.get(int(j),t1))-t0
                if common>=0:
                    distance=closest(np.asarray(before["defenders"][i][:3]),velocity[i],
                        np.asarray(before["defenders"][j][:3]),velocity[j],common)
                    separation=distance if separation is None else min(separation,distance)
        offline={}
        for key,control in after["control"]["pairs"].items():
            i=int(key)
            j=control["assigned_target"]
            mode=control["guidance_mode"]
            warmup+=mode=="WARMUP_PP"
            fallback+=mode=="FALLBACK_PP"
            no_root+=control["prediction_status"]=="NO_POSITIVE_PIP_SOLUTION"
            horizon=control["prediction_horizon_s"]
            if horizon is None:
                continue
            predictive+=1
            horizons.append(horizon)
            point=np.asarray(control["predicted_target_position_m"])
            absolute=t0+horizon
            beyond+=absolute>paths[j].duration
            row=dict(predicted_absolute_time_s=absolute,
                     prediction_beyond_reference=absolute>paths[j].duration)
            if absolute<=paths[j].duration:
                truth=paths[j].sample(absolute)[0]
                error=float(np.linalg.norm(point-truth))
                row["future_position_error_m"]=error
                errors.append(error)
            if method=="zuo_tp":
                own=np.asarray(before["defenders"][i][:3])
                target=np.asarray(before["intruders"][j][:3])
                los=(target-own)/np.linalg.norm(target-own)
                direction=np.asarray(control["guidance_direction_unit"])
                pair=diagnostics[i]
                pair["predictive_steps"]+=1
                pair["forward"]+=float(los@direction)>0
                pair["backward"]+=float(los@direction)<=0
                pair["in_bounds"]+=bool(np.all(point>=config.lower) and np.all(point<=config.upper))
                pair["below_floor"]+=point[2]<config.lower[2]
            offline[key]=row
        after["offline_evaluation"]={"pairs":offline}
    result=trace["result"]
    contacts=[event["time"] for event in result["events"]
              if event["kind"]=="interception_contact"]
    total=sum(row["predictive_steps"] for row in diagnostics.values())
    strict={key:(sum(row[key] for row in diagnostics.values())/total if total else None)
            for key in ("forward","backward","in_bounds","below_floor")}
    for row in diagnostics.values():
        n=row["predictive_steps"]
        for key in ("forward","backward","in_bounds","below_floor"):
            row[key+"_fraction"]=row[key]/n if n else None
    return dict(intercepted_count=result["intercepted_count"],
        breached_count=result["breached_count"],
        other_resolved_count=result["other_resolved_count"],
        remaining_intruder_count=result["remaining_intruder_count"],
        interception_ratio=result["interception_ratio"],breach_ratio=result["breach_ratio"],
        mission_success=result["mission_success"],outcome=result["outcome"],
        episode_resolution_time_s=result["time"],
        first_interception_time_s=min(contacts) if contacts else None,
        mean_interception_time_s=float(np.mean(contacts)) if contacts else None,
        last_interception_time_s=max(contacts) if contacts else None,
        defender_boundary_count=result["defender_boundary_count"],
        friendly_collision_count=result["friendly_collision_count"],
        intruder_collision_count=result["intruder_collision_count"],
        minimum_defender_defender_separation_m=separation,
        mean_defender_path_length_m=float(np.mean(lengths[:config.defender_count])),
        total_defender_path_length_m=float(sum(lengths[:config.defender_count])),
        predictive_steps_total=predictive,warmup_steps=warmup,fallback_steps=fallback,
        no_positive_pip_solution_steps=no_root,
        prediction_horizon_s=_stats(horizons),
        future_truth_scored_steps=len(errors),
        future_truth_coverage=len(errors)/predictive if predictive else None,
        prediction_beyond_reference_fraction=beyond/predictive if predictive else None,
        future_position_error_m={key:value for key,value in _stats(errors).items()
                                 if key in ("mean","p95","max")},
        strict_zuo_pair_diagnostics=diagnostics if method=="zuo_tp" else None,
        prediction_forward_fraction=strict["forward"] if method=="zuo_tp" else None,
        prediction_backward_fraction=strict["backward"] if method=="zuo_tp" else None,
        predicted_point_in_bounds_fraction=strict["in_bounds"] if method=="zuo_tp" else None,
        prediction_below_floor_fraction=strict["below_floor"] if method=="zuo_tp" else None)


def run_episode(config=None,method="zuo_tp",seed=17):
    config=formal_6v6_config() if config is None else config
    if config.scale!="6v6" or config.assignment_policy!="hungarian_fixed" or method not in METHODS:
        raise ValueError("6v6 fixed-assignment config and named guidance method required")
    history=IntruderHistory(config.dt,slots=INTRUDER_SLOTS)
    predictor=PREDICTORS[method]
    def decide(info):
        history.observe(float(info["time"]),info["intruders"][:,:3],
                        info["intruder_mask"],info["scenario_id"])
        action=np.full((DEFENDER_SLOTS,3),.5)
        pairs={}
        for i,j in sorted(info["assignment"].items()):
            own=np.asarray(info["defenders"][i,:3])
            target=np.asarray(info["intruders"][j,:3])
            prediction=predictor(history,j,own,config.defender_speeds[i])
            if prediction.status=="READY":
                point=np.asarray(prediction.predicted_target_position_m)
                mode=MODES[method]
            else:
                point=target
                mode="WARMUP_PP" if prediction.status=="INSUFFICIENT_HISTORY" else "FALLBACK_PP"
            delta=point-own
            norm=float(np.linalg.norm(delta))
            if not np.isfinite(norm) or norm<=1e-12:
                delta=target-own
                norm=float(np.linalg.norm(delta))
                mode="FALLBACK_PP"
            if not np.isfinite(norm) or norm<=1e-12:
                raise FloatingPointError("no finite multi-guidance direction")
            direction=delta/norm
            action[i]=.5+.5*direction
            pairs[i]=dict(assigned_target=j,guidance_mode=mode,
                prediction_status=prediction.status,prediction_horizon_s=prediction.horizon_s,
                predicted_target_position_m=prediction.predicted_target_position_m,
                estimated_target_velocity_mps=prediction.estimated_target_velocity_mps,
                guidance_direction_unit=direction.tolist(),
                pip_equation_degree=prediction.pip_equation_degree,
                pip_discriminant=prediction.pip_discriminant)
        return action,dict(decision_time_s=float(info["time"]),method=method,
                           action=action.tolist(),pairs=pairs)
    trace=record(config,seed,actions=decide,scenario_type="region_crossing")
    first=trace["frames"][0]
    specs=[row["specification"] for row in trace["references"]]
    result=dict(method=method,seed=seed,scenario_id=trace["scenario_id"],
        defender_starts=[first["defenders"][i][:3] for i in range(config.defender_count)],
        intruder_starts=[first["intruders"][j][:3] for j in range(config.intruder_count)],
        intruder_paths=[dict(start=spec["start"],goal=spec["end"],
            control_points=spec["controls"],duration_s=spec["duration"],
            path_sha256=spec["polyline_sha256"],
            protected_region_entry_time_s=trace["scene_diagnostics"]["paths"][j]["protected_region_entry_time"])
            for j,spec in enumerate(specs)],
        path_hashes=[spec["polyline_sha256"] for spec in specs],
        original_assignment=trace["original_assignment"],
        assignment_total_cost_m=trace["assignment_total_cost_m"],
        per_pair_initial_distance_m=trace["per_pair_initial_distance_m"],
        defender_final_status=trace["result"]["defender_status"])
    result.update(_analyze(trace,config,method))
    return serializable(result),serializable(trace)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method",required=True,choices=METHODS)
    parser.add_argument("--seed",type=int,default=17)
    parser.add_argument("--profile",choices=("unified","paper_like"),default="unified")
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args(argv)
    config=(formal_6v6_config() if args.profile=="unified" else
            formal_6v6_config(**dict(dt=.1,defender_speeds=(25.,)*DEFENDER_SLOTS,
                intruder_speeds=(30.,)*INTRUDER_SLOTS)))
    summary,trace=run_episode(config,args.method,args.seed)
    args.out.mkdir(parents=True,exist_ok=False)
    with (args.out/"summary.json").open("x",encoding="utf-8") as stream:
        json.dump(summary,stream,indent=2,allow_nan=False)
    write_trace(args.out/"episode.json",trace)
    print(f"method={args.method} intercepted={summary['intercepted_count']}/6 "
          f"breached={summary['breached_count']} outcome={summary['outcome']} "
          f"time={summary['episode_resolution_time_s']:.3f}s")


if __name__=="__main__":
    main()
