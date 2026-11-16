"""Post-episode diagnostics; never called from the online guidance callback."""
import math

import numpy as np

from env.ideal_interception.trajectory import Reference
from prediction.intercept_prediction import solve_constant_velocity_pip


def _stats(values, prefix):
    if not values:
        return {f"{prefix}_{name}":None for name in ("mean","median","p95","max")}
    return {f"{prefix}_mean":float(np.mean(values)),
            f"{prefix}_median":float(np.median(values)),
            f"{prefix}_p95":float(np.percentile(values,95)),
            f"{prefix}_max":float(max(values))}


def analyze_trace(trace, config):
    """Attach offline truth comparisons to frames after the real run has ended."""
    frames=trace["frames"]
    spec=trace["references"][0]["specification"]
    reference=Reference(spec["start"],spec["end"],spec["speed"],spec["mode"],
                        control_points=spec["controls"] if spec["mode"]=="bezier" else None)
    horizon_errors=[]
    point_errors=[]
    angle_errors=[]
    future_errors=[]
    absolute_times=[]
    predicted_points=[]
    ranges=[]
    closing_rates=[]
    decreasing=0
    predictive_total=0
    beyond_reference=0
    zuo_alignments=[]
    zuo_in_bounds=[]
    zuo_below=zuo_above=0
    first_zuo_ready=None
    last_predicted_center_time=None
    for before,after in zip(frames[:-1],frames[1:]):
        control=after["control"]
        time=float(before["time"])
        own=np.asarray(before["defenders"][0][:3],float)
        target=np.asarray(before["intruders"][0][:3],float)
        end_own=np.asarray(after["defenders"][0][:3],float)
        end_target=np.asarray(after["intruders"][0][:3],float)
        distance=float(np.linalg.norm(target-own))
        end_distance=float(np.linalg.norm(end_target-end_own))
        ranges.append(distance)
        change=end_distance-distance
        duration=float(after["time"]-time)
        closing=(distance-end_distance)/duration if duration>0 else None
        if closing is not None:
            closing_rates.append(closing)
            decreasing+=change<0
        offline=dict(current_range_m=distance,range_change_m=change,
                     closing_rate_observed_mps=closing)
        horizon=control["prediction_horizon_s"]
        point=control["predicted_target_position_m"]
        if horizon is not None and point is not None:
            predictive_total+=1
            point=np.asarray(point,float)
            absolute_time=time+horizon
            offline["predicted_absolute_pip_time_s"]=absolute_time
            beyond=bool(absolute_time>reference.duration)
            offline["prediction_beyond_reference"]=beyond
            beyond_reference+=beyond
            absolute_times.append(absolute_time)
            predicted_points.append(point)
            last_predicted_center_time=absolute_time
            if control["method"]=="zuo_tp":
                los=(target-own)/distance
                direction=np.asarray(control["guidance_direction_unit"],float)
                alignment=float(los@direction)
                in_bounds=bool(np.all(point>=config.lower) and np.all(point<=config.upper))
                zuo_alignments.append(alignment)
                zuo_in_bounds.append(in_bounds)
                zuo_below+=bool(point[2]<config.lower[2])
                zuo_above+=bool(point[2]>config.upper[2])
                offline.update(prediction_los_alignment=alignment,
                               predicted_point_in_bounds=in_bounds)
                if first_zuo_ready is None:
                    first_zuo_ready=dict(decision_time_s=time,
                        defender_position_m=own.tolist(),intruder_position_m=target.tolist(),
                        estimated_target_velocity_mps=control["estimated_target_velocity_mps"],
                        estimated_target_speed_mps=float(np.linalg.norm(
                            control["estimated_target_velocity_mps"])),
                        horizon_s=horizon,predicted_point_m=point.tolist(),
                        predicted_point_in_bounds=in_bounds,
                        current_los_direction_unit=los.tolist(),
                        predicted_direction_unit=direction.tolist(),alignment=alignment,
                        angle_deg=math.degrees(math.acos(float(np.clip(alignment,-1,1)))))
            if config.intruder_mode=="straight":
                true_velocity=reference.sample(time)[1]
                oracle=solve_constant_velocity_pip(own,config.defender_speeds[0],
                                                    target,true_velocity)
                offline["oracle_status"]=oracle.status
                if oracle.status=="READY":
                    oracle_point=np.asarray(oracle.predicted_interception_position_m)
                    error=horizon-oracle.time_to_pip_s
                    point_error=float(np.linalg.norm(point-oracle_point))
                    oracle_direction=oracle_point-own
                    oracle_direction/=np.linalg.norm(oracle_direction)
                    direction=np.asarray(control["guidance_direction_unit"],float)
                    angle=math.degrees(math.acos(float(np.clip(direction@oracle_direction,-1,1))))
                    offline.update(oracle_horizon_s=oracle.time_to_pip_s,
                        oracle_point_m=oracle.predicted_interception_position_m,
                        oracle_absolute_pip_time_s=time+oracle.time_to_pip_s,
                        horizon_error_s=error,absolute_horizon_error_s=abs(error),
                        predicted_point_error_m=point_error,guidance_angle_error_deg=angle)
                    horizon_errors.append(abs(error))
                    point_errors.append(point_error)
                    angle_errors.append(angle)
            if absolute_time<=reference.duration:
                truth=reference.sample(absolute_time)[0]
                error=float(np.linalg.norm(point-truth))
                offline["future_position_error_m"]=error
                future_errors.append(error)
        after["offline_evaluation"]=offline
    contact=next((event for event in trace["result"]["events"]
                  if event["kind"]=="interception_contact"),None)
    final_range=float(np.linalg.norm(np.asarray(frames[-1]["intruders"][0][:3])-
                                     np.asarray(frames[-1]["defenders"][0][:3])))
    contact_centers=contact["positions"] if contact else None
    contact_midpoint=(np.mean(contact_centers,axis=0).tolist() if contact else None)
    defender_positions=np.asarray([frame["defenders"][0][:3] for frame in frames],float)
    boundary=next((event for event in trace["result"]["events"]
                   if event["kind"]=="boundary_exit" and 0 in event["members"]),None)
    result=dict(initial_range_m=ranges[0],minimum_range_m=min(ranges+[final_range]),
        final_range_m=final_range,
        defender_final_position_m=defender_positions[-1].tolist(),
        defender_min_altitude_m=float(defender_positions[:,2].min()),
        defender_max_altitude_m=float(defender_positions[:,2].max()),
        defender_boundary_event=boundary,
        first_zuo_ready_prediction=first_zuo_ready,
        prediction_forward_fraction=(sum(a>0 for a in zuo_alignments)/len(zuo_alignments)
            if zuo_alignments else None),
        prediction_backward_fraction=(sum(a<=0 for a in zuo_alignments)/len(zuo_alignments)
            if zuo_alignments else None),
        predicted_point_in_bounds_fraction=(sum(zuo_in_bounds)/len(zuo_in_bounds)
            if zuo_in_bounds else None),
        predicted_point_below_floor_count=zuo_below,
        predicted_point_above_ceiling_count=zuo_above,
        predicted_point_behind_defender_count=sum(a<=0 for a in zuo_alignments),
        mean_positive_closing_rate_mps=float(np.mean([v for v in closing_rates if v>0]))
            if any(v>0 for v in closing_rates) else None,
        fraction_of_steps_range_decreasing=decreasing/len(closing_rates) if closing_rates else None,
        actual_contact_time_s=contact["time"] if contact else None,
        actual_contact_midpoint_m=contact_midpoint,
        actual_contact_centers_m=contact_centers,
        final_predicted_center_intercept_time_s=last_predicted_center_time if contact else None,
        time_lead_due_to_contact_radius_s=(last_predicted_center_time-contact["time"])
            if contact and last_predicted_center_time is not None else None,
        predicted_absolute_pip_time_std_s=float(np.std(absolute_times)) if absolute_times else None,
        predicted_pip_position_drift_m=(float(np.max(np.linalg.norm(
            np.asarray(predicted_points)-predicted_points[0],axis=1)))
            if predicted_points else None),
        oracle_scored_steps=len(horizon_errors),
        predictive_steps_total=predictive_total,
        future_truth_scored_steps=len(future_errors),
        future_truth_unscored_steps=predictive_total-len(future_errors),
        future_truth_coverage=len(future_errors)/predictive_total if predictive_total else None,
        prediction_beyond_reference_steps=beyond_reference,
        prediction_beyond_reference_fraction=beyond_reference/predictive_total if predictive_total else None)
    result.update(_stats(horizon_errors,"absolute_horizon_error_s"))
    result.update(_stats(point_errors,"predicted_point_error_m"))
    result.update(_stats(angle_errors,"guidance_angle_error_deg"))
    result.update(_stats(future_errors,"future_position_error_m"))
    return result
