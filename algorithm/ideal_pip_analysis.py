"""Offline, same-state diagnostics for two- and three-point PIP estimates."""
import numpy as np

from env.ideal_interception.trajectory import Reference
from prediction.intercept_prediction import solve_constant_velocity_pip


FIELDS=("velocity_estimate_difference_mps","pip_horizon_difference_s",
        "pip_point_difference_m","two_point_velocity_error_mps",
        "three_point_velocity_error_mps","true_local_acceleration_mps2")


def stats(values):
    return (dict(mean=float(np.mean(values)),p95=float(np.percentile(values,95)),
                 max=float(np.max(values))) if values else
            dict(mean=None,p95=None,max=None))


def summarize_samples(rows):
    return dict(samples=len(rows),**{key:stats([row[key] for row in rows
        if row.get(key) is not None]) for key in FIELDS})


def acceleration_bins(rows, thresholds=None):
    valid=[row for row in rows if row.get("true_local_acceleration_mps2") is not None]
    magnitudes=[row["true_local_acceleration_mps2"] for row in valid]
    if thresholds is None:
        thresholds=tuple(float(x) for x in np.quantile(magnitudes,[1/3,2/3])) if valid else (None,None)
    result={}
    for label,subset in (("Low",[r for r in valid if r["true_local_acceleration_mps2"]<=thresholds[0]]),
        ("Medium",[r for r in valid if thresholds[0]<r["true_local_acceleration_mps2"]<=thresholds[1]]),
        ("High",[r for r in valid if r["true_local_acceleration_mps2"]>thresholds[1]])) if valid else ():
        result[label]=dict(samples=len(subset),
            two_point_velocity_error_mps=stats([r["two_point_velocity_error_mps"] for r in subset]),
            three_point_velocity_error_mps=stats([r["three_point_velocity_error_mps"] for r in subset]))
    return dict(thresholds_mps2=thresholds,bins=result)


def compare_on_trace(trace,config):
    """Use recorded states only, after the closed-loop episode has finished."""
    paths=[]
    for entry in trace["references"]:
        spec=entry["specification"]
        paths.append(Reference(spec["start"],spec["end"],spec["speed"],spec["mode"],
            control_points=spec["controls"] if spec["mode"]=="bezier" else None))
    frames=trace["frames"]
    rows=[]
    for k in range(2,len(frames)-1):
        before,after=frames[k:k+2]
        pairs=after["control"]["pairs"]
        for key,control in pairs.items():
            i,j=int(key),control["assigned_target"]
            if not all(frames[q]["intruder_mask"][j] for q in (k-2,k-1,k)):
                continue
            points=np.asarray([frames[q]["intruders"][j][:3] for q in (k-2,k-1,k)])
            v2=(points[2]-points[1])/config.dt
            v3=(3*points[2]-4*points[1]+points[0])/(2*config.dt)
            t=float(before["time"])
            if t>paths[j].duration:
                continue
            true_velocity=paths[j].sample(t)[1]
            row=dict(velocity_estimate_difference_mps=float(np.linalg.norm(v2-v3)),
                two_point_velocity_error_mps=float(np.linalg.norm(v2-true_velocity)),
                three_point_velocity_error_mps=float(np.linalg.norm(v3-true_velocity)),
                true_local_acceleration_mps2=None,
                pip_horizon_difference_s=None,pip_point_difference_m=None)
            if t>=config.dt and t+config.dt<=paths[j].duration:
                preceding=paths[j].sample(t-config.dt)[1]
                following=paths[j].sample(t+config.dt)[1]
                row["true_local_acceleration_mps2"]=float(np.linalg.norm(
                    (following-preceding)/(2*config.dt)))
            defender=before["defenders"][i][:3]
            two=solve_constant_velocity_pip(defender,config.defender_speeds[i],points[2],v2)
            three=solve_constant_velocity_pip(defender,config.defender_speeds[i],points[2],v3)
            if two.status==three.status=="READY":
                row["pip_horizon_difference_s"]=abs(two.time_to_pip_s-three.time_to_pip_s)
                row["pip_point_difference_m"]=float(np.linalg.norm(np.asarray(
                    two.predicted_interception_position_m)-three.predicted_interception_position_m))
            after["offline_evaluation"]["pairs"].setdefault(str(key),{}).update(row)
            rows.append(row)
    return rows
