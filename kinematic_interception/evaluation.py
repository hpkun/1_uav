"""Paired deterministic experiments with shared, pre-generated reference paths."""
from collections import defaultdict
from dataclasses import asdict
import math
import time

import numpy as np

from algorithm.ideal_collision_avoidance import AvoidanceConfig, apply_collision_avoidance

from .controllers import Controller
from .dynamics import Command
from .environment import KinematicInterceptionEnv, one_vs_one_scene, twenty_scene
from .prediction import TargetHistory, forecast


def run_episode(scene, model, method, *, avoidance=None, substep=.01, trace=False):
    env = KinematicInterceptionEnv(scene, model, substep=substep)
    history = TargetHistory(slots=len(scene.paths))
    controller = Controller(method,env.dynamics,history)
    avoidance = avoidance or AvoidanceConfig(mode="none")
    if len(env.states) == 1 and avoidance.mode != "none":
        raise ValueError("1v1 comparison uses no friendly avoidance")
    if method == "G4" and avoidance.mode != "none":
        raise ValueError("G4 direct-control MPC has no CPA postprocessor")
    frames = [env.snapshot()] if trace else None
    controls = [] if trace else None
    planning_time = 0.
    planning_calls = 0
    statuses = defaultdict(int)
    predicted_minima = []
    solve_times=[]
    while env.outcome == "ongoing":
        before = env.snapshot()
        history.observe(env.time, before["intruders"][:,:3], env.intruder_mask)
        commands = [None]*len(env.states)
        details = {}
        for i,j in sorted(env.assignment.items()):
            start = time.perf_counter()
            command,detail = controller.command(i,env.states[i],j,
                before["intruders"][j,:3],env.time)
            planning_time += time.perf_counter()-start
            planning_calls += 1
            statuses[detail["status"]] += 1
            if "minimum_distance_m" in detail:
                predicted_minima.append(detail["minimum_distance_m"])
            if "solve_time_s" in detail:
                solve_times.append(detail["solve_time_s"])
            details[i] = detail
            if avoidance.mode != "none":
                direction,_ = apply_collision_avoidance(avoidance,i,
                    before["defenders"][:,:3],before["defenders"][:,3:],
                    env.defender_mask,np.asarray(command.direction))
                command = Command(tuple(direction),command.speed)
            commands[i] = command
        if trace:
            controls.append(dict(time=env.time,assignment=env.assignment.copy(),
                details=details,
                commands=[None if command is None else (
                    dict(direction=command.direction,speed=command.speed)
                    if isinstance(command,Command) else dict(control=vars(command)))
                    for command in commands]))
        after = env.step(commands)
        if trace:
            frames.append(after)
    contacts = [event["time"] for event in env.events
                if event["kind"] == "interception_contact"]
    result = dict(protocol=after["protocol"],scene_id=scene.identity,model=model,
        method=method,avoidance_mode=avoidance.mode,
        dynamics=asdict(env.dynamics),substep_s=substep,
        defender_count=len(env.states),intruder_count=len(scene.paths),
        outcome=env.outcome,intercepted_count=env.interception_count,
        assigned_isr=env.interception_count/len(scene.paths),
        breached_count=env.breach_count,breach_ratio=env.breach_count/len(scene.paths),
        unassigned_contact_count=env.unassigned_contact_count,
        friendly_collision_count=env.friendly_collision_count,
        intruder_collision_count=env.intruder_collision_count,
        defender_boundary_count=env.defender_boundary_count,
        minimum_defender_separation_m=env.minimum_separation,
        first_interception_time_s=min(contacts) if contacts else None,
        mean_interception_time_s=float(np.mean(contacts)) if contacts else None,
        episode_resolution_time_s=env.time,
        defender_trajectory_length_m=env.trajectory_length.tolist(),
        minimum_active_speed_mps=env.min_active_speed,
        maximum_applied_acceleration_mps2=env.max_applied_acceleration,
        maximum_yaw_rate_radps=env.max_yaw_rate if model == "B" else None,
        maximum_flightpath_rate_radps=env.max_flightpath_rate if model == "B" else None,
        maximum_lateral_acceleration_mps2=env.max_lateral_acceleration if model == "B" else None,
        planning_calls=planning_calls,
        mean_planning_time_s=planning_time/planning_calls if planning_calls else None,
        guidance_status_counts=dict(statuses),
        solver_calls=len(solve_times),
        solver_fallbacks=statuses.get("FALLBACK",0),
        mean_solver_time_s=float(np.mean(solve_times)) if solve_times else None,
        p95_solver_time_s=float(np.percentile(solve_times,95)) if solve_times else None,
        p99_solver_time_s=float(np.percentile(solve_times,99)) if solve_times else None,
        mean_predicted_candidate_minimum_distance_m=(
            float(np.mean(predicted_minima)) if predicted_minima else None),
        path_hashes=[path.specification()["polyline_sha256"] for path in scene.paths])
    if trace:
        return result,dict(summary=result,events=env.events,
                           frames=frames,controls=controls,
                           scene=dict(lower=scene.lower,upper=scene.upper,
                               protected_center=scene.protected_center,
                               protected_radius=scene.protected_radius,
                               protected_height=scene.protected_height,
                               contact_distance=scene.contact_distance),
                           references=[dict(specification=path.specification(),
                                            points=path.points.tolist())
                                       for path in scene.paths])
    return result


def paired_experiment(seeds=(17,), *, scale=1, methods=("G0","G1","G2","G3"),
                      models=("A","B"), avoidance=None, substep=.01):
    rows = []
    for seed in seeds:
        scene = one_vs_one_scene(seed) if scale == 1 else twenty_scene(seed,scale)
        for model in models:
            for method in methods:
                row = run_episode(scene,model,method,avoidance=avoidance,
                                  substep=substep)
                row["seed"] = seed
                rows.append(row)
    return rows


def offline_forecast_experiment(scenes, horizons=(.5,1.,2.,3.,5.), dt=.05):
    """Per-path time-averaged errors; paths, not adjacent timesteps, are units."""
    records = []
    for scene in scenes:
        for j,path in enumerate(scene.paths):
            history = TargetHistory()
            samples = defaultdict(list)
            steps = int(math.floor(path.duration/dt))
            for step_index in range(steps+1):
                t = step_index*dt
                history.observe(t,np.asarray([path.sample(t)[0]]),np.array([True]))
                for method in ("CV","CA"):
                    try:
                        prediction = forecast(history,0,method)
                    except ValueError:
                        continue
                    for horizon in horizons:
                        if t+horizon > path.duration:
                            continue
                        offsets = np.linspace(0.,horizon,
                            max(2,int(math.ceil(horizon/.1))+1))
                        errors = np.array([np.linalg.norm(prediction.at(offset)-
                                           path.sample(t+offset)[0])
                                           for offset in offsets])
                        samples[(method,horizon)].append((float(np.mean(errors)),
                            float(errors[-1]),float(np.mean(errors**2))))
            for (method,horizon),errors in samples.items():
                values = np.asarray(errors)
                records.append(dict(scene_id=scene.identity,target=j,
                    path_hash=path.specification()["polyline_sha256"],
                    method=method,horizon_s=horizon,samples=len(errors),
                    ade_m=float(np.mean(values[:,0])),fde_m=float(np.mean(values[:,1])),
                    rmse_m=float(np.sqrt(np.mean(values[:,2])))))
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["method"],row["horizon_s"])].append(row)
    summary = [dict(method=method,horizon_s=horizon,paths=len(group),
                    mean_path_ade_m=float(np.mean([r["ade_m"] for r in group])),
                    mean_path_fde_m=float(np.mean([r["fde_m"] for r in group])),
                    mean_path_rmse_m=float(np.mean([r["rmse_m"] for r in group])))
               for (method,horizon),group in sorted(grouped.items())]
    return dict(summary=summary,per_path=records)
