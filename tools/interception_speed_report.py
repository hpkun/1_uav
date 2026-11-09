"""Plots, deterministic case selection and replay exports for completed tier audits."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import json
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from statistics import median
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from algorithm.common.research import atomic_json
from algorithm.common.serialization import serializable
from algorithm.interception import episode, reference_action
from env.interception.config import VERSION
from env.interception.motion import advance, execute_velocity
from tools.interception_replay import render, render_video
from tools.interception_timeline import Timeline
from tools.interception_speed_tiers import TASK_NAMES, THRESHOLDS, TIERS, config
from tools.interception_high_speed_audit import POLICIES


def turn_evidence():
    results = {}
    for tier, (red, blue) in TIERS.items():
        c = config(tier, "S2")
        for side, limit in (("red", red), ("blue", blue)):
            for angle in (90,180):
                state = np.r_[np.zeros(3), limit, 0., 0.]
                request = np.array([0.,limit,0.] if angle == 90 else [-limit,0.,0.])
                rows, crossing = [], None
                for k in range(101):
                    command = execute_velocity(request, state[3:], limit, 2.5, c)
                    accel = (command-state[3:])/2.5
                    speed = np.linalg.norm(state[3:5])
                    horizontal = np.linalg.norm(accel[:2])
                    curvature = abs(state[3]*accel[1]-state[4]*accel[0])/speed**3 if speed >= 1 else None
                    assert speed <= limit+1e-12 and horizontal <= 6+1e-12
                    rows.append(dict(time=k/10, state=state.tolist(), horizontal_acceleration=horizontal,
                                     horizontal_curvature=curvature))
                    if crossing is None and (state[4] >= state[3]-1e-12 if angle == 90 else state[3] <= 0):
                        crossing = k/10
                    state = advance(state, command, 2.5, .1)
                results[f"{tier}_{side}_{angle}"] = dict(
                    half_turn_or_reverse_s=crossing,
                    max_acceleration=max(row["horizontal_acceleration"] for row in rows),
                    max_curvature=max(row["horizontal_curvature"] for row in rows if row["horizontal_curvature"] is not None),
                    samples=rows)
    return results


def choose(group):
    rows = group["episodes"]
    hits = [r for r in rows if r["outcome"] == "success"]
    if hits:
        middle = median(r["time"] for r in hits)
        return min(hits, key=lambda r: (abs(r["time"]-middle),r["seed"])), "median_success_time"
    return min(rows, key=lambda r: (r["minimum_red_blue_m"],r["seed"])), "smallest_continuous_miss"


def record(output, tier, task, chosen, criterion):
    c = config(tier, task)
    seed = chosen["seed"]
    row, frames = episode(c, seed, {"scenario":"random"}, "predictive", trace=True, audit=True)
    assert row["outcome"] == chosen["outcome"]
    np.testing.assert_allclose(row["time"], chosen["time"], rtol=0, atol=1e-12)
    output.mkdir()
    closest = min(row["closest_targets"], key=lambda p: p["center_distance"])
    diagnostics = [closest] if row["outcome"] != "success" else []
    data = dict(version=VERSION, trace_version=2, task=task, seed=seed, policy="predictive",
                selection=criterion, config=serializable(asdict(c)), result=row, frames=frames,
                diagnostic_events=diagnostics)
    path = output/"trajectory.json.gz"
    with gzip.open(path,"wt",encoding="utf-8") as stream:
        json.dump(serializable(data),stream,allow_nan=False)
    timeline=Timeline(path)
    for contact in row["contact_events"]:
        assert abs(contact["center_distance"]-1.) < 1e-7
        at_contact=timeline.at(contact["time"])
        at_end=timeline.at(timeline.duration)
        for i, (_,_,_,_,_,entity) in enumerate(timeline.entities):
            if entity in contact["entity_ids"]:
                endpoint=contact["entity_ids"].index(entity)
                for sample in (at_contact,at_end):
                    assert sample["entities"][i]["captured"] and not sample["entities"][i]["active"]
                    np.testing.assert_array_equal(sample["entities"][i]["position"],contact["positions"][endpoint])
    render(path, output/"replay.html")
    details = dict(tier=tier, task=task, seed=seed, criterion=criterion,
                   time=row["time"], outcome=row["outcome"], closest=closest, contacts=row["contact_events"])
    atomic_json(output/"selection.json",details)
    return data, details


def plots(groups, output, caption):
    labels = [f"{p}/{e}" for p,e in TIERS.values()]
    fig, axes = plt.subplots(2,2,figsize=(11,7),layout="constrained")
    for task, ax in zip(TASK_NAMES,axes.flat):
        x=np.arange(3)
        for i, policy in enumerate(POLICIES):
            values=[groups[tier,task,policy]["statistics"]["success_rate"] for tier in TIERS]
            ax.bar(x+(i-1.5)*.2,values,.2,label=policy)
        ax.set(title=task,xticks=x,xticklabels=labels,ylim=(0,1.05),ylabel="Full contact success rate")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Full interception by scene and controller\n"+caption)
    fig.savefig(output/"fig1_success.png",dpi=150)
    plt.close(fig)

    fig, ax=plt.subplots(figsize=(8,5),layout="constrained")
    values=[[r["minimum_red_blue_m"] for r in groups[t,"S2","predictive"]["episodes"]] for t in TIERS]
    ax.boxplot(values)
    ax.set(xticks=range(1,4),xticklabels=labels,ylabel="Continuous minimum distance (m)",
           title="S2 · predictive (10 s)\n"+caption)
    ax.axhline(1,color="k",ls="--",label="1 m"); ax.legend()
    fig.savefig(output/"fig2_s2_distance.png",dpi=150); plt.close(fig)

    fig, ax=plt.subplots(figsize=(8,5),layout="constrained")
    for t,label in zip(TIERS,labels):
        m=groups[t,"S2","predictive"]["statistics"]
        ax.plot(range(5),[m["per_target_entered_counts"][str(d)]/m["target_samples"] for d in THRESHOLDS],
                "o-",label=label)
    ax.set(xticks=range(5),xticklabels=THRESHOLDS,xlabel="Radius (m)",ylabel="Target entry fraction",
           ylim=(-.02,1.03),title="S2 · predictive (10 s)\n"+caption); ax.legend()
    fig.savefig(output/"fig3_thresholds.png",dpi=150); plt.close(fig)

    fig, axes=plt.subplots(2,2,figsize=(11,7),layout="constrained")
    for task, ax in zip(TASK_NAMES,axes.flat):
        breach=[[r["time"] for r in groups[t,task,"predictive"]["episodes"] if r["outcome"]=="breach"]
                for t in TIERS]
        for i, values in enumerate(breach,1):
            if values:
                ax.boxplot([values],positions=[i],widths=.5)
        ax.set(title=task,xticks=range(1,4),
               xticklabels=[f"{label}\n(n={len(v)})" for label,v in zip(labels,breach)],ylabel="Breach time (s)")
    fig.suptitle("Breach times among breaches · predictive (10 s)\n"+caption)
    fig.savefig(output/"fig4_breach_time.png",dpi=150); plt.close(fig)


def trajectories(cases, output, caption):
    fig=plt.figure(figsize=(6*len(cases),5),layout="constrained")
    for i, (data,details) in enumerate(cases,1):
        ax=fig.add_subplot(1,len(cases),i,projection="3d")
        for side,mask,color in (("pursuers","active_mask","#d64040"),("evaders","evader_mask","#2378bf")):
            for j, active in enumerate(data["frames"][0][mask]):
                if active:
                    path=np.array([row[side][j][:3] for row in data["frames"]])
                    ax.plot(*path.T,color=color,lw=1.3)
                    ax.scatter(*path[0],color=color,s=12)
        theta,z=np.meshgrid(np.linspace(0,2*np.pi,36),[0,35])
        ax.plot_surface(25*np.cos(theta),25*np.sin(theta),z,alpha=.15,color="green")
        marker=np.mean(details["closest"]["positions"],axis=0)
        ax.scatter(*marker,marker="x",s=65,color="#1aa575" if details["contacts"] else "#9c56bd")
        for frame in data["frames"]:
            for event in frame["events"]:
                if event["kind"]=="breach":
                    ax.scatter(*event["positions"][0],marker="s",s=35,color="#f0c400")
        p,e=TIERS[details["tier"]]
        ax.set(title=f'{details["task"]} · {p}/{e} m/s · seed {details["seed"]}\n'
                      f'{details["outcome"]}, closest {details["closest"]["center_distance"]:.3f} m',
               xlabel="X (m)",ylabel="Y (m)",zlabel="Z (m)")
    fig.suptitle("Selected cases · predictive (10 s) · red defenders / blue intruders\n"+caption)
    fig.savefig(output/"fig5_trajectories.png",dpi=150);plt.close(fig)


def paired_difference(a, b):
    """Seed-paired percentile bootstrap for descriptive 95% intervals."""
    assert [r["seed"] for r in a] == [r["seed"] for r in b]
    delta = np.array([[x["minimum_red_blue_m"]-y["minimum_red_blue_m"],
                       int(x["outcome"]=="success")-int(y["outcome"]=="success")] for x,y in zip(a,b)])
    rng=np.random.default_rng(9201)
    bootstrap=delta[rng.integers(0,len(delta),size=(10000,len(delta)))].mean(axis=1)
    return {key:dict(mean=float(delta[:,i].mean()), ci95=np.quantile(bootstrap[:,i],[.025,.975]).tolist())
            for i,key in enumerate(("minimum_distance_m","success_fraction"))}


def prediction_benchmark(cases):
    data, _ = cases[0]
    frames = data["frames"][1:101]
    samples = [{key:np.asarray(f[key]) for key in ("pursuers","evaders")} for f in frames]
    results={}
    for h in (10,20,40):
        from dataclasses import replace
        c=replace(config(cases[0][1]["tier"],"S2"),predictive_horizon=h)
        measurements=[]
        for _ in range(5):
            start=perf_counter()
            for _ in range(10):
                for state in samples:
                    reference_action(state,c,[(0,0)],predictive=True)
            measurements.append((perf_counter()-start)/ (10*len(samples))*1e6)
        results[str(h)]=dict(candidates=int(h/c.predictive_step),median_us=median(measurements))
    return results


def export_group_table(audit, output):
    fields = ["tier","task","suite","policy","episodes","success","breach","timeout",
              "friendly_collision","pursuer_boundary_violation","evader_escape","partial_capture_episodes",
              "mean_duration_s","mean_first_contact_s","mean_capture_time_s","mean_minimum_red_blue_m",
              "mean_breach_s","controller_us_per_step","mean_episode_wall_s",
              *[f"targets_within_{t}m" for t in THRESHOLDS],"target_samples"]
    with (output/"groups.csv").open("x",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fields)
        writer.writeheader()
        for tier in TIERS:
            for task in TASK_NAMES:
                d=json.loads((audit/tier/task/"summary.json").read_text(encoding="utf-8"))
                for group in d["groups"]:
                    s=group["statistics"]
                    row=group | s | s["outcomes"] | {f"targets_within_{t}m":s["per_target_entered_counts"][str(t)]
                                                       for t in THRESHOLDS}
                    writer.writerow({k:row[k] for k in fields})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    groups={(t,s,p):json.loads((args.audit/t/s/f"main_{p}.json").read_text(encoding="utf-8"))
            for t in TIERS for s in TASK_NAMES for p in POLICIES}
    seeds=groups["low","S2","predictive"]["seeds"]
    caption=f"n={len(seeds)} per group, seeds {seeds[0]}–{seeds[-1]} · speed labels: red/blue m/s"
    plots(groups,args.out,caption)
    export_group_table(args.audit,args.out)
    cases=[]
    for tier in TIERS:
        row, criterion=choose(groups[tier,"S2","predictive"])
        cases.append(record(args.out/tier,tier,"S2",row,criterion))
    for _, details in cases:
        if details["outcome"]!="success":
            tier,seed=details["tier"],details["seed"]
            same=next(r for r in groups[tier,"S1","predictive"]["episodes"] if r["seed"]==seed)
            if same["outcome"]=="success":
                record(args.out/f"{tier}_matched_S1",tier,"S1",same,"same_seed_navigation_control")
    if not any(row["outcome"]=="success" for _,row in cases):
        tier=cases[0][1]["tier"]
        seed=cases[0][1]["seed"]
        other=next(r for r in groups[tier,"S1","predictive"]["episodes"] if r["seed"]==seed)
        if other["outcome"]=="success":
            cases.append(record(args.out/"matched_S1",tier,"S1",other,"same_seed_navigation_control"))
    if all(row["outcome"]=="success" for _,row in cases):
        failures = [(r["minimum_red_blue_m"],tier,r["seed"],r) for tier in TIERS
                    for r in groups[tier,"S2","predictive"]["episodes"] if r["outcome"]!="success"]
        if failures:
            _,tier,_,row=min(failures)
            cases.append(record(args.out/"nearest_failure",tier,"S2",row,"smallest_continuous_miss"))
    trajectories(cases,args.out,caption)
    atomic_json(args.out/"selections.json",[details for _,details in cases])
    atomic_json(args.out/"turns.json",turn_evidence())
    comparisons={}
    for a,b in combinations(TIERS,2):
        comparisons[f"{a}_minus_{b}"]=paired_difference(
            groups[a,"S2","predictive"]["episodes"],groups[b,"S2","predictive"]["episodes"])
    for tier in TIERS:
        single=json.loads((args.audit/tier/"S2"/"single_predictive.json").read_text(encoding="utf-8"))
        comparisons[f"{tier}_three_minus_one"]=paired_difference(groups[tier,"S2","predictive"]["episodes"],
                                                                 single["episodes"])
    atomic_json(args.out/"paired_differences.json",comparisons)
    atomic_json(args.out/"prediction_benchmark.json",prediction_benchmark(cases))
    with ProcessPoolExecutor(max_workers=3) as pool:
        jobs=[pool.submit(render_video, path, path.with_name("replay.mp4"),12,960,540)
              for path in sorted(args.out.glob("*/trajectory.json.gz"))]
        for job in jobs:
            job.result()


if __name__=="__main__":
    main()
