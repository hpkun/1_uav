"""Scripted potential audit and paired learning curves for static navigation."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from statistics import mean

import numpy as np

from algorithm.common.research import atomic_json
from algorithm.navigation import ARMS, NavigationAccumulator
from env.heading_potential import direction_quality, heading_transition
from env.navigation import NavigationConfig, NavigationEnv


def scripted(config, command):
    env = NavigationEnv(config)
    _,info = env.reset(2200)
    tracker = NavigationAccumulator()
    tracker.reset(info)
    actions = np.broadcast_to(.5+np.asarray(command)/(2*config.vmax),(5,3))
    trace = []
    while True:
        _,rewards,terminated,truncated,info = env.step(actions)
        tracker.add(info,rewards)
        trace.append({k:info[k] for k in ("heading","reward_parts","states","time","outcome")})
        if terminated or truncated:
            return dict(metrics=tracker.summary(info),trace=trace)


def reward_audit():
    base = NavigationConfig(heading_enabled=True,initial_velocity=(1.,0.,0.),start_jitter=(0.,0.,0.))
    results = {}
    for name,weight in ARMS.items():
        cfg = replace(base,heading_enabled=weight is not None,heading_weight=weight if weight is not None else .3)
        results[name] = dict(arrival=scripted(cfg,[3.,0.,0.]),
                             timeout=scripted(replace(cfg,max_steps=5),[0.,2.,0.]))
    for outcome in ("arrival","timeout"):
        assert results["off"][outcome]["metrics"]["outcome"] == ("success" if outcome=="arrival" else "timeout")
        for name in ARMS:
            case=results[name][outcome]
            assert abs(case["metrics"]["shaping_identity_error"]) < 1e-12
            for ref,row in zip(results["off"][outcome]["trace"],case["trace"],strict=True):
                np.testing.assert_array_equal(ref["states"],row["states"])
                assert ref["reward_parts"] == {k:v for k,v in row["reward_parts"].items() if k!="heading"}
    gate = {str(speed):direction_quality([[0,0,0,speed,0,0]],[10,0,0],[True],base)["quality"][0]
            for speed in (0.,.01,.1,1.,4.,7.,14.,21.)}
    held = {}
    for speed in (1.,-1.):
        s = [[0,0,0,speed,0,0]]
        held[str(speed)] = heading_transition(s,s,[10,0,0],[True],[True],1,False,base)
    return dict(scripted=results,aligned_quality_by_speed=gate,constant_quality_controls=held)


def plots(study, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs=json.loads((study/"summary.json").read_text(encoding="utf-8"))["runs"]
    initial = {}
    for run in runs:
        for point in run["curve"]:
            result=json.loads((study/run["arm"]/f"seed_{run['seed']}"/
                               f"evaluation_{point['update']:04}.json").read_text(encoding="utf-8"))
            geometry=[(row["seed"],row["initial_states"]) for row in result["episodes"]]
            assert geometry==initial.setdefault("eval",geometry)
        train=(study/run["arm"]/f"seed_{run['seed']}"/"train.jsonl").read_text(encoding="utf-8").splitlines()
        starts=[(r["worker_id"],r["worker_episode_index"],r["initial_states"])
                for line in train for r in json.loads(line)["episodes"]]
        # Different policies can finish a different number of episodes. The
        # paired worker/episode IDs still have identical initial states.
        for worker,index,states in starts:
            key=(run["seed"],worker,index)
            assert states==initial.setdefault(key,states)
    metrics=[("route_completion","Route completion"),("tracking_error_m","Cross-track error (m)"),
             ("mean_heading_error_deg","Heading error (deg)"),("turning_deg_per_s","Turning (deg/s)"),
             ("mean_speed_mps","Mean speed (m/s)"),("speed_change_mps2","Speed change (m/s²)")]
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout="constrained")
    final={}
    for arm in ARMS:
        selected=[r for r in runs if r["arm"]==arm]
        for ax,(key,title) in zip(axes.flat,metrics):
            points=list(zip(*(r["curve"] for r in selected)))
            values=[mean(row["evaluation"][key] for row in point) for point in points]
            ax.plot([point[0]["update"] for point in points],values,marker="o",label=arm)
            ax.set(title=title,xlabel="PPO update")
        final[arm]=[dict(seed=r["seed"],**r["curve"][-1]["evaluation"]) for r in selected]
    axes.flat[0].legend()
    fig.suptitle("Static waypoint navigation · paired deterministic evaluations")
    fig.savefig(output/"learning_curves.png",dpi=150)
    plt.close(fig)
    atomic_json(output/"paired_final.json",final)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--study",type=Path)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    atomic_json(args.out/"potential_audit.json",reward_audit())
    if args.study:
        plots(args.study,args.out)


if __name__=="__main__":
    main()
