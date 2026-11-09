"""Read completed A/B journals and plot matched tracking results."""
import argparse
from pathlib import Path

import numpy as np

from algorithm.common.research import atomic_json, commit_header, journal, read_json
from env.preset_trajectory import build_path
from env.trajectory_rl import TrackingConfig


def paired_runs(straight, bezier):
    runs = []
    for directory,mode in ((straight,"straight"),(bezier,"bezier")):
        head = commit_header(directory/"latest.pt")
        run = head["run"]
        if run!=read_json(directory/"config.json") or run["environment"]["trajectory_mode"]!=mode:
            raise ValueError("A/B run identity mismatch")
        records = journal(directory,head)
        evaluations = [(r["update"],r["evaluation"]) for r in records if r["evaluation"] is not None]
        if not evaluations:
            raise ValueError("A/B runs must have evaluations")
        runs.append((run,records,evaluations))
    a,b = [r[0] for r in runs]
    for key in ("training","total_steps","interval","evaluation_seeds","device","protocol"):
        if a[key]!=b[key]:
            raise ValueError(f"A/B {key} differs")
    if any(a["environment"][k]!=b["environment"][k] for k in a["environment"] if k!="trajectory_mode"):
        raise ValueError("A/B environment differs beyond trajectory mode")
    if len(runs[0][1])!=len(runs[1][1]):
        raise ValueError("A/B run lengths differ")
    eval_pairs = 0
    for (ua,ea),(ub,eb) in zip(runs[0][2],runs[1][2],strict=True):
        if ua!=ub:
            raise ValueError("A/B evaluation updates differ")
        for ar,br in zip(ea["episodes"],eb["episodes"],strict=True):
            assert ar["seed"]==br["seed"] and ar["reset_seed"]==br["reset_seed"]
            np.testing.assert_array_equal(ar["initial_states"],br["initial_states"])
            for pa,pb in zip(ar["reference_paths"],br["reference_paths"],strict=True):
                np.testing.assert_array_equal(pa["points"][-1],pb["points"][-1])
            eval_pairs += 1
    def starts(records):
        return {(row["worker_id"],row["worker_episode_index"]):row["initial_states"] for record in records
                for row in record["completed_episodes"]}
    ta,tb = [starts(r[1]) for r in runs]
    keys = ta.keys()&tb.keys()
    for k in keys:
        np.testing.assert_array_equal(ta[k],tb[k])
    return runs,dict(evaluation_initial_pairs=eval_pairs,completed_training_initial_pairs=len(keys))


def report(straight, bezier, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs,pairing = paired_runs(straight,bezier)
    output.mkdir(parents=True,exist_ok=False)
    fig,axes = plt.subplots(3,3,figsize=(13,9),layout="constrained")
    training = ("policy_loss","critic_loss","entropy","valid_samples","actor_grad_norm","critic_grad_norm")
    evaluation = ("success_rate","mean_position_error","mean_velocity_error")
    for run,records,evals in runs:
        mode = run["environment"]["trajectory_mode"]
        for ax,key in zip(axes.flat,training):
            ax.plot([r["joint_steps"] for r in records],[r[key] for r in records],".-",label=mode)
            ax.set(title=key,xlabel="Joint environment steps")
        for ax,key in zip(axes.flat[6:],evaluation):
            by_update = {r["update"]:r["joint_steps"] for r in records}
            ax.plot([by_update[u] for u,_ in evals],[e["summary"][key] for _,e in evals],"o-",label=mode)
            ax.set(title=key,xlabel="Joint environment steps")
    axes[0,0].legend()
    fig.suptitle("Preset tracking MAPPO · independent A/B learning curves")
    fig.savefig(output/"learning_curves.png",dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(12,5.6),layout="constrained")
    finals = {}
    for j,(run,_,evals) in enumerate(runs,1):
        c = TrackingConfig(**run["environment"])
        update,data = evals[-1]
        example = data["episodes"][0]
        ax = fig.add_subplot(1,2,j,projection="3d")
        for i,path in enumerate(example["reference_paths"]):
            geometry = build_path(c.trajectory_mode,path["points"],c)
            xyz = np.asarray([geometry.sample(t)[0] for t in np.linspace(0,geometry.duration,200)])
            actual = np.asarray([r["actual_states"][i][:3] for r in data["first_trace"]])
            color = f"C{i}"
            ax.plot(*xyz.T,"--",color=color,label=f"U{i} reference")
            ax.plot(*actual.T,color=color,label=f"U{i} actual")
            ax.scatter(*xyz[0],color=color,marker="o",s=16)
            ax.scatter(*xyz[-1],color=color,marker="x",s=30)
        ax.set(xlabel="X (m)",ylabel="Y (m)",zlabel="Z (m)",
               title=f"{c.trajectory_mode} · seed {example['seed']} · {example['outcome']}")
        ax.legend(fontsize=8)
        finals[c.trajectory_mode] = dict(update=update,**data["summary"])
    fig.savefig(output/"evaluation_paths.png",dpi=150)
    plt.close(fig)
    atomic_json(output/"comparison.json",dict(pairing=pairing,final=finals))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("straight","bezier","out"):
        p.add_argument("--"+key,type=Path,required=True)
    a = p.parse_args()
    report(a.straight,a.bezier,a.out)


if __name__=="__main__":
    main()
