"""Paired open-loop geometry and closed-loop tracking audit, without training."""
import argparse
from collections import Counter
from dataclasses import asdict, replace
import gzip
import json
from pathlib import Path
from statistics import mean

import numpy as np

from algorithm.common.serialization import serializable, write_json
from env.preset_tracking import run_tracking
from env.preset_trajectory import MODES, VERSION, TrajectoryConfig, build_path


def _write_episode(path,result):
    with gzip.open(path,"wt",encoding="utf-8",compresslevel=3) as f:
        json.dump(serializable(result),f,allow_nan=False,separators=(",",":"))


def plot_pair(pair,config,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6*len(pair),5.8),layout="constrained")
    for j,result in enumerate(pair,1):
        ax = fig.add_subplot(1,len(pair),j,projection="3d")
        for i,item in enumerate(result["members"]):
            reference = build_path(result["kind"],item["reference"]["points"],config)
            path = np.array([reference.sample(t)[0] for t in np.linspace(0,reference.duration,250)])
            actual = np.asarray([f["actual_states"][i,:3] for f in result["frames"]])
            ax.plot(*path.T,ls="--",lw=2,color="#2765ad",label="Reference" if i==0 else None)
            ax.plot(*actual.T,lw=1.3,color="#d44d43",label="Actual" if i==0 else None)
            ax.scatter(*path[0],color="#222",marker="o",label="Start" if i==0 else None)
            ax.scatter(*path[-1],color="#328657",marker="X",label="Reference end" if i==0 else None)
            ax.scatter(*actual[-1],edgecolors="#d44d43",facecolors="none",marker="o",
                       label="Actual end" if i==0 else None)
        primary = result["members"][0]
        ax.set(xlabel="X (m)",ylabel="Y (m)",zlabel="Z (m)",
               title=f'{result["kind"]} · seed {result["seed"]}\n'
                     f'L ref/actual {primary["reference"]["length"]:.2f}/{primary["actual_length"]:.2f} m'
                     f' · mean error {primary["mean_error"]:.3f} m\n'
                     f'Actual vmax {primary["max_speed"]:.2f} m/s · aH max {primary["max_horizontal_acceleration"]:.2f} m/s²')
        ax.legend(loc="upper left",fontsize=8)
    fig.suptitle(f"Paired initial states · speed limit {config.vmax:g} m/s · reference limit {config.reference_speed:g} m/s")
    fig.savefig(output,dpi=160)
    plt.close(fig)


def aggregate(runs):
    members = [item for run in runs for item in run["members"]]
    errors = np.concatenate([np.asarray([f["position_errors"] for f in r["frames"]]).ravel() for r in runs])
    return dict(episodes=len(runs),outcomes=dict(Counter(r["outcome"] for r in runs)),
                reference_completed=sum(m["reference_completed"] for m in members),
                settled=sum(m["settled"] for m in members),members=len(members),
                mean_reference_length=mean(m["reference"]["length"] for m in members),
                mean_actual_length=mean(m["actual_length"] for m in members),
                mean_reference_duration=mean(m["reference"]["duration"] for m in members),
                mean_nominal_duration=mean(m["reference"]["nominal_duration"] for m in members),
                mean_position_error=mean(m["mean_error"] for m in members),max_position_error=max(errors),
                pooled_position_percentiles=dict(zip(("p50","p90","p99"),np.percentile(errors,[50,90,99]).tolist())),
                mean_velocity_error=mean(m["mean_velocity_error"] for m in members),
                max_actual_speed=max(m["max_speed"] for m in members),
                max_horizontal_acceleration=max(m["max_horizontal_acceleration"] for m in members),
                max_up_speed=max(m["max_vertical_speed"] for m in members),
                max_down_speed=-min(m["min_vertical_speed"] for m in members),
                max_endpoint_error=max(m["endpoint_error"] for m in members),
                constraints={name:sum(m["constraint_steps"][name] for m in members)
                             for name in ("speed","vertical","horizontal")},
                limiting_constraints=dict(Counter(m["reference"]["limiting_constraint"] for m in members)))


def audit(output,base,limits,seeds,kinds=MODES):
    output.mkdir(parents=True,exist_ok=False)
    summaries=[]
    for vmax in limits:
        config = base if vmax==base.vmax else replace(base,vmax=vmax,reference_speed=min(base.reference_speed,.8*vmax))
        folder = output/f"vmax_{vmax:g}"
        folder.mkdir()
        write_json(folder/"config.json",asdict(config))
        runs = {kind:[] for kind in kinds}
        for seed in seeds:
            pair = [run_tracking(kind,seed,config) for kind in kinds]
            if len(pair) == 2:
                np.testing.assert_array_equal(pair[0]["initial_states"],pair[1]["initial_states"])
                np.testing.assert_array_equal(pair[0]["active_mask"],pair[1]["active_mask"])
                for a,b in zip(pair[0]["members"],pair[1]["members"],strict=True):
                    np.testing.assert_array_equal(a["reference"]["points"][-1],b["reference"]["points"][-1])
            for run in pair:
                runs[run["kind"]].append(run)
                _write_episode(folder/f'{run["kind"]}_{seed}.json.gz',run)
        plot_pair([runs[k][0] for k in kinds],config,folder/"comparison.png")
        summaries.append(dict(config=asdict(config),results={k:aggregate(v) for k,v in runs.items()}))
        print(vmax,[(k,s["outcomes"]) for k,s in summaries[-1]["results"].items()],flush=True)
    write_json(output/"summary.json",dict(version=VERSION,seeds=seeds,groups=summaries,
               actual_length_method="sum of 3D chords at control-frame times",
               representative_seed=seeds[0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--config",type=Path)
    parser.add_argument("--mode",choices=MODES,help="omit to compare both reference modes")
    parser.add_argument("--limits",type=float,nargs="+")
    parser.add_argument("--episodes",type=int,default=50)
    parser.add_argument("--seed",type=int,default=1200)
    args = parser.parse_args()
    if args.episodes < 1 or args.seed < 0:
        parser.error("invalid episode count or seed")
    base = TrajectoryConfig(**(json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}))
    limits = args.limits if args.limits is not None else ([base.vmax] if args.config else [4.,7.,12.])
    if any(not np.isfinite(v) or v <= 0 for v in limits) or len(set(limits)) != len(limits):
        parser.error("speed limits must be positive, finite and distinct")
    audit(args.out,base,limits,list(range(args.seed,args.seed+args.episodes)),
          (args.mode,) if args.mode else MODES)


if __name__ == "__main__":
    main()
