"""Run isolated deterministic kinematic interception comparisons."""
import argparse
import json
from pathlib import Path

import torch

from algorithm.common.serialization import serializable
from algorithm.ideal_collision_avoidance import AvoidanceConfig

from .environment import one_vs_one_scene, twenty_scene
from .evaluation import (offline_forecast_experiment, paired_experiment,
                         run_episode)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=int, choices=(1,3,6,20), default=1)
    parser.add_argument("--seed", type=int, nargs="+", default=[17])
    parser.add_argument("--model", choices=("A","B"))
    parser.add_argument("--method", choices=("G0","G1","G2","G3"))
    parser.add_argument("--avoidance", choices=("none","predictive_cpa_knn"),
                        default="none")
    parser.add_argument("--offline-prediction", action="store_true")
    parser.add_argument("--substep", type=float, default=.01)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        parser.error("CUDA is required for repository evaluation audits")
    if args.scale == 1 and args.avoidance != "none":
        parser.error("friendly avoidance applies only to multi-member scenes")
    if args.out is not None and args.out.exists():
        parser.error(f"output already exists: {args.out}")
    scenes = [one_vs_one_scene(seed) if args.scale == 1 else twenty_scene(seed,args.scale)
              for seed in args.seed]
    if args.offline_prediction:
        payload = offline_forecast_experiment(scenes)
        for row in payload["summary"]:
            print(f"[forecast] method={row['method']} | horizon={row['horizon_s']} s | "
                  f"paths={row['paths']} | ADE={row['mean_path_ade_m']:.3f} m | "
                  f"FDE={row['mean_path_fde_m']:.3f} m | "
                  f"RMSE={row['mean_path_rmse_m']:.3f} m")
    else:
        avoidance = AvoidanceConfig(mode=args.avoidance)
        if args.model and args.method and len(scenes) == 1 and args.out is not None:
            summary,trace = run_episode(scenes[0],args.model,args.method,
                avoidance=avoidance,substep=args.substep,trace=True)
            summary["seed"] = args.seed[0]
            payload = dict(summary=summary,trace=trace)
            rows = [summary]
        else:
            rows = paired_experiment(args.seed,scale=args.scale,
                models=(args.model,) if args.model else ("A","B"),
                methods=(args.method,) if args.method else ("G0","G1","G2","G3"),
                avoidance=avoidance,substep=args.substep)
            payload = dict(rows=rows)
        for row in rows:
            print(f"[kinematic] seed={row['seed']} | scene={row['scene_id'][:24]} | "
                  f"model={row['model']} | method={row['method']} | "
                  f"assigned={row['intercepted_count']}/{row['intruder_count']} | "
                  f"breach={row['breached_count']} | friendly={row['friendly_collision_count']} | "
                  f"outcome={row['outcome']} | time={row['episode_resolution_time_s']:.3f} s")
    if args.out is not None:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with args.out.open("x",encoding="utf-8") as stream:
            json.dump(serializable(payload),stream,ensure_ascii=False,
                      separators=(",", ":"),allow_nan=False)


if __name__ == "__main__":
    main()
