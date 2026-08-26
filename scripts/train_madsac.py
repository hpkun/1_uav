"""Configuration-driven formal and smoke MADSAC runner."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import yaml
from uav_combat.training.runner import MADSACTrainingRunner


def main()->None:
    p=argparse.ArgumentParser(); p.add_argument("--device",choices=["cpu","cuda"]); p.add_argument("--seed",type=int); p.add_argument("--total-sampled-steps",type=int); p.add_argument("--num-envs",type=int); p.add_argument("--output-dir"); p.add_argument("--resume"); p.add_argument("--env-config",default="configs/combat_environment.yaml"); p.add_argument("--smoke",action="store_true")
    args=p.parse_args(); root=Path(__file__).resolve().parents[1]
    env_path=Path(args.env_config); env_path=env_path if env_path.is_absolute() else root/env_path
    env_cfg=yaml.safe_load(env_path.read_text(encoding="utf-8")); alg_cfg=yaml.safe_load((root/"configs/madsac.yaml").read_text(encoding="utf-8"))
    total=args.total_sampled_steps or (24000 if args.smoke else None); runner=MADSACTrainingRunner(env_cfg,alg_cfg,args.num_envs,total,args.device,args.seed,args.output_dir,args.smoke)
    print(runner.start_log_line(), flush=True)
    if args.resume: runner.resume(args.resume)
    summary=runner.run(); (runner.output_dir/"run_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8"); print(runner.done_log_line(summary), flush=True)


if __name__=="__main__": main()
