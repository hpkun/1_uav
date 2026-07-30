"""Train or resume the project feed-forward MAPPO baseline."""
from __future__ import annotations
import argparse
from uav_env.algorithms.mappo.config import load_mappo_config
from uav_env.algorithms.mappo.runner import MAPPORunner

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--seed",type=int); p.add_argument("--device"); p.add_argument("--total-env-steps",type=int); p.add_argument("--num-envs",type=int); p.add_argument("--resume"); p.add_argument("--load-actor-only"); p.add_argument("--run-name",default="mappo"); p.add_argument("--run-id"); p.add_argument("--log-interval",type=int)
    a=p.parse_args(); c=load_mappo_config(a.config)
    for key,value in (("seed",a.seed),("device",a.device),("total_env_steps",a.total_env_steps),("num_envs",a.num_envs),("run_id",a.run_id),("log_interval",a.log_interval)):
        if value is not None: c[key]=value
    runner=MAPPORunner(c,a.run_name)
    if a.resume: runner.resume(a.resume)
    if a.load_actor_only: runner.resume(a.load_actor_only,actor_only=True)
    print(f"Output: {runner.run().resolve()}")
if __name__=="__main__": main()
