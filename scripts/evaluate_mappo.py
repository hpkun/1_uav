"""Evaluate a MAPPO checkpoint with deterministic or sampled actions."""
from __future__ import annotations
import argparse,yaml,torch
from uav_env.algorithms.mappo.networks import SharedActor,CentralizedCritic
from uav_env.algorithms.mappo.checkpoint import load_checkpoint
from uav_env.algorithms.mappo.runner import MAPPORunner
from uav_env.algorithms.mappo.value_normalizer import ValueNormalizer

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--episodes",type=int,default=100); p.add_argument("--seed-start",type=int,default=100000); p.add_argument("--deterministic",action=argparse.BooleanOptionalAction,default=True); p.add_argument("--scenario"); p.add_argument("--opponent"); a=p.parse_args()
    data=torch.load(a.checkpoint,map_location="cpu",weights_only=False); c=data["config"]
    if a.scenario: c["environment"]["scenario"]=a.scenario
    if a.opponent: c["environment"]["opponent"]=a.opponent
    runner=MAPPORunner(c,"evaluation"); runner.resume(a.checkpoint,actor_only=True); print(yaml.safe_dump(runner.evaluate(a.episodes,a.seed_start,a.deterministic),sort_keys=False))
if __name__=="__main__": main()
