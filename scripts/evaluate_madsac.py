"""Evaluate a checkpoint on twenty disjoint seeds."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from uav_combat.madsac import MADSACTrainer
from uav_combat.training.evaluator import evaluate
from uav_combat.environment.observation import OBSERVATION_DIM


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--checkpoint", required=True); p.add_argument("--output",default="outputs/evaluation_smoke.json"); p.add_argument("--env-config", default="configs/combat_environment.yaml"); args = p.parse_args(); root = Path(__file__).resolve().parents[1]
    checkpoint=root/args.checkpoint
    state=torch.load(checkpoint,map_location="cpu",weights_only=False)
    hidden=int(state["actor"]["backbone.0.weight"].shape[0])
    env_path=Path(args.env_config); env_path=env_path if env_path.is_absolute() else root/env_path
    trainer = MADSACTrainer(observation_dim=OBSERVATION_DIM, hidden_dim=hidden); trainer.load(checkpoint); result=evaluate(trainer, env_path)
    output=root/args.output; output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
