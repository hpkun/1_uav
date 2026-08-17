"""Evaluate a checkpoint on twenty disjoint seeds."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from uav_combat.madsac import MADSACTrainer
from uav_combat.training.evaluator import evaluate


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--checkpoint", required=True); p.add_argument("--output",default="outputs/evaluation_smoke.json"); args = p.parse_args(); root = Path(__file__).resolve().parents[1]
    trainer = MADSACTrainer(); trainer.load(root / args.checkpoint); result=evaluate(trainer, root / "configs/paper_environment.yaml")
    output=root/args.output; output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
