"""Evaluate a checkpoint on twenty disjoint seeds."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from uav_combat.madsac import MADSACTrainer
from uav_combat.training.evaluator import evaluate


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--checkpoint", required=True); args = p.parse_args(); root = Path(__file__).resolve().parents[1]
    trainer = MADSACTrainer(); trainer.load(root / args.checkpoint); print(json.dumps(evaluate(trainer, root / "configs/paper_environment.yaml"), indent=2))


if __name__ == "__main__": main()
