"""Holdout evaluation for a formal MAPPO checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from uav_combat.mappo.trainer import MAPPOTrainer
from uav_combat.training.checkpoint import validate_checkpoint_environment
from uav_combat.training.evaluator import evaluate


ROOT = Path(__file__).resolve().parents[1]


def resolved(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def build_trainer(config: dict, device: str):
    network = config["network"]
    training = config["training"]
    implementation = config["implementation"]
    common = {
        "observation_dim": int(network["observation_dim"]),
        "action_dim": int(network["action_dim"]),
        "num_agents": int(network["num_agents"]),
        "hidden_dim": int(network["actor_hidden_layers"][0]),
        "attention_heads": int(network["attention_heads"]),
        "device": device,
        "actor_activation": implementation["actor_activation"],
        "critic_activation": implementation["critic_activation"],
        "log_std_min": float(implementation["log_std_min"]),
        "log_std_max": float(implementation["log_std_max"]),
    }
    return MAPPOTrainer(
        **common,
        actor_learning_rate=float(training["actor_learning_rate"]),
        critic_learning_rate=float(training["critic_learning_rate"]),
        gamma=float(training["gamma"]),
        gae_lambda=float(training["gae_lambda"]),
        clip_ratio=float(training["clip_ratio"]),
        value_loss_coefficient=float(training["value_loss_coefficient"]),
        entropy_coefficient=float(training["entropy_coefficient"]),
        max_grad_norm=float(training["max_grad_norm"]),
        ppo_epochs=int(training["ppo_epochs"]),
        minibatch_size=int(training["minibatch_size"]),
        normalize_advantages=bool(implementation["normalize_advantages"]),
        clip_value_loss=bool(implementation["clip_value_loss"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=("mappo",), default="mappo")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env-config", required=True)
    parser.add_argument("--algorithm-config", required=True)
    parser.add_argument("--seed-base", type=int, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")

    env_config = yaml.safe_load(resolved(args.env_config).read_text(encoding="utf-8"))
    algorithm_config = yaml.safe_load(
        resolved(args.algorithm_config).read_text(encoding="utf-8")
    )
    checkpoint = resolved(args.checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint_environment(state, env_config)
    trainer = build_trainer(algorithm_config, args.device)
    trainer.load(checkpoint)
    seeds = range(args.seed_base, args.seed_base + args.episodes)
    result = evaluate(trainer, env_config, seeds)
    result.update({
        "algorithm": "MAPPO",
        "checkpoint": str(checkpoint),
        "holdout_seed_base": args.seed_base,
    })
    output = resolved(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
