"""Formal runner with an intentionally short smoke mode."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
import yaml
from uav_combat.environment import PaperUAVCombatEnv
from uav_combat.madsac import MADSACTrainer


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--smoke", action="store_true"); parser.add_argument("--steps", type=int); parser.add_argument("--output", default="outputs/madsac_smoke.pt")
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "configs/madsac.yaml").read_text(encoding="utf-8")); t = cfg["training"]; assumptions = cfg["reproduction_assumptions"]
    steps = args.steps or (2048 if args.smoke else t["total_env_steps"]); batch = 16 if args.smoke else t["batch_size"]; capacity = 4096 if args.smoke else t["replay_buffer_size"]
    trainer = MADSACTrainer(learning_rate=t["learning_rate"], gamma=t["gamma"], tau=t["tau"], alpha=t["alpha"], policy_delay=assumptions["policy_delay"], replay_capacity=capacity, batch_size=batch)
    initial = {"actor": next(trainer.actor.parameters()).detach().clone(), "critic1": next(trainer.critic1.parameters()).detach().clone(), "critic2": next(trainer.critic2.parameters()).detach().clone(), "target_actor": next(trainer.target_actor.parameters()).detach().clone()}
    env = PaperUAVCombatEnv(root / "configs/paper_environment.yaml"); obs, _ = env.reset(0); metrics = {}; episode_return = 0.0; completed = []
    for step in range(steps):
        action = trainer.act(obs); next_obs, reward, terminated, truncated, info = env.step(action)
        trainer.replay.push(obs, action, reward, next_obs, terminated or truncated); episode_return += float(reward[0]); obs = next_obs
        if trainer.replay.size >= batch: metrics = trainer.update(batch)
        if terminated or truncated:
            completed.append({"episode_return": episode_return, **info}); obs, _ = env.reset(step + 1); episode_return = 0.0
    trainer.save(root / args.output)
    summary = {"sampled_steps": steps, "episodes": len(completed), "last_metrics": metrics, "critic_updates": trainer.update_count, "actor_updates": trainer.actor_update_count, "target_updates": trainer.target_update_count, "actor_param_changed": not torch.equal(initial["actor"], next(trainer.actor.parameters())), "critic1_param_changed": not torch.equal(initial["critic1"], next(trainer.critic1.parameters())), "critic2_param_changed": not torch.equal(initial["critic2"], next(trainer.critic2.parameters())), "target_param_changed": not torch.equal(initial["target_actor"], next(trainer.target_actor.parameters())), "replay_size": trainer.replay.size, "all_finite": bool(all(np.isfinite(v) for v in metrics.values() if isinstance(v, float)))}
    (root / "outputs").mkdir(exist_ok=True); (root / "outputs/madsac_smoke_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8"); print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
