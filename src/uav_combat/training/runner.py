"""Paper Algorithm 1 runner with 24 synchronous environments."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
import numpy as np

from ..madsac.trainer import MADSACTrainer
from .evaluator import episode_return_metrics, evaluate
from .vector_env import SyncVectorEnv


class PaperTrainingRunner:
    def __init__(
        self,
        env_config: dict,
        algorithm_config: dict,
        num_envs: int | None = None,
        total_sampled_steps: int | None = None,
        device: str | None = None,
        seed: int | None = None,
        output_dir: str | Path | None = None,
        smoke: bool = False,
    ) -> None:
        self.env_config, self.algorithm_config = env_config, algorithm_config
        training = algorithm_config["training"]
        assumptions = algorithm_config["reproduction_assumptions"]
        self.num_envs = int(num_envs or training["num_train_envs"])
        self.total_sampled_steps = int(total_sampled_steps or training["total_sampled_steps"])
        self.device = str(device or training["device"])
        self.seed = int(training["seed"] if seed is None else seed)
        self.steps_per_update = int(assumptions["steps_per_update"])
        self.update_steps_n = int(assumptions["update_steps_n"])
        self.policy_delay_d = int(assumptions["policy_delay_d"])
        self.algorithm1_t_counter = str(assumptions["algorithm1_t_counter"])
        if self.algorithm1_t_counter != "global_vector_step":
            raise ValueError("only the documented global_vector_step assumption is implemented")
        if min(self.steps_per_update, self.update_steps_n, self.policy_delay_d) <= 0:
            raise ValueError("Algorithm 1 scheduler values must be positive")

        batch_size = 64 if smoke else int(training["batch_size"])
        replay_capacity = 50_000 if smoke else int(training["replay_buffer_size"])
        hidden_dim = 64 if smoke else int(algorithm_config["network"]["actor_hidden_layers"][0])
        self.trainer = MADSACTrainer(
            hidden_dim=hidden_dim,
            attention_heads=int(algorithm_config["network"]["attention_heads"]),
            learning_rate=float(training["learning_rate"]),
            gamma=float(training["gamma"]),
            tau=float(training["tau"]),
            alpha=float(training["alpha"]),
            replay_capacity=replay_capacity,
            batch_size=batch_size,
            device=self.device,
            seed=self.seed,
            actor_activation=assumptions["actor_activation"],
            critic_activation=assumptions["critic_activation"],
            log_std_min=float(assumptions["log_std_min"]),
            log_std_max=float(assumptions["log_std_max"]),
        )
        evaluation_base = int(assumptions["evaluation_seed_base"])
        self.evaluation_seeds = list(range(evaluation_base, evaluation_base + int(training["evaluation_episodes"])))
        if self.seed + self.total_sampled_steps + self.num_envs >= evaluation_base:
            raise ValueError("configured training seed range can overlap evaluation seeds")
        self.vector = SyncVectorEnv(self.num_envs, env_config, self.seed, self.evaluation_seeds)
        self.observations = self.vector.reset()
        self.alive_masks = self.vector.current_alive_masks.copy()

        base_output = Path(output_dir or training["output_dir"])
        self.output_dir = base_output / f"run_seed_{self.seed}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scheduler_T = 0
        self.scheduler_update_blocks = 0
        self.agent_episode_returns = np.zeros((self.num_envs, 4), dtype=float)
        self.completed_records: list[dict[str, Any]] = []
        self.last_metrics: dict[str, float] = {}
        self.evaluation_history: list[dict[str, float]] = []
        cycle_steps = int(assumptions["assumed_sampled_steps_per_training_cycle"])
        self.evaluation_interval = int(training["evaluation_every_training_cycles"]) * cycle_steps
        self.next_evaluation = self.evaluation_interval
        self.checkpoint_interval = int(assumptions["checkpoint_interval_sampled_steps"])
        self.next_checkpoint = self.checkpoint_interval

    def startup_summary(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "num_envs_M": self.num_envs,
            "seed": self.seed,
            "total_sampled_steps": self.total_sampled_steps,
            "batch_size": self.trainer.batch_size,
            "replay_capacity": self.trainer.replay.capacity,
            "steps_per_update": self.steps_per_update,
            "update_steps_n": self.update_steps_n,
            "policy_delay_d": self.policy_delay_d,
            "algorithm1_t_counter": self.algorithm1_t_counter,
        }

    def _algorithm1_updates(self) -> tuple[int, int]:
        """Execute the update block printed in Algorithm 1."""
        if self.scheduler_T < self.steps_per_update or self.trainer.replay.size < self.trainer.batch_size:
            return 0, 0
        critic_metrics: list[dict[str, float]] = []
        actor_metrics: list[dict[str, float]] = []
        for _ in range(self.update_steps_n):
            critic_metrics.append(self.trainer.update_critics())
        actor_branch = self.trainer.vector_steps % self.policy_delay_d == 0
        if actor_branch:
            for _ in range(self.update_steps_n):
                actor_metrics.append(self.trainer.update_actor())
            self.trainer.update_targets()
        self.scheduler_T = 0
        self.scheduler_update_blocks += 1
        keys = set().union(*(row.keys() for row in critic_metrics + actor_metrics))
        self.last_metrics = {
            key: float(np.mean([row[key] for row in critic_metrics + actor_metrics if key in row]))
            for key in keys
        }
        return len(critic_metrics), len(actor_metrics)

    def vector_step(self) -> dict[str, Any]:
        actions = self.trainer.act(self.observations, self.alive_masks)
        result = self.vector.step_batch(actions)
        dones = result.terminated | result.truncated
        executed = np.stack([info["executed_red_actions"] for info in result.infos])
        self.trainer.replay.push_batch(
            self.observations, executed, result.rewards, result.transition_next_observations,
            dones, result.alive_masks, result.next_alive_masks,
        )
        self.agent_episode_returns += result.rewards
        completed_now = []
        for i, done in enumerate(dones):
            if done:
                per_agent = self.agent_episode_returns[i].copy()
                team_return, mean_agent_return = episode_return_metrics(per_agent)
                record = {
                    "episode_return": team_return,
                    "team_episode_return": team_return,
                    "mean_agent_episode_return": mean_agent_return,
                    "per_agent_episode_returns": per_agent,
                    **result.infos[i],
                }
                self.completed_records.append(record)
                completed_now.append(record)
                self.agent_episode_returns[i].fill(0.0)
        self.observations = result.observations
        self.alive_masks = self.vector.current_alive_masks.copy()
        self.trainer.sampled_steps += self.num_envs
        self.trainer.vector_steps += 1
        self.scheduler_T += self.num_envs
        critic_updates, actor_updates = self._algorithm1_updates()

        completed_mean = lambda key: (
            float(np.mean([row[key] for row in completed_now])) if completed_now else None
        )
        metric_record = {
            "sampled_steps": self.trainer.sampled_steps,
            "team_episode_return": completed_mean("team_episode_return"),
            "mean_agent_episode_return": completed_mean("mean_agent_episode_return"),
            "win_rate": completed_mean("red_success"),
            "red_uav_losses": completed_mean("red_losses"),
            "critic_loss": (
                (self.last_metrics.get("critic1_loss", 0.0) + self.last_metrics.get("critic2_loss", 0.0)) / 2.0
                if self.last_metrics else None
            ),
            "actor_loss": self.last_metrics.get("actor_loss"),
            "q_value": self.last_metrics.get("q_value"),
            "entropy": self.last_metrics.get("entropy"),
        }
        with (self.output_dir / "training_metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metric_record) + "\n")
        return {
            "sampled_steps": self.trainer.sampled_steps,
            "new_transitions": self.num_envs,
            "critic_updates": critic_updates,
            "actor_updates": actor_updates,
            "scheduler_T": self.scheduler_T,
            "completed_episodes": len(completed_now),
        }

    def run(self) -> dict[str, Any]:
        while self.trainer.sampled_steps < self.total_sampled_steps:
            self.vector_step()
            if self.trainer.sampled_steps >= self.next_evaluation:
                record = {
                    "sampled_steps": self.trainer.sampled_steps,
                    **evaluate(self.trainer, self.env_config, self.evaluation_seeds),
                }
                self.evaluation_history.append(record)
                self._write_evaluation()
                self.next_evaluation += self.evaluation_interval
            if self.trainer.sampled_steps >= self.next_checkpoint:
                self.save_checkpoint(self.output_dir / f"checkpoint_{self.trainer.sampled_steps}.pt")
                self.next_checkpoint += self.checkpoint_interval
        self.save_checkpoint(self.output_dir / "latest.pt")
        return self.summary()

    def _write_evaluation(self) -> None:
        if not self.evaluation_history:
            return
        with (self.output_dir / "evaluation_history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.evaluation_history[0]))
            writer.writeheader()
            writer.writerows(self.evaluation_history)

    def save_checkpoint(self, path: str | Path) -> None:
        self.trainer.save(path, {
            "scheduler_T": self.scheduler_T,
            "scheduler_update_blocks": self.scheduler_update_blocks,
            "episode_indices": self.vector.episode_indices.tolist(),
        })

    def resume(self, path: str | Path) -> None:
        """Resume networks/counters with an empty replay and fresh episodes."""
        extra = self.trainer.load(path)
        self.scheduler_T = int(extra.get("scheduler_T", 0))
        # The fallback reads checkpoints made before the counter was renamed;
        # both values count scheduler update blocks, never paper training cycles.
        self.scheduler_update_blocks = int(
            extra.get("scheduler_update_blocks", extra.get("training_cycles", 0))
        )
        previous = np.asarray(extra.get("episode_indices", [0] * self.num_envs), dtype=np.int64)
        if previous.shape != (self.num_envs,):
            raise RuntimeError("checkpoint environment count mismatch")
        self.vector.episode_indices = previous + 1
        self.observations = self.vector.reset()
        self.alive_masks = self.vector.current_alive_masks.copy()

    def summary(self) -> dict[str, Any]:
        mean = lambda key: (
            float(np.mean([record[key] for record in self.completed_records]))
            if self.completed_records else 0.0
        )
        return {
            **self.startup_summary(),
            "sampled_steps": self.trainer.sampled_steps,
            "vector_steps": self.trainer.vector_steps,
            "scheduler_update_blocks": self.scheduler_update_blocks,
            "critic_updates": self.trainer.critic_update_count,
            "actor_updates": self.trainer.actor_update_count,
            "target_updates": self.trainer.target_update_count,
            "replay_size": self.trainer.replay.size,
            "completed_episodes": len(self.completed_records),
            "average_return": mean("episode_return"),
            "average_agent_return": mean("mean_agent_episode_return"),
            "win_rate": mean("red_success"),
            "average_red_loss": mean("red_losses"),
            "last_update_metrics": self.last_metrics,
            "evaluation_history": self.evaluation_history,
        }
