"""MADSAC Algorithm-1 runner with synchronous environments."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
import numpy as np

from ..madsac.trainer import MADSACTrainer
from .evaluator import episode_return_metrics, evaluate
from .vector_env import SyncVectorEnv


class MADSACTrainingRunner:
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
        assumptions = algorithm_config["implementation"]
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
            observation_dim=int(algorithm_config["network"]["observation_dim"]),
            action_dim=int(algorithm_config["network"]["action_dim"]),
            num_agents=int(algorithm_config["network"]["num_agents"]),
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
        self.last_critic_metrics: dict[str, float] = {}
        self.last_actor_metrics: dict[str, float] = {}
        self.last_metrics: dict[str, float] = {}
        self.evaluation_history: list[dict[str, float]] = []
        runtime_logging = algorithm_config["runtime_logging"]
        self.console_interval = int(runtime_logging["console_interval_sampled_steps"])
        self.recent_episode_window = int(runtime_logging["recent_episode_window"])
        if min(self.console_interval, self.recent_episode_window) <= 0:
            raise ValueError("runtime logging interval and window must be positive")
        self.next_console_log = self.console_interval
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

    def start_log_line(self) -> str:
        summary = self.startup_summary()
        return (
            f"[START] device={summary['device']} | envs={summary['num_envs_M']} | seed={summary['seed']} "
            f"| total={summary['total_sampled_steps']} | batch={summary['batch_size']} "
            f"| replay={summary['replay_capacity']} | T={summary['steps_per_update']} "
            f"| n={summary['update_steps_n']} | d={summary['policy_delay_d']}"
        )

    def recent_episode_metrics(self) -> dict[str, float | None]:
        records = self.completed_records[-self.recent_episode_window:]
        if not records:
            return {"return": None, "win": None, "red_loss": None}
        return {
            "return": float(np.mean([row["team_episode_return"] for row in records])),
            "win": float(np.mean([row["red_success"] for row in records])),
            "red_loss": float(np.mean([row["red_losses"] for row in records])),
        }

    @staticmethod
    def _display(value: float | None, digits: int) -> str:
        return "NA" if value is None else f"{value:.{digits}f}"

    def _last_critic_loss(self) -> float | None:
        critic1 = self.last_critic_metrics.get("critic1_loss")
        critic2 = self.last_critic_metrics.get("critic2_loss")
        if critic1 is None or critic2 is None:
            return None
        return (critic1 + critic2) / 2.0

    def train_log_line(self) -> str:
        recent = self.recent_episode_metrics()
        critic = self._last_critic_loss()
        percent = 100.0 * self.trainer.sampled_steps / self.total_sampled_steps
        return (
            f"[TRAIN] steps={self.trainer.sampled_steps}/{self.total_sampled_steps} ({percent:.1f}%) "
            f"| eps={len(self.completed_records)} | return={self._display(recent['return'], 2)} "
            f"| win={self._display(recent['win'], 2)} | red_loss={self._display(recent['red_loss'], 2)} "
            f"| critic={self._display(critic, 4)} "
            f"| actor={self._display(self.last_actor_metrics.get('actor_loss'), 3)} "
            f"| Q={self._display(self.last_critic_metrics.get('q_value'), 3)} "
            f"| H={self._display(self.last_actor_metrics.get('entropy'), 2)}"
        )

    @staticmethod
    def evaluation_log_line(record: dict[str, float]) -> str:
        return (
            f"[EVAL] steps={int(record['sampled_steps'])} | return={record['average_return']:.2f} "
            f"| agent_return={record['average_agent_return']:.2f} | win={record['win_rate']:.2f} "
            f"| red_loss={record['average_red_loss']:.2f} "
            f"| ep_len={record['average_episode_length']:.1f}"
        )

    @staticmethod
    def checkpoint_log_line(sampled_steps: int, path: str | Path) -> str:
        return f"[CKPT] steps={sampled_steps} | saved={Path(path).name}"

    @staticmethod
    def done_log_line(summary: dict[str, Any]) -> str:
        return (
            f"[DONE] steps={summary['sampled_steps']} | episodes={summary['completed_episodes']} "
            f"| return={summary['average_return']:.2f} | win={summary['win_rate']:.2f} "
            f"| red_loss={summary['average_red_loss']:.2f}"
        )

    def _console_log_due(self) -> bool:
        if self.trainer.sampled_steps < self.next_console_log:
            return False
        self.next_console_log += self.console_interval
        return True

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
        critic_keys = set().union(*(row.keys() for row in critic_metrics))
        self.last_critic_metrics = {
            key: float(np.mean([row[key] for row in critic_metrics if key in row]))
            for key in critic_keys
        }
        if actor_metrics:
            actor_keys = set().union(*(row.keys() for row in actor_metrics))
            self.last_actor_metrics = {
                key: float(np.mean([row[key] for row in actor_metrics if key in row]))
                for key in actor_keys
            }
        self.last_metrics = {**self.last_critic_metrics, **self.last_actor_metrics}
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
            "loss_rate": completed_mean("blue_win"),
            "draw_rate": completed_mean("draw"),
            "red_uav_losses": completed_mean("red_losses"),
            "blue_uav_losses": completed_mean("blue_losses"),
            "critic_loss": self._last_critic_loss(),
            "actor_loss": self.last_actor_metrics.get("actor_loss"),
            "q_value": self.last_critic_metrics.get("q_value"),
            "entropy": self.last_actor_metrics.get("entropy"),
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
            if self._console_log_due():
                print(self.train_log_line(), flush=True)
            if self.trainer.sampled_steps >= self.next_evaluation:
                record = {
                    "sampled_steps": self.trainer.sampled_steps,
                    **evaluate(self.trainer, self.env_config, self.evaluation_seeds),
                }
                self.evaluation_history.append(record)
                self._write_evaluation()
                print(self.evaluation_log_line(record), flush=True)
                self.next_evaluation += self.evaluation_interval
            if self.trainer.sampled_steps >= self.next_checkpoint:
                checkpoint_path = self.output_dir / f"checkpoint_{self.trainer.sampled_steps}.pt"
                self.save_checkpoint(checkpoint_path)
                print(
                    self.checkpoint_log_line(self.trainer.sampled_steps, checkpoint_path),
                    flush=True,
                )
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
        self.next_console_log = (
            self.trainer.sampled_steps // self.console_interval + 1
        ) * self.console_interval

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
            "loss_rate": mean("blue_win"),
            "draw_rate": mean("draw"),
            "average_red_loss": mean("red_losses"),
            "average_blue_loss": mean("blue_losses"),
            "total_red_attack_kills": int(sum(
                record["red_attack_kills"] for record in self.completed_records
            )),
            "total_blue_attack_kills": int(sum(
                record["blue_attack_kills"] for record in self.completed_records
            )),
            "average_red_boundary_loss": mean("red_boundary_losses"),
            "average_blue_boundary_loss": mean("blue_boundary_losses"),
            "average_red_horizontal_boundary_loss": mean("red_horizontal_boundary_losses"),
            "average_blue_horizontal_boundary_loss": mean("blue_horizontal_boundary_losses"),
            "average_red_low_altitude_loss": mean("red_low_altitude_losses"),
            "average_blue_low_altitude_loss": mean("blue_low_altitude_losses"),
            "average_red_high_altitude_loss": mean("red_high_altitude_losses"),
            "average_blue_high_altitude_loss": mean("blue_high_altitude_losses"),
            "first_attackable_episode_rate": float(np.mean([
                record["first_attackable_step"] is not None for record in self.completed_records
            ])) if self.completed_records else 0.0,
            "first_kill_episode_rate": float(np.mean([
                record["first_kill_step"] is not None for record in self.completed_records
            ])) if self.completed_records else 0.0,
            "last_update_metrics": self.last_metrics,
            "evaluation_history": self.evaluation_history,
        }
