"""Formal-grade modular MAPPO runner with bounded memory and strict lineage."""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from env.config import ENVIRONMENT_VERSION
from algorithm.common.checkpoint import evaluation_selection_key
from algorithm.common.protocol import config_sha256
from algorithm.common.vector_env import ParallelVectorEnv
from algorithm.mappo.networks import SharedMAPPOActor
from algorithm.mappo.trainer import MAPPO_IMPL_VERSION
from .buffer import ModularRolloutBatch
from .evaluation import evaluate_modular
from .factory import build_modular_mappo_trainer
from .protocol import checkpoint_architecture, validate_modular_checkpoint
from .trainer import MODULAR_MAPPO_IMPL_VERSION


class ModularMAPPOTrainingRunner:
    def __init__(self, env_config: dict, algorithm_config: dict,
                 num_envs: int | None = None, total_sampled_steps: int | None = None,
                 device: str | None = None, seed: int | None = None,
                 output_dir: str | Path | None = None, smoke: bool = False,
                 warm_start_checkpoint: str | None = None,
                 reference_checkpoint: str | None = None,
                 resume_mode: bool = False) -> None:
        self.env_config = deepcopy(env_config)
        self.algorithm_config = deepcopy(algorithm_config)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for name in ("training_metrics.jsonl", "optimization_metrics.jsonl"):
            (self.output_dir / name).touch(exist_ok=True)
        self.smoke = bool(smoke)
        training = algorithm_config["training"]
        implementation = algorithm_config["implementation"]
        logging = algorithm_config["runtime_logging"]
        self.num_envs = int(num_envs or training["num_train_envs"])
        self.total_sampled_steps = int(total_sampled_steps or training["total_sampled_steps"])
        self.seed = int(training["seed"] if seed is None else seed)
        self.device = str(device or training["device"])
        configured = deepcopy(algorithm_config)
        configured["training"]["seed"] = self.seed
        warm_enabled = bool(configured.get("modules", {}).get("warm_start", {}).get("enabled", False))
        anchor_enabled = bool(configured.get("modules", {}).get("policy_anchor", {}).get("enabled", False))
        self.effective_hidden_dim = int(configured["network"]["actor_hidden_layers"][0])
        if self.smoke and not (warm_enabled or anchor_enabled):
            self.effective_hidden_dim = 64
        self.trainer = build_modular_mappo_trainer(configured, self.device, self.effective_hidden_dim)
        self.rollout_steps = 4 if self.smoke else int(training["rollout_steps"])
        self.eval_episodes = min(2, int(training["evaluation_episodes"])) if self.smoke else int(training["evaluation_episodes"])
        self.eval_base = int(implementation["evaluation_seed_base"])
        self.evaluation_interval = int(training["evaluation_interval_sampled_steps"])
        self.checkpoint_interval = int(implementation["checkpoint_interval_sampled_steps"])
        self.console_interval = int(logging["console_interval_sampled_steps"])
        self.recent_window = int(logging["recent_episode_window"])
        self.next_evaluation = self.evaluation_interval
        self.next_checkpoint = self.checkpoint_interval
        self.next_console = self.console_interval
        self.current_stage, self.current_waves = self.trainer.curriculum.stage(0)
        self.runtime_env_config = self.trainer.curriculum.runtime_config(self.env_config, 0)
        self.vector: ParallelVectorEnv | None = None
        self._make_vector()
        self.recent_episodes: deque[dict[str, Any]] = deque(maxlen=self.recent_window)
        self.completed_episode_count = 0
        self.raw_episode_returns = np.zeros((self.num_envs, 4), dtype=np.float64)
        self.training_episode_returns = np.zeros((self.num_envs, 4), dtype=np.float64)
        self.paper_episode_blue = np.zeros(self.num_envs, dtype=np.float64)
        self.paper_episode_red = np.zeros(self.num_envs, dtype=np.float64)
        self.paper_episode_by_wave = np.zeros((self.num_envs, 3), dtype=np.float64)
        self.transition_counts = np.zeros(3, dtype=np.int64)
        self.alive_agent_counts = np.zeros(3, dtype=np.int64)
        self.wave_clear_transition_counts = np.zeros(3, dtype=np.int64)
        self.reward_bonus_totals = np.zeros(4, dtype=np.float64)
        self.paper_reward_totals = np.zeros(5, dtype=np.float64)
        self.death_index_totals = np.zeros((2, self.alive.shape[1]), dtype=np.int64)
        self.death_cause_names = (
            "red_weapon_deaths", "red_boundary_deaths", "red_ground_deaths",
            "blue_weapon_deaths", "blue_boundary_deaths", "blue_ground_deaths",
        )
        self.death_cause_info_keys = (
            "blue_attack_kills", "red_boundary_exits", "red_ground_losses",
            "red_attack_kills", "blue_boundary_exits", "blue_ground_losses",
        )
        self.previous_cause_counts = np.zeros((self.num_envs, len(self.death_cause_names)), dtype=np.int64)
        self.death_cause_totals = np.zeros(len(self.death_cause_names), dtype=np.int64)
        self.hidden_reset_count = 0
        self.curriculum_transitions = [{"sampled_steps": 0, "stage": self.current_stage, "total_waves": self.current_waves}]
        self.evaluation_history: list[dict[str, Any]] = []
        self.best_evaluation: dict[str, Any] | None = None
        self.best_sampled_steps: int | None = None
        self.latest_evaluation: dict[str, Any] | None = None
        self.last_metrics: dict[str, float] = {}
        self.last_rollout_metrics: dict[str, float] = {}
        self.resume_count = 0
        if not resume_mode:
            if warm_start_checkpoint:
                self.trainer.warm_start_provenance = self.trainer.warm_start.initialize(self.trainer, warm_start_checkpoint)
            if self.trainer.anchor.enabled:
                if not reference_checkpoint:
                    raise ValueError("fresh policy-anchor training requires --reference-checkpoint")
                self._attach_reference(reference_checkpoint)

    def _attach_reference(self, checkpoint: str) -> None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        network = self.algorithm_config["network"]
        reference = SharedMAPPOActor(
            int(network["observation_dim"]), int(network["action_dim"]),
            int(network["actor_hidden_layers"][0]),
            float(self.algorithm_config["implementation"]["log_std_min"]),
            float(self.algorithm_config["implementation"]["log_std_max"]),
            str(self.algorithm_config["implementation"]["actor_activation"]),
        ).to(self.trainer.device)
        reference.load_state_dict(state["actor"])
        self.trainer.anchor.attach(reference, str(checkpoint))
        extra = state.get("extra", {})
        with open(checkpoint, "rb") as stream:
            digest = hashlib.sha256(stream.read()).hexdigest()
        self.trainer.anchor_provenance = {
            "reference_checkpoint": str(checkpoint),
            "source_algorithm": state.get("algorithm"),
            "source_sampled_steps": int(state.get("sampled_steps", 0)),
            "source_environment_variant": extra.get("environment_variant"),
            "source_training_seed": extra.get("training_seed"),
            "source_checkpoint_sha256": digest,
        }

    def _make_vector(self, episode_indices: np.ndarray | None = None) -> None:
        if self.vector is not None:
            self.vector.close()
        self.vector = ParallelVectorEnv(
            self.num_envs, self.runtime_env_config, self.seed,
            range(self.eval_base, self.eval_base + self.eval_episodes),
        )
        if episode_indices is not None:
            self.vector.episode_indices = np.asarray(episode_indices, dtype=np.int64)
        self.observations = self.vector.reset()
        self.alive = self.vector.current_alive_masks.copy()
        self.blue_alive = np.ones_like(self.alive, dtype=np.float32)
        self.wave = np.ones(self.num_envs, dtype=np.int64)
        self.total = np.full(self.num_envs, self.current_waves, dtype=np.int64)
        self.episode_mask = np.zeros(self.num_envs, dtype=np.float32)
        self.actor_hidden, self.critic_hidden = self.trainer.initial_hidden(self.num_envs)
        (self.output_dir / "runtime_env_config.yaml").write_text(
            yaml.safe_dump(self.runtime_env_config, sort_keys=False), encoding="utf-8"
        )

    def _maybe_curriculum(self) -> None:
        stage, waves = self.trainer.curriculum.stage(self.trainer.sampled_steps)
        if (stage, waves) == (self.current_stage, self.current_waves):
            return
        previous = self.vector.episode_indices.copy() + 1
        self.current_stage, self.current_waves = stage, waves
        self.curriculum_transitions.append({"sampled_steps": self.trainer.sampled_steps, "stage": stage, "total_waves": waves})
        self.runtime_env_config = self.trainer.curriculum.runtime_config(self.env_config, self.trainer.sampled_steps)
        self._make_vector(previous)

    @staticmethod
    def _fractions(counts: np.ndarray) -> np.ndarray:
        return counts.astype(np.float64) / max(float(counts.sum()), 1.0)

    def _write_episode(self, info: dict[str, Any], raw_return: np.ndarray,
                       training_return: np.ndarray, sampled_steps: int,
                       paper_blue: float = 0.0, paper_red: float = 0.0,
                       paper_by_wave: np.ndarray | None = None) -> None:
        waves = int(info.get("waves_cleared", 0))
        paper_wave = np.zeros(3, dtype=np.float64) if paper_by_wave is None else np.asarray(paper_by_wave)
        record = {
            "sampled_steps": int(sampled_steps),
            "episode_length": int(info["episode_length"]),
            "team_raw_environment_return": float(raw_return.sum()),
            "team_training_return": float(training_return.sum()),
            "raw_environment_reward": float(raw_return.sum()),
            "jiao_training_reward": float(training_return.sum()),
            "paper_R2_blue_kill_component": float(paper_blue),
            "paper_R2_red_loss_component": float(paper_red),
            **{f"paper_R2_wave{k}": float(paper_wave[k - 1]) for k in (1, 2, 3)},
            "red_success": float(info["red_success"]),
            "blue_win": float(info["blue_win"]),
            "red_losses": int(info["red_losses"]),
            "blue_losses": int(info["blue_losses"]),
            "red_attack_kills": int(info["red_attack_kills"]),
            "blue_attack_kills": int(info["blue_attack_kills"]),
            "red_boundary_exits": int(info["red_boundary_exits"]),
            "blue_boundary_exits": int(info["blue_boundary_exits"]),
            "red_ground_losses": int(info["red_ground_losses"]),
            "blue_ground_losses": int(info["blue_ground_losses"]),
            "red_weapon_deaths": int(info["blue_attack_kills"]),
            "red_boundary_deaths": int(info["red_boundary_exits"]),
            "red_ground_deaths": int(info["red_ground_losses"]),
            "blue_weapon_deaths": int(info["red_attack_kills"]),
            "blue_boundary_deaths": int(info["blue_boundary_exits"]),
            "blue_ground_deaths": int(info["blue_ground_losses"]),
            "waves_cleared": waves,
            "total_waves": int(info.get("total_waves", 1)),
            **{f"wave_{k}_cleared": float(waves >= k) for k in (1, 2, 3)},
        }
        with (self.output_dir / "training_metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        self.recent_episodes.append(record)
        self.completed_episode_count += 1

    def collect_rollout(self, steps: int | None = None) -> ModularRolloutBatch:
        keys = ("observations", "actions", "raw_actions", "old_log_probs", "rewards",
                "raw_environment_rewards", "dones", "alive_masks", "next_observations",
                "next_alive_masks", "wave_indices", "total_waves", "contexts",
                "next_contexts", "actor_hidden_before_step", "critic_hidden_before_step",
                "episode_masks")
        storage = {key: [] for key in keys}
        rollout_transition = np.zeros(3, dtype=np.int64)
        rollout_alive = np.zeros(3, dtype=np.int64)
        reward_rows: list[dict[str, float]] = []
        actor_hidden_norms: list[np.ndarray] = []
        critic_hidden_norms: list[np.ndarray] = []
        rollout_hidden_resets = 0
        rollout_causes = np.zeros(len(self.death_cause_names), dtype=np.int64)
        rollout_death_indices = np.zeros_like(self.death_index_totals)
        for _ in range(int(steps or self.rollout_steps)):
            obs, alive, pre_wave = self.observations.copy(), self.alive.copy(), self.wave.copy()
            blue_alive = self.blue_alive.copy()
            context = self.trainer.context_numpy(pre_wave, self.total)
            actor_before = None if self.actor_hidden is None else self.actor_hidden.copy()
            critic_before = None if self.critic_hidden is None else self.critic_hidden.copy()
            actions, raw, log_prob, new_actor = self.trainer.act(
                obs, alive, False, True, context, self.actor_hidden, self.episode_mask
            )
            _, new_critic = self.trainer.values_step(
                obs, alive, context, self.critic_hidden, self.episode_mask
            )
            result = self.vector.step_batch(actions)
            done = result.terminated | result.truncated
            training_reward, reward_metrics = self.trainer.reward_adapter.adapt(
                result.rewards, result.infos, pre_wave, alive, blue_alive
            )
            paper = self.trainer.reward_adapter.last_transition
            blue_deaths = np.asarray(paper["blue_death_mask"], dtype=bool)
            red_deaths = np.asarray(paper["red_death_mask"], dtype=bool)
            step_indices = np.stack((blue_deaths.sum(0), red_deaths.sum(0)))
            rollout_death_indices += step_indices
            self.death_index_totals += step_indices
            current_causes = np.asarray([
                [int(info.get(key, 0)) for key in self.death_cause_info_keys]
                for info in result.infos
            ], dtype=np.int64)
            cause_delta = current_causes - self.previous_cause_counts
            # Counters are cumulative within an episode; auto-reset happens
            # after terminal info is returned, so the next baseline is zero.
            cause_delta = np.maximum(cause_delta, 0)
            step_causes = cause_delta.sum(0)
            rollout_causes += step_causes
            self.death_cause_totals += step_causes
            self.previous_cause_counts = np.where(done[:, None], 0, current_causes)
            self.paper_episode_blue += paper["blue_component"]
            self.paper_episode_red += paper["red_component"]
            self.paper_episode_by_wave += paper["per_wave"]
            self.paper_reward_totals += np.asarray([
                paper["blue_component"].sum(), paper["red_component"].sum(),
                paper["per_wave"][:, 0].sum(), paper["per_wave"][:, 1].sum(),
                paper["per_wave"][:, 2].sum(),
            ])
            reward_rows.append(reward_metrics)
            self.reward_bonus_totals += np.asarray([reward_metrics.get("reward_bonus_total",0.0),reward_metrics.get("reward_bonus_wave1",0.0),reward_metrics.get("reward_bonus_wave2",0.0),reward_metrics.get("reward_bonus_wave3",0.0)])
            next_wave = np.asarray([int(row.get("wave_index", 1)) for row in result.infos])
            next_total = np.asarray([int(row.get("total_waves", self.current_waves)) for row in result.infos])
            next_context = self.trainer.context_numpy(next_wave, next_total)
            for k in (1, 2, 3):
                transition = pre_wave == k
                rollout_transition[k - 1] += int(transition.sum())
                rollout_alive[k - 1] += int(alive[transition].sum())
                self.transition_counts[k - 1] += int(transition.sum())
                self.alive_agent_counts[k - 1] += int(alive[transition].sum())
            for env_id, info in enumerate(result.infos):
                if info.get("wave_cleared_this_step", False):
                    self.wave_clear_transition_counts[max(1, min(3, int(pre_wave[env_id]))) - 1] += 1
            values = (obs, actions, raw, log_prob, training_reward, result.rewards.copy(),
                      done.astype(np.float32), alive, result.transition_next_observations,
                      result.next_alive_masks, pre_wave, self.total.copy(), context,
                      next_context, actor_before, critic_before, self.episode_mask.copy())
            for key, value in zip(keys, values):
                storage[key].append(value)
            self.raw_episode_returns += result.rewards
            self.training_episode_returns += training_reward
            step_after = self.trainer.sampled_steps + self.num_envs
            for env_id, is_done in enumerate(done):
                if is_done:
                    self._write_episode(result.infos[env_id], self.raw_episode_returns[env_id],
                                        self.training_episode_returns[env_id], step_after,
                                        self.paper_episode_blue[env_id], self.paper_episode_red[env_id],
                                        self.paper_episode_by_wave[env_id])
                    self.raw_episode_returns[env_id].fill(0)
                    self.training_episode_returns[env_id].fill(0)
                    self.paper_episode_blue[env_id] = 0.0
                    self.paper_episode_red[env_id] = 0.0
                    self.paper_episode_by_wave[env_id].fill(0.0)
            self.observations = result.observations
            self.alive = self.vector.current_alive_masks.copy()
            post_blue = np.stack([np.asarray(row["blue_alive_mask"], dtype=np.float32) for row in result.infos])
            self.blue_alive = np.where(done[:, None], np.ones_like(post_blue), post_blue)
            self.wave = np.where(done, 1, next_wave)
            self.total = np.where(done, self.current_waves, next_total)
            self.episode_mask = (~done).astype(np.float32)
            self.actor_hidden = self.trainer.recurrent.apply_alive(new_actor, self.alive)
            self.critic_hidden = self.trainer.recurrent.apply_alive(new_critic, self.alive)
            self.trainer.recurrent.reset_for_episode(self.actor_hidden, done)
            self.trainer.recurrent.reset_for_episode(self.critic_hidden, done)
            resets = int(done.sum())
            rollout_hidden_resets += resets
            self.hidden_reset_count += resets
            if self.actor_hidden is not None:
                actor_hidden_norms.append(np.linalg.norm(self.actor_hidden, axis=-1))
            if self.critic_hidden is not None:
                critic_hidden_norms.append(np.linalg.norm(self.critic_hidden, axis=-1))
            self.trainer.sampled_steps += self.num_envs
            self.trainer.vector_steps += 1
        transition_fraction = self._fractions(rollout_transition)
        alive_fraction = self._fractions(rollout_alive)
        self.last_rollout_metrics = {
            **{f"transition_samples_wave_{k}": float(rollout_transition[k-1]) for k in (1,2,3)},
            **{f"transition_fraction_wave_{k}": float(transition_fraction[k-1]) for k in (1,2,3)},
            **{f"alive_agent_samples_wave_{k}": float(rollout_alive[k-1]) for k in (1,2,3)},
            **{f"alive_agent_fraction_wave_{k}": float(alive_fraction[k-1]) for k in (1,2,3)},
            "actor_hidden_norm": float(np.concatenate([x.ravel() for x in actor_hidden_norms]).mean()) if actor_hidden_norms else 0.0,
            "critic_hidden_norm": float(np.concatenate([x.ravel() for x in critic_hidden_norms]).mean()) if critic_hidden_norms else 0.0,
            "actor_hidden_norm_max": float(np.concatenate([x.ravel() for x in actor_hidden_norms]).max()) if actor_hidden_norms else 0.0,
            "critic_hidden_norm_max": float(np.concatenate([x.ravel() for x in critic_hidden_norms]).max()) if critic_hidden_norms else 0.0,
            "hidden_norm_mean": float(np.mean([
                value for rows in (actor_hidden_norms, critic_hidden_norms)
                for array in rows for value in array.ravel()
            ])) if (actor_hidden_norms or critic_hidden_norms) else 0.0,
            "hidden_reset_count": float(rollout_hidden_resets),
            "hidden_reset_count_total": float(self.hidden_reset_count),
            **{f"blue_deaths_index_{index}": float(rollout_death_indices[0, index]) for index in range(self.death_index_totals.shape[1])},
            **{f"red_deaths_index_{index}": float(rollout_death_indices[1, index]) for index in range(self.death_index_totals.shape[1])},
            **{name: float(rollout_causes[index]) for index, name in enumerate(self.death_cause_names)},
            "red_death_cause_unattributed": float(rollout_death_indices[1].sum() - rollout_causes[:3].sum()),
            "blue_death_cause_unattributed": float(rollout_death_indices[0].sum() - rollout_causes[3:].sum()),
        }
        if reward_rows:
            for key in reward_rows[0]:
                self.last_rollout_metrics[key] = float(np.mean([row[key] for row in reward_rows]))
        kwargs = {key: (None if not values or values[0] is None else np.asarray(values)) for key, values in storage.items()}
        return ModularRolloutBatch(**kwargs)

    def checkpoint_extra(self, evaluation: dict[str, Any] | None = None) -> dict[str, Any]:
        network = self.algorithm_config["network"]
        return {
            "environment_version": str(self.env_config.get("environment_version", ENVIRONMENT_VERSION)),
            "environment_variant": self.env_config.get("environment_variant", "direct_v2_3"),
            "observation_dim": int(network["observation_dim"]),
            "action_dim": int(network["action_dim"]),
            "num_agents": int(network["num_agents"]),
            "training_seed": self.seed, "training_gamma": self.trainer.gamma,
            "training_num_envs": self.num_envs,
            "training_total_sampled_steps": self.total_sampled_steps,
            "training_smoke": self.smoke,
            "environment_config_sha256": config_sha256(self.env_config),
            "algorithm_config_sha256": config_sha256(self.algorithm_config),
            "environment_config": self.env_config,
            "algorithm_config": self.algorithm_config,
            "network_architecture": checkpoint_architecture(self.trainer),
            "curriculum_stage": self.current_stage,
            "current_total_waves": self.current_waves,
            "curriculum_config": self.algorithm_config.get("modules", {}).get("curriculum", {}),
            "episode_indices": self.vector.episode_indices.tolist(),
            "evaluation_history": self.evaluation_history,
            "best_evaluation": self.best_evaluation,
            "best_sampled_steps": self.best_sampled_steps,
            "transition_counts": self.transition_counts.tolist(),
            "alive_agent_counts": self.alive_agent_counts.tolist(),
            "wave_clear_transition_counts": self.wave_clear_transition_counts.tolist(),
            "reward_bonus_totals": self.reward_bonus_totals.tolist(),
            "paper_reward_totals": self.paper_reward_totals.tolist(),
            "death_index_totals": self.death_index_totals.tolist(),
            "death_cause_totals": self.death_cause_totals.tolist(),
            "hidden_reset_count": self.hidden_reset_count,
            "curriculum_transitions": self.curriculum_transitions,
            "resume_count": self.resume_count,
            "evaluation": evaluation,
        }

    def save_checkpoint(self, path: str | Path, evaluation: dict[str, Any] | None = None) -> None:
        self.trainer.save(path, self.checkpoint_extra(evaluation))

    def _evaluation_key(self, row: dict[str, Any]) -> tuple[float, ...]:
        return evaluation_selection_key(row, self.env_config.get("environment_variant", "direct_v2_3"))

    def _record_evaluation(self) -> dict[str, Any]:
        row = {"sampled_steps": self.trainer.sampled_steps,
               "evaluation_seed_base": self.eval_base,
               "evaluation_seed_end": self.eval_base + self.eval_episodes - 1,
               **evaluate_modular(
            self.trainer, self.env_config, range(self.eval_base, self.eval_base + self.eval_episodes)
        )}
        self.latest_evaluation = row
        self.evaluation_history.append(row)
        path = self.output_dir / "evaluation_history.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.evaluation_history[0]))
            writer.writeheader(); writer.writerows(self.evaluation_history)
        print(self.evaluation_log_line(row), flush=True)
        if self.best_evaluation is None or self._evaluation_key(row) > self._evaluation_key(self.best_evaluation):
            old = None if self.best_evaluation is None else self._evaluation_key(self.best_evaluation)
            self.best_evaluation = dict(row); self.best_sampled_steps = self.trainer.sampled_steps
            self.save_checkpoint(self.output_dir / "best_eval.pt", row)
            print(f"[BEST] old={old} | new={self._evaluation_key(row)} | sampled_steps={self.trainer.sampled_steps}", flush=True)
        return row

    def restore_best_from_disk(self, checkpoint_steps: int) -> None:
        path = self.output_dir / "evaluation_history.csv"
        rows: list[dict[str, Any]] = []
        if path.exists():
            with path.open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    converted = {key: (float(value) if key != "sampled_steps" else int(value)) for key, value in row.items()}
                    if converted["sampled_steps"] <= checkpoint_steps:
                        rows.append(converted)
        self.evaluation_history = rows
        if rows:
            self.best_evaluation = max(rows, key=self._evaluation_key)
            self.best_sampled_steps = int(self.best_evaluation["sampled_steps"])
        best_path = self.output_dir / "best_eval.pt"
        if best_path.exists():
            state = torch.load(best_path, map_location="cpu", weights_only=False)
            if int(state.get("sampled_steps", -1)) <= checkpoint_steps:
                validate_modular_checkpoint(state, self.env_config, self.algorithm_config)
                stored = state.get("extra", {}).get("evaluation")
                if stored is not None and (self.best_evaluation is None or self._evaluation_key(stored) >= self._evaluation_key(self.best_evaluation)):
                    self.best_evaluation = stored; self.best_sampled_steps = int(state["sampled_steps"])

    def resume(self, path: str | Path) -> None:
        state = torch.load(path, map_location="cpu", weights_only=False)
        validate_modular_checkpoint(state, self.env_config, self.algorithm_config, {
            "training_seed": self.seed, "training_num_envs": self.num_envs,
            "training_smoke": self.smoke,
        })
        extra = self.trainer.load(path)
        self.current_stage = int(extra["curriculum_stage"])
        self.current_waves = int(extra["current_total_waves"])
        self.runtime_env_config = self.trainer.curriculum.runtime_config(self.env_config, self.trainer.sampled_steps)
        previous = np.asarray(extra.get("episode_indices", [0] * self.num_envs), dtype=np.int64) + 1
        self._make_vector(previous)
        self.transition_counts = np.asarray(extra.get("transition_counts", [0,0,0]), dtype=np.int64)
        self.alive_agent_counts = np.asarray(extra.get("alive_agent_counts", [0,0,0]), dtype=np.int64)
        self.wave_clear_transition_counts = np.asarray(extra.get("wave_clear_transition_counts", [0,0,0]), dtype=np.int64)
        self.reward_bonus_totals = np.asarray(extra.get("reward_bonus_totals", [0,0,0,0]), dtype=np.float64)
        self.paper_reward_totals = np.asarray(extra.get("paper_reward_totals", [0,0,0,0,0]), dtype=np.float64)
        self.death_index_totals = np.asarray(extra.get("death_index_totals", np.zeros((2,self.alive.shape[1])).tolist()), dtype=np.int64)
        self.death_cause_totals = np.asarray(extra.get("death_cause_totals", [0]*len(self.death_cause_names)), dtype=np.int64)
        self.hidden_reset_count = int(extra.get("hidden_reset_count", 0))
        self.curriculum_transitions = list(extra.get("curriculum_transitions", self.curriculum_transitions))
        self.resume_count = int(extra.get("resume_count", 0)) + 1
        self.restore_best_from_disk(self.trainer.sampled_steps)
        self.next_console = (self.trainer.sampled_steps // self.console_interval + 1) * self.console_interval
        self.next_evaluation = (self.trainer.sampled_steps // self.evaluation_interval + 1) * self.evaluation_interval
        self.next_checkpoint = (self.trainer.sampled_steps // self.checkpoint_interval + 1) * self.checkpoint_interval

    def startup_summary(self) -> dict[str, Any]:
        return {"algorithm":"modular_mappo","mode":"smoke" if self.smoke else "formal",
                "device":self.device,"seed":self.seed,"num_envs":self.num_envs,
                "total_sampled_steps":self.total_sampled_steps,"rollout_steps":self.rollout_steps,
                "gamma":self.trainer.gamma,"enabled_modules":self.trainer.module_protocol()["enabled_modules"],
                "environment_variant":self.env_config.get("environment_variant","direct_v2_3")}

    def train_log_line(self) -> str:
        rows = list(self.recent_episodes)
        mean = lambda key: float(np.mean([row[key] for row in rows])) if rows else float("nan")
        module=(f"hiddenA/C={self.last_metrics.get('actor_hidden_norm',0):.3f}/{self.last_metrics.get('critic_hidden_norm',0):.3f} "
                f"| resets={self.last_metrics.get('hidden_reset_count',0):.0f} "
                f"| chunks={self.last_metrics.get('sequence_chunks',0):.0f} "
                f"| rsteps={self.last_metrics.get('recurrent_optimizer_steps_this_update',0):.0f} "
                f"| gru_grad={self.last_metrics.get('gru_gradient_norm',0):.3f} "
                f"| popart={self.last_metrics.get('popart_std',1):.3f} "
                f"| wmean={self.last_metrics.get('effective_wave_weight_mean',1):.3f} "
                f"| bonus={self.reward_bonus_totals[0]:.2f} "
                f"| anchor={self.last_metrics.get('anchor_kl',0):.3f} "
                f"| stage={self.current_stage}/{self.current_waves}")
        return (f"[TRAIN] steps={self.trainer.sampled_steps}/{self.total_sampled_steps} | episodes={self.completed_episode_count} "
                f"| raw_return={mean('team_raw_environment_return'):.2f} | waves={mean('waves_cleared'):.2f} "
                f"| red_loss={mean('red_losses'):.2f} | blue_loss={mean('blue_losses'):.2f} "
                f"| transition={tuple(round(self.last_rollout_metrics.get(f'transition_fraction_wave_{k}',0),3) for k in (1,2,3))} "
                f"| alive={tuple(round(self.last_rollout_metrics.get(f'alive_agent_fraction_wave_{k}',0),3) for k in (1,2,3))} "
                f"| actor={self.last_metrics.get('actor_loss',float('nan')):.4f} | value={self.last_metrics.get('value_loss',float('nan')):.4f} "
                f"| H={self.last_metrics.get('entropy',float('nan')):.3f} | KL={self.last_metrics.get('approx_kl',float('nan')):.5f} "
                f"| logR=[{self.last_metrics.get('log_ratio_min',float('nan')):.2f},{self.last_metrics.get('log_ratio_max',float('nan')):.2f}] "
                f"| underflow={self.last_metrics.get('ratio_underflow_fraction',0):.4f} | {module}")

    @staticmethod
    def evaluation_log_line(row: dict[str, Any]) -> str:
        return (f"[EVAL] steps={int(row['sampled_steps'])} | W1/W2/W3="
                f"{row.get('clear_wave_1_probability',0):.2f}/{row.get('clear_wave_2_probability',0):.2f}/{row.get('clear_wave_3_probability',0):.2f} "
                f"| waves={row.get('average_waves_cleared',0):.2f} | return={row['average_return']:.2f} "
                f"| red_loss={row['average_red_loss']:.2f} | blue_loss={row['average_blue_loss']:.2f} "
                f"| K/L={row.get('kill_loss_ratio',0):.2f} | boundary={row['average_red_boundary_exits']:.2f} "
                f"| ground={row['average_red_ground_losses']:.2f}")

    def summary(self) -> dict[str, Any]:
        transition_fraction = self._fractions(self.transition_counts)
        alive_fraction = self._fractions(self.alive_agent_counts)
        pretraining = int(self.trainer.warm_start_provenance.get("pretraining_sampled_steps", 0))
        return {
            "algorithm":"modular_mappo",
            "modular_mappo_impl_version":MODULAR_MAPPO_IMPL_VERSION,
            "baseline_mappo_impl_version":MAPPO_IMPL_VERSION,
            "protocol": {**self.trainer.module_protocol(), "network_architecture":checkpoint_architecture(self.trainer),
                         "environment_config_sha256":config_sha256(self.env_config),
                         "algorithm_config_sha256":config_sha256(self.algorithm_config)},
            "sampled_steps": self.trainer.sampled_steps,
            "current_pw_training_sampled_steps": self.trainer.sampled_steps,
            "pretraining_sampled_steps": pretraining,
            "effective_total_experience_budget": pretraining + self.trainer.sampled_steps,
            "best_checkpoint_step": self.best_sampled_steps,
            "best_evaluation": self.best_evaluation,
            "latest_step": self.trainer.sampled_steps,
            "latest_evaluation": self.latest_evaluation,
            "completed_episodes": self.completed_episode_count,
            "wave_transition_counts": {f"wave_{k}":int(self.transition_counts[k-1]) for k in (1,2,3)},
            "wave_transition_fractions": {f"wave_{k}":float(transition_fraction[k-1]) for k in (1,2,3)},
            "wave_alive_agent_sample_counts": {f"wave_{k}":int(self.alive_agent_counts[k-1]) for k in (1,2,3)},
            "wave_alive_agent_sample_fractions": {f"wave_{k}":float(alive_fraction[k-1]) for k in (1,2,3)},
            "wave_clear_transition_counts": {f"wave_{k}":int(self.wave_clear_transition_counts[k-1]) for k in (1,2,3)},
            "reward_adapter_totals": {"reward_bonus_total":float(self.reward_bonus_totals[0]),**{f"reward_bonus_wave{k}":float(self.reward_bonus_totals[k]) for k in (1,2,3)}},
            "paper_R2_totals": {"blue_kill_component":float(self.paper_reward_totals[0]),
                                "red_loss_component":float(self.paper_reward_totals[1]),
                                **{f"wave_{k}":float(self.paper_reward_totals[k+1]) for k in (1,2,3)}},
            "death_index_totals": {
                **{f"blue_deaths_index_{index}":int(self.death_index_totals[0,index]) for index in range(self.death_index_totals.shape[1])},
                **{f"red_deaths_index_{index}":int(self.death_index_totals[1,index]) for index in range(self.death_index_totals.shape[1])},
            },
            "death_cause_totals": {name:int(self.death_cause_totals[index]) for index,name in enumerate(self.death_cause_names)},
            "hidden_reset_count": self.hidden_reset_count,
            "module_protocol": self.trainer.module_protocol(),
            "warm_start_provenance": self.trainer.warm_start_provenance,
            "anchor_provenance": self.trainer.anchor_provenance,
            "curriculum_transitions": self.curriculum_transitions,
            "resume_count": self.resume_count,
            "final_optimization_metrics": self.last_metrics,
        }

    def run(self) -> dict[str, Any]:
        print(f"[START] {self.startup_summary()}", flush=True)
        try:
            while self.trainer.sampled_steps < self.total_sampled_steps:
                self._maybe_curriculum()
                remaining = math.ceil((self.total_sampled_steps - self.trainer.sampled_steps) / self.num_envs)
                rollout = self.collect_rollout(min(self.rollout_steps, remaining))
                self.last_metrics = {**self.trainer.update(rollout), **self.last_rollout_metrics,
                                     "curriculum_stage":float(self.current_stage),
                                     "current_total_waves":float(self.current_waves)}
                record = {"sampled_steps":self.trainer.sampled_steps,"rollout_update":self.trainer.ppo_update_count,**self.last_metrics}
                with (self.output_dir / "optimization_metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record) + "\n")
                if self.trainer.sampled_steps >= self.next_console:
                    print(self.train_log_line(), flush=True)
                    while self.next_console <= self.trainer.sampled_steps:self.next_console += self.console_interval
                if self.trainer.sampled_steps >= self.next_evaluation:
                    self._record_evaluation()
                    while self.next_evaluation <= self.trainer.sampled_steps:self.next_evaluation += self.evaluation_interval
                if self.trainer.sampled_steps >= self.next_checkpoint:
                    path = self.output_dir / f"checkpoint_{self.trainer.sampled_steps}.pt";self.save_checkpoint(path)
                    print(f"[CHECKPOINT] path={path.name} | sampled_steps={self.trainer.sampled_steps}", flush=True)
                    while self.next_checkpoint <= self.trainer.sampled_steps:self.next_checkpoint += self.checkpoint_interval
            if self.latest_evaluation is None or int(self.latest_evaluation["sampled_steps"]) != self.trainer.sampled_steps:
                self._record_evaluation()
            self.save_checkpoint(self.output_dir / "latest.pt")
            self.save_checkpoint(self.output_dir / "final.pt")
            result = self.summary()
            (self.output_dir / "run_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"[DONE] sampled_steps={self.trainer.sampled_steps} | best={self.best_sampled_steps} | latest={self.trainer.sampled_steps}", flush=True)
            return result
        finally:
            self.vector.close()


__all__ = ["ModularMAPPOTrainingRunner"]
