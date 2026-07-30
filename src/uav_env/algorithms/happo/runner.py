"""Independent HAPPO runner for fixed homogeneous 3v3 red-team learning."""

from __future__ import annotations

import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from uav_env.algorithms.common.output_safety import prepare_output_dir
from uav_env.algorithms.common.progress_logging import format_evaluation_log, format_training_log
from uav_env.algorithms.common.reward_diagnostics import allows_truncation_bootstrap, restore_reward_component_accumulators
from uav_env.algorithms.happo.checkpoint import load_happo_checkpoint, save_happo_checkpoint
from uav_env.algorithms.happo.networks import IndependentActorSet, JointCentralizedCritic
from uav_env.algorithms.happo.rollout_buffer import HAPPORolloutBuffer
from uav_env.algorithms.happo.trainer import HAPPOTrainer
from uav_env.algorithms.mappo.adapter import CombatEnvDescription, ParallelCombatVectorEnv, SyncCombatVectorEnv, make_adapter_from_description
from uav_env.algorithms.mappo.checkpoint import schema_metadata
from uav_env.algorithms.mappo.metrics import append_csv, combat_outcome_rates, evaluation_key
from uav_env.algorithms.mappo.value_normalizer import ValueNormalizer
from uav_env.algorithms.mappo.runner import resolve_device

REWARD_COMPONENT_NAMES = (
    "situation_reward",
    "geometry_event_reward",
    "raw_shape_reward",
    "assigned_shape_reward",
    "combat_event_reward",
    "dense_reward",
    "terminal_reward",
    "terminal_base_reward",
    "mission_success_bonus",
    "hit_event_reward",
    "destroy_event_reward",
    "attacked_event_penalty",
    "destroyed_event_penalty",
    "boundary_collision_penalty",
    "support_position_raw",
    "support_coverage_raw",
    "support_safety_raw",
    "support_team_event_reward",
    "support_loss_adjustment",
)


class HAPPORunner:
    """Paper-aligned HAPPO baseline with joint rewards and scalar critic."""

    def __init__(self, config: dict[str, Any], run_name: str, output_root: str | Path = "outputs/happo") -> None:
        self.config = config
        self.seed = int(config["seed"])
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        self.device = resolve_device(str(config["device"]))
        print(f"HAPPO device: {self.device}")
        run_id = str(config["run_id"]) if "run_id" in config and config["run_id"] is not None else datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = prepare_output_dir(output_root, run_name, run_id)
        env_cfg = config["environment"]
        self.description = CombatEnvDescription(str(env_cfg["kind"]), str(env_cfg["scenario"]), str(env_cfg["opponent"]), env_cfg.get("multi_terminal_reward_profile"), env_cfg.get("functional_mode"), tuple(env_cfg["red_roles"]) if "red_roles" in env_cfg else None, env_cfg.get("relay_enabled"))
        probe = make_adapter_from_description(self.description)
        self.num_agents, self.obs_dim, self.state_dim = probe.num_agents, probe.obs_dim, probe.state_dim
        if self.num_agents != 3:
            raise ValueError("HAPPO baseline is scoped to fixed red_count=3")
        if hasattr(probe.env, "config"):
            for key in ("environment_schema_version", "observation_schema", "global_state_schema", "reward_profile", "scenario_profile"):
                if key in probe.env.config:
                    self.config["environment"][key] = probe.env.config[key]
        probe.env.close()
        self.schema_metadata = schema_metadata(self.config, self.obs_dim, self.state_dim, self.num_agents)
        if str(self.schema_metadata["environment_schema_version"]) != "homogeneous_3v3_v2_timeaware":
            raise ValueError("HAPPO baseline requires homogeneous_3v3_v2_timeaware")
        self.vector = (
            ParallelCombatVectorEnv(self.description, int(config["num_envs"]), self.seed)
            if config.get("vector_env", "sync") == "parallel"
            else SyncCombatVectorEnv([lambda description=self.description: make_adapter_from_description(description) for _ in range(int(config["num_envs"]))], self.seed)
        )
        self.actors = IndependentActorSet(
            [self.obs_dim] * self.num_agents,
            [15] * self.num_agents,
            config["actor_hidden_sizes"],
            config["activation"],
            self.seed,
        ).to(self.device)
        self.critic = JointCentralizedCritic(self.state_dim, config["critic_hidden_sizes"], config["activation"]).to(self.device)
        self.normalizer = ValueNormalizer()
        self.trainer = HAPPOTrainer(self.actors, self.critic, config, self.normalizer, self.device)
        (self.output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        self.writer = SummaryWriter(self.output_dir / "tensorboard")
        self.environment_steps = 0
        self.update_index = 0
        self.best_evaluation: dict[str, Any] | None = None
        self.current = self.vector.reset()
        self.episodes = 0
        self.episode_team_return_accumulators = np.zeros(int(config["num_envs"]), dtype=np.float64)
        self.episode_agent_sum_return_accumulators = np.zeros(int(config["num_envs"]), dtype=np.float64)
        self.reward_component_episode_accumulators = {
            name: np.zeros(int(config["num_envs"]), dtype=np.float64) for name in REWARD_COMPONENT_NAMES
        }
        self.last_evaluation_step: int | None = None
        self._closed = False

    def _values(self, states: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            values = self.critic(torch.as_tensor(states, device=self.device))
        return values.cpu().numpy().astype(np.float32)

    def _action_distributions(self, obs: np.ndarray, available: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
        actions = np.zeros((obs.shape[0], self.num_agents), dtype=np.int64)
        log_probs = np.zeros((obs.shape[0], self.num_agents), dtype=np.float32)
        entropies = np.zeros((obs.shape[0], self.num_agents), dtype=np.float32)
        logits_by_actor: list[np.ndarray] = []
        margins_by_actor: list[np.ndarray] = []
        with torch.no_grad():
            for agent_id in range(self.num_agents):
                logits = self.actors[agent_id](
                    torch.as_tensor(obs[:, agent_id, :], device=self.device),
                    torch.as_tensor(available[:, agent_id, :], device=self.device),
                )
                dist = torch.distributions.Categorical(logits=logits)
                sampled = dist.sample()
                actions[:, agent_id] = sampled.cpu().numpy()
                log_probs[:, agent_id] = dist.log_prob(sampled).cpu().numpy()
                entropies[:, agent_id] = dist.entropy().cpu().numpy()
                logits_np = logits.cpu().numpy()
                logits_by_actor.append(logits_np)
                top2 = np.sort(np.partition(logits_np, -2, axis=-1)[:, -2:], axis=-1)
                margins_by_actor.append(top2[:, 1] - top2[:, 0])
        return actions, log_probs, entropies, logits_by_actor, margins_by_actor

    def _allows_truncation_bootstrap(self, step: Any) -> bool:
        if not step.truncated:
            return False
        reason = str(step.info["outcome"].termination_reason)
        return allows_truncation_bootstrap(str(self.schema_metadata["environment_schema_version"]), True, reason)

    def collect(self) -> tuple[HAPPORolloutBuffer, dict[str, float]]:
        t, e = int(self.config["rollout_length"]), int(self.config["num_envs"])
        buffer = HAPPORolloutBuffer(t, e, self.num_agents, self.obs_dim, self.state_dim)
        buffer.set_initial(self.current["local_obs"], self.current["global_state"], self.current["available_actions"], self.current["alive_masks"])
        rollout_returns: list[float] = []
        rollout_agent_sum_returns: list[float] = []
        action_counts = np.zeros((self.num_agents, 15), dtype=np.float64)
        entropy_records = [[] for _ in range(self.num_agents)]
        margin_records = [[] for _ in range(self.num_agents)]
        episode_timeouts: list[float] = []
        side_names = ("attack_attempts", "hits", "effective_damage", "nominal_damage", "overkill_damage", "attack_area_steps", "ground_crashes", "ceiling_violations", "collisions", "survivors")
        side_rollout = {f"{team}_{name}": [] for team in ("red", "blue") for name in side_names}
        component_values = {name: [] for name in REWARD_COMPONENT_NAMES}
        component_totals = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
        component_abs_totals = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
        component_episode_values = {name: [] for name in REWARD_COMPONENT_NAMES}
        truncated_episode_count = 0
        truncation_no_bootstrap_count = 0
        for _ in range(t):
            values = self._values(self.current["global_state"])
            actions, log_probs, entropies, _, margins = self._action_distributions(self.current["local_obs"], self.current["available_actions"])
            alive = np.asarray(self.current["alive_masks"], dtype=bool)
            for agent_id in range(self.num_agents):
                active_actions = actions[:, agent_id][alive[:, agent_id]]
                for action in active_actions:
                    action_counts[agent_id, int(action)] += 1
                entropy_records[agent_id].extend(entropies[:, agent_id][alive[:, agent_id]].tolist())
                margin_records[agent_id].extend(margins[agent_id][alive[:, agent_id]].tolist())
            result = self.vector.step(actions)
            terminal_values = np.zeros(e, dtype=np.float32)
            truncation_bootstrap_mask = np.zeros(e, dtype=np.float32)
            self.episode_team_return_accumulators += np.asarray(result["team_rewards"], dtype=np.float64)
            self.episode_agent_sum_return_accumulators += np.asarray(result["agent_reward_sums"], dtype=np.float64)
            for env_index, info in enumerate(result.get("infos", [])):
                step_components = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
                for breakdown in info.get("agent_reward_breakdowns", {}).values():
                    per_agent = {
                        "situation_reward": float(breakdown.situation),
                        "geometry_event_reward": float(breakdown.geometry_event),
                        "raw_shape_reward": float(breakdown.raw_shape),
                        "assigned_shape_reward": float(breakdown.assigned_shape),
                        "combat_event_reward": float(breakdown.combat_event),
                        "dense_reward": float(breakdown.dense_reward),
                        "terminal_reward": float(breakdown.terminal),
                        "terminal_base_reward": float(breakdown.terminal_base_reward),
                        "mission_success_bonus": float(breakdown.mission_success_bonus),
                        "hit_event_reward": float(breakdown.hit_event_reward),
                        "destroy_event_reward": float(breakdown.destroy_event_reward),
                        "attacked_event_penalty": float(breakdown.attacked_event_penalty),
                        "destroyed_event_penalty": float(breakdown.destroyed_event_penalty),
                        "boundary_collision_penalty": float(breakdown.boundary_collision_penalty),
                        "support_position_raw": float(breakdown.support_position),
                        "support_coverage_raw": float(breakdown.support_coverage),
                        "support_safety_raw": float(breakdown.support_safety),
                        "support_team_event_reward": float(breakdown.support_team_event),
                        "support_loss_adjustment": float(breakdown.support_loss_adjustment),
                    }
                    for name, value in per_agent.items():
                        component_values[name].append(value)
                        step_components[name] += value
                for name, value in step_components.items():
                    component_totals[name] += value
                    component_abs_totals[name] += abs(value)
                    self.reward_component_episode_accumulators[name][env_index] += value
            for index, step in enumerate(result["terminal_steps"]):
                if step.truncated:
                    truncated_episode_count += 1
                    if self._allows_truncation_bootstrap(step):
                        terminal_values[index] = self._values(step.global_state[None, :])[0]
                        truncation_bootstrap_mask[index] = 1.0
                    else:
                        truncation_no_bootstrap_count += 1
                if step.terminated or step.truncated:
                    self.episodes += 1
                    rollout_returns.append(float(self.episode_team_return_accumulators[index]))
                    rollout_agent_sum_returns.append(float(self.episode_agent_sum_return_accumulators[index]))
                    self.episode_team_return_accumulators[index] = 0.0
                    self.episode_agent_sum_return_accumulators[index] = 0.0
                    for name in REWARD_COMPONENT_NAMES:
                        component_episode_values[name].append(float(self.reward_component_episode_accumulators[name][index]))
                        self.reward_component_episode_accumulators[name][index] = 0.0
                    outcome = step.info["outcome"]
                    episode_timeouts.append(float(outcome.termination_reason == "timeout"))
                    stats = step.info.get("statistics", {})
                    aircraft = stats.get("aircraft", {})
                    for team in ("red", "blue"):
                        ids = [f"{team}_{i}" for i in range(self.num_agents)]
                        for name in ("attack_attempts", "hits", "effective_damage", "nominal_damage", "overkill_damage", "attack_area_steps", "ground_crashes", "ceiling_violations", "collisions"):
                            side_rollout[f"{team}_{name}"].append(float(sum(float(aircraft.get(key, {}).get(name, 0.0)) for key in ids)))
                    side_rollout["red_survivors"].append(float(outcome.red_survivors))
                    side_rollout["blue_survivors"].append(float(outcome.blue_survivors))
            buffer.insert(
                actions,
                log_probs,
                values,
                np.asarray(result["team_rewards"], dtype=np.float32),
                np.asarray(result["rewards"], dtype=np.float32),
                result["terminated"],
                result["truncated"],
                self.current["alive_masks"],
                result["next_local_obs"],
                result["next_global_state"],
                result["next_available_actions"],
                result["next_alive_masks"],
                terminal_values,
                truncation_bootstrap_mask,
            )
            self.current = {
                "local_obs": result["next_local_obs"],
                "global_state": result["next_global_state"],
                "alive_masks": result["next_alive_masks"],
                "available_actions": result["next_available_actions"],
            }
        buffer.finish(self._values(self.current["global_state"]), float(self.config["gamma"]), float(self.config["gae_lambda"]))
        diagnostics: dict[str, float] = {
            "rollout_team_episode_return_mean": float(np.mean(rollout_returns)) if rollout_returns else 0.0,
            "rollout_agent_sum_episode_return_mean": float(np.mean(rollout_agent_sum_returns)) if rollout_agent_sum_returns else 0.0,
            "rollout_mean_per_agent_episode_return": float(np.mean(rollout_agent_sum_returns)) / self.num_agents if rollout_agent_sum_returns else 0.0,
            "rollout_episode_count": float(len(rollout_returns)),
            "timeout_rate": float(np.mean(episode_timeouts)) if episode_timeouts else 0.0,
            "truncated_episode_count": float(truncated_episode_count),
            "truncation_no_bootstrap_count": float(truncation_no_bootstrap_count),
            "team_reward_mean": float(buffer.team_rewards.mean()),
            "joint_advantage_raw_mean": float(buffer.advantages.mean()),
            "joint_advantage_raw_std": float(buffer.advantages.std()),
            "agent_reward_sum_mean": float(buffer.agent_rewards.sum(axis=2).mean()),
        }
        for agent_id in range(self.num_agents):
            total = max(action_counts[agent_id].sum(), 1.0)
            for action_id in range(15):
                diagnostics[f"actor_{agent_id}_action_{action_id}_frequency"] = float(action_counts[agent_id, action_id] / total)
            diagnostics[f"actor_{agent_id}_policy_entropy_collect"] = float(np.mean(entropy_records[agent_id])) if entropy_records[agent_id] else 0.0
            diagnostics[f"actor_{agent_id}_top1_top2_margin_collect"] = float(np.mean(margin_records[agent_id])) if margin_records[agent_id] else 0.0
        for name, values in side_rollout.items():
            diagnostics[f"rollout_{name}_mean"] = float(np.mean(values)) if values else 0.0
        for name in REWARD_COMPONENT_NAMES:
            diagnostics[f"{name}_mean"] = float(np.mean(component_values[name])) if component_values[name] else 0.0
            diagnostics[f"{name}_per_step"] = float(component_totals[name] / max(t * e, 1))
            diagnostics[f"{name}_abs_mean"] = float(component_abs_totals[name] / max(t * e, 1))
            diagnostics[f"{name}_per_episode"] = float(np.mean(component_episode_values[name])) if component_episode_values[name] else 0.0
        return buffer, diagnostics

    def evaluate(self, episodes: int | None = None, seed_start: int = 100000, deterministic: bool | None = None) -> dict[str, float]:
        count = int(episodes or self.config["validation_episodes"])
        deterministic = bool(self.config.get("deterministic_evaluation", True)) if deterministic is None else deterministic
        outcomes, returns, agent_sum_returns, steps = [], [], [], []
        terminal_proportions: list[float] = []
        red_crashes: list[float] = []
        blue_crashes: list[float] = []
        frequencies = np.zeros((self.num_agents, 15), dtype=np.float64)
        entropy_records = [[] for _ in range(self.num_agents)]
        margin_records = [[] for _ in range(self.num_agents)]
        side_metrics = {f"{team}_{name}": [] for team in ("red", "blue") for name in ("attack_attempts", "hits", "effective_damage", "nominal_damage", "overkill_damage", "attack_area_steps", "ground_crashes", "ceiling_violations", "collisions", "survivors")}
        for episode in range(count):
            env = make_adapter_from_description(self.description)
            current = env.reset(seed_start + episode)
            total, agent_sum_total, absolute_total, terminal_absolute, done = 0.0, 0.0, 0.0, 0.0, False
            while not done:
                action = np.zeros(self.num_agents, dtype=np.int64)
                with torch.no_grad():
                    for agent_id in range(self.num_agents):
                        logits = self.actors[agent_id](
                            torch.as_tensor(current.local_obs[agent_id:agent_id + 1], device=self.device),
                            torch.as_tensor(current.available_action_mask[agent_id:agent_id + 1], device=self.device),
                        )
                        dist = torch.distributions.Categorical(logits=logits)
                        selected = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
                        action[agent_id] = int(selected.item())
                        if current.agent_alive_mask[agent_id]:
                            entropy_records[agent_id].append(float(dist.entropy().item()))
                            top2 = torch.topk(logits, 2, dim=-1).values
                            margin_records[agent_id].append(float((top2[0, 0] - top2[0, 1]).item()))
                            frequencies[agent_id, int(action[agent_id])] += 1
                current = env.step(action)
                total += current.team_reward
                agent_sum_total += current.agent_reward_sum
                absolute_total += abs(current.team_reward)
                if "agent_reward_breakdowns" in current.info:
                    terminal_absolute += abs(float(np.mean([v.terminal_base_reward + v.mission_success_bonus for v in current.info["agent_reward_breakdowns"].values()])))
                done = current.terminated or current.truncated
            outcome = current.info["outcome"]
            outcomes.append(outcome)
            returns.append(total)
            agent_sum_returns.append(agent_sum_total)
            terminal_proportions.append(terminal_absolute / max(absolute_total, 1.0e-12))
            steps.append(outcome.decision_steps)
            aircraft = current.info.get("statistics", {}).get("aircraft", {})
            red_crashes.append(float(any(float(aircraft.get(f"red_{i}", {}).get("ground_crashes", 0.0)) > 0.0 for i in range(self.num_agents))))
            blue_crashes.append(float(any(float(aircraft.get(f"blue_{i}", {}).get("ground_crashes", 0.0)) > 0.0 for i in range(self.num_agents))))
            for team in ("red", "blue"):
                ids = [f"{team}_{i}" for i in range(self.num_agents)]
                for name in ("attack_attempts", "hits", "effective_damage", "nominal_damage", "overkill_damage", "attack_area_steps", "ground_crashes", "ceiling_violations", "collisions"):
                    side_metrics[f"{team}_{name}"].append(float(sum(aircraft.get(key, {}).get(name, 0.0) for key in ids)))
            side_metrics["red_survivors"].append(float(outcome.red_survivors))
            side_metrics["blue_survivors"].append(float(outcome.blue_survivors))
            env.env.close()
        combat_rates = combat_outcome_rates(outcomes)
        result = {
            **combat_rates,
            "red_win_rate": combat_rates["overall_red_win_rate"],
            "blue_win_rate": sum(o.winner == "blue" for o in outcomes) / count,
            "red_crash_rate": float(np.mean(red_crashes)),
            "blue_crash_rate": float(np.mean(blue_crashes)),
            "mean_episode_return": float(np.mean(returns)),
            "mean_team_episode_return": float(np.mean(returns)),
            "mean_agent_sum_episode_return": float(np.mean(agent_sum_returns)),
            "mean_per_agent_episode_return": float(np.mean(agent_sum_returns)) / self.num_agents,
            "mean_episode_steps": float(np.mean(steps)),
            "terminal_reward_proportion": float(np.mean(terminal_proportions)),
        }
        for agent_id in range(self.num_agents):
            total = max(frequencies[agent_id].sum(), 1.0)
            for action_id in range(15):
                result[f"actor_{agent_id}_action_{action_id}_frequency"] = float(frequencies[agent_id, action_id] / total)
            result[f"actor_{agent_id}_policy_entropy"] = float(np.mean(entropy_records[agent_id])) if entropy_records[agent_id] else 0.0
            result[f"actor_{agent_id}_top1_top2_logit_margin"] = float(np.mean(margin_records[agent_id])) if margin_records[agent_id] else 0.0
        for name, values in side_metrics.items():
            result[f"mean_{name}"] = float(np.mean(values)) if values else 0.0
        result["mean_effective_damage"] = result["mean_red_effective_damage"]
        result["mean_hits"] = result["mean_red_hits"]
        result["mean_attack_area_steps"] = result["mean_red_attack_area_steps"]
        result["mean_red_survivors"] = result["mean_red_survivors"]
        result["mean_blue_survivors"] = result["mean_blue_survivors"]
        result["mean_survivor_difference"] = result["mean_red_survivors"] - result["mean_blue_survivors"]
        return result

    def resume(self, path: str, actor_only: bool = False) -> None:
        data = load_happo_checkpoint(
            path,
            self.actors,
            self.critic,
            self.trainer.actor_optimizers,
            self.trainer.critic_optimizer,
            self.normalizer,
            actor_only,
            self.device,
            self.schema_metadata,
        )
        if actor_only:
            return
        self.environment_steps = int(data["environment_steps"])
        self.update_index = int(data["update_index"])
        self.best_evaluation = data["best_evaluation"]
        state = data["runner_state"]
        self.vector.set_state(state["vector_env_state"])
        self.current = state["current"]
        self.episodes = int(state["episodes"])
        self.episode_team_return_accumulators = np.asarray(state["episode_team_return_accumulators"], dtype=np.float64)
        self.episode_agent_sum_return_accumulators = np.asarray(state["episode_agent_sum_return_accumulators"], dtype=np.float64)
        self._restore_reward_component_accumulators(state.get("reward_component_episode_accumulators"))
        self.last_evaluation_step = state.get("last_evaluation_step")
        self.trainer.order_rng.bit_generator.state = state["agent_order_rng_state"]
        for rng, rng_state in zip(self.trainer.actor_minibatch_rngs, state["actor_minibatch_rng_states"]):
            rng.bit_generator.state = rng_state
        self.trainer.critic_minibatch_rng.bit_generator.state = state["critic_minibatch_rng_state"]

    def _restore_reward_component_accumulators(self, state: Any) -> None:
        expected_shape = (int(self.config["num_envs"]),)
        self.reward_component_episode_accumulators = restore_reward_component_accumulators(
            state,
            REWARD_COMPONENT_NAMES,
            expected_shape,
            error_prefix="HAPPO reward_component_episode_accumulators",
        )

    def _save(self, name: str) -> None:
        runner_state = {
            "vector_env_state": self.vector.get_state(),
            "current": self.current,
            "episodes": self.episodes,
            "episode_team_return_accumulators": self.episode_team_return_accumulators,
            "episode_agent_sum_return_accumulators": self.episode_agent_sum_return_accumulators,
            "reward_component_episode_accumulators": self.reward_component_episode_accumulators,
            "last_evaluation_step": self.last_evaluation_step,
            "agent_order_rng_state": self.trainer.order_rng.bit_generator.state,
            "actor_minibatch_rng_states": [rng.bit_generator.state for rng in self.trainer.actor_minibatch_rngs],
            "critic_minibatch_rng_state": self.trainer.critic_minibatch_rng.bit_generator.state,
        }
        save_happo_checkpoint(
            self.output_dir / "checkpoints" / name,
            self.actors,
            self.critic,
            self.trainer.actor_optimizers,
            self.trainer.critic_optimizer,
            self.normalizer,
            self.config,
            self.environment_steps,
            self.update_index,
            self.best_evaluation,
            runner_state,
            self.schema_metadata,
        )

    def _run_impl(self) -> Path:
        started = time.time()
        start_steps = self.environment_steps
        total = int(self.config["total_env_steps"])
        log_interval = int(self.config.get("log_interval", 1))
        if self.environment_steps == 0:
            self._save("initial.pt")
        while self.environment_steps < total:
            buffer, rollout = self.collect()
            metrics = self.trainer.update(buffer)
            self.environment_steps += int(self.config["rollout_length"]) * int(self.config["num_envs"])
            self.update_index += 1
            elapsed = time.time() - started
            row = {
                "environment_steps": self.environment_steps,
                "decisions": self.environment_steps * self.num_agents,
                "episodes": self.episodes,
                "update_index": self.update_index,
                "wall_time": elapsed,
                "samples_per_second": (self.environment_steps - start_steps) / max(elapsed, 1e-9),
                **metrics,
                **rollout,
            }
            append_csv(self.output_dir / "metrics.csv", row)
            if log_interval > 0 and self.update_index % log_interval == 0:
                print(format_training_log("HAPPO", row), flush=True)
            for key, value in row.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, self.environment_steps)
            if self.environment_steps % int(self.config["evaluation_interval"]) < int(self.config["rollout_length"]) * int(self.config["num_envs"]):
                evaluation = {"environment_steps": self.environment_steps, "evaluation_split": "validation", **self.evaluate(int(self.config["validation_episodes"]), int(self.config["validation_seed_start"]))}
                append_csv(self.output_dir / "evaluations.csv", evaluation)
                if log_interval > 0:
                    print(format_evaluation_log("HAPPO", evaluation), flush=True)
                self.last_evaluation_step = self.environment_steps
                if self.best_evaluation is None or evaluation_key(evaluation, str(self.config["checkpoint_selection"])) > evaluation_key(self.best_evaluation, str(self.config["checkpoint_selection"])):
                    self.best_evaluation = evaluation
                    self._save("best.pt")
            if self.environment_steps < total and self.environment_steps % int(self.config["checkpoint_interval"]) < int(self.config["rollout_length"]) * int(self.config["num_envs"]):
                self._save(f"step_{self.environment_steps}.pt")
        self._save("last.pt")
        if self.last_evaluation_step != self.environment_steps:
            evaluation = {
                "environment_steps": self.environment_steps,
                "evaluation_split": "validation",
                **self.evaluate(int(self.config["validation_episodes"]), int(self.config["validation_seed_start"])),
            }
            append_csv(self.output_dir / "evaluations.csv", evaluation)
            if log_interval > 0:
                print(format_evaluation_log("HAPPO", evaluation), flush=True)
            self.last_evaluation_step = self.environment_steps
            if self.best_evaluation is None or evaluation_key(evaluation, str(self.config["checkpoint_selection"])) > evaluation_key(self.best_evaluation, str(self.config["checkpoint_selection"])):
                self.best_evaluation = evaluation
                self._save("best.pt")
        if self.best_evaluation is None:
            self.best_evaluation = {
                "environment_steps": self.environment_steps,
                "evaluation_split": "validation",
                **self.evaluate(int(self.config["validation_episodes"]), int(self.config["validation_seed_start"])),
            }
            append_csv(self.output_dir / "evaluations.csv", self.best_evaluation)
            if log_interval > 0:
                print(format_evaluation_log("HAPPO", self.best_evaluation), flush=True)
            self.last_evaluation_step = self.environment_steps
            self._save("best.pt")
        if not (self.output_dir / "checkpoints" / "best.pt").is_file():
            self._save("best.pt")
        test_evaluations = {}
        for label in ("initial", "last", "best"):
            self.resume(str(self.output_dir / "checkpoints" / f"{label}.pt"), actor_only=True)
            test_evaluations[label] = self.evaluate(
                int(self.config["test_episodes"]),
                int(self.config["test_seed_start"]),
                deterministic=True,
            )
            if log_interval > 0:
                print(format_evaluation_log("HAPPO", {"environment_steps": self.environment_steps, "evaluation_split": f"test_{label}", **test_evaluations[label]}), flush=True)
        wall_time = time.time() - started
        summary = {
            "environment_steps": self.environment_steps,
            "updates": self.update_index,
            "episodes": self.episodes,
            "device": str(self.device),
            "schema_metadata": self.schema_metadata,
            "validation_best_evaluation": self.best_evaluation,
            "test_evaluations": test_evaluations,
            "wall_time": wall_time,
        }
        (self.output_dir / "final_summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
        self.writer.flush()
        return self.output_dir

    def run(self) -> Path:
        try:
            return self._run_impl()
        except KeyboardInterrupt:
            interrupted = self.output_dir / "checkpoints" / "interrupted.pt"
            try:
                self._save("interrupted.pt")
                print(f"HAPPO interrupted; checkpoint saved to {interrupted.resolve()}")
            except Exception as error:
                print(f"HAPPO interrupted; failed to save interrupted checkpoint: {error}")
            raise
        finally:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.vector.close()
        self.writer.close()
