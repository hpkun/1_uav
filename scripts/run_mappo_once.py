"""Run one configured combat episode and print rewards without training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from uav_env.algorithms.happo.config import load_happo_config
from uav_env.algorithms.mappo.adapter import CombatEnvDescription, make_adapter_from_description
from uav_env.algorithms.mappo.config import load_mappo_config


def _load_config(path: str, algorithm: str) -> dict[str, Any]:
    if algorithm == "mappo":
        return load_mappo_config(path)
    if algorithm == "happo":
        return load_happo_config(path)
    try:
        return load_mappo_config(path)
    except Exception:
        return load_happo_config(path)


def _sample_actions(rng: np.random.Generator, available: np.ndarray, policy: str) -> np.ndarray:
    if policy == "hold":
        return np.zeros(available.shape[0], dtype=np.int64)
    actions = np.zeros(available.shape[0], dtype=np.int64)
    for agent_id, mask in enumerate(available):
        valid = np.flatnonzero(mask)
        actions[agent_id] = int(rng.choice(valid)) if valid.size else 0
    return actions


def _reward_components(info: dict[str, Any]) -> dict[str, float]:
    breakdowns = info.get("agent_reward_breakdowns", {})
    fields = ("situation", "geometry_event", "combat_event", "assigned_dense", "terminal", "total")
    return {
        field: float(sum(float(getattr(breakdown, field, 0.0)) for breakdown in breakdowns.values()))
        for field in fields
    }


def _side_statistics(info: dict[str, Any], num_agents: int) -> dict[str, float]:
    aircraft = info.get("statistics", {}).get("aircraft", {})
    result: dict[str, float] = {}
    for team in ("red", "blue"):
        ids = [f"{team}_{index}" for index in range(num_agents)]
        for name in ("attack_attempts", "hits", "effective_damage", "attack_area_steps", "ground_crashes", "collisions"):
            result[f"{team}_{name}"] = float(sum(float(aircraft.get(agent_id, {}).get(name, 0.0)) for agent_id in ids))
    return result


def run_once(config_path: str, algorithm: str, seed: int, policy: str, max_steps: int | None, step_log_interval: int) -> dict[str, Any]:
    config = _load_config(config_path, algorithm)
    env_cfg = config["environment"]
    description = CombatEnvDescription(
        str(env_cfg["kind"]),
        str(env_cfg["scenario"]),
        str(env_cfg["opponent"]),
        env_cfg.get("multi_terminal_reward_profile"),
    )
    adapter = make_adapter_from_description(description, seed)
    rng = np.random.default_rng(seed)
    current = adapter.reset(seed)
    limit = int(max_steps or adapter.env.config["max_decision_steps"])
    team_return = 0.0
    agent_sum_return = 0.0
    step_count = 0
    last_info: dict[str, Any] = current.info
    try:
        while step_count < limit:
            actions = _sample_actions(rng, current.available_action_mask, policy)
            current = adapter.step(actions)
            step_count += 1
            team_return += float(current.team_reward)
            agent_sum_return += float(current.agent_reward_sum)
            last_info = current.info
            components = _reward_components(current.info)
            if step_log_interval > 0 and step_count % step_log_interval == 0:
                print(
                    f"[step {step_count:04d}] "
                    f"team_reward={current.team_reward:.4f} "
                    f"agent_sum={current.agent_reward_sum:.4f} "
                    f"dense={components['assigned_dense']:.4f} "
                    f"event={components['combat_event']:.4f} "
                    f"terminal={components['terminal']:.4f} "
                    f"actions={actions.tolist()}",
                    flush=True,
                )
            if current.terminated or current.truncated:
                break
    finally:
        adapter.env.close()
    outcome = last_info.get("outcome")
    summary = {
        "config": str(Path(config_path)),
        "algorithm_config_type": "happo" if config.get("algorithm") == "happo" else "mappo",
        "scenario": description.scenario,
        "opponent": description.opponent,
        "policy": policy,
        "seed": seed,
        "decision_steps": step_count,
        "stopped_by_max_steps": step_count >= limit and not (current.terminated or current.truncated),
        "terminated": bool(current.terminated),
        "truncated": bool(current.truncated),
        "winner": str(getattr(outcome, "winner", "none")),
        "termination_reason": str(getattr(outcome, "termination_reason", "none")),
        "team_return": team_return,
        "agent_sum_return": agent_sum_return,
        "mean_per_agent_return": agent_sum_return / max(float(adapter.num_agents), 1.0),
        "last_step_agent_rewards": current.agent_rewards.astype(float).tolist(),
        "last_step_components_sum": _reward_components(last_info),
        "side_statistics": _side_statistics(last_info, adapter.num_agents),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one configured UAV episode and print rewards; no training, no checkpoint, no audit.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--algorithm", choices=["auto", "mappo", "happo"], default="auto")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--policy", choices=["hold", "random"], default="hold")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--step-log-interval", type=int, default=1, help="Print every N steps; use 0 for final JSON only")
    args = parser.parse_args()
    summary = run_once(args.config, args.algorithm, args.seed, args.policy, args.max_steps, args.step_log_interval)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
