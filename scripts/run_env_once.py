"""Run one configured UAV combat environment episode without training."""

from __future__ import annotations

import argparse
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np

from uav_env.actions.discrete_15 import DiscreteAction15
from uav_env.algorithms.happo.config import load_happo_config
from uav_env.algorithms.mappo.adapter import CombatEnvDescription, make_adapter_from_description
from uav_env.algorithms.mappo.config import load_mappo_config


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a nonnegative integer")
    return parsed


def load_algorithm_config(path: str, algorithm: str) -> dict[str, Any]:
    """Load only the selected algorithm config type; do not auto-fallback."""

    if algorithm == "mappo":
        return load_mappo_config(path)
    if algorithm == "happo":
        return load_happo_config(path)
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def sample_actions(rng: np.random.Generator, available: np.ndarray, policy: str) -> np.ndarray:
    if policy not in {"hold", "random"}:
        raise ValueError("policy must be 'hold' or 'random'")
    mask_array = np.asarray(available, dtype=bool)
    hold = int(DiscreteAction15.LEVEL_HOLD)
    if policy == "hold":
        actions = np.full(mask_array.shape[0], hold, dtype=np.int64)
        for agent_id, mask in enumerate(mask_array):
            valid = np.flatnonzero(mask)
            if valid.size and hold not in valid:
                # Mask-consistency fallback: keep the probe valid without changing the environment mask.
                actions[agent_id] = int(valid[0])
        return actions
    actions = np.zeros(mask_array.shape[0], dtype=np.int64)
    for agent_id, mask in enumerate(mask_array):
        valid = np.flatnonzero(mask)
        actions[agent_id] = int(rng.choice(valid)) if valid.size else hold
    return actions


def reward_component_sums(info: dict[str, Any]) -> dict[str, float]:
    breakdowns = info.get("agent_reward_breakdowns", {})
    fields = {
        "situation": "situation",
        "geometry_event": "geometry_event",
        "combat_event": "combat_event",
        "assigned_shape": "assigned_shape",
        "assigned_dense": "assigned_dense",
        "dense_reward": "dense_reward",
        "terminal": "terminal",
        "terminal_base_reward": "terminal_base_reward",
        "mission_success_bonus": "mission_success_bonus",
        "support_position_raw": "support_position",
        "support_coverage_raw": "support_coverage",
        "support_safety_raw": "support_safety",
        "support_team_event": "support_team_event",
        "support_loss_adjustment": "support_loss_adjustment",
        "total": "total",
    }
    return {
        output: float(sum(float(getattr(breakdown, attr, 0.0)) for breakdown in breakdowns.values()))
        for output, attr in fields.items()
    }


def side_statistics(info: dict[str, Any], num_agents: int) -> dict[str, float]:
    statistics = info.get("statistics", {})
    aircraft = info.get("statistics", {}).get("aircraft", {})
    result: dict[str, float] = {}
    for team in ("red", "blue"):
        ids = [f"{team}_{index}" for index in range(num_agents)]
        for name in (
            "attack_attempts",
            "hits",
            "effective_damage",
            "ground_crashes",
            "ceiling_violations",
            "collisions",
        ):
            result[f"{team}_{name}"] = float(sum(float(aircraft.get(agent_id, {}).get(name, 0.0)) for agent_id in ids))
    result["collisions"] = float(statistics.get("collisions", 0.0))
    return result


def assert_finite_numbers(value: Any, path: str = "summary") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite_numbers(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite_numbers(item, f"{path}[{index}]")
    elif isinstance(value, Real) and not isinstance(value, bool):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"{path} is not finite: {value}")


def zero_reward_components() -> dict[str, float]:
    return {
        key: 0.0
        for key in (
            "situation",
            "geometry_event",
            "combat_event",
            "assigned_shape",
            "assigned_dense",
            "dense_reward",
            "terminal",
            "terminal_base_reward",
            "mission_success_bonus",
            "support_position_raw",
            "support_coverage_raw",
            "support_safety_raw",
            "support_team_event",
            "support_loss_adjustment",
            "total",
        )
    }


def add_components(total: dict[str, float], step: dict[str, float]) -> None:
    for key in total:
        total[key] += float(step.get(key, 0.0))


def _validate_run_once_inputs(algorithm: str, seed: int, policy: str, max_steps: int | None, step_log_interval: int) -> int:
    if algorithm not in {"mappo", "happo"}:
        raise ValueError("algorithm must be 'mappo' or 'happo'")
    if policy not in {"hold", "random"}:
        raise ValueError("policy must be 'hold' or 'random'")
    if max_steps is not None:
        if isinstance(max_steps, bool) or not isinstance(max_steps, Integral) or int(max_steps) <= 0:
            raise ValueError("max_steps must be None or a positive integer")
    if isinstance(step_log_interval, bool) or not isinstance(step_log_interval, Integral) or int(step_log_interval) < 0:
        raise ValueError("step_log_interval must be a nonnegative integer")
    if isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if isinstance(seed, Integral):
        return int(seed)
    if isinstance(seed, str):
        text = seed.strip()
        if text and (text.isdigit() or (text[0] in {"+", "-"} and text[1:].isdigit())):
            return int(text)
    raise ValueError("seed must be an integer")


def run_once(config_path: str, algorithm: str, seed: int, policy: str, max_steps: int | None, step_log_interval: int) -> dict[str, Any]:
    seed = _validate_run_once_inputs(algorithm, seed, policy, max_steps, step_log_interval)
    config = load_algorithm_config(config_path, algorithm)
    env_cfg = config["environment"]
    description = CombatEnvDescription(
        str(env_cfg["kind"]),
        str(env_cfg["scenario"]),
        str(env_cfg["opponent"]),
        env_cfg.get("multi_terminal_reward_profile"),
        env_cfg.get("functional_mode"),
        tuple(env_cfg["red_roles"]) if "red_roles" in env_cfg else None,
        env_cfg.get("relay_enabled"),
    )
    adapter = make_adapter_from_description(description, seed)
    rng = np.random.default_rng(seed)
    current = adapter.reset(seed)
    limit = int(max_steps or adapter.env.config["max_decision_steps"])
    team_return = 0.0
    agent_sum_return = 0.0
    step_count = 0
    last_info: dict[str, Any] = current.info
    cumulative_components = zero_reward_components()
    last_step_components = zero_reward_components()
    try:
        while step_count < limit:
            actions = sample_actions(rng, current.available_action_mask, policy)
            current = adapter.step(actions)
            step_count += 1
            team_return += float(current.team_reward)
            agent_sum_return += float(current.agent_reward_sum)
            last_info = current.info
            components = reward_component_sums(current.info)
            last_step_components = components
            add_components(cumulative_components, components)
            if step_log_interval > 0 and step_count % step_log_interval == 0:
                print(
                    f"[step {step_count:04d}] "
                    f"team_reward={current.team_reward:.4f} "
                    f"agent_reward_sum={current.agent_reward_sum:.4f} "
                    f"assigned_dense={components['assigned_dense']:.4f} "
                    f"combat_event={components['combat_event']:.4f} "
                    f"terminal={components['terminal']:.4f} "
                    f"actions={actions.tolist()}",
                    flush=True,
                )
            if current.terminated or current.truncated:
                break
    finally:
        adapter.env.close()
    outcome = last_info.get("outcome")
    stopped_by_max_steps = bool(max_steps is not None and step_count >= int(max_steps) and not (current.terminated or current.truncated))
    winner_value = getattr(outcome, "winner", None)
    termination_reason = "ongoing" if not (current.terminated or current.truncated) else str(getattr(outcome, "termination_reason", "ongoing"))
    summary = {
        "config": str(Path(config_path)),
        "algorithm": algorithm,
        "scenario": description.scenario,
        "opponent": description.opponent,
        "policy": policy,
        "seed": seed,
        "decision_steps": step_count,
        "stopped_by_max_steps": stopped_by_max_steps,
        "terminated": bool(current.terminated),
        "truncated": bool(current.truncated),
        "winner": None if winner_value is None else str(winner_value),
        "termination_reason": termination_reason,
        "team_return": team_return,
        "agent_sum_return": agent_sum_return,
        "mean_per_agent_return": agent_sum_return / max(float(adapter.num_agents), 1.0),
        "last_step_agent_rewards": current.agent_rewards.astype(float).tolist(),
        "reward_component_sums": cumulative_components,
        "last_step_reward_component_sums": last_step_components,
        "side_statistics": side_statistics(last_info, adapter.num_agents),
        "final_survivors": {
            "red": float(getattr(outcome, "red_survivors", 0.0) or 0.0),
            "blue": float(getattr(outcome, "blue_survivors", 0.0) or 0.0),
        },
    }
    assert_finite_numbers(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one configured UAV environment episode; no model, no training, no checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--algorithm", choices=["mappo", "happo"], default="mappo")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--policy", choices=["hold", "random"], default="hold")
    parser.add_argument("--max-steps", type=positive_int)
    parser.add_argument("--step-log-interval", type=nonnegative_int, default=1)
    args = parser.parse_args()
    summary = run_once(args.config, args.algorithm, args.seed, args.policy, args.max_steps, args.step_log_interval)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
