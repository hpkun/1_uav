"""Diagnose fixed 3v3 V2 reward ordering with paired rule-policy episodes."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import torch

from uav_env.algorithms.mappo.checkpoint import load_checkpoint
from uav_env.algorithms.mappo.networks import SharedActor
from uav_env.envs import make_3v3_env
from uav_env.envs.combat_multi_env import CombatMultiEnv
from uav_env.opponents.pursuit import PursuitOpponent
from uav_env.opponents.random import RandomOpponent
from uav_env.opponents.straight import StraightOpponent
from uav_env.opponents.team_controller import TeamRuleController


POLICIES = ("pursuit", "straight", "random")
PAIR_METRICS = (
    "team_episode_return",
    "red_attack_attempts",
    "red_hits",
    "red_effective_damage",
    "red_survivors",
    "survivor_difference",
    "timeout",
)
REWARD_COMPONENTS = (
    "situation_reward",
    "geometry_event_reward",
    "raw_shape_reward",
    "assigned_shape_reward",
    "combat_event_reward",
    "dense_reward",
    "terminal_reward",
    "hit_event_reward",
    "destroy_event_reward",
    "attacked_event_penalty",
    "destroyed_event_penalty",
    "boundary_collision_penalty",
)
BREAKDOWN_FIELDS = {
    "situation_reward": "situation",
    "geometry_event_reward": "geometry_event",
    "raw_shape_reward": "raw_shape",
    "assigned_shape_reward": "assigned_shape",
    "combat_event_reward": "combat_event",
    "dense_reward": "dense_reward",
    "terminal_reward": "terminal",
    "hit_event_reward": "hit_event_reward",
    "destroy_event_reward": "destroy_event_reward",
    "attacked_event_penalty": "attacked_event_penalty",
    "destroyed_event_penalty": "destroyed_event_penalty",
    "boundary_collision_penalty": "boundary_collision_penalty",
}


def finite_float(value: Any) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"Non-finite diagnostic value: {value!r}")
    return result


def make_policy(name: str, env: CombatMultiEnv) -> Any:
    if name == "pursuit":
        pursuit = {key: float(value) for key, value in env.config["pursuit"].items()}
        return PursuitOpponent(
            env.profile,
            env.attack_config,
            float(env.config["physics_dt"]),
            int(env.config["physics_steps_per_action"]),
            float(env.config["gravity"]),
            float(env.config["max_altitude"]),
            **pursuit,
        )
    if name == "straight":
        return StraightOpponent()
    if name == "random":
        return RandomOpponent()
    raise ValueError(name)


def build_actor(checkpoint_path: Path, device: str) -> SharedActor:
    data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = data.get("config", {})
    metadata = data.get("schema_metadata", {})
    obs_dim = int(metadata.get("obs_dim") or 63)
    hidden = config.get("actor_hidden_sizes", [128, 128])
    activation = str(config.get("activation", "relu"))
    actor = SharedActor(obs_dim, 15, hidden, activation).to(device)
    load_checkpoint(checkpoint_path, actor, actor_only=True, map_location=device)
    actor.eval()
    return actor


def run_episode(policy_label: str, seed: int, checkpoint_path: str | None = None, actor: SharedActor | None = None, device: str = "cpu") -> dict[str, Any]:
    env = make_3v3_env("head_on_mirrored_jitter_v2", "pursuit", seed=seed, multi_terminal_reward_profile="paper_2024_exact")
    observation, info = env.reset(seed=seed)
    controller = None
    if actor is None:
        policy = make_policy(policy_label, env)
        controller = TeamRuleController(policy_label, policy, seed + {"pursuit": 1_000_003, "straight": 1_100_003, "random": 1_200_003}[policy_label])

    terminated = truncated = False
    team_return = 0.0
    agent_sum_return = 0.0
    component_totals = {name: 0.0 for name in REWARD_COMPONENTS}
    while not (terminated or truncated):
        if actor is None:
            actions, _ = controller.select_actions(env.red_aircraft, env.blue_aircraft)
            action_values = np.asarray([int(action) for action in actions], dtype=np.int64)
        else:
            mask = torch.as_tensor(info["available_action_mask"], device=device)
            obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
            with torch.no_grad():
                action_values = torch.argmax(actor(obs_tensor, mask), dim=-1).cpu().numpy().astype(np.int64)
        observation, reward, terminated, truncated, info = env.step(action_values)
        team_return += float(reward)
        agent_rewards = info.get("agent_rewards", {})
        agent_sum_return += sum(float(value) for value in agent_rewards.values())
        for breakdown in info.get("agent_reward_breakdowns", {}).values():
            for component, field in BREAKDOWN_FIELDS.items():
                component_totals[component] += float(getattr(breakdown, field))

    outcome = info["outcome"]
    aircraft = info["statistics"]["aircraft"]
    red_ids = [f"red_{index}" for index in range(3)]
    blue_ids = [f"blue_{index}" for index in range(3)]

    def side_total(team: list[str], field: str) -> float:
        return finite_float(sum(float(aircraft[key].get(field, 0.0)) for key in team))

    row = {
        "policy_label": policy_label,
        "checkpoint_path": checkpoint_path or "",
        "seed": seed,
        "winner": str(outcome.winner or "none"),
        "termination_reason": str(outcome.termination_reason),
        "decision_steps": int(outcome.decision_steps),
        "simulation_time": finite_float(outcome.simulation_time),
        "red_survivors": int(outcome.red_survivors or 0),
        "blue_survivors": int(outcome.blue_survivors or 0),
        "survivor_difference": int(outcome.red_survivors or 0) - int(outcome.blue_survivors or 0),
        "team_episode_return": finite_float(team_return),
        "agent_sum_episode_return": finite_float(agent_sum_return),
        "mean_per_agent_episode_return": finite_float(agent_sum_return / 3.0),
        "red_attack_attempts": side_total(red_ids, "attack_attempts"),
        "blue_attack_attempts": side_total(blue_ids, "attack_attempts"),
        "red_hits": side_total(red_ids, "hits"),
        "blue_hits": side_total(blue_ids, "hits"),
        "red_nominal_damage": side_total(red_ids, "nominal_damage"),
        "blue_nominal_damage": side_total(blue_ids, "nominal_damage"),
        "red_effective_damage": side_total(red_ids, "effective_damage"),
        "blue_effective_damage": side_total(blue_ids, "effective_damage"),
        "red_overkill_damage": side_total(red_ids, "overkill_damage"),
        "blue_overkill_damage": side_total(blue_ids, "overkill_damage"),
        "red_attack_area_steps": side_total(red_ids, "attack_area_steps"),
        "blue_attack_area_steps": side_total(blue_ids, "attack_area_steps"),
        "red_ground_crashes": side_total(red_ids, "ground_crashes"),
        "blue_ground_crashes": side_total(blue_ids, "ground_crashes"),
        "red_collisions": side_total(red_ids, "collisions"),
        "blue_collisions": side_total(blue_ids, "collisions"),
        "timeout": int(outcome.termination_reason == "timeout"),
        "red_elimination_win": int(outcome.winner == "red" and outcome.termination_reason == "blue_eliminated"),
        "red_timeout_survival_win": int(outcome.winner == "red" and outcome.termination_reason == "timeout"),
        "draw": int(outcome.winner == "draw"),
    }
    for component, total in component_totals.items():
        row[f"{component}_team_total"] = finite_float(total)
        row[f"{component}_per_agent"] = finite_float(total / 3.0)
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_policy[str(row["policy_label"])].append(row)
    summary: dict[str, Any] = {"policies": {}, "paired_differences": {}, "diagnosis": {}}
    for policy, policy_rows in sorted(by_policy.items()):
        values = lambda key: [float(row[key]) for row in policy_rows]
        entry = {
            "episode_count": len(policy_rows),
            "mean_team_episode_return": float(np.mean(values("team_episode_return"))),
            "std_team_episode_return": float(np.std(values("team_episode_return"), ddof=1)) if len(policy_rows) > 1 else 0.0,
            "median_team_episode_return": float(median(values("team_episode_return"))),
            "mean_per_agent_episode_return": float(np.mean(values("mean_per_agent_episode_return"))),
            "overall_red_win_rate": float(np.mean([row["winner"] == "red" for row in policy_rows])),
            "elimination_win_rate": float(np.mean(values("red_elimination_win"))),
            "timeout_survival_win_rate": float(np.mean(values("red_timeout_survival_win"))),
            "draw_rate": float(np.mean(values("draw"))),
            "timeout_rate": float(np.mean(values("timeout"))),
            "mean_episode_steps": float(np.mean(values("decision_steps"))),
            "mean_red_survivors": float(np.mean(values("red_survivors"))),
            "mean_blue_survivors": float(np.mean(values("blue_survivors"))),
            "mean_survivor_difference": float(np.mean(values("survivor_difference"))),
            "mean_red_attack_attempts": float(np.mean(values("red_attack_attempts"))),
            "mean_red_hits": float(np.mean(values("red_hits"))),
            "mean_red_effective_damage": float(np.mean(values("red_effective_damage"))),
            "mean_red_attack_area_steps": float(np.mean(values("red_attack_area_steps"))),
            "red_ground_crash_rate": float(np.mean([row["red_ground_crashes"] > 0 for row in policy_rows])),
            "red_collision_rate": float(np.mean([row["red_collisions"] > 0 for row in policy_rows])),
        }
        for component in REWARD_COMPONENTS:
            totals = values(f"{component}_team_total")
            entry[f"{component}_mean"] = float(np.mean(totals))
            entry[f"{component}_std"] = float(np.std(totals, ddof=1)) if len(totals) > 1 else 0.0
        summary["policies"][policy] = entry
    summary["paired_differences"] = paired_differences(rows)
    summary["diagnosis"] = classify(summary["policies"])
    return summary


def paired_differences(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_policy_seed = {(str(row["policy_label"]), int(row["seed"])): row for row in rows}
    pairs = [("pursuit", "straight"), ("pursuit", "random"), ("learned_actor", "pursuit"), ("learned_actor", "straight"), ("learned_actor", "random")]
    rng = np.random.default_rng(24681357)
    result: dict[str, Any] = {}
    for left, right in pairs:
        seeds = sorted({seed for policy, seed in by_policy_seed if policy == left} & {seed for policy, seed in by_policy_seed if policy == right})
        if not seeds:
            continue
        label = f"{left} - {right}"
        result[label] = {}
        for metric in PAIR_METRICS:
            diffs = np.asarray([float(by_policy_seed[(left, seed)][metric]) - float(by_policy_seed[(right, seed)][metric]) for seed in seeds], dtype=np.float64)
            boot = []
            for _ in range(1000):
                sample = rng.choice(diffs, size=len(diffs), replace=True)
                boot.append(float(np.mean(sample)))
            lo, hi = np.percentile(boot, [2.5, 97.5])
            result[label][metric] = {
                "mean_paired_difference": float(np.mean(diffs)),
                "standard_deviation": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0,
                "median": float(np.median(diffs)),
                "positive_count": int(np.sum(diffs > 0.0)),
                "negative_count": int(np.sum(diffs < 0.0)),
                "zero_count": int(np.sum(diffs == 0.0)),
                "bootstrap_95_ci": [float(lo), float(hi)],
            }
    return result


def classify(policies: dict[str, dict[str, float]]) -> dict[str, Any]:
    pursuit = policies.get("pursuit")
    straight = policies.get("straight")
    random = policies.get("random")
    learned = policies.get("learned_actor")
    if not pursuit or not straight or not random:
        return {"label": "insufficient_evidence", "evidence": ["Missing fixed policy summaries."]}
    evidence = [
        f"pursuit return={pursuit['mean_team_episode_return']:.3f}, attacks={pursuit['mean_red_attack_attempts']:.3f}, hits={pursuit['mean_red_hits']:.3f}, damage={pursuit['mean_red_effective_damage']:.3f}",
        f"straight return={straight['mean_team_episode_return']:.3f}, random return={random['mean_team_episode_return']:.3f}",
    ]
    if pursuit["mean_red_attack_attempts"] < 0.05 and pursuit["mean_red_hits"] < 0.05 and pursuit["mean_red_effective_damage"] < 1.0:
        return {"label": "environment_reachability_failure", "evidence": evidence}
    combat_better = (
        pursuit["mean_red_attack_attempts"] > max(straight["mean_red_attack_attempts"], random["mean_red_attack_attempts"]) + 0.1
        or pursuit["mean_red_hits"] > max(straight["mean_red_hits"], random["mean_red_hits"]) + 0.1
        or pursuit["mean_red_effective_damage"] > max(straight["mean_red_effective_damage"], random["mean_red_effective_damage"]) + 1.0
    )
    return_not_better = pursuit["mean_team_episode_return"] <= max(straight["mean_team_episode_return"], random["mean_team_episode_return"])
    if combat_better and return_not_better:
        return {"label": "reward_misalignment", "evidence": evidence}
    if learned:
        evidence.append(f"learned return={learned['mean_team_episode_return']:.3f}, attacks={learned['mean_red_attack_attempts']:.3f}, hits={learned['mean_red_hits']:.3f}, damage={learned['mean_red_effective_damage']:.3f}, win={learned['overall_red_win_rate']:.3f}")
        learned_signal = learned["mean_red_attack_attempts"] > 0.05 or learned["mean_red_hits"] > 0.05 or learned["mean_red_effective_damage"] > 1.0
        if learned_signal and learned["overall_red_win_rate"] <= max(straight["overall_red_win_rate"], random["overall_red_win_rate"]):
            return {"label": "partially_learned", "evidence": evidence}
        if learned_signal and (
            learned["mean_red_effective_damage"] > max(straight["mean_red_effective_damage"], random["mean_red_effective_damage"])
            or learned["overall_red_win_rate"] > max(straight["overall_red_win_rate"], random["overall_red_win_rate"])
        ):
            return {"label": "learnable_signal_confirmed", "evidence": evidence}
        if pursuit["mean_team_episode_return"] > max(straight["mean_team_episode_return"], random["mean_team_episode_return"]) and combat_better and not learned_signal:
            return {"label": "exploration_or_optimization_failure", "evidence": evidence}
    return {"label": "insufficient_evidence", "evidence": evidence + ["No learned checkpoint was provided, or fixed-policy ordering is ambiguous."]}


def write_outputs(rows: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "reward_ordering_episodes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "reward_ordering_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# 3v3 Reward Ordering Diagnosis", "", f"diagnosis: `{summary['diagnosis']['label']}`", ""]
    lines.extend(f"- {item}" for item in summary["diagnosis"]["evidence"])
    lines.append("")
    for policy, entry in summary["policies"].items():
        lines.append(f"## {policy}")
        lines.append(f"- episodes: {entry['episode_count']}")
        lines.append(f"- mean return: {entry['mean_team_episode_return']:.6f}")
        lines.append(f"- red win / elimination / timeout / draw: {entry['overall_red_win_rate']:.3f} / {entry['elimination_win_rate']:.3f} / {entry['timeout_rate']:.3f} / {entry['draw_rate']:.3f}")
        lines.append(f"- attacks / hits / effective damage: {entry['mean_red_attack_attempts']:.3f} / {entry['mean_red_hits']:.3f} / {entry['mean_red_effective_damage']:.3f}")
        lines.append("")
    (output_dir / "reward_ordering_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=300000)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reward_ordering"))
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    actors = [(Path(path), build_actor(Path(path), device)) for path in args.checkpoint]
    rows: list[dict[str, Any]] = []
    for offset in range(args.episodes):
        seed = args.seed_start + offset
        for policy in POLICIES:
            rows.append(run_episode(policy, seed, device=device))
        for index, (path, actor) in enumerate(actors):
            label = "learned_actor" if index == 0 else f"learned_actor_{index + 1}"
            rows.append(run_episode(label, seed, str(path), actor, device))
    summary = summarize(rows)
    write_outputs(rows, summary, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "diagnosis": summary["diagnosis"], "episodes": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
