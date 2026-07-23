"""Run and summarize one complete homogeneous 2v2 episode."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from uav_env.actions.discrete_15 import DiscreteAction15
from uav_env.envs import make_2v2_env
from uav_env.envs.combat_multi_env import CombatMultiEnv
from uav_env.opponents.pursuit import PursuitOpponent
from uav_env.opponents.random import RandomOpponent
from uav_env.opponents.straight import StraightOpponent
from uav_env.opponents.team_controller import TeamRuleController


@dataclass(frozen=True)
class MultiEpisodeSummary:
    winner: str
    termination_reason: str
    decision_steps: int
    red_survivors: int
    blue_survivors: int
    health: dict[str, float]
    red_effective_damage: dict[str, float]
    red_hits: dict[str, int]
    red_contribution: dict[str, float]
    red_cumulative_rewards: dict[str, float]
    team_cumulative_reward: float
    agent_sum_cumulative_reward: float
    red_ground_crashes: int
    blue_ground_crashes: int
    timeout: bool


def run_2v2_episode(scenario: str, opponent: str, seed: int, red_policy: str, max_steps: int | None = None, terminal_reward_profile: str | None = None) -> tuple[CombatMultiEnv, MultiEpisodeSummary]:
    """Run one deterministic-seed 2v2 rule-policy episode."""

    env = make_2v2_env(scenario, opponent, seed=seed, multi_terminal_reward_profile=terminal_reward_profile)
    env.reset(seed=seed)
    pursuit_cfg = {key: float(value) for key, value in env.config["pursuit"].items()}
    pursuit = PursuitOpponent(env.profile, env.attack_config, float(env.config["physics_dt"]), int(env.config["physics_steps_per_action"]), float(env.config["gravity"]), float(env.config["max_altitude"]), **pursuit_cfg)
    rule = {"straight": StraightOpponent(), "random": RandomOpponent(), "pursuit": pursuit}.get(red_policy)
    if rule is None:
        raise ValueError(f"Unknown red policy: {red_policy!r}")
    controller = TeamRuleController(red_policy, rule, seed + 1_000_003)
    team_total = 0.0
    terminated = truncated = False
    info: dict[str, object] = {}
    limit = max_steps or int(env.config["max_decision_steps"])
    while not (terminated or truncated) and env.decision_step < limit:
        selected, _ = controller.select_actions(env.red_aircraft, env.blue_aircraft)
        actions = [int(action) for action in selected]
        _, reward, terminated, truncated, info = env.step(np.asarray(actions, dtype=np.int64))
        team_total += reward
    outcome = info.get("outcome", env._outcome(False))
    stats = env.get_statistics()["aircraft"]
    summary = MultiEpisodeSummary(
        str(outcome.winner or "none"), str(outcome.termination_reason), env.decision_step,
        int(outcome.red_survivors or 0), int(outcome.blue_survivors or 0),
        {u.uav_id: u.state.health for u in env.all_aircraft},
        {u.uav_id: float(stats[u.uav_id]["effective_damage"]) for u in env.red_aircraft},
        {u.uav_id: int(stats[u.uav_id]["hits"]) for u in env.red_aircraft},
        {u.uav_id: float(stats[u.uav_id]["contribution_score"]) for u in env.red_aircraft},
        {u.uav_id: float(stats[u.uav_id]["cumulative_reward"]) for u in env.red_aircraft},
        team_total, sum(float(stats[u.uav_id]["cumulative_reward"]) for u in env.red_aircraft),
        sum(int(stats[u.uav_id]["ground_crashes"]) for u in env.red_aircraft),
        sum(int(stats[u.uav_id]["ground_crashes"]) for u in env.blue_aircraft),
        str(outcome.termination_reason) == "timeout",
    )
    return env, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["head_on_formation", "offset_formation", "balanced_random"], default="head_on_formation")
    parser.add_argument("--opponent", choices=["straight", "random", "pursuit"], default="straight")
    parser.add_argument("--red-policy", choices=["straight", "random", "pursuit"], default="pursuit")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()
    _, summary = run_2v2_episode(args.scenario, args.opponent, args.seed, args.red_policy, args.max_steps)
    print(f"Winner: {summary.winner}")
    print(f"Termination reason: {summary.termination_reason}")
    print(f"Decision steps: {summary.decision_steps}")
    print(f"Survivors: red={summary.red_survivors}, blue={summary.blue_survivors}")
    print(f"Remaining health: {summary.health}")
    print(f"Red effective damage: {summary.red_effective_damage}")
    print(f"Red hits: {summary.red_hits}")
    print(f"Red contribution: {summary.red_contribution}")
    print(f"Red cumulative rewards: {summary.red_cumulative_rewards}")
    print(f"Team cumulative reward: {summary.team_cumulative_reward:.6f}")
    print(f"Agent-sum cumulative reward: {summary.agent_sum_cumulative_reward:.6f}")
    print(f"Ground crashes: red={summary.red_ground_crashes}, blue={summary.blue_ground_crashes}")
    print(f"Timeout: {summary.timeout}")


if __name__ == "__main__":
    main()
