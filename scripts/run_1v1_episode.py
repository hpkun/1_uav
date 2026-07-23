"""Run and summarize one complete homogeneous 1v1 episode."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from uav_env.actions.discrete_15 import DiscreteAction15
from uav_env.envs import make_1v1_env
from uav_env.envs.combat_1v1_env import Combat1v1Env
from uav_env.opponents.pursuit import PursuitOpponent


@dataclass(frozen=True)
class EpisodeSummary:
    """Compact terminal statistics printed by experiment scripts."""

    outcome: str
    termination_reason: str
    decision_steps: int
    red_health: float
    blue_health: float
    red_hits: int
    blue_hits: int
    red_damage: float
    blue_damage: float
    cumulative_reward: float
    red_attack_area_steps: int
    blue_attack_area_steps: int
    red_ground_crash: bool
    blue_ground_crash: bool
    collision: bool
    timeout: bool


def run_episode(
    scenario: str,
    opponent: str,
    seed: int,
    red_policy: str,
    max_steps: int | None = None,
) -> tuple[Combat1v1Env, EpisodeSummary]:
    """Run one reproducible episode with a simple external red policy."""

    env = make_1v1_env(scenario=scenario, opponent=opponent, seed=seed)
    _, _ = env.reset(seed=seed)
    policy_rng = np.random.default_rng(seed + 1_000_003)
    pursuit = PursuitOpponent(
        env.profile,
        env.attack_config,
        float(env.config["physics_dt"]),
        int(env.config["physics_steps_per_action"]),
        float(env.config["gravity"]),
        float(env.config["max_altitude"]),
        **{key: float(value) for key, value in env.config["pursuit"].items()},
    )
    cumulative_reward = 0.0
    terminated = truncated = False
    info: dict[str, object] = {}
    limit = max_steps if max_steps is not None else int(env.config["max_decision_steps"])
    while not (terminated or truncated) and env.decision_step < limit:
        if red_policy == "straight":
            action = DiscreteAction15.LEVEL_HOLD
        elif red_policy == "random":
            action = DiscreteAction15(int(policy_rng.integers(0, 15)))
        elif red_policy == "pursuit":
            action = pursuit.select_action(env.red.state, env.blue.state)
        else:
            raise ValueError(f"Unknown red policy: {red_policy!r}")
        _, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += reward
    outcome = info.get("outcome")
    winner = getattr(outcome, "winner", None) or "none"
    reason = getattr(outcome, "termination_reason", "script_step_limit")
    statistics = info.get("statistics", env.get_statistics())
    summary = EpisodeSummary(
        outcome=str(winner),
        termination_reason=str(reason),
        decision_steps=env.decision_step,
        red_health=env.red.state.health,
        blue_health=env.blue.state.health,
        red_hits=int(statistics["red_hits"]),  # type: ignore[index]
        blue_hits=int(statistics["blue_hits"]),  # type: ignore[index]
        red_damage=float(statistics["red_effective_damage"]),  # type: ignore[index]
        blue_damage=float(statistics["blue_effective_damage"]),  # type: ignore[index]
        cumulative_reward=cumulative_reward,
        red_attack_area_steps=int(statistics["red_attack_area_steps"]),  # type: ignore[index]
        blue_attack_area_steps=int(statistics["blue_attack_area_steps"]),  # type: ignore[index]
        red_ground_crash="red_ground_crash" in str(reason),
        blue_ground_crash="blue_ground_crash" in str(reason),
        collision=str(reason) == "collision",
        timeout=str(reason) == "timeout",
    )
    return env, summary


def main() -> None:
    """Parse CLI arguments, run one episode, and print terminal statistics."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["tail_chase", "head_on", "balanced_random"], default="tail_chase")
    parser.add_argument("--opponent", choices=["straight", "random", "pursuit"], default="straight")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--red-policy", choices=["random", "straight", "pursuit"], default="pursuit")
    args = parser.parse_args()
    _, summary = run_episode(args.scenario, args.opponent, args.seed, args.red_policy, args.max_steps)
    print(f"Outcome: {summary.outcome}")
    print(f"Termination reason: {summary.termination_reason}")
    print(f"Decision steps: {summary.decision_steps}")
    print(f"Remaining health: red={summary.red_health:.1f}, blue={summary.blue_health:.1f}")
    print(f"Hits: red={summary.red_hits}, blue={summary.blue_hits}")
    print(f"Cumulative damage: red={summary.red_damage:.1f}, blue={summary.blue_damage:.1f}")
    print(f"Red cumulative reward: {summary.cumulative_reward:.6f}")


if __name__ == "__main__":
    main()
