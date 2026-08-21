"""V2 unit-level invariants and two deterministic combat-chain baselines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from uav_combat.environment.env import MultiUAVCombatEnv
from uav_combat.environment.geometry import engagement_geometry
from uav_combat.models import AircraftState


def state(x, y, psi, speed=225.0, altitude=3000.0, alive=True):
    return AircraftState(x, y, -altitude, speed, 0.0, psi, alive)


def one_alive(primary):
    dead = [state(0, 0, 0, alive=False) for _ in range(3)]
    return [primary, *dead]


def straight_head_on(config: Path) -> dict:
    env = MultiUAVCombatEnv(config)
    env.reset(1001)
    env.red = one_alive(state(-3000, 0, 0))
    env.blue = one_alive(state(3000, 0, np.pi))
    env.scenario_mode = "scripted_head_on"
    initial_attackable = env._in_fire_window(env.red[0], env.blue[0])
    first_kill = None
    minimum_distance = float("inf")
    for _ in range(40):
        _, _, terminated, truncated, info = env.step(
            np.zeros((4, 3), dtype=np.float32),
            np.zeros((4, 3), dtype=np.float32),
        )
        minimum_distance = min(
            minimum_distance,
            engagement_geometry(env.red[0], env.blue[0]).distance,
        )
        if info["red_first_kill_step"] is not None or info["blue_first_kill_step"] is not None:
            first_kill = env.steps
            break
        if terminated or truncated:
            break
    return {
        "initial_attackable": initial_attackable,
        "steps": env.steps,
        "minimum_distance": minimum_distance,
        "first_kill_step": first_kill,
        "passed": not initial_attackable and first_kill is None,
    }


def maneuver_combat(config: Path) -> dict:
    env = MultiUAVCombatEnv(config)
    env.reset(2001)
    env.red = one_alive(state(0, -500, 0, speed=250.0))
    env.blue = one_alive(state(4000, 0, 0, speed=200.0))
    env.scenario_mode = "scripted_tail_offset"
    initial_distance = engagement_geometry(env.red[0], env.blue[0]).distance
    maximum_heading_change = 0.0
    previous_heading = env.red[0].psi
    approached = False
    fire_opportunity = False
    completed_lock = False
    killed = False
    reward_components = {key: [] for key in ("progress", "tactical", "fire", "event")}
    for _ in range(env.max_steps):
        red_actions = env.fixed_policy.team_actions(env.red, env.blue)
        blue_actions = np.zeros((4, 3), dtype=np.float32)
        blue_actions[0, 2] = -1.0
        _, _, terminated, truncated, info = env.step(red_actions, blue_actions)
        heading_change = abs(((env.red[0].psi - previous_heading + np.pi) % (2*np.pi)) - np.pi)
        maximum_heading_change = max(maximum_heading_change, heading_change)
        previous_heading = env.red[0].psi
        if env.red[0].alive and env.blue[0].alive:
            approached |= engagement_geometry(env.red[0], env.blue[0]).distance < initial_distance - 500.0
        fire_opportunity |= info["red_first_attackable_step"] is not None
        completed_lock |= info["red_first_lock_step"] is not None
        killed |= info["red_attack_kills"] > 0
        reward_components["progress"].extend(map(float, info["progress_rewards"]))
        reward_components["tactical"].extend(map(float, info["tactical_rewards"]))
        reward_components["fire"].extend(map(float, info["fire_opportunity_rewards"]))
        reward_components["event"].extend(map(float, info["event_rewards"]))
        if terminated or truncated:
            break
    maneuvered = maximum_heading_change > 1e-3
    finite = all(np.all(np.isfinite(values)) for values in reward_components.values())
    return {
        "initial_distance": initial_distance,
        "steps": env.steps,
        "approached": approached,
        "maneuvered": maneuvered,
        "maximum_step_heading_change": maximum_heading_change,
        "attack_opportunity": fire_opportunity,
        "completed_lock": completed_lock,
        "kill": killed,
        "red_first_attackable_step": env.red_first_attackable_step,
        "red_first_lock_step": env.red_first_lock_step,
        "red_first_kill_step": env.red_first_kill_step,
        "reward_finite": finite,
        "reward_component_mean": {
            key: float(np.mean(value)) for key, value in reward_components.items()
        },
        "passed": all((approached, maneuvered, fire_opportunity, completed_lock, killed, finite)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/combat_environment_validation_v2.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = root / "configs/combat_environment.yaml"
    result = {
        "environment_version": "2.0",
        "straight_head_on": straight_head_on(config),
        "maneuver_combat": maneuver_combat(config),
    }
    result["passed"] = all(row["passed"] for key, row in result.items() if isinstance(row, dict))
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
