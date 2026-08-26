"""Pure-environment stress audit for persistent-wave replacement geometry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from uav_combat.config import load_config
from uav_combat.environment.geometry import engagement_geometry
from uav_combat.environment.persistent_env import PersistentWaveCombatEnv
from uav_combat.models import AircraftState


ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ("center", "edge", "dispersed", "altitude_heading")


def percentile_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("mean", "p10", "p50", "p90", "p99", "max")}
    data = np.asarray(values, dtype=float)
    return {
        "mean": float(data.mean()),
        "p10": float(np.percentile(data, 10)),
        "p50": float(np.percentile(data, 50)),
        "p90": float(np.percentile(data, 90)),
        "p99": float(np.percentile(data, 99)),
        "max": float(data.max()),
    }


def red_layout(category: str, survivors: int, rng: np.random.Generator) -> list[AircraftState]:
    if category == "center":
        xy = [(-450, -250), (-150, 250), (150, -250), (450, 250)]
    elif category == "edge":
        bearing = float(rng.uniform(-np.pi, np.pi))
        radial = np.array([np.cos(bearing), np.sin(bearing)])
        lateral = np.array([-radial[1], radial[0]])
        xy = [tuple(4100.0 * radial + offset * lateral) for offset in (-300, -100, 100, 300)]
    elif category == "dispersed":
        rotation = float(rng.uniform(-np.pi, np.pi))
        xy = [(3000.0 * np.cos(rotation + k * np.pi / 2),
               3000.0 * np.sin(rotation + k * np.pi / 2)) for k in range(4)]
    else:
        xy = [(-2200, -1500), (-800, 2100), (1800, -1700), (2400, 1200)]
    states = []
    for index, (x, y) in enumerate(xy):
        altitude = 1800.0 + 700.0 * index if category == "altitude_heading" else 3000.0 + rng.uniform(-300, 300)
        states.append(AircraftState(
            x=float(x), y=float(y), z=-float(altitude),
            v=float(rng.uniform(175, 275)), theta=float(rng.uniform(-0.25, 0.25)),
            psi=float(rng.uniform(-np.pi, np.pi)), alive=index < survivors,
        ))
    return states


def run_stress(config: dict[str, Any], cases: int) -> dict[str, Any]:
    env = PersistentWaveCombatEnv(config)
    attempts: list[float] = []
    minimum_distances: list[float] = []
    first_red_window_steps: list[float] = []
    first_blue_window_steps: list[float] = []
    immediate_red_pairs = immediate_blue_pairs = failures = 0
    failure_examples: list[dict[str, Any]] = []
    strata: dict[str, dict[str, int]] = {}

    for case_index in range(cases):
        category = CATEGORIES[case_index % len(CATEGORIES)]
        survivors = (case_index // len(CATEGORIES)) % 4 + 1
        key = f"{category}/red_{survivors}"
        stratum = strata.setdefault(key, {"cases": 0, "successes": 0, "failures": 0})
        stratum["cases"] += 1
        seed = 70_000_000 + case_index
        env.reset(seed)
        env.red = red_layout(category, survivors, env.rng)
        try:
            env._spawn_next_wave()
        except RuntimeError as error:
            failures += 1
            stratum["failures"] += 1
            if len(failure_examples) < 20:
                failure_examples.append({
                    "case": case_index, "seed": seed, "category": category,
                    "red_survivors": survivors, "error": str(error),
                })
            continue
        stratum["successes"] += 1
        attempts.append(float(env.last_spawn_attempts))
        alive_red = [state for state in env.red if state.alive]
        pair_distances = [
            engagement_geometry(red, blue).distance
            for red in alive_red for blue in env.blue
        ]
        minimum_distances.append(float(min(pair_distances)))
        red_pairs = env._window_pair_count(alive_red, env.blue)
        blue_pairs = env._window_pair_count(env.blue, alive_red)
        immediate_red_pairs += red_pairs
        immediate_blue_pairs += blue_pairs

        red_first = blue_first = None
        zero_actions = np.zeros((4, 3), dtype=np.float32)
        for step in range(1, 11):
            _, _, terminated, truncated, info = env.step(zero_actions)
            if red_first is None and info["red_fire_window_pairs"]:
                red_first = step
            if blue_first is None and info["blue_fire_window_pairs"]:
                blue_first = step
            if terminated or truncated:
                break
        if red_first is not None:
            first_red_window_steps.append(float(red_first))
        if blue_first is not None:
            first_blue_window_steps.append(float(blue_first))

    successes = cases - failures
    result = {
        "environment_variant": env.environment_variant,
        "cases": cases,
        "successes": successes,
        "failures": failures,
        "success_rate": successes / cases,
        "spawn_attempts": percentile_summary(attempts),
        "minimum_3d_red_blue_distance_m": percentile_summary(minimum_distances),
        "immediate_red_fire_window_pairs": immediate_red_pairs,
        "immediate_blue_fire_window_pairs": immediate_blue_pairs,
        "first_red_fire_window_within_1s_rate": len(first_red_window_steps) / max(successes, 1),
        "first_blue_fire_window_within_1s_rate": len(first_blue_window_steps) / max(successes, 1),
        "first_red_fire_window_step": percentile_summary(first_red_window_steps),
        "first_blue_fire_window_step": percentile_summary(first_blue_window_steps),
        "strata": strata,
        "failure_examples": failure_examples,
    }
    result["passed"] = bool(
        failures == 0
        and immediate_red_pairs == 0
        and immediate_blue_pairs == 0
        and (not minimum_distances or min(minimum_distances) >= env.min_red_distance)
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=10_000)
    parser.add_argument(
        "--config", default="configs/persistent_wave_environment.yaml"
    )
    parser.add_argument(
        "--output", default="outputs/persistent_wave_spawn_stress.json"
    )
    args = parser.parse_args()
    if args.cases <= 0:
        raise ValueError("cases must be positive")
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    result = run_stress(load_config(config_path), args.cases)
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
