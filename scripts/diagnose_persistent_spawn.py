"""Reproduce and diagnose the known bounded spawn-rejection case."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from uav_combat.config import load_config
from uav_combat.environment.persistent_env import PersistentWaveCombatEnv
from validate_persistent_wave_environment import CATEGORIES, red_layout


ROOT = Path(__file__).resolve().parents[1]
KNOWN_SEED = 70_003_167
ATTEMPT_BUDGETS = (256, 512, 1024, 2048, 4096)


def known_environment(max_attempts: int) -> PersistentWaveCombatEnv:
    config = load_config(ROOT / "configs/persistent_wave_environment.yaml")
    config["persistent_waves"]["max_spawn_attempts"] = max_attempts
    env = PersistentWaveCombatEnv(config)
    env.reset(KNOWN_SEED)
    env.red = red_layout("altitude_heading", 4, env.rng)
    return env


def stress_environment(case_index: int, max_attempts: int) -> PersistentWaveCombatEnv:
    config = load_config(ROOT / "configs/persistent_wave_environment.yaml")
    config["persistent_waves"]["max_spawn_attempts"] = max_attempts
    env = PersistentWaveCombatEnv(config)
    seed = 70_000_000 + case_index
    env.reset(seed)
    category = CATEGORIES[case_index % len(CATEGORIES)]
    survivors = (case_index // len(CATEGORIES)) % 4 + 1
    env.red = red_layout(category, survivors, env.rng)
    return env


def angular_scan(env: PersistentWaveCombatEnv) -> dict:
    resolution_degrees = 0.01
    angles = np.deg2rad(np.arange(-180.0, 180.0, resolution_degrees))
    valid_angles = []
    for angle in angles:
        if env._valid_blue_wave(env._candidate_blue_wave(float(angle))):
            valid_angles.append(float(np.rad2deg(angle)))
    return {
        "resolution_degrees": resolution_degrees,
        "angles_tested": len(angles),
        "valid_candidates": len(valid_angles),
        "valid_fraction": len(valid_angles) / len(angles),
        "first_valid_angles_degrees": valid_angles[:20],
    }


def main() -> None:
    attempts = {}
    for budget in ATTEMPT_BUDGETS:
        env = known_environment(budget)
        try:
            env._spawn_next_wave()
            attempts[str(budget)] = {
                "success": True, "candidate": env.last_spawn_attempts,
            }
        except RuntimeError:
            attempts[str(budget)] = {"success": False, "candidate": None}

    scan_env = known_environment(256)
    additional_cases = {}
    for case_index in (22, 23):
        env = stress_environment(case_index, 4096)
        try:
            env._spawn_next_wave()
            random_result = {"success": True, "candidate": env.last_spawn_attempts}
        except RuntimeError:
            random_result = {"success": False, "candidate": None}
        additional_cases[str(case_index)] = {
            "category": CATEGORIES[case_index % len(CATEGORIES)],
            "random_4096": random_result,
            "angular_scan": angular_scan(stress_environment(case_index, 4096)),
        }
    result = {
        "seed": KNOWN_SEED,
        "category": "altitude_heading",
        "red_survivors": 4,
        "attempt_budget_results": attempts,
        "angular_scan": angular_scan(scan_env),
        "additional_stress_cases": additional_cases,
    }
    output = ROOT / "outputs/persistent_wave_spawn_case_3167_diagnosis.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
