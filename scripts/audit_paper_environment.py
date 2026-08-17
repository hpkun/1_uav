"""Small paper audit: classifications, Eq.(6) truth table, and Eq.(8) probabilities."""
from __future__ import annotations

import argparse
from math import erf, pi, sqrt
from pathlib import Path
import numpy as np
import yaml

from uav_combat.environment.geometry import PaperAirCombatGeometry, compute_paper_geometry
from uav_combat.environment.env import PaperUAVCombatEnv
from uav_combat.environment.weapon import WeaponModel
from uav_combat.models import AircraftState


ROOT = Path(__file__).resolve().parents[1]


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def theoretical_hit_probability(geometry: PaperAirCombatGeometry, weapon: WeaponModel) -> float:
    """Exact probability for Eq.(8)'s one shared epsilon_fire."""
    threshold = pi * np.exp(-geometry.distance / weapon.d_hit)
    intervals = []
    for angle, scale in ((geometry.ata, weapon.c4), (geometry.ha, weapon.c5)):
        if scale == 0.0:
            if abs(angle) > threshold:
                return 0.0
            continue
        intervals.append(((-threshold - angle) / scale, (threshold - angle) / scale))
    if not intervals:
        return 1.0
    low = max(interval[0] for interval in intervals)
    high = min(interval[1] for interval in intervals)
    return max(0.0, normal_cdf(high) - normal_cdf(low))


def truth_table() -> list[tuple[str, float, float, float, float, float]]:
    def state(x, y, z=0.0, psi=0.0):
        return AircraftState(x, y, z, 200.0, 0.0, psi)
    cases = [
        ("A red behind blue", state(0, 0), state(100, 0)),
        ("B head-on", state(0, 0), state(100, 0, psi=pi)),
        ("C red at blue side", state(0, -100), state(0, 0)),
        ("D blue behind red", state(100, 0), state(0, 0)),
        ("E blue above red", state(0, 0), state(100, 0, z=-100)),
    ]
    rows = []
    for label, red, blue in cases:
        red_geometry = compute_paper_geometry(red, blue)
        blue_geometry = compute_paper_geometry(blue, red)
        rows.append((label, red_geometry.ata, red_geometry.aa, blue_geometry.ata, blue_geometry.aa, red_geometry.ha))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weapon-samples", type=int, default=10_000)
    parser.add_argument("--rule-episodes", type=int, default=0)
    args = parser.parse_args()
    env = yaml.safe_load((ROOT / "configs/paper_environment.yaml").read_text(encoding="utf-8"))
    alg = yaml.safe_load((ROOT / "configs/madsac.yaml").read_text(encoding="utf-8"))
    ea, ma, training = env["reproduction_assumptions"], alg["reproduction_assumptions"], alg["training"]
    rows = [
        ("PAPER", "dt", env["simulation"]["dt"]),
        ("PAPER", "4v4 / diameter", [env["scenario"]["red_count"], env["scenario"]["blue_count"], env["battlefield"]["diameter"]]),
        ("PAPER", "action ranges", env["action"]),
        ("PAPER", "weapon max/radar angles", {key: env["weapon"][key] for key in ("distance_max", "ata_max", "ha_max")}),
        ("PAPER", "actor/critic", [alg["network"]["actor_hidden_layers"], alg["network"]["attention_heads"], alg["network"]["critic_hidden_layers"]]),
        ("PAPER", "replay/batch/gamma/tau/alpha", [training["replay_buffer_size"], training["batch_size"], training["gamma"], training["tau"], training["alpha"]]),
        ("PAPER", "train/evaluation/run protocol", [training["num_train_envs"], training["evaluation_episodes"], training["independent_training_runs"], training["confidence_interval"]]),
        ("DERIVED", "observation dimension", 45),
        ("DERIVED", "battle radius", env["battlefield"]["diameter"] / 2),
        ("UNSPECIFIED", "sensor coefficients", ea["sensor"]),
        ("UNSPECIFIED", "weapon coefficients", ea["weapon"] | {"distance_min": env["weapon"]["distance_min"]}),
        ("UNSPECIFIED", "controller", ea["controller"]),
        ("UNSPECIFIED", "formation", ea["formation"]),
        ("UNSPECIFIED", "Algorithm 1 n/d/threshold", {key: ma[key] for key in ("steps_per_update", "update_steps_n", "policy_delay_d")}),
    ]
    for classification, name, value in rows:
        print(f"[{classification:11}] {name}: {value}")

    print("\nEq.(6) truth table, radians: ATA_r AA_r ATA_b AA_b HA_r")
    for row in truth_table():
        print(f"{row[0]:22} " + " ".join(f"{value:+.6f}" for value in row[1:]))

    weapon = WeaponModel(**(env["weapon"] | ea["weapon"]))
    rng = np.random.default_rng(2023)
    print("\nEq.(8) theoretical / Monte-Carlo hit probability")
    for distance in (500.0, 2000.0, 4000.0):
        for angle_degrees in (0.0, 15.0, 30.0):
            angle = np.deg2rad(angle_degrees)
            geometry = PaperAirCombatGeometry(distance, angle, 0.0, angle, 0.0)
            theoretical = theoretical_hit_probability(geometry, weapon)
            monte_carlo = float(np.mean([weapon.sample_hit(geometry, rng) for _ in range(args.weapon_samples)]))
            print(f"d={distance:4.0f} ATA=HA={angle_degrees:2.0f}deg: {theoretical:.6f} / {monte_carlo:.6f}")

    if args.rule_episodes:
        records = []
        for seed in range(args.rule_episodes):
            combat = PaperUAVCombatEnv(env)
            observation, _ = combat.reset(seed)
            episode_return = 0.0
            while True:
                red_actions = np.stack([combat.fixed_policy.action(state, combat.blue)[0] for state in combat.red])
                observation, rewards, terminated, truncated, info = combat.step(red_actions)
                episode_return += float(rewards[0])
                if terminated or truncated:
                    records.append({"return": episode_return, **info})
                    break
        print("\nRule-vs-rule smoke")
        print(f"episodes={len(records)} finite={all(np.isfinite(row['return']) for row in records)} ")
        print(f"win_rate={np.mean([row['red_success'] for row in records]):.6f} average_return={np.mean([row['return'] for row in records]):.6f} average_length={np.mean([row['episode_length'] for row in records]):.2f}")
        print("termination_reasons=" + str([row["termination_reason"] for row in records]))


if __name__ == "__main__":
    main()
