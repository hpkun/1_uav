"""Read-only diagnostics for paper-unspecified combat-environment assumptions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml

from uav_combat.environment.env import PaperUAVCombatEnv
from uav_combat.environment.geometry import PaperAirCombatGeometry, compute_paper_geometry
from uav_combat.environment.weapon import WeaponModel
from uav_combat.madsac.actor import SharedSquashedGaussianActor


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "paper_environment.yaml"
DEFAULT_ALGORITHM_CONFIG = ROOT / "configs" / "madsac.yaml"
DEFAULT_CHECKPOINT = ROOT / "outputs" / "madsac_8m_seed2023" / "run_seed_2023" / "checkpoint_8000016.pt"
DEFAULT_OUTPUT = ROOT / "outputs" / "environment_diagnosis" / "diagnosis.json"


def distribution(values: list[float] | np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {key: None for key in ("count", "mean", "std", "min", "p10", "p50", "p90", "max")}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "p10": float(np.quantile(array, 0.10)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(array.max()),
    }


def canonical_weapon(config: dict[str, Any]) -> WeaponModel:
    return WeaponModel(**(config["weapon"] | config["reproduction_assumptions"]["weapon"]))


def weapon_diagnosis(
    config: dict[str, Any], samples: int = 100_000, seed: int = 2023,
) -> dict[str, Any]:
    weapon = canonical_weapon(config)
    distances = [500.0, 1000.0, 2000.0, 3000.0, 3500.0, 4000.0]
    angles = [0.0, 5.0, 15.0, 25.0, 30.0]
    rows = []
    rng = np.random.default_rng(seed)
    for distance in distances:
        threshold = float(np.pi * np.exp(-distance / weapon.d_hit))
        for angle_degrees in angles:
            angle = float(np.deg2rad(angle_degrees))
            epsilon = rng.normal(size=samples)
            hit = (
                (np.abs(angle + weapon.c4 * epsilon) <= threshold)
                & (np.abs(angle + weapon.c5 * epsilon) <= threshold)
            )
            rows.append({
                "distance_m": distance,
                "ata_degrees": angle_degrees,
                "ha_degrees": angle_degrees,
                "threshold_radians": threshold,
                "threshold_degrees": float(np.rad2deg(threshold)),
                "monte_carlo_hit_probability": float(hit.mean()),
            })

    critical = PaperAirCombatGeometry(4000.0, 0.0, 0.0, 0.0, 0.0)
    method_rng = np.random.default_rng(seed)
    method_hits = sum(weapon.sample_hit(critical, method_rng) for _ in range(samples))
    critical_probability = method_hits / samples
    selected = {
        row["ata_degrees"]: row["monte_carlo_hit_probability"]
        for row in rows if row["distance_m"] == 4000.0
    }
    repeated = {
        f"ata_ha_{int(angle)}deg": {
            str(attempts): float(1.0 - (1.0 - probability) ** attempts)
            for attempts in (1, 2, 5, 10)
        }
        for angle, probability in selected.items()
    }
    return {
        "samples_per_geometry": samples,
        "rng_seed": seed,
        "rows": rows,
        "weapon_model_sample_hit_probability_at_4000m_0deg": critical_probability,
        "repeated_attempt_probability_at_4000m": repeated,
    }


def initial_geometry_diagnosis(
    config: dict[str, Any], reset_count: int = 1000, seed_base: int = 30_000_000,
) -> dict[str, Any]:
    metrics: dict[str, list[float]] = {
        key: [] for key in (
            "red_center_x_m", "red_center_y_m", "red_center_z_m", "red_center_radius_m",
            "blue_center_x_m", "blue_center_y_m", "blue_center_z_m", "blue_center_radius_m",
            "center_distance_m", "minimum_pair_distance_m", "maximum_pair_distance_m",
            "absolute_initial_ata_degrees", "absolute_initial_ha_degrees",
            "absolute_initial_aa_degrees", "closing_speed_mps",
        )
    }
    env = PaperUAVCombatEnv(config)
    for seed in range(seed_base, seed_base + reset_count):
        env.reset(seed)
        red_center = np.mean([state.as_array()[:3] for state in env.red], axis=0)
        blue_center = np.mean([state.as_array()[:3] for state in env.blue], axis=0)
        metrics["red_center_x_m"].append(red_center[0])
        metrics["red_center_y_m"].append(red_center[1])
        metrics["red_center_z_m"].append(red_center[2])
        metrics["red_center_radius_m"].append(float(np.hypot(*red_center[:2])))
        metrics["blue_center_x_m"].append(blue_center[0])
        metrics["blue_center_y_m"].append(blue_center[1])
        metrics["blue_center_z_m"].append(blue_center[2])
        metrics["blue_center_radius_m"].append(float(np.hypot(*blue_center[:2])))
        metrics["center_distance_m"].append(float(np.linalg.norm(blue_center - red_center)))
        pair_distances = []
        for red in env.red:
            for blue in env.blue:
                geometry = compute_paper_geometry(red, blue)
                pair_distances.append(geometry.distance)
                metrics["absolute_initial_ata_degrees"].append(abs(float(np.rad2deg(geometry.ata))))
                metrics["absolute_initial_ha_degrees"].append(abs(float(np.rad2deg(geometry.ha))))
                metrics["absolute_initial_aa_degrees"].append(abs(float(np.rad2deg(geometry.aa))))
                relative_position = blue.as_array()[:3] - red.as_array()[:3]
                relative_velocity = blue.velocity_vector() - red.velocity_vector()
                closing = -float(np.dot(relative_position, relative_velocity)) / geometry.distance
                metrics["closing_speed_mps"].append(closing)
        metrics["minimum_pair_distance_m"].append(min(pair_distances))
        metrics["maximum_pair_distance_m"].append(max(pair_distances))

    formation = config["reproduction_assumptions"]["formation"]
    distance_max = float(config["weapon"]["distance_max"])
    head_on_closing = 2.0 * float(formation["speed"])
    time_to_envelope = (2.0 * float(formation["center_distance"]) - distance_max) / head_on_closing
    return {
        "reset_count": reset_count,
        "seed_start": seed_base,
        "seed_end": seed_base + reset_count - 1,
        "statistics": {key: distribution(values) for key, values in metrics.items()},
        "head_on_theory": {
            "initial_center_distance_m": 2.0 * float(formation["center_distance"]),
            "closing_speed_mps": head_on_closing,
            "time_to_4000m_seconds": time_to_envelope,
            "time_to_4000m_steps": time_to_envelope / float(config["simulation"]["dt"]),
        },
    }


class DeterministicCheckpointActor:
    def __init__(self, checkpoint: Path, algorithm_config: dict[str, Any], device: str = "cpu") -> None:
        network = algorithm_config["network"]
        assumptions = algorithm_config["reproduction_assumptions"]
        self.device = torch.device(device)
        self.actor = SharedSquashedGaussianActor(
            observation_dim=int(network["observation_dim"]),
            action_dim=int(network["action_dim"]),
            hidden_dim=int(network["actor_hidden_layers"][0]),
            log_std_min=float(assumptions["log_std_min"]),
            log_std_max=float(assumptions["log_std_max"]),
            activation=str(assumptions["actor_activation"]),
        ).to(self.device)
        state = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(state["actor"])
        self.actor.eval()

    @torch.no_grad()
    def act(self, observations: np.ndarray, alive_mask: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        actions = self.actor.deterministic(tensor)
        actions *= torch.as_tensor(alive_mask, dtype=torch.float32, device=self.device).unsqueeze(-1)
        return actions.cpu().numpy()


def _first(values: list[dict[str, Any]], team: str) -> dict[str, Any] | None:
    return next((row for row in values if row["team"] == team), None)


def run_engagements(
    config: dict[str, Any], mode: str, episode_count: int, seed_base: int,
    actor: DeterministicCheckpointActor | None = None,
) -> dict[str, Any]:
    records = []
    dt = float(config["simulation"]["dt"])
    for seed in range(seed_base, seed_base + episode_count):
        events: list[tuple[str, dict[str, Any]]] = []
        env = PaperUAVCombatEnv(config, diagnostic_observer=lambda event, payload: events.append((event, payload)))
        observation, _ = env.reset(seed)
        red_center = np.mean([state.as_array()[:3] for state in env.red], axis=0)
        blue_center = np.mean([state.as_array()[:3] for state in env.blue], axis=0)
        initial_distance = float(np.linalg.norm(blue_center - red_center))
        previous_red = int(env.red_alive_mask.sum())
        previous_blue = int(env.blue_alive_mask.sum())
        deaths_by_step: dict[int, dict[str, int]] = {}
        attempts: list[dict[str, Any]] = []
        resolutions: list[dict[str, Any]] = []
        first_casualty_step: int | None = None
        while True:
            actions = (
                np.zeros((4, 3), dtype=np.float32)
                if mode == "straight"
                else actor.act(observation, env.red_alive_mask)
            )
            before = len(events)
            observation, _, terminated, truncated, info = env.step(actions)
            for event, payload in events[before:]:
                if event == "weapon_attempt":
                    attempts.append(payload)
                elif event == "hit_resolution":
                    resolutions.append(payload)
            current_red = int(env.red_alive_mask.sum())
            current_blue = int(env.blue_alive_mask.sum())
            red_deaths = previous_red - current_red
            blue_deaths = previous_blue - current_blue
            if red_deaths or blue_deaths:
                deaths_by_step[env.steps] = {"red": red_deaths, "blue": blue_deaths}
                if first_casualty_step is None:
                    first_casualty_step = env.steps
            previous_red, previous_blue = current_red, current_blue
            if terminated or truncated:
                break

        first_red = _first(attempts, "red")
        first_blue = _first(attempts, "blue")
        fire_steps = [row["step"] for row in (first_red, first_blue) if row is not None]
        first_fire_step = min(fire_steps) if fire_steps else None
        first_success = resolutions[0] if resolutions else None
        first_step_attempts = [row for row in attempts if row["step"] == first_fire_step]

        def casualties_within(team: str, seconds: float) -> int | None:
            if first_fire_step is None:
                return None
            final_step = first_fire_step + int(np.ceil(seconds / dt)) - 1
            return sum(row[team] for step, row in deaths_by_step.items() if first_fire_step <= step <= final_step)

        def fire_geometry(row: dict[str, Any] | None) -> dict[str, float | int] | None:
            if row is None:
                return None
            return {
                "step": row["step"],
                "distance_m": row["distance"],
                "ata_degrees": float(np.rad2deg(row["ata"])),
                "ha_degrees": float(np.rad2deg(row["ha"])),
            }

        records.append({
            "seed": seed,
            "initial_center_distance_m": initial_distance,
            "first_can_fire_step_red": None if first_red is None else first_red["step"],
            "first_can_fire_step_blue": None if first_blue is None else first_blue["step"],
            "first_fire_step": first_fire_step,
            "first_fire_red": fire_geometry(first_red),
            "first_fire_blue": fire_geometry(first_blue),
            "first_fire_attempts_red": sum(row["team"] == "red" for row in first_step_attempts),
            "first_fire_attempts_blue": sum(row["team"] == "blue" for row in first_step_attempts),
            "total_weapon_attempts_red": sum(row["team"] == "red" for row in attempts),
            "total_weapon_attempts_blue": sum(row["team"] == "blue" for row in attempts),
            "total_successful_proposals_red": sum(row["red_successful_proposals"] for row in resolutions),
            "total_successful_proposals_blue": sum(row["blue_successful_proposals"] for row in resolutions),
            "first_successful_hit_step": None if first_success is None else first_success["step"],
            "first_casualty_step": first_casualty_step,
            "first_successful_exchange": first_success,
            "red_casualties_within_0_5s": casualties_within("red", 0.5),
            "red_casualties_within_1_0s": casualties_within("red", 1.0),
            "red_casualties_within_2_0s": casualties_within("red", 2.0),
            "blue_casualties_within_0_5s": casualties_within("blue", 0.5),
            "blue_casualties_within_1_0s": casualties_within("blue", 1.0),
            "blue_casualties_within_2_0s": casualties_within("blue", 2.0),
            "episode_length": info["episode_length"],
            "episode_length_minus_first_fire": None if first_fire_step is None else info["episode_length"] - first_fire_step,
            "red_losses": info["red_losses"],
            "blue_losses": 4 - info["blue_survivors"],
            "red_success": info["red_success"],
            "termination_reason": info["termination_reason"],
        })

    def values(key: str) -> list[float]:
        return [float(row[key]) for row in records if row[key] is not None]

    first_exchange = [row["first_successful_exchange"] for row in records if row["first_successful_exchange"] is not None]
    exchanges: dict[str, int] = {}
    for row in first_exchange:
        label = f"{row['blue_actual_kills']}red_vs_{row['red_actual_kills']}blue_deaths"
        exchanges[label] = exchanges.get(label, 0) + 1
    termination: dict[str, int] = {}
    for row in records:
        termination[row["termination_reason"]] = termination.get(row["termination_reason"], 0) + 1

    summary_keys = (
        "initial_center_distance_m", "first_can_fire_step_red", "first_can_fire_step_blue",
        "first_fire_step", "first_successful_hit_step", "first_casualty_step",
        "episode_length", "episode_length_minus_first_fire", "red_losses", "blue_losses",
        "red_casualties_within_0_5s", "red_casualties_within_1_0s", "red_casualties_within_2_0s",
        "blue_casualties_within_0_5s", "blue_casualties_within_1_0s", "blue_casualties_within_2_0s",
        "first_fire_attempts_red", "first_fire_attempts_blue",
        "total_weapon_attempts_red", "total_weapon_attempts_blue",
        "total_successful_proposals_red", "total_successful_proposals_blue",
    )
    statistics = {key: distribution(values(key)) for key in summary_keys}
    nested_metrics: dict[str, list[float]] = {
        key: [] for key in (
            "first_fire_distance_red_m", "first_fire_distance_blue_m",
            "absolute_first_fire_ata_red_degrees", "absolute_first_fire_ata_blue_degrees",
            "absolute_first_fire_ha_red_degrees", "absolute_first_fire_ha_blue_degrees",
            "first_exchange_red_successful_proposals", "first_exchange_blue_successful_proposals",
            "first_exchange_red_actual_kills", "first_exchange_blue_actual_kills",
        )
    }
    for row in records:
        for team in ("red", "blue"):
            fire = row[f"first_fire_{team}"]
            if fire is not None:
                nested_metrics[f"first_fire_distance_{team}_m"].append(fire["distance_m"])
                nested_metrics[f"absolute_first_fire_ata_{team}_degrees"].append(abs(fire["ata_degrees"]))
                nested_metrics[f"absolute_first_fire_ha_{team}_degrees"].append(abs(fire["ha_degrees"]))
        exchange = row["first_successful_exchange"]
        if exchange is not None:
            for key in (
                "red_successful_proposals", "blue_successful_proposals",
                "red_actual_kills", "blue_actual_kills",
            ):
                nested_metrics[f"first_exchange_{key}"].append(exchange[key])
    statistics.update({key: distribution(metric) for key, metric in nested_metrics.items()})
    severe_exchange_rate = float(np.mean([
        row["first_successful_exchange"] is not None
        and row["first_successful_exchange"]["red_actual_kills"] >= 2
        and row["first_successful_exchange"]["blue_actual_kills"] >= 2
        for row in records
    ]))
    return {
        "mode": mode,
        "episode_count": episode_count,
        "seed_start": seed_base,
        "seed_end": seed_base + episode_count - 1,
        "statistics": statistics,
        "red_win_rate": float(np.mean([row["red_success"] for row in records])),
        "termination_reasons": termination,
        "first_successful_exchange_counts": exchanges,
        "first_exchange_at_least_2_deaths_each_side_rate": severe_exchange_rate,
        "episodes": records,
    }


def print_tables(result: dict[str, Any]) -> None:
    print("Weapon Eq.(8) Monte Carlo")
    print("distance_m angle_deg threshold_rad threshold_deg hit_probability")
    for row in result["weapon"]["rows"]:
        print(
            f"{row['distance_m']:10.0f} {row['ata_degrees']:9.0f} "
            f"{row['threshold_radians']:13.6f} {row['threshold_degrees']:13.3f} "
            f"{row['monte_carlo_hit_probability']:15.6f}"
        )
    print("\nInitial geometry")
    for key, stats in result["initial_geometry"]["statistics"].items():
        print(f"{key}: {stats}")
    for mode, report in result["engagements"].items():
        print(f"\nFirst engagement: {mode}")
        for key, stats in report["statistics"].items():
            print(f"{key}: {stats}")
        print(f"red_win_rate: {report['red_win_rate']:.6f}")
        print(f"termination_reasons: {report['termination_reasons']}")
        print(f"first_successful_exchange_counts: {report['first_successful_exchange_counts']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--algorithm-config", type=Path, default=DEFAULT_ALGORITHM_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reset-count", type=int, default=1000)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--mc-samples", type=int, default=100_000)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    algorithm_config = yaml.safe_load(args.algorithm_config.read_text(encoding="utf-8"))
    result: dict[str, Any] = {
        "canonical_config": str(args.config),
        "weapon": weapon_diagnosis(config, args.mc_samples),
        "initial_geometry": initial_geometry_diagnosis(config, args.reset_count),
        "engagements": {},
    }
    result["engagements"]["straight"] = run_engagements(
        config, "straight", args.episodes, 40_000_000,
    )
    if args.checkpoint.is_file():
        actor = DeterministicCheckpointActor(args.checkpoint, algorithm_config, args.device)
        result["checkpoint"] = str(args.checkpoint)
        result["engagements"]["madsac"] = run_engagements(
            config, "madsac", args.episodes, 41_000_000, actor,
        )
    else:
        result["checkpoint"] = None
        result["checkpoint_status"] = "not found; MADSAC mode skipped"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print_tables(result)
    print(f"\nWrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
