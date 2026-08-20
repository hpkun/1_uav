"""Monte Carlo and trajectory validation of the isolated paper attack prototype."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from uav_combat.diagnostics.paper_weapon_prototype import (
    C4, C5, D_HIT, EVIDENCE, EntryTriggeredAttempt, PaperWeaponGeometry,
    fire_gate, hit_samples, hit_threshold, paper_weapon_geometry,
)
from uav_combat.models import AircraftState


DT = 0.1
GRID_SAMPLES = 100_000
TRAJECTORY_EPISODES = 100_000
DISTANCES = (0.0, 500.0, 1000.0, 2000.0, 3000.0, 4000.0)
ANGLE_CASES = (
    (0, 0), (15, 0), (0, 15), (15, 15), (30, 0), (0, 30),
    (30, 30), (30, -30), (-30, 30), (-30, -30), (31, 0), (0, 31),
)


def probability_grid(samples: int = GRID_SAMPLES, seed: int = 202_308) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for distance in DISTANCES:
        for ata_deg, ha_deg in ANGLE_CASES:
            geometry = PaperWeaponGeometry(distance, np.deg2rad(ata_deg), np.deg2rad(ha_deg))
            gate = fire_gate(geometry)
            results = {}
            for semantics in ("shared", "independent"):
                hits = hit_samples(geometry, rng, samples, semantics) if gate else np.zeros(samples, dtype=bool)
                probability = float(np.mean(hits))
                results[f"hit_probability_{semantics}"] = probability
                results[f"standard_error_{semantics}"] = float(np.sqrt(probability * (1.0 - probability) / samples))
            rows.append({
                "distance_m": distance, "ata_deg": ata_deg, "ha_deg": ha_deg,
                "fire_gate": gate, "threshold_deg": float(np.degrees(hit_threshold(distance))),
                "samples": samples, **results,
            })
    return rows


def cadence_table() -> list[dict]:
    rows = []
    for probability in (0.05, 0.10, 0.16, 0.25, 0.50):
        for duration in (0.1, 0.5, 1.0, 2.0):
            attempts = int(round(duration / DT))
            rows.append({
                "single_attempt_probability": probability, "duration_s": duration,
                "dt": DT, "attempts": attempts,
                "per_step_cumulative_kill_probability": 1.0 - (1.0 - probability) ** attempts,
                "entry_triggered_attempts_one_window": 1,
                "entry_triggered_kill_probability_one_window": probability,
            })
    return rows


def state(x, y, psi, speed=225.0, z=-3000.0) -> AircraftState:
    return AircraftState(float(x), float(y), float(z), float(speed), 0.0, float(psi))


def head_on_pass() -> list[PaperWeaponGeometry]:
    result = []
    for step in range(201):
        time = step * DT
        own = state(-3000.0 + 225.0 * time, 0.0, 0.0)
        target = state(3000.0 - 225.0 * time, 0.0, np.pi)
        result.append(paper_weapon_geometry(own, target))
    return result


def tail_chase() -> list[PaperWeaponGeometry]:
    result = []
    for step in range(301):
        time = step * DT
        own = state(260.0 * time, 0.0, 0.0, 260.0)
        target = state(3000.0 + 225.0 * time, 0.0, 0.0)
        result.append(paper_weapon_geometry(own, target))
    return result


def horizontal_crossing() -> list[PaperWeaponGeometry]:
    result = []
    for step in range(201):
        time = step * DT
        own = state(225.0 * time, 0.0, 0.0)
        target = state(3000.0, -3000.0 + 225.0 * time, np.pi / 2.0)
        result.append(paper_weapon_geometry(own, target))
    return result


def pure_pursuit() -> list[PaperWeaponGeometry]:
    own_xy = np.array([-3000.0, 0.0])
    target_xy = np.array([3000.0, 1000.0])
    target_heading = np.pi
    result = []
    for _ in range(301):
        delta = target_xy - own_xy
        own_heading = float(np.arctan2(delta[1], delta[0])) if np.linalg.norm(delta) > 1e-12 else 0.0
        own = state(*own_xy, own_heading, speed=260.0)
        target = state(*target_xy, target_heading)
        result.append(paper_weapon_geometry(own, target))
        own_xy += DT * 260.0 * np.array([np.cos(own_heading), np.sin(own_heading)])
        target_xy += DT * 225.0 * np.array([np.cos(target_heading), np.sin(target_heading)])
    return result


def brief_crossing() -> list[PaperWeaponGeometry]:
    distances = np.concatenate([np.linspace(4500, 3500, 21), np.linspace(3550, 4500, 20)])
    return [PaperWeaponGeometry(float(distance), 0.0, 0.0) for distance in distances]


def long_dwell() -> list[PaperWeaponGeometry]:
    return [PaperWeaponGeometry(2000.0, 0.0, 0.0) for _ in range(101)]


TRAJECTORIES = {
    "head_on_pass": head_on_pass,
    "tail_chase": tail_chase,
    "horizontal_crossing": horizontal_crossing,
    "pure_pursuit": pure_pursuit,
    "brief_firing_envelope_crossing": brief_crossing,
    "long_firing_envelope_dwell": long_dwell,
}


def window_indices(gates: list[bool]) -> list[list[int]]:
    windows, current = [], []
    for index, gate in enumerate(gates):
        if gate:
            current.append(index)
        elif current:
            windows.append(current); current = []
    if current:
        windows.append(current)
    return windows


def attempt_indices(gates: list[bool], cadence: str) -> list[int]:
    if cadence == "per_step":
        return [index for index, gate in enumerate(gates) if gate]
    trigger, result = EntryTriggeredAttempt(), []
    for index, gate in enumerate(gates):
        if trigger.update(gate):
            result.append(index)
    return result


def episode_monte_carlo(
    geometries: list[PaperWeaponGeometry], indices: list[int], semantics: str,
    episodes: int, seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    alive = np.ones(episodes, dtype=bool)
    for index in indices:
        active = np.flatnonzero(alive)
        if active.size == 0:
            break
        hits = hit_samples(geometries[index], rng, active.size, semantics)
        alive[active[hits]] = False
    probability = float(np.mean(~alive))
    standard_error = float(np.sqrt(probability * (1.0 - probability) / episodes))
    return probability, standard_error


def trajectory_results(episodes: int = TRAJECTORY_EPISODES) -> list[dict]:
    rows, seed = [], 990_000
    for name, builder in TRAJECTORIES.items():
        geometries = builder()
        gates = [fire_gate(geometry) for geometry in geometries]
        windows = window_indices(gates)
        for cadence in ("per_step", "entry_triggered"):
            indices = attempt_indices(gates, cadence)
            for semantics in ("shared", "independent"):
                probability, standard_error = episode_monte_carlo(
                    geometries, indices, semantics, episodes, seed
                )
                seed += 1
                rows.append({
                    "trajectory": name, "noise_semantics": semantics, "cadence": cadence,
                    "trajectory_steps": len(geometries), "fire_gate_entry_count": len(windows),
                    "continuous_firing_windows": len(windows),
                    "total_fire_gate_steps": sum(gates), "total_fire_gate_duration_s": sum(gates) * DT,
                    "longest_window_s": max((len(window) * DT for window in windows), default=0.0),
                    "attempt_count": len(indices), "monte_carlo_episodes": episodes,
                    "episode_kill_probability": probability, "standard_error": standard_error,
                    "first_attempt_distance_m": geometries[indices[0]].distance if indices else None,
                    "first_attempt_ata_deg": np.degrees(geometries[indices[0]].ata) if indices else None,
                    "first_attempt_ha_deg": np.degrees(geometries[indices[0]].ha) if indices else None,
                })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/paper_environment_prototypes/weapon")
    parser.add_argument("--grid-samples", type=int, default=GRID_SAMPLES)
    parser.add_argument("--trajectory-episodes", type=int, default=TRAJECTORY_EPISODES)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    grid = probability_grid(args.grid_samples)
    cadence = cadence_table()
    trajectories = trajectory_results(args.trajectory_episodes)
    write_csv(output / "weapon_probability_grid.csv", grid)
    write_csv(output / "weapon_cadence_comparison.csv", cadence)
    write_csv(output / "weapon_trajectory_results.csv", trajectories)
    ideal = [row for row in grid if row["distance_m"] == 4000 and row["ata_deg"] == 0 and row["ha_deg"] == 0][0]
    sign_rows = [row for row in grid if row["distance_m"] == 4000 and abs(row["ata_deg"]) == 30 and abs(row["ha_deg"]) == 30]
    summary = {
        "evidence": EVIDENCE, "dt": DT, "D_hit": D_HIT, "c4": C4, "c5": C5,
        "grid_samples_per_case": args.grid_samples,
        "trajectory_monte_carlo_episodes": args.trajectory_episodes,
        "four_km_ideal_aim": ideal,
        "four_km_signed_30deg_cases": sign_rows,
        "distance_threshold_degrees": {str(int(distance)): float(np.degrees(hit_threshold(distance))) for distance in DISTANCES},
        "all_probability_finite": all(np.isfinite(row["hit_probability_shared"]) and np.isfinite(row["hit_probability_independent"]) for row in grid),
    }
    (output / "weapon_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    recommendation = {
        "D_hit": D_HIT,
        "D_hit_status": "DERIVED candidate: 4000/ln(6)",
        "c4_c5": [C4, C5],
        "c4_c5_status": "PREDECESSOR-supported RECONSTRUCTION",
        "noise_semantics": "independent epsilon_ATA and epsilon_HA",
        "noise_status": "RECONSTRUCTION selected because paper-literal shared epsilon has sign-coupling pathology",
        "shared_noise_pathology": "SHARED-NOISE SIGN-COUPLING PATHOLOGY",
        "cadence": "ENTRY_TRIGGERED_ATTEMPT",
        "cadence_status": "minimal RECONSTRUCTION; one attempt per continuous Eq.(7) window",
        "cadence_reason": "PER_STEP_RESAMPLE makes every benchmark firing dwell approximately certain kill and is implicitly dt-dependent",
        "fixed_cooldown": "NOT YET DETERMINED; no paper ammo/cooldown/interval and no additional parameter is justified by this audit",
        "entry_triggered_risk": "re-entry can create another attempt; this must remain explicit in the future environment specification",
        "head_on_note": "Eq.(7) has no AA gate, so head-on geometry legitimately enters the firing gate",
        "formal_parameters_frozen": False, "training_executed": False,
        "active_weapon_modified": False,
    }
    (output / "weapon_recommendation.json").write_text(json.dumps(recommendation, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "grid_rows": len(grid),
                      "trajectory_rows": len(trajectories), "D_hit": D_HIT}, indent=2), flush=True)


if __name__ == "__main__":
    main()
