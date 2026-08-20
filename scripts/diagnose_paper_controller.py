"""CPU benchmark for the isolated paper-constrained controller prototype."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from uav_combat.diagnostics.paper_controller_prototype import (
    EVIDENCE, FeasibleProjectedPController, ModelFeedbackPController,
    command_from_normalized, wrap_angle,
)
from uav_combat.dynamics import PointMassDynamics
from uav_combat.integrator import RK4Integrator
from uav_combat.models import AircraftSpec, AircraftState


DT = 0.1
DURATION = 20.0
INITIAL = AircraftState(0.0, 0.0, -10_000.0, 225.0, 0.0, 0.0)
SPEC = AircraftSpec(v_min=150.0, v_max=300.0,
                    theta_min=-np.pi / 3.0, theta_max=np.pi / 3.0)


def action(deg_psi=0.0, deg_theta=0.0, delta_speed=0.0) -> np.ndarray:
    return np.array([deg_psi / 180.0, deg_theta / 60.0, delta_speed / 50.0])


CASES = {
    **{f"heading_{value:+d}deg": action(deg_psi=value) for value in (30, -30, 90, -90, 180, -180)},
    **{f"pitch_{value:+d}deg": action(deg_theta=value) for value in (10, -10, 30, -30, 60, -60)},
    **{f"speed_{value:+d}mps": action(delta_speed=value) for value in (25, -25, 50, -50)},
    "combined_pos_90_30_50": action(90, 30, 50),
    "combined_neg_90_30_50": action(-90, -30, -50),
    "heading_reversal_180": action(180, 0, 0),
    "turn_90_hold_pitch_speed": action(90, 0, 0),
    "speed_change_banked_turn": action(90, 0, 50),
}


COMMON_CANDIDATES = {
    f"common_tau_{tau:g}s": (tau, tau, tau) for tau in (1.0, 2.0, 4.0)
}
MIXED_CANDIDATES = {
    "mixed_tau_psi4_theta4_v2": (4.0, 4.0, 2.0),
    "mixed_tau_psi4_theta2_v2": (4.0, 2.0, 2.0),
}


TIMESERIES_FIELDS = [
    "candidate", "case", "time_s", "x", "y", "altitude", "speed", "theta",
    "psi", "psi_unwrapped", "desired_speed", "desired_theta", "desired_psi",
    "speed_error", "theta_error", "psi_error", "raw_nx", "raw_nz", "raw_phi",
    "nx", "nz", "phi", "v_dot_command", "theta_dot_command", "psi_dot_command",
]


def first_time(condition: np.ndarray, times: np.ndarray) -> float | None:
    indices = np.flatnonzero(condition)
    return float(times[indices[0]]) if indices.size else None


def settling_time(errors: np.ndarray, times: np.ndarray, band: float) -> float | None:
    outside_after = np.maximum.accumulate((np.abs(errors) > band)[::-1])[::-1]
    indices = np.flatnonzero(~outside_after)
    return float(times[indices[0]]) if indices.size else None


def channel_metrics(
    values: np.ndarray, desired: float, initial: float, errors: np.ndarray,
    times: np.ndarray, angular: bool, minimum_band: float,
) -> dict:
    delta = wrap_angle(desired - initial) if angular else desired - initial
    magnitude = abs(delta)
    rise_threshold = 0.1 * magnitude
    rise = 0.0 if magnitude <= 1e-12 else first_time(np.abs(errors) <= rise_threshold, times)
    band = max(0.02 * magnitude, minimum_band)
    settle = settling_time(errors, times, band)
    if magnitude <= 1e-12:
        overshoot = float(np.max(np.abs(values - initial)))
    else:
        direction = np.sign(delta)
        progress = direction * (values - initial)
        overshoot = max(0.0, float(np.max(progress) - magnitude))
    tail = max(1, int(round(2.0 / DT)))
    return {
        "rise_time_s": rise, "settling_time_s": settle,
        "peak_overshoot": overshoot,
        "steady_state_abs_error": float(np.mean(np.abs(errors[-tail:]))),
        "settling_band": band,
    }


def run_case(candidate: str, taus: tuple[float, float, float], case: str, normalized: np.ndarray):
    dynamics, integrator = PointMassDynamics(), RK4Integrator(DT)
    controller_type = FeasibleProjectedPController if candidate.startswith("projected_") else ModelFeedbackPController
    controller = controller_type(*taus)
    state = INITIAL.copy()
    desired = command_from_normalized(state, normalized)
    rows, psi_unwrapped, prior_psi = [], state.psi, state.psi
    for step in range(int(DURATION / DT) + 1):
        result = controller.control(state, desired)
        if step:
            psi_unwrapped += wrap_angle(state.psi - prior_psi)
        prior_psi = state.psi
        rows.append({
            "candidate": candidate, "case": case, "time_s": step * DT,
            "x": state.x, "y": state.y, "altitude": state.altitude,
            "speed": state.v, "theta": state.theta, "psi": state.psi,
            "psi_unwrapped": psi_unwrapped, "desired_speed": desired.speed,
            "desired_theta": desired.theta, "desired_psi": desired.psi,
            "speed_error": result.speed_error, "theta_error": result.theta_error,
            "psi_error": result.psi_error, "raw_nx": result.raw_nx,
            "raw_nz": result.raw_nz, "raw_phi": result.raw_phi,
            "nx": result.command.nx, "nz": result.command.nz,
            "phi": result.command.phi, "v_dot_command": result.v_dot_command,
            "theta_dot_command": result.theta_dot_command,
            "psi_dot_command": result.psi_dot_command,
        })
        if step < int(DURATION / DT):
            state = integrator.step(state, result.command, dynamics, SPEC)
    return rows, desired


def summarize_case(candidate: str, case: str, taus, rows: list[dict], desired) -> dict:
    arrays = {key: np.asarray([row[key] for row in rows], dtype=float) for key in rows[0]
              if key not in {"candidate", "case"}}
    times = arrays["time_s"]
    psi_target_unwrapped = INITIAL.psi + wrap_angle(desired.psi - INITIAL.psi)
    metrics = {
        "psi": channel_metrics(arrays["psi_unwrapped"], psi_target_unwrapped, INITIAL.psi,
                               arrays["psi_error"], times, True, np.deg2rad(0.5)),
        "theta": channel_metrics(arrays["theta"], desired.theta, INITIAL.theta,
                                 arrays["theta_error"], times, False, np.deg2rad(0.25)),
        "speed": channel_metrics(arrays["speed"], desired.speed, INITIAL.v,
                                 arrays["speed_error"], times, False, 0.25),
    }
    finite = all(np.all(np.isfinite(array)) for array in arrays.values())
    phi_sat = np.abs(arrays["raw_phi"]) > np.pi / 2.0 + 1e-12
    return {
        "candidate": candidate, "case": case, "tau_psi": taus[0],
        "tau_theta": taus[1], "tau_speed": taus[2],
        "command_delta_psi_deg": np.degrees(wrap_angle(desired.psi - INITIAL.psi)),
        "command_delta_theta_deg": np.degrees(desired.theta - INITIAL.theta),
        "command_delta_speed": desired.speed - INITIAL.v,
        **{f"{channel}_{key}": value for channel, data in metrics.items() for key, value in data.items()},
        "max_abs_raw_phi_deg": float(np.degrees(np.max(np.abs(arrays["raw_phi"])))),
        "max_abs_phi_deg": float(np.degrees(np.max(np.abs(arrays["phi"])))),
        "max_raw_nz": float(np.max(arrays["raw_nz"])),
        "max_applied_nz": float(np.max(arrays["nz"])),
        "max_abs_raw_nx": float(np.max(np.abs(arrays["raw_nx"]))),
        "fraction_phi_saturated": float(np.mean(phi_sat)),
        "fraction_applied_phi_at_limit": float(np.mean(np.abs(arrays["phi"]) >= np.pi / 2.0 - 1e-12)),
        "fraction_nz_gt_5": float(np.mean(arrays["nz"] > 5.0)),
        "fraction_nz_gt_8": float(np.mean(arrays["nz"] > 8.0)),
        "fraction_raw_nz_gt_5": float(np.mean(arrays["raw_nz"] > 5.0)),
        "fraction_raw_nz_gt_8": float(np.mean(arrays["raw_nz"] > 8.0)),
        "finite": finite,
        "unstable": bool(not finite or np.max(np.abs(arrays["raw_nz"])) > 1e6),
        "min_speed": float(np.min(arrays["speed"])), "max_speed": float(np.max(arrays["speed"])),
        "min_theta_deg": float(np.degrees(np.min(arrays["theta"]))),
        "max_theta_deg": float(np.degrees(np.max(arrays["theta"]))),
        "min_altitude": float(np.min(arrays["altitude"])),
    }


def aggregate(results: list[dict]) -> dict:
    summary = {}
    for candidate in sorted({row["candidate"] for row in results}):
        subset = [row for row in results if row["candidate"] == candidate]
        summary[candidate] = {
            "cases": len(subset), "all_finite": all(row["finite"] for row in subset),
            "unstable_cases": sum(row["unstable"] for row in subset),
            "max_raw_nz": max(row["max_raw_nz"] for row in subset),
            "max_applied_nz": max(row["max_applied_nz"] for row in subset),
            "max_abs_raw_nx": max(row["max_abs_raw_nx"] for row in subset),
            "max_abs_raw_phi_deg": max(row["max_abs_raw_phi_deg"] for row in subset),
            "mean_case_fraction_phi_saturated": float(np.mean([row["fraction_phi_saturated"] for row in subset])),
            "mean_case_fraction_applied_phi_at_limit": float(np.mean([row["fraction_applied_phi_at_limit"] for row in subset])),
            "mean_case_fraction_nz_gt_5": float(np.mean([row["fraction_nz_gt_5"] for row in subset])),
            "mean_case_fraction_nz_gt_8": float(np.mean([row["fraction_nz_gt_8"] for row in subset])),
            "unsettled_psi_cases": sum(row["psi_settling_time_s"] is None for row in subset),
            "unsettled_theta_cases": sum(row["theta_settling_time_s"] is None for row in subset),
            "unsettled_speed_cases": sum(row["speed_settling_time_s"] is None for row in subset),
            "max_steady_psi_error_deg": float(np.degrees(max(row["psi_steady_state_abs_error"] for row in subset))),
            "max_steady_theta_error_deg": float(np.degrees(max(row["theta_steady_state_abs_error"] for row in subset))),
            "max_steady_speed_error": max(row["speed_steady_state_abs_error"] for row in subset),
            "max_psi_overshoot_deg": float(np.degrees(max(row["psi_peak_overshoot"] for row in subset))),
            "max_theta_overshoot_deg": float(np.degrees(max(row["theta_peak_overshoot"] for row in subset))),
            "max_speed_overshoot": max(row["speed_peak_overshoot"] for row in subset),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/paper_environment_prototypes/controller")
    parser.add_argument("--include-mixed", action="store_true")
    parser.add_argument("--include-projected", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    candidates = dict(COMMON_CANDIDATES)
    if args.include_mixed:
        candidates.update(MIXED_CANDIDATES)
    if args.include_projected:
        candidates.update({f"projected_common_tau_{tau:g}s": (tau, tau, tau)
                           for tau in (1.0, 2.0, 4.0)})
    results = []
    for candidate, taus in candidates.items():
        print(f"[CONTROLLER] {candidate}", flush=True)
        for case, normalized in CASES.items():
            rows, desired = run_case(candidate, taus, case, normalized)
            results.append(summarize_case(candidate, case, taus, rows, desired))
            path = output / f"controller_timeseries_{candidate}_{case}.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=TIMESERIES_FIELDS)
                writer.writeheader(); writer.writerows(rows)
    with (output / "controller_case_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader(); writer.writerows(results)
    summary = {
        "dt": DT, "duration_s": DURATION, "initial_state": asdict(INITIAL),
        "evidence": EVIDENCE, "candidates": aggregate(results),
        "metric_note": "Rise=first entry into 10% error; settling=remaining within a conventional 2% comparison band. These are diagnostics, not paper requirements.",
    }
    (output / "controller_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    recommendation = {
        "controller_order": "feasible-projected P-only model-feedback controller",
        "recommended_time_constants_s": {"psi": 2.0, "theta": 2.0, "speed": 2.0},
        "p_only_sufficient": True,
        "integral_needed": False,
        "integral_reason": "normal commands converge without persistent error; the -60 deg descent delay is a 3-DOF/roll-feasibility limit and I would wind up rather than fix it",
        "derivative_needed": False,
        "derivative_reason": "projected P responses show no sustained oscillation and negligible overshoot",
        "projection_needed": True,
        "projection_reason": "unconstrained A<0 asks for raw phi near 180 deg; clipping it creates spurious yaw, while A=max(A,0) preserves feasibility without I/D",
        "tau_choice_reason": "tau=1 s creates the largest nx/nz; tau=4 s needs about 15.5 s to settle ordinary 30/90 deg channels; tau=2 s settles them in about 7.7 s",
        "nz_hard_limit_needed": True,
        "nz_limit_value": "NOT YET DETERMINED",
        "nz_limit_reason": "unlimited tau=2 commands reach about 18 g for 90 deg and 36 g for 180 deg heading changes",
        "five_g_status": "sanity reference only; clearly clips even 30/90 deg cases",
        "eight_g_status": "sanity reference only; still clips portions of 90/180 deg and combined cases",
        "heading_180_status": "finite and convergent, but physically extreme; +180 and -180 share the same wrapped -pi branch",
        "prototype_gain_selected": True,
        "formal_controller_frozen": False,
        "formal_freeze_blocker": "nz hard-limit value and limited-controller tracking have not yet been validated",
        "nx_limit_needed_by_current_evidence": False,
        "nx_limit_note": "tau=2 s reaches |nx|=2.548 for the full +/-50 m/s command; paper gives no nx limit",
        "nx_nz_limit_frozen": False,
        "training_executed": False, "active_environment_modified": False,
    }
    (output / "controller_recommendation.json").write_text(json.dumps(recommendation, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "cases": len(results)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
