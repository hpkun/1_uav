"""Reproducible V1.4 translation-invariant environment validation."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import numpy as np

from diagnose_action_stability import ACTOR_SEEDS, finalize_short, short_worker
from uav_combat.diagnostics.action_stability import vertical_balance
from uav_combat.environment.env import MultiUAVCombatEnv
from uav_combat.environment.geometry import engagement_geometry
from uav_combat.math_utils import wrap_angle
from uav_combat.models import AircraftState


def distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "min": None, "p10": None, "p50": None, "p90": None, "max": None}
    a = np.asarray(values, dtype=float)
    return {"count": len(values), "min": float(a.min()), "p10": float(np.percentile(a, 10)),
            "p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)), "max": float(a.max())}


def statistics(values: list[float]) -> dict:
    a = np.asarray(values, dtype=float)
    return {"count": int(a.size), "mean": float(a.mean()), "std": float(a.std()),
            "p10": float(np.percentile(a, 10)), "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)), "max_abs": float(np.abs(a).max())}


def reset_statistics(config: Path, count: int) -> dict:
    env = MultiUAVCombatEnv(config)
    speeds, altitudes, separations, pairs, perturbations, spans = [], [], [], [], [], []
    for seed in range(count):
        _, info = env.reset(seed)
        nominal_red, nominal_blue = info["radial_angle"], wrap_angle(info["radial_angle"] + np.pi)
        red_center = np.mean([[s.x, s.y] for s in env.red], axis=0)
        blue_center = np.mean([[s.x, s.y] for s in env.blue], axis=0)
        separations.append(float(np.linalg.norm(blue_center-red_center)))
        spans.extend(float(np.hypot(team[-1].x-team[0].x, team[-1].y-team[0].y)) for team in (env.red, env.blue))
        pairs.extend(float(np.linalg.norm([b.x-r.x, b.y-r.y, b.z-r.z])) for r in env.red for b in env.blue)
        perturbations.extend(wrap_angle(s.psi-nominal_red) for s in env.red)
        perturbations.extend(wrap_angle(s.psi-nominal_blue) for s in env.blue)
        speeds.extend(s.v for s in env.red+env.blue); altitudes.extend(s.altitude for s in env.red+env.blue)
    return {"resets": count, "speed": statistics(speeds), "altitude": statistics(altitudes),
            "team_center_separation": distribution(separations), "red_blue_pair_distance": distribution(pairs),
            "formation_span": distribution(spans), "heading_perturbation_deg": distribution(list(np.rad2deg(perturbations)))}


def flank_actions(env: MultiUAVCombatEnv, nominal_heading: float) -> np.ndarray:
    actions = []
    for index, own in enumerate(env.red):
        if not own.alive:
            actions.append(np.zeros(3, dtype=np.float32)); continue
        offset = np.deg2rad(30.0) if index < 2 else -np.deg2rad(30.0)
        actions.append(env.fixed_policy.action_toward(own, wrap_angle(nominal_heading+offset), 0.0, 260.0))
    return np.stack(actions)


def spread_snapshot(env: MultiUAVCombatEnv) -> tuple[float, float, float]:
    alive = [s for s in env.red+env.blue if s.alive]
    pairs = [np.hypot(a.x-b.x, a.y-b.y) for i, a in enumerate(alive) for b in alive[i+1:]]
    nearest = []
    for team, opponents in ((env.red, env.blue), (env.blue, env.red)):
        targets = [s for s in opponents if s.alive]
        for own in team:
            if own.alive and targets:
                nearest.append(min(np.linalg.norm([t.x-own.x, t.y-own.y, t.z-own.z]) for t in targets))
    return float(max(pairs, default=0.0)), float(max(nearest, default=0.0)), float(min(nearest, default=0.0))


def run_episode(task: tuple[str, str, int]) -> dict:
    config, scenario, seed = task
    env = MultiUAVCombatEnv(config); _, reset_info = env.reset(seed)
    shaping, events, max_pair, max_nearest, min_enemy, final_nearest = [], [], 0.0, 0.0, float("inf"), 0.0
    while True:
        pair, nearest_max, nearest_min = spread_snapshot(env)
        max_pair, max_nearest = max(max_pair, pair), max(max_nearest, nearest_max)
        if nearest_min > 0.0: min_enemy, final_nearest = min(min_enemy, nearest_min), nearest_min
        if scenario == "straight":
            red_actions = blue_actions = np.zeros((4, 3), dtype=np.float32)
        elif scenario == "rule":
            red_actions, blue_actions = env.fixed_policy.team_actions(env.red, env.blue), None
        elif scenario == "flank":
            red_actions = flank_actions(env, reset_info["radial_angle"]) if env.steps < 50 else env.fixed_policy.team_actions(env.red, env.blue)
            blue_actions = None
        else: raise ValueError(scenario)
        _, _, terminated, truncated, info = env.step(red_actions, blue_actions)
        shaping.extend(map(float, info["shaping_rewards"])); events.extend(map(float, info["event_rewards"]))
        if terminated or truncated:
            record = dict(info); record["_shaping"], record["_event"] = shaping, events
            record["max_horizontal_pair_separation"], record["max_nearest_enemy_distance"] = max_pair, max_nearest
            record["minimum_enemy_distance"] = min_enemy if np.isfinite(min_enemy) else 0.0
            record["final_nearest_enemy_distance"] = final_nearest
            return record


def summarize(records: list[dict]) -> dict:
    episodes = len(records); mean = lambda key: float(np.mean([r[key] for r in records]))
    reasons = ("red_win", "blue_win", "draw_mutual_destruction", "draw_timeout")
    altitude_total = lambda r: sum(r[k] for k in ("red_low_altitude_losses", "blue_low_altitude_losses", "red_high_altitude_losses", "blue_high_altitude_losses"))
    return {
        "episodes": episodes,
        **{
            f"{side}_{event}_episodes": sum(
                r[f"{side}_first_{event}_step"] is not None for r in records
            )
            for side in ("red", "blue") for event in ("attackable", "lock", "kill")
        },
        **{
            f"{side}_first_{event}_step": distribution([
                r[f"{side}_first_{event}_step"] for r in records
                if r[f"{side}_first_{event}_step"] is not None
            ])
            for side in ("red", "blue") for event in ("attackable", "lock", "kill")
        },
        "red_attack_kills_total": int(sum(r["red_attack_kills"] for r in records)), "blue_attack_kills_total": int(sum(r["blue_attack_kills"] for r in records)),
        **{f"{key}_total": int(sum(r[key] for r in records)) for key in ("red_low_altitude_losses", "blue_low_altitude_losses", "red_high_altitude_losses", "blue_high_altitude_losses")},
        "altitude_losses_per_episode": float(np.mean([altitude_total(r) for r in records])),
        "win_rate": mean("red_win"), "loss_rate": mean("blue_win"), "draw_rate": mean("draw"),
        "termination_counts": {reason: sum(r["termination_reason"] == reason for r in records) for reason in reasons},
        "episode_length": distribution([r["episode_length"] for r in records]), "episode_length_mean": mean("episode_length"),
        "minimum_enemy_distance": distribution([r["minimum_enemy_distance"] for r in records]),
        "max_horizontal_pair_separation": distribution([r["max_horizontal_pair_separation"] for r in records]),
        "max_nearest_enemy_distance": distribution([r["max_nearest_enemy_distance"] for r in records]),
        "final_nearest_enemy_distance": distribution([r["final_nearest_enemy_distance"] for r in records]),
        "shaping_reward_statistics": statistics([v for r in records for v in r["_shaping"]]),
        "event_reward_statistics": statistics([v for r in records for v in r["_event"]]),
    }


def run_scenario(config: Path, scenario: str, episodes: int, seed_base: int, workers: int) -> dict:
    tasks = [(str(config), scenario, seed_base+i) for i in range(episodes)]
    if workers == 1: return summarize([run_episode(task) for task in tasks])
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return summarize(list(executor.map(run_episode, tasks, chunksize=1)))


def fresh_stochastic_short(config: Path, workers: int) -> dict:
    tasks = [(str(config), "fresh_stochastic", seed, 42_000_000+i*100, 100) for i, seed in enumerate(ACTOR_SEEDS)]
    with ProcessPoolExecutor(max_workers=workers) as executor: return finalize_short(list(executor.map(short_worker, tasks)))


def translation_invariance(config: Path, steps: int = 50) -> dict:
    first, shifted = MultiUAVCombatEnv(config), MultiUAVCombatEnv(config); first.reset(7_700_001); shifted.reset(7_700_001)
    translation = np.array([50_000.0, -30_000.0])
    for s in shifted.red+shifted.blue: s.x += translation[0]; s.y += translation[1]
    errors = {key: 0.0 for key in ("observation", "action", "reward", "state", "translation")}; terminal = True
    for _ in range(steps):
        errors["observation"] = max(errors["observation"], float(np.max(np.abs(first._observations()-shifted._observations()))))
        red_a, red_b = first.fixed_policy.team_actions(first.red, first.blue), shifted.fixed_policy.team_actions(shifted.red, shifted.blue)
        blue_a, blue_b = first.fixed_policy.team_actions(first.blue, first.red), shifted.fixed_policy.team_actions(shifted.blue, shifted.red)
        errors["action"] = max(errors["action"], float(np.max(np.abs(red_a-red_b))), float(np.max(np.abs(blue_a-blue_b))))
        out_a, out_b = first.step(red_a, blue_a), shifted.step(red_b, blue_b)
        errors["reward"] = max(errors["reward"], float(np.max(np.abs(out_a[1]-out_b[1])))); terminal &= out_a[2:4] == out_b[2:4]
        for a, b in zip(first.red+first.blue, shifted.red+shifted.blue):
            errors["state"] = max(errors["state"], float(np.max(np.abs(a.as_array()[2:]-b.as_array()[2:]))))
            errors["translation"] = max(errors["translation"], abs((b.x-a.x)-translation[0]), abs((b.y-a.y)-translation[1]))
        if out_a[2] or out_a[3]: break
    return {"steps": steps, "translation": translation.tolist(), "max_errors": errors, "matching_terminal": terminal,
            "passed": max(errors.values()) < 1e-5 and terminal}


def invariant_checks(config: Path) -> dict:
    env = MultiUAVCombatEnv(config); observation, _ = env.reset(8_800_001)
    rotated = MultiUAVCombatEnv(config); rotated.reset(8_800_001); angle = 1.234; c, s = np.cos(angle), np.sin(angle)
    for state in rotated.red+rotated.blue:
        state.x, state.y = c*state.x-s*state.y, s*state.x+c*state.y; state.psi = wrap_angle(state.psi+angle)
    rotation_error = float(np.max(np.abs(observation-rotated._observations())))
    own, head_on, tail = AircraftState(0,0,-3000,225,0,0), AircraftState(1000,0,-3000,225,0,np.pi), AircraftState(1000,0,-3000,225,0,0)
    theta, a2 = np.deg2rad(np.array([-30,0,30]))[:,None], np.linspace(-1,1,9)[None,:]
    trim_error = float(np.max(np.abs(vertical_balance(theta, 0.0, a2))))
    return {"trim_max_abs_vertical_balance": trim_error, "trim_passed": trim_error < 1e-12,
            "observation_shape": list(observation.shape), "observation_finite": bool(np.all(np.isfinite(observation))),
            "rotation_max_abs_error": rotation_error, "rotation_invariant": rotation_error < 2e-6,
            "head_on_not_attackable": not env.weapon.attackable(engagement_geometry(own, head_on)),
            "tail_attackable": env.weapon.attackable(engagement_geometry(own, tail)), "lock_steps_required": env.weapon.lock_steps_required,
            "horizontal_origin_irrelevant": set(env.config["flight_envelope"]) == {"altitude_min", "altitude_max"}}


def all_finite(value) -> bool:
    if isinstance(value, dict): return all(all_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)): return all(all_finite(v) for v in value)
    return not isinstance(value, (float, np.floating)) or np.isfinite(value)


def acceptance(result: dict) -> dict:
    short, straight, rule, flank, inv = result["fresh_stochastic_100_step_stability"], result["straight_vs_straight"], result["rule_vs_rule"], result["flank_vs_rule"], result["invariant_checks"]
    checks = {
        "B_trim_relative_action": inv["trim_passed"],
        "C_fresh_stochastic_vertical_stability": abs(short["altitude_change"]["mean"]) < 25 and abs(short["theta_change"]["mean"]) < 0.025,
        "D_straight_baseline": all(
            straight[f"{side}_{event}_episodes"] == 0
            for side in ("red", "blue") for event in ("attackable", "lock", "kill")
        ) and straight["altitude_losses_per_episode"] == 0 and straight["termination_counts"]["draw_timeout"] == straight["episodes"],
        "E_rule_flank_finite": all_finite(rule) and all_finite(flank),
        "F_combat_chain_reachable": any(
            x[f"{side}_attackable_episodes"] > 0
            and x[f"{side}_lock_episodes"] > 0
            and x[f"{side}_kill_episodes"] > 0
            for x in (rule, flank) for side in ("red", "blue")
        ),
        "G_no_mass_vertical_self_destruction": rule["altitude_losses_per_episode"] <= 0.5 and flank["altitude_losses_per_episode"] <= 0.5,
        "H_observation_52d": inv["observation_shape"] == [4,52] and inv["observation_finite"],
        "I_rotation_invariance": inv["rotation_invariant"], "J_translation_invariance": result["translation_invariance"]["passed"],
        "K_frozen_core_regressions": inv["head_on_not_attackable"] and inv["tail_attackable"] and inv["lock_steps_required"] == 3 and inv["horizontal_origin_irrelevant"],
    }
    return {"checks": checks, "passed_excluding_external_pytest_gate_A": all(checks.values())}


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--reset-count",type=int,default=1000); p.add_argument("--straight-episodes",type=int,default=100); p.add_argument("--rule-episodes",type=int,default=200); p.add_argument("--flank-episodes",type=int,default=200); p.add_argument("--workers",type=int,default=min(4,os.cpu_count() or 1)); p.add_argument("--output",default="outputs/combat_environment_validation_v1_4.json"); args=p.parse_args()
    if min(args.reset_count,args.straight_episodes,args.rule_episodes,args.flank_episodes,args.workers)<=0: raise ValueError("all counts and workers must be positive")
    root=Path(__file__).resolve().parents[1]; config=root/"configs/combat_environment.yaml"
    result={"reset_statistics":reset_statistics(config,args.reset_count), "invariant_checks":invariant_checks(config),
            "translation_invariance":translation_invariance(config), "fresh_stochastic_100_step_stability":fresh_stochastic_short(config,args.workers),
            "straight_vs_straight":run_scenario(config,"straight",args.straight_episodes,1_000_000,args.workers),
            "rule_vs_rule":run_scenario(config,"rule",args.rule_episodes,2_000_000,args.workers),
            "flank_vs_rule":run_scenario(config,"flank",args.flank_episodes,3_000_000,args.workers)}
    result["acceptance"]=acceptance(result); output=root/args.output; output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result,indent=2))


if __name__ == "__main__": main()
