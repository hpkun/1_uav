"""Read-only trajectory audits for historical V2.2 checkpoints."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from uav_combat.environment.env import MultiUAVCombatEnv
from uav_combat.environment.geometry import engagement_geometry
from uav_combat.environment.reward import paper_state_reward_components
from uav_combat.madsac.actor import SharedSquashedGaussianActor
from uav_combat.mappo.networks import SharedMAPPOActor


ROOT = Path(__file__).resolve().parents[1]
EVAL_SEEDS = range(10_000_000, 10_000_020)


def load_actor(kind: str, checkpoint: Path):
    config = yaml.safe_load((ROOT / f"configs/{kind}.yaml").read_text())
    network, impl = config["network"], config["implementation"]
    actor_class = SharedSquashedGaussianActor if kind == "madsac" else SharedMAPPOActor
    actor = actor_class(
        observation_dim=int(network["observation_dim"]),
        action_dim=int(network["action_dim"]),
        hidden_dim=int(network["actor_hidden_layers"][0]),
        log_std_min=float(impl["log_std_min"]),
        log_std_max=float(impl["log_std_max"]),
        activation=str(impl["actor_activation"]),
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor.load_state_dict(state["actor"])
    actor.eval()
    return actor


def policy_values(actor, observation: np.ndarray, deterministic: bool):
    tensor = torch.as_tensor(observation, dtype=torch.float32)
    with torch.no_grad():
        distribution = actor.distribution(tensor)
        raw = distribution.mean if deterministic else distribution.sample()
        action = torch.tanh(raw)
    return (
        distribution.mean.numpy(), distribution.scale.log().numpy(),
        raw.numpy(), action.numpy(),
    )


def prehit_snapshot(env: MultiUAVCombatEnv, red_alive, blue_alive):
    red = [state.copy() for state in env.red]
    blue = [state.copy() for state in env.blue]
    for states, before in ((red, red_alive), (blue, blue_alive)):
        for state, was_alive in zip(states, before):
            state.alive = bool(
                was_alive and np.hypot(state.x, state.y) <= env.arena_radius
                and state.altitude > 0.0
            )
    return red, blue


def nearest_rows(red, blue, components):
    rows = []
    for index, own in enumerate(red):
        candidates = [
            (engagement_geometry(own, target).distance, target_index)
            for target_index, target in enumerate(blue) if own.alive and target.alive
        ]
        if not candidates:
            rows.append({"red_index": index, "target_index": None,
                         "distance": None, "off_boresight_deg": None,
                         "ata_deg": None, "ha_deg": None, "aa_deg": None,
                         "blue_reverse_off_boresight_deg": None,
                         "r3": float(components["r3"][index]),
                         "r4": float(components["r4"][index])})
            continue
        _, target_index = min(candidates)
        forward = engagement_geometry(own, blue[target_index])
        reverse = engagement_geometry(blue[target_index], own)
        rows.append({
            "red_index": index, "target_index": target_index,
            "distance": forward.distance,
            "off_boresight_deg": float(np.rad2deg(forward.off_boresight)),
            "ata_deg": float(np.rad2deg(forward.ata)),
            "ha_deg": float(np.rad2deg(forward.ha)),
            "aa_deg": float(np.rad2deg(forward.aa)),
            "blue_reverse_off_boresight_deg": float(np.rad2deg(reverse.off_boresight)),
            "r3": float(components["r3"][index]),
            "r4": float(components["r4"][index]),
        })
    return rows


def episode(kind, actor, config, seed: int, deterministic: bool, step_stream=None):
    torch.manual_seed(seed + (0 if deterministic else 1_000_000))
    env = MultiUAVCombatEnv(config)
    observation, _ = env.reset(seed)
    total_return = 0.0
    action_rows = []
    radial_rows = []
    reward_rows = []
    while True:
        alive = env.red_alive_mask.copy()
        red_alive_before = [state.alive for state in env.red]
        blue_alive_before = [state.alive for state in env.blue]
        mean, log_std, raw, action = policy_values(actor, observation, deterministic)
        action *= alive[:, None]
        next_observation, reward, terminated, truncated, info = env.step(action)
        total_return += float(reward.sum())
        red_pre, blue_pre = prehit_snapshot(env, red_alive_before, blue_alive_before)
        components = paper_state_reward_components(red_pre, blue_pre, config["reward"])
        geometries = nearest_rows(red_pre, blue_pre, components)
        for index, state in enumerate(env.red):
            if alive[index] <= .5:
                continue
            radius = float(np.hypot(state.x, state.y))
            horizontal_velocity = state.velocity_vector()[:2]
            outward = np.asarray([state.x, state.y]) / max(radius, 1e-12)
            radial_velocity = float(np.dot(horizontal_velocity, outward))
            action_rows.append(action[index].copy())
            radial_rows.append(radial_velocity)
            reward_rows.append(geometries[index])
            if step_stream is not None:
                step_stream.write(json.dumps({
                    "algorithm": kind, "checkpoint": getattr(actor, "audit_checkpoint", ""),
                    "deterministic": deterministic, "seed": seed, "step": env.steps,
                    "red_index": index, "mu_psi": float(mean[index, 0]),
                    "mu_theta": float(mean[index, 1]), "mu_v": float(mean[index, 2]),
                    "logstd_psi": float(log_std[index, 0]),
                    "logstd_theta": float(log_std[index, 1]),
                    "logstd_v": float(log_std[index, 2]),
                    "raw_action": raw[index].tolist(), "action": action[index].tolist(),
                    "psi": float(state.psi), "theta": float(state.theta), "v": float(state.v),
                    "x": float(state.x), "y": float(state.y), "altitude": float(state.altitude),
                    "arena_radius": radius, "radial_velocity_outward": radial_velocity,
                    "distance_to_boundary": float(env.arena_radius - radius),
                    **geometries[index],
                    "fire_window_pairs": int(info["red_fire_window_pairs"]),
                    "attempts": int(info["red_step_fire_attempts"]),
                    "hits": int(info["red_step_weapon_hits"]),
                    "kills": int(info["red_step_attack_kills"]),
                    "boundary_exits_total": int(info["red_boundary_exits"]),
                }, separators=(",", ":")) + "\n")
        observation = next_observation
        if terminated or truncated:
            break
    actions = np.asarray(action_rows) if action_rows else np.zeros((0, 3))
    return {
        "seed": seed, "deterministic": deterministic, "return": total_return,
        "win": bool(info["red_win"]), "red_losses": int(info["red_losses"]),
        "boundary_exits": int(info["red_boundary_exits"]),
        "fire_attempts": int(info["red_fire_attempts"]),
        "weapon_hits": int(info["red_weapon_hits"]),
        "kills": int(info["red_attack_kills"]),
        "episode_length": int(info["episode_length"]),
        "action_mean": actions.mean(0).tolist() if len(actions) else [0.0] * 3,
        "action_abs_mean": np.abs(actions).mean(0).tolist() if len(actions) else [0.0] * 3,
        "action_std": actions.std(0).tolist() if len(actions) else [0.0] * 3,
        "positive_fraction": (actions > 0).mean(0).tolist() if len(actions) else [0.0] * 3,
        "negative_fraction": (actions < 0).mean(0).tolist() if len(actions) else [0.0] * 3,
        "radial_velocity_outward_mean": float(np.mean(radial_rows)) if radial_rows else 0.0,
        "reward_geometry_rows": reward_rows,
    }


def aggregate(rows):
    array = lambda key: np.asarray([row[key] for row in rows], dtype=float)
    actions = np.asarray([row["action_mean"] for row in rows])
    return {
        "episodes": len(rows), "return_mean": float(array("return").mean()),
        "win_rate": float(array("win").mean()),
        "red_losses_mean": float(array("red_losses").mean()),
        "boundary_exits_mean": float(array("boundary_exits").mean()),
        "fire_attempts_mean": float(array("fire_attempts").mean()),
        "weapon_hits_mean": float(array("weapon_hits").mean()),
        "kills_mean": float(array("kills").mean()),
        "action_mean": actions.mean(0).tolist(),
        "delta_psi_degrees_per_step": float(actions.mean(0)[0] * 180.0),
        "delta_theta_degrees_per_step": float(actions.mean(0)[1] * 60.0),
        "delta_v_mps_per_step": float(actions.mean(0)[2] * 50.0),
        "radial_velocity_outward_mean": float(array("radial_velocity_outward_mean").mean()),
    }


def reward_summary(rows):
    geometry = [g for row in rows for g in row["reward_geometry_rows"]
                if g["off_boresight_deg"] is not None]
    subset = lambda predicate: [g for g in geometry if predicate(g)]
    ratio = lambda selected, predicate: (
        float(np.mean([predicate(g) for g in selected])) if selected else None
    )
    r3 = subset(lambda g: g["r3"] > 0)
    r4p = subset(lambda g: g["r4"] > 0)
    r4n = subset(lambda g: g["r4"] < 0)
    fire = subset(lambda g: g["distance"] <= 4000 and g["off_boresight_deg"] <= 30)
    return {
        "geometry_samples": len(geometry),
        "r3_positive_samples": len(r3),
        "r3_positive_off_boresight_gt_30_rate": ratio(r3, lambda g: g["off_boresight_deg"] > 30),
        "r4_positive_samples": len(r4p),
        "r4_positive_off_boresight_gt_30_rate": ratio(r4p, lambda g: g["off_boresight_deg"] > 30),
        "r4_negative_samples": len(r4n),
        "r4_negative_blue_true_threat_within_30_rate": ratio(
            r4n, lambda g: g["blue_reverse_off_boresight_deg"] <= 30),
        "fire_window_samples": len(fire),
        "fire_window_r3_positive_rate": ratio(fire, lambda g: g["r3"] > 0),
        "fire_window_r4_positive_rate": ratio(fire, lambda g: g["r4"] > 0),
        "fire_window_r4_negative_rate": ratio(fire, lambda g: g["r4"] < 0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mappo", type=Path, required=True)
    parser.add_argument("--madsac-best", type=Path, required=True)
    parser.add_argument("--madsac-drift", type=Path, nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "configs/combat_environment.yaml").read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"environment_version": config["environment_version"], "reward_audit": {},
              "madsac_policy_drift": []}
    with gzip.open(args.output_dir / "trajectory_steps.jsonl.gz", "wt", encoding="utf-8") as stream:
        for kind, checkpoint in (("mappo", args.mappo), ("madsac", args.madsac_best)):
            actor = load_actor(kind, checkpoint); actor.audit_checkpoint = str(checkpoint)
            rows = [episode(kind, actor, config, seed, True, stream) for seed in EVAL_SEEDS]
            report["reward_audit"][kind] = {
                "checkpoint": str(checkpoint), "performance": aggregate(rows),
                "reward_geometry": reward_summary(rows),
            }
        for checkpoint in args.madsac_drift:
            actor = load_actor("madsac", checkpoint); actor.audit_checkpoint = str(checkpoint)
            item = {"checkpoint": str(checkpoint)}
            for deterministic in (True, False):
                rows = [episode("madsac", actor, config, seed, deterministic, stream)
                        for seed in EVAL_SEEDS]
                item["deterministic" if deterministic else "stochastic"] = aggregate(rows)
            report["madsac_policy_drift"].append(item)
            print(f"completed {checkpoint}", flush=True)
    (args.output_dir / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.output_dir / "madsac_policy_drift.csv").open("w", newline="", encoding="utf-8") as stream:
        flat = []
        for item in report["madsac_policy_drift"]:
            for mode in ("deterministic", "stochastic"):
                flat.append({"checkpoint": item["checkpoint"], "mode": mode, **item[mode]})
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader(); writer.writerows(flat)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
