"""Select a shared representative evaluation seed and plot best-model 3D tracks."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics

import matplotlib.pyplot as plt
import numpy as np
import yaml

from uav_combat.environment.factory import make_combat_environment
from uav_combat.madsac.trainer import MADSACTrainer
from uav_combat.mappo.trainer import MAPPOTrainer


ROOT = Path(__file__).resolve().parents[1]


def build_madsac(checkpoint: Path, device: str):
    cfg = yaml.safe_load((ROOT / "configs/madsac.yaml").read_text(encoding="utf-8"))
    network, training, implementation = cfg["network"], cfg["training"], cfg["implementation"]
    trainer = MADSACTrainer(
        observation_dim=int(network["observation_dim"]),
        action_dim=int(network["action_dim"]),
        num_agents=int(network["num_agents"]),
        hidden_dim=int(network["actor_hidden_layers"][0]),
        attention_heads=int(network["attention_heads"]),
        learning_rate=float(training["learning_rate"]), gamma=float(training["gamma"]),
        tau=float(training["tau"]), alpha=float(training["alpha"]),
        replay_capacity=1, batch_size=1, device=device,
        actor_activation=implementation["actor_activation"],
        critic_activation=implementation["critic_activation"],
        log_std_min=float(implementation["log_std_min"]),
        log_std_max=float(implementation["log_std_max"]),
    )
    trainer.load(checkpoint)
    trainer.actor.eval()
    return trainer


def build_mappo(checkpoint: Path, device: str):
    cfg = yaml.safe_load((ROOT / "configs/mappo.yaml").read_text(encoding="utf-8"))
    network, training, implementation = cfg["network"], cfg["training"], cfg["implementation"]
    trainer = MAPPOTrainer(
        observation_dim=int(network["observation_dim"]),
        action_dim=int(network["action_dim"]), num_agents=int(network["num_agents"]),
        hidden_dim=int(network["actor_hidden_layers"][0]),
        attention_heads=int(network["attention_heads"]),
        actor_learning_rate=float(training["actor_learning_rate"]),
        critic_learning_rate=float(training["critic_learning_rate"]),
        gamma=float(training["gamma"]), gae_lambda=float(training["gae_lambda"]),
        clip_ratio=float(training["clip_ratio"]),
        value_loss_coefficient=float(training["value_loss_coefficient"]),
        entropy_coefficient=float(training["entropy_coefficient"]),
        max_grad_norm=float(training["max_grad_norm"]),
        ppo_epochs=int(training["ppo_epochs"]),
        minibatch_size=int(training["minibatch_size"]),
        normalize_advantages=bool(implementation["normalize_advantages"]),
        clip_value_loss=bool(implementation["clip_value_loss"]), device=device,
        actor_activation=implementation["actor_activation"],
        critic_activation=implementation["critic_activation"],
        log_std_min=float(implementation["log_std_min"]),
        log_std_max=float(implementation["log_std_max"]),
    )
    trainer.load(checkpoint, allow_legacy_diagnostic=True)
    trainer.actor.eval()
    trainer.critic.eval()
    return trainer


def rollout(actor, environment_config: dict, seed: int, capture: bool = False):
    env = make_combat_environment(environment_config)
    observation, _ = env.reset(seed)
    returns = np.zeros(4, dtype=np.float64)
    tracks = {(side, index): [] for side in ("red", "blue") for index in range(4)}

    def append(step: int, side: str, states, include):
        if not capture:
            return
        for index, (state, keep) in enumerate(zip(states, include)):
            if keep:
                tracks[(side, index)].append({
                    "step": step, "x_m": float(state.x), "y_m": float(state.y),
                    "altitude_m": float(state.altitude), "alive": bool(state.alive),
                })

    append(0, "red", env.red, [True] * 4)
    append(0, "blue", env.blue, [True] * 4)
    while True:
        red_before = [state.alive for state in env.red]
        blue_before = [state.alive for state in env.blue]
        actions = actor.act(observation, env.red_alive_mask, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(actions)
        returns += reward
        append(env.steps, "red", env.red, red_before)
        append(env.steps, "blue", env.blue, blue_before)
        if terminated or truncated:
            break
    summary = {
        "seed": seed, "team_return": float(returns.sum()),
        "mean_agent_return": float(returns.mean()),
        "termination_reason": info["termination_reason"],
        "episode_length": int(info["episode_length"]),
        "red_losses": int(info["red_losses"]), "blue_losses": int(info["blue_losses"]),
        "red_attack_kills": int(info["red_attack_kills"]),
        "blue_attack_kills": int(info["blue_attack_kills"]),
        "red_boundary_exits": int(info["red_boundary_exits"]),
        "red_ground_losses": int(info["red_ground_losses"]),
        "episode_r1_total": float(info["episode_r1_total"]),
        "episode_r2_total": float(info["episode_r2_total"]),
        "episode_r3_total": float(info["episode_r3_total"]),
        "episode_r4_total": float(info["episode_r4_total"]),
    }
    return summary, tracks


def shared_representative(results: dict[str, list[dict]]) -> int:
    common = sorted(set(
        row["seed"] for row in results["madsac"]
        if row["termination_reason"] == "red_win"
    ) & set(
        row["seed"] for row in results["mappo"]
        if row["termination_reason"] == "red_win"
    ))
    candidates = common or sorted(set(row["seed"] for row in results["madsac"]))
    by_algorithm = {
        algorithm: {row["seed"]: row for row in rows}
        for algorithm, rows in results.items()
    }
    medians = {
        algorithm: statistics.median(by_algorithm[algorithm][seed]["team_return"]
                                     for seed in candidates)
        for algorithm in results
    }
    scales = {
        algorithm: max(statistics.pstdev(
            by_algorithm[algorithm][seed]["team_return"] for seed in candidates
        ), 1.0) for algorithm in results
    }
    return min(candidates, key=lambda seed: sum(
        abs(by_algorithm[algorithm][seed]["team_return"] - medians[algorithm])
        / scales[algorithm] for algorithm in results
    ))


def plot_tracks(algorithm: str, checkpoint: Path, seed: int, summary: dict,
                tracks: dict, output: Path) -> None:
    figure = plt.figure(figsize=(11, 8.5), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    red_colors = ["#7f0000", "#b2182b", "#d6604d", "#f4a582"]
    blue_colors = ["#053061", "#2166ac", "#4393c3", "#92c5de"]
    all_points = []
    for side, colors, linestyle in (("red", red_colors, "-"), ("blue", blue_colors, "--")):
        for index in range(4):
            points = tracks[(side, index)]
            if not points:
                continue
            xyz = np.asarray([[p["x_m"], p["y_m"], p["altitude_m"]] for p in points]) / 1000.0
            all_points.append(xyz)
            axis.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], linestyle,
                      color=colors[index], linewidth=2.0,
                      label=f"{side.capitalize()} {index + 1}")
            axis.scatter(*xyz[0], color=colors[index], marker="o", s=28)
            end_marker = "X" if not points[-1]["alive"] else "^"
            axis.scatter(*xyz[-1], color=colors[index], marker=end_marker, s=55)
    angle = np.linspace(0.0, 2.0 * np.pi, 240)
    axis.plot(5.0 * np.cos(angle), 5.0 * np.sin(angle), np.zeros_like(angle),
              color="#666666", linewidth=1.0, alpha=0.65, label="5 km arena")
    points = np.concatenate(all_points)
    x_range = max(np.ptp(points[:, 0]), 1.0)
    y_range = max(np.ptp(points[:, 1]), 1.0)
    z_range = max(np.ptp(points[:, 2]), 1.0)
    axis.set_box_aspect((x_range, y_range, z_range * 1.4))
    axis.set_xlabel("North x (km)")
    axis.set_ylabel("East y (km)")
    axis.set_zlabel("Altitude (km)")
    axis.view_init(elev=25, azim=-56)
    axis.set_title(
        f"{algorithm.upper()} best saved checkpoint: deterministic 4v4 trajectory\n"
        f"seed={seed} | {summary['termination_reason']} | return={summary['team_return']:.2f} "
        f"| Red loss={summary['red_losses']} | Blue loss={summary['blue_losses']} "
        f"| steps={summary['episode_length']}"
    )
    axis.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), ncol=2, fontsize=8)
    axis.grid(True, alpha=0.25)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--madsac-checkpoint", type=Path, required=True)
    parser.add_argument("--mappo-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed-base", type=int, default=10_000_000)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--selected-seed", type=int)
    parser.add_argument("--env-config", default="configs/combat_environment.yaml")
    args = parser.parse_args()
    environment_path = Path(args.env_config)
    if not environment_path.is_absolute():
        environment_path = ROOT / environment_path
    environment = yaml.safe_load(environment_path.read_text(encoding="utf-8"))
    actors = {
        "madsac": build_madsac(args.madsac_checkpoint, args.device),
        "mappo": build_mappo(args.mappo_checkpoint, args.device),
    }
    checkpoints = {"madsac": args.madsac_checkpoint, "mappo": args.mappo_checkpoint}
    results = {algorithm: [rollout(actor, environment, seed)[0]
              for seed in range(args.seed_base, args.seed_base + args.episodes)]
              for algorithm, actor in actors.items()}
    seed = (
        int(args.selected_seed)
        if args.selected_seed is not None
        else shared_representative(results)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection = (
        "explicit selected seed"
        if args.selected_seed is not None
        else "shared winning seed closest to both models' median winning return"
    )
    report = {"selection": selection,
              "selected_seed": seed, "evaluation_results": results, "selected": {}}
    for algorithm, actor in actors.items():
        summary, tracks = rollout(actor, environment, seed, capture=True)
        checkpoint = checkpoints[algorithm]
        stem = f"{algorithm}_best_checkpoint_3d_trajectory_seed_{seed}"
        png = args.output_dir / f"{stem}.png"
        csv_path = args.output_dir / f"{stem}.csv"
        plot_tracks(algorithm, checkpoint, seed, summary, tracks, png)
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=(
                "side", "aircraft", "step", "x_m", "y_m", "altitude_m", "alive"
            ))
            writer.writeheader()
            for (side, index), points in tracks.items():
                for point in points:
                    writer.writerow({"side": side, "aircraft": index + 1, **point})
        report["selected"][algorithm] = {
            "checkpoint": str(checkpoint), "summary": summary,
            "png": str(png), "csv": str(csv_path),
        }
    (args.output_dir / "best_model_trajectory_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["selected"], indent=2))


if __name__ == "__main__":
    main()
