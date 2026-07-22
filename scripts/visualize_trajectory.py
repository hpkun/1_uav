"""Run one complete episode and save five separate trajectory figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from run_1v1_episode import run_episode


def _save_line_plot(
    times: list[float],
    red_values: list[float],
    blue_values: list[float],
    ylabel: str,
    path: Path,
) -> None:
    fig = plt.figure()
    plt.plot(times, red_values, label="red")
    plt.plot(times, blue_values, label="blue")
    plt.xlabel("time (s)")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    """Save 3D, top-view, altitude, speed, and health figures."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["tail_chase", "head_on", "balanced_random"], default="tail_chase")
    parser.add_argument("--opponent", choices=["straight", "random", "pursuit"], default="straight")
    parser.add_argument("--red-policy", choices=["straight", "random", "pursuit"], default="pursuit")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    env, _ = run_episode(args.scenario, args.opponent, args.seed, args.red_policy)
    trajectory = env.get_trajectory()
    output = Path("outputs/trajectories")
    output.mkdir(parents=True, exist_ok=True)
    times = [float(record["simulation_time"]) for record in trajectory]
    red = [record["red_state"] for record in trajectory]
    blue = [record["blue_state"] for record in trajectory]

    fig = plt.figure()
    axis = fig.add_subplot(111, projection="3d")
    axis.plot([s.x for s in red], [s.y for s in red], [s.z for s in red], label="red")
    axis.plot([s.x for s in blue], [s.y for s in blue], [s.z for s in blue], label="blue")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_zlabel("z (m)")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "trajectory_3d.png", dpi=160)
    plt.close(fig)

    fig = plt.figure()
    plt.plot([s.x for s in red], [s.y for s in red], label="red")
    plt.plot([s.x for s in blue], [s.y for s in blue], label="blue")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    fig.savefig(output / "trajectory_xy.png", dpi=160)
    plt.close(fig)

    _save_line_plot(times, [s.z for s in red], [s.z for s in blue], "altitude (m)", output / "altitude.png")
    _save_line_plot(times, [s.speed for s in red], [s.speed for s in blue], "speed (m/s)", output / "speed.png")
    _save_line_plot(times, [s.health for s in red], [s.health for s in blue], "health", output / "health.png")
    print(f"Saved figures to {output.resolve()}")


if __name__ == "__main__":
    main()
