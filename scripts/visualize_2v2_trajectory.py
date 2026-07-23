"""Save five separate plots for one complete homogeneous 2v2 episode."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from run_2v2_episode import run_2v2_episode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="head_on_formation")
    parser.add_argument("--opponent", default="straight")
    parser.add_argument("--red-policy", default="pursuit")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    env, _ = run_2v2_episode(args.scenario, args.opponent, args.seed, args.red_policy)
    trajectory = env.get_trajectory()
    output = Path("outputs/trajectories/2v2"); output.mkdir(parents=True, exist_ok=True)
    ids = [u.uav_id for u in env.all_aircraft]
    colors = {"red_0":"red", "red_1":"darkred", "blue_0":"blue", "blue_1":"navy"}
    times = [record["simulation_time"] for record in trajectory]
    for name, ylabel, getter in (("altitude", "altitude (m)", lambda s:s.z), ("speed", "speed (m/s)", lambda s:s.speed), ("health", "health", lambda s:s.health)):
        fig=plt.figure()
        for uid in ids: plt.plot(times, [getter(record["states"][uid]) for record in trajectory], label=uid, color=colors[uid])
        plt.xlabel("time (s)"); plt.ylabel(ylabel); plt.legend(); plt.tight_layout(); fig.savefig(output/f"{name}.png", dpi=160); plt.close(fig)
    fig=plt.figure(); axis=fig.add_subplot(111, projection="3d")
    for uid in ids:
        states=[record["states"][uid] for record in trajectory]; axis.plot([s.x for s in states],[s.y for s in states],[s.z for s in states],label=uid,color=colors[uid])
    axis.legend(); fig.tight_layout(); fig.savefig(output/"trajectory_3d.png",dpi=160); plt.close(fig)
    fig=plt.figure()
    for uid in ids:
        states=[record["states"][uid] for record in trajectory]; plt.plot([s.x for s in states],[s.y for s in states],label=uid,color=colors[uid])
    plt.axis("equal"); plt.legend(); plt.tight_layout(); fig.savefig(output/"trajectory_xy.png",dpi=160); plt.close(fig)
    print(f"Saved figures to {output.resolve()}")


if __name__ == "__main__":
    main()
