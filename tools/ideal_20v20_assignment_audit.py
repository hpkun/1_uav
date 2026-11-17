"""Write-once visual audit of nominal 20v20 distance-only Hungarian assignment."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from env.ideal_interception.large_scale_20v20 import build_audit


CSV_COLUMNS = ("defender", "intruder", "defender_x", "defender_y", "defender_z",
               "intruder_x", "intruder_y", "intruder_z", "distance_m")


def write_csv(path, pairs):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for pair in pairs:
            writer.writerow(dict(defender=pair["defender"], intruder=pair["intruder"],
                **{f"defender_{axis}": pair["defender_position"][index]
                   for index, axis in enumerate("xyz")},
                **{f"intruder_{axis}": pair["intruder_position"][index]
                   for index, axis in enumerate("xyz")},
                distance_m=pair["distance_m"]))


def write_figures(out, summary):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    defenders = np.asarray(summary["defender_positions"])
    intruders = np.asarray(summary["intruder_positions"])
    assignment = {int(i): j for i, j in summary["assignment"].items()}

    fig, ax = plt.subplots(figsize=(13, 8), layout="constrained")
    for i, j in sorted(assignment.items()):
        ax.plot((defenders[i, 0], intruders[j, 0]),
                (defenders[i, 1], intruders[j, 1]), color="#7a7a7a", lw=.8,
                alpha=.65, zorder=1,
                label="Distance-Hungarian assignment" if i == 0 else "_nolegend_")
    ax.scatter(defenders[:, 0], defenders[:, 1], s=48, c="#d64040", marker="o",
               label="Defender", zorder=3)
    ax.scatter(intruders[:, 0], intruders[:, 1], s=55, c="#2378bf", marker="^",
               label="Intruder", zorder=3)
    ax.add_patch(Circle(summary["protected_center"][:2],
                        summary["protected_radius_m"], fill=False,
                        edgecolor="#2b8c5b", lw=2, label="Protected area", zorder=2))
    for i, point in enumerate(defenders):
        ax.annotate(f"D{i}", point[:2], xytext=(4, 4), textcoords="offset points",
                    color="#8c2020", fontsize=8)
    for j, point in enumerate(intruders):
        ax.annotate(f"I{j}", point[:2], xytext=(4, 4), textcoords="offset points",
                    color="#164f80", fontsize=8)
    ax.set(title="20v20 Initial Distance-Hungarian Assignment", xlabel="X (m)",
           ylabel="Y (m)", xlim=(-260, 310), ylim=(-185, 185))
    ax.set_aspect("equal")
    ax.grid(alpha=.2)
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(out/"assignment_xy.png", dpi=180)
    plt.close(fig)

    distances = np.linalg.norm(defenders[:, None, :]-intruders[None, :, :], axis=2)
    fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
    image = ax.imshow(distances, origin="upper", cmap="viridis", aspect="equal")
    for i, j in sorted(assignment.items()):
        ax.add_patch(Rectangle((j-.48, i-.48), .96, .96, fill=False,
                               edgecolor="#ef4444", linewidth=1.8))
    ax.set_xticks(range(len(intruders)), [f"I{j}" for j in range(len(intruders))],
                  rotation=90, fontsize=7)
    ax.set_yticks(range(len(defenders)), [f"D{i}" for i in range(len(defenders))],
                  fontsize=7)
    ax.set(xlabel="Intruder", ylabel="Defender",
           title="Initial 3D distance matrix (m); red boxes mark assignments")
    fig.colorbar(image, ax=ax, label="Initial 3D distance (m)")
    fig.savefig(out/"distance_matrix.png", dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error(f"output directory already exists: {args.out}")
    summary = build_audit()
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out/"summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=False)
    write_csv(args.out/"assignment.csv", summary["pairs"])
    write_figures(args.out, summary)
    stats = summary["statistics"]
    print("20v20 distance assignment")
    print(f"pairs={len(summary['pairs'])}")
    print(f"total_distance={stats['total_distance_m']:.3f} m")
    print(f"mean_distance={stats['mean_distance_m']:.3f} m")
    print(f"min_distance={stats['minimum_distance_m']:.3f} m")
    print(f"max_distance={stats['maximum_distance_m']:.3f} m")
    print(f"crossings={stats['assignment_line_crossings_xy']}")
    print(f"y_inversions={stats['monotonic_y_inversions']}")
    print(f"out={args.out}")


if __name__ == "__main__":
    main()
