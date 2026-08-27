"""Plot trajectory CSV produced by audit_persistent_wave_behavior.py."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    colors = {"red": "#b2182b", "blue": "#2166ac"}
    for category in sorted({row["category"] for row in rows}):
        selected = [row for row in rows if row["category"] == category]
        seed = selected[0]["seed"]
        fig, (xy, altitude) = plt.subplots(
            1, 2, figsize=(13, 5.5), constrained_layout=True
        )
        groups = sorted({
            (row["side"], int(row["aircraft"]), int(row["wave_index"]))
            for row in selected
        })
        for side, aircraft, wave in groups:
            points = [row for row in selected
                      if row["side"] == side
                      and int(row["aircraft"]) == aircraft
                      and int(row["wave_index"]) == wave
                      and row["alive"].lower() == "true"]
            if not points:
                continue
            x = np.asarray([float(row["x"]) for row in points]) / 1000.0
            y = np.asarray([float(row["y"]) for row in points]) / 1000.0
            t = np.asarray([float(row["step"]) for row in points]) * 0.1
            z = np.asarray([float(row["altitude"]) for row in points]) / 1000.0
            xy.plot(x, y, color=colors[side], alpha=0.55, linewidth=1.0,
                    label=f"{side[0].upper()}{aircraft + 1} W{wave}")
            altitude.plot(t, z, color=colors[side], alpha=0.55, linewidth=1.0)
            for index, row in enumerate(points):
                if row["event"]:
                    marker = "X" if "loss" in row["event"] or "kill" in row["event"] else "o"
                    xy.scatter(x[index], y[index], color=colors[side], marker=marker, s=34)
                    altitude.scatter(t[index], z[index], color=colors[side], marker=marker, s=34)
        angle = np.linspace(0.0, 2.0 * np.pi, 240)
        xy.plot(5.0 * np.cos(angle), 5.0 * np.sin(angle), color="black", linewidth=0.8)
        xy.set_aspect("equal"); xy.set_xlabel("x (km)"); xy.set_ylabel("y (km)")
        altitude.set_xlabel("time (s)"); altitude.set_ylabel("altitude (km)")
        fig.suptitle(f"{category} | seed={seed}")
        fig.savefig(args.output_dir / f"{category}_seed_{seed}.png", dpi=170)
        plt.close(fig)


if __name__ == "__main__":
    main()
