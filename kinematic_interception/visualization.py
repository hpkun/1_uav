"""Render independent recorded trajectories; defenders red, intruders blue."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_trace(trace, destination):
    frames = trace["frames"]
    if not frames:
        raise ValueError("trace has no frames")
    defenders = np.asarray([frame["defenders"] for frame in frames],float)
    intruders = np.asarray([frame["intruders"] for frame in frames],float)
    if defenders.ndim != 3 or intruders.ndim != 3:
        raise ValueError("invalid 3D trajectory trace")
    figure = plt.figure(figsize=(9,7))
    axes = figure.add_subplot(111,projection="3d")
    for i in range(defenders.shape[1]):
        xyz = defenders[:,i,:3]
        axes.plot(*xyz.T,color="tab:red",alpha=.8,linewidth=1.1,
                  label="Defender" if i == 0 else None)
        axes.scatter(*xyz[0],color="tab:red",s=13)
    for j in range(intruders.shape[1]):
        xyz = intruders[:,j,:3]
        axes.plot(*xyz.T,color="tab:blue",alpha=.8,linewidth=1.1,
                  label="Intruder" if j == 0 else None)
        axes.scatter(*xyz[0],color="tab:blue",s=13)
    axes.set(xlabel="X (m)",ylabel="Y (m)",zlabel="Z (m)",
             title=f"{trace['summary']['model']} / {trace['summary']['method']} / "
                   f"{trace['summary']['scene_id'][:20]}")
    axes.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(destination,dpi=160)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args(argv)
    with args.trace.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    plot_trace(payload.get("trace",payload),args.output)


if __name__ == "__main__":
    main()
