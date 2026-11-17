"""Export an independent 20v20 episode trace as an MP4 replay."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from algorithm.ideal_20v20_nearest_guidance import TRACE_VERSION
from env.ideal_interception.large_scale_20v20_env import PROTOCOL_20V20


def render(source: Path, output: Path, fps: int = 20):
    if fps <= 0:
        raise ValueError("fps must be positive")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required for MP4 export")
    with source.open(encoding="utf-8") as stream:
        trace = json.load(stream)
    if (trace.get("trace_version") != TRACE_VERSION or
            trace.get("environment_protocol") != PROTOCOL_20V20):
        raise ValueError("expected an independent 20v20 episode trace")
    frames = trace["frames"]
    if len(frames) < 2 or any(len(f["defenders"]) != 20 or
                              len(f["intruders"]) != 20 for f in frames):
        raise ValueError("invalid 20v20 frame sequence")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = trace["config"]
    lower, upper = config["lower"], config["upper"]
    fig = plt.figure(figsize=(11, 8), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]),
           zlim=(lower[2], upper[2]), xlabel="X (m)", ylabel="Y (m)",
           zlabel="Z (m)")
    ax.set_box_aspect((upper[0]-lower[0], upper[1]-lower[1],
                       upper[2]-lower[2]), zoom=.85)
    defenders = ax.scatter([], [], [], c="#d64040", s=28, label="Defenders")
    intruders = ax.scatter([], [], [], c="#2378bf", s=28, label="Intruders")
    ax.legend(loc="upper right")
    title = ax.set_title("")
    ax.view_init(elev=25, azim=-65)

    times = np.asarray([f["time"] for f in frames], dtype=float)
    if times[0] != 0 or not np.all(np.diff(times) > 0):
        raise ValueError("invalid trace times")
    duration = float(times[-1])
    sample_times = np.arange(0, duration, 1/fps)
    sample_times = np.append(sample_times, duration)
    with tempfile.TemporaryDirectory(prefix="ideal_20v20_replay_") as folder:
        folder_path = Path(folder)
        for number, t in enumerate(sample_times):
            frame = frames[min(np.searchsorted(times, t, side="right")-1,
                               len(frames)-1)]
            for artist, key, mask_key in ((defenders, "defenders", "defender_mask"),
                                          (intruders, "intruders", "intruder_mask")):
                points = np.asarray(frame[key], dtype=float)[
                    np.asarray(frame[mask_key], dtype=bool), :3]
                artist._offsets3d = (points[:, 0], points[:, 1], points[:, 2])
            title.set_text(f"20v20  |  t={t:.2f} s  |  {frame['outcome']}")
            fig.savefig(folder_path/f"frame_{number:05d}.png")
        plt.close(fig)
        temporary_video = folder_path/"replay.mp4"
        encoders = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                                  capture_output=True, text=True, check=True).stdout
        codec = "libx264" if " libx264 " in encoders else (
            "libopenh264" if " libopenh264 " in encoders else "mpeg4")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-framerate", str(fps), "-i", str(folder_path/"frame_%05d.png"),
                        "-c:v", codec, "-pix_fmt", "yuv420p", str(temporary_video)],
                       check=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(temporary_video), output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args(argv)
    if args.output.suffix.lower() != ".mp4":
        parser.error("output must end in .mp4")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    render(args.trace, args.output, args.fps)


if __name__ == "__main__":
    main()
