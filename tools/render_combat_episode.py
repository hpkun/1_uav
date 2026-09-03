"""Headless Matplotlib renderer for a recorded combat trace."""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, writers
import numpy as np

from tools.combat_visualization import read_trace


def _frame_artists(ax, trace, frame, trail_length, heading):
    ax.collections.clear() if False else None
    artists = []
    start = max(0, frame - trail_length) if trail_length else 0
    for i in range(4):
        red = trace["red_kinematics"][start:frame + 1, i]
        valid = np.isfinite(red[:, 0])
        if valid.any():
            artists.append(ax.plot(red[valid, 0], red[valid, 1], -red[valid, 2], color="crimson", lw=1.2)[0])
        if trace["red_alive"][frame, i]:
            state = trace["red_kinematics"][frame, i]
            artists.append(ax.scatter([state[0]], [state[1]], [-state[2]], color="crimson", s=28, label="Red" if i == 0 else None))
            if heading:
                theta, psi = state[4], state[5]
                artists.append(ax.quiver(state[0], state[1], -state[2], np.cos(theta)*np.cos(psi), np.cos(theta)*np.sin(psi), np.sin(theta), color="crimson", length=120, normalize=True))
    styles = ["-", "--", ":"]
    for wave in range(trace["blue_kinematics"].shape[1]):
        for i in range(4):
            blue = trace["blue_kinematics"][start:frame + 1, wave, i]
            valid = np.isfinite(blue[:, 0])
            if valid.any():
                artists.append(ax.plot(blue[valid, 0], blue[valid, 1], -blue[valid, 2], color="royalblue", ls=styles[min(wave, 2)], lw=1.0)[0])
            if trace["blue_alive"][frame, wave, i]:
                state = trace["blue_kinematics"][frame, wave, i]
                artists.append(ax.scatter([state[0]], [state[1]], [-state[2]], color="royalblue", s=25, label="Blue" if wave == 0 and i == 0 else None))
                if heading:
                    theta, psi = state[4], state[5]
                    artists.append(ax.quiver(state[0], state[1], -state[2], np.cos(theta)*np.cos(psi), np.cos(theta)*np.sin(psi), np.sin(theta), color="royalblue", length=120, normalize=True))
    return artists


def render(trace_path: Path, metadata_path: Path, preview_path: Path, mp4_path: Path | None, fps=30, stride=4, dpi=100, elev=25, azim=-55, trail_length=0, heading=True, preview_frame=None):
    if not trace_path.is_file():
        raise FileNotFoundError(f"trace not found: {trace_path}; run record_combat_episode.py successfully first")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata not found: {metadata_path}; run record_combat_episode.py successfully first")
    trace = read_trace(trace_path); metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata.get("trace_schema_version", -1)) != 1: raise ValueError("unsupported trace schema")
    fig = plt.figure(figsize=(12.8, 7.2), dpi=dpi); ax = fig.add_subplot(111, projection="3d")
    radius = float(metadata["arena_radius"]); ax.set_xlim(-radius, radius); ax.set_ylim(-radius, radius)
    altitude = -trace["red_kinematics"][..., 2].tolist() if False else -trace["red_kinematics"][..., 2]
    b_altitude = -trace["blue_kinematics"][..., 2]
    valid = np.concatenate([altitude[np.isfinite(altitude)], b_altitude[np.isfinite(b_altitude)]])
    upper = max(500.0, float(np.nanmax(valid) + 100.0)) if valid.size else 500.0
    ax.set_zlim(0, upper); ax.set_xlabel("X / m"); ax.set_ylabel("Y / m"); ax.set_zlabel("Altitude / m"); ax.view_init(elev=elev, azim=azim)
    circle = np.linspace(0, 2*np.pi, 128); ax.plot(radius*np.cos(circle), radius*np.sin(circle), np.zeros_like(circle), color="lightgray", lw=.7)
    overlay = fig.text(.02, .94, "", va="top", family="monospace")
    frames = list(range(0, len(trace["steps"]), max(1, stride))); frames = frames if frames[-1] == len(trace["steps"])-1 else frames + [len(trace["steps"])-1]
    def draw(index):
        for artist in list(ax.lines[1:]) + list(ax.collections) + list(ax.artists):
            try: artist.remove()
            except (NotImplementedError, ValueError): pass
        frame = frames[index]; artists = _frame_artists(ax, trace, frame, trail_length, heading)
        if index == 0 or index == len(frames) - 1 or index % 25 == 0:
            print(f"[RENDER] frame {index + 1}/{len(frames)} (trace step {int(frame)})", flush=True)
        overlay.set_text(f"EA-WB-MAPPO | seed={metadata['episode_seed']}\nSim Time: {trace['time_s'][frame]:.1f} s   Wave: {trace['active_wave'][frame]} / {metadata['total_waves']}\nWaves Cleared: {trace['waves_cleared'][frame]}   Red Alive: {int(trace['red_alive'][frame].sum())} / 4   Blue Alive: {int(trace['blue_alive'][frame, trace['active_wave'][frame]-1].sum())} / 4")
        return artists + [overlay]
    target = frames.index(preview_frame) if preview_frame in frames else (len(frames)-1 if preview_frame is None else min(range(len(frames)), key=lambda i: abs(frames[i]-preview_frame)))
    draw(target); fig.savefig(preview_path, dpi=dpi, bbox_inches="tight")
    if mp4_path is not None:
        if not writers.is_available("ffmpeg"): raise RuntimeError("ffmpeg is required for MP4 rendering")
        animation = FuncAnimation(fig, draw, frames=len(frames), interval=1000/fps, blit=False)
        animation.save(mp4_path, writer=FFMpegWriter(fps=fps), dpi=dpi)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--input-dir"); parser.add_argument("--trace"); parser.add_argument("--metadata"); parser.add_argument("--output"); parser.add_argument("--preview"); parser.add_argument("--fps", type=int, default=30); parser.add_argument("--frame-stride", type=int, default=4); parser.add_argument("--dpi", type=int, default=100); parser.add_argument("--elev", type=float, default=25); parser.add_argument("--azim", type=float, default=-55); parser.add_argument("--trail-length", type=int, default=0); parser.add_argument("--no-heading", action="store_true"); parser.add_argument("--preview-frame", type=int)
    args = parser.parse_args(); directory = Path(args.input_dir) if args.input_dir else None
    trace = Path(args.trace) if args.trace else directory / "episode_trace.npz"; metadata = Path(args.metadata) if args.metadata else directory / "metadata.json"
    output = Path(args.output) if args.output else (None if args.preview else (directory / "episode.mp4" if directory else Path("episode.mp4")))
    preview = Path(args.preview) if args.preview else (output.with_name("preview.png") if output else Path("preview.png"))
    render(trace, metadata, preview, output, args.fps, args.frame_stride, args.dpi, args.elev, args.azim, args.trail_length, not args.no_heading, args.preview_frame)


if __name__ == "__main__": main()
