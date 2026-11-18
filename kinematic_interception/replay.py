"""Render a recorded A/B episode as an MP4 with 3D and XY views."""
import argparse
from bisect import bisect_right
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np

from . import PROTOCOL


def _validated_trace(payload):
    trace = payload.get("trace", payload)
    summary = trace["summary"]
    if summary.get("protocol") != PROTOCOL:
        raise ValueError("not an A/B kinematic interception trace")
    frames = trace["frames"]
    times = np.asarray([frame["time"] for frame in frames],float)
    if (len(times) < 2 or times[0] != 0 or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError("invalid episode clock")
    defenders = np.asarray([frame["defenders"] for frame in frames],float)
    intruders = np.asarray([frame["intruders"] for frame in frames],float)
    if (defenders.shape != (len(times),summary["defender_count"],6)
            or intruders.shape != (len(times),summary["intruder_count"],6)
            or not np.isfinite(defenders).all() or not np.isfinite(intruders).all()):
        raise ValueError("invalid recorded member states")
    if len(trace["references"]) != summary["intruder_count"]:
        raise ValueError("reference count mismatch")
    for index, reference in enumerate(trace["references"]):
        points = np.asarray(reference["points"],float)
        digest = hashlib.sha256(points.tobytes()).hexdigest()
        if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
                or digest != reference["specification"]["polyline_sha256"]
                or digest != summary["path_hashes"][index]):
            raise ValueError("recorded target reference hash mismatch")
    for event in trace["events"]:
        if not 0 <= event["time"] <= times[-1]:
            raise ValueError("event outside episode")
    return trace,times,defenders,intruders


def render_mp4(source, output, fps=10):
    if fps <= 0 or fps > 60:
        raise ValueError("FPS must be between 1 and 60")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg is required for MP4 rendering")
    encoders = subprocess.check_output(["ffmpeg","-hide_banner","-encoders"],text=True)
    available = {line.split()[1] for line in encoders.splitlines()
                 if line.startswith(" V") and len(line.split()) > 2}
    codec = next((name for name in ("libx264","libopenh264","mpeg4")
                  if name in available),None)
    if codec is None:
        raise RuntimeError("FFmpeg needs an H.264 or MPEG-4 video encoder")
    source,output = Path(source),Path(output)
    if output.exists():
        raise FileExistsError(output)
    with source.open(encoding="utf-8") as stream:
        trace,times,defenders,intruders = _validated_trace(json.load(stream))
    summary,scene = trace["summary"],trace["scene"]
    count_d,count_i = summary["defender_count"],summary["intruder_count"]
    duration = float(times[-1])
    all_positions = np.concatenate((defenders[:,:,:3],intruders[:,:,:3]),axis=1)
    masks = np.concatenate((np.asarray([f["defender_mask"] for f in trace["frames"]],bool),
                            np.asarray([f["intruder_mask"] for f in trace["frames"]],bool)),axis=1)
    exit_at = np.full(count_d+count_i,np.inf)
    for event in trace["events"]:
        for member in event["members"]:
            if member < len(exit_at):
                exit_at[member] = min(exit_at[member],event["time"])
    figure = plt.figure(figsize=(14.4,7.2),dpi=90)
    ax3 = figure.add_subplot(121,projection="3d")
    ax2 = figure.add_subplot(122)
    lower,upper = np.asarray(scene["lower"]),np.asarray(scene["upper"])
    ax3.set(xlim=(lower[0],upper[0]),ylim=(lower[1],upper[1]),
            zlim=(lower[2],upper[2]),xlabel="X (m)",ylabel="Y (m)",zlabel="Z (m)")
    ax3.set_box_aspect(upper-lower)
    ax3.view_init(elev=24,azim=-62)
    ax2.set(xlim=(lower[0],upper[0]),ylim=(lower[1],upper[1]),
            xlabel="X (m)",ylabel="Y (m)")
    ax2.set_aspect("equal",adjustable="box")
    center = np.asarray(scene["protected_center"])
    radius = scene["protected_radius"]
    theta = np.linspace(0,2*np.pi,64)
    z = np.linspace(center[2],center[2]+scene["protected_height"],6)
    mesh_t,mesh_z = np.meshgrid(theta,z)
    ax3.plot_surface(center[0]+radius*np.cos(mesh_t),
                     center[1]+radius*np.sin(mesh_t),mesh_z,
                     color="#58a66b",alpha=.10,shade=False)
    ax2.plot(center[0]+radius*np.cos(theta),center[1]+radius*np.sin(theta),
             color="#58a66b",linewidth=1.4,label="Protected region")
    for reference in trace["references"]:
        points = np.asarray(reference["points"],float)
        ax3.plot(*points.T,color="#2378bf",alpha=.15,linewidth=.7)
        ax2.plot(points[:,0],points[:,1],color="#2378bf",alpha=.15,linewidth=.7)
    artists = []
    for member in range(count_d+count_i):
        color = "#d64040" if member < count_d else "#2378bf"
        label = "Red Defender" if member == 0 else "Blue Intruder" if member == count_d else None
        line3, = ax3.plot([],[],[],color=color,linewidth=1.5,alpha=.8,label=label)
        dot3, = ax3.plot([],[],[],marker="o",linestyle="none",color=color,markersize=4)
        line2, = ax2.plot([],[],color=color,linewidth=1.5,alpha=.8)
        dot2, = ax2.plot([],[],marker="o",linestyle="none",color=color,markersize=4)
        artists.append((line3,dot3,line2,dot2))
    ax3.legend(loc="upper left",fontsize=8)
    event_points=[]
    for event in trace["events"]:
        if event["positions"]:
            xyz=np.mean(np.asarray(event["positions"],float),axis=0)
            mark3,=ax3.plot([xyz[0]],[xyz[1]],[xyz[2]],marker="x",linestyle="none",
                            color="#e69a23",markersize=5,visible=False)
            mark2,=ax2.plot([xyz[0]],[xyz[1]],marker="x",linestyle="none",
                            color="#e69a23",markersize=5,visible=False)
            event_points.append((event["time"],mark3,mark2))
    figure.subplots_adjust(left=.04,right=.98,bottom=.08,top=.9,wspace=.08)
    figure.suptitle(f"{count_d}v{count_i} | Model {summary['model']} | "
                    f"{summary['method']} + {summary['avoidance_mode']} | "
                    "Red Defender / Blue Intruder",fontsize=13)
    video_times = np.linspace(0,duration,max(2,int(np.ceil(duration*fps))+1))
    matplotlib.rcParams["animation.ffmpeg_path"] = shutil.which("ffmpeg")
    try:
        encoder = FFMpegWriter(fps=fps,codec=codec,extra_args=["-pix_fmt","yuv420p"])
        with encoder.saving(figure,str(output),figure.dpi) as writer:
            for now in video_times:
                index = min(bisect_right(times,now)-1,len(times)-2)
                later = index+1
                weight = float(np.clip((now-times[index])/(times[later]-times[index]),0.,1.))
                current = all_positions[index]*(1-weight)+all_positions[later]*weight
                for member,(line3,dot3,line2,dot2) in enumerate(artists):
                    trail = np.vstack((all_positions[:index+1,member],current[member]))
                    line3.set_data_3d(*trail.T)
                    line2.set_data(trail[:,0],trail[:,1])
                    xyz = current[member]
                    dot3.set_data_3d([xyz[0]],[xyz[1]],[xyz[2]])
                    dot2.set_data([xyz[0]],[xyz[1]])
                    active = bool(masks[index,member] and now < exit_at[member])
                    dot3.set_visible(active)
                    dot2.set_visible(active)
                for event_time,mark3,mark2 in event_points:
                    mark3.set_visible(event_time <= now)
                    mark2.set_visible(event_time <= now)
                assigned = sum(e["kind"] == "interception_contact" and e["time"] <= now
                               for e in trace["events"])
                breached = sum(e["kind"] == "region_breach" and e["time"] <= now
                                for e in trace["events"])
                ax2.set_title(f"t={now:.2f} s  |  assigned={assigned}/{count_i}  |  breach={breached}")
                writer.grab_frame()
    finally:
        plt.close(figure)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("MP4 rendering produced no video")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--fps",type=int,default=10)
    args = parser.parse_args(argv)
    render_mp4(args.trace,args.output,args.fps)
    print(f"[replay] file={args.output} | fps={args.fps}")


if __name__ == "__main__":
    main()
