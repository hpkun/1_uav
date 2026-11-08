"""Export an offline Plotly player or a fixed-camera MP4 from saved trajectories."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from .interception_timeline import Timeline


PLAYER = r"""
const plot = document.getElementById('{plot_id}');
const {frames, keys, events, result} = DATA;
const slider = document.querySelector('#clock'), rate = document.querySelector('#rate');
const panel = document.querySelector('#details'), time = document.querySelector('#time');
const count = frames[0].entities.length;
const trails = frames[0].entities.map((_, j) =>
  [0,1,2].map(axis => frames.map(row => row.entities[j].position[axis])));
const indices = Array.from({length: count + 3}, (_, i) => i + plot.data.length - count - 3);
const nf = (x, digits=3) => x == null ? 'N/A' : x.toFixed(digits);
const vector = x => x == null ? 'N/A' : x.map(v => nf(v, 2)).join(', ');
const duration = frames.at(-1).time;
let wanted = 0, shown = -1, pending = false;
let cursor = 0, playing = false, origin = 0, started = 0, request;
slider.max = keys.length - 1;

function floorIndex(length, timeAt, t) {
  let low = 0, high = length;
  while (low+1 < high) {
    const middle = (low+high) >> 1;
    if (timeAt(middle) <= t) low = middle;
    else high = middle;
  }
  return low;
}
const keyAt = t => floorIndex(keys.length, k => frames[keys[k]].time, t);
const visibleEvents = row => events.filter(e => e.time <= row.time);
function arrowLines(entities) {
  const points = [];
  for (const e of entities) {
    if (!e.velocity || e.speed <= 1e-9) continue;
    const dir = e.velocity.map(v => v/e.speed);
    const horizontal = Math.hypot(dir[0],dir[1]);
    const side = horizontal > 1e-9 ? [-dir[1]/horizontal,dir[0]/horizontal,0] : [1,0,0];
    const length = Math.min(3,.35*e.speed);
    const tip = e.position.map((p,i) => p+dir[i]*length);
    const wing = sign => tip.map((p,i) => p+length*(-.3*dir[i]+sign*.16*side[i]));
    points.push(e.position, tip, null, wing(1), tip, wing(-1), null);
  }
  return points;
}
function diagnostics(row, visible) {
  const pair = row.nearest_pair;
  const pairs = row.pairs.map(p =>
    p.pursuer+'/'+p.evader+': '+nf(p.distance)+' m, '+nf(p.relative_speed)+' m/s');
  const dwell = row.last_recorded_dwell;
  const dwellText = dwell == null ? 'N/A' : row.entities.filter(e => e.name.startsWith('P')).map(p => {
    const values = row.entities.filter(e => e.name.startsWith('E')).map(e =>
      e.name+' '+nf(dwell[Number(p.name.slice(1))][Number(e.name.slice(1))])+' s');
    return p.name+': '+values.join(', ');
  }).join(' | ');
  const rewards = row.rewards == null ? 'N/A' : Object.entries(row.rewards)
    .map(([name, value]) => name+' '+nf(value)).join(' | ');
  panel.textContent = [
    'Final result: '+result+' | Current: '+row.outcome+' | State: '+row.state_basis,
    ...row.entities.map(e => e.name+' '+(e.captured ? 'captured / inactive' : e.active ? 'active' : 'inactive')+
      ' | position (m): '+vector(e.position)+' | physical velocity (m/s): '+vector(e.velocity)+
      ' | speed: '+nf(e.speed)+' m/s' +
      ' | normalized policy action: '+vector(e.action)+' | executed velocity command (m/s): '+vector(e.command)+
      (e.requested == null ? '' : ' | requested (m/s): '+vector(e.requested)+
       ' | step-start acceleration (m/s²): '+vector(e.initial_acceleration))),
    'Nearest active pair: '+(pair ? pair.pursuer+'/'+pair.evader+' · '+nf(pair.distance)+
       ' m · relative speed '+nf(pair.relative_speed)+' m/s' : 'N/A'),
    'Active pairs: '+(pairs.join(' | ') || 'N/A'),
    'Last recorded diagnostics at '+nf(row.recorded_time)+' s',
    'Minimum separation: '+nf(row.minimum)+' m'+(dwell == null ? '' : ' | Capture dwell: '+dwellText),
    'Minimum surface clearance: '+nf(row.minimum_surface_clearance)+' m',
    'Reward components: '+rewards,
    'Events: '+(visible.map(e => e.label).join(' | ') || 'none')
  ].join('\n');
}
function show(i) {
  wanted = i;
  if (!pending && wanted !== shown) draw();
}
async function draw() {
  pending = true;
  try {
    while (wanted !== shown) {
      const target = wanted;
      const row = frames[target], visible = visibleEvents(row);
      const current = row.entities.filter(e => !e.captured);
      const arrows = arrowLines(current);
      const xyz = [0,1,2].map(axis => [
        ...trails.map(trail => trail[axis].slice(0,target+1)),
        current.map(e => e.position[axis]),
        arrows.map(point => point?.[axis] ?? null),
        visible.map(e => e.position[axis])
      ]);
      const update = {x: xyz[0], y: xyz[1], z: xyz[2]};
      update.text = Array(count).fill(undefined).concat([
        current.map(e => e.name), undefined,
        visible.map(e => e.label + '<br>Recorded coordinates (m): ' + e.positions.map(vector).join(' / '))
      ]);
      update['marker.color'] = Array(count).fill(undefined).concat([
        current.map(e => e.active ? e.color : '#888888'),undefined,visible.map(e => e.color)
      ]);
      update['marker.symbol'] = Array(count+2).fill(undefined).concat([visible.map(e => e.plotly_marker)]);
      // Wheel motion can finish after Plotly saves its layout camera.
      // Carry the actual WebGL camera into the same update as the frame.
      const scene = plot._fullLayout.scene._scene;
      const camera = scene?.getCamera?.() ?? plot.layout.scene.camera;
      await Plotly.update(plot, update, {'scene.camera': camera}, indices);
      shown = target;
      // Commit the panel with the rendered scene. New scrub requests can
      // replace queued work while this one Plotly update is in flight.
      time.textContent = row.time.toFixed(3)+' s · step '+row.step+
        ' · keyframe '+(keyAt(row.time)+1)+'/'+keys.length;
      plot.dataset.frame = String(target);
      diagnostics(row, visible);
      if (playing || wanted === shown) slider.value = keyAt(row.time);
      slider.setAttribute('aria-valuetext', row.time.toFixed(3)+' seconds');
    }
  } finally { pending = false; }
}
const speed = () => Number(rate.value);
function tick(now) {
  if (!playing) return;
  cursor = Math.min(duration, origin + (now-started)*speed()/1000);
  show(floorIndex(frames.length, i => frames[i].time, cursor));
  if (cursor >= duration) playing = false;
  else request = requestAnimationFrame(tick);
}
function pause() {
  if (playing) cursor = Math.min(duration, origin + (performance.now()-started)*speed()/1000);
  playing = false;
  cancelAnimationFrame(request);
}
function seek(t) {
  pause();
  cursor = t;
  show(floorIndex(frames.length, i => frames[i].time, t));
}
function play() {
  if (playing || duration === 0) return;
  if (cursor >= duration) seek(0);
  origin = cursor;
  started = performance.now();
  playing = true;
  request = requestAnimationFrame(tick);
}
document.querySelector('#play').onclick = play;
document.querySelector('#pause').onclick = pause;
document.querySelector('#restart').onclick = () => { seek(0); play(); };
document.querySelector('#prev').onclick = () => seek(frames[keys[Math.max(0,keyAt(frames[wanted].time)-1)]].time);
document.querySelector('#next').onclick = () => seek(frames[keys[Math.min(keys.length-1,keyAt(frames[wanted].time)+1)]].time);
slider.oninput = () => seek(frames[keys[Number(slider.value)]].time);
let previousRate = speed();
rate.onchange = () => {
  if (playing) {
    const now = performance.now();
    origin = Math.min(duration, origin + (now-started)*previousRate/1000);
    started = now;
  }
  previousRate = speed();
};
show(0);
"""


def render(source, output):
    import plotly.graph_objects as go

    timeline = Timeline(source)
    times = sorted(set(timeline.key_times + [
        k / 30 for k in range(math.ceil(timeline.duration * 30)) if k / 30 < timeline.duration
    ]))
    samples = [timeline.at(timestamp) for timestamp in times]
    time_indices = {timestamp: i for i, timestamp in enumerate(times)}
    config = timeline.data["config"]
    center, radius = np.array(config["center"]), config["radius"]
    if config.get("protected_zone") == "ground_cylinder":
        theta, z = np.meshgrid(np.linspace(0, 2*np.pi, 32),
                               np.linspace(center[2], center[2] + config["protected_height"], 16))
        surface = np.array([center[0] + radius*np.cos(theta),
                            center[1] + radius*np.sin(theta), z])
    else:
        theta, phi = np.meshgrid(np.linspace(0, 2*np.pi, 32), np.linspace(0, np.pi, 16))
        surface = radius * np.array([np.cos(theta)*np.sin(phi),
                                    np.sin(theta)*np.sin(phi), np.cos(phi)])
        surface += center[:, None, None]
    figure = go.Figure(go.Surface(x=surface[0], y=surface[1], z=surface[2], opacity=.22,
                                  name="Protected zone", showscale=False, showlegend=True,
                                  colorscale=[[0, "#55a868"], [1, "#55a868"]]))
    if config.get("protected_zone") == "ground_cylinder":
        figure.add_trace(go.Scatter3d(x=[center[0]], y=[center[1]], z=[center[2]], mode="markers",
                                     name="Ground facility", marker=dict(size=7, color="#246d48", symbol="square")))
    for entity in samples[0]["entities"]:
        role = "Red defender" if entity["name"].startswith("P") else "Blue intruder"
        figure.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="lines", name=f"{role} {entity['name']}",
                                     line=dict(color=entity["color"], width=3)))
    figure.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="markers+text", textposition="top center",
                                 name="Current UAV", marker=dict(size=5)))
    figure.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="lines", name="Velocity direction",
                                 hoverinfo="skip", line=dict(color="#485864", width=3)))
    figure.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="markers", name="Events",
                                 hoverinfo="text", marker=dict(size=12, symbol="x")))
    bounds = timeline.bounds(samples)
    figure.update_layout(uirevision="keep-camera", height=640, margin=dict(t=30, b=0),
                         scene=dict(aspectmode="data", dragmode="orbit",
                                    camera=dict(eye=dict(x=2.2, y=2.2, z=1.7)),
                                    **{axis+"axis": dict(title=axis+" (m)", range=bounds[i])
                                                          for i, axis in enumerate("xyz")}))
    data = json.dumps(dict(frames=samples, keys=[time_indices[t] for t in timeline.key_times],
                           events=timeline.display_events, result=timeline.data["result"]["outcome"]),
                      separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    script = "const DATA = " + data + ";\n" + PLAYER
    graph = figure.to_html(full_html=False, include_plotlyjs=True, post_script=script,
                           config={"scrollZoom": True})
    controls = """
<h2>UAV interception replay</h2><div>
<button id="play">Play</button> <button id="pause">Pause</button>
<button id="restart">Replay</button> <button id="prev">Previous step</button>
<button id="next">Next step</button> <label>Speed <select id="rate">
<option value=".5">0.5×</option><option value="1" selected>1×</option>
<option value="2">2×</option><option value="4">4×</option></select></label>
<label>Timeline <input id="clock" type="range" min="0" step="1" value="0"></label>
<span id="time"></span></div><pre id="details"></pre>
"""
    with Path(output).open("x", encoding="utf-8") as stream:
        stream.write('<!doctype html><meta charset="utf-8"><title>Interception replay</title>'
                     '<style>body{font:15px system-ui;margin:16px;display:flex;flex-direction:column}'
                     '#details{order:1;white-space:pre-wrap}'
                     '#clock{width:min(36vw,420px)}</style>' + controls + graph)


def video_times(duration, fps):
    if not math.isfinite(duration) or duration < 0 or fps <= 0:
        raise ValueError("finite nonnegative duration and positive fps required")
    count = max(2, math.ceil(duration * fps)) if duration else 1
    return np.linspace(0., duration, count).tolist()


def render_video(source, output, fps=30, width=1280, height=720, elevation=25, azimuth=-55):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("FFmpeg required. Install with conda install -c conda-forge ffmpeg and retry.")
    encoders = subprocess.check_output([ffmpeg, "-hide_banner", "-encoders"], text=True).splitlines()
    names = {line.split()[1] for line in encoders if len(line.split()) > 2 and line.startswith(" V")}
    codec = next((name for name in ("libx264", "libopenh264", "mpeg4") if name in names), None)
    if codec is None:
        raise RuntimeError("Install an FFmpeg build with an H.264 or MPEG-4 video encoder.")
    if fps <= 0 or width < 2 or height < 2 or width % 2 or height % 2:
        raise ValueError("positive fps and even video dimensions required")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg
    timeline = Timeline(source)
    samples = [timeline.at(t) for t in video_times(timeline.duration, fps)]
    history = [timeline.at(t) for t in timeline.key_times]
    bounds = timeline.bounds(samples + history)
    config = timeline.data["config"]
    center, radius = config["center"], config["radius"]
    if config.get("protected_zone") == "ground_cylinder":
        theta, vertical = np.meshgrid(np.linspace(0, 2*np.pi, 28),
                                      np.linspace(center[2], center[2] + config["protected_height"], 14))
        surface = (center[0] + radius*np.cos(theta),
                   center[1] + radius*np.sin(theta), vertical)
    else:
        theta, phi = np.meshgrid(np.linspace(0, 2*np.pi, 28), np.linspace(0, np.pi, 14))
        surface = (center[0] + radius*np.cos(theta)*np.sin(phi),
                   center[1] + radius*np.sin(theta)*np.sin(phi),
                   center[2] + radius*np.cos(phi))
    figure = plt.figure(figsize=(width/100, height/100), dpi=100)
    axes = figure.add_subplot(projection="3d", computed_zorder=False)
    axes.view_init(elev=elevation, azim=azimuth)
    axes.set(xlim=bounds[0], ylim=bounds[1], zlim=bounds[2],
             xlabel="x (m)", ylabel="y (m)", zlabel="z (m)")
    axes.set_box_aspect(np.ptp(bounds, axis=1))
    axes.plot_surface(*surface, color="#55a868", alpha=.13, zorder=0)
    if config.get("protected_zone") == "ground_cylinder":
        axes.scatter(*center, color="#246d48", marker="s", label="Ground facility")
    paths, points, labels = [], [], []
    from matplotlib.lines import Line2D
    roles = {entity["name"][0]: entity["color"] for entity in samples[0]["entities"]}
    axes.legend(handles=[
        Line2D([], [], color=roles["P"], label="Red defenders (P)"),
        Line2D([], [], color=roles["E"], label="Blue intruders (E)"),
        Line2D([], [], color="#246d48", marker="s", linestyle="none", label="Facility / protected zone"),
        Line2D([], [], color="#1aa575", marker="x", linestyle="none", label="Contact interception"),
    ], loc="upper left", fontsize=8)
    for entity in samples[0]["entities"]:
        paths.append(axes.plot([], [], [], color=entity["color"], zorder=2)[0])
        points.append(axes.plot([], [], [], linestyle="none", marker="o",
                                color=entity["color"], zorder=4)[0])
        labels.append(axes.text(0, 0, 0, entity["name"], zorder=5))
    history_positions = np.array([[e["position"] for e in row["entities"]] for row in history])
    events = []
    for event in timeline.display_events:
        marker = axes.plot(*event["position"], linestyle="none", marker=event["mpl_marker"],
                           markerfacecolor="none" if event["kind"] == "evader_escape" else event["color"],
                           markersize=12, markeredgewidth=2.4, color=event["color"], zorder=10)[0]
        marker.set_visible(False)
        events.append(marker)
    title = axes.set_title("")
    event_text = figure.text(.02, .025, "", fontsize=9)
    arrows = []
    history_end = 0
    output = Path(output)
    reserved = False
    try:
        with output.open("xb"):
            pass
        reserved = True
        with tempfile.TemporaryDirectory(dir=output.parent) as folder:
            target = Path(folder) / "video.mp4"
            writer = FFMpegWriter(fps=fps, codec=codec, extra_args=["-pix_fmt", "yuv420p"])
            with writer.saving(figure, str(target), 100):
                for row in samples:
                    timestamp = row["time"]
                    while history_end < len(history) and history[history_end]["time"] < timestamp:
                        history_end += 1
                    for arrow in arrows:
                        arrow.remove()
                    arrows.clear()
                    for i, entity in enumerate(row["entities"]):
                        position = entity["position"]
                        path = np.vstack((history_positions[:history_end, i], position))
                        paths[i].set_data_3d(*path.T)
                        points[i].set_data_3d(*([v] for v in position))
                        points[i].set_visible(not entity["captured"])
                        labels[i].set_position_3d(position)
                        labels[i].set_visible(not entity["captured"])
                        if entity["velocity"] is not None and entity["speed"] > 1e-9:
                            arrows.append(axes.quiver(*position, *entity["velocity"], length=.3,
                                                     color=entity["color"], zorder=3))
                    visible = []
                    for event, marker in zip(timeline.display_events, events):
                        show = event["time"] <= timestamp
                        marker.set_visible(show)
                        if show:
                            visible.append(event["label"])
                    event_text.set_text(" | ".join(visible))
                    title.set_text(f"{timestamp:.6f} s | step {row['step']} | "
                                   f"{row['outcome']} | result: {timeline.data['result']['outcome']}")
                    writer.grab_frame()
            os.replace(target, output)
    except BaseException:
        if reserved:
            output.unlink(missing_ok=True)
        raise
    finally:
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory")
    parser.add_argument("output")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--elevation", type=float, default=25)
    parser.add_argument("--azimuth", type=float, default=-55)
    args = parser.parse_args()
    if Path(args.output).suffix.lower() == ".mp4":
        render_video(args.trajectory, args.output, args.fps, args.width, args.height,
                     args.elevation, args.azimuth)
    else:
        render(args.trajectory, args.output)


if __name__ == "__main__":
    main()
