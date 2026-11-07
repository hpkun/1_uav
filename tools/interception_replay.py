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
const frames = DATA.frames, events = DATA.events, result = DATA.result;
const slider = document.querySelector('#clock'), rate = document.querySelector('#rate');
const panel = document.querySelector('#details');
const count = frames[0].entities.length;
const trails = frames[0].entities.map((_, j) =>
  [0,1,2].map(axis => frames.map(row => row.entities[j].position[axis])));
const indices = Array.from({length: count + 3}, (_, i) => i + 1);
const nf = (x, digits=3) => x === null || x === undefined ? 'N/A' : x.toFixed(digits);
const pretty = x => x === null || x === undefined ? 'N/A' : JSON.stringify(x);
let index = 0, playing = false, started, origin, pending = false, dirty = false;
slider.max = frames.length - 1;
function show(i) {
  index = Math.max(0, Math.min(i, frames.length - 1));
  slider.value = index;
  const row = frames[index];
  document.querySelector('#time').textContent =
    row.time.toFixed(3) + ' s · step ' + row.step + ' · frame ' + (index+1) + '/' + frames.length;
  slider.setAttribute('aria-valuetext', row.time.toFixed(3) + ' seconds');
  const p = row.entities.filter(e => e.name.startsWith('P') && e.active).length;
  const e = row.entities.filter(e => e.name.startsWith('E') && e.active).length;
  const visible = events.filter(event => event.time <= row.time);
  panel.textContent = [
    'Episode: ' + result + ' | Current: ' + row.outcome,
    'Active pursuers: ' + p + ' | Active evaders: ' + e,
    ...row.entities.map(entity => entity.name + ': ' +
      (entity.active ? nf(entity.speed) + ' m/s' : 'inactive')),
    'Nearest P–E distance (m): ' + nf(row.distance),
    'Minimum separation so far (m): ' + nf(row.minimum),
    'Dwell (s), P × E: ' + pretty(row.dwell),
    'Velocity commands (m/s), P then E: ' + pretty(row.commands),
    'Reward components: ' + pretty(row.rewards),
    'Events: ' + (visible.map(event => event.label).join(' | ') || 'none')
  ].join('\n');
  dirty = true;
  if (!pending) draw();
}
async function draw() {
  pending = true;
  try {
    while (dirty) {
      dirty = false;
      const row = frames[index], entities = row.entities;
      const visible = events.filter(event => event.time <= row.time);
      const arrows = entities.filter(e => e.velocity && e.speed > 1e-9);
      const xyz = [0,1,2].map(axis => [
        ...trails.map(trail => trail[axis].slice(0,index+1)),
        entities.map(e => e.position[axis]),
        arrows.map(e => e.position[axis]),
        visible.flatMap(e => e.positions.map(p => p[axis]))
      ]);
      const update = {x: xyz[0], y: xyz[1], z: xyz[2]};
      for (const [axis,key] of ['u','v','w'].entries())
        update[key] = Array(count+1).fill(undefined).concat([arrows.map(e => e.velocity[axis]),undefined]);
      update.text = Array(count).fill(undefined).concat([
        entities.map(e => e.name + (e.active ? '' : ' (inactive)')), undefined,
        visible.flatMap(e => e.positions.map(() => e.label))
      ]);
      update['marker.color'] = Array(count).fill(undefined).concat([
        entities.map(e => e.active ? e.color : '#888888'),undefined,undefined
      ]);
      await Plotly.restyle(plot, update, indices);
    }
  } finally { pending = false; }
}
function tick(now) {
  if (!playing) return;
  const elapsed = origin + (now-started) * Number(rate.value) / 1000;
  let next = index;
  while (next+1 < frames.length && frames[next+1].time <= elapsed) next++;
  if (next !== index) show(next);
  if (index === frames.length-1) playing = false;
  else request = requestAnimationFrame(tick);
}
function pause() { playing = false; cancelAnimationFrame(request); }
let request;
function play() {
  pause();
  if (index === frames.length-1) show(0);
  origin = frames[index].time;
  started = performance.now();
  playing = true;
  request = requestAnimationFrame(tick);
}
document.querySelector('#play').onclick = play;
document.querySelector('#pause').onclick = pause;
document.querySelector('#restart').onclick = () => { pause(); show(0); play(); };
document.querySelector('#prev').onclick = () => { pause(); show(index-1); };
document.querySelector('#next').onclick = () => { pause(); show(index+1); };
slider.oninput = () => { pause(); show(Number(slider.value)); };
rate.onchange = () => { if (playing) play(); };
show(0);
"""


def render(source, output):
    import plotly.graph_objects as go

    timeline = Timeline(source)
    samples = [timeline.at(timestamp) for timestamp in timeline.key_times]
    config = timeline.data["config"]
    center, radius = np.array(config["center"]), config["radius"]
    theta, phi = np.meshgrid(np.linspace(0, 2*np.pi, 32), np.linspace(0, np.pi, 16))
    sphere = radius * np.array([np.cos(theta)*np.sin(phi), np.sin(theta)*np.sin(phi), np.cos(phi)])
    sphere += center[:, None, None]
    figure = go.Figure(go.Surface(x=sphere[0], y=sphere[1], z=sphere[2], opacity=.22,
                                  name="Protected zone", showscale=False, showlegend=True,
                                  colorscale=[[0, "#55a868"], [1, "#55a868"]]))
    for entity in samples[0]["entities"]:
        figure.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="lines", name=entity["name"],
                                     line=dict(color=entity["color"], width=3)))
    figure.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="markers+text", textposition="top center",
                                 name="Current UAV", marker=dict(size=5)))
    figure.add_trace(go.Cone(x=[], y=[], z=[], u=[], v=[], w=[], anchor="tail", sizemode="absolute",
                             sizeref=2, showscale=False, colorscale=[[0, "#485864"], [1, "#485864"]]))
    figure.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="markers", name="Events",
                                 hoverinfo="text", marker=dict(size=7, symbol="diamond", color="#2f9c63")))
    bounds = timeline.bounds(samples)
    figure.update_layout(uirevision="keep-camera", height=640, margin=dict(t=30, b=0),
                         scene=dict(aspectmode="data", **{axis+"axis": dict(title=axis+" (m)", range=bounds[i])
                                                          for i, axis in enumerate("xyz")}))
    events = [event | {"label": f"{event['kind']} · {event['time']:.6f} s · step {event['step']}"}
              for event in timeline.events]
    data = json.dumps(dict(frames=samples, events=events, result=timeline.data["result"]["outcome"]),
                      separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    script = "const DATA = " + data + ";\n" + PLAYER
    graph = figure.to_html(full_html=False, include_plotlyjs=True, post_script=script)
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
                     '<style>body{font:15px system-ui;margin:16px}pre{white-space:pre-wrap}'
                     '#clock{width:min(36vw,420px)}</style>' + controls + graph)


def video_times(duration, fps):
    count = max(2, math.ceil(duration * fps)) if duration else 1
    return [k / fps for k in range(count-1)] + [duration]


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
    center, radius = timeline.data["config"]["center"], timeline.data["config"]["radius"]
    theta, phi = np.meshgrid(np.linspace(0, 2*np.pi, 28), np.linspace(0, np.pi, 14))
    figure = plt.figure(figsize=(width/100, height/100), dpi=100)
    axes = figure.add_subplot(projection="3d")
    output = Path(output)
    with output.open("xb"):
        pass
    try:
        with tempfile.TemporaryDirectory(dir=output.parent) as folder:
            target = Path(folder) / "video.mp4"
            writer = FFMpegWriter(fps=fps, codec=codec, extra_args=["-pix_fmt", "yuv420p"])
            with writer.saving(figure, str(target), 100):
                for row in samples:
                    axes.clear()
                    axes.view_init(elev=elevation, azim=azimuth)
                    axes.set(xlim=bounds[0], ylim=bounds[1], zlim=bounds[2],
                             xlabel="x (m)", ylabel="y (m)", zlabel="z (m)")
                    axes.set_box_aspect(np.ptp(bounds, axis=1))
                    axes.plot_surface(center[0]+radius*np.cos(theta)*np.sin(phi),
                                      center[1]+radius*np.sin(theta)*np.sin(phi),
                                      center[2]+radius*np.cos(phi), color="#55a868", alpha=.18)
                    for i, entity in enumerate(row["entities"]):
                        past = [sample["entities"][i]["position"] for sample in history
                                if sample["time"] < row["time"]]
                        path = np.array(past + [entity["position"]])
                        axes.plot(*path.T, color=entity["color"])
                        position = entity["position"]
                        axes.scatter(*position, color=entity["color"] if entity["active"] else "#888888")
                        axes.text(*position, entity["name"] + ("" if entity["active"] else " inactive"))
                        if entity["velocity"] is not None:
                            axes.quiver(*position, *entity["velocity"], length=.3, color=entity["color"])
                    for event in timeline.events:
                        if event["time"] <= row["time"]:
                            axes.scatter(*np.array(event["positions"]).T, marker="D", color="#2f9c63")
                    axes.set_title(f"{row['time']:.3f} s | step {row['step']} | "
                                   f"{row['outcome']} | result: {timeline.data['result']['outcome']}")
                    writer.grab_frame()
            os.replace(target, output)
    except BaseException:
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
