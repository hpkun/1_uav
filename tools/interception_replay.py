"""Render a saved interception trajectory into a standalone interactive HTML file."""
import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from env.interception.config import VERSION, MAX_PURSUERS


def render(source, output):
    data = json.loads(Path(source).read_text(encoding="utf-8"))
    if data.get("version") != VERSION:
        raise ValueError("unsupported interception trajectory")
    frames, config = data["frames"], data["config"]
    figure = go.Figure()
    for prefix, mask, color in (("pursuers", "active_mask", "#2378bf"),
                                ("evaders", "evader_mask", "#df6a36")):
        for i, present in enumerate(frames[0][mask]):
            if not present:
                continue
            rows = [frame for k, frame in enumerate(frames) if k == 0 or frames[k-1][mask][i]]
            states = np.asarray([row[prefix][i] for row in rows])
            times = [row["time"] for row in rows]
            steps = [row["step"] for row in rows]
            if prefix == "evaders" and not rows[-1][mask][i]:
                capture = next(event for event in rows[-1]["events"]
                               if event["kind"] == "capture" and event["entities"][1] == MAX_PURSUERS + i)
                times[-1] = capture["time"]
            figure.add_trace(go.Scatter3d(x=states[:, 0], y=states[:, 1], z=states[:, 2],
                                         mode="lines+markers", marker=dict(size=2), line=dict(color=color),
                                         name=f"{prefix[:-1].title()} {i}",
                                         customdata=np.c_[steps, times, np.linalg.norm(states[:, 3:], axis=1)],
                                         hovertemplate="Step %{customdata[0]} · %{customdata[1]:.3f} s"
                                                       "<br>speed %{customdata[2]:.2f} m/s"))
            arrows = states[::max(1, len(states) // 12)]
            figure.add_trace(go.Cone(x=arrows[:, 0], y=arrows[:, 1], z=arrows[:, 2],
                                     u=arrows[:, 3], v=arrows[:, 4], w=arrows[:, 5], anchor="tail",
                                     sizemode="absolute", sizeref=2, showscale=False, showlegend=False,
                                     colorscale=[[0, color], [1, color]]))
    theta, phi = np.meshgrid(np.linspace(0, 2*np.pi, 36), np.linspace(0, np.pi, 18))
    center, radius = config["center"], config["radius"]
    figure.add_trace(go.Surface(x=center[0] + radius*np.cos(theta)*np.sin(phi),
                                y=center[1] + radius*np.sin(theta)*np.sin(phi),
                                z=center[2] + radius*np.cos(phi), opacity=.25,
                                name="Protected zone", showscale=False, showlegend=True,
                                colorscale=[[0, "#55a868"], [1, "#55a868"]]))
    for frame in frames:
        for event in frame["events"]:
            positions = np.asarray(event["positions"])
            names = [f"P{i}" if i < MAX_PURSUERS else f"E{i-MAX_PURSUERS}" for i in event["entities"]]
            label = f"{event['kind']} {' / '.join(names)} · step {event['step']} · {event['time']:.3f} s"
            figure.add_trace(go.Scatter3d(x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
                                         mode="markers", marker=dict(size=7, symbol="diamond"),
                                         name=label, text=[label] * len(positions), hoverinfo="text"))
    result = data["result"]
    figure.update_layout(title=f"{data['policy']} · {config['scale']} · {result['outcome']} · "
                               f"{result['steps']} steps · {result['time']:.3f} s",
                         scene=dict(aspectmode="data", xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)"))
    with Path(output).open("x", encoding="utf-8") as stream:
        stream.write(figure.to_html(include_plotlyjs=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory")
    parser.add_argument("output")
    args = parser.parse_args()
    render(args.trajectory, args.output)


if __name__ == "__main__":
    main()
