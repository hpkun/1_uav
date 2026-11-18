"""Audit and plot the independent 20v20 delayed-convergence intruder references."""
import argparse
import json
from pathlib import Path

import numpy as np

from env.ideal_interception.large_scale_20v20_env import TwentyConfig, TwentyVTwentyEnv
from env.ideal_interception.large_scale_20v20_trajectory import attack_frame, PROFILE


def audit(seed=17):
    env = TwentyVTwentyEnv(TwentyConfig(seed=seed))
    _, info = env.reset(seed)
    e_long, e_lat = attack_frame(env.config)
    goal_xy = np.asarray(env.config.protected_center[:2])
    fractions = (0., 1/3, 2/3)
    lanes = np.asarray([[float((path.sample(f*path.duration)[0][:2]-goal_xy)@e_lat)
                         for path in env.paths] for f in fractions])
    spans = np.ptp(lanes, axis=1)
    if spans[0] <= 0:
        raise ValueError("initial formation has no lateral spread")
    geometry = [path.geometry_diagnostics() for path in env.paths]
    correlation = [float(np.corrcoef(lanes[0], lanes[k])[0, 1]) for k in (1, 2)]
    summary = dict(experiment="20v20_delayed_convergence_path_audit", seed=seed,
        scenario_id=info["scenario_id"], trajectory_profile=PROFILE,
        attack_frame=dict(e_long=e_long.tolist(), e_lat=e_lat.tolist()),
        lateral_span_initial_m=float(spans[0]),
        lateral_span_one_third_m=float(spans[1]),
        lateral_span_two_thirds_m=float(spans[2]),
        retention_one_third=float(spans[1]/spans[0]),
        retention_two_thirds=float(spans[2]/spans[0]),
        lateral_std_initial_m=float(np.std(lanes[0])),
        lateral_std_one_third_m=float(np.std(lanes[1])),
        lateral_std_two_thirds_m=float(np.std(lanes[2])),
        lane_correlation_one_third=correlation[0],
        lane_correlation_two_thirds=correlation[1],
        maximum_chord_deviation_range_m=[float(min(g["maximum_chord_deviation_m"] for g in geometry)),
                                         float(max(g["maximum_chord_deviation_m"] for g in geometry))],
        maximum_heading_deviation_range_deg=[float(min(g["maximum_heading_deviation_from_chord_deg"] for g in geometry)),
                                             float(max(g["maximum_heading_deviation_from_chord_deg"] for g in geometry))],
        path_hashes=[path.specification()["polyline_sha256"] for path in env.paths],
        path_metadata=[path.plan_metadata for path in env.paths])
    return summary, env.paths, env.config


def write_plots(paths, config, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    for view in ("xy", "3d"):
        figure = plt.figure(figsize=(11, 8), dpi=150)
        axis = figure.add_subplot(111, projection="3d" if view == "3d" else None)
        for index, path in enumerate(paths):
            p = path.points
            color = plt.cm.viridis(index/19)
            if view == "xy":
                axis.plot(p[:, 0], p[:, 1], color=color, linewidth=1.4)
                axis.scatter(p[0, 0], p[0, 1], color=color, s=20)
            else:
                axis.plot(p[:, 0], p[:, 1], p[:, 2], color=color, linewidth=1.4)
                axis.scatter(*p[0], color=color, s=20)
        cx, cy, cz = config.protected_center
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = cx+config.protected_radius*np.cos(theta)
        circle_y = cy+config.protected_radius*np.sin(theta)
        if view == "xy":
            axis.add_patch(Circle((cx, cy), config.protected_radius,
                                  fill=False, edgecolor="red", linewidth=2))
            axis.set(xlabel="x (m)", ylabel="y (m)", xlim=(config.lower[0], config.upper[0]),
                     ylim=(config.lower[1], config.upper[1]), aspect="equal")
        else:
            axis.plot(circle_x, circle_y, np.full_like(theta, cz), color="red", linewidth=2)
            axis.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)",
                     xlim=(config.lower[0], config.upper[0]),
                     ylim=(config.lower[1], config.upper[1]),
                     zlim=(config.lower[2], config.upper[2]))
        axis.set_title(f"20v20 intruder references | delayed convergence | {view.upper()}")
        figure.savefig(folder/f"intruder_reference_{view}.png", bbox_inches="tight")
        plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    summary, paths, config = audit(args.seed)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    write_plots(paths, config, args.out)
    print(f"span={summary['lateral_span_initial_m']:.2f}/"
          f"{summary['lateral_span_one_third_m']:.2f}/"
          f"{summary['lateral_span_two_thirds_m']:.2f}m "
          f"retention={summary['retention_one_third']:.3f}/"
          f"{summary['retention_two_thirds']:.3f} "
          f"corr={summary['lane_correlation_one_third']:.3f}/"
          f"{summary['lane_correlation_two_thirds']:.3f}")


if __name__ == "__main__":
    main()
