"""Offline reporting from committed research journals. No model deserialization."""
import argparse
from collections import deque
from pathlib import Path
from statistics import mean, stdev

from algorithm.common.research import (commit_header, evaluation_records, journal,
                                        read_json, score, summarize)
from algorithm.common.serialization import write_json
from env.interception.config import VERSION


def series(records):
    recent = deque(maxlen=100)
    output = []
    for row in records:
        recent.extend(row["completed_episodes"])
        stats = summarize(list(recent))
        output.append({k: v for k, v in row.items() if k not in ("completed_episodes", "previous")}
                      | {"window": stats})
    return output


def experiment_signature(run, planned):
    return (run["environment"], {k: v for k, v in run["training"].items() if k != "seed"},
            run["protocol"], run["evaluation"], run["training_scenario"],
            run["initial_budget_updates"], planned)


def aggregate(curves, getter):
    result = []
    for update in sorted({row["update"] for curve in curves for row in curve}):
        values = [getter(curve[update-1]) for curve in curves if len(curve) >= update]
        values = [value for value in values if value is not None]
        result.append(dict(update=update, mean=mean(values) if values else None,
                           std=stdev(values) if len(values) > 1 else None, n=len(values)))
    return result


def plot(output, curves, evaluations):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 4, figsize=(16, 10), layout="constrained")
    panels = [
        ("Success rate", "success_rate"), ("Return", "mean_episode_return"),
        ("Capture time (s)", "mean_capture_time_s"), ("Breach rate", "breach_rate"),
        ("Collision rate", "collision_rate"), ("Timeout rate", "timeout_rate"),
        ("Actor loss", "actor_total_loss"), ("Critic loss", "critic_loss"),
        ("Entropy", "entropy"), ("Approx KL", "approx_kl")
    ]
    combined = {}
    for index, (title, key) in enumerate(panels):
        axis = axes.flat[index]
        getter = (lambda row: row["window"].get(key)) if index < 6 else (lambda row: row.get(key))
        values = aggregate(curves, getter)
        combined[key] = values
        axis.plot([row["update"] for row in values], [row["mean"] for row in values], marker=".")
        band = [row for row in values if row["std"] is not None]
        if band:
            axis.fill_between([row["update"] for row in band], [row["mean"]-row["std"] for row in band],
                              [row["mean"]+row["std"] for row in band], alpha=.18)
        if all(row["mean"] is None for row in values):
            axis.text(.5, .5, "N/A", ha="center", transform=axis.transAxes)
        axis.set(title=title + (" · last 100 episodes" if index < 6 else ""), xlabel="Update")
    rewards = axes.flat[10]
    parts = sorted({key for curve in curves for row in curve
                    for key in row["window"]["mean_reward_components"]})
    for key in parts:
        values = aggregate(curves, lambda row: row["window"]["mean_reward_components"].get(key))
        combined["reward/" + key] = values
        rewards.plot([row["update"] for row in values], [row["mean"] for row in values], label=key)
    if parts:
        rewards.legend(fontsize=7)
    else:
        rewards.text(.5, .5, "N/A", ha="center", transform=rewards.transAxes)
    rewards.set(title="Reward components · last 100 episodes", xlabel="Update")
    evaluation = axes.flat[11]
    combined["evaluation"] = {}
    for metric in ("success_rate", "breach_rate", "friendly_collision_rate", "evader_collision_rate", "pursuer_boundary_violation_rate",
                   "evader_escape_rate", "protection_success_rate", "partial_capture_rate"):
        values = []
        for update in sorted({r["update"] for rows in evaluations for r in rows}):
            points = [r["summary"][metric] for rows in evaluations for r in rows
                      if r["update"] == update and metric in r["summary"]]
            if not points:
                continue
            values.append(dict(update=update, mean=mean(points), n=len(points),
                               std=stdev(points) if len(points) > 1 else None))
        combined["evaluation"][metric] = values
        if values:
            evaluation.plot([r["update"] for r in values], [r["mean"] for r in values], ".-", label=metric)
    if any(evaluations):
        evaluation.legend(fontsize=7)
    else:
        evaluation.text(.5, .5, "N/A", ha="center", transform=evaluation.transAxes)
    evaluation.set(title="Independent evaluation", xlabel="Update")
    figure.suptitle("Mean across seeds · shading: sample SD where available")
    figure.savefig(output, dpi=140)
    plt.close(figure)
    return combined


def generate(folders, output):
    output = Path(output)
    curves, evaluations, runs, seeds = [], [], [], set()
    signature = None
    for folder in map(Path, folders):
        head = commit_header(folder / "latest.pt")
        run = head["run"]
        if run["protocol"]["version"] != VERSION:
            raise ValueError("historical run requires its original reporting protocol")
        if run != read_json(folder / "config.json"):
            raise ValueError("run config differs from committed metadata")
        records = journal(folder, head)
        evaluations.append(evaluation_records(folder, run, head["update"]))
        curves.append(series(records))
        latest = records[-1] if records else None
        planned = latest["planned_updates"] if latest else run["initial_budget_updates"]
        settings = experiment_signature(run, planned)
        seed = run["training"]["seed"]
        if signature is not None and (settings != signature or seed in seeds):
            raise ValueError("multi-seed report requires distinct seeds and matching protocol, settings and budgets")
        signature = settings
        seeds.add(seed)
        latest_evaluation = next((r["summary"] for r in evaluations[-1] if r["update"] == head["update"]), None)
        best = read_json(folder / "best.json") if (folder / "best.json").exists() else None
        best_evaluation = None
        if best and evaluations[-1]:
            winner = min(evaluations[-1], key=score)
            if best["best_update"] != winner["update"] or best["checkpoint_sha256"] != winner["checkpoint_sha256"]:
                raise ValueError("Best metadata is pending reconciliation")
            best_evaluation = winner["summary"]
        runs.append(dict(path=str(folder), config=run, planned_updates=planned,
                         planned_joint_steps=planned * run["training"]["workers"] * run["training"]["horizon"],
                         completed_updates=head["update"], joint_steps=latest["joint_steps"] if latest else 0,
                         completed_episodes=latest["completed_episodes_total"] if latest else 0,
                         best=best, best_evaluation=best_evaluation, latest_evaluation=latest_evaluation,
                         update_series=curves[-1], evaluations=evaluations[-1]))
    output.mkdir(parents=True, exist_ok=False)
    combined = plot(output / "training_curves.png", curves, evaluations)
    write_json(output / "summary.json", dict(runs=runs, aggregate=combined, window_episodes=100,
                                              axis="update", std="sample"))
    lines = ["# Interception training report", "",
             "Red defenders are Pursuers controlled by MAPPO. Blue intruders are Evaders using the configured rule.", "",
             "The horizontal axis is PPO update. Training outcome and reward curves use the last "
             "100 completed episodes. Across seeds, each point includes the mean, sample standard "
             "deviation when n > 1, and the effective sample count in summary.json.", "",
             "![Training curves](training_curves.png)", "",
             "| Run / seed | Updates | Joint steps | Episodes | Best update / success | Latest success |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in runs:
        seed = row["config"]["training"]["seed"]
        best = row["best"]
        selected = f'{best["best_update"]} / {best["best_score"]["success_rate"]:.1%}' if best else "N/A"
        recent = row["latest_evaluation"]
        latest = f'{recent["success_rate"]:.1%}' if recent else "N/A"
        lines.append(f'| {row["path"]} / {seed} | {row["completed_updates"]}/{row["planned_updates"]} | '
                     f'{row["joint_steps"]}/{row["planned_joint_steps"]} | {row["completed_episodes"]} | '
                     f'{selected} | {latest} |')
    lines.extend(["", "| Seed | Model | All captured | Breach | Friendly collision | Evader collision | Boundary | Escape | Timeout | "
                  "Protection | Partial capture | Mean successful capture (s) |",
                  "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in runs:
        for name, metrics in (("Best", row["best_evaluation"]),
                              ("Latest", row["latest_evaluation"])):
            rates = [f"{metrics[key]:.1%}" if metrics else "N/A"
                     for key in ("success_rate", "breach_rate", "friendly_collision_rate", "evader_collision_rate",
                                 "pursuer_boundary_violation_rate", "evader_escape_rate", "timeout_rate",
                                 "protection_success_rate", "partial_capture_rate")]
            duration = metrics["mean_capture_time_s"] if metrics else None
            duration_text = f"{duration:.3f}" if duration is not None else "N/A"
            lines.append(f"| {row['config']['training']['seed']} | {name} | "
                         + " | ".join(rates) + f" | {duration_text} |")
    lines.extend(["", "The JSON includes the environment and optimizer settings, individual evaluation "
                  "episodes, outcome counts, reward components, and the complete per-update series.", ""])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    generate(args.runs, args.out)


if __name__ == "__main__":
    main()
