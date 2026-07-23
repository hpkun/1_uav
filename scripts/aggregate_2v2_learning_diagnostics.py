"""Flatten formal 2v2 evaluation curves and attach nearest training diagnostics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--input-root",required=True); parser.add_argument("--output",default="outputs/metrics/2v2_learning_diagnostics.csv")
    args=parser.parse_args(); output=[]
    for seed_dir in sorted(Path(args.input_root).glob("seed_*")):
        run=seed_dir/"run"; evaluations=run/"evaluations.csv"; metrics=run/"metrics.csv"
        if not evaluations.exists() or not metrics.exists(): continue
        training=_rows(metrics); by_step={int(float(row["environment_steps"])):row for row in training}
        seen_non_timeout=False; seen_damage=False
        for evaluation in _rows(evaluations):
            step=int(float(evaluation["environment_steps"])); prior=max((value for value in by_step if value<=step),default=max(by_step)); train=by_step[prior]
            timeout=float(evaluation["timeout_rate"]); damage=float(evaluation["mean_effective_damage"])
            first_non_timeout=(not seen_non_timeout) and timeout<1.0; first_damage=(not seen_damage) and damage>0.0
            seen_non_timeout |= timeout<1.0; seen_damage |= damage>0.0
            output.append({
                "seed":seed_dir.name.removeprefix("seed_"),"environment_steps":step,
                "red_win_rate":evaluation["red_win_rate"],"timeout_rate":timeout,
                "mean_team_episode_return":evaluation["mean_team_episode_return"],
                "mean_effective_damage":damage,"mean_hits":evaluation["mean_hits"],
                "mean_episode_steps":evaluation["mean_episode_steps"],
                "policy_entropy_mean":evaluation["policy_entropy_mean"],
                "logits_top1_top2_margin_mean":evaluation["logits_top1_top2_margin_mean"],
                "terminal_reward_proportion":evaluation["terminal_reward_proportion"],
                "observation_saturation_ratio":evaluation["mean_observation_saturation_ratio"],
                "first_non_timeout":int(first_non_timeout),"first_effective_damage":int(first_damage),
                "training_entropy":train.get("entropy",""),"training_clip_fraction":train.get("clip_fraction",""),
                "training_approx_kl":train.get("approx_kl",""),"critic_explained_variance":train.get("explained_variance",""),
                # Not retrospectively available in the current evaluation trajectory schema.
                "mean_closest_enemy_distance":"","attack_angle_mean":"","escape_angle_mean":"",
                "advantage_area_entries":"","attack_area_entries":"","first_attack_area_step":"",
            })
    if not output: raise FileNotFoundError("No completed 2v2 evaluation curves")
    destination=Path(args.output); destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)
    print(destination.resolve())


if __name__=="__main__": main()
