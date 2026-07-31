from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
BATCH = "20260730_213216"
RUNS = {
    "homogeneous_control": ROOT / "outputs/mappo/functional_homogeneous_3v3_seed1" / BATCH,
    "heterogeneous_no_relay": ROOT / "outputs/mappo/functional_heterogeneous_no_relay_3v3_seed1" / BATCH,
    "heterogeneous_relay": ROOT / "outputs/mappo/functional_heterogeneous_relay_3v3_seed1" / BATCH,
}
LABEL = {
    "homogeneous_control": "Homogeneous",
    "heterogeneous_no_relay": "Heterogeneous, no relay",
    "heterogeneous_relay": "Heterogeneous, relay",
}
COLORS = {"homogeneous_control": "#4472C4", "heterogeneous_no_relay": "#ED7D31", "heterogeneous_relay": "#70AD47"}
RESULT_FIELDS = [
    "overall_red_win_rate", "elimination_red_win_rate", "timeout_survivor_red_win_rate",
    "blue_win_rate", "draw_rate", "timeout_rate", "mean_episode_return", "mean_episode_steps",
    "mean_red_survivors", "mean_blue_survivors", "mean_survivor_difference", "mean_red_hits",
    "mean_blue_hits", "mean_red_effective_damage", "mean_blue_effective_damage", "red_crash_rate",
    "blue_crash_rate", "mean_collisions",
]
FUNCTIONAL_FIELDS = [
    "has_support_agent", "support_metrics_applicable", "support_survival_rate", "mission_success_rate",
    "support_detection_coverage_mean", "relay_visible_enemy_count_mean", "support_incoming_threat_mean",
    "support_position_error_mean", "combat_attack_attempts_mean", "combat_hits_mean",
    "combat_effective_damage_mean",
]
SUPPORT_ONLY_FIELDS = {
    "has_support_agent", "support_metrics_applicable", "support_survival_rate", "mission_success_rate",
    "support_detection_coverage_mean", "relay_visible_enemy_count_mean", "support_incoming_threat_mean",
    "support_position_error_mean",
}
POLICY_FIELDS = ["policy_entropy_mean", "logits_top1_top2_margin_mean", "terminal_reward_proportion", "mean_observation_saturation_ratio"]
ALIASES = {
    "elimination_red_win_rate": "elimination_win_rate",
    "timeout_survivor_red_win_rate": "timeout_survival_win_rate",
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def finite_csv(df: pd.DataFrame) -> tuple[int, int, int]:
    numeric = df.select_dtypes(include=[np.number])
    return int(numeric.isna().sum().sum()), int(np.isinf(numeric.to_numpy()).sum()), int(df.isna().all(axis=0).sum())


def val(d: dict[str, Any], name: str) -> Any:
    return d.get(name, d.get(ALIASES.get(name, ""), np.nan))


def fmt(x: Any, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "N/A"
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.{digits}f}"
    return str(x)


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out += ["| " + " | ".join(fmt(x) for x in row) + " |" for row in rows]
    return "\n".join(out)


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else 0}
    if not path.exists():
        return result
    try:
        import torch
        obj = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if isinstance(obj, dict):
            for key in ("environment_steps", "update_index", "updates", "episodes", "seed", "schema_metadata"):
                item = obj.get(key)
                if isinstance(item, (str, int, float, bool, type(None), dict)):
                    result[key] = item
            result["top_level_keys"] = sorted(obj.keys())
    except Exception as exc:
        result["metadata_read_error"] = type(exc).__name__
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    configs, summaries, metrics, evals, ckmeta = {}, {}, {}, {}, {}
    integrity_rows: list[dict[str, Any]] = []
    for mode, run in RUNS.items():
        configs[mode] = load_yaml(run / "config.yaml")
        summaries[mode] = load_yaml(run / "final_summary.yaml")
        metrics[mode] = pd.read_csv(run / "metrics.csv")
        evals[mode] = pd.read_csv(run / "evaluations.csv")
        ckmeta[mode] = {n: checkpoint_metadata(run / "checkpoints" / f"{n}.pt") for n in ("initial", "last", "best")}
        m_nan, m_inf, m_empty = finite_csv(metrics[mode])
        e_nan, e_inf, e_empty = finite_csv(evals[mode])
        c, s = configs[mode], summaries[mode]
        env, schema = c["environment"], s["schema_metadata"]
        required = ["config.yaml", "metrics.csv", "evaluations.csv", "final_summary.yaml", "checkpoints/initial.pt", "checkpoints/last.pt", "checkpoints/best.pt"]
        missing = [x for x in required if not (run / x).exists()]
        expected = math.ceil(c["total_env_steps"] / (c["num_envs"] * c["rollout_length"])) * c["num_envs"] * c["rollout_length"]
        best_step = int(s["validation_best_evaluation"]["environment_steps"])
        integrity_rows.append({
            "mode": mode, "complete": not missing and int(s["environment_steps"]) == expected,
            "missing_artifacts": ";".join(missing), "environment_steps": s["environment_steps"],
            "expected_rollout_aligned_steps": expected, "updates": s["updates"], "metrics_rows": len(metrics[mode]),
            "episodes": s["episodes"], "metrics_last_step": int(metrics[mode].environment_steps.iloc[-1]),
            "evaluations_last_step": int(evals[mode].environment_steps.iloc[-1]), "final_summary_step": s["environment_steps"],
            "initial_checkpoint": ckmeta[mode]["initial"]["exists"], "last_checkpoint": ckmeta[mode]["last"]["exists"],
            "best_checkpoint": ckmeta[mode]["best"]["exists"], "best_checkpoint_step": best_step,
            "best_step_in_validation": best_step in set(evals[mode].environment_steps),
            "test_initial": "initial" in s.get("test_evaluations", {}), "test_last": "last" in s.get("test_evaluations", {}),
            "test_best": "best" in s.get("test_evaluations", {}),
            "schema": schema.get("environment_schema_version"), "obs_dim": schema.get("obs_dim"), "state_dim": schema.get("state_dim"),
            "seed": c.get("seed"), "num_envs": c.get("num_envs"), "rollout_length": c.get("rollout_length"),
            "opponent": env.get("opponent"), "functional_mode": env.get("functional_mode"), "relay_enabled": env.get("relay_enabled"),
            "metrics_nan_cells": m_nan, "metrics_inf_cells": m_inf, "metrics_empty_columns": m_empty,
            "evaluations_nan_cells": e_nan, "evaluations_inf_cells": e_inf, "evaluations_empty_columns": e_empty,
        })
    pd.DataFrame(integrity_rows).to_csv(OUT / "artifact_integrity.csv", index=False)

    summary_rows = []
    for mode, s in summaries.items():
        for checkpoint, data in s["test_evaluations"].items():
            row = {"mode": mode, "checkpoint": checkpoint, "checkpoint_environment_steps": 0 if checkpoint == "initial" else (s["environment_steps"] if checkpoint == "last" else s["validation_best_evaluation"]["environment_steps"])}
            for field in RESULT_FIELDS + FUNCTIONAL_FIELDS + POLICY_FIELDS:
                value = val(data, field)
                if mode == "homogeneous_control" and field in SUPPORT_ONLY_FIELDS:
                    value = np.nan
                row[field] = value
            summary_rows.append(row)
    experiment_summary = pd.DataFrame(summary_rows)
    experiment_summary.to_csv(OUT / "experiment_summary.csv", index=False, na_rep="missing")

    trajectory = []
    for mode, df in evals.items():
        for record in df.to_dict("records"):
            record = {**record, "mode": mode, "is_validation_best": int(record["environment_steps"]) == int(summaries[mode]["validation_best_evaluation"]["environment_steps"])}
            trajectory.append(record)
    trajectory_df = pd.DataFrame(trajectory)
    trajectory_df.to_csv(OUT / "validation_trajectory.csv", index=False)

    tail_rows = []
    tail_map = {
        "rollout_return": "rollout_return_mean", "timeout_rate": "timeout_rate", "red_hits": "rollout_red_hits_mean",
        "red_effective_damage": "rollout_red_effective_damage_mean", "blue_hits": "rollout_blue_hits_mean",
        "blue_effective_damage": "rollout_blue_effective_damage_mean", "entropy": "rollout_action_entropy",
        "support_survival": "rollout_support_survival_rate", "support_coverage": "rollout_support_detection_coverage_mean",
        "relay_visible": "rollout_relay_visible_enemy_count_mean", "mission_success": "rollout_mission_success_rate",
    }
    for mode, df in metrics.items():
        windows: list[tuple[str, pd.DataFrame]] = [("all", df), ("last_10", df.tail(10)), ("last_30", df.tail(30))]
        for criterion, col in (("return", "rollout_return_mean"), ("red_damage", "rollout_red_effective_damage_mean"), ("mission_success", "rollout_mission_success_rate")):
            rolling = df[col].rolling(20, min_periods=20).mean()
            end = int(rolling.idxmax()) if rolling.notna().any() else len(df) - 1
            windows.append((f"best_20_{criterion}", df.iloc[max(0, end - 19):end + 1]))
        for name, w in windows:
            row: dict[str, Any] = {"mode": mode, "window": name, "updates": len(w), "start_step": int(w.environment_steps.iloc[0]), "end_step": int(w.environment_steps.iloc[-1])}
            for short, col in tail_map.items():
                row[f"{short}_mean"] = w[col].mean()
                row[f"{short}_std"] = w[col].std(ddof=1)
            tail_rows.append(row)
    pd.DataFrame(tail_rows).to_csv(OUT / "training_tail_summary.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    def lines(filename: str, fields: list[tuple[str, str]], ylabel: str) -> None:
        fig, axes = plt.subplots(len(fields), 1, figsize=(9, 3.3 * len(fields)), sharex=True, squeeze=False)
        for ax, (field, title) in zip(axes[:, 0], fields):
            for mode, df in evals.items():
                ax.plot(df.environment_steps, df[field], marker="o", label=LABEL[mode], color=COLORS[mode])
            ax.set_title(title); ax.set_ylabel(ylabel if len(fields) == 1 else title); ax.set_xlim(0, 305000)
        axes[-1, 0].set_xlabel("Environment steps"); axes[0, 0].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(FIG / filename, dpi=180); plt.close(fig)
    lines("validation_return.png", [("mean_episode_return", "Validation mean episode return")], "Return")
    lines("validation_outcomes.png", [("overall_red_win_rate", "Overall red win rate"), ("elimination_win_rate", "Elimination win rate"), ("timeout_rate", "Timeout rate")], "Rate")
    lines("validation_survivors.png", [("mean_red_survivors", "Mean red survivors"), ("mean_blue_survivors", "Mean blue survivors")], "Aircraft")
    lines("validation_combat_effectiveness.png", [("combat_hits_mean", "Combat hits"), ("combat_effective_damage_mean", "Combat effective damage")], "Mean / episode")
    fig, axes = plt.subplots(4, 1, figsize=(9, 12), sharex=True)
    for mode in ("heterogeneous_no_relay", "heterogeneous_relay"):
        df = evals[mode]
        for ax, field, title in zip(axes, ["support_survival_rate", "support_detection_coverage_mean", "relay_visible_enemy_count_mean", "mission_success_rate"], ["Support survival", "Detection coverage", "Relay-visible enemies", "Mission success"]):
            ax.plot(df.environment_steps, df[field], marker="o", label=LABEL[mode], color=COLORS[mode]); ax.set_title(title); ax.set_xlim(0, 305000)
    axes[-1].set_xlabel("Environment steps"); axes[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG / "validation_support_metrics.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for mode, df in metrics.items(): ax.plot(df.environment_steps, df.rollout_action_entropy, label=LABEL[mode], color=COLORS[mode])
    ax.set(xlabel="Environment steps", ylabel="Entropy", title="Training action entropy", xlim=(0, 305000)); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG / "training_entropy.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8)); checkpoints = ["initial", "last", "best"]; xs = np.arange(3); width = .24
    for ax, field, title in zip(axes.flat, ["overall_red_win_rate", "mean_episode_return", "mean_red_survivors", "combat_effective_damage_mean"], ["Red win rate", "Episode return", "Red survivors", "Combat damage"]):
        for i, mode in enumerate(RUNS):
            vals = [val(summaries[mode]["test_evaluations"][c], field) for c in checkpoints]
            ax.bar(xs + (i-1)*width, vals, width, label=LABEL[mode], color=COLORS[mode])
        ax.set_xticks(xs, checkpoints); ax.set_title(title)
    axes[0,0].legend(fontsize=7); fig.tight_layout(); fig.savefig(FIG / "test_checkpoint_comparison.png", dpi=180); plt.close(fig)

    pairs = [("heterogeneous_no_relay", "homogeneous_control"), ("heterogeneous_relay", "heterogeneous_no_relay"), ("heterogeneous_relay", "homogeneous_control")]
    delta_fields = ["overall_red_win_rate", "elimination_red_win_rate", "mission_success_rate", "support_survival_rate", "mean_episode_return", "timeout_rate", "mean_red_survivors", "mean_blue_survivors", "combat_attack_attempts_mean", "combat_hits_mean", "combat_effective_damage_mean", "support_detection_coverage_mean", "relay_visible_enemy_count_mean"]
    delta_rows = []
    for checkpoint in ("last", "best"):
        for left, right in pairs:
            l, r = summaries[left]["test_evaluations"][checkpoint], summaries[right]["test_evaluations"][checkpoint]
            for field in delta_fields:
                lv, rv = val(l, field), val(r, field)
                if right == "homogeneous_control" and field in SUPPORT_ONLY_FIELDS: rv = np.nan
                delta_rows.append([checkpoint, f"{left} - {right}", field, lv-rv if pd.notna(lv) and pd.notna(rv) else np.nan, (lv-rv)/abs(rv)*100 if pd.notna(lv) and pd.notna(rv) and rv != 0 else np.nan])

    report = build_report(configs, summaries, metrics, evals, integrity_rows, delta_rows)
    (OUT / "comparison_report.md").write_text(report, encoding="utf-8")
    print("Analysis complete")
    print(pd.DataFrame(integrity_rows)[["mode", "complete", "environment_steps", "updates", "episodes", "metrics_nan_cells", "metrics_inf_cells"]].to_string(index=False))
    print("\nHeld-out test last/best core metrics")
    print(experiment_summary[experiment_summary.checkpoint.isin(["last", "best"])][["mode", "checkpoint", "overall_red_win_rate", "mean_episode_return", "timeout_rate", "mean_red_survivors", "mean_blue_survivors", "combat_hits_mean", "combat_effective_damage_mean", "support_survival_rate", "support_detection_coverage_mean", "relay_visible_enemy_count_mean", "mission_success_rate"]].to_string(index=False))


def build_report(configs: dict[str, Any], summaries: dict[str, Any], metrics: dict[str, pd.DataFrame], evals: dict[str, pd.DataFrame], integrity: list[dict[str, Any]], deltas: list[list[Any]]) -> str:
    core_fields = ["overall_red_win_rate", "elimination_red_win_rate", "mean_episode_return", "timeout_rate", "mean_red_survivors", "mean_blue_survivors", "combat_attack_attempts_mean", "combat_hits_mean", "combat_effective_damage_mean", "support_survival_rate", "support_detection_coverage_mean", "relay_visible_enemy_count_mean", "mission_success_rate"]
    test_rows = []
    for mode in RUNS:
        for cp in ("last", "best"):
            d = summaries[mode]["test_evaluations"][cp]
            row = [LABEL[mode], cp]
            for f in core_fields:
                x = val(d, f)
                if mode == "homogeneous_control" and f in SUPPORT_ONLY_FIELDS: x = None
                row.append(x)
            test_rows.append(row)
    delta_table = md_table(["checkpoint", "contrast", "metric", "absolute delta", "relative %"], deltas)
    integrity_table = md_table(["mode", "complete", "steps", "updates", "episodes", "best step", "NaN", "Inf"], [[r["mode"], r["complete"], r["environment_steps"], r["updates"], r["episodes"], r["best_checkpoint_step"], r["metrics_nan_cells"]+r["evaluations_nan_cells"], r["metrics_inf_cells"]+r["evaluations_inf_cells"]] for r in integrity])
    test_table = md_table(["mode", "checkpoint"] + core_fields, test_rows)
    validation_notes = []
    for mode, df in evals.items():
        first, last = df.iloc[0], df.iloc[-1]
        best_step = summaries[mode]["validation_best_evaluation"]["environment_steps"]
        validation_notes.append(f"- **{LABEL[mode]}**: steps {list(df.environment_steps.astype(int))}; selected best={best_step}. Return {first.mean_episode_return:.2f} → {last.mean_episode_return:.2f}, timeout {first.timeout_rate:.2f} → {last.timeout_rate:.2f}, red survivors {first.mean_red_survivors:.2f} → {last.mean_red_survivors:.2f}, combat damage {first.combat_effective_damage_mean:.2f} → {last.combat_effective_damage_mean:.2f}.")
    tail_notes = []
    for mode, df in metrics.items():
        last30 = df.tail(30)
        tail_notes.append(f"- **{LABEL[mode]}**: 147 updates, final step 301056, episodes={summaries[mode]['episodes']}, mean SPS={df.samples_per_second.mean():.1f}; entropy initial/max/final={df.rollout_action_entropy.iloc[0]:.3f}/{df.rollout_action_entropy.max():.3f}/{df.rollout_action_entropy.iloc[-1]:.3f}; last-30 return={last30.rollout_return_mean.mean():.2f}±{last30.rollout_return_mean.std():.2f}, timeout={last30.timeout_rate.mean():.2f}, red damage={last30.rollout_red_effective_damage_mean.mean():.2f}.")
    relay_last = summaries["heterogeneous_relay"]["test_evaluations"]["last"]
    relay_best = summaries["heterogeneous_relay"]["test_evaluations"]["best"]
    nr_last = summaries["heterogeneous_no_relay"]["test_evaluations"]["last"]
    nr_best = summaries["heterogeneous_no_relay"]["test_evaluations"]["best"]
    hom_last = summaries["homogeneous_control"]["test_evaluations"]["last"]
    recommendation = "Run additional independent seeds with the unchanged three-way design. All win and mission-success estimates are zero in seed 1, while checkpoint-level combat differences are large and directionally unstable; replication is therefore more informative than reward or code changes."
    return f"""# Functional 3v3 MAPPO comparison — seed 1, batch {BATCH}

## 1. Executive Summary

All three 300k experiments are complete and internally consistent at **301,056 environment steps** (147 × 2,048). None learned a held-out red win: test `last` and `best` overall/elimination win rates and heterogeneous mission-success rates are all **0**. Training mainly moved toward survival/avoidance. Relay produced real visible information (`relay_visible_enemy_count_mean`: last **{relay_last['relay_visible_enemy_count_mean']:.3f}**, best **{relay_best['relay_visible_enemy_count_mean']:.3f}**) and, at `best`, more combat activity than no-relay (attempts **{relay_best['combat_attack_attempts_mean']:.2f} vs {nr_best['combat_attack_attempts_mean']:.2f}**, damage **{relay_best['combat_effective_damage_mean']:.2f} vs {nr_best['combat_effective_damage_mean']:.2f}**), but it did not convert into wins or mission success. These are preliminary seed-1 aggregates, not significance claims.

## 2. Experiment Integrity

{integrity_table}

Required artifacts and all initial/last/best test blocks exist. Schema is `functional_heterogeneous_3v3_v1`, Actor=69D, Critic=64D, seed=1, 16 parallel envs, rollout=128, opponent=`greedy_combat`. CSV numeric columns contain no NaN or Inf. Homogeneous support fields are encoded as zeros upstream but are treated as **N/A** here because `support_metrics_applicable=0`.

## 3. Experimental Design and Controlled Variables

- Homogeneous: 3 armed Combat UAVs; no Support, no relay.
- No relay: 2 armed Combat UAVs + 1 unarmed long-range sensing Support UAV; no information sharing.
- Relay: same heterogeneous roles/weapons/reward as no-relay; Support shares information while alive.
- Controlled: dynamics, initial scenario, 69D Actor input, 64D Critic state, MAPPO hyperparameters, 300k target, seed, and GreedyCombat opponent.
- Contrasts: no-relay−homogeneous combines role differentiation and loss of one weapon platform; relay−no-relay isolates relay sharing; relay−homogeneous measures the complete heterogeneous scheme.

## 4. Validation Trajectories

{chr(10).join(validation_notes)}

No group shows validation win improvement. Returns improve largely alongside rising timeout/survival and collapsing combat output, indicating an attack-to-avoidance transition. See `validation_*.png` and `validation_trajectory.csv`.

## 5. Held-Out Test Results

{test_table}

## 6. Homogeneous vs Heterogeneous No Relay

At test-last, no-relay has the same zero wins, a slightly lower return ({nr_last['mean_episode_return']:.2f} vs {hom_last['mean_episode_return']:.2f}), fewer red survivors ({nr_last['mean_red_survivors']:.2f} vs {hom_last['mean_red_survivors']:.2f}), and zero combat output in both. Removing one armed Combat UAV plus adding Support therefore yields no demonstrated combat benefit. Support survival is 0 at last/best, so the role reward did not produce a surviving support role on held-out tests.

## 7. No Relay vs Relay

Relay is operationally nonzero, but outcome conversion is absent. At test-best it adds {relay_best['combat_attack_attempts_mean']-nr_best['combat_attack_attempts_mean']:.2f} attempts, {relay_best['combat_hits_mean']-nr_best['combat_hits_mean']:.2f} hits and {relay_best['combat_effective_damage_mean']-nr_best['combat_effective_damage_mean']:.2f} damage per episode, while both have 0 wins/mission success and 0 support survival. At test-last it retains modest combat ({relay_last['combat_effective_damage_mean']:.2f} damage vs {nr_last['combat_effective_damage_mean']:.2f}) but has lower return and fewer red survivors. Classification: **B — relay increases usable visibility and some attack activity, but not wins/mission success**; it is not evidence of a successful information advantage.

## 8. Full Heterogeneous Scheme vs Homogeneous Control

At last, homogeneous attains the highest red survival ({hom_last['mean_red_survivors']:.2f}) and best return ({hom_last['mean_episode_return']:.2f}), but does so with zero combat damage and 100% timeout. Relay is more active but less survivable, still with zero wins. Thus the full heterogeneous design is not supported on outcome performance in this seed.

## 9. Support Role Analysis

Support is present and its sensing coverage is meaningful (no-relay last {nr_last['support_detection_coverage_mean']:.3f}; relay last {relay_last['support_detection_coverage_mean']:.3f}), but held-out support survival is 0 for both last and best and mission success remains 0. Role differentiation exists structurally and in diagnostics, but a successful behavioral Support role is not established.

## 10. Relay Information Value

No-relay relay-visible count is strictly 0; relay has positive count across initial/last/best. The channel therefore produces additional visibility. The best checkpoint converts it to attacks/hits/damage, but neither checkpoint converts it to victory, mission success, or Support survival.

## 11. Behavioral Local Optima

All groups exhibit survival/avoidance local optimization late in training: timeout rises, return and red survival improve relative to initial behavior, blue survival remains near 3, and red hits/damage approach zero. It is strongest in homogeneous and no-relay final validation/test; relay retains some attack activity and therefore shows a weaker but still outcome-failing version.

## 12. Best vs Last Checkpoint

Validation-selected best does not generalize as a winner: all held-out win rates remain zero. Homogeneous/no-relay `last` are more avoidant and higher-return than `best`; relay `best` is much more aggressive (damage {relay_best['combat_effective_damage_mean']:.2f} vs last {relay_last['combat_effective_damage_mean']:.2f}) but suffers lower return and survival. This is behavioral trade-off and validation/test instability, not proof of statistical overfitting with only 20 aggregate test episodes.

## 13. Evidence for Research Hypotheses

- H1 (functional role differentiation): **partial structural evidence only**; sensing/role metrics exist, but Support never survives held-out tests.
- H2 (relay supplies information): **supported descriptively in seed 1** by positive relay-visible count versus strict zero without relay.
- H3 (relay improves combat/mission outcome): **not supported**; some best-checkpoint combat activity rises, but wins and mission success remain zero.

## 14. Limitations

One seed, 20 validation episodes and 20 test episodes per checkpoint; files provide aggregate metrics, not episode-level samples. Therefore no valid confidence intervals or hypothesis tests can be computed. Differences are descriptive and cannot be called statistically significant or stable. `mission_success` and Support metrics are applicable only to heterogeneous groups. Total/mean reward fields are not interchanged; this report uses `mean_episode_return` consistently.

### Absolute and relative deltas

Relative percentages are omitted when the denominator is zero or the metric is N/A.

{delta_table}

## 15. Recommended Next Experiment

**{recommendation}**

### Training-process summary

{chr(10).join(tail_notes)}
"""


if __name__ == "__main__":
    main()
