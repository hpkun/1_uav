from __future__ import annotations

import math
import zipfile
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
SUPPORT_POSITION_TOLERANCE_M = 900.0

RUNS = {
    "homogeneous_control": ROOT / "outputs/mappo/functional_homogeneous_3v3_seed1" / BATCH,
    "heterogeneous_no_relay": ROOT / "outputs/mappo/functional_heterogeneous_no_relay_3v3_seed1" / BATCH,
    "heterogeneous_relay": ROOT / "outputs/mappo/functional_heterogeneous_relay_3v3_seed1" / BATCH,
}
LABEL = {
    "homogeneous_control": "Homogeneous",
    "heterogeneous_no_relay": "Heterogeneous No Relay",
    "heterogeneous_relay": "Heterogeneous Relay",
}
COLORS = {
    "homogeneous_control": "#4472C4",
    "heterogeneous_no_relay": "#ED7D31",
    "heterogeneous_relay": "#70AD47",
}

RESULT_FIELDS = [
    "overall_red_win_rate",
    "elimination_red_win_rate",
    "timeout_survivor_red_win_rate",
    "blue_win_rate",
    "draw_rate",
    "timeout_rate",
    "mean_episode_return",
    "mean_episode_steps",
    "mean_red_survivors",
    "mean_blue_survivors",
    "mean_survivor_difference",
    "mean_red_hits",
    "mean_blue_hits",
    "mean_red_effective_damage",
    "mean_blue_effective_damage",
    "red_crash_rate",
    "blue_crash_rate",
    "mean_collisions",
]
FUNCTIONAL_FIELDS = [
    "has_support_agent",
    "support_metrics_applicable",
    "support_survival_rate",
    "mission_success_rate",
    "support_detection_coverage_mean",
    "relay_visible_enemy_count_mean",
    "support_incoming_threat_mean",
    "support_position_error_mean",
    "combat_attack_attempts_mean",
    "combat_hits_mean",
    "combat_effective_damage_mean",
]
SUPPORT_PERFORMANCE_FIELDS = {
    "support_survival_rate",
    "mission_success_rate",
    "support_detection_coverage_mean",
    "relay_visible_enemy_count_mean",
    "support_incoming_threat_mean",
    "support_position_error_mean",
}
POLICY_FIELDS = [
    "policy_entropy_mean",
    "logits_top1_top2_margin_mean",
    "terminal_reward_proportion",
    "mean_observation_saturation_ratio",
]
ALIASES = {
    "elimination_red_win_rate": "elimination_win_rate",
    "timeout_survivor_red_win_rate": "timeout_survival_win_rate",
}
PERCENT_METRICS = {
    "overall_red_win_rate",
    "elimination_red_win_rate",
    "mission_success_rate",
    "support_survival_rate",
    "timeout_rate",
    "mean_red_survivors",
    "mean_blue_survivors",
    "combat_attack_attempts_mean",
    "combat_hits_mean",
    "combat_effective_damage_mean",
    "support_detection_coverage_mean",
    "relay_visible_enemy_count_mean",
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def finite_csv(df: pd.DataFrame) -> tuple[int, int, int]:
    numeric = df.select_dtypes(include=[np.number])
    return (
        int(numeric.isna().sum().sum()),
        int(np.isinf(numeric.to_numpy()).sum()),
        int(df.isna().all(axis=0).sum()),
    )


def val(d: dict[str, Any], name: str) -> Any:
    return d.get(name, d.get(ALIASES.get(name, ""), np.nan))


def applicable_value(mode: str, field: str, value: Any) -> Any:
    if mode == "homogeneous_control" and field in SUPPORT_PERFORMANCE_FIELDS:
        return np.nan
    return value


def fmt(x: Any, digits: int = 3) -> str:
    if x is None:
        return "N/A"
    try:
        if pd.isna(x):
            return "N/A"
    except TypeError:
        pass
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.{digits}f}"
    return str(x)


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    output.extend("| " + " | ".join(fmt(item) for item in row) + " |" for row in rows)
    return "\n".join(output)


def read_checkpoint_metadata(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "metadata_read_ok": False,
        "metadata_read_error": "",
        "environment_steps": np.nan,
        "schema": "",
        "obs_dim": np.nan,
        "state_dim": np.nan,
    }
    if not path.exists():
        result["metadata_read_error"] = "missing checkpoint"
        return result
    try:
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(checkpoint, dict):
            raise TypeError(f"unexpected checkpoint type: {type(checkpoint).__name__}")
        schema = checkpoint.get("schema_metadata")
        if schema is None:
            schema = checkpoint.get("metadata", {}).get("schema_metadata", {})
        if not isinstance(schema, dict):
            schema = {}
        result.update(
            {
                "metadata_read_ok": True,
                "environment_steps": checkpoint.get("environment_steps", np.nan),
                "schema": schema.get("environment_schema_version", ""),
                "obs_dim": schema.get("obs_dim", np.nan),
                "state_dim": schema.get("state_dim", np.nan),
            }
        )
    except Exception as exc:  # pragma: no cover - diagnostic path
        result["metadata_read_error"] = f"{type(exc).__name__}: {exc}"
        fallback = read_checkpoint_metadata_from_pickle(path)
        if fallback:
            fallback["metadata_read_error"] = ""
            return {**result, **fallback}
    return result


def read_checkpoint_metadata_from_pickle(path: Path) -> dict[str, Any]:
    if not zipfile.is_zipfile(path):
        return {}
    with zipfile.ZipFile(path) as archive:
        data_name = next((name for name in archive.namelist() if name.endswith("data.pkl")), "")
        if not data_name:
            return {}
        data = archive.read(data_name)

    def parse_int(key: bytes) -> int | float:
        pos = data.find(key)
        if pos < 0:
            return np.nan
        pos += len(key)
        for idx in range(pos, min(pos + 64, len(data))):
            code = data[idx : idx + 1]
            if code == b"K":
                return int(data[idx + 1])
            if code == b"M":
                return int.from_bytes(data[idx + 1 : idx + 3], "little", signed=False)
            if code == b"J":
                return int.from_bytes(data[idx + 1 : idx + 5], "little", signed=True)
            if code in {b"X", b"}", b"u"}:
                break
        return np.nan

    def parse_string_after(key: bytes) -> str:
        pos = data.find(key)
        if pos < 0:
            return ""
        pos += len(key)
        marker = data.find(b"X", pos, min(pos + 64, len(data)))
        if marker < 0:
            return ""
        length = int.from_bytes(data[marker + 1 : marker + 5], "little", signed=False)
        return data[marker + 5 : marker + 5 + length].decode("utf-8", errors="replace")

    return {
        "metadata_read_ok": True,
        "environment_steps": parse_int(b"environment_steps"),
        "schema": parse_string_after(b"environment_schema_version"),
        "obs_dim": parse_int(b"obs_dim"),
        "state_dim": parse_int(b"state_dim"),
    }


def safe_int_equal(value: Any, expected: int) -> bool:
    try:
        if pd.isna(value):
            return False
        return int(value) == int(expected)
    except (TypeError, ValueError):
        return False


def checkpoint_environment_steps(summary: dict[str, Any], checkpoint: str) -> int:
    if checkpoint == "initial":
        return 0
    if checkpoint == "last":
        return int(summary["environment_steps"])
    return int(summary["validation_best_evaluation"]["environment_steps"])


def validation_best_reason(summary: dict[str, Any]) -> str:
    best = summary["validation_best_evaluation"]
    return (
        "combat-ranked validation best checkpoint selected by tuple "
        f"({best.get('elimination_win_rate', 0.0):.3f}, "
        f"{best.get('overall_red_win_rate', 0.0):.3f}, "
        f"{best.get('mean_effective_damage', best.get('mean_red_effective_damage', 0.0)):.3f}, "
        f"{best.get('mean_survivor_difference', 0.0):.3f}, "
        f"{best.get('mean_hits', best.get('mean_red_hits', 0.0)):.3f}, "
        f"{best.get('mean_attack_area_steps', best.get('mean_red_attack_area_steps', 0.0)):.3f}, "
        f"{best.get('mean_team_episode_return', best.get('mean_episode_return', 0.0)):.3f}, "
        f"{-best.get('red_crash_rate', 0.0):.3f}, {-best.get('timeout_rate', 0.0):.3f})"
    )


def best_window(df: pd.DataFrame, col: str, criterion: str) -> tuple[str, pd.DataFrame | None, float]:
    rolling = df[col].rolling(10, min_periods=10).mean()
    valid = rolling.dropna()
    if valid.empty:
        return "missing", None, np.nan
    max_value = float(valid.max())
    if max_value == 0.0 and (valid == 0.0).all():
        return "tied_all_zero", None, 0.0
    if (valid == max_value).all():
        return "tied_all", None, max_value
    winners = valid.index[valid == max_value].tolist()
    if len(winners) != 1:
        return "tied_best", None, max_value
    end = int(winners[0])
    return "selected", df.iloc[end - 9 : end + 1], max_value


def window_row(mode: str, name: str, status: str, window: pd.DataFrame | None, metric_value: float | None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "mode": mode,
        "window": name,
        "status": status,
        "updates": 0 if window is None else len(window),
        "start_update": np.nan if window is None else int(window.update_index.iloc[0]),
        "end_update": np.nan if window is None else int(window.update_index.iloc[-1]),
        "start_environment_steps": np.nan if window is None else int(window.environment_steps.iloc[0]),
        "end_environment_steps": np.nan if window is None else int(window.environment_steps.iloc[-1]),
        "selection_metric_value": metric_value if metric_value is not None else np.nan,
    }
    tail_map = {
        "rollout_return": "rollout_return_mean",
        "timeout_rate": "timeout_rate",
        "red_hits": "rollout_red_hits_mean",
        "red_effective_damage": "rollout_red_effective_damage_mean",
        "blue_hits": "rollout_blue_hits_mean",
        "blue_effective_damage": "rollout_blue_effective_damage_mean",
        "entropy": "rollout_action_entropy",
        "support_survival": "rollout_support_survival_rate",
        "support_coverage": "rollout_support_detection_coverage_mean",
        "relay_visible": "rollout_relay_visible_enemy_count_mean",
        "support_incoming_threat": "rollout_support_incoming_threat_mean",
        "support_position_error": "rollout_support_position_error_mean",
        "mission_success": "rollout_mission_success_rate",
    }
    for short, col in tail_map.items():
        row[f"{short}_mean"] = np.nan if window is None else float(window[col].mean())
        row[f"{short}_std"] = np.nan if window is None else float(window[col].std(ddof=1))
    if window is None:
        row["final_cumulative_sps"] = np.nan
        row["min_cumulative_sps"] = np.nan
        row["max_cumulative_sps"] = np.nan
    else:
        row["final_cumulative_sps"] = float(window.samples_per_second.iloc[-1])
        row["min_cumulative_sps"] = float(window.samples_per_second.min())
        row["max_cumulative_sps"] = float(window.samples_per_second.max())
    return row


def build_training_windows(metrics: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for mode, df in metrics.items():
        for name, window in (("all", df), ("last_10", df.tail(10)), ("last_20", df.tail(20))):
            rows.append(window_row(mode, name, "selected", window, None))
        for criterion, col in (
            ("return", "rollout_return_mean"),
            ("red_damage", "rollout_red_effective_damage_mean"),
            ("mission_success", "rollout_mission_success_rate"),
        ):
            status, window, metric_value = best_window(df, col, criterion)
            rows.append(window_row(mode, f"best_10_{criterion}", status, window, metric_value))
    return pd.DataFrame(rows)


def add_support_na(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    mask = output["mode"] == "homogeneous_control"
    for field in SUPPORT_PERFORMANCE_FIELDS:
        if field in output:
            output.loc[mask, field] = np.nan
    return output


def make_figures(summaries: dict[str, Any], metrics: dict[str, pd.DataFrame], evals: dict[str, pd.DataFrame]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    def line_plot(filename: str, fields: list[tuple[str, str]]) -> None:
        fig, axes = plt.subplots(len(fields), 1, figsize=(9, 3.3 * len(fields)), sharex=True, squeeze=False)
        for ax, (field, title) in zip(axes[:, 0], fields):
            for mode, df in evals.items():
                if mode == "homogeneous_control" and field in SUPPORT_PERFORMANCE_FIELDS:
                    continue
                ax.plot(df.environment_steps, df[field], marker="o", label=LABEL[mode], color=COLORS[mode])
            ax.set_title(title)
            ax.set_xlim(0, 305000)
        axes[-1, 0].set_xlabel("Environment steps")
        axes[0, 0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / filename, dpi=180)
        plt.close(fig)

    line_plot("validation_return.png", [("mean_episode_return", "Validation mean episode return")])
    line_plot(
        "validation_outcomes.png",
        [
            ("overall_red_win_rate", "Overall red win rate"),
            ("blue_win_rate", "Blue win rate"),
            ("draw_rate", "Draw rate"),
            ("timeout_rate", "Timeout rate (not a winner class)"),
        ],
    )
    line_plot("validation_survivors.png", [("mean_red_survivors", "Mean red survivors"), ("mean_blue_survivors", "Mean blue survivors")])
    line_plot("validation_combat_effectiveness.png", [("combat_hits_mean", "Combat hits"), ("combat_effective_damage_mean", "Combat effective damage")])
    line_plot(
        "validation_support_metrics.png",
        [
            ("support_survival_rate", "Support survival"),
            ("support_detection_coverage_mean", "Support detection coverage"),
            ("relay_visible_enemy_count_mean", "Relay-visible enemies"),
            ("mission_success_rate", "Mission success"),
        ],
    )

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for mode, df in metrics.items():
        ax.plot(df.environment_steps, df.rollout_action_entropy, label=LABEL[mode], color=COLORS[mode])
    ax.set(xlabel="Environment steps", ylabel="Entropy", title="Training action entropy", xlim=(0, 305000))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "training_entropy.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    checkpoints = ["initial", "last", "best"]
    xs = np.arange(3)
    width = 0.24
    for ax, field, title in zip(
        axes.flat,
        ["overall_red_win_rate", "mean_episode_return", "mean_red_survivors", "combat_effective_damage_mean"],
        ["Red win rate", "Episode return", "Red survivors", "Combat damage"],
    ):
        for i, mode in enumerate(RUNS):
            values = [val(summaries[mode]["test_evaluations"][checkpoint], field) for checkpoint in checkpoints]
            ax.bar(xs + (i - 1) * width, values, width, label=LABEL[mode], color=COLORS[mode])
        ax.set_xticks(xs, ["initial", "last", "combat-ranked validation best"])
        ax.set_title(title)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "test_checkpoint_comparison.png", dpi=180)
    plt.close(fig)


def absolute_deltas(summaries: dict[str, Any]) -> pd.DataFrame:
    pairs = [
        ("heterogeneous_no_relay", "homogeneous_control"),
        ("heterogeneous_relay", "heterogeneous_no_relay"),
        ("heterogeneous_relay", "homogeneous_control"),
    ]
    fields = [
        "overall_red_win_rate",
        "elimination_red_win_rate",
        "blue_win_rate",
        "draw_rate",
        "timeout_rate",
        "mean_episode_return",
        "mean_survivor_difference",
        "mean_red_survivors",
        "mean_blue_survivors",
        "combat_attack_attempts_mean",
        "combat_hits_mean",
        "combat_effective_damage_mean",
        "support_survival_rate",
        "mission_success_rate",
        "support_detection_coverage_mean",
        "relay_visible_enemy_count_mean",
        "support_incoming_threat_mean",
        "support_position_error_mean",
    ]
    rows = []
    for checkpoint in ("last", "best"):
        for left, right in pairs:
            left_data = summaries[left]["test_evaluations"][checkpoint]
            right_data = summaries[right]["test_evaluations"][checkpoint]
            for field in fields:
                left_value = applicable_value(left, field, val(left_data, field))
                right_value = applicable_value(right, field, val(right_data, field))
                rows.append(
                    [
                        checkpoint,
                        f"{left} - {right}",
                        field,
                        left_value - right_value if pd.notna(left_value) and pd.notna(right_value) else np.nan,
                    ]
                )
    return pd.DataFrame(rows, columns=["checkpoint", "contrast", "metric", "absolute_delta"])


def build_report(
    configs: dict[str, Any],
    summaries: dict[str, Any],
    metrics: dict[str, pd.DataFrame],
    evals: dict[str, pd.DataFrame],
    integrity_df: pd.DataFrame,
    experiment_summary: pd.DataFrame,
    training_summary: pd.DataFrame,
    deltas: pd.DataFrame,
) -> str:
    core_fields = [
        "overall_red_win_rate",
        "blue_win_rate",
        "draw_rate",
        "timeout_rate",
        "elimination_red_win_rate",
        "mean_episode_return",
        "mean_red_survivors",
        "mean_blue_survivors",
        "combat_attack_attempts_mean",
        "combat_hits_mean",
        "combat_effective_damage_mean",
        "support_survival_rate",
        "support_detection_coverage_mean",
        "relay_visible_enemy_count_mean",
        "support_incoming_threat_mean",
        "support_position_error_mean",
        "mission_success_rate",
    ]
    rows = []
    for mode in RUNS:
        for checkpoint in ("last", "best"):
            record = experiment_summary[(experiment_summary["mode"] == mode) & (experiment_summary["checkpoint"] == checkpoint)].iloc[0]
            rows.append([LABEL[mode], checkpoint] + [record[field] for field in core_fields])
    test_table = md_table(["mode", "checkpoint"] + core_fields, rows)

    integrity_table = md_table(
        [
            "mode",
            "complete",
            "steps",
            "updates",
            "episodes",
            "best step",
            "ckpt metadata ok",
            "schema",
            "obs/state",
        ],
        [
            [
                row.mode,
                row.complete,
                row.environment_steps,
                row.updates,
                row.episodes,
                row.best_checkpoint_step,
                bool(row.initial_metadata_read_ok and row.last_metadata_read_ok and row.best_metadata_read_ok),
                row.best_checkpoint_schema,
                f"{row.best_obs_dim}/{row.best_state_dim}",
            ]
            for row in integrity_df.itertuples()
        ],
    )

    best_rows = []
    for mode in RUNS:
        best_rows.append([LABEL[mode], summaries[mode]["validation_best_evaluation"]["environment_steps"], validation_best_reason(summaries[mode])])

    support_rows = []
    for mode in ("heterogeneous_no_relay", "heterogeneous_relay"):
        for checkpoint in ("last", "best"):
            d = summaries[mode]["test_evaluations"][checkpoint]
            support_rows.append(
                [
                    LABEL[mode],
                    checkpoint,
                    d["support_position_error_mean"],
                    d["support_position_error_mean"] <= SUPPORT_POSITION_TOLERANCE_M,
                    d["support_incoming_threat_mean"],
                    d["support_detection_coverage_mean"],
                    d["relay_visible_enemy_count_mean"],
                    d["support_survival_rate"],
                    d["mission_success_rate"],
                ]
            )

    final_sps_rows = [
        [LABEL[mode], metrics[mode].samples_per_second.iloc[-1], metrics[mode].samples_per_second.min(), metrics[mode].samples_per_second.max()]
        for mode in RUNS
    ]

    relay_delta = deltas[(deltas["checkpoint"] == "best") & (deltas["contrast"] == "heterogeneous_relay - heterogeneous_no_relay")]
    relay_delta_last = deltas[(deltas["checkpoint"] == "last") & (deltas["contrast"] == "heterogeneous_relay - heterogeneous_no_relay")]
    delta_table = md_table(["checkpoint", "contrast", "metric", "absolute delta"], deltas.values.tolist())

    window_table = md_table(
        ["mode", "window", "status", "start update", "end update", "start steps", "end steps", "selection value"],
        training_summary[
            [
                "mode",
                "window",
                "status",
                "start_update",
                "end_update",
                "start_environment_steps",
                "end_environment_steps",
                "selection_metric_value",
            ]
        ].values.tolist(),
    )

    def metric_delta(df: pd.DataFrame, name: str) -> float:
        return float(df[df["metric"] == name]["absolute_delta"].iloc[0])

    hlast = summaries["homogeneous_control"]["test_evaluations"]["last"]
    nr_last = summaries["heterogeneous_no_relay"]["test_evaluations"]["last"]
    nr_best = summaries["heterogeneous_no_relay"]["test_evaluations"]["best"]
    r_last = summaries["heterogeneous_relay"]["test_evaluations"]["last"]
    r_best = summaries["heterogeneous_relay"]["test_evaluations"]["best"]

    return f"""# Functional 3v3 MAPPO Comparison, Revised Seed-1 Analysis

Batch: `{BATCH}`

## Scope

This report reads the three completed result directories only. It does not retrain, reevaluate checkpoints, or modify environment, reward, algorithm, configuration, output, log, or checkpoint files. The canonical analysis products are regenerated in `analysis/functional_3v3_seed1_20260730_213216/`.

## Research Questions and Judgments

H1: whether functional role differentiation forms. Evidence includes Combat vs Support structural role differences, Support being unarmed with longer-range sensing, support coverage, support position error, incoming threat, support survival, mission success, and whether stable rear support, survival, and information coverage behavior appears. Judgment: **partially supported** at the structural and sensing levels, **not supported** at the successful behavioral level. Support metrics exist and coverage is nonzero, but held-out Support survival and mission success are 0.

H2: whether relay compensates for the firepower loss caused by removing one armed Combat UAV. The primary comparison is heterogeneous relay vs homogeneous control, with relay vs no-relay as a mechanism check. Judgment: **not supported**. Relay is accompanied by nonzero extra visibility and, at the combat-ranked validation best checkpoint, more attacks/hits/damage than no-relay, but it does not reduce blue survivors, create red wins, produce elimination wins, preserve Support survival, or produce mission success.

H3: whether the experiment successfully isolates functional heterogeneity under identical maneuverability. Judgment: **design condition satisfied**. The three runs keep dynamics, network dimensions, initial scenario, MAPPO hyperparameters, seed, opponent, rollout length, environment steps, observation/state schemas, and training schedule fixed. The intended differences are role/weapon/sensing/reward/relay semantics.

## Checkpoint Selection

`checkpoint_selection: combat` uses the code path in `src/uav_env/algorithms/mappo/metrics.py::evaluation_key`. Validation checkpoints are ranked lexicographically by:

1. elimination red win rate, higher is better
2. overall red win rate, higher is better
3. red effective damage through `mean_effective_damage` or `mean_red_effective_damage`, higher is better
4. survivor difference, higher is better
5. red hits through `mean_hits` or `mean_red_hits`, higher is better
6. attack-area steps through `mean_attack_area_steps` or `mean_red_attack_area_steps`, higher is better
7. mean team episode return or mean episode return, higher is better
8. red crash rate, lower is better by negation
9. timeout rate, lower is better by negation

The selected checkpoint should be called **combat-ranked validation best checkpoint**. It is not the best held-out test checkpoint, not the best return checkpoint, not the best survival checkpoint, and not a global optimum. Because all validation red win rates are 0 here, later tuple fields such as effective damage determine the selected `best`.

{md_table(["mode", "best environment steps", "ranking tuple"], best_rows)}

Homogeneous best vs last reflects an attack/survival trade-off: `best` has more combat damage (4.80 vs 0.00 held-out) but much worse return and red survival. Relay best vs last reflects a stronger attack/avoidance trade-off: `best` has 74.00 damage and 3.50 hits but lower return and fewer red survivors than `last`. No-relay best is selected by validation combat ranking despite held-out combat output remaining 0. Validation combat ranking and held-out test behavior can diverge.

## Artifact and Metadata Integrity

{integrity_table}

All three runs reach 301,056 rollout-aligned environment steps. All required artifacts are present. Checkpoint metadata is readable for `initial.pt`, `last.pt`, and `best.pt`; checkpoint schemas are `functional_heterogeneous_3v3_v1`, obs dim is 69, and state dim is 64. `last.pt` records 301,056 steps in all runs, while `best.pt` steps match `final_summary.yaml` and validation records.

## Held-Out Test Core Table

`timeout_rate` is a termination condition, not a winner class; it should not be added to red/blue/draw rates. Homogeneous Support performance fields are N/A because no Support agent exists.

{test_table}

Homogeneous test-last is a strong timeout survival/delay policy: overall red win 0.00, blue win 0.85, draw 0.15, timeout 1.00, red survivors 2.05, blue survivors 3.00, combat damage 0.00. It partially forms draws through timeout but never red victory.

## Support Behavior

Support position tolerance is {SUPPORT_POSITION_TOLERANCE_M:.0f} m.

{md_table(["mode", "checkpoint", "position error m", "<=900m", "incoming threat", "coverage", "relay-visible enemies", "support survival", "mission success"], support_rows)}

No-relay last has perfect coverage (1.000) but position error is 1038.23 m, above tolerance, and Support survival is 0. No-relay best has position error 630.14 m and coverage 0.990, but still has Support survival 0 and mission success 0.

Relay last has lower position error (531.50 m) than relay best and is within tolerance, yet Support still does not survive; the likely descriptive interpretation is that rear positioning and visibility alone did not prevent lethal exposure under the current policy/opponent. Relay best has much stronger combat activity, but this is accompanied by much larger position error (1810.12 m), higher incoming threat (0.384), Support survival 0, and mission success 0. Coverage and relay-visible information are meaningful, but current held-out results prove sensing capability, not stable support-role behavior.

## Relay vs No-Relay Absolute Deltas

At held-out last, relay minus no-relay has return delta {metric_delta(relay_delta_last, "mean_episode_return"):.2f}, red-survivor delta {metric_delta(relay_delta_last, "mean_red_survivors"):.2f}, attack-attempt delta {metric_delta(relay_delta_last, "combat_attack_attempts_mean"):.2f}, hit delta {metric_delta(relay_delta_last, "combat_hits_mean"):.2f}, damage delta {metric_delta(relay_delta_last, "combat_effective_damage_mean"):.2f}, coverage delta {metric_delta(relay_delta_last, "support_detection_coverage_mean"):.3f}, relay-visible delta {metric_delta(relay_delta_last, "relay_visible_enemy_count_mean"):.3f}, support-survival delta {metric_delta(relay_delta_last, "support_survival_rate"):.2f}, and mission-success delta {metric_delta(relay_delta_last, "mission_success_rate"):.2f}.

At the combat-ranked validation best checkpoint, relay minus no-relay has attack-attempt delta {metric_delta(relay_delta, "combat_attack_attempts_mean"):.2f}, hit delta {metric_delta(relay_delta, "combat_hits_mean"):.2f}, damage delta {metric_delta(relay_delta, "combat_effective_damage_mean"):.2f}, red-survivor delta {metric_delta(relay_delta, "mean_red_survivors"):.2f}, blue-survivor delta {metric_delta(relay_delta, "mean_blue_survivors"):.2f}, win delta {metric_delta(relay_delta, "overall_red_win_rate"):.2f}, elimination-win delta {metric_delta(relay_delta, "elimination_red_win_rate"):.2f}, Support-survival delta {metric_delta(relay_delta, "support_survival_rate"):.2f}, and mission-success delta {metric_delta(relay_delta, "mission_success_rate"):.2f}. These are absolute descriptive deltas; no relative percentages are reported for returns, survivor difference, position error, incoming threat, or zero-denominator cases.

{delta_table}

## Training Windows

Training-window statistics use `last_10`, `last_20`, and 10-update sliding windows. `best_10_mission_success` is marked `tied_all_zero` for all three modes because mission success is zero throughout training; there is no meaningful best mission-success window.

{window_table}

Final cumulative SPS is the last cumulative `samples_per_second` value, not the mean of cumulative SPS values.

{md_table(["mode", "final cumulative SPS", "min cumulative SPS", "max cumulative SPS"], final_sps_rows)}

## Validation Trajectory Summary

Validation contains six scheduled evaluations per mode. All validation red win rates are 0. Homogeneous/no-relay late checkpoints mostly move toward timeout survival and low combat output. Relay has a combat-active validation best at 51,200 steps, but later training reduces combat activity while not creating held-out wins.

## Evidence Boundary

The following statements are supported: relay produced nonzero extra visible enemies; relay combat-ranked best was accompanied by more attack attempts, hits, and damage than no-relay best; these phenomena appeared together in seed 1; the extra information did not convert to red win rate, elimination, lower blue survival, Support survival, or mission success.

The following stronger statements are not supported by this dataset: relay caused an attack capability improvement; relay significantly improved attack; relay proved effective; relay compensated for the lost Combat UAV; functional heterogeneity improved mission performance.

## Final Conclusions

1. All three groups have zero held-out red wins and zero elimination wins.
2. Homogeneous and no-relay last checkpoints mainly show survival/avoidance local optima.
3. The relay link actually provides additional visible targets.
4. Relay combat-ranked best shows more combat activity than no-relay best.
5. Attack activity does not convert into lower blue survival, red wins, eliminations, Support survival, or mission success.
6. H1 is partially supported only structurally and perceptually; behavioral role formation is not supported.
7. H2 is not supported.
8. H3's controlled design condition is satisfied.
9. Results are descriptive for seed 1 and 20 held-out test episodes; statistical significance is not claimed.
10. The highest-priority next step is to keep configuration unchanged and add independent seed replications.
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    configs: dict[str, Any] = {}
    summaries: dict[str, Any] = {}
    metrics: dict[str, pd.DataFrame] = {}
    evals: dict[str, pd.DataFrame] = {}
    integrity_rows: list[dict[str, Any]] = []

    for mode, run in RUNS.items():
        configs[mode] = load_yaml(run / "config.yaml")
        summaries[mode] = load_yaml(run / "final_summary.yaml")
        metrics[mode] = pd.read_csv(run / "metrics.csv")
        evals[mode] = pd.read_csv(run / "evaluations.csv")
        ckpt = {name: read_checkpoint_metadata(run / "checkpoints" / f"{name}.pt") for name in ("initial", "last", "best")}

        metric_nan, metric_inf, metric_empty = finite_csv(metrics[mode])
        eval_nan, eval_inf, eval_empty = finite_csv(evals[mode])
        config = configs[mode]
        summary = summaries[mode]
        schema = summary["schema_metadata"]
        required = [
            "config.yaml",
            "metrics.csv",
            "evaluations.csv",
            "final_summary.yaml",
            "checkpoints/initial.pt",
            "checkpoints/last.pt",
            "checkpoints/best.pt",
        ]
        missing = [name for name in required if not (run / name).exists()]
        expected_steps = math.ceil(config["total_env_steps"] / (config["num_envs"] * config["rollout_length"])) * config["num_envs"] * config["rollout_length"]
        best_step = int(summary["validation_best_evaluation"]["environment_steps"])
        best_step_in_validation = best_step in set(evals[mode].environment_steps.astype(int))

        row: dict[str, Any] = {
            "mode": mode,
            "complete": not missing and int(summary["environment_steps"]) == expected_steps,
            "missing_artifacts": ";".join(missing) if missing else "none",
            "environment_steps": summary["environment_steps"],
            "expected_rollout_aligned_steps": expected_steps,
            "updates": summary["updates"],
            "metrics_rows": len(metrics[mode]),
            "episodes": summary["episodes"],
            "metrics_last_step": int(metrics[mode].environment_steps.iloc[-1]),
            "evaluations_last_step": int(evals[mode].environment_steps.iloc[-1]),
            "final_summary_step": summary["environment_steps"],
            "best_checkpoint_step": best_step,
            "best_step_in_validation": best_step_in_validation,
            "test_initial": "initial" in summary.get("test_evaluations", {}),
            "test_last": "last" in summary.get("test_evaluations", {}),
            "test_best": "best" in summary.get("test_evaluations", {}),
            "schema": schema.get("environment_schema_version"),
            "obs_dim": schema.get("obs_dim"),
            "state_dim": schema.get("state_dim"),
            "seed": config.get("seed"),
            "num_envs": config.get("num_envs"),
            "rollout_length": config.get("rollout_length"),
            "opponent": config["environment"].get("opponent"),
            "functional_mode": config["environment"].get("functional_mode"),
            "relay_enabled": config["environment"].get("relay_enabled"),
            "metrics_nan_cells": metric_nan,
            "metrics_inf_cells": metric_inf,
            "metrics_empty_columns": metric_empty,
            "evaluations_nan_cells": eval_nan,
            "evaluations_inf_cells": eval_inf,
            "evaluations_empty_columns": eval_empty,
        }
        errors = []
        for name in ("initial", "last", "best"):
            meta = ckpt[name]
            row[f"{name}_checkpoint"] = meta["exists"]
            row[f"{name}_metadata_read_ok"] = meta["metadata_read_ok"]
            row[f"{name}_checkpoint_environment_steps"] = meta["environment_steps"]
            row[f"{name}_checkpoint_schema"] = meta["schema"]
            row[f"{name}_obs_dim"] = meta["obs_dim"]
            row[f"{name}_state_dim"] = meta["state_dim"]
            if meta["metadata_read_error"]:
                errors.append(f"{name}: {meta['metadata_read_error']}")
        row["metadata_read_error"] = " | ".join(errors)
        row["last_checkpoint_step_ok"] = safe_int_equal(row["last_checkpoint_environment_steps"], int(summary["environment_steps"]))
        row["best_checkpoint_step_ok"] = safe_int_equal(row["best_checkpoint_environment_steps"], best_step)
        row["initial_checkpoint_step_reasonable"] = safe_int_equal(row["initial_checkpoint_environment_steps"], 0)
        row["checkpoint_schema_ok"] = all(row[f"{name}_checkpoint_schema"] == "functional_heterogeneous_3v3_v1" for name in ("initial", "last", "best"))
        row["checkpoint_dims_ok"] = all(int(row[f"{name}_obs_dim"]) == 69 and int(row[f"{name}_state_dim"]) == 64 for name in ("initial", "last", "best"))
        integrity_rows.append(row)

    integrity_df = pd.DataFrame(integrity_rows)
    integrity_df.to_csv(OUT / "artifact_integrity.csv", index=False, na_rep="missing")

    summary_rows: list[dict[str, Any]] = []
    for mode, summary in summaries.items():
        for checkpoint, data in summary["test_evaluations"].items():
            row = {
                "mode": mode,
                "checkpoint": checkpoint,
                "checkpoint_environment_steps": checkpoint_environment_steps(summary, checkpoint),
            }
            for field in RESULT_FIELDS + FUNCTIONAL_FIELDS + POLICY_FIELDS:
                row[field] = applicable_value(mode, field, val(data, field))
            summary_rows.append(row)
    experiment_summary = pd.DataFrame(summary_rows)
    experiment_summary.to_csv(OUT / "experiment_summary.csv", index=False, na_rep="missing")

    trajectory_rows = []
    for mode, df in evals.items():
        for record in df.to_dict("records"):
            item = {
                **record,
                "mode": mode,
                "is_validation_best": int(record["environment_steps"]) == int(summaries[mode]["validation_best_evaluation"]["environment_steps"]),
            }
            for field in SUPPORT_PERFORMANCE_FIELDS:
                if mode == "homogeneous_control" and field in item:
                    item[field] = np.nan
            trajectory_rows.append(item)
    trajectory_df = pd.DataFrame(trajectory_rows)
    trajectory_df.to_csv(OUT / "validation_trajectory.csv", index=False, na_rep="missing")

    training_summary = build_training_windows(metrics)
    training_summary.to_csv(OUT / "training_tail_summary.csv", index=False, na_rep="missing")

    make_figures(summaries, metrics, evals)
    deltas = absolute_deltas(summaries)
    report = build_report(configs, summaries, metrics, evals, integrity_df, experiment_summary, training_summary, deltas)
    (OUT / "comparison_report.md").write_text(report, encoding="utf-8")

    print("Analysis complete")
    print(integrity_df[["mode", "complete", "environment_steps", "updates", "episodes", "best_checkpoint_step", "last_checkpoint_step_ok", "best_checkpoint_step_ok"]].to_string(index=False))
    print()
    print("Held-out test last/best core metrics")
    print(
        experiment_summary[experiment_summary["checkpoint"].isin(["last", "best"])][
            [
                "mode",
                "checkpoint",
                "overall_red_win_rate",
                "blue_win_rate",
                "draw_rate",
                "timeout_rate",
                "mean_episode_return",
                "mean_red_survivors",
                "mean_blue_survivors",
                "combat_hits_mean",
                "combat_effective_damage_mean",
                "support_survival_rate",
                "support_detection_coverage_mean",
                "relay_visible_enemy_count_mean",
                "support_position_error_mean",
                "support_incoming_threat_mean",
                "mission_success_rate",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
