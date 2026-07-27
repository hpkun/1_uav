"""Analyze an interrupted or completed MAPPO run directory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return rows


def as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def finite_rows(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        for value in row.values():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(numeric):
                return False
    return True


def series(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [as_float(row, key) for row in rows if key in row and row.get(key) not in ("", None)]


def change(rows: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    values = series(rows, key)
    if not values:
        return {"first": None, "last": None, "delta": None, "mean": None}
    return {"first": values[0], "last": values[-1], "delta": values[-1] - values[0], "mean": float(np.mean(values))}


def last_nonzero_step(rows: list[dict[str, Any]], metric_key: str) -> int | None:
    result = None
    for row in rows:
        if as_float(row, metric_key) != 0.0:
            result = int(as_float(row, "environment_steps", as_float(row, "step", 0.0)))
    return result


def classify(metrics: list[dict[str, Any]], evaluations: list[dict[str, Any]], has_nonfinite: bool) -> str:
    if has_nonfinite:
        return "numerical_instability"
    if not metrics and not evaluations:
        return "insufficient_evidence"
    timeout = change(metrics or evaluations, "timeout_rate")
    returns = change(metrics, "rollout_team_episode_return_mean")
    if returns["delta"] is None:
        returns = change(evaluations, "mean_team_episode_return")
    attack = change(evaluations, "mean_red_attack_attempts")
    hits = change(evaluations, "mean_red_hits")
    damage = change(evaluations, "mean_red_effective_damage")
    win = change(evaluations, "overall_red_win_rate")
    attack_improved = any(item["delta"] is not None and float(item["delta"]) > 0.0 for item in (attack, hits, damage))
    if attack_improved and win["delta"] is not None and float(win["delta"]) > 0.0:
        return "active_combat_learning"
    if (
        timeout["delta"] is not None
        and float(timeout["delta"]) > 0.05
        and returns["delta"] is not None
        and float(returns["delta"]) > 0.0
        and not attack_improved
    ):
        return "stalling_or_survival_local_optimum"
    red_survivors = change(evaluations, "mean_red_survivors")
    if red_survivors["last"] is not None and float(red_survivors["last"]) <= 0.5 and timeout["last"] is not None and float(timeout["last"]) < 0.2:
        return "rapid_failure"
    return "insufficient_evidence"


def analyze_run(run_dir: Path, output_dir: Path | None = None) -> dict[str, Any]:
    output_dir = output_dir or run_dir
    metrics = read_csv(run_dir / "metrics.csv")
    evaluations = read_csv(run_dir / "evaluations.csv")
    config_path = run_dir / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    summary_path = run_dir / "final_summary.yaml"
    final_summary = yaml.safe_load(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else None
    checkpoints = sorted((run_dir / "checkpoints").glob("*.pt")) if (run_dir / "checkpoints").is_dir() else []
    has_nonfinite = not finite_rows(metrics + evaluations)
    last_metric = metrics[-1] if metrics else {}
    diagnosis = {
        "run_dir": str(run_dir),
        "has_config": bool(config),
        "has_final_summary": final_summary is not None,
        "has_last_checkpoint": (run_dir / "checkpoints" / "last.pt").is_file(),
        "checkpoint_count": len(checkpoints),
        "latest_step_checkpoint": str(max(checkpoints, key=lambda path: path.stat().st_mtime)) if checkpoints else None,
        "last_environment_step": int(as_float(last_metric, "environment_steps", as_float(last_metric, "step", 0.0))) if metrics else None,
        "updates": int(as_float(last_metric, "update", as_float(last_metric, "update_index", len(metrics)))) if metrics else 0,
        "completed_episodes": int(sum(as_float(row, "rollout_episode_count") for row in metrics)),
        "elapsed_seconds": as_float(last_metric, "elapsed_seconds", as_float(last_metric, "wall_time_seconds", 0.0)) if metrics else None,
        "samples_per_second": as_float(last_metric, "samples_per_second", as_float(last_metric, "samples_per_sec", 0.0)) if metrics else None,
        "has_nan_or_inf": has_nonfinite,
        "last_nonzero_red_attack_attempts_step": last_nonzero_step(evaluations, "mean_red_attack_attempts"),
        "last_nonzero_red_hits_step": last_nonzero_step(evaluations, "mean_red_hits"),
        "last_nonzero_red_effective_damage_step": last_nonzero_step(evaluations, "mean_red_effective_damage"),
        "trends": {
            "timeout_rate": change(metrics or evaluations, "timeout_rate"),
            "overall_red_win_rate": change(evaluations, "overall_red_win_rate"),
            "elimination_win_rate": change(evaluations, "elimination_win_rate"),
            "red_attack_attempts": change(evaluations, "mean_red_attack_attempts"),
            "red_hits": change(evaluations, "mean_red_hits"),
            "red_effective_damage": change(evaluations, "mean_red_effective_damage"),
            "red_survivors": change(evaluations, "mean_red_survivors"),
            "blue_survivors": change(evaluations, "mean_blue_survivors"),
            "training_episode_return": change(metrics, "rollout_team_episode_return_mean"),
            "validation_episode_return": change(evaluations, "mean_team_episode_return"),
            "policy_entropy": change(metrics, "policy_entropy"),
            "rollout_action_entropy": change(metrics, "rollout_action_entropy"),
            "approx_kl": change(metrics, "approx_kl"),
            "clip_fraction": change(metrics, "clip_fraction"),
            "explained_variance": change(metrics, "explained_variance"),
            "value_loss": change(metrics, "value_loss"),
            "observation_saturation": change(metrics, "observation_saturation_mean"),
        },
        "outcome_semantics": {
            "elimination_win_rate_field": "elimination_win_rate",
            "timeout_survival_win_rate_field": "timeout_survival_win_rate",
            "draw_rate_field": "draw_rate",
        },
    }
    diagnosis["behavior_pattern"] = classify(metrics, evaluations, has_nonfinite)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_diagnosis.json").write_text(json.dumps(diagnosis, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# MAPPO Run Diagnosis",
        "",
        f"behavior_pattern: `{diagnosis['behavior_pattern']}`",
        f"last_environment_step: {diagnosis['last_environment_step']}",
        f"updates: {diagnosis['updates']}",
        f"completed_episodes: {diagnosis['completed_episodes']}",
        f"has_nan_or_inf: {diagnosis['has_nan_or_inf']}",
        f"has_final_summary: {diagnosis['has_final_summary']}",
        f"has_last_checkpoint: {diagnosis['has_last_checkpoint']}",
        "",
        "Timeout survivor-count wins, elimination wins, and draws are reported as separate fields.",
    ]
    (output_dir / "run_diagnosis.md").write_text("\n".join(lines), encoding="utf-8")
    return diagnosis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    diagnosis = analyze_run(args.run_dir, args.output_dir)
    print(json.dumps({"run_dir": str(args.run_dir), "behavior_pattern": diagnosis["behavior_pattern"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
