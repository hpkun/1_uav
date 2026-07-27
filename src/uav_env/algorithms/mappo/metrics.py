"""CSV metrics and evaluation ranking helpers."""

from __future__ import annotations
import csv
from pathlib import Path
from typing import Any

def combat_outcome_rates(outcomes: list[Any]) -> dict[str,float]:
    """Separate combat termination wins from timeout survivor-count wins."""

    if not outcomes: raise ValueError("At least one outcome is required")
    count=len(outcomes)
    return {
        "overall_red_win_rate":sum(o.winner=="red" for o in outcomes)/count,
        "elimination_win_rate":sum(o.winner=="red" and o.termination_reason=="blue_eliminated" for o in outcomes)/count,
        "timeout_survival_win_rate":sum(o.winner=="red" and o.termination_reason=="timeout" for o in outcomes)/count,
        "decisive_win_rate":sum(o.winner=="red" and o.termination_reason!="timeout" for o in outcomes)/count,
        "draw_rate":sum(o.winner=="draw" for o in outcomes)/count,
        "timeout_rate":sum(o.termination_reason=="timeout" for o in outcomes)/count,
    }

def append_csv(path: Path, row: dict[str,Any]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); exists=path.exists()
    with path.open("a",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(row));
        if not exists: writer.writeheader()
        writer.writerow(row)

def evaluation_key(result: dict[str,Any], selection: str = "smoke") -> tuple[float, ...]:
    """Rank validation evaluations using either smoke or combat semantics."""

    if selection == "combat":
        survivor_difference = result.get("mean_survivor_difference")
        if survivor_difference is None:
            survivor_difference = float(result.get("mean_red_survivors", 0.0)) - float(result.get("mean_blue_survivors", 0.0))
        return (
            float(result.get("elimination_win_rate", 0.0)),
            float(result.get("overall_red_win_rate", 0.0)),
            float(result.get("mean_effective_damage", result.get("mean_red_effective_damage", 0.0))),
            float(survivor_difference),
            float(result.get("mean_hits", result.get("mean_red_hits", 0.0))),
            float(result.get("mean_attack_area_steps", result.get("mean_red_attack_area_steps", 0.0))),
            float(result.get("mean_team_episode_return", result.get("mean_episode_return", 0.0))),
            -float(result.get("red_crash_rate", 0.0)),
            -float(result.get("timeout_rate", 0.0)),
        )
    if selection == "smoke":
        return float(result["overall_red_win_rate"]),-float(result["red_crash_rate"])-float(result["blue_crash_rate"]),float(result["mean_episode_return"])
    raise ValueError(f"Unknown checkpoint_selection: {selection}")
