"""CSV metrics and evaluation ranking helpers."""

from __future__ import annotations
import csv
from pathlib import Path
from typing import Any

def append_csv(path: Path, row: dict[str,Any]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); exists=path.exists()
    with path.open("a",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(row));
        if not exists: writer.writeheader()
        writer.writerow(row)

def evaluation_key(result: dict[str,Any]) -> tuple[float,float,float]:
    return float(result["red_win_rate"]),-float(result["red_crash_rate"])-float(result["blue_crash_rate"]),float(result["mean_episode_return"])
