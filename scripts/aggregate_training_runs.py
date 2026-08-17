"""Aggregate five paper runs into Figure 8/9 data with 95% confidence intervals."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import numpy as np


T_95_DF4 = 2.7764451051977987


def read_history(run_dir: Path) -> dict[int, dict[str, float]]:
    path = run_dir / "evaluation_history.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        return {
            int(row["sampled_steps"]): {key: float(value) for key, value in row.items() if key != "sampled_steps"}
            for row in csv.DictReader(stream)
        }


def mean_ci(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(T_95_DF4 * array.std(ddof=1) / np.sqrt(len(array)))


def write_rows(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs=5, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/aggregate"))
    args = parser.parse_args()
    histories = [read_history(path) for path in args.run_dirs]
    common_steps = sorted(set.intersection(*(set(history) for history in histories)))
    if not common_steps:
        raise RuntimeError("the five runs have no common evaluation sampled_steps")
    figure8, figure9 = [], []
    for step in common_steps:
        return_mean, return_ci = mean_ci([history[step]["average_return"] for history in histories])
        win_mean, win_ci = mean_ci([history[step]["win_rate"] for history in histories])
        loss_mean, loss_ci = mean_ci([history[step]["average_red_loss"] for history in histories])
        figure8.append({
            "sampled_steps": step,
            "average_return_mean": return_mean,
            "average_return_ci95": return_ci,
            "win_rate_mean": win_mean,
            "win_rate_ci95": win_ci,
        })
        figure9.append({
            "sampled_steps": step,
            "average_red_loss_mean": loss_mean,
            "average_red_loss_ci95": loss_ci,
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "figure8_data.csv", figure8)
    write_rows(args.output_dir / "figure9_data.csv", figure9)
    print(f"wrote {len(common_steps)} common evaluation points to {args.output_dir}")


if __name__ == "__main__":
    main()
