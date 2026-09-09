"""Analyze the completed 3x3 learnability ladder; never evaluates a policy."""
from __future__ import annotations

import csv, json, math, shutil
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments/mappo_baseline_learnability_manifest.json"
DEST = ROOT / "outputs/mappo_baseline_learnability_audit"
ENDPOINTS = (900_000, 3_000_000)
SUMMARY_STEPS = (900_000, 1_500_000, 2_000_000, 3_000_000)
BASE_FIELDS = ["sampled_steps", "average_return", "clear_wave_1_probability",
    "clear_wave_2_probability", "clear_wave_3_probability", "average_waves_cleared",
    "average_red_loss", "average_blue_loss", "average_red_ground_losses",
    "average_red_boundary_exits", "average_episode_length", "timeout_rate"]


def number(value):
    if value in (None, ""): return None
    try: return float(value)
    except (TypeError, ValueError): return value


def read_eval(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = [{k: number(v) for k, v in row.items()} for row in csv.DictReader(f)]
    rows.sort(key=lambda r: int(r["sampled_steps"]))
    return rows


def metric(row, key):
    value = row.get(key)
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None


def safe_ratio(numerator, denominator):
    if numerator is None or denominator is None or denominator == 0: return None
    return float(numerator) / float(denominator)


def completed_training_episodes(path: Path, sampled_steps: int) -> int | None:
    """Count actually recorded completed episodes through a real sampled step."""
    if not path.is_file(): return None
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        row = json.loads(line)
        if int(row["sampled_steps"]) <= sampled_steps: count += 1
    return count


def select_endpoint(rows, step):
    """Use the nearest real scheduled evaluation; never interpolate."""
    selected = min(rows, key=lambda row: (abs(int(row["sampled_steps"]) - step), int(row["sampled_steps"])))
    if abs(int(selected["sampled_steps"]) - step) > 25_000:
        raise RuntimeError(f"no evaluation sufficiently near endpoint {step}")
    return selected


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None):
    fields = fields or sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def enriched(run, row):
    out = {"condition": run["condition"], "training_seed": run["training_seed"]}
    for key in BASE_FIELDS: out[key] = metric(row, key)
    for wave in range(1, 4):
        out[f"w{wave}"] = metric(row, f"clear_wave_{wave}_probability")
        out[f"wave_{wave}_clear_time"] = metric(row, f"wave_{wave}_duration")
        out[f"red_survivors_after_wave_{wave}"] = metric(row, f"red_survivors_after_wave_{wave}")
        out[f"red_ground_losses_wave_{wave}"] = metric(row, f"red_ground_losses_after_wave_{wave}")
        out[f"red_boundary_losses_wave_{wave}"] = metric(row, f"red_boundary_losses_after_wave_{wave}")
    out["reach_w2"] = out["w1"] if run["condition"] in ("L2", "L3") else None
    out["reach_w3"] = out["w2"] if run["condition"] == "L3" else None
    out["red_survivors_entering_w2"] = out["red_survivors_after_wave_1"] if run["condition"] in ("L2", "L3") else None
    out["red_survivors_entering_w3"] = out["red_survivors_after_wave_2"] if run["condition"] == "L3" else None
    out["q2_w2_given_w1"] = safe_ratio(out["w2"], out["w1"]) if run["condition"] in ("L2", "L3") else None
    out["q3_w3_given_w2"] = safe_ratio(out["w3"], out["w2"]) if run["condition"] == "L3" else None
    return out


def mean_sd(values):
    values = [float(v) for v in values if v is not None]
    return (float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else None,
            min(values) if values else None, max(values) if values else None) if values else (None, None, None, None)


def diagnostic_labels(summary: list[dict]) -> dict:
    lookup = {(r["condition"], r["sampled_steps"]): r for r in summary}
    l1_900, l1 = lookup[("L1", 900_000)], lookup[("L1", 3_000_000)]
    l2, l3 = lookup[("L2", 3_000_000)], lookup[("L3", 3_000_000)]
    stable = (l1["w1_mean"] or 0) >= .7 and (l1["w1_min"] or 0) >= .5
    single = "SINGLE_WAVE_BASELINE_STABLE" if stable else "SINGLE_WAVE_BASELINE_UNSTABLE"
    budget = "TRAINING_BUDGET_900K_LIKELY_INSUFFICIENT" if (l1["w1_mean"] or 0) - (l1_900["w1_mean"] or 0) >= .1 else "TRAINING_BUDGET_900K_NOT_PRIMARY_LIMITATION"
    delta_l2 = None if l1["w1_mean"] is None or l2["w1_mean"] is None else l2["w1_mean"] - l1["w1_mean"]
    delta_l3 = None if l1["w1_mean"] is None or l3["w1_mean"] is None else l3["w1_mean"] - l1["w1_mean"]
    interference = stable and any(v is not None and v <= -.10 for v in (delta_l2, delta_l3))
    conditional = [l2.get("q2_w2_given_w1_mean"), l3.get("q2_w2_given_w1_mean"), l3.get("q3_w3_given_w2_mean")]
    later = stable and not interference and any(v is not None and v < .80 for v in conditional)
    candidates = []
    if interference: candidates.append("LONG_HORIZON_OPTIMIZATION_INTERFERENCE")
    if later: candidates.append("LATER_WAVE_DISTRIBUTION_DIFFICULTY")
    if not stable:
        persistent = "PERSISTENT_WAVE_DIFFICULTY_UNRESOLVED"
        reason = "SINGLE_WAVE_BASELINE_UNSTABLE"
    elif interference or later:
        persistent = "PERSISTENT_WAVE_DIFFICULTY_SUPPORTED"; reason = None
    else:
        persistent = "PERSISTENT_WAVE_DIFFICULTY_NOT_SUPPORTED"; reason = None
    return {"single_wave": single, "training_budget": budget, "persistent_wave": persistent,
            "persistent_wave_reason": reason, "diagnostic_candidates": candidates,
            "delta_w1_l2_minus_l1": delta_l2, "delta_w1_l3_minus_l1": delta_l3}


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    missing = []
    for run in manifest["runs"]:
        run_dir = ROOT / run["output_dir"]
        for name in ("run_config.json", "evaluation_history.csv", "training_metrics.jsonl", "latest.pt"):
            if not (run_dir / name).is_file(): missing.append(str((run_dir / name).relative_to(ROOT)))
    if missing: raise RuntimeError("results incomplete; no audit written: " + ", ".join(missing))
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required for checkpoint integrity audit; CPU fallback forbidden")

    histories, integrity = {}, {"complete": True, "runs": []}
    endpoints = {step: [] for step in SUMMARY_STEPS}
    curves = []
    for run in manifest["runs"]:
        run_dir = ROOT / run["output_dir"]
        config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        state = torch.load(run_dir / "latest.pt", map_location="cuda", weights_only=False)
        if int(state["sampled_steps"]) != 3_000_000: raise RuntimeError(f"incomplete checkpoint: {run_dir}")
        if (int(config["seed"]) != run["training_seed"] or config["device"] != "cuda"
                or config["environment_variant"] != "persistent_wave_v2"
                or config["enabled_modules"] != ["actor_lr_decay"]):
            raise RuntimeError(f"run protocol mismatch: {run_dir}")
        rows = read_eval(run_dir / "evaluation_history.csv")
        histories[(run["condition"], run["training_seed"])] = rows
        for row in rows: curves.append(enriched(run, row))
        for step in SUMMARY_STEPS:
            selected = enriched(run, select_endpoint(rows, step)); selected["target_endpoint"] = step
            selected["endpoint_name"] = "900k scheduled evaluation endpoint" if step == 900_000 else ("3M final/latest endpoint" if step == 3_000_000 else "scheduled learning-curve endpoint")
            selected["completed_training_episodes"] = completed_training_episodes(
                run_dir / "training_metrics.jsonl", int(selected["sampled_steps"]))
            endpoints[step].append(selected)
        integrity["runs"].append({"condition": run["condition"], "training_seed": run["training_seed"],
            "sampled_steps": int(state["sampled_steps"]), "device": config["device"],
            "evaluation_rows": len(rows), "finite_checkpoint": all(torch.isfinite(v).all().item() for group in (state["actor"], state["critic"]) for v in group.values())})

    DEST.mkdir(parents=True, exist_ok=True)
    write_csv(DEST / "latest_900k.csv", endpoints[900_000])
    write_csv(DEST / "latest_3m.csv", endpoints[3_000_000])
    write_csv(DEST / "learning_curves.csv", curves)

    summary = []
    for condition in ("L1", "L2", "L3"):
        for step in SUMMARY_STEPS:
            subset = [r for r in endpoints[step] if r["condition"] == condition]
            row = {"condition": condition, "sampled_steps": step, "n_training_seeds": 3}
            for key in ("w1", "w2", "w3", "q2_w2_given_w1", "q3_w3_given_w2",
                        "completed_training_episodes", "average_waves_cleared", "average_return",
                        "average_red_ground_losses", "average_red_boundary_exits"):
                mean, sd, low, high = mean_sd([r.get(key) for r in subset])
                row.update({f"{key}_mean": mean, f"{key}_sample_sd": sd, f"{key}_min": low, f"{key}_max": high})
            xs, ys = [], []
            for seed in manifest["training_seeds"]:
                for r in histories[(condition, seed)]:
                    xs.append(r["sampled_steps"]); ys.append(metric(r, "average_waves_cleared"))
            # Mean per-step AUC, keeping training seed as the replicate unit.
            aucs = []
            for seed in manifest["training_seeds"]:
                h = histories[(condition, seed)]; x = np.asarray([r["sampled_steps"] for r in h], float)
                y = np.asarray([metric(r, "average_waves_cleared") for r in h], float)
                aucs.append(float(np.trapz(y, x) / max(x[-1] - x[0], 1)))
            row["waves_learning_curve_auc_mean"] = float(np.mean(aucs))
            summary.append(row)
    write_csv(DEST / "condition_summary.csv", summary)

    effects = []
    for condition in ("L1", "L2", "L3"):
        for seed in manifest["training_seeds"]:
            a = next(r for r in endpoints[900_000] if r["condition"] == condition and r["training_seed"] == seed)
            b = next(r for r in endpoints[3_000_000] if r["condition"] == condition and r["training_seed"] == seed)
            row = {"condition": condition, "training_seed": seed}
            for key in ("w1", "average_waves_cleared", "average_return", "average_red_ground_losses", "average_red_boundary_exits"):
                row[f"{key}_900k"] = a[key]; row[f"{key}_3m"] = b[key]
                row[f"delta_{key}"] = None if a[key] is None or b[key] is None else b[key] - a[key]
            effects.append(row)
    write_csv(DEST / "budget_effects.csv", effects)

    progression = []
    for step in ENDPOINTS:
        for seed in manifest["training_seeds"]:
            rows = {c: next(r for r in endpoints[step] if r["condition"] == c and r["training_seed"] == seed) for c in ("L1", "L2", "L3")}
            progression.append({"sampled_steps": step, "training_seed": seed,
                "w1_l1": rows["L1"]["w1"], "w1_l2": rows["L2"]["w1"], "w1_l3": rows["L3"]["w1"],
                "delta_w1_l2_minus_l1": rows["L2"]["w1"] - rows["L1"]["w1"],
                "delta_w1_l3_minus_l1": rows["L3"]["w1"] - rows["L1"]["w1"],
                "w2_l2": rows["L2"]["w2"], "q2_l2": rows["L2"]["q2_w2_given_w1"],
                "w2_l3": rows["L3"]["w2"], "q2_l3": rows["L3"]["q2_w2_given_w1"],
                "w3_l3": rows["L3"]["w3"], "q3_l3": rows["L3"]["q3_w3_given_w2"]})
    write_csv(DEST / "wave_progression.csv", progression)
    safety = [{k: r.get(k) for k in ("condition", "training_seed", "sampled_steps", "average_red_ground_losses", "average_red_boundary_exits", "timeout_rate", "red_ground_losses_wave_1", "red_ground_losses_wave_2", "red_ground_losses_wave_3", "red_boundary_losses_wave_1", "red_boundary_losses_wave_2", "red_boundary_losses_wave_3")} for step in ENDPOINTS for r in endpoints[step]]
    write_csv(DEST / "safety_summary.csv", safety)
    (DEST / "integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")

    source = ROOT / "outputs/mappo_baseline_learnability_preflight_audit"
    for name in ("blue_policy_audit.md", "literature_design_mapping.md"):
        if (source / name).is_file(): shutil.copy2(source / name, DEST / name)
    labels = diagnostic_labels(summary)
    (DEST / "final_report.md").write_text(f"""# MAPPO baseline learnability diagnostic

Primary endpoints are the nearest real **900k scheduled evaluation endpoint** and the **3M final/latest endpoint**; no interpolation or re-evaluation is performed. Best checkpoints are not used as evidence. The replication unit is the training seed (n=3); the 50 deterministic scenarios are evaluation cases, not independent training replicates.

## Diagnostic labels

- {labels['single_wave']}
- {labels['training_budget']}
- {labels['persistent_wave']} {('reason=' + labels['persistent_wave_reason']) if labels['persistent_wave_reason'] else ''}
- Diagnostic candidates: {labels['diagnostic_candidates'] or ['NONE']}
- Mean first-wave interference: ΔW1_L2-L1={labels['delta_w1_l2_minus_l1']}, ΔW1_L3-L1={labels['delta_w1_l3_minus_l1']}

See `condition_summary.csv`, `budget_effects.csv`, and `wave_progression.csv` for the predeclared comparisons. Missing W2/W3 fields for L1 and missing W3 for L2 are intentionally blank, not zero.
""", encoding="utf-8")
    print(json.dumps({"status": "ANALYSIS_COMPLETE", "output_dir": str(DEST), "labels": labels}, indent=2))


if __name__ == "__main__": main()
