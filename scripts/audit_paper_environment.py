"""Print a classification-labelled paper environment audit."""
from __future__ import annotations
from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parents[1]
env = yaml.safe_load((ROOT / "configs/paper_environment.yaml").read_text(encoding="utf-8"))
alg = yaml.safe_load((ROOT / "configs/madsac.yaml").read_text(encoding="utf-8"))
a = env["aircraft"]; t = alg["training"]
rows = [
    ("number of red", env["scenario"]["red_count"], "PAPER"), ("number of blue", env["scenario"]["blue_count"], "PAPER"),
    ("team size", 4, "PAPER"), ("dt", env["simulation"]["dt"], "ASSUMPTION"), ("battle diameter", env["battlefield"]["diameter"], "PAPER"),
    ("speed range", [a["v_min"], a["v_max"]], "PAPER"), ("theta range", [a["theta_min"], a["theta_max"]], "PAPER"), ("phi range", [a["phi_min"], a["phi_max"]], "PAPER"),
    ("action physical ranges", env["action"], "PAPER"), ("observation dimension", 45, "DERIVED"),
    ("weapon max range", env["weapon"]["distance_max"], "PAPER"), ("weapon angle limits", [env["weapon"]["ata_max"], env["weapon"]["ha_max"]], "PAPER"),
    ("sensor noise enabled", env["reproduction_assumptions"]["sensor"]["enabled"], "ASSUMPTION"), ("reward distance threshold", 4000.0, "PAPER"),
    ("fixed blue strategy", env["scenario"]["blue_strategy"], "PAPER"), ("actor hidden size", alg["network"]["actor_hidden_layers"], "PAPER"),
    ("critic heads", alg["network"]["attention_heads"], "PAPER"), ("buffer size", t["replay_buffer_size"], "PAPER"), ("batch size", t["batch_size"], "PAPER"),
    ("gamma", t["gamma"], "PAPER"), ("tau", t["tau"], "PAPER"), ("alpha", t["alpha"], "PAPER"),
]
for name, value, classification in rows: print(f"[{classification:10}] {name}: {value}")
