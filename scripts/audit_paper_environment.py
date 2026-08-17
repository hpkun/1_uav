"""Classification-labelled single-source-of-truth paper audit."""
from __future__ import annotations
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
env=yaml.safe_load((ROOT/"configs/paper_environment.yaml").read_text(encoding="utf-8")); alg=yaml.safe_load((ROOT/"configs/madsac.yaml").read_text(encoding="utf-8"))
a=env["aircraft"]; ea=env["reproduction_assumptions"]; ma=alg["reproduction_assumptions"]; t=alg["training"]
rows=[
    ("number of red",env["scenario"]["red_count"],"PAPER"),("number of blue",env["scenario"]["blue_count"],"PAPER"),("team size",4,"PAPER"),
    ("dt", env["simulation"]["dt"], "PAPER"),("episode horizon",env["simulation"]["max_steps"],"ASSUMPTION"),("battle diameter",env["battlefield"]["diameter"],"PAPER"),
    ("speed range",[a["v_min"],a["v_max"]],"PAPER"),("theta range",[a["theta_min"],a["theta_max"]],"PAPER"),("phi range",[a["phi_min"],a["phi_max"]],"PAPER"),("action physical ranges",env["action"],"PAPER"),
    ("observation dimension",45,"DERIVED"),("observation scalar encoding",ea["observation_normalization"],"ASSUMPTION"),("dead slot representation","zero plus alive mask in learning pipeline","ASSUMPTION"),
    ("formation center distance",ea["formation"]["center_distance"],"ASSUMPTION"),("formation spacing",ea["formation"]["formation_spacing"],"ASSUMPTION"),
    ("formal sensor noise",ea["sensor"]["enabled"],"PAPER"),("sensor coefficients",{k:ea["sensor"][k] for k in ("c1","c2","c3","b1","b2","b3")},"ASSUMPTION"),
    ("weapon max range",env["weapon"]["distance_max"],"PAPER"),("weapon angle limits",[env["weapon"]["ata_max"],env["weapon"]["ha_max"]],"PAPER"),("D_firemin",env["weapon"]["distance_min"],"ASSUMPTION"),("D_hit/c4/c5",ea["weapon"],"ASSUMPTION"),
    ("reward distance threshold",4000.0,"PAPER"),("reward aggregation",ea["reward_aggregation"],"ASSUMPTION"),("reward timing","pre-attack geometry plus post-event R1/R2","ASSUMPTION"),
    ("win criterion","all opponents dead while own team has survivors; mutual elimination draw","PAPER"),("fixed blue strategy",env["scenario"]["blue_strategy"],"PAPER"),
    ("controller limits",{k:ea["controller"][k] for k in ("nx_min","nx_max","nz_min","nz_max","yaw_rate_max","pitch_rate_max","acceleration_max")},"ASSUMPTION"),("controller gains",{k:ea["controller"][k] for k in ("k_yaw","k_pitch","k_speed")},"ASSUMPTION"),
    ("actor hidden size",alg["network"]["actor_hidden_layers"],"PAPER"),("actor activation",ma["actor_activation"],"ASSUMPTION"),("log_std bounds",[ma["log_std_min"],ma["log_std_max"]],"ASSUMPTION"),
    ("critic heads",alg["network"]["attention_heads"],"PAPER"),("critic activation",ma["critic_activation"],"ASSUMPTION"),("buffer size",t["replay_buffer_size"],"PAPER"),("batch size",t["batch_size"],"PAPER"),
    ("gamma",t["gamma"],"PAPER"),("tau",t["tau"],"PAPER"),("alpha",t["alpha"],"PAPER"),("policy delay",ma["policy_delay"],"ASSUMPTION"),("learning starts",ma["learning_starts"],"ASSUMPTION"),("update/data ratio",ma["updates_per_transition"],"ASSUMPTION"),
    ("dead-agent mask semantics","zero executed actions; masked attention/loss/entropy/bootstrap","ASSUMPTION"),("training env count",t["num_train_envs"],"PAPER"),("testing seed count",t["evaluation_episodes"],"PAPER"),
]
for name,value,classification in rows: print(f"[{classification:10}] {name}: {value}")
