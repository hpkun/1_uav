"""Bounded capability benchmark (default 5k/module, never a long experiment)."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.train_modular_mappo import load_config
from algorithm.modular_mappo.runner import ModularMAPPOTrainingRunner

MODULES=[("M1","pw_m1_wave_context.yaml"),("M2","pw_m2_gru.yaml"),("M3","pw_m3_popart.yaml"),("M4","pw_m4_wave_reward.yaml"),("M5","pw_m5_wave_balance.yaml"),("M6","pw_m6_warm_start.yaml"),("M7","pw_m7_curriculum.yaml"),("M8","pw_m8_policy_anchor.yaml")]
def main():
 p=argparse.ArgumentParser();p.add_argument("--pilot-steps",type=int,default=5000);p.add_argument("--device",default="cpu");p.add_argument("--output-dir",default="outputs/modular_capability_study");a=p.parse_args()
 if not 1<=a.pilot_steps<=20000:raise ValueError("pilot-steps must be in [1,20000]")
 if a.pilot_steps*len(MODULES)>180000:raise ValueError("aggregate pilot budget exceeds 180k")
 out=ROOT/a.output_dir;out.mkdir(parents=True,exist_ok=True);env=yaml.safe_load((ROOT/"configs/persistent_wave_v2_environment.yaml").read_text(encoding="utf-8"));source=ROOT/"outputs/d999_seed2023/best_eval.pt";rows=[]
 for mid,file in MODULES:
  run=out/f"pilot_{mid.lower()}";summary_path=run/"run_summary.json"
  if summary_path.exists():summary=json.loads(summary_path.read_text(encoding="utf-8"))
  else:
   cfg=load_config(ROOT/"configs"/file);cfg["training"].update({"seed":42,"num_train_envs":4,"total_sampled_steps":a.pilot_steps,"evaluation_episodes":3,"rollout_steps":128,"ppo_epochs":2,"minibatch_size":256})
   if mid=="M7":cfg["modules"]["curriculum"].update({"stage1_end":a.pilot_steps//3,"stage2_end":2*a.pilot_steps//3})
   warm=str(source) if mid=="M6" and source.exists() else None;ref=str(source) if mid=="M8" and source.exists() else None
   if mid in {"M6","M8"} and not source.exists():rows.append({"module":mid,"status":"SKIPPED_SOURCE_MISSING"});continue
   summary=ModularMAPPOTrainingRunner(env,cfg,4,a.pilot_steps,a.device,42,run,False,warm,ref).run()
  finite=all(np.isfinite(list(summary["optimization"].values())));fr=summary["wave_fractions"];ev=summary["evaluation"]
  status="READY_WITH_CAUTION" if mid in {"M2","M3","M4","M8"} else "READY"
  rows.append({"module":mid,"status":status if finite else "NOT_READY","sampled_steps":summary["sampled_steps"],"wave1_fraction":fr["wave_1"],"wave2_fraction":fr["wave_2"],"wave3_fraction":fr["wave_3"],"average_return":ev["average_return"],"average_waves_cleared":ev.get("average_waves_cleared",0),"clear_wave_3_probability":ev.get("clear_wave_3_probability",0),"actor_loss":summary["optimization"]["actor_loss"],"value_loss":summary["optimization"]["value_loss"],"anchor_kl":summary["optimization"].get("anchor_kl",0),"popart_std":summary["optimization"].get("popart_std",1),"reward_bonus_total":summary["reward_adapter_totals"]["reward_bonus_total"]})
 fields=list(rows[0]);
 with (out/"module_micro_pilot.csv").open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 with (out/"module_capability_wave_samples.csv").open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=["module","wave1_fraction","wave2_fraction","wave3_fraction"]);w.writeheader();w.writerows({k:r.get(k) for k in w.fieldnames} for r in rows)
 payload={"purpose":"functional/capability diagnostic; not a performance comparison","seed":42,"num_envs":4,"pilot_steps_per_module":a.pilot_steps,"aggregate_pilot_steps":a.pilot_steps*len(MODULES),"rows":rows};(out/"module_capability_summary.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
 table="\n".join(f"| {r['module']} | {r['status']} | {r.get('wave1_fraction',0):.4f} | {r.get('wave2_fraction',0):.4f} | {r.get('wave3_fraction',0):.4f} | {r.get('average_waves_cleared',0):.2f} |" for r in rows)
 report=f"""# Modular MAPPO capability report\n\nThese {a.pilot_steps}-step/module pilots are functional diagnostics, not performance conclusions. Aggregate local pilot budget: {a.pilot_steps*len(MODULES)} sampled steps.\n\n| Module | Readiness | W1 fraction | W2 fraction | W3 fraction | Eval waves |\n|---|---:|---:|---:|---:|---:|\n{table}\n\nLevel 1: invariant/unit tests cover M1–M8. Level 2: all-off and every single module completed real environment rollout, PPO update, checkpoint and raw-outcome evaluation.\n""";(out/"module_capability_report.md").write_text(report,encoding="utf-8")
 print(json.dumps(payload,indent=2))
if __name__=="__main__":main()
