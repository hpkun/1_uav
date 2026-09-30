#!/usr/bin/env python3
"""Read-only exact-endpoint analyzer for the future six-run DAWE screen."""
from __future__ import annotations
import csv,hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SEEDS=(5301,5302,5303);TARGET=1_805_280
OUT=ROOT/"outputs/dawe_fixed10_300k_analysis"
RUNS={"Control":lambda s:ROOT/f"outputs/dev_dawe_control_seed{s}_300k","DAWE":lambda s:ROOT/f"outputs/dev_dawe_v1_seed{s}_300k"}
FIELDS={"AW":"average_waves_cleared","W1":"clear_wave_1_probability","W2":"clear_wave_2_probability","W3":"clear_wave_3_probability","Return":"average_return","RedLoss":"average_red_loss","Boundary":"average_red_boundary_exits","Ground":"average_red_ground_losses","EpisodeLength":"average_episode_length"}
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as stream:
  for block in iter(lambda:stream.read(1024*1024),b""):h.update(block)
 return h.hexdigest()
def rows(path):
 with path.open(newline="",encoding="utf-8") as stream:return list(csv.DictReader(stream))
def jsonl(path):return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
def metrics(row):
 value={key:float(row[field]) for key,field in FIELDS.items()};value["Q2"]=None if value["W1"]==0 else value["W2"]/value["W1"];value["Q3"]=None if value["W2"]==0 else value["W3"]/value["W2"];return value
def delta(a,b,key):return None if a[key] is None or b[key] is None else a[key]-b[key]
def main():
 if OUT.exists():raise FileExistsError(OUT)
 endpoint={};paired=[];mechanism=[];runtime_shas=set();protocol={}
 for seed in SEEDS:
  endpoint[seed]={};source=ROOT/f"outputs/diag_mappo_learnability/l3_seed{seed}/checkpoint_1505280.pt";source_sha=sha(source)
  for method,pathfn in RUNS.items():
   path=pathfn(seed)
   required=("run_config.json","run_summary.json","branch_from.json","evaluation_history.csv","optimization_metrics.jsonl","latest.pt","final.pt")
   if not path.is_dir() or any(not (path/name).is_file() for name in required):raise RuntimeError(f"incomplete run: {path}")
   run=json.loads((path/"run_config.json").read_text());branch=json.loads((path/"branch_from.json").read_text());history=rows(path/"evaluation_history.csv");opt=jsonl(path/"optimization_metrics.jsonl")
   expected="dawe_fixed10_control" if method=="Control" else "dawe_fixed10_v1"
   if (run.get("development_method"),int(run.get("seed",-1)),int(run.get("total_sampled_steps",-1)))!=(expected,seed,TARGET):raise RuntimeError(f"run identity mismatch: {path}")
   if branch.get("parent_checkpoint_sha256")!=source_sha or int(branch.get("source_training_seed",-1))!=seed:raise RuntimeError(f"parent mismatch: {path}")
   final=[row for row in history if int(row["sampled_steps"])==TARGET]
   if len(final)!=1 or (int(final[0]["evaluation_episodes"]),int(final[0]["evaluation_seed_base"]),int(final[0]["evaluation_seed_end"]))!=(50,44_000_000,44_000_049):raise RuntimeError(f"endpoint mismatch: {path}")
   if any(45_000_000<=int(row["evaluation_seed_base"])<=45_000_199 for row in history):raise RuntimeError(f"45M used: {path}")
   if len(opt)!=49 or any(int(row["ppo_epochs_executed"])!=10 for row in opt):raise RuntimeError(f"optimization row/epoch mismatch: {path}")
   actor_steps=sum(int(row["actor_optimizer_steps_this_update"]) for row in opt);critic_steps=sum(int(row["critic_optimizer_steps_this_update"]) for row in opt)
   if (actor_steps,critic_steps)!=(5860,5860):raise RuntimeError(f"optimizer step mismatch: {path}: {(actor_steps,critic_steps)}")
   manifest={item["path"]:item["sha256"] for item in run["runtime_source_manifest_files"]};runtime_shas.add(run["runtime_source_manifest_sha256"])
   endpoint[seed][method]=metrics(final[0]);protocol[f"{method}_seed{seed}"]={"status":"PASS","runtime_source_sha":run["runtime_source_manifest_sha256"],"parent_sha":source_sha,"optimization_rows":49,"actor_steps":actor_steps,"critic_steps":critic_steps}
   windows={"early":opt[:10],"late":opt[-16:],"final5":opt[-5:]}
   for window,selected in windows.items():
    for wave in (1,2,3):
     def mean(key):
      values=[float(row[key]) for row in selected if row.get(key) is not None]
      return statistics.mean(values) if values else None
     mechanism.append({"method":method,"seed":seed,"window":window,"wave":wave,
      "base_std_mean":mean(f"dawe_wave{wave}_base_std_mean"),"effective_std_mean":mean(f"dawe_wave{wave}_effective_std_mean"),
      "effective_to_base_std_ratio":mean(f"dawe_wave{wave}_effective_to_base_std_ratio"),
      "base_log_std_mean":mean(f"dawe_wave{wave}_base_log_std_mean"),"latent_deviation_abs_mean":mean(f"dawe_wave{wave}_latent_deviation_abs_mean")})
  for key in (*FIELDS,"Q2","Q3"):
   paired.append({"seed":seed,"metric":key,"Control":endpoint[seed]["Control"][key],"DAWE":endpoint[seed]["DAWE"][key],"delta":delta(endpoint[seed]["DAWE"],endpoint[seed]["Control"],key)})
 if len(runtime_shas)!=1:raise RuntimeError("six runs do not share one runtime source SHA")
 aggregate={}
 for key in (*FIELDS,"Q2","Q3"):
  values=[row["delta"] for row in paired if row["metric"]==key and row["delta"] is not None]
  aggregate[key]={"mean_delta":statistics.mean(values) if values else None,"wins":sum(value>0 for value in values),"defined_n":len(values),"per_seed":{str(row["seed"]):row["delta"] for row in paired if row["metric"]==key}}
 safety=all(aggregate["AW"]["per_seed"][str(seed)]>-.50 and aggregate["W1"]["per_seed"][str(seed)]>-.25 for seed in SEEDS) and aggregate["W1"]["mean_delta"]>=-.05
 efficacy=aggregate["AW"]["wins"]>=2 and aggregate["AW"]["mean_delta"]>0 and aggregate["W3"]["wins"]>=2 and aggregate["W3"]["mean_delta"]>0
 label="SAFETY_FAIL" if not safety else "PROMISING" if efficacy else "NOT_SUPPORTED"
 report={"status":"DAWE_FIXED10_300K_ANALYSIS_COMPLETE","replication_unit":"training_seed","n":3,"primary_endpoint":TARGET,
  "protocol":protocol,"endpoint":endpoint,"paired_DAWE_minus_Control":aggregate,"mechanism_diagnostics":mechanism,
  "gates":{"safety":safety,"efficacy":efficacy},"DAWE_SCREEN":label,"evaluation_rerun":False,"uses_45m":False,
  "runtime_source_sha256":next(iter(runtime_shas))}
 OUT.mkdir(parents=True);(OUT/"analysis.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
 fields=list(paired[0]);
 with (OUT/"paired_endpoint.csv").open("w",newline="",encoding="utf-8") as stream:w=csv.DictWriter(stream,fieldnames=fields);w.writeheader();w.writerows(paired)
 with (OUT/"mechanism_diagnostics.csv").open("w",newline="",encoding="utf-8") as stream:w=csv.DictWriter(stream,fieldnames=list(mechanism[0]));w.writeheader();w.writerows(mechanism)
 (OUT/"decision_support.txt").write_text(f"DAWE_SCREEN={label}\nSafety={safety}\nEfficacy={efficacy}\n",encoding="utf-8")
 print(json.dumps({"status":report["status"],"label":label,"output":str(OUT)},indent=2))
if __name__=="__main__":main()
