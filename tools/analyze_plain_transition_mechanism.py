"""Strictly offline analysis of the Plain transition-mechanism diagnostic."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.plain_transition_diagnostic_common import OUTPUT_DIR,POLICY_SEEDS


def scalar(value: str):
    if value=="" or value is None:return None
    if value in ("True","False"):return value=="True"
    try:
        number=float(value)
        return int(number) if number.is_integer() else number
    except ValueError:return value


def read_csv(path: Path) -> list[dict[str,Any]]:
    with path.open(newline="",encoding="utf-8") as stream:
        return [{k:scalar(v) for k,v in row.items()} for row in csv.DictReader(stream)]


def write_csv(path: Path,rows: list[dict[str,Any]]) -> None:
    if not rows:path.write_text("",encoding="utf-8");return
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),extrasaction="ignore");writer.writeheader();writer.writerows(rows)


def stats(values) -> dict[str,Any]:
    x=np.asarray([float(v) for v in values if v is not None],dtype=float)
    if not x.size:return {"N":0,"mean":None,"median":None,"q1":None,"q3":None}
    return {"N":int(x.size),"mean":float(x.mean()),"median":float(np.median(x)),"q1":float(np.quantile(x,.25)),"q3":float(np.quantile(x,.75))}


def rankdata(values: list[float]) -> np.ndarray:
    order=np.argsort(values);ranks=np.empty(len(values),float);i=0
    while i<len(values):
        j=i+1
        while j<len(values) and values[order[j]]==values[order[i]]:j+=1
        ranks[order[i:j]]=(i+j-1)/2+1;i=j
    return ranks


def spearman(values,targets):
    pairs=[(float(x),float(y)) for x,y in zip(values,targets) if x is not None and y is not None]
    if len(pairs)<3:return None
    x,y=zip(*pairs);rx,ry=rankdata(list(x)),rankdata(list(y))
    if np.std(rx)==0 or np.std(ry)==0:return None
    return float(np.corrcoef(rx,ry)[0,1])


def paired_outcome_summary(rows, left_key, right_key):
    pairs=[(int(r[left_key]),int(r[right_key])) for r in rows if r.get(left_key) is not None and r.get(right_key) is not None]
    wins=sum(a>b for a,b in pairs);losses=sum(a<b for a,b in pairs);ties=len(pairs)-wins-losses
    return {"N":len(pairs),"wins":wins,"losses":losses,"ties":ties,
            "mean_paired_outcome_delta":float(np.mean([a-b for a,b in pairs])) if pairs else None}


def effect_label(deltas):
    values=[float(x) for x in deltas if x is not None]
    if not values:return "ENTRY_NOT_SUPPORTED"
    mean=float(np.mean(values));positive=sum(x>0 for x in values)
    if all(x>0 for x in values):return "ENTRY_SUPPORTED_STRONG"
    if mean>0 and positive>=4:return "ENTRY_SUPPORTED_DIRECTIONALLY"
    if mean>0:return "ENTRY_MIXED_POSITIVE"
    return "ENTRY_NOT_SUPPORTED"


def controller_label(deltas):
    values=[float(x) for x in deltas if x is not None]
    if len(values)>=2 and all(x>0 for x in values):return "CONTROLLER_SUPPORTED_DIRECTIONALLY"
    if values and float(np.mean(values))>0:return "CONTROLLER_MIXED_POSITIVE"
    return "CONTROLLER_NOT_SUPPORTED"


def classify_mechanism(entry_label, controller_label_value):
    entry_ok=entry_label in {"ENTRY_SUPPORTED_STRONG","ENTRY_SUPPORTED_DIRECTIONALLY"}
    controller_ok=controller_label_value=="CONTROLLER_SUPPORTED_DIRECTIONALLY"
    if entry_ok and controller_ok:return "STATE_AND_CONTROLLER_BOTH"
    if entry_ok:return "STATE_QUALITY_DOMINANT"
    if controller_ok:return "CONTROLLER_QUALITY_DOMINANT"
    return "NEITHER_RESOLVED"


def case_matrix(direct):
    by={(int(r["evaluation_seed"]),int(r["policy_training_seed"])):r for r in direct};rows=[]
    for case in sorted({k[0] for k in by}):
        row={"evaluation_seed":case}
        for seed in POLICY_SEEDS:
            source=by[(case,seed)]
            for out,name in (("waves_cleared","waves"),("ground_losses","ground"),("boundary_losses","boundary"),("red_losses","red_loss"),("blue_losses","blue_loss"),("episode_return","return"),("episode_length","episode_length")):
                row[f"{name}_{seed}"]=source[out]
        rows.append(row)
    return rows


def case_counts(rows):
    result={"all_same_waves":0,"all_zero":0,"all_three":0,"5303_strictly_better_than_both":0,"5303_unique_three_wave":0,"5301_or_5302_three_but_5303_not":0}
    diffs={5301:{i:0 for i in range(-3,4)},5302:{i:0 for i in range(-3,4)}}
    disagreement={"5301_vs_5302":0,"5301_vs_5303":0,"5302_vs_5303":0}
    discriminating=[]
    for r in rows:
        a,b,c=(int(r[f"waves_{s}"]) for s in POLICY_SEEDS)
        result["all_same_waves"]+=int(a==b==c);result["all_zero"]+=int(a==b==c==0);result["all_three"]+=int(a==b==c==3)
        result["5303_strictly_better_than_both"]+=int(c>a and c>b)
        result["5303_unique_three_wave"]+=int(c==3 and a<3 and b<3)
        result["5301_or_5302_three_but_5303_not"]+=int(c<3 and (a==3 or b==3))
        disagreement["5301_vs_5302"]+=int(a!=b);disagreement["5301_vs_5303"]+=int(a!=c);disagreement["5302_vs_5303"]+=int(b!=c)
        diffs[5301][c-a]+=1;diffs[5302][c-b]+=1
        discriminating.append((max(a,b,c)-min(a,b,c),r["evaluation_seed"],a,b,c))
    result["disagreement"]=disagreement;result["5303_minus_others_distribution"]={str(k):v for s in (5301,5302) for k,v in diffs[s].items() if False}
    result["5303_minus_5301"]={str(k):v for k,v in diffs[5301].items()};result["5303_minus_5302"]={str(k):v for k,v in diffs[5302].items()}
    result["mean_paired_waves_difference"]={"5303-5301":float(np.mean([r["waves_5303"]-r["waves_5301"] for r in rows])),"5303-5302":float(np.mean([r["waves_5303"]-r["waves_5302"] for r in rows]))}
    result["most_discriminating_cases"]=[{"evaluation_seed":x[1],"waves_5301":x[2],"waves_5302":x[3],"waves_5303":x[4]} for x in sorted(discriminating,reverse=True)[:10]]
    result["5303_three_while_both_others_at_most_one"] = sum(int(r["waves_5303"]==3 and r["waves_5301"]<=1 and r["waves_5302"]<=1) for r in rows)
    return result


def survivor_summary(direct):
    rows=[]
    for seed in POLICY_SEEDS:
        rr=[r for r in direct if r["policy_training_seed"]==seed]
        for wave in (1,2,3):
            record=[r[f"wave_{wave}_red_survivors_end"] for r in rr if r[f"wave_{wave}_recorded"]]
            clear=[r[f"wave_{wave}_red_survivors_end"] for r in rr if r[f"wave_{wave}_cleared"]]
            rows.append({"policy_seed":seed,"wave":wave,"record_N":len(record),"conditional_on_record":float(np.mean(record)) if record else None,"clear_N":len(clear),"conditional_on_clear":float(np.mean(clear)) if clear else None})
    return rows


FEATURES=("red_survivor_count","red_altitude_min","red_altitude_mean","red_boundary_margin_min","red_boundary_margin_mean","red_formation_spread","red_pairwise_distance_mean","remaining_horizon","minimum_spawn_distance","minimum_red_blue_distance","red_current_time_to_ground_min")


def matched_transition(transitions):
    rows=[];summary=[]
    for nxt in (2,3):
        available={seed:{int(r["evaluation_seed"]) for r in transitions if r["source_policy_seed"]==seed and r["next_wave"]==nxt} for seed in POLICY_SEEDS}
        matched=set.intersection(*(available[s] for s in POLICY_SEEDS))
        for case in sorted(matched):
            for seed in POLICY_SEEDS:
                r=next(x for x in transitions if x["next_wave"]==nxt and x["source_policy_seed"]==seed and x["evaluation_seed"]==case)
                rows.append(r)
        for feature in FEATURES+("source_wave_ground_risk_ratio",):
            for other in (5301,5302):
                deltas=[]
                for case in matched:
                    a=next(x for x in rows if x["next_wave"]==nxt and x["evaluation_seed"]==case and x["source_policy_seed"]==5303).get(feature)
                    b=next(x for x in rows if x["next_wave"]==nxt and x["evaluation_seed"]==case and x["source_policy_seed"]==other).get(feature)
                    if a is not None and b is not None:deltas.append(float(a)-float(b))
                summary.append({"next_wave":nxt,"feature":feature,"contrast":f"5303-{other}","N":len(deltas),"mean_difference":float(np.mean(deltas)) if deltas else None,"median_difference":float(np.median(deltas)) if deltas else None,"positive_direction_fraction":float(np.mean(np.asarray(deltas)>0)) if deltas else None})
    return rows,summary


def ground_summary(risks):
    rows=[]
    for seed in POLICY_SEEDS:
        for wave in (0,1,2,3):
            rr=[r for r in risks if r["policy_training_seed"]==seed and r["wave"]==wave]
            living=sum(r["living_agent_decision_steps"] for r in rr);risk=sum(r["ground_risk_steps"] for r in rr)
            rows.append({"policy_seed":seed,"wave":"all" if wave==0 else wave,"cases":len(rr),"living_agent_decision_steps":living,"ground_risk_steps":risk,"ground_risk_ratio":risk/max(living,1),"ground_risk_activation_count":sum(r["ground_risk_activation_count"] for r in rr),"max_consecutive_ground_risk_steps":max((r["max_consecutive_ground_risk_steps"] for r in rr),default=0),"minimum_boundary_margin":min((r["minimum_boundary_margin"] for r in rr if r["minimum_boundary_margin"] is not None),default=None),"minimum_positive_time_to_boundary":min((r["minimum_positive_time_to_boundary"] for r in rr if r["minimum_positive_time_to_boundary"] is not None),default=None)})
    return rows


def death_summary(path: Path):
    events=[json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows=[]
    mapping={1:0,5:4,10:9,25:24,50:49}
    fields=("altitude","pitch","vertical_speed","action_pitch","commanded_pitch","time_to_ground")
    for seed in POLICY_SEEDS:
        ground=[e for e in events if e["policy_seed"]==seed and e["death_type"]=="ground"]
        for label,index in mapping.items():
            samples=[e["trace"][index] for e in ground if len(e["trace"])>index]
            row={"policy_seed":seed,"t_minus":label,"death_event_N":len(ground),"available_N":len(samples)}
            for field in fields:row[f"median_{field}"]=statistics.median([x[field] for x in samples if x.get(field) is not None]) if any(x.get(field) is not None for x in samples) else None
            row["guard_risk_fraction"]=float(np.mean([x["would_trigger_blue_ground_guard"] for x in samples])) if samples else None
            rows.append(row)
    return rows,events


def feature_associations(transitions):
    rows=[]
    for nxt in (2,3):
        for scope in ("pooled",*POLICY_SEEDS):
            rr=[r for r in transitions if r["next_wave"]==nxt and (scope=="pooled" or r["source_policy_seed"]==scope)]
            targets=[int(r["next_wave_clear"]) for r in rr]
            for feature in FEATURES+("source_wave_ground_risk_ratio","spawn_candidate_index","spawn_radial_angle"):
                success=[r.get(feature) for r in rr if r["next_wave_clear"]]
                failure=[r.get(feature) for r in rr if not r["next_wave_clear"]]
                ss,fs=stats(success),stats(failure)
                rows.append({"scope":scope,"next_wave":nxt,"feature":feature,"success_N":ss["N"],"success_mean":ss["mean"],"success_median":ss["median"],"success_q1":ss["q1"],"success_q3":ss["q3"],"failure_N":fs["N"],"failure_mean":fs["mean"],"failure_median":fs["median"],"failure_q1":fs["q1"],"failure_q3":fs["q3"],"spearman":spearman([r.get(feature) for r in rr],targets)})
    return rows


def survivor_probability(transitions):
    rows=[]
    for nxt in (2,3):
        for scope in ("pooled",*POLICY_SEEDS):
            rr=[r for r in transitions if r["next_wave"]==nxt and (scope=="pooled" or r["source_policy_seed"]==scope)]
            for n in (1,2,3,4):
                x=[r for r in rr if r["red_survivor_count"]==n]
                rows.append({"scope":scope,"next_wave":nxt,"survivors":n,"N":len(x),"clear_probability":float(np.mean([r["next_wave_clear"] for r in x])) if x else None})
    return rows


def continuation_matrix(rows,next_wave):
    out=[]
    for source in POLICY_SEEDS:
        for controller in POLICY_SEEDS:
            rr=[r for r in rows if r["next_wave"]==next_wave and r["source_policy_seed"]==source and r["continuation_policy_seed"]==controller]
            out.append({"state_source":source,"continuation_policy":controller,"N":len(rr),"clear_rate":float(np.mean([r["next_wave_clear"] for r in rr])) if rr else None,"mean_survivors_end":float(np.mean([r["red_survivors_end"] for r in rr])) if rr else None,"mean_ground_losses":float(np.mean([r["ground_losses"] for r in rr])) if rr else None,"mean_boundary_losses":float(np.mean([r["boundary_losses"] for r in rr])) if rr else None,"mean_continuation_steps":float(np.mean([r["continuation_steps"] for r in rr])) if rr else None})
    return out


def continuation_effects(rows,next_wave):
    sources={s:{r["evaluation_seed"] for r in rows if r["next_wave"]==next_wave and r["source_policy_seed"]==s} for s in POLICY_SEEDS}
    matched=set.intersection(*(sources[s] for s in POLICY_SEEDS));state=[]
    for controller in POLICY_SEEDS:
        for source in POLICY_SEEDS:
            rr=[r for r in rows if r["next_wave"]==next_wave and r["continuation_policy_seed"]==controller and r["source_policy_seed"]==source and r["evaluation_seed"] in matched]
            state.append({"next_wave":next_wave,"continuation_policy":controller,"state_source":source,"matched_case_N":len(rr),"clear_rate":float(np.mean([r["next_wave_clear"] for r in rr])) if rr else None})
    control=[]
    for source in POLICY_SEEDS:
        for controller in POLICY_SEEDS:
            rr=[r for r in rows if r["next_wave"]==next_wave and r["source_policy_seed"]==source and r["continuation_policy_seed"]==controller]
            control.append({"next_wave":next_wave,"state_source":source,"continuation_policy":controller,"N":len(rr),"clear_rate":float(np.mean([r["next_wave_clear"] for r in rr])) if rr else None})
    return state,control


def duration_summary(direct):
    rows=[]
    for seed in POLICY_SEEDS:
        rr=[r for r in direct if r["policy_training_seed"]==seed]
        for wave in (1,2,3):
            record=[r[f"wave_{wave}_duration"] for r in rr if r[f"wave_{wave}_recorded"]]
            clear=[r[f"wave_{wave}_duration"] for r in rr if r[f"wave_{wave}_cleared"]]
            rows.append({"policy_seed":seed,"wave":wave,"record_duration_mean":float(np.mean(record)) if record else None,"clear_duration_mean":float(np.mean(clear)) if clear else None,"record_N":len(record),"clear_N":len(clear)})
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument("--input-dir",default=str(OUTPUT_DIR));a=p.parse_args();base=Path(a.input_dir);analysis=base/"analysis";analysis.mkdir(exist_ok=True)
    direct=read_csv(base/"direct_case_results.csv");transitions=read_csv(base/"transition_states.csv");risks=read_csv(base/"ground_risk_cases.csv");continuations=read_csv(base/"continuation_results.csv")
    matrix=case_matrix(direct);write_csv(analysis/"case_policy_matrix.csv",matrix)
    survivors=survivor_summary(direct);write_csv(analysis/"transition_summary.csv",survivors)
    matched,matched_summary=matched_transition(transitions);write_csv(analysis/"matched_transition_case_results.csv",matched);write_csv(analysis/"matched_transition_summary.csv",matched_summary)
    ground=ground_summary(risks);write_csv(analysis/"ground_risk_summary.csv",ground)
    death,events=death_summary(base/"death_pretrace.jsonl");write_csv(analysis/"ground_death_summary.csv",death)
    associations=feature_associations(transitions);write_csv(analysis/"feature_success_association.csv",associations)
    surv_prob=survivor_probability(transitions);write_csv(analysis/"survivor_success_probability.csv",surv_prob)
    m2=continuation_matrix(continuations,2);m3=continuation_matrix(continuations,3);write_csv(analysis/"continuation_matrix_wave2.csv",m2);write_csv(analysis/"continuation_matrix_wave3.csv",m3)
    s2,c2=continuation_effects(continuations,2);s3,c3=continuation_effects(continuations,3);write_csv(analysis/"state_source_effect.csv",s2+s3);write_csv(analysis/"controller_effect.csv",c2+c3)
    duration=duration_summary(direct);write_csv(analysis/"wave_duration_summary.csv",duration)
    integrity=json.loads((base/"replay_integrity.json").read_text(encoding="utf-8"));metadata=json.loads((base/"run_metadata.json").read_text(encoding="utf-8"))
    simultaneous=sum(bool(e.get("simultaneous_boundary_ground",False)) for e in events)
    result={"status":"ANALYSIS_COMPLETE","integrity":integrity,"run_metadata":metadata,"case_outcomes":case_counts(matrix),"clear_and_record_survivors":survivors,"matched_transition_summary":matched_summary,"ground_risk":ground,"ground_death_precursors":death,"simultaneous_boundary_ground_count":simultaneous,"survivor_success_probability":surv_prob,"wave_duration":duration,"continuation_wave2":m2,"continuation_wave3":m3,"state_source_effect":s2+s3,"controller_effect":c2+c3,"transition_weapon_state_reset":metadata["transition_weapon_state_reset"],"effect_interpretation_note":"Native results are NATIVE_ENTRY_CONDITION_EFFECT; canonical results are CANONICAL_ENTRY_CONDITION_EFFECT; controller tables are CONTROLLER_EFFECT. No PURE_RED_STATE_CAUSAL_EFFECT is claimed.","matched_future_rng_note":"matched initial future RNG stream, not event-wise coupled randomness","statistical_unit_note":"Training seed is the algorithm replication unit; 44M cases are matched mechanism diagnostics."}
    (analysis/"analysis.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    report=["# Plain transition mechanism diagnostic closure audit","",f"- Status: `{result['status']}`",f"- Replay: `{integrity['status']}`",f"- Weapon reset: `{metadata['transition_weapon_state_reset']}`",f"- Direct episodes: {metadata['direct_episode_count']}",f"- Native transitions: {metadata['transition_count']}",f"- Native continuations: {metadata['continuation_count']}",f"- Canonical transitions: {metadata.get('canonical_transition_count',0)}",f"- Canonical continuations: {metadata.get('canonical_continuation_count',0)}",f"- Simultaneous boundary+ground deaths: {simultaneous}","", "Native continuation is labeled NATIVE_ENTRY_CONDITION_EFFECT.","Canonical continuation is labeled CANONICAL_ENTRY_CONDITION_EFFECT.","Controller comparisons are CONTROLLER_EFFECT; no PURE_RED_STATE_CAUSAL_EFFECT is claimed.","Matched randomness means a matched initial future RNG stream, not event-wise coupled randomness.","", "Detailed machine-readable results are in `analysis.json` and the accompanying CSV tables."]
    (analysis/"report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print(json.dumps({"status":result["status"],"replay":integrity["status"],"analysis_dir":str(analysis)},indent=2))


if __name__=="__main__":main()
