"""No-avoidance radius audit before the paired avoidance experiment."""
import argparse
import json
from pathlib import Path

import numpy as np

from algorithm.ideal_collision_avoidance import AvoidanceConfig
from algorithm.ideal_multi_predictive_guidance import run_episode
from env.ideal_interception.config import (formal_6v6_level2_config,
    formal_6v6_avoidance_stress_config,formal_15v15_level2_config)

RADII=(30.,20.,15.,12.,10.)


def choose_radius(rows):
    valid=[row for row in rows if row["valid"]]
    if not valid:
        return None,"no valid initial formation"
    if all(row["episodes_below_8m_rate"]<.2 for row in valid):
        return None,"even the smallest valid radius has insufficient alert pressure"
    bracket=[row for row in valid if .2<=row["episodes_below_8m_rate"]<=.6]
    if bracket:
        chosen=max(bracket,key=lambda row:row["radius_m"])
        return chosen["radius_m"],"largest radius with 20–60% alert-exposure episodes"
    chosen=min(valid,key=lambda row:(abs(row["episodes_below_8m_rate"]-.4),
                                     -row["radius_m"]))
    return chosen["radius_m"],"closest rate to 40%, larger radius on ties"


def audit_config(config,seeds):
    seeds=list(seeds)
    count_defenders=config.defender_count
    starts=np.asarray(config.defender_starts[:count_defenders],float)
    initial=min(float(np.linalg.norm(starts[i]-starts[j]))
                for i in range(count_defenders) for j in range(i+1,count_defenders))
    radius=config.defender_circle_radius_m or (60. if config.scale=="15v15" else 50.)
    if initial<=8.:
        return dict(radius_m=radius,initial_minimum_friendly_distance_m=initial,
            valid=False,invalid_reason="initial neighbor distance <= 8 m",
            episodes=len(seeds))
    episodes=[]
    for seed in seeds:
        summary,_=run_episode(config,"three_point_pip",seed,AvoidanceConfig(mode="none"))
        episodes.append(dict(seed=seed,scenario_id=summary["scenario_id"],
            minimum_friendly_distance_m=summary["minimum_friendly_distance_m"],
            minimum_predicted_cpa_distance_m=summary["minimum_predicted_cpa_distance_m"],
            maximum_cpa_risk=summary["maximum_cpa_risk"],
            minimum_preavoidance_predicted_cpa_distance_m=summary["minimum_preavoidance_predicted_cpa_distance_m"],
            maximum_preavoidance_cpa_risk=summary["maximum_preavoidance_cpa_risk"],
            friendly_collision_count=summary["friendly_collision_count"],
            alert_entry_count=summary["alert_entry_count"],
            safety_violation_count=summary["safety_violation_count"],
            alert_exposure_pair_seconds=summary["alert_exposure_pair_seconds"]))
    distances=[row["minimum_friendly_distance_m"] for row in episodes]
    cpa=[row["minimum_predicted_cpa_distance_m"] for row in episodes]
    count=len(episodes)
    below_alert=sum(d<8. for d in distances)
    below_safe=sum(d<3. for d in distances)
    return dict(radius_m=radius,initial_minimum_friendly_distance_m=initial,
        valid=True,episodes=count,
        friendly_collision_episodes=sum(row["friendly_collision_count"]>0 for row in episodes),
        friendly_collision_count=sum(row["friendly_collision_count"] for row in episodes),
        episodes_below_8m=below_alert,episodes_below_8m_rate=below_alert/count,
        episodes_below_3m=below_safe,episodes_below_3m_rate=below_safe/count,
        episodes_with_min_distance_below_8m=below_alert,
        episodes_with_min_distance_below_8m_rate=below_alert/count,
        episodes_with_min_distance_below_3m=below_safe,
        episodes_with_min_distance_below_3m_rate=below_safe/count,
        mean_minimum_friendly_distance_m=float(np.mean(distances)),
        p05_minimum_friendly_distance_m=float(np.percentile(distances,5)),
        minimum_friendly_distance_m=float(min(distances)),
        global_minimum_friendly_distance_m=float(min(distances)),
        alert_entry_count=sum(row["alert_entry_count"] for row in episodes),
        safety_violation_count=sum(row["safety_violation_count"] for row in episodes),
        alert_exposure_pair_seconds=sum(row["alert_exposure_pair_seconds"] for row in episodes),
        minimum_preavoidance_predicted_cpa_distance_m=float(min(cpa)),
        maximum_preavoidance_cpa_risk=float(max(row["maximum_preavoidance_cpa_risk"] for row in episodes)),
        minimum_predicted_cpa_distance_m=float(min(cpa)),
        maximum_cpa_risk=float(max(row["maximum_cpa_risk"] for row in episodes)),
        per_seed=episodes)


def audit(seeds):
    seeds=list(seeds)
    nominal=audit_config(formal_6v6_level2_config(),seeds)
    rows=[audit_config(formal_6v6_avoidance_stress_config(radius),seeds)
          for radius in RADII]
    selected,reason=choose_radius(rows)
    return dict(method="three_point_pip",avoidance_mode="none",
        intruder_motion_level="level2_maneuvering",seeds=seeds,
        nominal=nominal,candidates=rows,
        stress_profile=dict(version="avoidance_stress_v1",
            selected_radius_m=selected,selection_rule=reason,
            adequate=selected is not None))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale",choices=("6v6","15v15"),default="6v6")
    parser.add_argument("--scenario-profile",choices=("nominal","stress"),default="stress")
    parser.add_argument("--seed-start",type=int,default=17)
    parser.add_argument("--trials",type=int,default=50)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args(argv)
    if args.seed_start<0 or args.trials<1:
        parser.error("invalid seed range")
    if args.scale=="15v15":
        if args.scenario_profile!="nominal":
            parser.error("15v15 has only the nominal profile")
        seeds=list(range(args.seed_start,args.seed_start+args.trials))
        result=dict(method="three_point_pip",avoidance_mode="none",scale="15v15",
            intruder_motion_level="level2_maneuvering",seeds=seeds,
            nominal=audit_config(formal_15v15_level2_config(),seeds))
    else:
        result=audit(range(args.seed_start,args.seed_start+args.trials)) if args.scenario_profile=="stress" else dict(
            method="three_point_pip",avoidance_mode="none",scale="6v6",
            intruder_motion_level="level2_maneuvering",
            seeds=list(range(args.seed_start,args.seed_start+args.trials)),
            nominal=audit_config(formal_6v6_level2_config(),range(args.seed_start,args.seed_start+args.trials)))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    for row in result.get("candidates",[]):
        print(f"radius={row['radius_m']:.0f} valid={row['valid']} "
              f"alert={row.get('episodes_below_8m','N/A')}/{args.trials} "
              f"collision={row.get('friendly_collision_count','N/A')}")
    if "stress_profile" in result:
        print(f"selected={result['stress_profile']['selected_radius_m']} "
              f"reason={result['stress_profile']['selection_rule']}")
    else:
        row=result["nominal"]
        print(f"scale={args.scale} nominal episodes={row['episodes']} "
              f"collision={row['friendly_collision_count']} "
              f"alert={row['episodes_below_8m']}/{row['episodes']}")


if __name__=="__main__":
    main()
