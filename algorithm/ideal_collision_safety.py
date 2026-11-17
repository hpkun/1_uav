"""Post-episode friendly-pair safety and avoidance activity measurements."""
import numpy as np

from env.ideal_interception.geometry import closest,inside_intervals
from .ideal_collision_avoidance import AvoidanceConfig,pair_cpa


def _stats(values):
    return (dict(mean=float(np.mean(values)),p95=float(np.percentile(values,95)),
                 max=float(max(values))) if values else
            dict(mean=None,p95=None,max=None))


def analyze_safety(trace,avoidance=AvoidanceConfig()):
    frames=trace["frames"]
    minimum=None
    min_cpa=None
    max_risk=0.
    alert_entries=safety_entries=0
    exposure=0.
    active_steps=eligible_steps=early_steps=0
    deviations=[]
    for before,after in zip(frames[:-1],frames[1:]):
        t0,t1=before["time"],after["time"]
        exits={event["members"][0]:event["time"] for event in after["events"]
               if event["kind"]=="member_inactive"}
        states=np.asarray(before["defenders"],float)
        executed=np.asarray(after["executed_velocities"],float)
        active=np.flatnonzero(before["defender_mask"])
        for a,i in enumerate(active):
            for j in active[a+1:]:
                duration=max(0.,min(exits.get(int(i),t1),exits.get(int(j),t1))-t0)
                p,q=states[i,:3],states[j,:3]
                v,w=executed[i],executed[j]
                nearest=closest(p,v,q,w,duration)
                minimum=nearest if minimum is None else min(minimum,nearest)
                cpa=pair_cpa(p,states[i,3:],q,states[j,3:],avoidance)
                min_cpa=cpa["d_cpa_m"] if min_cpa is None else min(min_cpa,cpa["d_cpa_m"])
                max_risk=max(max_risk,cpa["combined_risk"])
                initial=float(np.linalg.norm(q-p))
                for threshold,key in ((avoidance.alert_distance_m,"alert"),
                                      (avoidance.safe_distance_m,"safe")):
                    intervals=inside_intervals(p,v,q,w,threshold,duration)
                    if nearest>=threshold-1e-10 or not intervals:
                        continue
                    measured=sum(max(0.,right-left) for left,right in intervals)
                    if key=="alert":
                        exposure+=measured
                        if initial>=threshold-1e-10 and measured>1e-10:
                            alert_entries+=1
                    elif initial>=threshold-1e-10 and measured>1e-10:
                        safety_entries+=1
        for control in after["control"]["pairs"].values():
            eligible_steps+=1
            active_steps+=bool(control["avoidance_active"])
            early_steps+=bool(control["predictive_early_activation"])
            deviations.append(control["avoidance_guidance_deviation_deg"])
    return dict(minimum_friendly_distance_m=minimum,
        minimum_predicted_cpa_distance_m=min_cpa,maximum_cpa_risk=max_risk,
        safety_violation_count=safety_entries,alert_entry_count=alert_entries,
        alert_exposure_pair_seconds=exposure,
        avoidance_active_steps=active_steps,
        eligible_active_defender_steps=eligible_steps,
        avoidance_active_fraction=active_steps/eligible_steps if eligible_steps else None,
        predictive_early_activation_count=early_steps,
        predictive_early_activation_fraction=early_steps/active_steps if active_steps else None,
        guidance_deviation_deg=_stats(deviations))
