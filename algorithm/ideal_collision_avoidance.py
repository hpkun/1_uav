"""Deterministic defender-only direction post-processing for ideal guidance."""
from dataclasses import dataclass
import math

import numpy as np


MODES=("none","knn_repulsion","predictive_cpa_knn")
EPS=1e-12


@dataclass(frozen=True)
class AvoidanceConfig:
    mode: str = "none"
    k: int = 4
    influence_radius_m: float = 6.
    alpha: float = 2.5
    repulsion_lambda: float = .40
    fixed_weight: float = .40
    cpa_horizon_s: float = 3.
    safe_distance_m: float = 3.
    alert_distance_m: float = 8.
    max_avoidance_weight: float = .40

    def __post_init__(self):
        if self.mode not in MODES or type(self.k) is not int or self.k<1:
            raise ValueError("invalid avoidance mode or k")
        values=(self.influence_radius_m,self.alpha,self.repulsion_lambda,
                self.fixed_weight,self.cpa_horizon_s,self.safe_distance_m,
                self.alert_distance_m,self.max_avoidance_weight)
        if (not all(math.isfinite(v) for v in values) or
                self.influence_radius_m<=0 or self.alpha<=0 or self.repulsion_lambda<0 or
                not 0<=self.fixed_weight<=1 or self.cpa_horizon_s<=0 or
                self.safe_distance_m<=0 or self.alert_distance_m<=self.safe_distance_m or
                not 0<=self.max_avoidance_weight<=1):
            raise ValueError("invalid avoidance parameters")


def pair_cpa(position_i,velocity_i,position_j,velocity_j,config=AvoidanceConfig()):
    """Current relative-velocity CPA and its bounded future risk."""
    r=np.asarray(position_j,float)-np.asarray(position_i,float)
    v=np.asarray(velocity_j,float)-np.asarray(velocity_i,float)
    if r.shape!=(3,) or v.shape!=(3,) or not np.isfinite(r).all() or not np.isfinite(v).all():
        raise ValueError("CPA requires finite 3D states")
    distance=float(np.linalg.norm(r))
    speed_sq=float(v@v)
    closing=float(r@v)<0.
    if speed_sq<=1e-18:
        t_raw=t_cpa=0.
        # Equal-velocity neighbors already inside the alert radius remain risky.
        approaching=distance<config.alert_distance_m
    else:
        t_raw=-float(r@v)/speed_sq
        t_cpa=float(np.clip(t_raw,0.,config.cpa_horizon_s))
        approaching=closing and t_raw>=0.
    r_cpa=r+v*t_cpa
    d_cpa=float(np.linalg.norm(r_cpa))
    distance_risk=float(np.clip(1.-(d_cpa-config.safe_distance_m)/
        (config.alert_distance_m-config.safe_distance_m),0.,1.))
    time_risk=float(np.clip(1.-t_cpa/config.cpa_horizon_s,0.,1.))
    risk=distance_risk*time_risk if approaching else 0.
    return dict(current_distance_m=distance,relative_closing=closing,
        t_raw_s=t_raw,t_cpa_s=t_cpa,d_cpa_m=d_cpa,
        distance_risk=distance_risk,time_risk=time_risk,
        combined_risk=float(risk),r_cpa=r_cpa)


def _escape(raw,r,index):
    """Deterministic orthogonal direction when CPA centers coincide."""
    candidate=np.cross(raw,r)
    if np.linalg.norm(candidate)<=EPS:
        axis=np.array((0.,0.,1.)) if abs(raw[2])<.9 else np.array((1.,0.,0.))
        candidate=np.cross(raw,axis)
    if np.linalg.norm(candidate)<=EPS:
        candidate=np.eye(3)[index%3]
    return candidate/np.linalg.norm(candidate)


def apply_collision_avoidance(config,defender_index,defender_positions,
                              defender_velocities,defender_mask,raw_guidance_direction):
    """Return a unit heading and audit metadata without changing any other state."""
    positions=np.asarray(defender_positions,float)
    velocities=np.asarray(defender_velocities,float)
    mask=np.asarray(defender_mask,bool)
    raw=np.asarray(raw_guidance_direction,float)
    if (positions.ndim!=2 or positions.shape[1]!=3 or velocities.shape!=positions.shape or
            mask.shape!=(len(positions),) or raw.shape!=(3,) or
            not np.isfinite(positions).all() or not np.isfinite(velocities).all() or
            not np.isfinite(raw).all() or not 0<=defender_index<len(mask) or
            not mask[defender_index] or not np.isclose(np.linalg.norm(raw),1.,atol=1e-9)):
        raise ValueError("invalid defender avoidance state")
    neighbors=[j for j in range(len(mask)) if j!=defender_index and mask[j]]
    selected=[]
    avoid_raw=np.zeros(3)
    risk_max=0.
    if config.mode=="knn_repulsion":
        ordered=sorted(neighbors,key=lambda j:(float(np.linalg.norm(
            positions[j]-positions[defender_index])),j))[:config.k]
        for j in ordered:
            delta=positions[defender_index]-positions[j]
            distance=float(np.linalg.norm(delta))
            if distance>=config.influence_radius_m:
                continue
            weight=config.repulsion_lambda*max(0.,1.-distance/config.influence_radius_m)**config.alpha
            away=delta/distance if distance>EPS else _escape(raw,delta,defender_index)
            avoid_raw+=weight*away
            selected.append(dict(neighbor_index=j,current_distance_m=distance,
                relative_closing=None,t_cpa_s=None,d_cpa_m=None,distance_risk=None,
                time_risk=None,combined_risk=float(weight)))
        weight=config.fixed_weight
    elif config.mode=="predictive_cpa_knn":
        candidates=[]
        for j in neighbors:
            row=pair_cpa(positions[defender_index],velocities[defender_index],
                         positions[j],velocities[j],config)
            if row["combined_risk"]>0:
                candidates.append((j,row))
        candidates.sort(key=lambda item:(-item[1]["combined_risk"],item[1]["t_cpa_s"],item[0]))
        for j,row in candidates[:config.k]:
            r_cpa=row.pop("r_cpa")
            direction=-r_cpa/row["d_cpa_m"] if row["d_cpa_m"]>EPS else _escape(
                raw,positions[j]-positions[defender_index],defender_index)
            avoid_raw+=row["combined_risk"]*direction
            selected.append(dict(neighbor_index=j,**{key:value for key,value in row.items()
                if key!="t_raw_s"}))
        risk_max=max((row["combined_risk"] for row in selected),default=0.)
        weight=config.max_avoidance_weight*risk_max
    else:
        weight=0.
    active=bool(weight>0 and np.linalg.norm(avoid_raw)>EPS)
    avoid=avoid_raw/np.linalg.norm(avoid_raw) if active else None
    if active:
        mixture=(1.-weight)*raw+weight*avoid
        final=mixture/np.linalg.norm(mixture)
    else:
        final=raw.copy()
        weight=0.
    deviation=math.degrees(math.acos(float(np.clip(raw@final,-1.,1.)))) if active else 0.
    early=bool(active and config.mode=="predictive_cpa_knn" and selected and
        min(row["current_distance_m"] for row in selected)>config.alert_distance_m)
    return final,dict(avoidance_mode=config.mode,avoidance_active=active,
        raw_guidance_direction_unit=raw.tolist(),
        final_guidance_direction_unit=final.tolist(),
        avoidance_direction_unit=avoid.tolist() if avoid is not None else None,
        avoidance_weight=float(weight),neighbor_count=len(neighbors),
        risk_neighbor_count=len(selected),selected_neighbors=selected,
        avoidance_guidance_deviation_deg=deviation,
        predictive_early_activation=early,maximum_cpa_risk=float(risk_max))
