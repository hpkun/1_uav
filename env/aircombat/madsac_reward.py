"""Li (2023) Fig.2 / Eq.(25), with explicitly disclosed Xu adaptations.

Reward-only geometry; never used by APH, weapons, observations or Blue.
The printed Eq.(6) ATA uses -V2 and reversed LOS, duplicating AA. We use
Fig.2's own horizontal velocity for ATA. All returned angles are magnitudes.
"""
import numpy as np

COMPONENTS = ('R1','R2','R3','R4')
ANGLE_ROUNDOFF_DEG = 1e-12


def geometry(attacker, target, epsilon=1e-9):
    """ATA own heading→horizontal LOS; AA target heading→same LOS; HA elevation.

    Use physical horizontal velocity directions, including cos(gamma)'s sign.
    Degenerate horizontal LOS/own velocity has finite sentinel ATA and valid=False.
    Missing target horizontal velocity only invalidates AA (not ATA/R3).
    HA is the angle to the horizontal plane,
    not an angle between the LOS and the attacker's 3D velocity.
    """
    a,b=np.asarray(attacker),np.asarray(target)
    los=b[:3]-a[:3]
    horizontal=float(np.linalg.norm(los[:2]))
    distance=float(np.linalg.norm(los))
    va=a[3]*np.cos(a[4])*np.array([np.cos(a[5]),np.sin(a[5])])
    vb=b[3]*np.cos(b[4])*np.array([np.cos(b[5]),np.sin(b[5])])
    valid=horizontal>epsilon and np.linalg.norm(va)>epsilon
    aa_valid=horizontal>epsilon and np.linalg.norm(vb)>epsilon
    def angle(v):
        return float(np.degrees(np.arctan2(abs(v[0]*los[1]-v[1]*los[0]),np.dot(v,los[:2]))))
    return dict(distance=distance,ATA=angle(va) if valid else 180.,
                AA=angle(vb) if aa_valid else 180.,
                HA=float(np.degrees(np.arctan2(abs(los[2]),horizontal))),valid=bool(valid),AA_valid=bool(aa_valid))


def within(angle, limit):
    return angle<=limit+ANGLE_ROUNDOFF_DEG


def tier(angles, config, values):
    """Inclusive nested tiers; highest matching tier, never accumulated."""
    if not angles['valid']:return 0.
    result=0.
    for limit,value in zip(config.madsac_angle_limits_deg,values):
        if within(angles['ATA'],limit) and within(angles['HA'],limit):result=value
    return result


def compute(state, current_mask, assignments, events, config, *, pre_red):
    if pre_red is None:raise ValueError('MADSAC boundary reward requires pre-action positions')
    center=np.asarray(state.metadata['madsac_battlefield_center'],dtype=float)
    if center.shape!=(2,) or not np.isfinite(center).all():
        raise ValueError('invalid MADSAC battlefield center')
    parts={k:np.zeros(len(current_mask)) for k in COMPONENTS}
    # R1 is event-based, including mutual kills and multiple owned victims.
    for event in events:
        if event['attacker_side']=='Red':
            parts['R1'][event['attacker_id']]+=config.madsac_event_reward
        else:
            parts['R1'][event['target_id']]-=config.madsac_event_reward
    red,blue=state.friendly_states,state.adversarial_states
    live_blue=np.flatnonzero(state.adversarial_alive)
    for i in np.flatnonzero(current_mask & state.friendly_alive):
        d0=np.linalg.norm(pre_red[i,:2]-center)
        d1=np.linalg.norm(red[i,:2]-center)
        if d0<=config.madsac_battlefield_radius<d1:
            parts['R2'][i]=-config.madsac_exit_penalty
        j=assignments[i]
        if 0<=j<len(blue) and state.adversarial_alive[j]:
            g=geometry(red[i],blue[j],config.geometry_epsilon)
            if (g['valid'] and g['distance']>=config.madsac_distance_threshold
                    and within(g['ATA'],config.madsac_angle_limits_deg[0])
                    and within(g['HA'],config.madsac_angle_limits_deg[0])):
                parts['R3'][i]=config.madsac_guidance_reward
            if g['AA_valid'] and g['distance']<=config.madsac_distance_threshold and within(g['AA'],config.madsac_aspect_limit_deg):
                parts['R4'][i]=tier(g,config,config.madsac_offense_rewards)
        threats=[]
        for j in live_blue:
            g=geometry(blue[j],red[i],config.geometry_epsilon)
            if g['AA_valid'] and g['distance']<=config.madsac_distance_threshold and within(g['AA'],config.madsac_aspect_limit_deg):
                threats.append(tier(g,config,config.madsac_defense_penalties))
        parts['R4'][i]-=max(threats,default=0.)
    for value in parts.values():value[~current_mask]=0.
    return sum(parts.values()),parts
