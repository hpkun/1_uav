"""Reward-only transition gate. Never used by geometry, APH or Blue scoring."""
import numpy as np

TOTAL_KEYS=('assigned_samples','unassigned_active_samples','close_samples','engagement_samples','far_samples',
    'posture_close_total','posture_engagement_total','posture_far_total','far_raw_total','far_applied_total',
    'far_gate_sum','far_approaching_samples','far_maintaining_samples','far_receding_samples',
    'far_self_approaching_samples','attack_range_entries','in_attack_range_samples','diagnostic_steps')


def add_statistics(total,step):
    for key in TOTAL_KEYS:
        if key in step:total[key]=total.get(key,0)+step[key]


def summarize_statistics(total):
    result=dict(total)
    far=total.get('far_samples',0);assigned=total.get('assigned_samples',0)
    result['far_gate_mean']=total.get('far_gate_sum',0)/far if far else None
    for name in ('approaching','maintaining','receding'):
        result[f'far_{name}_fraction']=total.get(f'far_{name}_samples',0)/far if far else None
    result['attack_range_entry_fraction']=total.get('attack_range_entries',0)/assigned if assigned else None
    result['in_attack_range_fraction']=total.get('in_attack_range_samples',0)/assigned if assigned else None
    return result


def settle_posture(raw, current_mask, assignments, pre_red, pre_blue, post_red, post_blue, pre_blue_alive, config):
    """Use physical displacement, including valid death endpoints this step.

    Missing/invalid targets contribute no posture; newly dead agents/targets
    remain eligible through their last valid transition, exactly as in paper.
    """
    result=np.zeros(len(current_mask));samples=[]
    for i in np.flatnonzero(current_mask):
        j=int(assignments[i])
        if j<0 or j>=len(pre_blue) or not pre_blue_alive[j]:continue
        los=pre_blue[j,:3]-pre_red[i,:3]
        d0=float(np.linalg.norm(los));d1=float(np.linalg.norm(post_blue[j,:3]-post_red[i,:3]))
        c_self=float((post_red[i,:3]-pre_red[i,:3])@(los/d0)/config.dt) if d0>0 else 0.
        c_rel=(d0-d1)/config.dt
        gate=float(np.clip(min(c_self,c_rel)/config.posture_gate_speed,0.,1.)) if d0>0 else 0.
        p=float(raw[i,j]);value=p
        band='close' if d1<config.tau_min else 'engagement' if d1<=config.tau_max else 'far'
        if band=='far' and config.posture_reward_mode=='approach_gated':
            value=min(p,0.)+gate*max(p,0.)
        result[i]=value
        samples.append(dict(red_id=int(i),target_id=j,d0=d0,d1=d1,c_self=c_self,c_rel=c_rel,
                            gate=gate,raw=p,applied=value,band=band))
    return result,samples


def posture_statistics(samples,current_mask,config):
    """Read-only counters. Exact sign categorization, no hidden velocity band."""
    stats=dict.fromkeys(TOTAL_KEYS,0)
    stats.update(assigned_samples=len(samples),unassigned_active_samples=int(current_mask.sum())-len(samples),diagnostic_steps=1)
    for s in samples:
        band=s['band'];stats[band+'_samples']+=1;stats['posture_'+band+'_total']+=s['applied']
        stats['in_attack_range_samples']+=int(s['d1']<=config.tau_max)
        # Same pre-action pair on both endpoints; target switches are not entries.
        stats['attack_range_entries']+=int(s['d0']>config.tau_max and s['d1']<=config.tau_max)
        if band=='far':
            stats['far_raw_total']+=s['raw'];stats['far_applied_total']+=s['applied'];stats['far_gate_sum']+=s['gate']
            direction='approaching' if s['c_rel']>0 else 'receding' if s['c_rel']<0 else 'maintaining'
            stats['far_'+direction+'_samples']+=1
            stats['far_self_approaching_samples']+=int(s['c_self']>0)
    return stats
