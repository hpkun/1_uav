"""Piecewise-linear swept residence, with union coverage and unique time credit."""
import numpy as np


def centers_at(anchors,phase,time,config):
    # A common Y/Z translation preserves disjoint spheres and x-reflection fairness.
    angle=2*np.pi*time/config.region_period+phase
    delta=np.array([0.,config.region_y_amplitude*np.sin(angle),config.region_z_amplitude*np.sin(angle/2)])
    return anchors+delta


def residence_interval(start,end,radius):
    """Fractional time inside a sphere for a relative straight segment, including tangency."""
    a=np.asarray(start,dtype=float); d=np.asarray(end,dtype=float)-a
    aa=float(d@d); bb=float(2*(a@d)); cc=float(a@a-radius*radius)
    if aa<=1e-20:
        return (0.,1.) if cc<=0 else None
    disc=bb*bb-4*aa*cc
    if disc<=0: return None  # A tangent has zero residence measure.
    root=np.sqrt(disc)
    lo=max(0.,(-bb-root)/(2*aa)); hi=min(1.,(-bb+root)/(2*aa))
    return (lo,hi) if hi>lo else None


def score_transition(previous,current,previous_centers,centers,masks,config,step):
    n=len(current['red'])
    credits={t:np.zeros(n) for t in ('red','blue')}
    dwell={t:np.zeros((n,len(centers))) for t in credits}
    coverage={t:np.zeros(len(centers)) for t in credits}
    exclusive={t:np.zeros(len(centers)) for t in credits}
    events=[]
    for r in range(len(centers)):
        intervals={t:{} for t in credits}; cuts={0.,1.}
        for team in credits:
            for i in np.flatnonzero(masks[team]):
                interval=residence_interval(previous[team][i,:3]-previous_centers[r],current[team][i,:3]-centers[r],config.region_radius)
                if interval is not None:
                    intervals[team][int(i)]=interval
                    cuts.update(interval)
                    dwell[team][i,r]=(interval[1]-interval[0])*config.dt
        cuts=sorted(cuts)
        for lo,hi in zip(cuts[:-1],cuts[1:]):
            # Coincident crossings can differ by a few floating-point ulps.
            # Do not emit fictitious sub-resolution exclusive occupancy events.
            if hi-lo<=config.residence_fraction_epsilon: continue
            mid=(lo+hi)/2
            occupants={t:[i for i,(a,b) in intervals[t].items() if a<mid<b] for t in credits}
            duration=(hi-lo)*config.dt
            for team in credits:
                if occupants[team]: coverage[team][r]+=duration
                other='blue' if team=='red' else 'red'
                if occupants[team] and not occupants[other]:
                    exclusive[team][r]+=duration
                    credits[team][occupants[team]]+=duration/len(occupants[team])
            if occupants['red'] or occupants['blue']:
                events.append(dict(step=step,region_id=r,start_fraction=lo,end_fraction=hi,
                                   duration=duration,red_ids=occupants['red'],blue_ids=occupants['blue']))
    return dict(credits=credits,dwell=dwell,coverage=coverage,exclusive=exclusive,events=events)
