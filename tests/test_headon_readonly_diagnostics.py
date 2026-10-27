"""Small pure telemetry checks; no training, model sampling or new episodes."""
import copy
import numpy as np
import pytest
from env.aircombat import AirCombatConfig
from env.aircombat.framed_dynamics import initial_frame
from env.aircombat.geometry import relative_geometry
from tools.diagnose_headon import transition_samples,summarize_rows,mirror_check,longest_run

C=AirCombatConfig()


def row(step=1,own=25.,blue=50.,p=.7,mu=0.,n=2.,distance=3000.):
    r0=np.array([[0,0,6000,200,0,0]],float);b0=r0.copy();b0[0,0]=distance
    r1=r0.copy();b1=b0.copy();r1[0,0]+=own;b1[0,0]+=blue
    g=relative_geometry(r1,b1)
    return dict(seed=10010000,step=step,pre_red=r0.tolist(),pre_blue=b0.tolist(),post_red=r1.tolist(),post_blue=b1.tolist(),
        pre_red_alive=[True],pre_blue_alive=[True],pre_assignments=[0],red_controls=[[mu,0,n]],
        pre_red_frames=[initial_frame(r0[0]).tolist()],reward_components={'posture':[p]},
        red_distance=g.distance.tolist(),red_alpha_i_deg=np.degrees(g.alpha_i).tolist())


def test_true_reward_motion_groups_counterfactual_only_and_readonly():
    rows=[row(),row(2,-25,0,.2),row(3,25,25,-.1),row(4,25,0,1.)];before=copy.deepcopy(rows)
    a=summarize_rows(rows,C)['posture']
    assert rows==before
    assert a['far']['n']==4 and a['far']['sum']==pytest.approx(1.8)
    assert a['far_directions']['receding']['positive_n']==2
    assert a['far_directions']['receding']['positive_sum']==pytest.approx(.9)
    assert a['receding_positive_share_of_far_positive']==pytest.approx(.9/1.9)
    assert a['receding_causes']['self_receding']['n']==1
    assert a['receding_causes']['self_closing_blue_faster']['n']==1
    assert a['counterfactual_far_gated_sum']==pytest.approx(.9)
    for distance,band in ((2000.,'engagement'),(2000.001,'far'),(0.,'close')):
        _,samples=transition_samples(row(own=0,blue=0,distance=distance),C)
        assert samples[0]['band']==band and samples[0]['posture']==.7
        assert np.isfinite([samples[0][k] for k in ('c_self','c_rel','counterfactual_gated_posture')]).all()
    old=row();old.pop('reward_components');old.pop('pre_red_frames')
    assert summarize_rows([old],C)['posture'] is None


def test_bank_aware_equilibrium_not_normal_greater_than_one():
    f,_=transition_samples(row(mu=np.pi/2,n=3),C)
    assert f[0]['normal']>1 and f[0]['vertical_margin']<0 and f[0]['gamma_rhs_deg_s']<0
    f,_=transition_samples(row(mu=0,n=2),C)
    assert f[0]['vertical_margin']==1 and f[0]['normal_required_constant_gamma']==1
    assert f[0]['gamma_rhs_deg_s']==pytest.approx(np.degrees(C.gravity/200))


def test_mirror_modes_and_same_target_consecutive_runs():
    a={'observation':[1,200,0,0,.3,6000,200,0,0,.3,6000,1000],'physical_control':[0,0,1]}
    b=copy.deepcopy(a);b['observation'][4]=-.3
    assert mirror_check([a,b],'signed_alpha_v1')==[0,0,0]  # Same action is allowed.
    assert mirror_check([a,copy.deepcopy(a)],'unsigned')==[0,0,0]
    samples=[]
    for step,target,p in ((1,0,.1),(2,0,.2),(3,1,.3),(5,1,.4)):
        samples.append(dict(seed=1,red_id=0,target=target,step=step,posture=p,c_self=-1,d0=3000,d1=3001))
    best=longest_run(samples,lambda s:True)
    assert best['start']==1 and best['end']==2 and best['samples']==2 and best['posture_sum']==pytest.approx(.3)
