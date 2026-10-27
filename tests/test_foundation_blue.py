"""Independent Table I / Eq.(21)-(23) checks; no learned policy."""
import copy
from dataclasses import replace
import pickle
import numpy as np
import pytest
from env.aircombat import AirCombatEnv,AirCombatConfig
from env.aircombat.opponent import TABLE_I,BLUE_CONTROLS,candidate_scores,choose_blue_controls
from env.aircombat.framed_dynamics import advance,initial_frame

C=AirCombatConfig()
EXPECTED=np.array([[0,0,1],[0,2,1],[0,-1,0],
    [-np.pi/3,0,8],[-np.pi/3,2,8],[-np.pi/3,-1,8],
    [np.pi/3,0,8],[np.pi/3,2,8],[np.pi/3,-1,8],
    [0,0,8],[0,2,8],[0,-1,8],[np.pi,0,8],[np.pi,2,8],[np.pi,-1,8]],float)


def independent_pair_score(blue,red,config=C):
    """Scalar paper equations, independent of runtime geometry/attack helpers."""
    def velocity(s):
        return s[3]*np.array([np.cos(s[4])*np.cos(s[5]),np.cos(s[4])*np.sin(s[5]),np.sin(s[4])])
    def angles(a,b):
        los=b[:3]-a[:3];d=float(np.linalg.norm(los));v,w=velocity(a),velocity(b)
        angle=lambda u:np.arctan2(np.linalg.norm(np.cross(u,los)),np.dot(u,los))
        return d,angle(v),angle(w)
    def eta(a,b):
        d,ai,aj=angles(a,b)
        return config.attack_eta if 0<d<=config.tau_max and ai<=config.alpha_max and aj<np.pi/2 else 0.
    d,ai,_=angles(blue,red)
    cosine=np.cos(ai) if d>config.geometry_epsilon else 0.
    return (blue[3]-config.v_min)/(config.v_max-config.v_min)*cosine+eta(blue,red)-eta(red,blue)


def test_all_15_physical_controls_table_order_and_units():
    np.testing.assert_array_equal(BLUE_CONTROLS,EXPECTED)
    np.testing.assert_array_equal(TABLE_I,EXPECTED[:,[0,2,1]])


@pytest.mark.parametrize('polar',[False,True])
def test_virtual_state_actual_commit_independent_scores_and_no_mutation(polar,monkeypatch):
    import env.aircombat.opponent as module
    e=AirCombatEnv(C);e.reset(29,{'scenario':'head_on_3v3'})
    e.state.adversarial_alive[1:]=False
    if polar:
        y=e.state.adversarial_states[0].copy();y[4]=np.pi/2
        y,f,_=advance(y,[.7,0,8],initial_frame(y),C)
        e.state.adversarial_states[0]=y;e.state.adversarial_frames[0]=f
    e._assign();before=pickle.dumps(e.snapshot(),protocol=5)
    captured=[];real=module.advance
    def spy(*args,**kwargs):
        result=real(*args,**kwargs);captured.append(copy.deepcopy(result));return result
    monkeypatch.setattr(module,'advance',spy)
    y=e.state.adversarial_states[0];r=e.state.friendly_states;f=e.state.adversarial_frames[0]
    scores=candidate_scores(y,r,C,f)
    assert len(captured)==15 and pickle.dumps(e.snapshot(),protocol=5)==before
    for k,(predicted,frame,_) in enumerate(captured):
        expected=sum(independent_pair_score(predicted,target) for target in r)
        assert scores[k]==pytest.approx(expected,abs=2e-14)
        direct,df,_=advance(y,EXPECTED[k],f,C)
        np.testing.assert_array_equal(direct,predicted);np.testing.assert_array_equal(df,frame)
    captured.clear()
    _,_,_,_,info=e.step(np.tile([.5,1/3,1/8],(3,1)))
    k=info['blue_action_indices'][0]
    assert len(captured)==15 and k in np.argsort(-scores,kind='stable')[:2]
    np.testing.assert_array_equal(e.state.adversarial_states[0],captured[k][0])
    np.testing.assert_array_equal(e.state.adversarial_frames[0],captured[k][1])
    assert info['blue_action_indices'][1:].tolist()==[-1,-1] and not info['blue_controls'][1:].any()


def test_offense_defense_directions_red_sensitivity_and_dead_red_filter():
    e=AirCombatEnv(C);e.reset(71)
    e.state.adversarial_states[:]=[[0,0,6000,200,0,0],[0,500,6000,210,0,.3],[0,1000,6000,180,.1,-.2]]
    e.state.adversarial_frames=np.array([initial_frame(y) for y in e.state.adversarial_states])
    e.state.friendly_states[:]=[[1000,0,6000,200,0,0],[-1000,0,6000,200,0,0],[1000,1000,6000,200,0,0]]
    e.state.friendly_alive[2]=False
    y=e.state.adversarial_states[0];r=e.state.friendly_states[:2]
    scores=candidate_scores(y,r,C,e.state.adversarial_frames[0])
    for k,control in enumerate(EXPECTED):
        p,_,_=advance(y,control,e.state.adversarial_frames[0],C)
        assert scores[k]==pytest.approx(sum(independent_pair_score(p,target) for target in r),abs=2e-14)
    assert independent_pair_score(y,r[0])==pytest.approx(5+1/3)
    assert independent_pair_score(y,r[1])==pytest.approx(-5-1/3)
    moved=r.copy();moved[0,0]=5000
    assert not np.array_equal(scores,candidate_scores(y,moved,C,e.state.adversarial_frames[0]))
    e.state.adversarial_alive[1]=False
    state_before=pickle.dumps(e.state,protocol=5);rng=np.random.default_rng(991);expected_rng=np.random.default_rng(991)
    controls,ids=choose_blue_controls(e.state,C,rng)
    for j in (0,2):
        q=candidate_scores(e.state.adversarial_states[j],r,C,e.state.adversarial_frames[j])
        ranked=np.argsort(-q,kind='stable');expected=ranked[0 if expected_rng.random()<.8 else 1]
        assert ids[j]==expected;np.testing.assert_array_equal(controls[j],EXPECTED[expected])
    assert ids[1]==-1 and not controls[1].any()
    assert pickle.dumps(e.state,protocol=5)==state_before and rng.bit_generator.state==expected_rng.bit_generator.state
    e.state.friendly_states[2,:3]=[1e8,-1e8,1e8]
    np.testing.assert_array_equal(choose_blue_controls(e.state,C,np.random.default_rng(991))[1],ids)


def test_actual_ranked_actions_at_exact_probability_boundary():
    e=AirCombatEnv(C);e.reset(77,{'scenario':'head_on_3v3'})
    class FixedRandom:
        def __init__(self,value):self.value=value
        def random(self):return self.value
    for j in range(3):
        q=candidate_scores(e.state.adversarial_states[j],e.state.friendly_states,C,e.state.adversarial_frames[j])
        order=np.argsort(-q,kind='stable')
        assert choose_blue_controls(e.state,C,FixedRandom(.8-1e-12))[1][j]==order[0]
        assert choose_blue_controls(e.state,C,FixedRandom(.8))[1][j]==order[1]
