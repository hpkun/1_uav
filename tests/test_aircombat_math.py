"""Independent scalar answers for Xu geometry, APH and Blue virtual scoring."""
import copy
import math
import numpy as np
import pytest
from env.aircombat import AirCombatConfig
from env.aircombat.geometry import relative_geometry,attack_advantage,posture_advantage,pair_advantages
from env.aircombat.assignment import aph_assign
from env.aircombat.scenarios import generate,paper_scenario,TABLES
from env.aircombat.opponent import TABLE_I,BLUE_CONTROLS,candidate_scores,choose_blue_controls
from env.dynamics import velocity_vector,physical_actions
from env.paper_dynamics import rhs
from env.integrator import rk4

C = AirCombatConfig()


def plane(x=0,y=0,z=6000,v=200,gamma=0,psi=0):
    return np.array([x,y,z,v,gamma,psi],dtype=float)


def test_velocity_and_control_equations_in_three_dimensions():
    s = plane(gamma=np.pi/6,psi=np.pi/3)
    np.testing.assert_allclose(velocity_vector(s),[50*np.sqrt(3),150,100])
    control = physical_actions([.5,1.,.5])
    np.testing.assert_allclose(control,[0,2,4])
    d = rhs(s,control,gravity=9.81)
    assert d[3]==pytest.approx(9.81*1.5)
    assert d[4]==pytest.approx(9.81/200*(4-np.sqrt(3)/2))
    assert rhs(plane(gamma=-.2),[0,0,1],gravity=9.81)[2]<0


@pytest.mark.parametrize('distance,angle,heading,expected',[
    (2000,0,0,5),(2000.000001,0,0,0),(1000,np.pi/4,0,5),
    (1000,np.pi/4+1e-7,0,0),(1000,0,np.pi/2,0),(1000,0,np.pi,0),
    (0,0,0,0),(1000,0,np.pi/2-1e-6,5)])
def test_cone_distance_and_strict_target_direction(distance,angle,heading,expected):
    a = plane(z=0)[None,:]
    b = plane(x=distance*np.cos(angle),z=distance*np.sin(angle),psi=heading)[None,:]
    g = relative_geometry(a,b)
    assert attack_advantage(g,C)[0,0]==expected


def test_target_angle_uses_same_los_and_swapping_is_consistent():
    a,b = plane()[None,:],plane(x=1000)[None,:]
    g,ha,hreverse,_ = pair_advantages(a,b,C)
    assert g.alpha_i[0,0]==g.alpha_j[0,0]==0
    assert ha[0,0]==5 and hreverse[0,0]==0
    np.testing.assert_array_equal(hreverse,attack_advantage(relative_geometry(b,a),C))
    # Above the horizontal cone: full 3D test must reject despite heading=0.
    assert attack_advantage(relative_geometry(a,plane(x=1,z=7000)[None,:]),C)[0,0]==0


@pytest.mark.parametrize('d,v,expected',[(50,200,-.02),(100,200,2),(2000,200,2),(2001,250,1.)])
def test_posture_piecewise_hand_answers(d,v,expected):
    a,b = plane(v=v)[None,:],plane(x=d)[None,:]
    assert posture_advantage(relative_geometry(a,b),a,C)[0,0]==pytest.approx(expected)
    zero = posture_advantage(relative_geometry(a,a),a,C)
    assert zero[0,0]==-1/C.geometry_epsilon


def test_far_posture_is_additive_and_head_on_cancels():
    a,b = plane(v=250)[None,:],plane(x=3000,psi=np.pi)[None,:]
    g = relative_geometry(a,b)
    assert posture_advantage(g,a,C)[0,0]==pytest.approx(.5)
    b[0,0]=1000
    assert posture_advantage(relative_geometry(a,b),a,C)[0,0]==pytest.approx(0)


@pytest.mark.parametrize('ha,hb,red,blue,expected',[
    ([[0,0],[0,0]],[[1,9],[8,2]],[1,1],[1,1],[1,0]),
    ([[5,0],[0,0]],[[0,100],[9,1]],[1,1],[1,1],[0,1]),
    ([[5,5],[0,0]],[[1,9],[8,2]],[1,1],[1,1],[1,0]),
    ([[5,0],[5,0]],[[1,9],[8,2]],[1,1],[1,1],[0,0]),
    ([[0,0,0],[0,0,0]],[[1,9,3],[8,2,1]],[1,1],[1,1,1],[1,0]),
    ([[0],[0],[0]],[[1],[2],[3]],[1,1,1],[1],[0,0,0]),
    ([[5],[0]],[[1],[2]],[1,1],[1],[0,-1]),
    ([[0,0],[0,0]],[[1,9],[8,2]],[0,1],[1,1],[-1,0]),
    ([[0,0],[0,0]],[[1,9],[8,2]],[1,1],[1,0],[0,0]),
    ([[0,0],[0,0]],[[1,9],[8,2]],[1,1],[0,0],[-1,-1]),
    ([[0,0],[0,0]],[[1,1],[1,1]],[1,1],[1,1],[0,1]),
    ([[5,5]],[[1,1]],[1],[1,1],[0]),
])
def test_aph_exact_assignment(ha,hb,red,blue,expected):
    for _ in range(2):
        assignments,f = aph_assign(ha,hb,red,blue)
        np.testing.assert_array_equal(assignments,expected)
        assert (f.sum(1)<=1).all()
        for i,j in enumerate(expected):
            assert f[i].sum()==int(j>=0)
            if j>=0:
                assert f[i,j]==1


def test_aph_replication_uses_global_cost_and_modulo_original_columns():
    ha = np.zeros((3,2))
    hb = [[9,0],[8,0],[0,10]]
    assignment,_ = aph_assign(ha,hb,[1]*3,[1]*2)
    np.testing.assert_array_equal(assignment,[0,0,1])


@pytest.mark.parametrize('name',list(TABLES))
def test_all_paper_table_columns_and_metadata(name):
    p = paper_scenario(name)
    _,n,rows = TABLES[name]
    raw = np.array(rows)
    states = np.concatenate((p['friendly_states'],p['adversarial_states']))
    assert states.shape==(len(rows),6)
    np.testing.assert_array_equal(states[:,4],raw[:,6])
    np.testing.assert_array_equal(states[:,5],raw[:,4])
    np.testing.assert_array_equal(p['metadata']['initial_mu'],raw[:,5])
    scale = f'{n}v{len(rows)-n}'
    s = generate(AirCombatConfig(scale=scale),np.random.default_rng(1),{'scenario':name})
    np.testing.assert_array_equal(s.friendly_states,states[:n])


def test_blue_table_virtual_scores_and_no_mutation():
    assert TABLE_I.shape==(15,3)
    np.testing.assert_array_equal(BLUE_CONTROLS[1],[0,2,1])
    np.testing.assert_array_equal(BLUE_CONTROLS[2],[0,-1,0])
    np.testing.assert_array_equal(BLUE_CONTROLS[4],[-np.pi/3,2,8])
    blue = plane(x=0)
    red = np.array([plane(x=1000),plane(x=1500)])
    before = blue.copy(),red.copy()
    scores = candidate_scores(blue,red,C)
    # Steady forward candidate x=50, speed=200, both red ahead and same heading.
    assert scores[0]==pytest.approx(2*(1/3+5))
    np.testing.assert_array_equal(blue,before[0]); np.testing.assert_array_equal(red,before[1])
    # Independent scalar reconstruction of all 15 candidates.
    for k,u in enumerate(BLUE_CONTROLS):
        candidate,ok = rk4(blue,u,C,dict(speed_clips=0,nonfinite_states=0))
        assert ok
        total = 0
        for target in red:
            r = target[:3]-candidate[:3]
            d = np.linalg.norm(r)
            va,vb = velocity_vector(candidate),velocity_vector(target)
            ci,cj = np.dot(va,r)/(np.linalg.norm(va)*d),np.dot(vb,r)/(np.linalg.norm(vb)*d)
            offense = 5 if d<=2000 and np.arccos(np.clip(ci,-1,1))<=np.pi/4 and cj>0 else 0
            defense = 5 if d<=2000 and np.arccos(np.clip(-cj,-1,1))<=np.pi/4 and ci<0 else 0
            total += (candidate[3]-100)/300*ci+offense-defense
        assert scores[k]==pytest.approx(total)
    s = generate(C,np.random.default_rng(2))
    old = copy.deepcopy(s)
    rng = np.random.default_rng(1)
    controls,chosen = choose_blue_controls(s,C,rng)
    for j,index in enumerate(chosen):
        ranking = np.argsort(-candidate_scores(s.adversarial_states[j],s.friendly_states,C),kind='stable')
        assert index in ranking[:2]
        np.testing.assert_array_equal(controls[j],BLUE_CONTROLS[index])
    np.testing.assert_array_equal(s.friendly_states,old.friendly_states)
    assert not s.friendly_dwell.any()


def test_blue_best_second_probability_and_tie_order(monkeypatch):
    import env.aircombat.opponent as module
    s = generate(C,np.random.default_rng(3))
    monkeypatch.setattr(module,'candidate_scores',lambda *args:np.ones(15))
    class FixedRandom:
        def __init__(self,x): self.x=x
        def random(self): return self.x
    assert np.all(choose_blue_controls(s,C,FixedRandom(.799))[1]==0)
    assert np.all(choose_blue_controls(s,C,FixedRandom(.8))[1]==1)
