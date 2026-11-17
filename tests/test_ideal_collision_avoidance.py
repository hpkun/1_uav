"""Analytic friendly-avoidance fixtures and profile isolation."""
from dataclasses import replace
import math

import numpy as np
import pytest

from algorithm.ideal_collision_avoidance import (AvoidanceConfig,MODES,
    apply_collision_avoidance,pair_cpa)
from algorithm.ideal_collision_safety import analyze_safety
from algorithm.ideal_multi_predictive_guidance import run_episode
from env.ideal_interception.config import (formal_6v6_level2_config,
    formal_6v6_avoidance_stress_config)
from env.ideal_interception.scenario import initialize
from tools.ideal_collision_risk_audit import choose_radius,audit_config
from tools.ideal_collision_avoidance_benchmark import benchmark


def apply(mode,positions,velocities,raw=(1.,0.,0.),index=0,mask=None):
    positions=np.asarray(positions,float)
    velocities=np.asarray(velocities,float)
    if mask is None:
        mask=np.ones(len(positions),bool)
    return apply_collision_avoidance(AvoidanceConfig(mode=mode),index,positions,
        velocities,np.asarray(mask,bool),np.asarray(raw,float))


def test_none_exact_and_no_active_friends():
    raw=np.array((0.,1.,0.))
    p=np.array(((0.,0.,60.),(1.,0.,60.)))
    v=np.array(((20.,0.,0.),(-20.,0.,0.)))
    final,row=apply("none",p,v,raw)
    assert np.array_equal(final,raw) and not row["avoidance_active"]
    for mode in MODES:
        final,row=apply(mode,p,v,raw,mask=[True,False])
        assert np.array_equal(final,raw) and row["neighbor_count"]==0


def test_knn_distance_threshold_direction_and_capacity():
    p=np.array(((0.,0.,60.),(6.,0.,60.)))
    v=np.zeros((2,3))
    raw=np.array((0.,1.,0.))
    final,row=apply("knn_repulsion",p,v,raw)
    assert np.array_equal(final,raw) and not row["avoidance_active"]
    p[1,0]=5.
    final,row=apply("knn_repulsion",p,v,raw)
    expected=np.array((-.4,.6,0.)); expected/=np.linalg.norm(expected)
    np.testing.assert_allclose(final,expected)
    assert row["selected_neighbors"][0]["neighbor_index"]==1
    assert row["selected_neighbors"][0]["combined_risk"]==pytest.approx(
        .4*(1.-5./6.)**2.5)
    p=np.array(((0.,0.,60.),(1.,0.,60.),(2.,0.,60.),(3.,0.,60.),
                (4.,0.,60.),(5.,0.,60.)))
    _,row=apply("knn_repulsion",p,np.zeros((6,3)),raw)
    assert [r["neighbor_index"] for r in row["selected_neighbors"]]==[1,2,3,4]
    assert row["neighbor_count"]==5 and row["risk_neighbor_count"]==4
    changed=apply("knn_repulsion",p,np.array(((0,0,0),(100,0,0),(0,100,0),
        (0,0,100),(-100,0,0),(0,-100,0))),raw)[0]
    np.testing.assert_array_equal(changed,apply("knn_repulsion",p,np.zeros((6,3)),raw)[0])


def test_head_on_and_perpendicular_crossing_analytic_cpa():
    config=AvoidanceConfig(mode="predictive_cpa_knn")
    head=pair_cpa((-10,0,60),(20,0,0),(10,0,60),(-20,0,0),config)
    assert head["t_cpa_s"]==pytest.approx(.5)
    assert head["d_cpa_m"]==pytest.approx(0.)
    assert head["combined_risk"]==pytest.approx(1.-.5/3.)
    final,row=apply("predictive_cpa_knn",[(-10,0,60),(10,0,60)],
                    [(20,0,0),(-20,0,0)])
    assert row["avoidance_active"] and np.isfinite(final).all()
    assert np.linalg.norm(final)==pytest.approx(1.)
    assert row["selected_neighbors"][0]["t_cpa_s"]==pytest.approx(.5)
    cross=pair_cpa((-10,0,60),(20,0,0),(0,10,60),(0,-20,0),config)
    assert cross["t_cpa_s"]==pytest.approx(.5)
    assert cross["d_cpa_m"]==pytest.approx(0.)
    assert apply("predictive_cpa_knn",[(-10,0,60),(0,10,60)],
        [(20,0,0),(0,-20,0)])[1]["avoidance_active"]


def test_diverging_parallel_immediate_and_far_cpa():
    config=AvoidanceConfig(mode="predictive_cpa_knn")
    away=pair_cpa((0,0,60),(-20,0,0),(5,0,60),(20,0,0),config)
    assert not away["relative_closing"] and away["combined_risk"]==0.
    final,row=apply("predictive_cpa_knn",[(0,0,60),(5,0,60)],
                    [(-20,0,0),(20,0,0)],raw=(-1,0,0))
    assert not row["avoidance_active"] and final.tolist()==[-1.,0.,0.]
    parallel_far=pair_cpa((0,0,60),(20,0,0),(10,0,60),(20,0,0),config)
    assert parallel_far["combined_risk"]==0. and math.isfinite(parallel_far["d_cpa_m"])
    parallel_near=pair_cpa((0,0,60),(20,0,0),(2,0,60),(20,0,0),config)
    assert parallel_near["t_cpa_s"]==0. and parallel_near["combined_risk"]==1.
    assert pair_cpa((0,0,60),(20,0,0),(8,0,60),(19,0,0),config)["combined_risk"]==0.


def test_three_way_risk_order_determinism_and_symmetric_cancellation():
    positions=np.array(((0.,0.,60.),(10.,0.,60.),(-10.,0.,60.)))
    velocities=np.array(((0.,0.,0.),(-20.,0.,0.),(20.,0.,0.)))
    first=apply("predictive_cpa_knn",positions,velocities,raw=(0,1,0))
    second=apply("predictive_cpa_knn",positions,velocities,raw=(0,1,0))
    assert first[1]==second[1] and np.array_equal(first[0],second[0])
    assert [row["neighbor_index"] for row in first[1]["selected_neighbors"]]==[1,2]
    assert np.isfinite(first[0]).all() and np.linalg.norm(first[0])==pytest.approx(1.)
    ordered=apply("predictive_cpa_knn",[(0,0,60),(5,0,60),(10,0,60)],
        [(0,0,0),(-20,0,0),(-20,0,0)])
    assert [row["neighbor_index"] for row in ordered[1]["selected_neighbors"]]==[1,2]
    symmetric=apply("knn_repulsion",[(0,0,60),(1,0,60),(-1,0,60)],
                    [(0,0,0)]*3,raw=(0,1,0))
    assert not symmetric[1]["avoidance_active"]
    np.testing.assert_array_equal(symmetric[0],[0,1,0])


def test_safety_counts_step_internal_entry_once():
    initial=dict(time=0.,defenders=[[-5,0,60,20,0,0],[5,0,60,-20,0,0]],
        defender_mask=[True,True])
    final=dict(time=.5,executed_velocities=[[20,0,0],[-20,0,0]],events=[],
        control=dict(pairs={}))
    report=analyze_safety(dict(frames=[initial,final]))
    assert report["minimum_friendly_distance_m"]==pytest.approx(0.)
    assert report["alert_entry_count"]==1
    assert report["safety_violation_count"]==1
    assert report["alert_exposure_pair_seconds"]==pytest.approx(.4)


def test_level2_identity_ignores_unused_level1_jitter_and_stress_identity():
    base=formal_6v6_level2_config()
    modified=replace(base,intruder_bezier_jitter_ratio=.2)
    a=initialize(base,17)
    b=initialize(modified,17)
    assert a[-1]==b[-1]
    assert a[3]==b[3]
    assert all(np.array_equal(x.points,y.points) for x,y in zip(a[2],b[2]))
    stress=formal_6v6_avoidance_stress_config(15.)
    assert initialize(stress,17)[-1]!=a[-1]
    assert stress.scenario_profile_version=="avoidance_stress_v1"
    assert stress.defender_circle_radius_m==15.


def test_radius_selection_rule_and_audit_determinism():
    rows=[dict(radius_m=radius,valid=True,episodes_below_8m_rate=rate)
          for radius,rate in ((30.,.1),(20.,.25),(15.,.45),(12.,.7),(10.,.9))]
    assert choose_radius(rows)[0]==20.
    rows[1]["episodes_below_8m_rate"]=.1
    rows[2]["episodes_below_8m_rate"]=.65
    assert choose_radius(rows)[0]==15.
    rows[0]["episodes_below_8m_rate"]=.1
    rows[1]["episodes_below_8m_rate"]=.1
    rows[2]["episodes_below_8m_rate"]=.1
    rows[3]["episodes_below_8m_rate"]=.1
    rows[4]["episodes_below_8m_rate"]=.1
    assert choose_radius(rows)[0] is None
    rows[0]["valid"]=False
    assert choose_radius(rows)[0] is None
    c=formal_6v6_avoidance_stress_config(8.)
    assert not audit_config(c,[17])["valid"]
    assert audit_config(formal_6v6_avoidance_stress_config(15.),[17])==audit_config(
        formal_6v6_avoidance_stress_config(15.),[17])


def test_none_seed17_physical_regression():
    summary,trace=run_episode(formal_6v6_level2_config(),"three_point_pip",17,
        AvoidanceConfig(mode="none"))
    assert summary["path_hashes"][0]=="7f937dd5c2fa552ae50efbd08c1836034125490d261cf3885803fc4c7e1db643"
    assert summary["original_assignment"]=={0:0,4:1,2:2,5:3,3:4,1:5}
    assert trace["frames"][1]["control"]["action"][0]==[
        .9999574035175591,.49356619731032175,.5010958334411767]
    contacts=[e["time"] for e in trace["result"]["events"]
              if e["kind"]=="interception_contact"]
    assert contacts==[6.691873068177305,8.257818078965725,8.27037009560584,
                      9.063721331596392,9.15351925770379,10.06967153319023]
    assert summary["episode_resolution_time_s"]==10.06967153319023
    assert all(not pair["avoidance_active"] for frame in trace["frames"][1:]
               for pair in frame["control"]["pairs"].values())


def test_stress_cpa_end_to_end_speed_and_paired_scene():
    config=formal_6v6_avoidance_stress_config(15.)
    summary,trace=run_episode(config,"three_point_pip",40,
        AvoidanceConfig(mode="predictive_cpa_knn"))
    assert summary["avoidance_active_steps"]>0
    assert summary["predictive_early_activation_count"]>0
    for before,after in zip(trace["frames"][:-1],trace["frames"][1:]):
        executed=np.asarray(after["executed_velocities"])
        for i,active in enumerate(before["defender_mask"]):
            if active:
                assert np.linalg.norm(executed[i])==pytest.approx(config.defender_speeds[i],abs=1e-10)
        for i,pair in after["control"]["pairs"].items():
            assert np.linalg.norm(pair["raw_guidance_direction_unit"])==pytest.approx(1.)
            assert np.linalg.norm(pair["avoidance_adjusted_direction_unit"])==pytest.approx(1.)
            assert all(0<=neighbor["neighbor_index"]<6 and neighbor["neighbor_index"]!=int(i)
                and before["defender_mask"][neighbor["neighbor_index"]]
                for neighbor in pair["selected_neighbors"])
    paired=benchmark(config,[40])
    assert paired["paired_scenarios"] and len(paired["trials"][0]["methods"])==3
    assert paired["aggregates"]["predictive_cpa_knn"]["avoidance_active_steps"]>0
