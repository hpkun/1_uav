"""Independent A/B kinematics, causal forecasts and substep interception tests."""
import math
import json

import numpy as np
import pytest

from env.ideal_interception.trajectory import Reference
from algorithm.common.serialization import serializable
from kinematic_interception.controllers import Controller
from kinematic_interception.dynamics import (Command, DynamicsConfig, PointState,
    WingState, propagate, step, velocity)
from kinematic_interception.environment import (KinematicInterceptionEnv, Scene,
    one_vs_one_scene, twenty_scene)
from kinematic_interception.evaluation import (offline_forecast_experiment,
    paired_experiment, run_episode)
from kinematic_interception.prediction import (Forecast, TargetHistory,
    feasible_intercept, forecast, minimum_segment_distance)
from kinematic_interception.visualization import plot_trace
from kinematic_interception.replay import _validated_trace


CONFIG = DynamicsConfig()


def test_point_mass_exact_constant_acceleration_and_reverse():
    state = PointState(np.zeros(3), np.array((20., 0., 0.)))
    command = Command((-1., 0., 0.), 20.)
    result, acceleration, diagnostic = step(state,command,CONFIG)
    assert np.allclose(acceleration,(-10.,0.,0.))
    assert np.allclose(result.velocity,(19.5,0.,0.))
    assert np.allclose(result.position,(.9875,0.,0.))
    assert diagnostic["acceleration_norm"] <= 10.
    for _ in range(90):
        result, acceleration, _ = step(result,command,CONFIG)
        assert np.linalg.norm(acceleration) <= 10.+1e-12
        assert np.linalg.norm(result.velocity) <= 20.+1e-12
    assert result.velocity[0] < 0.


def test_point_mass_rotation_covariance_and_prediction_execution_identity():
    state = PointState(np.array((1.,2.,3.)),np.array((14.,5.,-2.)))
    command = Command((.1,.8,.6),17.)
    angle = .7
    rotation = np.array(((math.cos(angle),-math.sin(angle),0.),
                         (math.sin(angle),math.cos(angle),0.),(0.,0.,1.)))
    lhs = step(PointState(rotation@state.position,rotation@state.velocity),
               Command(tuple(rotation@np.asarray(command.direction)),17.),CONFIG)[0]
    rhs = step(state,command,CONFIG)[0]
    assert np.allclose(lhs.position,rotation@rhs.position)
    assert np.allclose(lhs.velocity,rotation@rhs.velocity)
    rows = propagate(state,command,CONFIG,.05,.01)
    current = state.copy()
    for _ in range(5):
        current = step(current,command,CONFIG,.01)[0]
    assert np.array_equal(rows[-1][1].position,current.position)
    assert np.array_equal(rows[-1][1].velocity,current.velocity)


def test_fixed_wing_turn_climb_speed_and_joint_lateral_limits():
    state = WingState(np.zeros(3),20.,0.,0.)
    command = Command((0.,1.,1.),0.)
    for _ in range(100):
        next_state,_,row = step(state,command,CONFIG)
        assert 12. <= next_state.speed <= 20.
        assert abs(next_state.yaw-state.yaw) <= .5*CONFIG.dt+1e-12
        assert abs(next_state.flightpath-state.flightpath) <= .3*CONFIG.dt+1e-12
        assert abs(next_state.flightpath) <= CONFIG.flightpath_max+1e-12
        assert row["lateral_acceleration"] <= 10.+1e-12
        assert -4.-1e-12 <= row["tangential_acceleration"] <= 3.+1e-12
        state = next_state
    assert math.isclose(state.speed,12.)
    assert abs(state.flightpath) <= CONFIG.flightpath_max
    assert state.yaw > 0.
    assert np.isclose(np.linalg.norm(velocity(state)),state.speed)


def test_fixed_wing_does_not_reverse_in_one_control_step():
    state = WingState(np.zeros(3),20.,0.,0.)
    result,_,_ = step(state,Command((-1.,0.,0.),20.),CONFIG)
    assert abs(result.yaw) <= .5*CONFIG.dt+1e-12
    assert result.position[0] > 0.


def test_fixed_wing_midpoint_position_for_bounded_turn():
    state = WingState(np.zeros(3),20.,0.,0.)
    result,_,row = step(state,Command((0.,1.,0.),20.),CONFIG)
    assert np.isclose(row["yaw_rate"],.5)
    assert np.allclose(result.position,
                       (math.cos(.0125),math.sin(.0125),0.),atol=1e-12)


def _history(function, count=7, dt=.05):
    history = TargetHistory()
    for k in range(count):
        t = k*dt
        history.observe(t,np.array([function(t)]),np.array([True]))
    return history


def test_cv_and_ca_exact_on_matching_polynomials():
    linear = lambda t: np.array((2.+3*t,-4.+2*t,10.))
    h = _history(linear)
    for method in ("CV","CA"):
        prediction = forecast(h,0,method)
        assert np.allclose(prediction.at(1.),linear(.3+1.),atol=1e-10)
    quadratic = lambda t: np.array((2.+3*t+.5*t*t,-4.+2*t,10.))
    h = _history(quadratic)
    prediction = forecast(h,0,"CA")
    assert np.allclose(prediction.at(1.),quadratic(1.3),atol=1e-10)
    join = prediction.at(2.)
    assert np.allclose(prediction.at(2.+1e-8),join+
                       (prediction.velocity+prediction.acceleration*2.)*1e-8)


def test_history_causality_mask_and_speed_cap():
    history = TargetHistory(slots=2)
    for k in range(7):
        history.observe(k*.05,np.array(((100.*k,0.,0.),(0.,0.,0.))),
                        np.array((True,False)))
    prediction = forecast(history,0,"CA")
    assert np.linalg.norm(prediction.velocity) <= 20.+1e-12
    assert np.linalg.norm(prediction.velocity+prediction.acceleration*2.) <= 20.+1e-12
    assert np.array_equal(prediction.at(0.),np.array((600.,0.,0.)))
    with pytest.raises(ValueError):
        forecast(history,1,"CA")
    history.observe(.35,np.zeros((2,3)),np.array((False,False)))
    with pytest.raises(ValueError):
        history.observe(.4,np.zeros((2,3)),np.array((True,False)))


def test_kfip_uses_actual_dynamics_and_cannot_claim_global_unreachable():
    h = _history(lambda t: np.array((30.-5*t,20.,30.)))
    target = forecast(h,0,"CV")
    a = PointState(np.array((0.,0.,30.)),np.array((20.,0.,0.)))
    b = WingState(np.array((0.,0.,30.)),20.,0.,0.)
    plan_a = feasible_intercept(a,target,CONFIG)
    plan_b = feasible_intercept(b,target,CONFIG)
    assert plan_a.candidates_evaluated <= 5
    assert plan_b.candidates_evaluated <= 5
    assert plan_a.minimum_distance_m != plan_b.minimum_distance_m
    assert plan_a.status in ("CANDIDATE_CONTACT","NO_CONTACT_PREDICTED","FAR_TRACK")
    assert plan_b.status in ("CANDIDATE_CONTACT","NO_CONTACT_PREDICTED","FAR_TRACK")


def test_kfip_detects_contact_between_prediction_samples():
    target = Forecast("CV",np.array((.2,0.,0.)),np.array((-20.,0.,0.)),
                      np.zeros(3),0.,20.,3)
    state = PointState(np.zeros(3),np.array((20.,0.,0.)))
    plan = feasible_intercept(state,target,CONFIG,horizon=.1,interval=.1)
    assert plan.candidates_evaluated == 1
    assert plan.status == "CANDIDATE_CONTACT"
    assert plan.minimum_distance_m < 1e-12
    assert abs(plan.closest_time_s-.005) < 1e-4
    # The two 10 ms segment endpoints remain outside the 0.1 m contact radius.
    rows = propagate(state,plan.command,CONFIG,.01,.01)
    assert all(np.linalg.norm(member.position-target.at(t)) > .1
               for t,member in rows)


def test_segment_scoring_changes_candidate_order_without_smaller_substeps():
    target = Forecast("CV",np.zeros(3),np.zeros(3),np.zeros(3),0.,20.,3)
    crossing = [(0.,PointState(np.array((-.2,0.,0.)),np.zeros(3))),
                (.01,PointState(np.array((.2,0.,0.)),np.zeros(3)))]
    near_static = [(0.,PointState(np.array((.05,0.,0.)),np.zeros(3))),
                   (.01,PointState(np.array((.05,0.,0.)),np.zeros(3)))]
    crossing_distance,crossing_time = minimum_segment_distance(crossing,target)
    static_distance,_ = minimum_segment_distance(near_static,target)
    assert min(np.linalg.norm(member.position) for _,member in crossing) > static_distance
    assert crossing_distance < static_distance
    assert crossing_distance == 0.
    assert crossing_time == pytest.approx(.005)


def _crossing_scene():
    path = Reference((1.2,0.,10.),(-10.,0.,10.),20.,"straight")
    return Scene(np.array(((0.,0.,10.),)),(path,),(-30.,-30.,0.),
                 (30.,30.,30.),(-20.,20.,0.),1.,30.,max_steps=20)


def test_scene_rejects_initial_overlap_and_protected_region_start():
    path = Reference((.05,0.,10.),(-10.,0.,10.),20.,"straight")
    with pytest.raises(ValueError,match="initial geometric contact"):
        Scene(np.array(((0.,0.,10.),)),(path,),(-20.,-20.,0.),
              (20.,20.,20.),(-10.,10.,0.),1.,20.)
    path = Reference((5.,0.,10.),(-10.,0.,10.),20.,"straight")
    with pytest.raises(ValueError,match="protected region"):
        Scene(np.array(((0.,0.,10.),)),(path,),(-20.,-20.,0.),
              (20.,20.,20.),(5.,0.,0.),1.,20.)


@pytest.mark.parametrize("model",("A","B"))
def test_step_internal_contact_even_when_endpoints_are_outside(model):
    scene = _crossing_scene()
    assert np.linalg.norm(scene.defender_starts[0]-scene.paths[0].sample(0.)[0]) > .1
    for substep in (.01,.005):
        env = KinematicInterceptionEnv(scene,model,substep=substep)
        result = env.step([Command((1.,0.,0.),20.)])
        assert result["outcome"] == "all_intercepted"
        assert result["intercepted_count"] == 1
        assert abs(env.events[0]["time"]-.0275) < 1e-8
        assert np.allclose(env.events[0]["positions"][0],(.55,0.,10.),atol=1e-8)


def test_turning_path_contact_converges_under_finer_substeps():
    path = Reference((.8,.04,10.),(-1.,.04,10.),20.,"straight")
    scene = Scene(np.array(((0.,0.,10.),)),(path,),(-5.,-5.,0.),
                  (5.,5.,20.),(-4.,4.,0.),1.,20.)
    times = []
    for substep in (.01,.005,.0025):
        env = KinematicInterceptionEnv(scene,"B",substep=substep)
        env.step([Command((0.,1.,0.),20.)])
        assert env.events[0]["kind"] == "interception_contact"
        times.append(env.events[0]["time"])
    assert abs(times[1]-times[2]) < abs(times[0]-times[2])
    assert abs(times[0]-times[2]) < 2e-6


def test_terminal_event_does_not_advance_survivor_to_substep_end():
    path = Reference((1.2,0.,10.),(-10.,0.,10.),20.,"straight")
    scene = Scene(np.array(((0.,0.,10.),(-5.,0.,10.))),(path,),
                  (-30.,-30.,0.),(30.,30.,30.),(-20.,20.,0.),1.,30.)
    env = KinematicInterceptionEnv(scene,"A")
    row = env.step([Command((1.,0.,0.),20.)]*2)
    assert row["outcome"] == "all_intercepted"
    assert np.isclose(row["time"],.0275)
    assert row["defender_mask"][1]
    assert np.isclose(row["defenders"][1,0],-5.+20*row["time"])


def test_paired_1v1_shares_paths_and_is_deterministic():
    one = paired_experiment((17,),methods=("G0","G1"))
    two = paired_experiment((17,),methods=("G0","G1"))
    for left,right in zip(one,two):
        for key in ("outcome","assigned_isr","breach_ratio","episode_resolution_time_s"):
            assert left[key] == right[key]
    assert len({tuple(row["path_hashes"]) for row in one}) == 1
    assert len({row["model"] for row in one}) == 2


def test_defender_exit_does_not_end_active_target_episode():
    path = Reference((-5.,0.,10.),(1.,0.,10.),20.,"straight")
    scene = Scene(np.array(((0.,0.,10.),)),(path,),(-6.,-20.,0.),
                  (2.,20.,30.),(-5.,15.,0.),1.,30.,max_steps=20)
    env = KinematicInterceptionEnv(scene,"A")
    while env.time < .1-1e-10:
        row = env.step([Command((1.,0.,0.),20.)])
    assert not row["defender_mask"].any()
    assert row["intruder_mask"].any()
    while env.outcome == "ongoing":
        env.step([None])
    assert env.outcome == "path_complete"
    assert [event["kind"] for event in env.events].count("boundary_exit") == 1


def test_trace_can_render_independent_3d_path(tmp_path):
    _,trace = run_episode(one_vs_one_scene(17),"A","G1",trace=True)
    destination = tmp_path/"episode.png"
    plot_trace(trace,destination)
    assert destination.stat().st_size > 1000
    payload = json.loads(json.dumps(serializable(dict(trace=trace))))
    _,times,defenders,intruders = _validated_trace(payload)
    assert len(times) == len(trace["frames"])
    assert defenders.shape[1:] == intruders.shape[1:] == (1,6)
    payload["trace"]["references"][0]["points"][0][0] += .01
    with pytest.raises(ValueError,match="hash mismatch"):
        _validated_trace(payload)


def test_offline_prediction_reports_per_path_units():
    result = offline_forecast_experiment((one_vs_one_scene(17,False),),
                                         horizons=(.5,))
    assert {row["method"] for row in result["summary"]} == {"CV","CA"}
    for row in result["summary"]:
        assert row["paths"] == 1
        assert row["mean_path_ade_m"] < 1e-8


def test_multi_scene_uses_existing_path_identity_without_mutating_old_env():
    scene = twenty_scene(17,3)
    assert len(scene.defender_starts) == len(scene.paths) == 3
    a = KinematicInterceptionEnv(scene,"A")
    b = KinematicInterceptionEnv(scene,"B")
    assert np.array_equal(a.intruder_positions(),b.intruder_positions())
    assert a.snapshot()["protocol"] != "uav-ideal-interception-20v20-delayed-convergence-nearest-v3"
