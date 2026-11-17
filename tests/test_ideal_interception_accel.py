"""Acceleration-limited clone contracts; the original ideal environment stays frozen."""
from dataclasses import replace

import numpy as np
import pytest

from env.ideal_interception import InterceptionEnv as IdealEnv
from env.ideal_interception.config import InterceptionConfig as IdealConfig
from env.ideal_interception_accel import InterceptionConfig, InterceptionEnv
from env.ideal_interception_accel.config import DEFENDER_SLOTS, PROTOCOL
from env.ideal_interception_accel.events import limited_velocity
from env.ideal_interception_accel.config import formal_6v6_level2_config
from tools.ideal_interception_accel_trace import record, TRACE_VERSION
from algorithm.ideal_multi_predictive_guidance_accel import run_episode


def reverse_actions():
    actions = np.full((DEFENDER_SLOTS, 3), .5)
    actions[0] = (0., .5, .5)
    return actions


def test_accel_configuration_and_original_scene_are_independent():
    assert PROTOCOL != "uav-ideal-interception-v3.5.0"
    config = InterceptionConfig(scale="1v1", intruder_mode="straight")
    assert config.defender_acceleration_max == 8.
    assert replace(config, defender_acceleration_max=12.).defender_acceleration_max == 12.
    for value in (0., -1., float("inf"), float("nan")):
        with pytest.raises(ValueError, match="defender_acceleration_max"):
            replace(config, defender_acceleration_max=value)
    original = IdealEnv(IdealConfig(scale="1v1", intruder_mode="straight"))
    cloned = InterceptionEnv(config)
    _, old = original.reset(17)
    _, new = cloned.reset(17)
    np.testing.assert_array_equal(old["defenders"], new["defenders"])
    np.testing.assert_array_equal(old["intruders"], new["intruders"])
    assert [row["polyline_sha256"] for row in old["intruder_paths"]] == [
        row["polyline_sha256"] for row in new["intruder_paths"]]
    _, _, _, _, old = original.step(reverse_actions())
    _, _, _, _, new = cloned.step(reverse_actions())
    np.testing.assert_allclose(old["defenders"][0, 3:], (-20., 0., 0.))
    np.testing.assert_allclose(new["defenders"][0, 3:], (19.6, 0., 0.))


def test_velocity_limit_zero_turn_small_turn_and_reverse():
    old = np.array((20., 0., 0.))
    new, step, row, commanded = limited_velocity(old, (1., 0., 0.), 20., 8., .05)
    np.testing.assert_array_equal(new, old)
    np.testing.assert_array_equal(step, old)
    assert not row["acceleration_limited"] and commanded == 0.
    direction = np.array((np.cos(.1), np.sin(.1), 0.))
    new, _, row, _ = limited_velocity(old, direction, 20., 8., .05)
    assert row["acceleration_limited"]
    assert np.linalg.norm(new-old) == pytest.approx(.4)
    assert np.linalg.norm(new) <= 20.+1e-9
    new, step, row, commanded = limited_velocity(old, (-1., 0., 0.), 20., 8., .05)
    np.testing.assert_allclose(new, (19.6, 0., 0.))
    np.testing.assert_allclose(step, (19.8, 0., 0.))
    assert row["acceleration_norm"] == pytest.approx(8.)
    assert commanded == pytest.approx(800.)
    velocity = old
    for _ in range(100):
        following, _, _, _ = limited_velocity(velocity, (-1., 0., 0.), 20., 8., .05)
        assert np.linalg.norm(following-velocity) <= .4+1e-9
        assert np.linalg.norm(following) <= 20.+1e-9
        velocity = following
    np.testing.assert_allclose(velocity, (-20., 0., 0.), atol=1e-8)


def test_step_uses_mean_velocity_for_position_and_end_velocity_for_state():
    env = InterceptionEnv(InterceptionConfig(scale="1v1", intruder_mode="straight"))
    _, before = env.reset(17)
    _, _, _, _, after = env.step(reverse_actions())
    row = after["defender_acceleration"][0]
    np.testing.assert_allclose(after["defenders"][0, :3]-before["defenders"][0, :3],
                               np.array((19.8, 0., 0.))*.05)
    np.testing.assert_allclose(after["defenders"][0, 3:], row["velocity_after"])
    np.testing.assert_allclose(after["executed_velocities"][0], row["effective_step_velocity"])
    assert row["delta_v_norm"] == pytest.approx(.4)
    assert row["acceleration_limited"]
    assert after["acceleration_limited_fraction"] == 1.
    assert after["maximum_commanded_acceleration_mps2"] == pytest.approx(800.)
    assert after["maximum_executed_acceleration_mps2"] == pytest.approx(8.)


def test_continuous_contact_uses_effective_step_velocity():
    base = InterceptionConfig(scale="1v1", intruder_mode="straight", random_range=0.,
                              intruder_start_jitter_m=0.)
    defenders = ((0., 0., 60.),)+base.defender_starts[1:]
    intruders = ((1.1, 0., 60.),)+base.intruder_starts[1:]
    goals = ((-260., 0., 60.),)+base.intruder_ends[1:]
    env = InterceptionEnv(replace(base, defender_starts=defenders,
        intruder_starts=intruders, intruder_ends=goals))
    env.reset(17)
    _, _, done, _, info = env.step(reverse_actions())
    contact = next(event for event in info["events"] if event["kind"] == "interception_contact")
    expected_time = (1.1-base.contact_distance)/(19.8+20.)
    assert done and contact["time"] == pytest.approx(expected_time, abs=1e-9)
    assert contact["positions"][0][0] == pytest.approx(19.8*expected_time)
    assert contact["positions"][1][0] == pytest.approx(1.1-20.*expected_time)


def test_accel_trace_fields_and_seed_reproducibility():
    config = InterceptionConfig(scale="1v1", intruder_mode="straight", max_steps=2)
    first = record(config, 17, actions=lambda info: reverse_actions(),
                   scenario_type="geometry")
    second = record(config, 17, actions=lambda info: reverse_actions(),
                    scenario_type="geometry")
    assert first == second
    assert first["trace_version"] == TRACE_VERSION
    assert first["environment_protocol"] == PROTOCOL
    row = first["frames"][1]["defender_acceleration"][0]
    assert set(row) == {"velocity_before", "desired_velocity", "velocity_after",
                        "effective_step_velocity", "delta_v_norm", "acceleration_norm",
                        "acceleration_limited"}
    assert row["acceleration_limited"]
    assert first["result"]["acceleration_limited_steps_total"] == 2
    assert first["result"]["acceleration_limited_fraction"] == 1.


def test_predictive_guidance_keeps_prediction_and_heading_diagnostics():
    config = formal_6v6_level2_config(defender_acceleration_max=8.)
    summary, trace = run_episode(config, "three_point_pip", 17)
    assert summary["avoidance_mode"] == "none"
    assert summary["defender_acceleration_max_mps2"] == 8.
    assert summary["acceleration_limited_steps_total"] > 0
    assert 0. < summary["acceleration_limited_fraction"] <= 1.
    assert summary["maximum_executed_acceleration_mps2"] <= 8.+1e-9
    assert summary["prediction_horizon_s"]["mean"] is not None
    pair = trace["frames"][1]["control"]["pairs"][0]
    for name in ("prediction_horizon_s", "predicted_target_position_m",
                 "estimated_target_velocity_mps", "heading_error_rad",
                 "velocity_change_lower_bound_s"):
        assert name in pair
    old_velocity = np.asarray(trace["frames"][0]["defenders"][0][3:])
    direction = np.asarray(pair["guidance_direction_unit"])
    expected = np.linalg.norm(config.defender_speeds[0]*direction-old_velocity)/8.
    assert pair["velocity_change_lower_bound_s"] == pytest.approx(expected)
    assert pair["heading_error_rad"] == pytest.approx(
        np.arccos(np.clip(old_velocity@direction/np.linalg.norm(old_velocity),-1.,1.)))
