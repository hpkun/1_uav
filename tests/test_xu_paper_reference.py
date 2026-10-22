"""Independent known answers, not a replacement for historical trajectory fixtures."""
import math
from pathlib import Path

import numpy as np
import pytest

from algorithm.mappo.config import TrainConfig
from algorithm.mappo.paper_protocol import OBSERVATION_FIELDS, TableIII, validate_observation_slots
from env.paper_dynamics import DynamicsDomainError, rhs, rk4_step


def test_six_equations_against_independent_scalar_values():
    # Nonzero angles and all three controls exercise every term in Eqs.(1)-(2).
    y = [10, 20, 3000, 200, math.pi/6, math.pi/3]
    u = [math.pi/3, 1.5, 4]
    g = 9.81  # explicit test choice, not a claimed paper constant
    expected = [50*math.sqrt(3), 150, 100, g,
                g/200*(2-math.sqrt(3)/2), g/50]
    np.testing.assert_allclose(rhs(y, u, gravity=g), expected, rtol=2e-15, atol=1e-14)


def test_level_trim_and_constant_acceleration_rk4():
    y = np.array([0., 0, 3000, 200, 0, 0])
    trim = rk4_step(y, [0, 0, 1], gravity=9.81)
    np.testing.assert_array_equal(trim.state, [50, 0, 3000, 200, 0, 0])
    assert trim.accepted and trim.within_published_speed_range
    accelerated = rk4_step(y, [0, 2, 1], gravity=9.81)
    np.testing.assert_allclose(accelerated.state, [50+9.81*.25**2, 0, 3000, 204.905, 0, 0])


def test_bank_sign_and_vertical_sign_follow_original_coordinates():
    y = [0, 0, 3000, 200, .2, 0]
    left = rhs(y, [-.3, 0, 1], gravity=9.81)
    right = rhs(y, [.3, 0, 1], gravity=9.81)
    assert right[5] > 0 > left[5]
    assert right[2] > 0
    assert right[5] == pytest.approx(-left[5])


def test_no_unpublished_85_degree_projection():
    gamma = math.radians(86)
    y = [0, 0, 3000, 200, gamma, 0]
    # A mathematically stationary speed/gamma at this angle, not level trim.
    step = rk4_step(y, [0, math.sin(gamma), math.cos(gamma)], gravity=9.81)
    assert step.accepted
    assert step.state[4] == gamma
    assert step.state[2] == pytest.approx(3000+50*math.sin(gamma))


@pytest.mark.parametrize('speed,gamma', [(0, 0), (-200, 0), (200, math.pi/2), (200, -math.pi/2)])
def test_singular_coordinates_reject_without_fabricated_derivatives(speed, gamma):
    y = np.array([0., 0, 3000, speed, gamma, 0])
    with pytest.raises(DynamicsDomainError):
        rhs(y, [0, 0, 1], gravity=9.81)
    step = rk4_step(y, [0, 0, 1], gravity=9.81)
    assert not step.accepted and step.reason
    np.testing.assert_array_equal(step.state, y)
    assert np.isfinite(step.state).all()


def test_rk4_stage_crossing_a_pole_is_rejected():
    y = np.array([0., 0, 3000, 100, math.radians(89), 0])
    step = rk4_step(y, [0, 1, 8], gravity=9.81)
    assert not step.accepted and 'singularity' in step.reason
    np.testing.assert_array_equal(step.state, y)


def test_speed_bound_disclosed_without_silently_projecting_equations():
    step = rk4_step([0, 0, 3000, 400, 0, 0], [0, 2, 1], gravity=9.81)
    assert step.accepted and not step.within_published_speed_range
    assert step.state[3] == pytest.approx(404.905)


@pytest.mark.parametrize('control', [[math.pi+.1, 0, 1], [0, 2.1, 1], [0, 0, -1], [0, np.nan, 1]])
def test_physical_controls_are_not_reinterpreted_or_clipped(control):
    with pytest.raises(DynamicsDomainError):
        rhs([0, 0, 3000, 200, 0, 0], control, gravity=9.81)


def test_raw_observation_protocol_exact_index_and_no_normalization():
    assert [v[1] for v in OBSERVATION_FIELDS] == [
        'O_i_jhat', 'v_i', 'gamma_i', 'psi_i', 'alpha_i', 'z_i',
        'v_jhat', 'gamma_jhat', 'psi_jhat', 'alpha_jhat', 'z_jhat', 'd_i_jhat']
    a = np.array([[1, 200, .1, .2, .3, 3000, 250, .4, .5, .6, 4000, 2000], [0]*12])
    b = validate_observation_slots(a)
    np.testing.assert_array_equal(b, a)
    assert not np.shares_memory(a, b)
    for invalid in ([.5]+[0]*11, [0]+[1]*11, [1]+[np.nan]*11):
        with pytest.raises(ValueError):
            validate_observation_slots(invalid)


def test_published_rollout_defaults_are_distinct_from_episode_length():
    p = TableIII()
    c = TrainConfig()
    y = TrainConfig.load(Path(__file__).resolve().parents[1]/'configs/mappo_coop3d.yaml')
    for key in ('actor_lr', 'critic_lr', 'gamma', 'gae_lambda', 'clip', 'entropy_coef',
                'ppo_epochs', 'minibatch_size', 'num_envs', 'horizon'):
        assert getattr(c, key) == getattr(y, key) == getattr(p, key)
    assert p.horizon == 1500 and p.max_episode_steps == 250
    assert (p.actor_input, p.critic_input, p.action_dimension) == (12, 60, 3)
    assert p.actor_hidden == p.critic_hidden == (128, 128)
