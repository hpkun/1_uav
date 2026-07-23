from copy import deepcopy
from dataclasses import replace

import pytest

from conftest import make_state
from uav_env.combat.events import EpisodeOutcome
from uav_env.entities.uav import UAV
from uav_env.rewards.multi_reward import multi_terminal_reward_allocations


def aircraft(uid, profile):
    state = make_state(profile)
    return UAV(uid, state.team_id, state, profile)


def exact_config(experiment_config):
    config = deepcopy(experiment_config)
    config["multi_terminal_reward_profile"] = "paper_2024_exact"
    weights = [1.0 / 3.0] * 3
    config["project_assumptions"]["multi_terminal_reward"]["win_weights"] = weights
    config["project_assumptions"]["multi_terminal_reward"]["lose_weights"] = weights
    return config


def test_win_equations_21_and_22_are_decomposed_exactly(profile, experiment_config):
    config = exact_config(experiment_config)
    reds = [aircraft("red_0", profile), aircraft("red_1", profile)]
    reds[1].state = replace(reds[1].state, health=150.0)
    outcome = EpisodeOutcome("red", True, False, "blue_eliminated", 1, 0.5, 2, 0)
    allocations = multi_terminal_reward_allocations(outcome, reds, {"red_0": 3.0, "red_1": 1.0}, config)
    base = 50.0 * 2.0 * (0.75 + 0.25 * 399.0 / 400.0)
    expected = [
        ((1/3)/2, 0.06, (1/3)*3/4, (1/3)*(300/450)*(300/300)),
        ((1/3)/2, 0.06, (1/3)*1/4, (1/3)*(150/450)*(150/300)),
    ]
    for index, key in enumerate(("red_0", "red_1")):
        item = allocations[key]
        assert item.team_base == pytest.approx(base)
        assert item.base_share_component == pytest.approx(expected[index][0])
        assert item.survival_component == pytest.approx(expected[index][1])
        assert item.contribution_component == pytest.approx(expected[index][2])
        assert item.health_component == pytest.approx(expected[index][3])
        assert item.allocation_factor == pytest.approx(sum(expected[index]))
        assert item.reward == pytest.approx(base * sum(expected[index]))
        assert item.alive_count == 2
        assert item.contribution_denominator == pytest.approx(4.0)
        assert item.health_denominator == pytest.approx(450.0)


def test_win_last_step_time_factor_is_point_75(profile, experiment_config):
    config = exact_config(experiment_config)
    reds = [aircraft("red_0", profile), aircraft("red_1", profile)]
    outcome = EpisodeOutcome("red", True, False, "blue_eliminated", 400, 200.0, 2, 0)
    assert multi_terminal_reward_allocations(outcome, reds, {}, config)["red_0"].team_base == pytest.approx(50*2*0.75)


def test_zero_beta_uses_uniform_project_numerical_convention(profile, experiment_config):
    config = exact_config(experiment_config)
    reds = [aircraft("red_0", profile), aircraft("red_1", profile)]
    outcome = EpisodeOutcome("red", True, False, "blue_eliminated", 1, 0.5, 2, 0)
    allocations = multi_terminal_reward_allocations(outcome, reds, {}, config)
    assert allocations["red_0"].contribution_component == pytest.approx((1/3)/2)
    assert allocations["red_1"].contribution_component == pytest.approx((1/3)/2)
    assert allocations["red_0"].contribution_denominator == 0.0


def test_one_survivor_shared_term_and_health_double_factor(profile, experiment_config):
    config = exact_config(experiment_config)
    reds = [aircraft("red_0", profile), aircraft("red_1", profile)]
    reds[1].state = replace(reds[1].state, health=0.0, alive=False, damaged=True)
    outcome = EpisodeOutcome("red", True, False, "blue_eliminated", 20, 10.0, 1, 0)
    allocations = multi_terminal_reward_allocations(outcome, reds, {"red_0": 1.0, "red_1": 1.0}, config)
    assert allocations["red_0"].survival_component == pytest.approx(0.03)
    assert allocations["red_1"].survival_component == pytest.approx(0.03)
    assert allocations["red_0"].health_component == pytest.approx(1/3)
    assert allocations["red_1"].health_component == 0.0


def test_loss_equations_23_to_25_use_max_denominator_and_b0(profile, experiment_config):
    config = exact_config(experiment_config)
    reds = [aircraft("red_0", profile), aircraft("red_1", profile)]
    for red in reds:
        red.state = replace(red.state, health=0.0, alive=False, damaged=True)
    outcome = EpisodeOutcome("blue", False, True, "red_eliminated", 400, 200.0, 0, 2)
    allocations = multi_terminal_reward_allocations(outcome, reds, {"red_0": 4.0, "red_1": 1.0}, config)
    base = -50.0 * 2.0 * 0.80
    for key, beta_prime in (("red_0", 1.0), ("red_1", 4.0)):
        item = allocations[key]
        factor = (1/3)/2 + 0.0 + (1/3)*beta_prime/4 + (1/3)*310/300
        assert item.team_base == pytest.approx(base)
        assert item.contribution_denominator == 4.0
        assert item.health_denominator == 300.0
        assert item.health_component == pytest.approx((1/3)*310/300)
        assert item.reward == pytest.approx(base*factor)


def test_loss_one_survivor_uses_same_negative_shared_term(profile, experiment_config):
    config = exact_config(experiment_config)
    reds = [aircraft("red_0", profile), aircraft("red_1", profile)]
    reds[1].state = replace(reds[1].state, health=0.0, alive=False)
    outcome = EpisodeOutcome("blue", True, False, "red_eliminated", 50, 25.0, 1, 2)
    allocations = multi_terminal_reward_allocations(outcome, reds, {}, config)
    assert all(item.survival_component == pytest.approx(-0.02) for item in allocations.values())


def test_paper_and_project_profiles_are_distinct_and_reproducible(profile, experiment_config):
    config = exact_config(experiment_config)
    reds = [aircraft("red_0", profile), aircraft("red_1", profile)]
    outcome = EpisodeOutcome("red", True, False, "blue_eliminated", 100, 50.0, 2, 0)
    paper_a = multi_terminal_reward_allocations(outcome, reds, {"red_0": 3.0, "red_1": 1.0}, config)
    paper_b = multi_terminal_reward_allocations(outcome, reds, {"red_0": 3.0, "red_1": 1.0}, config)
    config["multi_terminal_reward_profile"] = "project_balanced"
    project_a = multi_terminal_reward_allocations(outcome, reds, {"red_0": 3.0, "red_1": 1.0}, config)
    project_b = multi_terminal_reward_allocations(outcome, reds, {"red_0": 3.0, "red_1": 1.0}, config)
    assert paper_a == paper_b and project_a == project_b
    assert [v.reward for v in paper_a.values()] != [v.reward for v in project_a.values()]


def test_paper_weights_need_not_sum_to_one_but_must_be_finite_nonnegative(profile, experiment_config):
    config = exact_config(experiment_config)
    config["project_assumptions"]["multi_terminal_reward"]["win_weights"] = [1.0, 2.0, 3.0]
    reds = [aircraft("red_0", profile), aircraft("red_1", profile)]
    outcome = EpisodeOutcome("red", True, False, "blue_eliminated", 10, 5.0, 2, 0)
    assert multi_terminal_reward_allocations(outcome, reds, {}, config)
    config["project_assumptions"]["multi_terminal_reward"]["win_weights"] = [1.0, float("nan"), 3.0]
    with pytest.raises(ValueError, match="finite"):
        multi_terminal_reward_allocations(outcome, reds, {}, config)
