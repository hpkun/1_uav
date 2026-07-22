from math import cos

import pytest

from uav_env.actions.discrete_15 import ACTION_CONTROLS, DiscreteAction15, get_control, validate_action_table


def test_action_count_and_unique_ids() -> None:
    actions = list(DiscreteAction15)
    assert len(actions) == 15
    assert len({int(action) for action in actions}) == 15
    assert set(ACTION_CONTROLS) == set(actions)


def test_turn_bank_angles_are_opposites() -> None:
    left = get_control(DiscreteAction15.LEFT_HOLD)
    right = get_control(DiscreteAction15.RIGHT_HOLD)
    assert left.bank_angle == pytest.approx(-right.bank_angle)


def test_turn_vertical_component_is_one() -> None:
    bank_angle = get_control(DiscreteAction15.LEFT_HOLD).bank_angle
    assert 3.5 * cos(bank_angle) == pytest.approx(1.0)
    validate_action_table()
