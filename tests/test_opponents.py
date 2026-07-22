from __future__ import annotations

import numpy as np

from conftest import make_state
from uav_env.actions.discrete_15 import DiscreteAction15
from uav_env.combat.attack_geometry import AttackZoneConfig
from uav_env.core.enums import Team
from uav_env.entities.type_profiles import UAVTypeProfile
from uav_env.opponents.pursuit import PursuitOpponent
from uav_env.opponents.random import RandomOpponent
from uav_env.opponents.straight import StraightOpponent


def test_straight_and_random_opponents(profile: UAVTypeProfile) -> None:
    own = make_state(profile, team=Team.BLUE)
    target = make_state(profile, x=1000.0)
    assert StraightOpponent().select_action(own, target) is DiscreteAction15.LEVEL_HOLD
    first = RandomOpponent().select_action(own, target, np.random.default_rng(4))
    second = RandomOpponent().select_action(own, target, np.random.default_rng(4))
    assert first == second


def test_pursuit_returns_valid_action(profile: UAVTypeProfile) -> None:
    attack = AttackZoneConfig(40.0, 900.0, 0.6, 1.1, 0.8, 40.0, 1300.0, 1.1)
    policy = PursuitOpponent(profile, attack)
    action = policy.select_action(make_state(profile, team=Team.BLUE), make_state(profile, x=1000.0))
    assert action in DiscreteAction15
