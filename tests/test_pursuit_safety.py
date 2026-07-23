from conftest import make_state
from uav_env.actions.discrete_15 import DiscreteAction15
from uav_env.combat.attack_geometry import AttackZoneConfig
from uav_env.core.enums import Team
from uav_env.entities.type_profiles import UAVTypeProfile
from uav_env.opponents.pursuit import PursuitOpponent


def policy(profile: UAVTypeProfile, **weights: float) -> PursuitOpponent:
    attack=AttackZoneConfig(40,900,0.6,1.1,0.8,40,1300,1.1)
    return PursuitOpponent(profile,attack,**weights)


def test_low_and_high_altitude_choose_recovery(profile: UAVTypeProfile) -> None:
    target=make_state(profile,x=1000,z=1000,team=Team.BLUE)
    low=policy(profile).select_action(make_state(profile,z=100),target)
    high=policy(profile).select_action(make_state(profile,z=4900),target)
    assert low in {DiscreteAction15.CLIMB_HOLD,DiscreteAction15.CLIMB_ACCELERATE,DiscreteAction15.CLIMB_DECELERATE}
    assert high in {DiscreteAction15.DIVE_HOLD,DiscreteAction15.DIVE_ACCELERATE,DiscreteAction15.DIVE_DECELERATE}
    assert low == policy(profile).select_action(make_state(profile,z=100),target)


def test_weights_are_active_configuration(profile: UAVTypeProfile) -> None:
    first=policy(profile,angle_weight=9.0,distance_weight=0.0,altitude_weight=0.0)
    second=policy(profile,angle_weight=0.0,distance_weight=9.0,altitude_weight=0.0)
    ownship = make_state(profile, x=0, y=0, z=2000, team=Team.RED)
    target = make_state(profile, x=200, y=-1500, z=1500, team=Team.BLUE)

    assert first.select_action(ownship, target) != second.select_action(ownship, target)
