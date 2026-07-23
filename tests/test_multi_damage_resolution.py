from math import pi

from conftest import make_state
from uav_env.combat.attack_geometry import AttackZoneConfig
from uav_env.combat.damage import DamageConfig
from uav_env.combat.multi_combat import resolve_multi_attacks
from uav_env.core.enums import Team
from uav_env.entities.uav import UAV
from uav_env.entities.type_profiles import UAVTypeProfile
import numpy as np


def test_multi_attack_resolution_is_order_independent(profile: UAVTypeProfile) -> None:
    aircraft=[UAV("red_0",1,make_state(profile,x=0),profile),UAV("red_1",1,make_state(profile,x=20),profile),UAV("blue_0",0,make_state(profile,x=200,heading=pi,team=Team.BLUE,health=3),profile)]
    cfg=AttackZoneConfig(40,900,pi,pi,pi,40,1300,pi)
    first=resolve_multi_attacks(aircraft,cfg,DamageConfig(),np.random.default_rng(3))
    second=resolve_multi_attacks(list(reversed(aircraft)),cfg,DamageConfig(),np.random.default_rng(3))
    assert first.updated_states["blue_0"].health==second.updated_states["blue_0"].health
    assert sum(a.effective_damage for a in first.resolved_attacks if a.target_id=="blue_0")<=3.0
    credits=[a for a in first.resolved_attacks if a.destroy_credit]
    assert len(credits)<=1
