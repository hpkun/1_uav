from conftest import make_state
from uav_env.combat.damage import DamageConfig, apply_damage
from uav_env.entities.type_profiles import UAVTypeProfile


def test_overkill_is_separated_from_effective_damage(profile: UAVTypeProfile) -> None:
    updated, result = apply_damage(make_state(profile, health=3.0), DamageConfig(), 0.0)
    assert result.nominal_damage == 51.0
    assert result.effective_damage == 3.0
    assert result.overkill_damage == 48.0
    assert result.hit and result.destroyed and updated.health == 0.0
