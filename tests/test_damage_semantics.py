from conftest import make_state
from uav_env.combat.damage import DamageConfig, apply_damage
from uav_env.entities.type_profiles import UAVTypeProfile


def test_nonfatal_hit_sets_ever_hit_not_damaged(profile: UAVTypeProfile) -> None:
    updated, result = apply_damage(make_state(profile), DamageConfig(), 0.5)
    assert result.nominal_damage == 11.0
    assert updated.ever_hit and updated.alive
    assert not updated.damaged


def test_destroyed_state_is_consistent(profile: UAVTypeProfile) -> None:
    updated, _ = apply_damage(make_state(profile, health=10.0), DamageConfig(), 0.0)
    assert updated.damaged and not updated.alive and updated.health == 0.0
    updated.validate_consistency()
