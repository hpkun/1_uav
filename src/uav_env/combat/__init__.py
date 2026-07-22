"""Combat geometry, damage, events, and episode outcomes."""

from uav_env.combat.attack_geometry import AttackZoneConfig, CombatGeometry, compute_combat_geometry
from uav_env.combat.damage import DamageConfig, DamageResult, sample_damage
from uav_env.combat.events import CombatEvent, EpisodeOutcome

__all__ = [
    "AttackZoneConfig",
    "CombatEvent",
    "CombatGeometry",
    "DamageConfig",
    "DamageResult",
    "EpisodeOutcome",
    "compute_combat_geometry",
    "sample_damage",
]
