"""Public low-fidelity 4v4 combat-environment components."""

from .env import MultiUAVCombatEnv
from .factory import make_combat_environment
from .geometry import EngagementGeometry, engagement_geometry
from .persistent_env import PersistentWaveCombatEnv
from .weapon import FireState, WeaponEnvelope

__all__ = [
    "EngagementGeometry", "FireState", "MultiUAVCombatEnv",
    "PersistentWaveCombatEnv", "WeaponEnvelope", "engagement_geometry",
    "make_combat_environment",
]
