"""Non-weaponized two-team moving-region competition, independent of Coop3D."""
from .config import RegionConfig, ENVIRONMENT_VERSION
from .environment import RegionCompetitionEnv

__all__ = ['RegionConfig', 'RegionCompetitionEnv', 'ENVIRONMENT_VERSION']
