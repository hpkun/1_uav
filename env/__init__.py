"""Environment and task rules; never depends on the learning algorithm."""
from .config import ENVIRONMENT_VERSION, EnvConfig
from .cooperative_env import Cooperative3DEnv

__all__ = ['Cooperative3DEnv', 'EnvConfig', 'ENVIRONMENT_VERSION']
