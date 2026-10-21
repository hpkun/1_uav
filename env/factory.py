"""The sole environment construction entrypoint for algorithms and tools."""
from pathlib import Path
from .config import EnvConfig
from .cooperative_env import Cooperative3DEnv


def make_environment(config=None):
    if isinstance(config, (str, Path)):
        config = EnvConfig.load(config)
    return Cooperative3DEnv(config)
