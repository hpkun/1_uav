from __future__ import annotations

from math import pi

import pytest

from uav_env.core.enums import Team
from uav_env.core.state import UAVState
from uav_env.entities.type_profiles import UAVTypeProfile, profile_from_config
from uav_env.utils.config import load_experiment_config


@pytest.fixture
def experiment_config() -> dict[str, object]:
    return load_experiment_config("paper_2024_homogeneous", "tail_chase")


@pytest.fixture
def profile(experiment_config: dict[str, object]) -> UAVTypeProfile:
    return profile_from_config(experiment_config)


def make_state(
    profile: UAVTypeProfile,
    *,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 1500.0,
    speed: float = 100.0,
    heading: float = 0.0,
    team: Team = Team.RED,
    health: float = 300.0,
) -> UAVState:
    return UAVState(x, y, z, speed, 0.0, heading % (2.0 * pi), health, True, int(team), profile.type_id)
