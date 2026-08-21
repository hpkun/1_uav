"""Configuration loading for the public combat environment."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml
from .models import AircraftSpec


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML configuration and validate required top-level sections."""
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    required = {
        "simulation", "action", "aircraft", "flight_envelope", "scenario",
        "weapon", "reward", "observation", "blue_policy",
    }
    if not isinstance(config, dict) or not required.issubset(config):
        raise ValueError(f"configuration must contain: {sorted(required)}")
    return config


def aircraft_spec(config: dict[str, Any]) -> AircraftSpec:
    """Build the immutable homogeneous aircraft specification."""
    return AircraftSpec(**config["aircraft"])
