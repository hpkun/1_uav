"""YAML configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from math import isfinite

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load one YAML mapping from *path*."""

    with Path(path).open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError("Configuration root must be a mapping")
    return data


def deep_merge(*mappings: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings from left to right without mutating inputs."""

    result: dict[str, Any] = {}
    for mapping in mappings:
        for key, value in mapping.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = deep_merge(result[key], value)
            elif isinstance(value, dict):
                result[key] = deep_merge(value)
            else:
                result[key] = value
    return result


def project_root() -> Path:
    """Return the editable-install project root containing ``configs``."""

    return Path(__file__).resolve().parents[3]


def validate_experiment_config(config: dict[str, Any]) -> None:
    """Validate timing consistency and finite physical experiment values."""

    def validate_numeric_values(value: Any, path: str) -> None:
        if isinstance(value, bool) or value is None or isinstance(value, str):
            return
        if isinstance(value, (int, float)):
            if not isfinite(float(value)):
                raise ValueError(f"Configuration value {path} must be finite")
            return
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                validate_numeric_values(child_value, f"{path}.{child_key}")
            return
        if isinstance(value, (list, tuple)):
            for index, child_value in enumerate(value):
                validate_numeric_values(child_value, f"{path}[{index}]")

    validate_numeric_values(config, "config")

    physics_dt = float(config["physics_dt"])
    decision_dt = float(config["decision_dt"])
    ratio = decision_dt / physics_dt
    if physics_dt <= 0.0 or decision_dt <= 0.0 or not isfinite(ratio):
        raise ValueError("Physics and decision time steps must be finite and positive")
    if abs(ratio - round(ratio)) > 1.0e-10:
        raise ValueError("decision_dt / physics_dt must be a positive integer")
    if int(config["physics_steps_per_action"]) != int(round(ratio)):
        raise ValueError("physics_steps_per_action is inconsistent with time steps")
    expected_steps = float(config["max_episode_seconds"]) / decision_dt
    if abs(expected_steps - int(config["max_decision_steps"])) > 1.0e-10:
        raise ValueError("max_decision_steps is inconsistent with episode duration")
    physical_keys = (
        "gravity",
        "min_altitude",
        "max_altitude",
        "min_speed",
        "max_speed",
        "min_flight_path_angle",
        "max_flight_path_angle",
        "min_tangential_overload",
        "max_tangential_overload",
        "min_normal_overload",
        "max_normal_overload",
        "initial_health",
    )
    if not all(isfinite(float(config[key])) for key in physical_keys):
        raise ValueError("All physical parameters must be finite")
    if float(config["min_speed"]) >= float(config["max_speed"]):
        raise ValueError("min_speed must be less than max_speed")
    if float(config["min_altitude"]) >= float(config["max_altitude"]):
        raise ValueError("min_altitude must be less than max_altitude")


def load_experiment_config(
    config_name: str = "paper_2024_homogeneous",
    scenario: str = "tail_chase",
    config_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Load and validate base, platform-paper, and 1v1 scenario settings."""

    directory = Path(config_directory) if config_directory is not None else project_root() / "configs"
    scenario_files = {
        "tail_chase": "scenario_1v1_tail_chase.yaml",
        "head_on": "scenario_1v1_head_on.yaml",
        "balanced_random": "scenario_1v1_balanced.yaml",
        "balanced": "scenario_1v1_balanced.yaml",
    }
    if scenario not in scenario_files:
        raise ValueError(f"Unknown 1v1 scenario: {scenario!r}")
    paper_path = directory / f"{config_name}.yaml"
    if not paper_path.is_file():
        raise ValueError(f"Unknown platform config: {config_name!r}")
    config = deep_merge(
        load_yaml(directory / "base.yaml"),
        load_yaml(paper_path),
        load_yaml(directory / scenario_files[scenario]),
    )
    validate_experiment_config(config)
    return config


def load_multi_experiment_config(
    config_name: str = "paper_2024_homogeneous",
    scenario: str = "head_on_formation",
    config_directory: str | Path | None = None,
    team_size: int = 2,
) -> dict[str, Any]:
    """Load a fixed homogeneous 2v2 scenario or the 3v3 head-on scenario."""

    directory = Path(config_directory) if config_directory is not None else project_root() / "configs"
    scenario_files = {
        "head_on_formation": "scenario_2v2_head_on.yaml",
        "offset_formation": "scenario_2v2_offset.yaml",
        "balanced_random": "scenario_2v2_balanced.yaml",
        "head_on_mirrored_jitter_v2": "scenario_3v3_v2.yaml",
        "symmetric_stress_test_v2": "scenario_3v3_symmetric_stress_v2.yaml",
    }
    if scenario not in scenario_files:
        raise ValueError(f"Unknown multi-aircraft scenario: {scenario!r}")
    if team_size not in {2, 3}:
        raise ValueError("team_size must be 2 or 3")
    if team_size == 3 and scenario not in {"head_on_formation", "head_on_mirrored_jitter_v2", "symmetric_stress_test_v2"}:
        raise ValueError("Fixed 3v3 currently supports only head-on legacy or V2 scenarios")
    if scenario in {"head_on_mirrored_jitter_v2", "symmetric_stress_test_v2"} and team_size != 3:
        raise ValueError("V2 fixed homogeneous scenarios require team_size=3")
    paper_path = directory / f"{config_name}.yaml"
    config = deep_merge(load_yaml(directory / "base.yaml"), load_yaml(paper_path), load_yaml(directory / scenario_files[scenario]))
    config["red_count"] = team_size
    config["blue_count"] = team_size
    if config.get("environment_schema_version") == "homogeneous_3v3_v2":
        raise ValueError("homogeneous_3v3_v2 was a development-only 62D/60D schema and is not runnable; use homogeneous_3v3_v2_timeaware")
    validate_experiment_config(config)
    return config
