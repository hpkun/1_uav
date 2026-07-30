"""Output-directory validation shared by training runners."""

from __future__ import annotations

import re
from pathlib import Path


_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9_.-]+")


def validate_safe_dir_component(value: str, label: str) -> str:
    """Validate that a directory component is a single safe path segment."""

    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty safe directory name")
    if value in {".", ".."}:
        raise ValueError(f"{label} cannot be '.' or '..'")
    if "/" in value or "\\" in value:
        raise ValueError(f"{label} must not contain path separators")
    if Path(value).is_absolute():
        raise ValueError(f"{label} must not be an absolute path")
    if any(ord(char) < 32 for char in value) or any(char.isspace() for char in value):
        raise ValueError(f"{label} must not contain whitespace or control characters")
    if not _SAFE_COMPONENT.fullmatch(value):
        raise ValueError(f"{label} may only contain letters, digits, underscore, dot, and dash")
    return value


def validate_run_id(run_id: str) -> str:
    """Validate a safe run id directory component."""

    return validate_safe_dir_component(run_id, "run_id")


def prepare_output_dir(output_root: str | Path, run_name: str, run_id: str) -> Path:
    """Create an empty output directory or fail before any files are written."""

    safe_name = validate_safe_dir_component(run_name, "run_name")
    safe_id = validate_run_id(run_id)
    output_dir = Path(output_root) / safe_name / safe_id
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
