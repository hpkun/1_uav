"""Output-directory validation shared by training runners."""

from __future__ import annotations

import re
from pathlib import Path


_SAFE_RUN_ID = re.compile(r"[A-Za-z0-9_.-]+")


def validate_run_id(run_id: str) -> str:
    """Validate that a run id is a single safe directory name."""

    if not run_id:
        raise ValueError("run_id must be a non-empty safe directory name")
    if run_id in {".", ".."}:
        raise ValueError("run_id cannot be '.' or '..'")
    if "/" in run_id or "\\" in run_id:
        raise ValueError("run_id must not contain path separators")
    if Path(run_id).is_absolute():
        raise ValueError("run_id must not be an absolute path")
    if not _SAFE_RUN_ID.fullmatch(run_id):
        raise ValueError("run_id may only contain letters, digits, underscore, dot, and dash")
    return run_id


def prepare_output_dir(output_root: str | Path, run_name: str, run_id: str) -> Path:
    """Create an empty output directory or fail before any files are written."""

    safe_id = validate_run_id(run_id)
    output_dir = Path(output_root) / run_name / safe_id
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
