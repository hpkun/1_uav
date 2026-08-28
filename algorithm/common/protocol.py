"""Canonical experiment-protocol serialization and fingerprints."""
from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    """Return the stable JSON representation used by experiment manifests."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def config_sha256(config: Any) -> str:
    """Return the SHA-256 fingerprint of a configuration value."""
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


__all__ = ["canonical_json", "config_sha256"]
