"""Common validation helpers."""

from math import isfinite


def require_finite(value: float, name: str) -> float:
    """Return *value* or raise when it is not finite."""

    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value
