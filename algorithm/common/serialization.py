"""JSON serialization shared by logging and artifact tools."""
import json
from pathlib import Path
import numpy as np


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(v) for v in value]
    return value


def write_json(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(serializable(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
