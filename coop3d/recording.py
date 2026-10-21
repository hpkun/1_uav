"""Lossless JSON trajectory recording; replay uses only stored simulator output."""
import json
from pathlib import Path
import numpy as np
from .config import ENVIRONMENT_VERSION


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


class TrajectoryRecorder:
    def __init__(self, config, seed, initial_info):
        self.data = dict(environment_version=ENVIRONMENT_VERSION, config=config.to_dict(), seed=int(seed),
                         frames=[serializable(initial_info)])

    def append(self, info):
        self.data['frames'].append(serializable(info))

    def save(self, path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.data, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        return p
