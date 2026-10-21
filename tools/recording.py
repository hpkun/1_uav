"""Lossless JSON trajectory recording; replay uses only stored simulator output."""
import json
from pathlib import Path
from env.config import ENVIRONMENT_VERSION
from algorithm.common.serialization import serializable


class TrajectoryRecorder:
    def __init__(self, config, seed, initial_info):
        self.data = dict(environment_version=ENVIRONMENT_VERSION, config=config.to_dict(), seed=int(seed),
                         frames=[serializable(initial_info)])

    def append(self, info, *, policy_diagnostics=None):
        frame = serializable(info)
        if policy_diagnostics is not None:
            frame['policy_diagnostics'] = serializable(policy_diagnostics)
        self.data['frames'].append(frame)

    def save(self, path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.data, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        return p
