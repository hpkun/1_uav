import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from env.interception import InterceptionEnv


# Absolute bounds for the fixed historical trajectories. See the fixture notes.
POSITION_ATOL = 4 * math.ulp(160.)
VELOCITY_ATOL = 16 * math.ulp(13.)
TIME_ATOL = 8 * math.ulp(12.)
PROGRESS_ATOL = 2 * POSITION_ATOL / 40.


def _close(actual, expected, atol, context):
    assert np.shape(actual) == np.shape(expected), context
    np.testing.assert_allclose(actual, expected, rtol=0., atol=atol, equal_nan=False, err_msg=context)


def _state(info, expected, context):
    for key in ("step", "outcome"):
        assert info[key] == expected[key], context
    for key in ("active_mask", "evader_mask"):
        np.testing.assert_array_equal(info[key], expected[key], err_msg=f"{context}: {key}")
    for side in ("pursuers", "evaders"):
        state = np.asarray(expected[side])
        _close(info[side][:, :3], state[:, :3], POSITION_ATOL, f"{context}: {side} position")
        _close(info[side][:, 3:], state[:, 3:], VELOCITY_ATOL, f"{context}: {side} velocity")
    _close(info["time"], expected["time"], TIME_ATOL, f"{context}: time")


@pytest.fixture(scope="session")
def check_interception_history():
    path = Path(__file__).with_name("fixtures") / "historical_interception.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        references = json.load(stream)

    def check(case, steps, digest, config=None):
        reference = references[case]
        frames = reference["frames"]
        fields = ("rewards", "reward_parts")
        if case != "default":
            fields = ("evaders", "commands", *fields)
        payload = json.dumps([[frame[key] for key in fields] for frame in frames],
                             sort_keys=True, separators=(",", ":")).encode()
        assert hashlib.sha256(payload).hexdigest() == digest, "historical reference fingerprint changed"
        assert len(frames) == steps

        env = InterceptionEnv(config)
        _, info = env.reset(1200, {"scenario": "fixed"})
        _state(info, reference["initial"], f"{case} reset")
        action = np.full((5, 3), .5)
        for step, expected in enumerate(frames, 1):
            context = f"{case} step {step}"
            _, rewards, terminated, truncated, info = env.step(action)
            assert (terminated, truncated) == (expected["terminated"], expected["truncated"]), context
            _state(info, expected, context)
            for key in ("commands", "requested_velocities"):
                _close(info[key], expected[key], VELOCITY_ATOL, f"{context}: {key}")
            assert rewards.dtype == np.float32, context
            expected_rewards = np.asarray(expected["rewards"], dtype=np.float32)
            np.testing.assert_array_equal(rewards.view(np.uint32), expected_rewards.view(np.uint32),
                                          err_msg=f"{context}: reward bits")
            parts = info["reward_parts"].copy()
            expected_parts = expected["reward_parts"].copy()
            _close(parts.pop("progress"), expected_parts.pop("progress"), PROGRESS_ATOL, f"{context}: progress reward")
            assert parts == expected_parts, context

            assert len(info["events"]) == len(expected["events"]), context
            for event, archived in zip(info["events"], expected["events"], strict=True):
                event, archived = event.copy(), archived.copy()
                for key, atol in (("time", TIME_ATOL), ("positions", POSITION_ATOL), ("velocities", VELOCITY_ATOL)):
                    _close(event.pop(key), archived.pop(key), atol, f"{context}: event {key}")
                assert event == archived, context
        assert terminated and not truncated and info["outcome"] == "breach"

    return check
