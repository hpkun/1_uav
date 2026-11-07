import numpy as np
import pytest

from algorithm.interception import episode
from env.interception import InterceptionConfig, InterceptionEnv, SimulationFailure
from env.interception.config import MAX_PURSUERS, OBS_DIM, STATE_DIM, SCALES
from env.interception.events import windows
from env.interception.motion import advance, command
from env.interception.reward import reward_parts


def actions(x=0., config=None):
    limit = (config or InterceptionConfig()).pursuer_vmax
    array = np.full((MAX_PURSUERS, 3), .5)
    array[:, 0] += np.asarray(x) / (2 * limit)
    return array


def custom(env, p, e):
    return env.reset(12, {"scenario": "custom", "pursuers": p, "evaders": e})


def test_analytic_motion_semigroup_and_rotation():
    state = np.array([3., 4., 15., 1., -2., 3.])
    target = np.array([2., 5., -1.])
    tau, duration = 1.5, .7
    decay = np.exp(-duration / tau)
    expected = np.r_[state[:3] + target * duration + tau * (state[3:] - target) * (1-decay),
                     target + (state[3:] - target) * decay]
    actual = advance(state, target, tau, duration)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
    np.testing.assert_allclose(advance(advance(state, target, tau, .2), target, tau, .5), actual)
    rotation = [2, 0, 1, 5, 3, 4]
    np.testing.assert_allclose(advance(state[rotation], target[[2, 0, 1]], tau, duration), actual[rotation])
    np.testing.assert_array_equal(advance(state, target, tau, duration), actual)


def test_hover_stop_and_vertical_tiny_step():
    state = np.array([0., 0., 15., 0., 0., 0.])
    np.testing.assert_array_equal(advance(state, [0, 0, 0], 1.5, .1), state)
    accelerated = advance(state, [0, 0, 8], 1.5, .1)
    stopped = advance(accelerated, [0, 0, 0], 1.5, 100.)
    assert accelerated[2] > 15 and 0 < stopped[5] < 1e-25
    assert stopped[2] == pytest.approx(15.8)
    assert advance(np.zeros(6), [0, 0, 8], 1.5, 1e-9)[2] == pytest.approx(8e-18/3, rel=1e-8, abs=0)


def test_speed_is_projected_in_three_dimensions():
    direction = command([[0., 1., 1.]], 8.)[0]
    assert np.linalg.norm(direction) == pytest.approx(8.)
    state = [0., 0., 15., 8., 0., 0.]
    for dt in (.1, 3., 100.):
        assert np.linalg.norm(advance(state, direction, 1.5, dt)[3:]) <= 8. + 1e-12


@pytest.mark.parametrize("scale", SCALES)
def test_reset_slots_and_normalization(scale):
    config = InterceptionConfig(scale=scale)
    env = InterceptionEnv(config)
    obs, info = env.reset(123)
    repeat, same = env.reset(123)
    np.testing.assert_array_equal(obs, repeat)
    assert (OBS_DIM, STATE_DIM) == (52, 54)
    assert obs.shape == (5, 52) and obs.dtype == np.float32
    assert info["centralized_state"].shape == (54,)
    n, m = SCALES[scale]
    assert (sum(info["active_mask"]), sum(info["evader_mask"])) == (n, m)
    np.testing.assert_array_equal(obs[n:], 0.)
    np.testing.assert_allclose(obs[0, :3], info["pursuers"][0, :3] / config.position_scale)
    np.testing.assert_allclose(obs[0, 10:13],
                               (info["evaders"][0, :3]-info["pursuers"][0, :3]) / config.position_scale)
    assert obs[0, 16] == 1 and np.all(obs[:, 10+7*m:24] == 0)
    assert info["centralized_state"][48] == (m == 2)
    _, other = env.reset(124)
    assert not np.array_equal(info["evaders"], other["evaders"])
    assert not hasattr(env.state, "frame")


@pytest.mark.parametrize("mode", ["constant_velocity", "goal_seeking", "bounded_maneuver", "reactive_evasion"])
def test_policy_speed_and_snapshot_reproducibility(mode):
    config = InterceptionConfig(evader_mode=mode, reactive_range=50.)
    env = InterceptionEnv(config)
    env.reset(42)
    snap = env.snapshot()
    result = env.step(actions())
    env.restore(snap)
    restored = env.step(actions())
    np.testing.assert_array_equal(result[0], restored[0])
    np.testing.assert_array_equal(result[4]["commands"], restored[4]["commands"])
    target = result[4]["commands"][5]
    assert np.linalg.norm(target) <= config.evader_vmax + 1e-12
    if mode == "constant_velocity":
        np.testing.assert_array_equal(target, [-6., 0., 0.])
    elif mode == "goal_seeking":
        ray = np.array(config.center) - snap["state"].evaders[0, :3]
        np.testing.assert_allclose(target, 6 * ray / np.linalg.norm(ray))
    elif mode == "reactive_evasion":
        ray = np.array(config.center) - snap["state"].evaders[0, :3]
        assert not np.allclose(target, 6 * ray / np.linalg.norm(ray))


def test_fixed_maneuver_is_seed_invariant():
    env = InterceptionEnv(InterceptionConfig(evader_mode="bounded_maneuver"))
    commands = []
    for seed in (1, 999):
        env.reset(seed, {"scenario": "fixed"})
        commands.append(env.step(actions())[4]["commands"])
    np.testing.assert_array_equal(*commands)


def test_invalid_initialization_and_explicit_failure():
    env = InterceptionEnv(InterceptionConfig(scale="1v1"))
    with pytest.raises(ValueError, match="overlap"):
        custom(env, [[10, 0, 15, 0, 0, 0]], [[10, 0, 15, 0, 0, 0]])
    with pytest.raises(ValueError, match="initial breach"):
        custom(env, [[20, 0, 15, 0, 0, 0]], [[0, 0, 15, 0, 0, 0]])
    with pytest.raises(ValueError, match="initialization rejected"):
        env.reset(1, {"scenario": "random", "distance": 1000})
    env.reset(12)
    with pytest.raises(SimulationFailure, match="simulation_failure"):
        env.step(actions(np.nan))
    assert env.state.step == 0


def test_swept_collision_with_both_endpoints_outside():
    env = InterceptionEnv(InterceptionConfig(dt=1., evader_mode="constant_velocity"))
    p = [[-2, 20, 15, 8, 0, 0], [2, 20, 15, -8, 0, 0], [0, 30, 15, 0, 0, 0]]
    e = [[50, 0, 15, 0, 0, 0]]
    custom(env, p, e)
    _, _, term, trunc, info = env.step(actions([8, -8, 0, 0, 0]))
    assert term and not trunc and info["outcome"] == "collision"
    assert info["time"] == pytest.approx((4-.6)/16, abs=1e-7)
    assert info["reward_parts"]["capture_event"] == 0


def test_swept_protected_zone_breach():
    config = InterceptionConfig(scale="1v1", dt=3., evader_mode="constant_velocity",
                                constant_direction=(1, 0, 0))
    env = InterceptionEnv(config)
    custom(env, [[20, 20, 15, 0, 0, 0]], [[-8, 0, 15, 6, 0, 0]])
    _, _, term, trunc, info = env.step(actions())
    assert term and not trunc and info["outcome"] == "breach"
    assert info["time"] == pytest.approx(.5, abs=1e-7)
    assert info["reward_parts"]["terminal_failure"] == -10


def test_tangency_and_curved_threshold_windows():
    # The chord misses this sphere, but the actual curved trajectory enters.
    curve = lambda t: np.array([2*t-1, 1.2-1.6*t*(1-t), 0.])
    found = windows(curve, 3.2, 1., 0., 1., 1e-7)
    assert len(found) == 1 and found[0][0] < .5 < found[0][1]
    tangent = windows(lambda t: np.array([2*t-1, 1., 0.]), 0., 1., 0., 1., 1e-7)
    assert tangent == [(.5, .5)]
    assert not windows(lambda t: np.array([2*t-1, 1.001, 0.]), 0., 1., 0., 1., 1e-7)


def test_collision_wins_capture_tie():
    config = InterceptionConfig(scale="1v1", dt=.5, capture_dwell_steps=2,
                                evader_mode="constant_velocity", evader_speed_fraction=.25,
                                constant_direction=(1, 0, 0))
    env = InterceptionEnv(config)
    custom(env, [[-25, 10, 15, 2.9, 0, 0]], [[-22.3, 10, 15, 1.5, 0, 0]])
    for _ in range(3):
        _, _, term, _, info = env.step(actions(2.9))
    assert term and info["outcome"] == "collision"
    assert info["time"] == pytest.approx(1.5, abs=1e-7)
    assert [e["kind"] for e in info["events"]] == ["collision"]
    assert info["reward_parts"]["capture_event"] == 0


@pytest.mark.parametrize(("closing", "expected"), [(1.3, "success"), (1.45, "collision")])
def test_first_event_wins_when_capture_and_collision_are_in_same_step(closing, expected):
    config = InterceptionConfig(scale="1v1", dt=1., capture_dwell_steps=1,
                                evader_mode="constant_velocity", evader_speed_fraction=.25,
                                constant_direction=(1, 0, 0))
    env = InterceptionEnv(config)
    p_speed = 1.5 + closing
    custom(env, [[-25, 10, 15, p_speed, 0, 0]], [[-22.5, 10, 15, 1.5, 0, 0]])
    assert not env.step(actions(p_speed))[2]
    _, _, terminated, _, info = env.step(actions(p_speed))
    capture_time = .5 / closing + 1.
    collision_time = 1.9 / closing
    assert terminated and info["outcome"] == expected
    assert info["time"] == pytest.approx(min(capture_time, collision_time), abs=1e-7)
    assert len(info["events"]) == 1


def test_dwell_resets_when_relative_speed_becomes_excessive():
    config = InterceptionConfig(scale="1v1", pursuer_tau=.001, evader_tau=.001,
                                evader_mode="constant_velocity", evader_speed_fraction=.01,
                                constant_direction=(1, 0, 0))
    env = InterceptionEnv(config)
    custom(env, [[-25, 10, 15, 1.06, 0, 0]], [[-22.95, 10, 15, .06, 0, 0]])
    for _ in range(2):
        _, _, _, _, info = env.step(actions(1.06))
    assert info["dwell_seconds"][0, 0] == pytest.approx(.15, abs=1e-7)
    _, _, _, _, info = env.step(actions(3.))
    assert info["dwell_seconds"][0, 0] == 0
    for _ in range(4):
        _, _, terminated, _, info = env.step(actions(.06))
        assert not terminated
    _, _, terminated, _, info = env.step(actions(.06))
    # Deceleration consumes a fraction of the first dwell step.
    assert not terminated
    _, _, terminated, _, info = env.step(actions(.06))
    assert terminated and info["outcome"] == "success"


def test_partial_capture_stable_ids_and_reward():
    config = InterceptionConfig(scale="5v2", evader_mode="constant_velocity",
                                evader_speed_fraction=.1, constant_direction=(1, 0, 0))
    env = InterceptionEnv(config)
    custom(env, [[-30, -12, 15, 1.2, 0, 0], [30, 12, 15, 1.2, 0, 0],
                 [-15, -20, 15, 0, 0, 0], [15, 20, 15, 0, 0, 0], [0, 25, 15, 0, 0, 0]],
           [[-27.8, -12, 15, .6, 0, 0], [32.6, 12, 15, .6, 0, 0]])
    recorded = []
    for _ in range(20):
        obs, _, term, _, info = env.step(actions([1.2, 1.2, 0, 0, 0]))
        if info["events"]:
            recorded.append(info)
        if not info["evader_mask"][0]:
            np.testing.assert_array_equal(obs[0, 10:17], 0.)
        if term:
            break
    assert [x["events"][0]["entities"] for x in recorded] == [[0,5],[1,6]]
    assert recorded[0]["outcome"] == "ongoing" and recorded[1]["outcome"] == "success"
    assert recorded[0]["events"][0]["time"] == pytest.approx(.2 / .6 + .5, abs=1e-7)
    assert recorded[0]["reward_parts"]["progress"] == pytest.approx(.001, abs=1e-8)
    assert [x["reward_parts"]["capture_event"] for x in recorded] == [5,5]
    np.testing.assert_array_equal(recorded[0]["evaders"][0], recorded[1]["evaders"][0])


def test_timeout_and_reward_purity():
    config = InterceptionConfig(scale="1v1", dt=1., max_steps=1, evader_mode="constant_velocity",
                                constant_direction=(0, 1, 0))
    env = InterceptionEnv(config)
    custom(env, [[20, 0, 15, 0, 0, 0]], [[23, 0, 15, 0, 0, 0]])
    _, _, terminated, truncated, info = env.step(actions())
    assert not terminated and truncated and info["outcome"] == "timeout"
    assert reward_parts([3], [3], 0, "timeout", 1., config)["progress"] == 0
    assert reward_parts([3], [4], 0, "ongoing", 1., config)["progress"] < 0
    assert info["reward_parts"]["terminal_success"] == 0


@pytest.mark.parametrize("scale", SCALES)
def test_reference_proves_safe_solvable_fixture(scale):
    config = InterceptionConfig(scale=scale)
    row, frames = episode(config, 1000, {"scenario": "fixed"}, "reference", trace=True)
    assert row["outcome"] == "success" and row["minimum_separation"] > config.collision_distance
    assert row["reward_parts"]["capture_event"] == 5 * SCALES[scale][1]
    assert frames[-1]["outcome"] == "success"
