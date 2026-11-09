import numpy as np
import pytest

from algorithm.interception import episode
from env.interception import InterceptionConfig, InterceptionEnv, SimulationFailure
from env.interception.config import MAX_PURSUERS, OBS_DIM, STATE_DIM, SCALES
from env.interception.events import boundary_contact, windows
from env.interception.motion import advance, command, execute_velocity
from env.interception.reward import reward_parts
from tools.interception_pair_audit import paired


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


def test_preset_evader_modes_share_initial_state_and_follow_distinct_paths():
    straight, bezier = (InterceptionEnv(InterceptionConfig(scale="5v2", evader_mode=mode))
                        for mode in ("straight", "preset_bezier"))
    _, a = straight.reset(42)
    _, b = bezier.reset(42)
    for key in ("pursuers", "evaders", "active_mask", "evader_mask"):
        np.testing.assert_array_equal(a[key], b[key])
    assert a["evader_reference"]["seed"] == b["evader_reference"]["seed"]
    for line, curve in zip(a["evader_reference"]["paths"], b["evader_reference"]["paths"], strict=True):
        assert len(line["points"]) == 2 and len(curve["points"]) == 4
        assert line["points"][0] == curve["points"][0]
        assert line["points"][-1] == curve["points"][-1]
    a = straight.step(actions())[4]
    b = bezier.step(actions())[4]
    assert not np.allclose(a["evaders"], b["evaders"], rtol=0, atol=1e-9)


@pytest.mark.parametrize("mode", ["straight", "preset_bezier"])
def test_preset_evader_contact_is_resolved_on_actual_motion(mode):
    config = InterceptionConfig(scale="1v1", evader_mode=mode)
    env = InterceptionEnv(config)
    custom(env, [[30, 0, 20, 5, 0, 0]], [[31.2, 0, 20, 0, 0, 0]])
    _, rewards, terminated, truncated, info = env.step(actions())
    assert terminated and not truncated and info["outcome"] == "success"
    assert not info["active_mask"].any() and not info["evader_mask"].any()
    assert info["events"][0]["kind"] == "interception_contact"
    assert 0 < info["time"] < config.dt
    assert rewards[0] > 0 and not rewards[1:].any()


@pytest.mark.parametrize("mode", ["constant_velocity", "goal_seeking", "bounded_maneuver", "reactive_evasion",
                                  "straight", "preset_bezier"])
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
        np.testing.assert_array_equal(result[4]["requested_velocities"][5], [-13., 0., 0.])
        np.testing.assert_allclose(target, execute_velocity([-13., 0., 0.], snap["state"].evaders[0, 3:],
                                                           13., 2.5, config))
    elif mode == "goal_seeking":
        ray = np.array([0., 0., 12.]) - snap["state"].evaders[0, :3]
        np.testing.assert_allclose(target, execute_velocity(13 * ray / np.linalg.norm(ray), snap["state"].evaders[0, 3:], 13., 2.5, config))
    elif mode == "reactive_evasion":
        ray = np.array([0., 0., 12.]) - snap["state"].evaders[0, :3]
        assert target @ ray > 0


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


def test_tangency_and_curved_threshold_windows():
    # The chord misses this sphere, but the actual curved trajectory enters.
    curve = lambda t: np.array([2*t-1, 1.2-1.6*t*(1-t), 0.])
    found = windows(curve, 3.2, 1., 0., 1., 1e-7)
    assert len(found) == 1 and found[0][0] < .5 < found[0][1]
    tangent = windows(lambda t: np.array([2*t-1, 1., 0.]), 0., 1., 0., 1., 1e-7)
    assert tangent == [(.5, .5)]
    assert not windows(lambda t: np.array([2*t-1, 1.001, 0.]), 0., 1., 0., 1., 1e-7)


def straight(scale="1v1", **changes):
    return InterceptionConfig(**(dict(scale=scale, evader_mode="constant_velocity",
        evader_speed_fraction=6/13, constant_direction=(1, 0, 0)) | changes))


@pytest.mark.parametrize(("offset", "contact"), [(.99, True), (1., True), (1.01, False)])
def test_swept_contact_and_tangency(offset, contact):
    config = straight(constant_direction=(-1, 0, 0))
    env = InterceptionEnv(config)
    custom(env, [[40,20,15,12,0,0]], [[40.9,20+offset,15,-6,0,0]])
    obs, reward, term, trunc, info = env.step(actions(12, config))
    assert (term, trunc) == (contact, False)
    if not contact:
        assert info["outcome"] == "ongoing" and not info["events"]
        return
    expected = (.9-np.sqrt(1-offset**2))/18
    event, = info["events"]
    assert info["time"] == pytest.approx(expected, abs=1e-7)
    assert event["kind"] == "interception_contact"
    assert event["relative_speed"] == pytest.approx(18.)
    assert event["center_distance"] == pytest.approx(1., abs=1e-6)
    assert not info["active_mask"].any() and not info["evader_mask"].any()
    assert not obs.any() and not info["centralized_state"][:49].any()
    assert reward[0] == info["team_reward"] and not reward[1:].any()
    assert info["reward_parts"]["capture_event"] == 5.


@pytest.mark.parametrize(("kind", "expected"), [("PP", "friendly_collision"), ("EE", "evader_collision")])
def test_same_side_contact_is_failure(kind, expected):
    config = straight("3v1" if kind == "PP" else "5v2", dt=1.)
    env = InterceptionEnv(config)
    if kind == "PP":
        p = [[-2,30,15,8,0,0], [2,30,15,-8,0,0], [0,40,15,0,0,0]]
        e = [[50,0,15,6,0,0]]
        u = [8,-8,0,0,0]
        contact = 3/16
    else:
        p = [[0,-20-4*i,15,0,0,0] for i in range(5)]
        e = [[-2,30,15,6,0,0], [2,30,15,-6,0,0]]
        u = 0
        # The reversing Evader executes -6 + tau*a = 4 m/s.
        contact = None
    custom(env, p, e)
    _, _, term, trunc, info = env.step(actions(u, config))
    assert term and not trunc and info["outcome"] == expected
    if contact is None:
        t = info["time"]
        gap = lambda t: advance(e[1], [4,0,0], 2.5, t)[0]-(-2+6*t)
        assert gap(t) == pytest.approx(1., abs=1e-6) and gap(t-1e-5) > 1.
    else:
        assert info["time"] == pytest.approx(contact, abs=1e-7)
    assert info["events"][-1]["kind"] == expected
    assert info["reward_parts"]["capture_event"] == 0
    assert info["reward_parts"]["collision_safety" if kind == "PP" else "terminal_failure"] == -10
    assert info["reward_parts"]["time_penalty"] == pytest.approx(-config.step_cost*config.max_steps)
    if kind == "PP":
        assert info["events"][0]["kind"] == "friendly_warning"
        assert info["events"][0]["time"] == pytest.approx(1/16)


@pytest.mark.parametrize(("when", "outcome"), [(.15,"success"), (.2,"breach"), (.25,"breach")])
def test_breach_contact_order(when, outcome):
    config = straight(dt=.4)
    env = InterceptionEnv(config)
    ex = -13.2
    custom(env, [[ex-1-2*when,0,12,8,0,0]], [[ex,0,12,6,0,0]])
    _, _, term, trunc, info = env.step(actions(8, config))
    assert term and not trunc and info["outcome"] == outcome
    assert info["time"] == pytest.approx(min(when,.2), abs=1e-7)
    assert info["reward_parts"]["capture_event"] == (5 if outcome == "success" else 0)


def test_multitarget_retirement_same_step_and_next_step():
    config = straight("5v2", dt=.5)
    env = InterceptionEnv(config)
    p = [[40,20,15,8,0,0], [40,40,15,8,0,0], [40,22,15,8,0,0],
         [0,-40,15,0,0,0], [0,-50,15,0,0,0]]
    e = [[41.4,20,15,6,0,0], [42.4,40,15,6,0,0]]
    custom(env, p, e)
    u = actions([8,8,8,0,0], config)
    u[2] = [.5+8/24, .5-2/24, .5]
    obs, reward, term, _, info = env.step(u)
    assert not term and not info["active_mask"][0] and not info["evader_mask"][0]
    assert not obs[0].any() and reward[0] == reward[1]
    first = next(x for x in info["events"] if x["kind"] == "interception_contact")
    assert first["time"] == pytest.approx(.2, abs=1e-7)
    frozen = info["pursuers"][0].copy(), info["evaders"][0].copy()
    _, reward, term, _, info = env.step(u)
    assert term and info["outcome"] == "success" and info["time"] == pytest.approx(.7, abs=1e-7)
    assert reward[0] == 0 and info["reward_parts"]["capture_event"] == 5
    np.testing.assert_array_equal(frozen, (info["pursuers"][0], info["evaders"][0]))


def test_simultaneous_competing_interceptions_pick_stable_pursuer():
    config = straight("3v1", dt=.5, constant_direction=(0,0,1), evader_speed_fraction=.01)
    env = InterceptionEnv(config)
    custom(env, [[40,28,15,0,8,0], [40,32,15,0,-8,0], [40,40,20,0,0,0]], [[40,30,15,0,0,.13]])
    u = actions(0, config)
    u[0,1], u[1,1] = .5+8/24, .5-8/24
    _, _, _, _, info = env.step(u)
    contact, = [event for event in info["events"] if event["kind"] == "interception_contact"]
    assert contact["entities"] == [0,5]
    assert info["active_mask"].tolist() == [False,True,True,False,False]


@pytest.mark.parametrize(("who", "outcome"), [("p", "pursuer_boundary_violation"), ("e", "evader_escape")])
def test_full_envelope_first_exit(who, outcome):
    config = straight(dt=3.)
    env = InterceptionEnv(config)
    p = [159,20,15,4,0,0] if who == "p" else [140,20,15,0,0,0]
    e = [159,-20,15,6,0,0] if who == "e" else [140,-20,15,6,0,0]
    if who == "p":
        assert advance(p, [-6,0,0], 2.5, config.dt)[0] < 159.5
    custom(env, [p], [e])
    _, reward, term, trunc, info = env.step(actions(-12, config))
    assert term and not trunc and info["outcome"] == outcome
    event, = info["events"]
    assert event["time"] == info["time"] and event["positions"][0][0] == pytest.approx(159.5, abs=1e-6)
    assert event["velocities"][0][0] > 0
    parts = info["reward_parts"]
    assert parts["terminal_success"] == parts["capture_event"] == 0
    assert parts["collision_safety" if who == "p" else "terminal_failure"] == -10
    assert parts["progress"] <= 0 and reward[0] == pytest.approx(sum(parts.values()))


def test_boundary_tangency_without_exit():
    state = np.array([79.,20.,15.,4.,0.,0.])
    request = np.array([-8.,0.,0.])
    at = lambda t: advance(state, request, 1.5, t)
    top = at(1.5*np.log1p(.5))[0]
    assert boundary_contact(state, request, 1.5, at, [-80,-80,3], [top,80,40], 3., 1e-7) is None


def test_paired_deployment():
    s1, s2 = (InterceptionConfig(task=task) for task in ("S1", "S2"))
    _, a = InterceptionEnv(s1).reset(1001)
    _, b = InterceptionEnv(s2).reset(1001)
    np.testing.assert_array_equal(a["pursuers"], b["pursuers"])
    np.testing.assert_array_equal(a["evaders"], b["evaders"])
    assert len(set(b["pursuers"][:3, 2])) == 3
    rows = paired(s1, 1001, "random")
    for count, row in enumerate(rows, 1):
        assert row["pair_id"] == rows[2]["pair_id"]
        assert row["initial"]["evaders"] == rows[2]["initial"]["evaders"]
        assert row["initial"]["pursuers"][:count] == rows[2]["initial"]["pursuers"][:count]
