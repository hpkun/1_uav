import itertools
import numpy as np
import pytest
from coop3d import Cooperative3DEnv, EnvConfig
from coop3d.assignment import assign, hungarian
from coop3d.models import TaskState
from coop3d.observation import centralized
from coop3d.reward import compute_rewards

LEVEL = np.tile([.5, 1 / 3, 1 / 8], (3, 1))


def fixed_env(**kwargs):
    return Cooperative3DEnv(EnvConfig(random_scenario=False, **kwargs))


def setup_case(points, states=None, **kwargs):
    env = fixed_env(**kwargs)
    if states is None:
        states = [[0, -300, 3000, 200, 0, 0], [0, 0, 3000, 200, 0, 0], [0, 300, 3000, 200, 0, 0]]
    env.reset(123, options=dict(states=states, waypoints=points))
    return env


@pytest.mark.parametrize('scale,n,m', [('3U3P', 3, 3), ('5U5P', 5, 5), ('3U5P', 3, 5)])
def test_scale_episode(scale, n, m):
    env = fixed_env(scale=scale)
    obs, info = env.reset(5)
    assert obs.shape == (n, 12)
    assert info['centralized_state'].shape == (60,)
    assert len(info['waypoints']) == m
    steps = 0
    while True:
        obs, reward, terminated, truncated, info = env.step(np.tile([.5, 1 / 3, 1 / 8], (n, 1)))
        steps += 1
        assert np.isfinite(obs).all() and np.isfinite(reward).all()
        if terminated or truncated:
            break
    assert steps <= 250 and info['episode_length'] == steps


@pytest.mark.parametrize('scale', ['3U3P', '5U5P', '3U5P'])
def test_reproducible_random(scale):
    a, b = Cooperative3DEnv(EnvConfig(scale=scale)), Cooperative3DEnv(EnvConfig(scale=scale))
    oa, ia = a.reset(71)
    ob, ib = b.reset(71)
    np.testing.assert_array_equal(oa, ob)
    np.testing.assert_array_equal(ia['states'], ib['states'])
    np.testing.assert_array_equal(ia['waypoints'], ib['waypoints'])
    distances = ia['pairwise_distances'] + np.eye(len(oa)) * 1e9
    assert distances.min() >= 200
    assert np.all((ia['states'][:, 2] >= 2500) & (ia['states'][:, 2] <= 3500))
    assert not np.array_equal(a.reset(72)[0], ob)


def test_single_waypoint_and_deduplication():
    env = setup_case([[50, 0, 3000], [9000, 0, 3000], [11000, 0, 3000]])
    _, rewards, _, _, info = env.step(LEVEL)
    assert len(info['events']) == 1
    assert info['events'][0]['uav_id'] == 0  # boundary at 300 m, lowest ID.
    assert sum(info['reward_components']['completion']) == 10
    assert info['completed_waypoints'] == 1
    _, _, _, _, second = env.step(LEVEL)
    assert not second['events']
    assert not second['reward_components']['completion'].any()
    assert 0 not in second['assignments']


def test_arrival_is_three_dimensional():
    env = setup_case([[50, 0, 3400], [9000, 0, 3000], [11000, 0, 3000]])
    assert not env.step(LEVEL)[4]['events']


def test_all_completed_terminates_once():
    env = setup_case([[50, -300, 3000], [50, 0, 3000], [50, 300, 3000]])
    _, _, terminated, truncated, info = env.step(LEVEL)
    assert terminated and not truncated and info['success']
    assert info['completed_waypoints'] == 3
    np.testing.assert_array_equal(info['reward_components']['terminal'], [20, 20, 20])
    with pytest.raises(RuntimeError):
        env.step(LEVEL)


def test_timeout():
    env = fixed_env(max_steps=1)
    env.reset(1)
    _, _, terminated, truncated, info = env.step(LEVEL)
    assert truncated and not terminated and info['incomplete'] and not info['success']
    assert not info['reward_components']['terminal'].any()


@pytest.mark.parametrize('n,m', [(1, 5), (3, 5), (5, 3), (5, 5), (0, 3), (3, 0)])
def test_hungarian_optimal_bruteforce(n, m):
    cost = np.random.default_rng(6).normal(size=(n, m))
    pairs = hungarian(cost)
    assert len(pairs) == min(n, m)
    actual = sum(cost[i, j] for i, j in pairs)
    if n <= m:
        expected = min(sum(cost[i, j] for i, j in enumerate(js)) for js in itertools.permutations(range(m), n))
    else:
        expected = min(sum(cost[i, j] for j, i in enumerate(ids)) for ids in itertools.permutations(range(n), m))
    assert actual == pytest.approx(expected)


def test_hungarian_known_and_ties():
    assert hungarian([[4, 1, 3], [2, 0, 5], [3, 2, 2]]) == [(0, 1), (1, 0), (2, 2)]
    assert hungarian(np.ones((3, 2))) == [(0, 0), (1, 1)]


@pytest.mark.parametrize('mode', ['hungarian', 'nearest'])
@pytest.mark.parametrize('scale', ['3U5P', '5U5P'])
def test_assignment_no_duplicates_idle_and_inactive(mode, scale):
    env = fixed_env(scale=scale, assignment_mode=mode)
    env.reset(1)
    first = assign(env.task, mode)
    assert len(np.unique(first[first >= 0])) == min(env.config.num_uavs, env.config.num_waypoints)
    env.task.completed[1:] = True
    env.task.active_mask[0] = False
    second = assign(env.task, mode)
    assert second[0] == -1
    assert (second >= 0).sum() == 1


def test_cost_uses_heading_and_priority():
    t = TaskState(np.array([[0, 0, 3000, 200, 0, 0.]]), np.array([[1000, 0, 3000], [0, 1000, 3000]]), np.array([0., 1.]), np.zeros(2, bool), np.ones(1, bool))
    assert assign(t)[0] == 0  # equal cost broken by ID
    t.priorities[1] = 0
    assert assign(t)[0] == 0
    t.waypoints[1] = [500, 0, 3000]
    assert assign(t)[0] == 1


def test_observation_contract_and_padding():
    env = setup_case([[1000, -300, 3100], [2000, 0, 3200], [3000, 300, 3300]])
    obs = env.reset(1, options={'states': env.task.states, 'waypoints': env.task.waypoints})[0]
    np.testing.assert_allclose(obs[0, :5], [1, 1 / 3, 0, 0, .6])
    np.testing.assert_allclose(obs[0, 5:8], [.1, 0, .01])
    assert obs[0, 8] == pytest.approx(np.sqrt(1000**2 + 100**2) / 10000)
    assert obs[0, 9] == 0 and obs[0, 10] == 1 and obs[0, 11] == pytest.approx(.03)
    s = centralized(obs)
    np.testing.assert_array_equal(s[:36], obs.flatten())
    assert not s[36:].any()
    assert centralized(np.stack([obs, obs])).shape == (2, 60)


def test_inactive_and_idle_observation():
    from coop3d.observation import observe
    env = fixed_env()
    env.reset(1)
    env.task.active_mask[0] = False
    env.task.completed[1:] = True
    assignments = assign(env.task)
    obs = observe(env.task, assignments, env.config)
    assert not obs[0].any()
    idle = np.flatnonzero((assignments < 0) & env.task.active_mask)[0]
    assert not obs[idle, 5:11].any()


def test_failure_transition_semantics_and_frozen_state():
    env = fixed_env(failure_events={1: [1]})
    env.reset(1)
    _, _, _, _, info = env.step(LEVEL)
    assert info['current_active_mask'][1] and not info['next_active_mask'][1]
    assert info['reward_components']['efficiency'][1] == -.01
    state = env.task.states[1].copy()
    actions = LEVEL.copy()
    actions[1] = np.nan  # ignored for inactive UAV
    obs, reward, _, _, info = env.step(actions)
    np.testing.assert_array_equal(state, env.task.states[1])
    assert reward[1] == 0 and not obs[1].any() and info['assignments'][1] == -1


def test_all_inactive_terminates_without_terminal_bonus():
    env = fixed_env(failure_events={1: [0, 1, 2]})
    env.reset(1)
    _, _, terminated, truncated, info = env.step(LEVEL)
    assert terminated and not truncated and info['all_inactive'] and not info['success']
    assert not info['reward_components']['terminal'].any()


def test_idle_uav_uses_trim_cruise():
    env = fixed_env()
    env.reset(1)
    env.task.completed[1:] = True
    env.assignments = assign(env.task)
    idle = np.flatnonzero(env.assignments < 0)
    old = env.task.states.copy()
    _, _, _, _, info = env.step(np.ones((3, 3)))
    np.testing.assert_allclose(env.task.states[idle, 3:], old[idle, 3:])
    np.testing.assert_allclose(info['physical_actions'][idle], np.tile([0, 0, 1], (len(idle), 1)))


@pytest.mark.parametrize('direction,sign', [(0, 1), (np.pi, -1)])
def test_progress_sign_and_reward_sum(direction, sign):
    env = fixed_env()
    states = np.array([[0, -300, 3000, 200, 0, direction], [0, 0, 3000, 200, 0, direction], [0, 300, 3000, 200, 0, direction]])
    env.reset(1, options={'states': states})
    _, reward, _, _, info = env.step(LEVEL)
    assert np.all(info['reward_components']['progress'] * sign > 0)
    np.testing.assert_allclose(sum(info['reward_components'].values()), reward, rtol=1e-6)
    reconstructed, components = compute_rewards(info['previous_states'], info['states'], info['waypoints'], info['transition_assignments'], info['current_active_mask'], info['next_active_mask'], info['events'], info['episode_length'], info['success'], env.config)
    np.testing.assert_allclose(reconstructed, reward, rtol=1e-6)


def test_separation_and_single_valid():
    c = EnvConfig()
    states = np.array([[0, 0, 3000, 200, 0, 0], [50, 0, 3000, 200, 0, 0], [200, 0, 3000, 200, 0, 0]])
    for mask, expected in [([1, 1, 1], [-.25, -.25, 0]), ([1, 0, 0], [0, 0, 0])]:
        total, p = compute_rewards(states, states, np.zeros((3, 3)), [-1]*3, np.array(mask), np.array(mask), [], 25, True, c)
        np.testing.assert_allclose(p['separation'], expected)
        np.testing.assert_allclose(p['efficiency'], -.01 * np.array(mask))
        np.testing.assert_allclose(p['terminal'], 19 * np.array(mask))
        np.testing.assert_allclose(sum(p.values()), total)


@pytest.mark.parametrize('kwargs', [{'scale': 'bad'}, {'dt': 0}, {'v_min': 401}, {'gamma_limit': np.pi / 2}, {'max_steps': 251}, {'failure_events': {1: [3]}}, {'scenario': {'initial_speed': [1, 5]}}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        EnvConfig(**kwargs)


def test_invalid_actions_and_options():
    env = fixed_env()
    env.reset(1)
    for actions in (np.zeros((2, 3)), LEVEL * np.nan, LEVEL + 2):
        with pytest.raises(ValueError):
            env.step(actions)
    with pytest.raises(ValueError):
        env.reset(1, options={'states': [[0]*6]})
