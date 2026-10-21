import ast
import gzip
import json
from pathlib import Path
import numpy as np
import pytest
from env import EnvConfig
from env.factory import make_environment
from env.observation import centralized


def compare(actual,expected,path=''):
    if isinstance(expected,dict):
        for k,v in expected.items(): compare(actual[k],v,path+'/'+k)
    elif isinstance(expected,list):
        if expected and isinstance(expected[0],dict):
            assert len(actual)==len(expected)
            for i,v in enumerate(expected): compare(actual[i],v,path+f'/{i}')
        else: np.testing.assert_array_equal(actual,expected,err_msg=path)
    else: assert actual==expected,path


@pytest.mark.parametrize('case_index',range(12))
def test_original_semantic_trajectory(case_index):
    with gzip.open(Path(__file__).parent/'fixtures/coop3d_v1_semantics.json.gz','rt',encoding='utf-8') as f: case=json.load(f)[case_index]
    env=make_environment(EnvConfig(**case['config']))
    obs,info=env.reset(case['seed'],options=case['options'])
    compare(obs,case['initial_observations']); compare(info,case['initial'])
    for step in case['steps']:
        obs,r,term,trunc,info=env.step(step['actions'])
        compare(obs,step['observations']); compare(r,step['rewards']); compare(info,step['info'])
        assert (term,trunc)==(step['terminated'],step['truncated'])


def test_environment_has_no_learning_dependency():
    for p in Path('env').glob('*.py'):
        tree=ast.parse(p.read_text(encoding='utf-8'))
        for n in ast.walk(tree):
            modules=[n.module or ''] if isinstance(n,ast.ImportFrom) else [a.name for a in n.names] if isinstance(n,ast.Import) else []
            assert not any(x.startswith(('algorithm','torch','tools')) for x in modules),p


@pytest.mark.parametrize('scale',['3U3P','3U5P','5U5P'])
def test_waypoint_complete_global_state(scale):
    env=make_environment(EnvConfig(scale=scale))
    obs,info=env.reset(11)
    state=info['centralized_state']; n,m=env.config.num_uavs,env.config.num_waypoints
    assert state.shape==(90,) and obs.shape==(n,12)
    np.testing.assert_array_equal(state[:n*12],obs.flatten())
    assert not state[n*12:60].any()
    points=state[60:].reshape(5,6)
    np.testing.assert_allclose(points[:m,:3],info['waypoints']/[10000,10000,5000],rtol=1e-6)
    np.testing.assert_array_equal(points[:m,3],info['priorities'])
    np.testing.assert_array_equal(points[:m,4],info['completed'])
    assert np.all(points[:m,5]==1) and not points[m:].any()
    if scale=='3U5P':
        unassigned=set(range(m))-set(info['assignments'])
        assert len(unassigned)==2
        for j in unassigned: assert points[j,5]==1
    env.task.completed[0]=True; env.task.active_mask[0]=False
    from env.assignment import assign
    from env.observation import observe
    obs=observe(env.task,assign(env.task),env.config)
    state=centralized(obs,env.task,env.config)
    assert not state[:12].any() and state[64]==1


def test_step_numerical_deltas():
    env=make_environment(EnvConfig())
    env.reset(3)
    prior=env.counters.copy()
    for _ in range(6):
        _,_,_,_,info=env.step(np.ones((3,3)))
        for key,total in info['numerical_counters'].items():
            assert info['numerical_delta'][key]==total-prior[key]
            assert info['numerical_delta'][key]==info['numerical_events'][key].sum()
        prior=info['numerical_counters']
