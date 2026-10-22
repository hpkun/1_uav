import ast
import gzip
import json
from pathlib import Path
import numpy as np
import pytest
from env import EnvConfig
from env.factory import make_environment
from env.observation import centralized


def compare(actual,expected,path='',*,rtol=1e-12,atol=1e-12):
    """Cross-platform numeric tolerance; structural and discrete values stay exact."""
    import math
    import numbers
    location=path or '/'
    if isinstance(expected,dict):
        assert isinstance(actual,dict),f'{location}: expected dictionary'
        assert actual.keys()==expected.keys(),f'{location}: missing keys={expected.keys()-actual.keys()}, extra keys={actual.keys()-expected.keys()}'
        for k,v in expected.items(): compare(actual[k],v,path+'/'+str(k),rtol=rtol,atol=atol)
    elif isinstance(actual,np.ndarray) or isinstance(expected,np.ndarray):
        a,b=np.asarray(actual),np.asarray(expected)
        assert a.shape==b.shape,f'{location}: shape {a.shape} != {b.shape}'
        if b.dtype.kind=='f':
            assert a.dtype.kind in 'fiu',f'{location}: expected numeric float array'
            assert np.isfinite(a).all() and np.isfinite(b).all(),f'{location}: nonfinite floating array'
            error=float(np.max(np.abs(a.astype(float)-b.astype(float)))) if a.size else 0.
            np.testing.assert_allclose(a,b,rtol=rtol,atol=atol,equal_nan=False,
                err_msg=f'{location}: max_abs_error={error:.17g}, rtol={rtol}, atol={atol}')
        else:
            if b.dtype.kind in 'iu': assert a.dtype.kind in 'iu',f'{location}: expected integer array'
            if b.dtype.kind=='b': assert a.dtype.kind=='b',f'{location}: expected boolean array'
            np.testing.assert_array_equal(a,b,err_msg=f'{location}: exact discrete comparison')
    elif isinstance(expected,(list,tuple)):
        assert isinstance(actual,(list,tuple)),f'{location}: expected sequence'
        assert len(actual)==len(expected),f'{location}: sequence length {len(actual)} != {len(expected)}'
        for i,v in enumerate(expected): compare(actual[i],v,path+f'/{i}',rtol=rtol,atol=atol)
    elif isinstance(expected,(bool,np.bool_)):
        assert isinstance(actual,(bool,np.bool_)) and actual==expected,f'{location}: boolean {actual!r} != {expected!r}'
    elif isinstance(expected,(int,np.integer)):
        assert isinstance(actual,(int,np.integer)) and not isinstance(actual,(bool,np.bool_)) and actual==expected,f'{location}: integer {actual!r} != {expected!r}'
    elif isinstance(expected,(float,np.floating)):
        assert isinstance(actual,numbers.Real) and not isinstance(actual,(bool,np.bool_)),f'{location}: expected real number'
        assert math.isfinite(actual) and math.isfinite(expected),f'{location}: nonfinite floating scalar'
        error=abs(actual-expected)
        assert error<=atol+rtol*abs(expected),f'{location}: actual={actual!r}, expected={expected!r}, abs_error={error:.17g}, rtol={rtol}, atol={atol}'
    else:
        assert actual==expected,f'{location}: {actual!r} != {expected!r}'


@pytest.mark.parametrize('case_index',range(12))
def test_unaffected_historical_transitions_with_explicit_v3_observation_units(case_index):
    with gzip.open(Path(__file__).parent/'fixtures/coop3d_v1_semantics.json.gz','rt',encoding='utf-8') as f: case=json.load(f)[case_index]
    # The frozen fixture is untouched. All its trajectories are away from the
    # retired gamma projection: continue comparing their physical states,
    # events and rewards at the original tolerance. Only the explicitly changed
    # gamma feature unit and the added zero failure counter need schema mapping.
    assert all(s['info']['numerical_counters']['gamma_clips']==0 for s in case['steps'])
    config=case['config'].copy()
    assert config.pop('gamma_limit')==np.deg2rad(85)
    env=make_environment(EnvConfig(**config))

    def observations_in_new_units(observations, state_info):
        result=np.array(observations,dtype=np.float32)
        states=np.asarray(state_info['states'])
        active=np.asarray(state_info['active_mask'],dtype=bool)
        result[active,2]=states[active,4]/np.pi
        return result

    def info_in_new_schema(expected):
        return {**expected,'numerical_counters':{**expected['numerical_counters'],'integration_failures':0}}

    obs,info=env.reset(case['seed'],options=case['options'])
    compare(obs,observations_in_new_units(case['initial_observations'],case['initial']),path='/initial/observations')
    compare({k:info[k] for k in case['initial'] if k in info},info_in_new_schema(case['initial']),path='/initial/info')
    for t,step in enumerate(case['steps']):
        obs,r,term,trunc,info=env.step(step['actions'])
        compare(obs,observations_in_new_units(step['observations'],step['info']),path=f'/steps/{t}/observations')
        compare(r,step['rewards'],path=f'/steps/{t}/rewards')
        compare({k:info[k] for k in step['info'] if k in info},info_in_new_schema(step['info']),path=f'/steps/{t}/info')
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
