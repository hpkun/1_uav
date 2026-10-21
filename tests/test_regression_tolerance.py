import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('regression_compare_checks',Path(__file__).with_name('test_environment_regression.py'))
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
compare=module.compare


def test_tiny_float_noise_passes_arrays_and_scalars():
    compare(np.array([1.,2.,3.])+1e-14,[1.,2.,3.],path='/states')
    compare({'reward':1.+1e-14,'positions':[[2.+1e-14]]},{'reward':1.,'positions':[[2.]]})


def test_large_float_difference_reports_path_and_error():
    with pytest.raises(AssertionError,match=r'/states.*max_abs_error='):
        compare(np.array([1.,2.+1e-6]),[1.,2.],path='/states')
    with pytest.raises(AssertionError,match='/reward.*abs_error='):
        compare(1.+1e-6,1.,path='/reward')


@pytest.mark.parametrize('a,b',[
    ({'assignments':np.array([0,2])},{'assignments':[0,1]}),
    ({'reward':1.1},{'reward':1.}),
    ({'events':[]},{'events':[{'uav_id':1,'waypoint_id':2}]}),
    ({'terminated':False},{'terminated':True}),
    ({'event_count':2},{'event_count':1}),
    (np.array([1.+1e-14]),[1]),
    ([{'uav_id':1.00000000000001}],[{'uav_id':1}]),
    ({'a':1,'extra':2},{'a':1}),
    (np.array([1]),[True]),
])
def test_discrete_structure_and_real_changes_fail(a,b):
    with pytest.raises(AssertionError): compare(a,b)


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-float('inf')])
def test_nonfinite_never_matches_itself(value):
    with pytest.raises(AssertionError,match='nonfinite'): compare(np.array([value]),[value],path='/states')
    with pytest.raises(AssertionError,match='nonfinite'): compare(value,value,path='/reward')
