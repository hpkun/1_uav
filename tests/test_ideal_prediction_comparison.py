import numpy as np
import pytest

from env.ideal_interception import InterceptionConfig
from prediction.zuo_linear import predict_zuo_two_steps
from prediction.comparison import compare,read_observations
from tools.ideal_blue_observations import generate,write_observations
from tools.ideal_prediction_compare import run


def test_zuo_linear_formula_and_input_checks():
    p=np.array([[1.,2.,3.],[3.,4.,5.]])
    np.testing.assert_allclose(predict_zuo_two_steps(p,.1),[[5.,6.,7.],[7.,8.,9.]])
    with pytest.raises(ValueError):
        predict_zuo_two_steps(p[:1],.1)
    with pytest.raises(ValueError):
        predict_zuo_two_steps(p,0)


@pytest.mark.parametrize("scale,mode,initialization",[
    ("1v1","straight","fixed"),("1v1","bezier","fixed"),
    ("5v2","bezier","random"),("5v2","straight","random")])
def test_methods_are_paired_on_complete_blue_paths(tmp_path,scale,mode,initialization):
    data=generate(InterceptionConfig(scale=scale,intruder_mode=mode,
                                     initialization=initialization),17)
    path=tmp_path/"observations.json.gz"
    write_observations(path,data)
    first=compare(read_observations(path))
    assert first==compare(read_observations(path))
    for target,summary in first["targets"].items():
        for horizon in (1,2):
            z=summary["methods"]["zuo_linear"][f"ValidSamples_{horizon}"]
            t=summary["methods"]["tan_quadratic"][f"ValidSamples_{horizon}"]
            assert z==t==summary["number_of_paired_predictions"][str(horizon)]>0
        j=int(target[1:])
        keyed={}
        for row in first["predictions"]:
            if row["target_index"]!=j:
                continue
            keyed.setdefault((row["observation_time"],row["horizon_steps"]),[]).append(row)
        assert all(len(pair)==2 and pair[0]["valid"]==pair[1]["valid"]
                   and pair[0]["predicted_time"]==pair[1]["predicted_time"]
                   and all(pair[0][f"actual_{axis}"]==pair[1][f"actual_{axis}"]
                           for axis in "xyz") for pair in keyed.values())
        assert all(row["predicted_time"]<=summary["last_regular_sample_time"] + 2*data["dt"]
                   for row in first["predictions"] if row["target_index"]==j)
        assert all(row["predicted_time"]<=summary["last_regular_sample_time"]+1e-9
                   for row in first["predictions"] if row["target_index"]==j and row["valid"])
        assert all(row["actual_x"] is None and row["error_m"] is None
                   for row in first["predictions"] if row["target_index"]==j and not row["valid"])
        if mode=="straight":
            assert summary["methods"]["zuo_linear"]["MaxError_2"]<1e-9
            assert summary["methods"]["tan_quadratic"]["MaxError_2"]<1e-9
    exported=run(path,tmp_path/"report",plot=True)
    assert exported==first
    for name in ("predictions.csv","metrics.json","comparison.csv","comparison.png","error_curve.png"):
        assert (tmp_path/"report"/name).is_file()
