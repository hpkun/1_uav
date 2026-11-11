import json
import torch
import pytest
import algorithm.ideal_kinematic as ideal

from algorithm.common.interception_checkpoint import read
from algorithm.ideal_kinematic import PROTOCOL, IdealMAPPO, evaluate, fit, main
from algorithm.mappo.interception import TrainConfig
from env.ideal_kinematic import IdealConfig
from env.ideal_kinematic.config import OBS_DIM, STATE_DIM


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required by repository policy")
def test_ideal_mappo_checkpoint_resume_and_protocol_isolation(tmp_path, capsys):
    config = IdealConfig(task="IK0", max_steps=4)
    training = TrainConfig(workers=1, horizon=2, epochs=1)
    model = IdealMAPPO(training)
    assert model.actor[0].in_features == OBS_DIM
    assert model.critic[0].in_features == STATE_DIM
    assert model.actor[-1].out_features == 6
    assert model.critic[-1].out_features == 5
    folder = tmp_path/"ideal"
    checkpoint = fit(folder, config, training, total_steps=4, until_steps=2,
                     eval_interval=1, evaluation_seeds=[2300])
    first = read(checkpoint, protocol=PROTOCOL)
    assert first["steps"] == 2 and first["protocol"] == PROTOCOL
    with pytest.raises(ValueError, match="incompatible checkpoint protocol"):
        read(checkpoint)
    with pytest.raises(ValueError, match="incompatible checkpoint protocol"):
        read(checkpoint, protocol=PROTOCOL | {"version": "uav-ideal-kinematic-v1.0.0"})
    assert "[pause]" in capsys.readouterr().out
    fit(folder, resume=True)
    final = read(checkpoint, protocol=PROTOCOL)
    assert final["steps"] == 4 and final["updates"] == 2
    assert "[resume]" in capsys.readouterr().out
    output = tmp_path/"evaluation"
    main(["evaluate", str(checkpoint), "--out", str(output),
          "--eval-episodes", "1"])
    result = json.loads((output/"evaluation.json").read_text())
    assert result["protocol"] == PROTOCOL
    assert result["first_trace"][0]["time"] == 0.
    assert result["summary"]["episodes"] == 1
    assert len(result["first_trace"]) >= 2
    restored = IdealMAPPO(training)
    restored.actor.load_state_dict(final["actor"])
    restored.critic.load_state_dict(final["critic"])
    assert evaluate(restored, config, [2300], deterministic=False) == evaluate(
        restored, config, [2300], deterministic=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required by repository policy")
@pytest.mark.parametrize("stage", ["checkpoint", "update", "evaluation"])
def test_ideal_recovery_at_commit_stages(tmp_path, monkeypatch, stage):
    config = IdealConfig(task="IK0", max_steps=3)
    training = TrainConfig(workers=1, horizon=2, epochs=1)
    baseline = tmp_path/"baseline"
    fit(baseline, config, training, total_steps=2, eval_interval=1, evaluation_seeds=[2300])
    target = tmp_path/"interrupted"
    original_save, original_json = ideal.save, ideal.atomic_json
    def fail_save(path, model, *args, **kwargs):
        if model.updates == 1:
            raise OSError("injected checkpoint failure")
        return original_save(path, model, *args, **kwargs)
    def fail_json(path, value, *args, **kwargs):
        if path.name == ("update_000001.json" if stage == "update" else "evaluation_000001.json"):
            raise OSError("injected derived record failure")
        return original_json(path, value, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(ideal, "save" if stage == "checkpoint" else "atomic_json",
                      fail_save if stage == "checkpoint" else fail_json)
        with pytest.raises(OSError, match="injected"):
            fit(target, config, training, total_steps=2, eval_interval=1,
                evaluation_seeds=[2300])
    fit(target, resume=True)
    expected, actual = (read(folder/"latest.pt", protocol=PROTOCOL) for folder in (baseline, target))
    assert actual["updates"] == expected["updates"] == 1
    assert actual["steps"] == expected["steps"] == 2
    for key in ("actor", "critic"):
        for name, value in expected[key].items():
            assert torch.equal(value, actual[key][name])
    assert torch.equal(expected["torch_rng"], actual["torch_rng"])
    assert len((target/"training_metrics.jsonl").read_text().splitlines()) == 1
    assert json.loads((target/"update_000001.json").read_text())["update"] == 1
    assert json.loads((target/"evaluation_000001.json").read_text()) == json.loads(
        (baseline/"evaluation_000001.json").read_text())
