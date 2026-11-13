import json
import torch
import pytest

import algorithm.ideal_interception as cli
from algorithm.common.interception_checkpoint import read
from algorithm.mappo.interception import TrainConfig
from env.ideal_interception import InterceptionConfig
from env.ideal_interception.config import OBS_DIM, STATE_DIM


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required by repository policy")
def test_bilateral_mappo_checkpoint_resume_evaluate_and_masks(tmp_path):
    config = InterceptionConfig(max_steps=4)
    training = TrainConfig(workers=1,horizon=2,epochs=1)
    model = cli.InterceptionMAPPO(training)
    assert model.actor[0].in_features==OBS_DIM
    assert model.critic[0].in_features==STATE_DIM
    assert model.actor[-1].out_features==6 and model.critic[-1].out_features==5
    action,logp = model.act(torch.zeros((5,OBS_DIM)).numpy(),
                            [True,False,False,False,False],deterministic=True)
    assert torch.all(action[1:]==.5) and torch.all(logp[1:]==0)
    folder = tmp_path/"run"
    checkpoint = cli.fit(folder,config,training,total_steps=4,until_steps=2,
                         eval_interval=1,evaluation_seeds=[2200,2201])
    first = read(checkpoint,protocol=cli.PROTOCOL)
    assert first["protocol"]==cli.PROTOCOL and first["steps"]==2
    assert json.loads((folder/"update_000001.json").read_text())["valid_samples"]==2
    with pytest.raises(ValueError,match="incompatible checkpoint protocol"):
        read(checkpoint)
    with pytest.raises(ValueError,match="incompatible checkpoint protocol"):
        read(checkpoint,protocol=cli.PROTOCOL|{"version":"uav-ideal-interception-v1.0.0"})
    cli.fit(folder,resume=True)
    last = read(checkpoint,protocol=cli.PROTOCOL)
    assert last["steps"]==4 and last["updates"]==2
    assert len(last["evaluations"])==2 and sum(
        "result" in entry for entry in last["evaluations"])==1
    output = tmp_path/"eval"
    cli.main(["evaluate",str(checkpoint),"--out",str(output),"--eval-episodes","2"])
    result = json.loads((output/"evaluation.json").read_text())
    assert result["protocol"]==cli.PROTOCOL
    assert result["evaluation_protocol"]["kind"]=="fixed_case"
    assert result["evaluation_protocol"]["unique_scenarios"]==1
    assert result["summary"]["episodes"]==2 and result["first_trace"][0]["time"]==0.
    assert result["episodes"][0]["intruder_paths"][0]["polyline_sha256"]
    assert result["model_sha256"]==cli.model_digest(last["actor"],last["critic"])


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required by repository policy")
def test_bilateral_recovery_from_missing_committed_record(tmp_path,monkeypatch):
    config = InterceptionConfig(max_steps=3)
    training = TrainConfig(workers=1,horizon=2,epochs=1)
    folder = tmp_path/"run"
    original = cli.atomic_json
    with monkeypatch.context() as patch:
        def interrupt(path,value,*args,**kwargs):
            if path.name=="update_000001.json":
                raise OSError("injected record failure")
            return original(path,value,*args,**kwargs)
        patch.setattr(cli,"atomic_json",interrupt)
        with pytest.raises(OSError,match="injected"):
            cli.fit(folder,config,training,total_steps=2,eval_interval=1,
                    evaluation_seeds=[2200])
    cli.fit(folder,resume=True)
    payload = read(folder/"latest.pt",protocol=cli.PROTOCOL)
    assert payload["updates"]==1 and payload["steps"]==2
    assert json.loads((folder/"update_000001.json").read_text())["update"]==1
    assert (folder/"evaluation_000001.json").is_file()
    assert len((folder/"training_metrics.jsonl").read_text().splitlines())==1


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required by repository policy")
def test_bilateral_resume_preserves_training_and_evaluation_values(tmp_path):
    config = InterceptionConfig(max_steps=3)
    training = TrainConfig(workers=1,horizon=2,epochs=1)
    full,split = tmp_path/"full",tmp_path/"split"
    cli.fit(full,config,training,total_steps=4,eval_interval=1,
            evaluation_seeds=[2200])
    cli.fit(split,config,training,total_steps=4,until_steps=2,
            eval_interval=1,evaluation_seeds=[2200])
    cli.fit(split,resume=True)
    first = read(full/"latest.pt",protocol=cli.PROTOCOL)
    second = read(split/"latest.pt",protocol=cli.PROTOCOL)
    assert cli.model_digest(first["actor"],first["critic"]) == cli.model_digest(
        second["actor"],second["critic"])
    assert first["evaluations"][-1]["result"]["summary"] == second["evaluations"][-1]["result"]["summary"]
    for update in (1,2):
        a = json.loads((full/f"update_{update:06}.json").read_text())
        b = json.loads((split/f"update_{update:06}.json").read_text())
        assert {key:value for key,value in a.items() if key not in ("sampling_speed","previous")} == {
            key:value for key,value in b.items() if key not in ("sampling_speed","previous")}
