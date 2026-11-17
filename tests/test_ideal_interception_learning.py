import json
import zipfile
import numpy as np
import torch
import pytest

import algorithm.ideal_interception as cli
from algorithm.common.interception_checkpoint import read
from algorithm.mappo.interception import TrainConfig
from env.ideal_interception.config import STATE_DIM,DEFENDER_SLOTS,formal_6v6_config
from env.ideal_interception.learning import PG_OBS_DIM,PredictionGuidedInterceptionEnv


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required by repository policy")
@pytest.mark.parametrize("deterministic",[True,False])
def test_pg_mappo_act_with_no_active_defenders(deterministic,monkeypatch):
    model=cli.InterceptionMAPPO(TrainConfig(workers=1,horizon=1))
    def unexpected_distribution(_):
        raise AssertionError("empty mask must not construct a Beta distribution")
    monkeypatch.setattr(model,"distribution",unexpected_distribution)
    action,logp=model.act(np.zeros((DEFENDER_SLOTS,PG_OBS_DIM),dtype=np.float32),
                          np.zeros(DEFENDER_SLOTS,dtype=bool),deterministic=deterministic)
    assert action.shape==(DEFENDER_SLOTS,3) and logp.shape==(DEFENDER_SLOTS,)
    assert torch.all(action==.5) and torch.all(logp==0)


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required by repository policy")
def test_pg_mappo_act_with_active_defender_keeps_beta_outputs():
    model=cli.InterceptionMAPPO(TrainConfig(workers=1,horizon=1))
    observations=np.zeros((DEFENDER_SLOTS,PG_OBS_DIM),dtype=np.float32)
    mask=np.zeros(DEFENDER_SLOTS,dtype=bool);mask[2]=True
    with torch.no_grad():
        distribution=model.distribution(model.tensor(observations[mask]).float())
        expected=distribution.mean.clamp(1e-6,1-1e-6)[0]
        expected_logp=distribution.log_prob(expected).sum()
    action,logp=model.act(observations,mask,deterministic=True)
    torch.testing.assert_close(action[2],expected,rtol=0,atol=0)
    torch.testing.assert_close(logp[2],expected_logp,rtol=0,atol=0)
    assert torch.all(action[~model.tensor(mask)]==.5)
    assert torch.all(logp[~model.tensor(mask)]==0)


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required by repository policy")
def test_pg_evaluation_continues_after_all_defenders_exit():
    original=formal_6v6_config()
    starts=tuple((-299.5,float(y),60.) for y in (-150,-90,-30,30,90,150))
    config=formal_6v6_config(dt=1.,max_steps=30,intruder_mode="straight",
        random_range=0.,intruder_start_jitter_m=0.,
        defender_starts=starts+original.defender_starts[6:],
        defender_directions=((-1.,0.,0.),)*DEFENDER_SLOTS)
    env=PredictionGuidedInterceptionEnv(config)
    env.reset(17)
    observation,_,terminated,truncated,info=env.step(np.full((DEFENDER_SLOTS,3),.5))
    assert not terminated and not truncated and info["outcome"]=="ongoing"
    assert not info["defender_mask"].any() and info["intruder_mask"].any()
    model=cli.InterceptionMAPPO(TrainConfig(workers=1,horizon=1))
    while not (terminated or truncated):
        action,logp=model.act(observation,info["defender_mask"],deterministic=True)
        assert torch.all(action==.5) and torch.all(logp==0)
        observation,_,terminated,truncated,info=env.step(action.cpu().numpy())
    assert info["outcome"]=="mixed_resolution"
    assert not info["intruder_mask"].any()
    assert info["step"]<=config.max_steps


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required by repository policy")
def test_bilateral_mappo_checkpoint_resume_evaluate_and_masks(tmp_path):
    config = formal_6v6_config(max_steps=4,intruder_mode="straight")
    training = TrainConfig(workers=1,horizon=2,epochs=1)
    model = cli.InterceptionMAPPO(training)
    assert model.actor[0].in_features==PG_OBS_DIM==48
    assert model.critic[0].in_features==STATE_DIM
    assert model.actor[-1].out_features==6 and model.critic[-1].out_features==DEFENDER_SLOTS
    action,logp = model.act(torch.zeros((DEFENDER_SLOTS,PG_OBS_DIM)).numpy(),
                            [True]+[False]*(DEFENDER_SLOTS-1),deterministic=True)
    assert torch.all(action[1:]==.5) and torch.all(logp[1:]==0)
    folder = tmp_path/"run"
    checkpoint = cli.fit(folder,config,training,total_steps=4,until_steps=2,
                         eval_interval=1,evaluation_seeds=[2200,2201])
    first = read(checkpoint,protocol=cli.PROTOCOL)
    assert first["protocol"]==cli.PROTOCOL and first["steps"]==2
    assert json.loads((folder/"update_000001.json").read_text())["valid_samples"]==12
    with pytest.raises(ValueError,match="incompatible checkpoint protocol"):
        read(checkpoint)
    former=tmp_path/"old_mappo_header.pt"
    with zipfile.ZipFile(former,"w") as archive:
        archive.writestr("archive/interception_protocol.json",json.dumps(
            cli.PROTOCOL|{"algorithm":"mappo-beta","observation":265}))
    with pytest.raises(ValueError,match="incompatible checkpoint protocol"):
        read(former,protocol=cli.PROTOCOL)
    with pytest.raises(ValueError,match="incompatible checkpoint protocol"):
        read(checkpoint,protocol=cli.PROTOCOL|{"version":"uav-ideal-interception-v1.0.0"})
    with pytest.raises(ValueError,match="incompatible checkpoint protocol"):
        read(checkpoint,protocol=cli.PROTOCOL|{"version":"uav-ideal-interception-v2.0.0"})
    with pytest.raises(ValueError,match="incompatible checkpoint protocol"):
        read(checkpoint,protocol=cli.PROTOCOL|{"version":"uav-ideal-interception-v3.0.0"})
    with pytest.raises(ValueError,match="incompatible checkpoint protocol"):
        read(checkpoint,protocol=cli.PROTOCOL|{"version":"uav-ideal-interception-v3.1.0"})
    with pytest.raises(ValueError,match="incompatible checkpoint protocol"):
        read(checkpoint,protocol=cli.PROTOCOL|{"version":"uav-ideal-interception-v3.2.0"})
    cli.fit(folder,resume=True)
    last = read(checkpoint,protocol=cli.PROTOCOL)
    assert last["steps"]==4 and last["updates"]==2
    assert len(last["evaluations"])==2 and sum(
        "result" in entry for entry in last["evaluations"])==1
    output = tmp_path/"eval"
    cli.main(["evaluate",str(checkpoint),"--out",str(output),"--eval-episodes","2","--trace"])
    result = json.loads((output/"evaluation.json").read_text())
    assert result["protocol"]==cli.PROTOCOL
    assert result["evaluation_protocol"]["kind"]=="fixed_case"
    assert result["evaluation_protocol"]["unique_scenarios"]==1
    assert result["summary"]["episodes"]==2 and result["first_trace"][0]["time"]==0.
    assert all("assignment" in frame for frame in result["first_trace"])
    assert result["episodes"][0]["intruder_paths"][0]["polyline_sha256"]
    assert result["model_sha256"]==cli.model_digest(last["actor"],last["critic"])
    from tools.ideal_interception_timeline import Timeline
    trace=Timeline(output/"episode_000.json")
    assert trace.duration==result["episodes"][0]["time"]
    assert trace.at(trace.duration)["outcome"]==result["episodes"][0]["outcome"]


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required by repository policy")
def test_bilateral_recovery_from_missing_committed_record(tmp_path,monkeypatch):
    config = formal_6v6_config(max_steps=3,intruder_mode="straight")
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
    config = formal_6v6_config(max_steps=3,intruder_mode="straight")
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
