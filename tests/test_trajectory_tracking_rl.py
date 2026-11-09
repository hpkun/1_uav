from dataclasses import replace

import numpy as np
import pytest
import torch

from algorithm.common.interception_checkpoint import read, restore, save
from algorithm.mappo.gae import gae
from algorithm.mappo.interception import TrainConfig
from algorithm.trajectory_tracking import PROTOCOL, TrackingMAPPO, evaluate, fit, main, make_vector
from env.interception.motion import advance, command, execute_velocity
from env.preset_tracking import _commands, _frame
from env.trajectory_rl import OBS_DIM, STATE_DIM, TrackingConfig, TrackingEnv, TrackingNumericalFailure


IDLE = np.full((5,3),.5)


def staged(**changes):
    fields = dict(starts=((0,0,12),(0,2,12)),endpoints=((0,0,12),(1,2,12)),
                  initial_velocities=((0,0,0),(0,0,0)),start_jitter=(0,0,0),hold_seconds=.15)
    return TrackingConfig(**(fields|changes))


def test_pairing_observation_and_analytic_step():
    a,b = [TrackingEnv(TrackingConfig(trajectory_mode=mode)) for mode in ("straight","bezier")]
    obs,ia = a.reset(np.random.SeedSequence([0,17,0,0]))
    ob,ib = b.reset(np.random.SeedSequence([0,17,0,0]))
    np.testing.assert_array_equal(ia["actual_states"],ib["actual_states"])
    np.testing.assert_array_equal(ia["active_mask"],ib["active_mask"])
    assert obs.shape==ob.shape==(5,OBS_DIM) and ia["centralized_state"].shape==(STATE_DIM,)
    assert not obs[2:].any()
    np.testing.assert_array_equal(obs[:,17],[1,1,0,0,0])
    np.testing.assert_allclose(obs[0,:3],ia["actual_states"][0,:3]/a.config.observation_position_scale)
    np.testing.assert_allclose(obs[1,18:21],obs[0,:3])
    for i in (0,1):
        assert ia["reference_paths"][i]["points"][-1]==ib["reference_paths"][i]["points"][-1]
        assert len(ib["reference_paths"][i]["points"])==4
    with pytest.raises(ValueError,match="dimensions"):
        a.step(np.zeros((2,3)))
    controls=IDLE.copy()
    controls[:2,0]=.65
    u = execute_velocity(command(controls[:2],a.config.vmax),ia["actual_states"][:2,3:],
                         a.config.vmax,a.config.tau,a.config)
    _,reward,_,_,info=a.step(controls)
    np.testing.assert_array_equal(info["actual_states"][:2],advance(ia["actual_states"][:2],u,a.config.tau,.1))
    assert reward[0]==reward[1] and not reward[2:].any()
    assert reward[0]==pytest.approx(sum(info["reward_parts"].values()))
    assert info["reward_parts"]["position"]==pytest.approx(-sum(r["position_error"] for r in info["samples"])/20)
    _,repeat=b.reset(np.random.SeedSequence([0,17,0,0]))
    assert repeat["reference_paths"]==ib["reference_paths"]


def test_partial_retirement_and_precision_failure():
    env=TrackingEnv(staged())
    env.reset(7)
    env.step(IDLE)
    obs,r,term,trunc,info=env.step(IDLE)
    assert not term and not trunc and info["active_mask"].tolist()==[False,True,False,False,False]
    assert not obs[0].any() and r[0]==r[1] and info["reward_parts"]["completion"]==2.5
    assert info["samples"][0]["completion_time"]==pytest.approx(.15)
    while not term:
        _,r,term,trunc,info=env.step(IDLE)
        assert r[0]==0
    assert not trunc and info["outcome"]=="precision_failure" and not info["active_mask"].any()
    assert info["status"]==["completed","missed"] and info["reward_parts"]["missed"]==-1.


def test_true_completion_and_boundary():
    env=TrackingEnv(staged(endpoints=((0,0,12),(0,2,12))))
    env.reset(1);env.step(IDLE)
    _,r,term,trunc,info=env.step(IDLE)
    assert term and not trunc and info["outcome"]=="success"
    assert info["reward_parts"]["completion"]==5 and info["time"]==pytest.approx(.15)
    config=staged(starts=((5.4,0,12),(0,2,12)),endpoints=((1,0,12),(0,2,12)),
                  upper=(6,20,30),initial_velocities=((3,0,0),(0,0,0)))
    env=TrackingEnv(config);env.reset(1)
    _,_,term,trunc,info=env.step(IDLE)
    assert term and not trunc and info["outcome"]=="boundary" and info["boundary_id"]==0
    assert 0<info["time"]<config.dt and not info["active_mask"].any()
    assert info["reward_parts"]["boundary"]==-10


@pytest.mark.parametrize("mode",["straight","bezier"])
def test_existing_reference_follower_completes_nonzero_paths(mode):
    c = TrackingConfig(trajectory_mode=mode)
    env = TrackingEnv(c)
    _,info = env.reset(1200)
    while True:
        ids = np.flatnonzero(info["active_mask"])
        paths = [env.paths[i] for i in ids]
        frame = _frame(paths,info["actual_states"][ids],info["time"])
        request,_,_,_ = _commands(paths,frame,c.dt,c)
        u = IDLE.copy()
        u[ids] += .5*request/np.maximum(c.vmax,np.linalg.norm(request,axis=1,keepdims=True))
        _,_,terminated,truncated,info = env.step(u)
        if terminated or truncated:
            break
    assert terminated and not truncated and info["outcome"]=="success"
    assert info["status"]==["completed","completed"]


def test_truncated_active_mask_bootstraps_and_failure_does_not_mutate():
    env=TrackingEnv(staged(max_steps=2))
    env.reset(1);env.step(IDLE)
    actions=IDLE.copy();actions[0,0]=np.nan
    with pytest.raises(TrackingNumericalFailure,match="nonfinite"):
        env.step(actions)
    assert env.steps==1
    _,r,term,trunc,info=env.step(IDLE)
    assert not term and trunc and info["outcome"]=="time_limit"
    assert info["active_mask"].tolist()==[False,True,False,False,False]
    values=torch.zeros((1,1,5))
    before=torch.tensor([[[1,1,0,0,0]]],dtype=torch.bool)
    after=torch.tensor(info["active_mask"]).reshape(1,1,5)
    _,ret=gae(torch.tensor(r).reshape(1,1,5),values,torch.ones_like(values),before,after,
              torch.tensor([[False]]),torch.tensor([[True]]),.99,.95)
    torch.testing.assert_close(ret[0,0,:2],torch.tensor([r[0],r[1]+.99],dtype=ret.dtype))


@pytest.mark.parametrize("mode",["straight","bezier"])
def test_cuda_rollout_update_resume_and_deterministic_eval(tmp_path,mode):
    config=staged(trajectory_mode=mode,max_steps=3)
    settings=TrainConfig(workers=1,horizon=4,epochs=1,minibatch=8)
    model=TrackingMAPPO(settings)
    with make_vector(config,settings) as vector:
        rollout,episodes=model.collect(vector)
        assert episodes and episodes[0]["outcome"]=="time_limit"
        before=next(model.actor.parameters()).detach().clone()
        update=model.update(rollout)
        assert not torch.equal(before,next(model.actor.parameters())) and update["valid_samples"]==7
        path=tmp_path/"latest.pt"
        save(path,model,config,vector,protocol=PROTOCOL,evaluations=[])
        with pytest.raises(ValueError,match="protocol"):
            read(path)
        payload=read(path,protocol=PROTOCOL)
        expected,_=model.collect(vector)
        with pytest.raises(ValueError,match="configuration"):
            restore(payload,model,replace(config,trajectory_mode="bezier" if mode=="straight" else "straight"),
                    vector,protocol=PROTOCOL)
        restore(payload,model,config,vector,protocol=PROTOCOL)
        actual,_=model.collect(vector)
        for key,value in vars(expected).items():
            torch.testing.assert_close(value,getattr(actual,key),rtol=0,atol=0)
        assert evaluate(model,config,[2200])==evaluate(model,config,[2200])


def test_periodic_evaluation_and_recovered_commit(tmp_path, capsys):
    config=staged(max_steps=2)
    training=TrainConfig(workers=1,horizon=2,epochs=1,minibatch=4)
    folder=tmp_path/"run"
    path=fit(folder,config,training,total_steps=6,until_steps=4,interval=2,evaluation_seeds=[2200])
    logs = capsys.readouterr().out
    assert "[Preset tracking START]" in logs and "[Preset tracking PAUSE]" in logs
    assert "[Preset tracking TRAIN] Update 1/3 (33.3%)" in logs
    assert logs.count("[Preset tracking EVAL]") == 1 and "[Preset tracking EVAL] Update 2" in logs
    assert "Tracking success" in logs and "Mean position error" in logs and "eval N/A" not in logs
    (folder/"update_000002.json").unlink()
    (folder/"evaluation_000002.json").unlink()
    with pytest.raises(ValueError,match="mode"):
        fit(folder,resume=True,expected_mode="bezier")
    fit(folder,resume=True,expected_mode="straight")
    logs = capsys.readouterr().out
    assert "[Preset tracking RESUME]" in logs and "[Preset tracking DONE]" in logs
    assert logs.count("[Preset tracking EVAL]") == 1 and "[Preset tracking EVAL] Update 3" in logs
    payload=read(path,protocol=PROTOCOL)
    assert payload["steps"]==6 and [e["update"] for e in payload["evaluations"]]==[2,3]
    assert (folder/"evaluation_000002.json").is_file() and (folder/"evaluation_000003.json").is_file()
    output = tmp_path/"standalone"
    main(["evaluate", str(path), "--out", str(output), "--eval-episodes", "1"])
    logs = capsys.readouterr().out
    assert "[Preset tracking EVAL]" in logs and "Checkpoint update 3" in logs
    assert (output/"evaluation.json").is_file()
