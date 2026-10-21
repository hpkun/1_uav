"""Independent known-answer records; no simulator or model inference."""
import copy
import json
import sys

import pytest

from tools.trajectory_audit import audit, main


def record():
    first = dict(states=[[0,0,3000,200,0,0], [0,500,3000,200,0,0]],
                 waypoints=[[1000,0,3000]], active_mask=[True,True], assignments=[0,-1],
                 completed=[False], success=False, completion_rate=0., episode_length=0)
    second = copy.deepcopy(first)
    second.update(states=[[50,0,3000,200,0,0], [50,500,3000,200,0,0]],
                  previous_states=copy.deepcopy(first['states']), current_active_mask=[True,True],
                  transition_assignments=[0,-1], physical_actions=[[0,0,1],[0,0,1]],
                  episode_length=1, events=[], rewards=[.09,-.01],
                  reward_components=dict(progress=[.1,0],completion=[0,0],separation=[0,0],efficiency=[-.01,-.01],terminal=[0,0]))
    return dict(config=dict(dt=.25,safe_distance=100,completion_radius=300,scale='fixture'),seed=7,frames=[first,second])


def test_known_distances_controls_and_reward_alignment_without_input_changes(tmp_path):
    path=tmp_path/'trajectory.json'
    path.write_text(json.dumps(record()),encoding='utf-8')
    before=path.read_bytes()
    text=audit(path)
    assert path.read_bytes()==before
    assert '1000.000/950.000/950.000m' in text
    assert 'path length=50.000m' in text
    assert '0.125 / 0.125 / 0.125' in text
    assert 'assigned=0; idle=1' in text
    assert 'raw Beta sample' in text
    assert "'progress': 0.0" in text
    assert 'Reward checks=2' in text


def test_detects_time_alignment_and_wrong_reward(tmp_path):
    data=record(); data['frames'][1]['previous_states'][0][0]=7
    data['frames'][1]['reward_components']['progress'][0]=.7
    path=tmp_path/'trajectory.json'; path.write_text(json.dumps(data),encoding='utf-8')
    text=audit(path)
    assert 'max error=7' in text
    assert "'progress': 0.6" in text


def test_independently_detects_false_completion_and_attribution(tmp_path):
    data=record(); second=data['frames'][1]
    second['events']=[dict(step=1,waypoint_id=0,uav_id=1)]
    second['completed']=[True]
    second['reward_components']['completion']=[0,10]
    path=tmp_path/'trajectory.json'; path.write_text(json.dumps(data),encoding='utf-8')
    text=audit(path)
    assert 'event mismatches=1; completed-flag mismatches=1' in text
    assert "'completion': 10.0" in text


@pytest.mark.parametrize('mutation', ['nonfinite', 'nonconsecutive', 'bad_assignment'])
def test_rejects_malformed_evidence(tmp_path, mutation):
    data=record()
    if mutation=='nonfinite': data['frames'][1]['states'][0][0]=float('nan')
    if mutation=='nonconsecutive': data['frames'][1]['episode_length']=3
    if mutation=='bad_assignment': data['frames'][1]['transition_assignments'][0]=99
    path=tmp_path/'trajectory.json'; path.write_text(json.dumps(data),encoding='utf-8')
    with pytest.raises(ValueError): audit(path)


def test_cli_refuses_source_and_existing_output(tmp_path, monkeypatch):
    path=tmp_path/'trajectory.json'; path.write_text(json.dumps(record()),encoding='utf-8')
    before=path.read_bytes()
    monkeypatch.setattr(sys,'argv',['trajectory_audit',str(path),'--output',str(path)])
    with pytest.raises(ValueError,match='source'): main()
    assert path.read_bytes()==before
    out=tmp_path/'report.md'; out.write_text('keep',encoding='utf-8')
    monkeypatch.setattr(sys,'argv',['trajectory_audit',str(path),'--output',str(out)])
    with pytest.raises(FileExistsError): main()
    assert out.read_text(encoding='utf-8')=='keep'


def test_optional_policy_recording_preserves_trajectory_rewards_and_rng(tmp_path):
    import torch
    from env.config import EnvConfig
    from algorithm.mappo.config import TrainConfig
    from algorithm.mappo.trainer import MAPPOTrainer
    from algorithm.mappo.evaluation import evaluate
    trainer=MAPPOTrainer(TrainConfig())
    config=EnvConfig(max_steps=2)
    plain=tmp_path/'plain.json'; detailed=tmp_path/'detailed.json'
    before=torch.cuda.get_rng_state().clone()
    summary_a=evaluate(trainer,config,[20000000],plain)
    summary_b=evaluate(trainer,config,[20000000],detailed,policy_diagnostics=True)
    assert torch.equal(before,torch.cuda.get_rng_state())
    assert summary_a==summary_b
    a=json.loads(plain.read_text(encoding='utf-8'))
    b=json.loads(detailed.read_text(encoding='utf-8'))
    for frame in b['frames'][1:]:
        assert frame.pop('policy_diagnostics')['selection']=='mean'
    assert a==b
    report=audit(detailed)
    assert 'Beta parameters present for 2' in report
    assert 'max residual on assigned active transitions: 0.0' in report
