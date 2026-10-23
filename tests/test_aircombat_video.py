"""Offline reporting must use recorded death endpoints and actual timestamps."""
import copy
import numpy as np
from tools.aircombat_video import export_details


def test_report_death_positions_windows_and_no_frozen_path_extension(tmp_path):
    frames=[]
    for step in range(4):
        frames.append(dict(step=step,friendly_states=[[step,0,6000,200,0,0]],
            adversarial_states=[[500+min(step,2),0,6000,200,0,0]],friendly_alive=[True],
            adversarial_alive=[step<2],assignments=[0 if step<2 else -1],
            red_attack_eligible=[[step in (1,2)]],blue_attack_eligible=[[False]]))
    data=dict(checkpoint='fixture',checkpoint_sha256='fixture',completed_training_episodes=2500,updates=36,sampled_steps=1,
        environment_version='fixture',scenario='fixture',seed=100,policy='fixture',environment=dict(dt=.25),
        frames=frames,summary={},assignment_changes=[],events=[dict(attacker_side='Red',attacker_id=0,target_id=0,step=2)])
    before=copy.deepcopy(data);report=export_details(data,tmp_path)
    assert data==before
    assert report['kills'][0]['attacker_xyz_m']==[2,0,6000]
    assert report['kills'][0]['target_xyz_m']==[502,0,6000]
    assert report['kills'][0]['time_s']==.5 and report['kills'][0]['distance_m']==500
    window=report['attack_windows'][0]
    assert window['start_step']==1 and window['end_step']==2 and window['eligible_duration_s']==.5
    blue=report['aircraft'][1]
    assert not blue['survived'] and blue['path_length_m']==2 and blue['last_step']==2
    assert (tmp_path/'trajectory.csv').exists() and (tmp_path/'battle_report.json').exists()
