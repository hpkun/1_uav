import copy
import math
import re
from algorithm.common.progress import (ProgressClock,duration,format_training,format_evaluation,format_final,loss_values)


def record():
    return dict(sampled_steps=50000,update=20,joint_episode_count=200,
        training=dict(success_rate=0.,completion_rate=1/3,mean_episode_return=-2.15),
        rollout=dict(mean_target_distance=4200.,target_distance_reduction=350.,speed_clip_rate=.012,gamma_clip_rate=0.),
        optimization=dict(policy_loss=-.0021,actor_total_loss=.00065,actor_loss=.00065,critic_loss=.0355,approx_kl=.00358,clip_fraction=.0304,entropy=-2.75))


def evaluation(success=.2):
    return dict(success_rate=success,completion_rate=.733,mean_completed_waypoints=2.2,mean_reward=-.3,mean_episode_length=200.,
        episodes=[{'success':i<round(5*success)} for i in range(5)])


def test_readable_training_and_no_metric_mutation():
    r=record(); before=copy.deepcopy(r)
    text=format_training(r,200000,speed=1200,eta=125)
    assert len(text.splitlines())==2
    for part in ('50,000/200,000 (25.0%)','Update 20','Episodes 200','1,200 joint steps/s','ETA 00:02:05',
        'Success 0.0%','Completion 33.3%','Return -2.150','Distance 4,200 m','Distance reduction +350 m',
        'Policy -0.0021','Actor total 0.0006','Critic 0.0355','KL 0.00358','Clip 3.04%','Entropy -2.75',
        'Speed clipping 1.2%','Gamma clipping 0.0%'): assert part in text
    assert r==before and '\x1b' not in text


def test_missing_training_episodes_remain_na():
    r=record(); r['training']=dict(success_rate=None,completion_rate=None,mean_episode_return=None)
    text=format_training(r,200000)
    assert 'Train: Success N/A | Completion N/A | Return N/A' in text


def test_independent_evaluation_and_best_identity():
    report=evaluation(); before=copy.deepcopy(report)
    text=format_evaluation(50000,report,3,best_updated=True)
    assert '[EVAL]' in text and 'n=5' in text and 'Success 20.0% (1/5)' in text
    assert 'Completed waypoints 2.2/3' in text and 'BEST UPDATED' in text
    assert 'BEST UNCHANGED' in format_evaluation(50000,report,3,best_updated=False)
    summary=dict(sampled_steps=200000,updates=80,joint_episode_count=800,best_evaluation=evaluation(.4),latest_evaluation=evaluation(0.),
        best_checkpoint='/run/best.pt',latest_checkpoint='/run/latest.pt')
    text=format_final(summary,'/run',3)
    assert 'BEST: n=5 | Success 40.0%' in text and 'LATEST: n=5 | Success 0.0%' in text
    for name in ('best.pt','latest.pt','training_metrics.jsonl','evaluation_history.csv','run_summary.json'): assert name in text
    assert 'converged' not in text.lower() and report==before


def test_clock_excludes_pre_resume_steps():
    moments=iter([100.,110.]); clock=ProgressClock(1000,clock=lambda:next(moments))
    speed,eta=clock.measure(2000,5000)
    assert speed==100 and eta==30 and duration(eta)=='00:00:30'
    moments=iter([100.,100.]); clock=ProgressClock(1000,clock=lambda:next(moments))
    assert clock.measure(1000,5000)==(None,None)


def test_nonfinite_console_numbers_are_not_printed():
    r=record()
    for group in ('training','rollout','optimization'):
        for key in r[group]: r[group][key]=float('nan') if 'loss' in key else float('inf')
    text=format_training(r,200000,speed=float('inf'),eta=float('nan'))
    assert 'N/A' in text and not re.search(r'\b(nan|inf|infinity)\b',text,re.I)
    assert duration(-1)=='N/A'


def test_historical_loss_interpretation():
    original={'actor_loss':-.02,'entropy':-2.5}
    values,legacy=loss_values(original)
    assert legacy and values['policy_loss']==-.02 and values['actor_total_loss'] is None
    assert original=={'actor_loss':-.02,'entropy':-2.5}
    values,legacy=loss_values(record()['optimization'])
    assert not legacy and values['policy_loss']!=values['actor_total_loss']


def test_presentation_leaves_random_states_untouched():
    import random
    import numpy as np
    import torch
    py=random.getstate(); numpy=np.random.get_state(); tensor=torch.get_rng_state().clone()
    format_training(record(),200000,speed=1200,eta=125)
    format_evaluation(50000,evaluation(),3,best_updated=True)
    assert random.getstate()==py
    after=np.random.get_state()
    assert after[0]==numpy[0] and after[2:]==numpy[2:]
    np.testing.assert_array_equal(after[1],numpy[1])
    torch.testing.assert_close(torch.get_rng_state(),tensor,rtol=0,atol=0)
