"""Single-factor reward formula, historical paper replay and exact CUDA resume."""
import copy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from env.aircombat import AirCombatConfig,AirCombatEnv
from env.aircombat.rewards import compute_rewards
from env.aircombat.posture_reward import settle_posture,posture_statistics,summarize_statistics,add_statistics
from env.aircombat.geometry import pair_advantages
from env.aircombat.assignment import assignment_matrices
from algorithm.experiments import configurations,manifest,export_plan,run_experiment,write_json
from tests.test_aircombat_training import tensor_equal

C=AirCombatConfig()
G=replace(C,posture_reward_mode='approach_gated')


def transition(red_speed=50.,blue_speed=0.,distance=3000.,raw=2.,config=G):
    red=np.array([[0.,0.,6000.,200.,0.,0.]])
    blue=np.array([[distance,0.,6000.,200.,0.,0.]])
    next_red=red.copy();next_blue=blue.copy()
    next_red[0,0]+=red_speed*config.dt;next_blue[0,0]+=blue_speed*config.dt
    value,samples=settle_posture(np.array([[raw]]),np.ones(1,bool),np.array([0]),red,blue,next_red,next_blue,np.ones(1,bool),config)
    return value[0],samples,red,blue,next_red,next_blue


@pytest.mark.parametrize('self_speed,enemy_speed,expected',[(50,0,1.),(200,0,2.),(50,50,0.),(-50,-100,0.),(100,50,1.),(0,-100,0.),(50,100,0.)])
def test_far_positive_gate_requires_both_own_motion_and_closure(self_speed,enemy_speed,expected):
    value,samples,*_=transition(self_speed,enemy_speed)
    assert value==expected and samples[0]['gate']==expected/2


@pytest.mark.parametrize('p',[-2.,-.0001,0.])
@pytest.mark.parametrize('motion',[-50.,0.,50.,200.])
def test_far_negative_and_zero_are_preserved(p,motion):
    assert transition(motion,0,raw=p)[0]==p


def test_gate_speed_is_real_configuration_not_hidden_constant():
    assert transition(50,0,config=replace(G,posture_gate_speed=50.))[0]==2.
    assert transition(50,0,config=replace(G,posture_gate_speed=200.))[0]==.5


@pytest.mark.parametrize('distance',[0.,99.999999,100.,100.000001,1999.999999,2000.,2000.000001])
def test_post_motion_distance_boundaries_use_exact_paper_intervals(distance):
    _,_,r,b,nr,nb=transition(0,0,distance)
    _,ra,ba,raw=pair_advantages(nr,nb,G)
    expected=float(raw[0,0])
    paper,_=settle_posture(raw,np.ones(1,bool),np.array([0]),r,b,nr,nb,np.ones(1,bool),C)
    gated,samples=settle_posture(raw,np.ones(1,bool),np.array([0]),r,b,nr,nb,np.ones(1,bool),G)
    assert paper[0]==expected
    assert gated[0]==(0. if distance>2000 and expected>0 else expected)
    assert samples[0]['band']==('close' if distance<100 else 'engagement' if distance<=2000 else 'far')
    assert np.isfinite(gated).all()


def test_zero_pre_distance_and_invalid_target_masks_are_finite():
    _,_,r,b,nr,nb=transition(0,12000,0)
    for index,alive in ((0,True),(-1,True),(1,True),(999,True),(0,False)):
        value,samples=settle_posture(np.array([[2.]]),np.ones(1,bool),np.array([index]),r,b,nr,nb,np.array([alive]),G)
        assert value[0]==0 and np.isfinite(value).all()
        if samples:assert samples[0]['gate']==0
    assert not settle_posture(np.ones((1,1)),np.zeros(1,bool),np.array([0]),r,b,nr,nb,np.ones(1,bool),G)[1]


def test_current_transition_deaths_keep_posture_and_other_four_rewards_identical():
    _,_,r,b,nr,nb=transition(50,0)
    for red_alive,blue_alive in ((True,True),(False,True),(True,False),(False,False)):
        s=SimpleNamespace(friendly_states=nr,adversarial_states=nb,friendly_alive=np.array([red_alive]),
            adversarial_alive=np.array([blue_alive]),step_count=5)
        args=(s,np.ones(1,bool),np.array([0]),np.array([[5.]]),np.array([[2.]]),np.array([[2.]]),
              [dict(attacker_side='Red',attacker_id=0)])
        p,parts=compute_rewards(*args,C)
        diag={};g,changed=compute_rewards(*args,G,pre_red=r,pre_blue=b,pre_blue_alive=np.ones(1,bool),posture_diagnostics=diag)
        assert changed['posture'][0]==1.
        for name in ('attack','allied','shootdown','terminal'):np.testing.assert_array_equal(parts[name],changed[name])
        np.testing.assert_array_equal(g,sum(changed.values()))
        assert diag['assigned_samples']==1 and diag['far_raw_total']==2 and diag['far_applied_total']==1


@pytest.mark.parametrize('mode',['paper','approach_gated'])
def test_preassigned_target_and_range_entry_are_not_reassigned_after_motion(mode):
    c=replace(C,posture_reward_mode=mode)
    r=np.array([[0,0,0,200,0,0]],float)
    b=np.array([[2010,0,0,200,0,0],[-4000,0,0,200,0,0]],float)
    nr=r.copy();nr[0,0]=20
    values,samples=settle_posture(np.array([[2.,-1.]]),np.ones(1,bool),np.array([0]),r,b,nr,b,np.ones(2,bool),c)
    stats=posture_statistics(samples,np.ones(1,bool),c)
    assert values[0]==2. and stats['attack_range_entries']==1 and samples[0]['target_id']==0
    assert stats['posture_engagement_total']==2 and stats['far_samples']==0
    # Selecting another pre-action target changes the measured pair, not the
    # actual crossing of the former target. Switching alone is not an entry.
    _,other=settle_posture(np.array([[2.,-1.]]),np.ones(1,bool),np.array([1]),r,b,nr,b,np.ones(2,bool),c)
    assert posture_statistics(other,np.ones(1,bool),c)['attack_range_entries']==0


def test_diagnostic_signs_and_weighted_ratios_not_mean_of_means():
    total={}
    for self_speed,enemy_speed in ((50,0),(50,50),(50,100),(-50,-100)):
        _,samples,*_=transition(self_speed,enemy_speed)
        add_statistics(total,posture_statistics(samples,np.ones(1,bool),G))
    result=summarize_statistics(total)
    assert result['far_approaching_samples']==2 and result['far_maintaining_samples']==1 and result['far_receding_samples']==1
    assert result['far_gate_mean']==.125 and result['far_approaching_fraction']==.5
    assert result['posture_far_total']==result['far_applied_total']==1.
    assert summarize_statistics({})['far_gate_mean'] is None


@pytest.mark.parametrize('options',[None,{'scenario':'head_on_3v3'},{'scenario':'rear_end_3v3'}])
def test_modes_keep_physics_aph_weapons_observations_and_rng_exact(options):
    a,b=AirCombatEnv(C),AirCombatEnv(G)
    a.reset(44,options);b.reset(44,options)
    rng=np.random.default_rng(48);differences=0
    for _ in range(100):
        actions=rng.uniform(.1,.9,(3,3));x=a.step(actions);y=b.step(actions)
        tensor_equal((x[0],x[2],x[3]),(y[0],y[2],y[3]))
        for name in ('events','assignments','H_a','H_b','F','friendly_states','adversarial_states','blue_controls','centralized_state'):
            tensor_equal(x[4][name],y[4][name])
        for name in ('attack','allied','shootdown','terminal'):tensor_equal(x[4]['reward_components'][name],y[4]['reward_components'][name])
        tensor_equal(a.rng.bit_generator.state,b.rng.bit_generator.state)
        for t in (x,y):np.testing.assert_array_equal(t[1],sum(t[4]['reward_components'].values()))
        for t in (x,y):
            d=t[4]['posture_diagnostics']
            assert sum(d['posture_'+band+'_total'] for band in ('close','engagement','far'))==pytest.approx(t[4]['reward_components']['posture'].sum())
        differences+=int(np.any(x[1]!=y[1]))
        if x[2] or x[3]:break
    assert differences>0


@pytest.mark.parametrize('bad',[0.,-1.,np.nan,np.inf,True])
def test_configuration_rejects_bad_gate_speed(bad):
    with pytest.raises(ValueError):AirCombatConfig(posture_gate_speed=bad)


def test_configuration_catalogue_isolated_and_excluded_from_paper_reports(tmp_path):
    with pytest.raises(ValueError):AirCombatConfig(posture_reward_mode='other')
    name='posture_approach_gated_3v3'
    for seed in range(1,6):
        for budget in (2500,5000):
            base,bc=configurations('main_3v3',seed,episodes=budget);gate,gc=configurations(name,seed,episodes=budget)
            assert bc==gc
            assert {k for k,v in base.to_dict().items() if v!=gate.to_dict()[k]}=={'posture_reward_mode'}
    assert configurations(name,smoke=True)[1]==configurations('main_3v3',smoke=True)[1]
    assert manifest(name)['paper_experiment'] is False
    plan=export_plan(tmp_path/'plan.json');assert len(plan['training_runs'])==37
    assert all(t['experiment']!=name for t in plan['training_runs'])
    from tools.report_aircombat import read_runs
    path=tmp_path/name/'seed_1';path.mkdir(parents=True)
    write_json(path/'experiment.json',manifest(name))
    assert read_runs(tmp_path)==[]


def test_missing_transition_context_is_rejected_not_silently_zeroed():
    s=SimpleNamespace(friendly_states=np.zeros((1,6)),adversarial_states=np.zeros((1,6)))
    with pytest.raises(ValueError,match='true pre-action'):
        compute_rewards(s,np.ones(1,bool),np.array([0]),np.zeros((1,1)),np.zeros((1,1)),np.ones((1,1)),[],G)


def test_cli_budget_output_roots_and_existing_data_are_preserved(tmp_path,monkeypatch):
    import algorithm.experiments as module
    monkeypatch.chdir(tmp_path);calls=[]
    def fake_train(ec,cfg,output,**kw):
        calls.append((ec,cfg,output));(output/'train.jsonl').write_text('old-data',encoding='utf-8')
    monkeypatch.setattr(module,'train',fake_train);monkeypatch.setattr(module,'final_evaluation',lambda *a,**kw:None)
    for budget in (2500,5000):
        module.run_experiment('posture_approach_gated_3v3',1,episodes=budget,console=False)
    module.run_experiment('main_3v3',1,episodes=2500,console=False)
    assert len({c[2] for c in calls})==3
    path=calls[0][2];before=(path/'experiment.json').read_bytes()
    with pytest.raises(FileExistsError):module.run_experiment('posture_approach_gated_3v3',1,episodes=2500,console=False)
    assert (path/'experiment.json').read_bytes()==before and (path/'train.jsonl').read_text()=='old-data'
    seen=[];monkeypatch.setattr(module,'run_experiment',lambda *a,**kw:seen.append((a,kw)))
    module.main(['train','--experiment','posture_approach_gated_3v3','--seed','5','--episodes','2500'])
    assert seen[0][0][1]==5 and seen[0][1]['episodes']==2500


def test_checkpoint_rejects_cross_mode_and_gate_speed_before_loading_parameters(tmp_path):
    from algorithm.common.aircombat_vector import AirCombatVectorEnv
    from algorithm.common.checkpoint import save_protocol_checkpoint,read_protocol_checkpoint,restore_protocol_checkpoint
    from algorithm.mappo.trainer import ProtocolMAPPOTrainer
    from algorithm.mappo.config import ProtocolTrainConfig
    from env.aircombat.config import ENVIRONMENT_VERSION
    cfg=ProtocolTrainConfig(num_envs=1,horizon=1,ppo_epochs=1)
    t=ProtocolMAPPOTrainer(cfg);before=copy.deepcopy(t.actor.state_dict())
    for ec in (C,G):
        with AirCombatVectorEnv(ec,1) as v:
            path=tmp_path/(ec.posture_reward_mode+'.pt');save_protocol_checkpoint(path,t,env_config=ec,vector=v)
            payload=read_protocol_checkpoint(path,environment_version=ENVIRONMENT_VERSION)
            assert payload['env_config']['posture_reward_mode']==ec.posture_reward_mode
            for wrong in (replace(ec,posture_reward_mode='paper' if ec==G else 'approach_gated'),replace(ec,posture_gate_speed=200.)):
                with pytest.raises(ValueError,match='incompatible aircombat'):restore_protocol_checkpoint(payload,t,env_config=wrong,vector=v)
            tensor_equal(before,t.actor.state_dict())
            bad=v.snapshot();bad['workers'][0]['config']['posture_reward_mode']='paper' if ec==G else 'approach_gated'
            with pytest.raises(RuntimeError,match='snapshot configuration mismatch'):v.restore(bad)


def test_diagnostic_calculation_has_no_reward_state_or_rng_effect(monkeypatch):
    from algorithm.common.aircombat_vector import AirCombatVectorEnv
    from algorithm.mappo.trainer import ProtocolMAPPOTrainer
    from algorithm.mappo.config import ProtocolTrainConfig
    import env.aircombat.posture_reward as module
    cfg=ProtocolTrainConfig(num_envs=1,horizon=3,ppo_epochs=1,minibatch_size=7)
    # Single-process environment wrapper so a diagnostics-only monkeypatch is
    # inherited directly. Actual spawn/checkpoint tests separately verify IPC.
    for mode in ('paper','approach_gated'):
        ec=replace(C,posture_reward_mode=mode,max_steps=3)
        a=AirCombatEnv(ec);a.reset(19);actions=np.random.default_rng(99).uniform(.1,.9,(3,3,3))
        ref=[a.step(x) for x in actions];refstate=a.snapshot()
        original=module.posture_statistics
        monkeypatch.setattr(module,'posture_statistics',lambda *args:{})
        b=AirCombatEnv(ec);b.reset(19);out=[b.step(x) for x in actions]
        for x,y in zip(ref,out):
            tensor_equal(x[:4],y[:4]);tensor_equal(x[4]['reward_components'],y[4]['reward_components'])
        tensor_equal(refstate,b.snapshot())
        monkeypatch.setattr(module,'posture_statistics',original)


@pytest.mark.parametrize('mode',['paper','approach_gated'])
def test_diagnostic_aggregation_has_no_actor_sampling_ppo_parameters_or_rng_effect(mode):
    import random
    from algorithm.common.aircombat_vector import AirCombatVectorEnv
    from algorithm.mappo.trainer import ProtocolMAPPOTrainer
    from algorithm.mappo.config import ProtocolTrainConfig
    cfg=ProtocolTrainConfig(num_envs=2,horizon=3,ppo_epochs=1,minibatch_size=11)
    ec=replace(C,posture_reward_mode=mode,max_steps=2)
    a=ProtocolMAPPOTrainer(cfg)
    with AirCombatVectorEnv(ec,2) as v:
        r1,e1=a.collect(v);m1=a.update(r1);snapshot=v.snapshot()
        rng1=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
    b=ProtocolMAPPOTrainer(cfg)
    with AirCombatVectorEnv(ec,2) as v:
        original=v.step
        def step_without_diagnostic(*args,**kwargs):
            result=original(*args,**kwargs)
            for info in result.infos:info.pop('posture_diagnostics',None)
            return result
        v.step=step_without_diagnostic
        r2,e2=b.collect(v);m2=b.update(r2)
        tensor_equal((r1,e1,m1),(r2,e2,m2));tensor_equal(snapshot,v.snapshot())
        tensor_equal(rng1,(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all()))
    tensor_equal(a.actor.state_dict(),b.actor.state_dict());tensor_equal(a.critic.state_dict(),b.critic.state_dict())
    tensor_equal(a.actor_optimizer.state_dict(),b.actor_optimizer.state_dict())
    tensor_equal(a.critic_optimizer.state_dict(),b.critic_optimizer.state_dict())


def test_gated_pending_rollout_diagnostics_worker_reset_and_update_resume_exact(tmp_path):
    import random
    from algorithm.common.aircombat_vector import AirCombatVectorEnv
    from algorithm.common.checkpoint import save_protocol_checkpoint,read_protocol_checkpoint,restore_protocol_checkpoint
    from algorithm.mappo.trainer import ProtocolMAPPOTrainer
    from algorithm.mappo.config import ProtocolTrainConfig
    from env.aircombat.config import ENVIRONMENT_VERSION
    cfg=ProtocolTrainConfig(num_envs=2,horizon=4,ppo_epochs=1,minibatch_size=13,checkpoint_interval_episodes=1)
    ec=replace(G,max_steps=2);a=ProtocolMAPPOTrainer(cfg);saved=[]
    with AirCombatVectorEnv(ec,2) as v:
        def checkpoint(t,vector):
            if not saved:
                path=tmp_path/'pending.pt';save_protocol_checkpoint(path,t,env_config=ec,vector=vector);saved.append(path)
        r1,e1=a.collect(v,checkpoint_callback=checkpoint);m1=a.update(r1)
        expected=copy.deepcopy((a.actor.state_dict(),a.critic.state_dict(),a.actor_optimizer.state_dict(),a.critic_optimizer.state_dict(),
            a.rollout_metrics,v.snapshot(),random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all()))
        payload=read_protocol_checkpoint(saved[0],environment_version=ENVIRONMENT_VERSION)
        assert payload['phase']=='collecting_rollout' and payload['pending_collection']['cursor']==1
        assert payload['pending_collection']['posture_totals']['diagnostic_steps']==3
        b=ProtocolMAPPOTrainer(cfg);restore_protocol_checkpoint(payload,b,env_config=ec,vector=v)
        r2,e2=b.collect(v);m2=b.update(r2)
        tensor_equal((r1,e1,m1),(r2,e2,m2))
        tensor_equal(expected,(b.actor.state_dict(),b.critic.state_dict(),b.actor_optimizer.state_dict(),b.critic_optimizer.state_dict(),
            b.rollout_metrics,v.snapshot(),random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all()))
