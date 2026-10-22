"""Section IV protocol isolation, exact checkpoint/rollout resume and statistics."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from env.aircombat import AirCombatConfig,AirCombatEnv
from env.aircombat.assignment import assignment_matrices
from env.aircombat.geometry import pair_advantages
from env.aircombat.rewards import allied_risk_terms,allied_penalty
from env.aircombat.opponent import aggregate_scores,candidate_scores
from env.aircombat.scenarios import generate
from algorithm.experiments import CATALOG,FACTORS,configurations,export_plan,manifest,run_directory,blue_sensitivity,write_json
from algorithm.aircombat import train,evaluate
from algorithm.mappo.config import ProtocolTrainConfig
from algorithm.mappo.trainer import ProtocolMAPPOTrainer,gae
from algorithm.common.aircombat_vector import AirCombatVectorEnv,CombatEpisodeAccumulator
from algorithm.common.checkpoint import save_protocol_checkpoint,read_protocol_checkpoint,restore_protocol_checkpoint
from env.aircombat.config import ENVIRONMENT_VERSION
from tools.report_aircombat import mean_std,moving_average,report,training_episodes
from tools.aircombat_trajectories import record
from tests.test_aircombat_training import tensor_equal


def changed(a,b):return {k for k in a if a[k]!=b[k]}


@pytest.mark.parametrize('name',tuple(CATALOG))
def test_all_catalogue_defaults_and_single_factor_changes(name):
    env,cfg=configurations(name)
    base,train_cfg=configurations('main_'+env.scale)
    assert cfg.horizon==1500 and cfg.num_envs==10 and cfg.training_episodes==5000
    assert env.max_steps==250 and cfg.ppo_epochs==10 and cfg.minibatch_size==128
    assert cfg.evaluation_episodes==200 and cfg.checkpoint_interval_episodes==50
    assert cfg.worker_seeds()==tuple(range(1000,1010))
    expected=CATALOG[name]
    assert changed(base.to_dict(),env.to_dict())==set(expected.get('env',{}))
    assert changed(train_cfg.to_dict(),cfg.to_dict())==set(expected.get('training',{}))
    assert env.opponent_aggregation=='aggregate-sum' and env.opponent_best_probability==.8


def test_plan_five_seeds_archiving_and_deduplicated_factor_baseline(tmp_path):
    data=export_plan(tmp_path/'plan.json')
    assert len(data['training_runs'])==37
    assert len({(r['experiment'],r['training_seed']) for r in data['training_runs']})==37
    for name in ('main_3v3','main_5v5','main_3v5','without_aph','without_allied','without_masking'):
        assert [r['training_seed'] for r in data['training_runs'] if r['experiment']==name]==[1,2,3,4,5]
    for _,(_,values,baseline) in FACTORS.items():assert len(values)==3 and baseline in values
    assert 4950 in manifest('main_5v5')['checkpoint_targets']
    assert run_directory(tmp_path,'main_3v3',1)!=run_directory(tmp_path,'main_3v3',2)
    assert data['blue_evaluation']['probabilities']['episodes_per_seed']==200
    assert data['blue_evaluation']['aggregation']['episodes_per_seed']==40


def test_existing_smoke_metadata_is_not_overwritten_by_formal_retry(tmp_path,monkeypatch):
    import algorithm.experiments as module
    path=run_directory(tmp_path,'main_3v3',1);path.mkdir(parents=True)
    original=manifest('main_3v3',1,smoke=True);write_json(path/'experiment.json',original)
    before=(path/'experiment.json').read_bytes()
    (path/'train.jsonl').write_text('{}\n',encoding='utf-8')
    monkeypatch.setattr(module,'train',lambda *args,**kwargs:pytest.fail('existing run must be rejected before training'))
    with pytest.raises(FileExistsError):module.run_experiment('main_3v3',1,tmp_path)
    assert (path/'experiment.json').read_bytes()==before


def test_nearest_targets_independent_reuse_ties_and_dead_slots():
    env,_=configurations('without_aph');s=generate(env,np.random.default_rng(3))
    s.friendly_states[:,:3]=[[0,0,0],[0,0,0],[300,0,0]]
    s.adversarial_states[:,:3]=[[100,0,0],[-100,0,0],[1000,0,0]]
    assignment,_,_,f=assignment_matrices(s,env)
    np.testing.assert_array_equal(assignment,[0,0,0])
    s.friendly_alive[1]=False;s.adversarial_alive[0]=False
    assignment,_,_,f=assignment_matrices(s,env)
    np.testing.assert_array_equal(assignment,[1,-1,1])


def test_allied_ablation_changes_reward_only_and_risks_still_record():
    c=AirCombatConfig();s=generate(c,np.random.default_rng(4))
    # Close Red for guaranteed nonzero proximity. Far Blue cannot kill in 1 step.
    red=np.tile([0,0,6000,200,0,0],(3,1));red[:,1]=[0,20,40]
    blue=red.copy();blue[:,0]=10000
    opts=dict(friendly_states=red,adversarial_states=blue)
    a,b=AirCombatEnv(c),AirCombatEnv(replace(c,allied_penalty_enabled=False))
    a.reset(17,opts);b.reset(17,opts)
    action=np.tile([.5,1/3,1/8],(3,1))
    x,y=a.step(action),b.step(action)
    np.testing.assert_array_equal(x[0],y[0])
    for k in ('friendly_states','adversarial_states','assignments','blue_action_indices'):
        np.testing.assert_array_equal(x[4][k],y[4][k])
    for k in ('attack','posture','shootdown','terminal'):
        np.testing.assert_array_equal(x[4]['reward_components'][k],y[4]['reward_components'][k])
    np.testing.assert_allclose(y[1],x[1]-x[4]['reward_components']['allied'])
    assert x[4]['reward_components']['allied'].sum()<0 and not y[4]['reward_components']['allied'].any()
    tensor_equal(x[4]['risk_terms'],y[4]['risk_terms'])
    assert a.rng.bit_generator.state==b.rng.bit_generator.state


def test_raw_risk_terms_hand_answer_and_cumulative_accounting():
    c=AirCombatConfig();red=np.array([[0,0,6000,200,0,0],[50,0,6000,200,0,0],[500,0,6000,200,0,0]],float)
    ba=np.array([[5,5,0],[0,5,5]],float)
    terms=allied_risk_terms(red,np.ones(3,bool),np.ones(2,bool),ba,c)
    np.testing.assert_array_equal(terms['proximity'],[.25,.25,0])
    np.testing.assert_array_equal(terms['shared_exposure'],[.5,1,.5])
    np.testing.assert_allclose(-terms['proximity']-terms['shared_exposure'],allied_penalty(red,np.ones(3,bool),np.ones(2,bool),ba,c))
    tracker=CombatEpisodeAccumulator();tracker.reset({'active_mask':np.ones(3,bool)})
    info=dict(current_active_mask=np.ones(3,bool),numerical_failure=False,
              reward_components=dict.fromkeys(('attack','posture','allied','shootdown','terminal'),np.zeros(3)),
              risk_terms=terms,outcome='timeout',win=False,blue_shot_down=0,red_survivors=3)
    for _ in range(4):tracker.add(info,np.zeros(3))
    summary=tracker.summary(info)
    assert summary['proximity_risk']==pytest.approx(4*.5/3)
    assert summary['shared_exposure_risk']==pytest.approx(4*2/3)


def test_blue_equation24_reductions_and_only_requested_parameter_changes():
    scores=np.array([3.,-7.,2.])
    assert aggregate_scores(scores)==-2
    assert aggregate_scores(scores,'opportunistic-max')==3
    assert aggregate_scores(scores,'threat-aware-maximin')==-7
    base=AirCombatConfig(scale='5v5')
    for p in (1.,.8,.6):
        assert changed(base.to_dict(),replace(base,opponent_best_probability=p).to_dict())==({'opponent_best_probability'} if p!=.8 else set())
    for mode in ('aggregate-sum','opportunistic-max','threat-aware-maximin'):
        assert changed(base.to_dict(),replace(base,opponent_aggregation=mode).to_dict())==({'opponent_aggregation'} if mode!='aggregate-sum' else set())
    with pytest.raises(ValueError,match='evaluation-only'):
        train(replace(base,opponent_best_probability=.6),ProtocolTrainConfig(),'outputs/should_not_exist')


def test_ppo_mask_ablation_only_loss_samples_keep_real_gae_and_physics():
    from tools.numerical_cases import hand_batch
    c=ProtocolTrainConfig(ppo_epochs=1,minibatch_size=64,normalize_advantages=False)
    a,b=ProtocolMAPPOTrainer(c),ProtocolMAPPOTrainer(replace(c,ppo_masking=False))
    batch=hand_batch((2,1,0),a.device,poison=False)
    with torch.no_grad():batch.log_probs.copy_(a.actor.evaluate(batch.observations,batch.actions)[0])
    original=copy.deepcopy(batch)
    m1=a.update(batch);m2=b.update(batch)
    assert m1['valid_samples']==m2['valid_samples']==3
    assert m2['loss_samples']==6 and m2['numerical_diagnostics']['invalid_samples_in_loss']==3
    tensor_equal(batch,original)
    assert m2['critic_loss']!=m1['critic_loss']
    assert all(np.isfinite(m2[k]) for k in ('policy_loss','critic_loss','entropy'))
    # Reconfiguration is on the trainer only, no environment alive/attack switch.
    assert configurations('without_masking')[0]==configurations('main_3v3')[0]


def test_exact_midwave_checkpoint_resume_preserves_horizon_and_update(tmp_path):
    ec=AirCombatConfig(max_steps=2)
    c=ProtocolTrainConfig(num_envs=2,horizon=4,ppo_epochs=1,minibatch_size=64,checkpoint_interval_episodes=1)
    t=ProtocolMAPPOTrainer(c);saved=[]
    with AirCombatVectorEnv(ec,2) as v:
        def callback(trainer,vector):
            p=tmp_path/f'{trainer.training_episode_count}.pt'
            save_protocol_checkpoint(p,trainer,env_config=ec,vector=vector);saved.append(p)
        r1,e1=t.collect(v,checkpoint_callback=callback)
        assert r1.rewards.shape[0]==4 and t.sampled_steps==8 and len(e1)==4
        m1=t.update(r1);weights=copy.deepcopy(t.actor.state_dict())
        payload=read_protocol_checkpoint(saved[0],environment_version=ENVIRONMENT_VERSION)
        assert payload['training_episode_count']==1 and payload['phase']=='collecting_rollout'
        assert payload['pending_collection']['cursor']==1
        clone=ProtocolMAPPOTrainer(c);restore_protocol_checkpoint(payload,clone,env_config=ec,vector=v)
        r2,e2=clone.collect(v,checkpoint_callback=lambda *args:None)
        m2=clone.update(r2)
        tensor_equal(r1,r2);tensor_equal(e1,e2);tensor_equal(m1,m2);tensor_equal(weights,clone.actor.state_dict())


def test_checkpoint_logging_does_not_change_actions_or_parameter_updates(tmp_path):
    ec=AirCombatConfig(max_steps=2)
    c=ProtocolTrainConfig(num_envs=2,horizon=4,ppo_epochs=1,minibatch_size=64,checkpoint_interval_episodes=1)
    a=ProtocolMAPPOTrainer(c)
    with AirCombatVectorEnv(ec,2) as v:
        r1,e1=a.collect(v);m1=a.update(r1)
    b=ProtocolMAPPOTrainer(c)
    with AirCombatVectorEnv(ec,2) as v:
        def callback(t,vector):save_protocol_checkpoint(tmp_path/'logging.pt',t,env_config=ec,vector=vector)
        r2,e2=b.collect(v,checkpoint_callback=callback);m2=b.update(r2)
    tensor_equal(r1,r2);tensor_equal(e1,e2);tensor_equal(m1,m2)
    tensor_equal(a.actor.state_dict(),b.actor.state_dict());tensor_equal(a.critic.state_dict(),b.critic.state_dict())


def test_final_target_checkpoint_can_resume_last_buffer_update(tmp_path):
    ec=AirCombatConfig(max_steps=2)
    c=ProtocolTrainConfig(num_envs=1,horizon=3,ppo_epochs=1,training_episodes=1,checkpoint_interval_episodes=1)
    a=train(ec,c,tmp_path/'first')
    b=train(ec,c,tmp_path/'resumed',resume=tmp_path/'first'/'episodes_000001.pt')
    tensor_equal(a.actor.state_dict(),b.actor.state_dict())
    tensor_equal(a.critic.state_dict(),b.critic.state_dict())
    assert a.updates==b.updates==1 and b.training_episode_count==1


def test_grouped_eval_uses_fresh_independent_streams_and_repeats_exactly():
    t=ProtocolMAPPOTrainer(ProtocolTrainConfig(num_envs=1,ppo_epochs=1))
    ec=AirCombatConfig(max_steps=1)
    before=torch.cuda.get_rng_state().clone()
    a=evaluate(t,ec,[100,101],episodes_per_seed=2)
    b=evaluate(t,ec,[100,101],episodes_per_seed=2)
    assert a==b and a['completed_episodes']==4
    assert [(e['seed'],e['episode_index']) for e in a['episodes']]==[(100,0),(100,1),(101,0),(101,1)]
    torch.testing.assert_close(before,torch.cuda.get_rng_state(),rtol=0,atol=0)
    assert a['termination_counts']==dict(normal=0,timeout=4,numerical_failure=0)


def test_blue_formal_budget_and_duplicate_bundle_rejected_before_evaluation(tmp_path,monkeypatch):
    import algorithm.experiments as module
    from types import SimpleNamespace
    ec,cfg=configurations('main_5v5',1)
    model=SimpleNamespace(config=cfg)
    monkeypatch.setattr(module,'load_model',lambda path:(model,ec,dict(training_episode_count=2)))
    monkeypatch.setattr(module,'evaluate',lambda *args,**kwargs:pytest.fail('invalid bundle must not evaluate'))
    with pytest.raises(ValueError,match='completed 5000'):
        blue_sensitivity(['model.pt'],tmp_path/'probability',kind='probability')
    with pytest.raises(ValueError,match='distinct training seeds'):
        blue_sensitivity(['model.pt']*5,tmp_path/'aggregation',kind='aggregation',smoke=True)


def test_mean_sample_std_win_rate_and_strict_twenty_update_average():
    s=mean_std([.2,.4,.6,.8,1.])
    assert s['mean']==pytest.approx(.6) and s['std']==pytest.approx(np.sqrt(.1))
    assert mean_std([.5])['std'] is None
    assert mean_std([])['mean'] is None
    assert not len(moving_average(range(19)))
    np.testing.assert_allclose(moving_average(range(21)),[9.5,10.5])


def test_logs_reconstruct_figures_and_tables_without_inventing_values(tmp_path):
    # Arithmetic fixture ONLY, kept outside actual experiment output/results.
    root=tmp_path/'fixture'
    for seed in range(1,6):
        run=run_directory(root,'main_3v3',seed);run.mkdir(parents=True)
        write_json(run/'experiment.json',manifest('main_3v3',seed,smoke=True))
        evaluation=dict(training_seed=seed,completed_training_episodes=2,completed_episodes=2,
                        win_rate=.1*seed,mean_red_survivors=2.,proximity_risk=.3,shared_exposure_risk=1.)
        write_json(run/'final_evaluation.json',evaluation)
        rows=[dict(updates=i,completed_episodes=i,ppo=dict(critic_loss=i,entropy=-i),episodes=[dict(episode_return=i)]) for i in range(1,22)]
        (run/'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    summary=report(root,tmp_path/'figures')
    assert summary['fig5']['3v3']['mean']==pytest.approx(.3)
    assert summary['fig5']['3v3']['std']==pytest.approx(np.sqrt(.025))
    assert summary['fig5']['3v3']['status']=='partial_or_smoke'
    assert summary['fig5']['5v5']['mean'] is None
    for number in (5,9,10,11,12):assert (tmp_path/'figures'/f'fig{number}.png').exists()
    assert (tmp_path/'figures'/'table_IV.csv').exists()
    assert training_episodes(rows)[0]['episode']==1


def test_trace_positions_events_and_actions_are_real_replay(tmp_path):
    ec=AirCombatConfig(max_steps=3,initial_scenario='head_on_3v3')
    c=ProtocolTrainConfig(num_envs=1,horizon=1,ppo_epochs=1)
    t=ProtocolMAPPOTrainer(c)
    with AirCombatVectorEnv(ec,1) as v:save_protocol_checkpoint(tmp_path/'model.pt',t,env_config=ec,vector=v)
    trace=record(tmp_path/'model.pt','head_on_3v3',tmp_path/'trace',seed=100)
    e=AirCombatEnv(ec);obs,info=e.reset(100,{'scenario':'head_on_3v3'})
    events=[]
    for frame in trace['frames'][1:]:
        obs,r,term,trunc,info=e.step(frame['red_actions'])
        np.testing.assert_array_equal(frame['friendly_states'],info['friendly_states'])
        np.testing.assert_array_equal(frame['adversarial_states'],info['adversarial_states'])
        np.testing.assert_array_equal(frame['assignments'],info['assignments'])
        np.testing.assert_array_equal(frame['reward'],r)
        events.extend(info['events'])
    assert trace['events']==events and trace['summary']['episode_length']==3
