"""Reuse the production gate; signed/fixed-scale baseline changes one field."""
import copy
from dataclasses import replace
import json
import numpy as np
import pytest
from algorithm.experiments import configurations,manifest,export_plan,run_experiment,load_model
from algorithm.common.checkpoint import read_protocol_checkpoint,restore_protocol_checkpoint
from env.aircombat import AirCombatEnv
from env.aircombat.config import ENVIRONMENT_VERSION
from env.aircombat.geometry import pair_advantages
from env.aircombat.posture_reward import settle_posture
from tests.test_aircombat_training import tensor_equal
from tests.test_experiment_a import options,TRIM

NAME='headon_xu_norm_signed_gated_3v3'
BASE,BC=configurations('headon_xu_norm_signed_3v3',episodes=2500)
GATED,GC=configurations(NAME,episodes=2500)


def test_inherited_single_variable_formal_protocol_and_plan(tmp_path):
    for budget in (2500,5000):
        b,bc=configurations('headon_xu_norm_signed_3v3',episodes=budget)
        g,gc=configurations(NAME,episodes=budget)
        assert bc==gc and replace(g,posture_reward_mode='paper')==b
        assert gc.initialization_mode=='symmetric' and gc.observation_normalization=='fixed_scale_v1'
        assert g.observation_encoding=='signed_alpha_v1' and g.initial_scenario=='head_on' and g.posture_gate_speed==100.
        assert gc.num_envs==10 and gc.horizon==1500 and gc.evaluation_episodes==200
    assert manifest(NAME)['family']=='reward_ablation' and not manifest(NAME)['paper_experiment']
    assert len(export_plan(tmp_path/'plan.json')['training_runs'])==37


@pytest.mark.parametrize('case',['far','A3'])
def test_real_physics_blue_aph_weapon_and_four_other_rewards_unchanged(case):
    opts=options('A1' if case=='far' else case)
    if case=='far':opts['adversarial_states'][0,0]=3500
    a,b=AirCombatEnv(BASE),AirCombatEnv(GATED);a.reset(100,opts);b.reset(100,opts)
    changed=0
    for _ in range(12):
        pre_r=a.state.friendly_states.copy();pre_b=a.state.adversarial_states.copy()
        mask=a.state.friendly_alive.copy();targets=a.state.assignments.copy()
        x,y=a.step(TRIM),b.step(TRIM)
        tensor_equal(x[0],y[0]);tensor_equal(x[2:4],y[2:4])
        for key in ('friendly_states','adversarial_states','friendly_alive','adversarial_alive','blue_controls',
                    'blue_action_indices','assignments','transition_assignments','F','H_a','H_b','events'):
            tensor_equal(x[4][key],y[4][key])
        tensor_equal(a.rng.bit_generator.state,b.rng.bit_generator.state)
        _,ra,ba,_=pair_advantages(a.state.friendly_states,a.state.adversarial_states,BASE)
        _,rg,bg,_=pair_advantages(b.state.friendly_states,b.state.adversarial_states,GATED)
        tensor_equal((ra,ba),(rg,bg))
        for key in ('attack','allied','shootdown','terminal'):
            tensor_equal(x[4]['reward_components'][key],y[4]['reward_components'][key])
        for i in np.flatnonzero(mask):
            j=targets[i]
            if j<0:continue
            p=x[4]['reward_components']['posture'][i]
            los=pre_b[j,:3]-pre_r[i,:3];d0=np.linalg.norm(los)
            d1=np.linalg.norm(a.state.adversarial_states[j,:3]-a.state.friendly_states[i,:3])
            own=np.dot(a.state.friendly_states[i,:3]-pre_r[i,:3],los/d0)/BASE.dt if d0>0 else 0.
            rel=(d0-d1)/BASE.dt;gate=np.clip(min(own,rel)/100.,0,1) if d0>0 else 0.
            expected=min(p,0)+gate*max(p,0) if d1>2000 else p
            assert y[4]['reward_components']['posture'][i]==pytest.approx(expected,abs=1e-15)
        changed+=int(not np.array_equal(x[1],y[1]))
        np.testing.assert_array_equal(y[1],sum(y[4]['reward_components'].values()))
        if x[2] or x[3]:break
    if case=='far':assert changed>0
    else:assert len(x[4]['events'])==2 and x[2] and not x[3]


@pytest.mark.parametrize('speed,enemy,p,distance,expected',[
    (200,0,2,3000,2),(50,0,2,3000,1),(-50,0,2,3000,0),
    (-50,-100,2,3000,0),(50,50,2,3000,0),(-50,0,-2,3000,-2),
    (0,0,2,2000,2),(0,0,2,1999.999,2),(0,0,2,2000.001,0),
])
def test_existing_production_gate_formulas_and_closed_boundary(speed,enemy,p,distance,expected):
    r=np.array([[0,0,6000,200,0,0]],float);b=np.array([[distance,0,6000,200,0,0]],float)
    nr=r.copy();nb=b.copy();nr[0,0]+=speed*GATED.dt;nb[0,0]+=enemy*GATED.dt
    value,_=settle_posture(np.array([[p]]),np.array([True]),np.array([0]),r,b,nr,nb,np.array([True]),GATED)
    assert value[0]==expected


def test_new_default_roots_and_existing_output_protection(tmp_path,monkeypatch):
    import algorithm.experiments as module
    monkeypatch.chdir(tmp_path);paths=[]
    def fake_train(ec,cfg,path,**kwargs):
        paths.append(path);(path/'train.jsonl').write_text('preserved',encoding='utf-8')
    monkeypatch.setattr(module,'train',fake_train);monkeypatch.setattr(module,'final_evaluation',lambda *a,**kw:None)
    for budget in (2500,5000):module.run_experiment(NAME,1,episodes=budget,console=False)
    assert str(paths[0]).startswith('outputs/headon_norm_signed_gated_2500/')
    assert paths[0]!=paths[1]
    before=(paths[0]/'experiment.json').read_bytes()
    with pytest.raises(FileExistsError):module.run_experiment(NAME,1,episodes=2500,console=False)
    assert (paths[0]/'train.jsonl').read_text()=='preserved' and (paths[0]/'experiment.json').read_bytes()==before


def test_cuda_real_experiment_smoke_loading_evaluation_and_strict_resume(tmp_path):
    # Actual entry workflow: existing --smoke budget, unchanged 250-step environment.
    t=run_experiment(NAME,1,tmp_path/'smoke',smoke=True,console=False)
    path=tmp_path/'smoke'/NAME/'seed_1'
    clone,env,payload=load_model(path/'latest.pt')
    tensor_equal(t.actor.state_dict(),clone.actor.state_dict())
    assert env==GATED and t.training_episode_count==2 and t.updates>0
    report=json.loads((path/'final_evaluation.json').read_text())
    assert report['completed_episodes']==2 and report['numerical_failures']==0
    before=copy.deepcopy(clone.actor.state_dict())
    with pytest.raises(ValueError,match='incompatible aircombat'):
        restore_protocol_checkpoint(payload,clone,env_config=BASE)
    tensor_equal(before,clone.actor.state_dict())
    for src,dst in ((BASE,GATED),(GATED,BASE)):
        e=AirCombatEnv(src);e.reset(1000)
        with pytest.raises(ValueError,match='configuration mismatch'):AirCombatEnv(dst).restore(e.snapshot())
    with pytest.raises(FileExistsError):run_experiment(NAME,1,tmp_path/'smoke',smoke=True,console=False)
    rows=[json.loads(s) for s in (path/'train.jsonl').read_text().splitlines()]
    assert all(r['rollout']['numerical_failures']==0 and r['ppo']['valid_samples']>0 for r in rows)
    assert all(np.isfinite(r['ppo'][k]) for r in rows for k in ('actor_total_loss','critic_loss','entropy'))
    result=dict(workers=t.config.num_envs,max_episode_steps=env.max_steps,horizon=t.config.horizon,
        training_episodes=t.training_episode_count,joint_steps=t.sampled_steps,updates=t.updates,
        evaluation_episodes=2,evaluation_win_rate=report['win_rate'],numerical_failures=0,
        valid_ppo_samples=sum(r['ppo']['valid_samples'] for r in rows),
        last_tanh=rows[-1]['ppo']['numerical_diagnostics']['tanh'],strict_resume_and_output_protection=True)
    (tmp_path/'acceptance.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('SIGNED_GATED_ACCEPTANCE '+json.dumps(result))
