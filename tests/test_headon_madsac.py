"""Controlled initialization/reward tests; actual weapons and bounded CUDA resume."""
import copy
from dataclasses import replace
import json
import random
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from env.aircombat import AirCombatConfig,AirCombatEnv
from env.aircombat.scenarios import generate
from env.aircombat.madsac_reward import geometry,tier
from env.aircombat.rewards import compute_rewards
from algorithm.experiments import configurations,manifest,export_plan,run_experiment,write_json
from tests.test_aircombat_training import tensor_equal
from tests.test_experiment_a import options,TRIM

C=AirCombatConfig(initial_scenario='head_on',posture_reward_mode='madsac_inspired')


def uav(x=0.,y=0.,z=6000.,psi=0.,gamma=0.):
    return np.array([x,y,z,200.,gamma,np.radians(psi)])


def reward(red,blue,*,pre=None,assignment=None,red_alive=None,blue_alive=None,current=None,center=(0.,0.),config=C):
    red=np.atleast_2d(red).copy();blue=np.atleast_2d(blue).copy()
    n,m=len(red),len(blue)
    s=SimpleNamespace(friendly_states=red,adversarial_states=blue,
        friendly_alive=np.ones(n,bool) if red_alive is None else np.array(red_alive,bool),
        adversarial_alive=np.ones(m,bool) if blue_alive is None else np.array(blue_alive,bool),
        metadata={'madsac_battlefield_center':np.array(center)},step_count=1)
    return compute_rewards(s,np.ones(n,bool) if current is None else np.array(current,bool),
        np.zeros(n,int) if assignment is None else np.array(assignment),
        np.zeros((n,m)),np.zeros((m,n)),np.zeros((n,m)),[],config,
        pre_red=red if pre is None else np.atleast_2d(pre))


@pytest.mark.parametrize('scale',['3v3','5v5','3v5'])
@pytest.mark.parametrize('seed',[1,19,1000])
def test_head_on_changes_only_angles_keeps_seeded_positions_speed_and_rng(scale,seed):
    c=AirCombatConfig(scale=scale);a=np.random.default_rng(seed);b=np.random.default_rng(seed)
    old=generate(c,a);new=generate(replace(c,initial_scenario='head_on'),b)
    for before,after,psi in ((old.friendly_states,new.friendly_states,0.),(old.adversarial_states,new.adversarial_states,np.pi)):
        np.testing.assert_array_equal(before[:,:4],after[:,:4])
        np.testing.assert_array_equal(after[:,4],0.)
        np.testing.assert_array_equal(after[:,5],psi)
    xyz=np.concatenate((new.friendly_states[:,:3],new.adversarial_states[:,:3]))
    distances=np.linalg.norm(xyz[:,None]-xyz[None,:],axis=-1);np.fill_diagonal(distances,np.inf)
    assert distances.min()>=c.random_min_separation
    tensor_equal(a.bit_generator.state,b.bit_generator.state)
    again=generate(replace(c,initial_scenario='head_on'),np.random.default_rng(seed))
    tensor_equal(new,again)


def test_random_training_original_sampler_regression():
    c=AirCombatConfig();rng=np.random.default_rng(1000);rows=[]
    for k in range(6):
        while True:
            xyz=rng.uniform([4500 if k<3 else 9500,3800,5000],[5500 if k<3 else 10500,9600,7800])
            if not rows or np.min(np.linalg.norm(np.array(rows)[:,:3]-xyz,axis=1))>=100:break
        rows.append([*xyz,rng.uniform(175,230),rng.uniform(-.3,.3),rng.uniform(-np.pi,np.pi)])
    state=generate(c,np.random.default_rng(1000))
    np.testing.assert_array_equal(np.concatenate((state.friendly_states,state.adversarial_states)),rows)


@pytest.mark.parametrize('target,expected',[
    (uav(3000), (0,0,0)),
    (uav(3000,psi=180),(0,180,0)),
    (uav(0,3000,psi=90),(90,0,0)),
    (uav(-3000),(180,180,0)),
    (uav(1000,z=7000),(0,0,45)),
    (uav(1000,z=5000),(0,0,45)),
])
def test_li_figure2_horizontal_angles_and_height(target,expected):
    g=geometry(uav(),target)
    np.testing.assert_allclose([g['ATA'],g['AA'],g['HA']],expected,atol=1e-12,rtol=0)
    reverse=geometry(target,uav())
    assert reverse['ATA']==pytest.approx(180-g['AA'])
    assert reverse['AA']==pytest.approx(180-g['ATA'])
    assert reverse['HA']==g['HA']


def test_li_angles_do_not_reuse_xu_3d_attack_angles():
    a=uav(gamma=.6);b=uav(3000,z=6500,gamma=-.4)
    g=geometry(a,b)
    assert g['ATA']==g['AA']==0 and g['HA']==pytest.approx(np.degrees(np.arctan2(500,3000)))
    from env.aircombat.geometry import relative_geometry
    assert relative_geometry(a[None],b[None]).alpha_i[0,0]!=0
    assert geometry(uav(gamma=2.),b)['ATA']==180  # actual horizontal direction reverses


@pytest.mark.parametrize('a,b',[(uav(),uav()),(uav(),uav(z=7000)),(uav(gamma=np.pi/2),uav(3000))])
def test_degenerate_geometry_is_finite_and_cannot_earn_process_rewards(a,b):
    assert not geometry(a,b)['valid']
    assert all(np.isfinite(v) for v in geometry(a,b).values())
    total,parts=reward(a,b)
    assert total[0]==0 and all(v[0]==0 for v in parts.values())


def test_vertical_target_only_invalidates_aspect_not_own_guidance():
    a,b=uav(),uav(5000,gamma=np.pi/2)
    g=geometry(a,b);assert g['valid'] and not g['AA_valid']
    r,p=reward(a,b);assert p['R3'][0]==.001 and p['R4'][0]==0 and r[0]==.001


def pair(distance,ata=0.,ha=0.,aa=0.):
    # Horizontal LOS lies on +x; HA independently controls LOS elevation.
    rad=np.radians(ha)
    return uav(psi=ata),uav(distance*np.cos(rad),z=6000+distance*np.sin(rad),psi=aa)


@pytest.mark.parametrize('distance,expected',[(3999.999,0.),(4000.,.001),(4000.001,.001)])
@pytest.mark.parametrize('angle_key',['ata','ha'])
@pytest.mark.parametrize('angle,enabled',[(29.999999,True),(30.,True),(30.000001,False)])
def test_r3_inclusive_range_and_angle_boundaries(distance,expected,angle_key,angle,enabled):
    a,b=pair(distance,**{angle_key:angle})
    _,p=reward(a,b)
    assert p['R3'][0]==(expected if enabled else 0.)


@pytest.mark.parametrize('angle,value',[(0,.1),(4.999999,.1),(5.,.1),(5.000001,.02),
    (14.999999,.02),(15.,.02),(15.000001,.01),(29.999999,.01),(30.,.01),(30.000001,0.)])
@pytest.mark.parametrize('angle_key',['ata','ha'])
def test_r41_exact_tiers_highest_only(angle,value,angle_key):
    a,b=pair(3000,**{angle_key:angle});r,p=reward(a,b)
    assert p['R4'][0]==value and r[0]==value
    assert set(p)=={'R1','R2','R3','R4'}


@pytest.mark.parametrize('aa,value',[(29.999999,.1),(30.,.1),(30.000001,0.),(180.,0.)])
def test_r41_aspect_gate_and_head_on_exclusion(aa,value):
    a,b=pair(3000,aa=aa);assert reward(a,b)[1]['R4'][0]==value


@pytest.mark.parametrize('angle,value',[(5.,-.15),(5.000001,-.025),(15.,-.025),(15.000001,-.015),(30.,-.015),(30.000001,0.)])
def test_r42_blue_centered_reverse_los_and_defense_tiers(angle,value):
    r,p=reward(uav(),uav(-3000,psi=angle),assignment=[-1])
    assert p['R4'][0]==value and r[0]==value


def test_r42_all_blue_worst_only_and_both_offense_and_defense():
    blue=np.array([uav(3000),uav(-3000,psi=15),uav(-2500,psi=5)])
    total,p=reward(uav(),blue)
    assert p['R4'][0]==pytest.approx(.1-.15) and total[0]==p['R4'][0]
    assert reward(uav(),blue,assignment=[-1])[1]['R4'][0]==-.15
    assert reward(uav(),blue,blue_alive=[1,1,0])[1]['R4'][0]==pytest.approx(.1-.025)
    assert reward(uav(),blue,assignment=[2],blue_alive=[1,1,0])[1]['R4'][0]==-.025
    assert reward(uav(),blue,red_alive=[0])[0][0]==0
    assert reward(uav(),blue,current=[0])[0][0]==0


def test_4000_boundary_r3_plus_r4_and_weapon_range_stays_2000():
    a,b=pair(4000);r,p=reward(a,b)
    assert p['R3'][0]==.001 and p['R4'][0]==.1 and r[0]==.101
    from env.aircombat.geometry import pair_advantages
    assert pair_advantages(a[None],b[None],C)[1][0,0]==0
    assert C.tau_max==2000 and C.alpha_max==np.pi/4 and C.attack_dwell_steps==2


def test_r2_crossings_once_per_exit_reentry_and_no_death_penalty():
    # Circle is reward-only; no z constraint or outside→outside penalty.
    c=replace(C,madsac_battlefield_radius=100)
    previous=uav(0);observed=[]
    for x in (100,101,150,99,101):
        current=uav(x,z=100000)
        observed.append(reward(current,uav(20000),pre=previous,assignment=[-1],config=c)[1]['R2'][0])
        previous=current
    assert observed==[0,-10,0,0,-10]
    assert reward(uav(101),uav(20000),pre=uav(99),red_alive=[0],config=c)[0][0]==0


@pytest.mark.parametrize('case,expected',[('A1',10.),('A2',0.),('A3',20.),('death',-10.)])
def test_r1_real_blue_two_step_weapon_events_kill_death_and_mutual(case,expected):
    opts=options('A1' if case=='death' else case)
    if case=='death':opts['adversarial_states'][0,0]=-1000
    e=AirCombatEnv(C);e.reset(100,opts)
    first=e.step(TRIM);second=e.step(TRIM)
    assert first[4]['reward_components']['R1'][0]==0 and not first[4]['events']
    info=second[4];assert info['events'] and info['reward_components']['R1'][0]==expected
    assert all(x['step']==2 for x in info['events'])
    if case=='A2':assert {x['attacker_side'] for x in info['events']}=={'Red','Blue'}
    assert all(not np.any(v[1:]) for v in info['reward_components'].values())
    np.testing.assert_array_equal(second[1],sum(info['reward_components'].values()))
    # Dead Red/target get no post-motion process reward, even on the kill step.
    for key in ('R2','R3','R4'):assert info['reward_components'][key][0]==0
    if case in ('A1','death'):assert second[2] and not second[3]


def test_reward_switch_keeps_real_aph_blue_weapons_physics_observations_and_rng():
    a=AirCombatEnv(replace(C,posture_reward_mode='paper'));b=AirCombatEnv(C)
    a.reset(100,options('A2'));b.reset(100,options('A2'))
    for _ in range(2):
        x=a.step(TRIM);y=b.step(TRIM)
        tensor_equal((x[0],x[2:4]),(y[0],y[2:4]))
        for key in ('H_a','H_b','F','events','assignments','transition_assignments','friendly_states',
                    'adversarial_states','friendly_alive','adversarial_alive','blue_controls','centralized_state'):
            tensor_equal(x[4][key],y[4][key])
        tensor_equal(a.rng.bit_generator.state,b.rng.bit_generator.state)
        np.testing.assert_array_equal(y[1],sum(y[4]['reward_components'].values()))


@pytest.mark.parametrize('budget',[2500,5000])
def test_new_experiments_share_initial_state_hparams_and_are_excluded_from_paper(budget,tmp_path):
    base,bc=configurations('main_3v3',episodes=budget)
    a,ac=configurations('headon_xu_3v3',episodes=budget);b,cc=configurations('headon_madsac_3v3',episodes=budget)
    assert bc==ac==cc and ac.num_envs==10 and ac.horizon==1500 and ac.evaluation_episodes==200
    assert {k for k,v in base.to_dict().items() if v!=a.to_dict()[k]}=={'initial_scenario'}
    assert {k for k,v in a.to_dict().items() if v!=b.to_dict()[k]}=={'posture_reward_mode'}
    aa=generate(a,np.random.default_rng(1000));bb=generate(b,np.random.default_rng(1000))
    np.testing.assert_array_equal(aa.friendly_states,bb.friendly_states)
    np.testing.assert_array_equal(aa.adversarial_states,bb.adversarial_states)
    center=np.concatenate((aa.friendly_states[:,:2],aa.adversarial_states[:,:2])).mean(axis=0)
    np.testing.assert_array_equal(bb.metadata['madsac_battlefield_center'],center)
    from tools.report_aircombat import read_runs
    for name in ('headon_xu_3v3','headon_madsac_3v3'):
        meta=manifest(name,episodes=budget);assert not meta['paper_experiment']
        write_json(tmp_path/name/'seed_1'/'experiment.json',meta)
    assert read_runs(tmp_path)==[] and len(export_plan(tmp_path/'plan.json')['training_runs'])==37
    assert AirCombatConfig(**json.loads(json.dumps(C.to_dict())))==C


def test_new_default_roots_budget_isolation_and_no_overwrite(tmp_path,monkeypatch):
    import algorithm.experiments as module
    monkeypatch.chdir(tmp_path);calls=[]
    def fake_train(ec,cfg,path,**kwargs):
        calls.append(path);(path/'train.jsonl').write_text('preserved',encoding='utf-8')
    monkeypatch.setattr(module,'train',fake_train);monkeypatch.setattr(module,'final_evaluation',lambda *a,**kw:None)
    for name in ('main_3v3','headon_xu_3v3','headon_madsac_3v3'):
        for budget in (2500,5000):run_experiment(name,1,episodes=budget,console=False)
    assert len(set(calls))==6
    path=calls[-1];before=(path/'experiment.json').read_bytes()
    with pytest.raises(FileExistsError):run_experiment('headon_madsac_3v3',1,episodes=5000,console=False)
    assert (path/'experiment.json').read_bytes()==before and (path/'train.jsonl').read_text()=='preserved'


@pytest.mark.parametrize('kw',[
    {'madsac_battlefield_radius':0},{'madsac_battlefield_radius':float('nan')},
    {'madsac_event_reward':True},{'madsac_distance_threshold':-1},
    {'madsac_angle_limits_deg':[5,15,30]},{'madsac_aspect_limit_deg':181},
    {'madsac_offense_rewards':[.1,.02,.01]}, {'madsac_defense_penalties':[.01,.02]},
])
def test_invalid_reward_parameters_rejected(kw):
    with pytest.raises(ValueError):replace(C,**kw)


def test_environment_snapshot_strict_identity_radius_center_and_rng():
    a=AirCombatEnv(C);a.reset(1000);a.step(TRIM);p=a.snapshot()
    for wrong in (replace(C,initial_scenario='random_training'),replace(C,posture_reward_mode='paper'),
                  replace(C,posture_reward_mode='approach_gated'),replace(C,madsac_battlefield_radius=9000),
                  replace(C,madsac_guidance_reward=.002)):
        with pytest.raises(ValueError,match='configuration mismatch'):AirCombatEnv(wrong).restore(p)
    expected=a.step(TRIM);b=AirCombatEnv(C);b.restore(p)
    tensor_equal(expected,b.step(TRIM));tensor_equal(a.snapshot(),b.snapshot())


def test_cuda_ten_spawn_pending_boundary_worker_reset_and_ppo_resume_exact(tmp_path):
    from algorithm.common.aircombat_vector import AirCombatVectorEnv,combat_metrics
    from algorithm.common.checkpoint import save_protocol_checkpoint,read_protocol_checkpoint,restore_protocol_checkpoint
    from algorithm.mappo.trainer import ProtocolMAPPOTrainer
    from algorithm.mappo.config import ProtocolTrainConfig
    from env.aircombat.config import ENVIRONMENT_VERSION
    # Explicit tiny diagnostic budget; formal experiment configs stay unchanged.
    ec=replace(C,max_steps=2)
    cfg=ProtocolTrainConfig(num_envs=10,horizon=3,ppo_epochs=1,minibatch_size=31,checkpoint_interval_episodes=1)
    a=ProtocolMAPPOTrainer(cfg);saved=[]
    with AirCombatVectorEnv(ec,10) as v:
        assert len(set(v.worker_pids))==10
        def checkpoint(t,vector):
            if not saved:
                p=tmp_path/'pending.pt';save_protocol_checkpoint(p,t,env_config=ec,vector=vector);saved.append(p)
        r1,e1=a.collect(v,checkpoint_callback=checkpoint);m1=a.update(r1)
        expected=copy.deepcopy((a.actor.state_dict(),a.critic.state_dict(),a.actor_optimizer.state_dict(),
            a.critic_optimizer.state_dict(),a.rollout_metrics,v.snapshot(),random.getstate(),np.random.get_state(),
            torch.get_rng_state(),torch.cuda.get_rng_state_all()))
        p=read_protocol_checkpoint(saved[0],environment_version=ENVIRONMENT_VERSION)
        assert p['phase']=='collecting_rollout' and p['pending_collection']['cursor']==1
        assert p['env_config']['posture_reward_mode']=='madsac_inspired'
        assert p['env_config']['initial_scenario']=='head_on'
        b=ProtocolMAPPOTrainer(cfg)
        for wrong in (replace(ec,posture_reward_mode='paper'),replace(ec,posture_reward_mode='approach_gated'),
                      replace(ec,initial_scenario='random_training'),replace(ec,madsac_battlefield_radius=9000)):
            with pytest.raises(ValueError,match='incompatible aircombat'):restore_protocol_checkpoint(p,b,env_config=wrong)
        restore_protocol_checkpoint(p,b,env_config=ec,vector=v)
        r2,e2=b.collect(v);m2=b.update(r2)
        tensor_equal((r1,e1,m1),(r2,e2,m2))
        tensor_equal(expected,(b.actor.state_dict(),b.critic.state_dict(),b.actor_optimizer.state_dict(),
            b.critic_optimizer.state_dict(),b.rollout_metrics,v.snapshot(),random.getstate(),np.random.get_state(),
            torch.get_rng_state(),torch.cuda.get_rng_state_all()))
    assert r1.current_masks.bool().all() and torch.isfinite(r1.rewards).all()
    assert m1['valid_samples']==90 and m1['optimizer_steps']>0 and len(e1)==10
    assert b.rollout_metrics['numerical_failures']==0 and all(e['numerical_failures']==0 for e in e1)
    assert set(b.rollout_metrics['reward_components_mean'])=={'R1','R2','R3','R4'}
    assert set(combat_metrics(e1)['reward_components_mean'])=={'R1','R2','R3','R4'}
    for e in e1:assert e['episode_return']==pytest.approx(sum(e['reward_components_total'].values()))
