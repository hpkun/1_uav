"""Mathematical and transaction tests for the explicitly assumed v2 continuation."""
import copy
from dataclasses import replace
from itertools import product
import math
import numpy as np
import pytest
from env.aircombat import AirCombatEnv, AirCombatConfig
from env.aircombat.framed_dynamics import initial_frame, frame_rhs, step, advance, frame_error, SimulationFault
from env.paper_dynamics import rhs, rk4_step
from env.dynamics import velocity_vector
from env.aircombat.opponent import candidate_scores, BLUE_CONTROLS
from env.aircombat.geometry import pair_advantages


C=AirCombatConfig()


def test_subnormal_horizontal_direction_has_no_overflowing_intermediate():
    f=np.array([[1e-320,-1.,0.],[0.,0.,1.],[1.,1e-320,0.]])
    value=np.r_[0.,0.,6000.,200.,f.ravel()]
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        derivative=frame_rhs(value,[.7,1.,8.],C)
    assert np.isfinite(derivative).all()
    # Exactly zero horizontal direction follows the same finite limiting rule.
    f[0,0]=f[2,1]=0.
    zero=frame_rhs(np.r_[value[:4],f.ravel()],[.7,1.,8.],C)
    np.testing.assert_allclose(derivative,zero,atol=1e-300,rtol=0.)


def test_random_normal_region_differential_equivalence_and_full_control_corners():
    rng=np.random.default_rng(29)
    controls=list(product([-math.pi,0,math.pi],[-1,2],[0,8]))
    controls += [rng.uniform([-math.pi,-1,0],[math.pi,2,8]) for _ in range(150)]
    for control in controls:
        y=np.r_[rng.normal(size=3),rng.uniform(100,400),rng.uniform(-1.2,1.2),rng.uniform(-math.pi,math.pi)]
        frame=initial_frame(y); d=rhs(y,control,gravity=C.gravity)
        actual=frame_rhs(np.r_[y[:4],frame.ravel()],control,C)
        u,a,b=frame.T
        expected_du=d[4]*a+math.cos(y[4])*d[5]*b
        np.testing.assert_allclose(actual[:4],d[:4],atol=3e-13,rtol=3e-13)
        np.testing.assert_allclose(actual[4:].reshape(3,3)[:,0],expected_du,atol=3e-14,rtol=3e-13)
        expected_acceleration=d[3]*u+y[3]*expected_du
        np.testing.assert_allclose(actual[3]*u+y[3]*actual[4:].reshape(3,3)[:,0],expected_acceleration,atol=3e-12)
        assert np.max(abs(frame.T@actual[4:].reshape(3,3)+(frame.T@actual[4:].reshape(3,3)).T))<1e-12


def normal_trajectory_error(config=C):
    old=np.array([0.,0.,6000.,220.,.1,.3]); new=old.copy(); frame=initial_frame(new)
    errors=np.zeros(4)
    for i in range(80):
        control=np.array([.35*math.sin(i/20),.3,1.3])
        reference=rk4_step(old,control,dt=config.dt,gravity=config.gravity)
        assert reference.accepted
        old=reference.state
        new,frame,_=step(new,control,frame,config)
        errors=np.maximum(errors,[np.linalg.norm(new[:3]-old[:3]),abs(new[3]-old[3]),abs(new[4]-old[4]),abs(new[5]-old[5])])
    return errors


def test_multistep_normal_trajectory_and_step_refinement():
    errors=normal_trajectory_error()
    assert errors[0]<.01 and errors[1]<1e-4 and max(errors[2:])<1e-5
    fine=normal_trajectory_error(replace(C,dt=.125))
    assert fine[0]<errors[0] and max(fine[2:])<max(errors[2:])


@pytest.mark.parametrize('sign,mu,zeta,normal',product([-1,1],[-math.pi,-.7,0,.7,math.pi],[-1,2],[0,1,8]))
def test_poles_full_control_boundaries_finite_and_continuous(sign,mu,zeta,normal):
    y=np.array([0.,0.,6000.,200.,sign*math.pi/2,.4]); frame=initial_frame(y)
    previous=velocity_vector(y)
    for _ in range(8):
        result,frame,d=step(y,[mu,zeta,normal],frame,C)
        assert np.isfinite(result).all() and frame_error(frame)<2e-14
        assert np.linalg.norm(result[:3]-y[:3])<=C.dt*C.v_max*1.01
        assert np.linalg.norm(velocity_vector(result)-previous)<C.dt*C.gravity*(abs(zeta)+normal+2)*1.05
        np.testing.assert_allclose(velocity_vector(result)/result[3],frame[:,0],atol=1e-13)
        previous=velocity_vector(result);y=result


@pytest.mark.parametrize('sign,offset,mu',product([-1,1],[-1e-3,-1e-8,0.,1e-8,1e-3],[0.,.7]))
def test_both_sides_of_pole_and_bounded_frame_spin(sign,offset,mu):
    y=np.array([0.,0.,6000.,200.,sign*math.pi/2+offset,.4]);f=initial_frame(y)
    result,f,d=step(y,[mu,2,8],f,C)
    lower_speed=y[3]-C.dt*C.gravity*3
    bound=1.125*C.gravity*9/(lower_speed*C.pole_cap_ratio)+C.frame_alignment_rate/2
    assert d['max_frame_spin_rate']<=1.2*bound
    assert np.isfinite(result).all() and d['continuation_substeps']>0
    assert np.linalg.norm(result[:3]-y[:3])<C.dt*C.v_max


@pytest.mark.parametrize('sign',[-1,1])
def test_meridional_pole_crossing_keeps_reference_heading(sign):
    y=np.array([0.,0.,6000.,200.,sign*(math.pi/2-.01),.4]);f=initial_frame(y)
    control=[0. if sign==1 else math.pi,1,8]
    assert not rk4_step(y,control,dt=C.dt,gravity=C.gravity).accepted
    result,f,d=step(y,control,f,C)
    assert sign*result[4]>math.pi/2
    assert abs(result[5]-y[5])<1e-12
    assert np.linalg.norm(velocity_vector(result)-velocity_vector(y))<25


def test_post_continuation_gauge_relaxes_without_instant_axis_reset():
    y=np.array([0.,0.,6000.,200.,.2,.4]);f=initial_frame(y)
    a,b=f[:,1].copy(),f[:,2].copy()
    f[:,1]=math.cos(.4)*a+math.sin(.4)*b
    f[:,2]=-math.sin(.4)*a+math.cos(.4)*b
    before=f.copy()
    # The reference equation does not define this extra gauge state. It must
    # remain continuous and relax, rather than being silently discarded.
    y,f,d=step(y,[0.,0.,0.],f,C)
    assert 0<d['max_reference_axis_error']<=.40001
    assert np.linalg.norm(f[:,1]-before[:,1])<.4
    for _ in range(12):y,f,d=step(y,[0.,0.,0.],f,C)
    assert d['max_reference_axis_error']<.002


@pytest.mark.parametrize('kind',['random','sustained'])
def test_long_control_stress(kind):
    rng=np.random.default_rng(447);y=np.array([0.,0.,6000.,200.,1.45,.3]);frame=initial_frame(y)
    continuations=0
    for _ in range(1500):
        control=rng.uniform([-math.pi,-1,0],[math.pi,2,8]) if kind=='random' else [0,1,8]
        y,frame,d=step(y,control,frame,C)
        continuations+=d['continuation_substeps']
        assert np.isfinite(y).all() and frame_error(frame)<2e-14
    if kind=='sustained': assert continuations>0


def test_blue_virtual_predictions_use_actual_frame_without_mutation():
    e=AirCombatEnv();e.reset(11)
    y=e.state.adversarial_states[0].copy(); y[4]=math.pi/2
    frame=initial_frame(y)
    # Create a genuine non-canonical continued frame.
    y,frame,_=step(y,[.7,0,8],frame,C)
    red=e.state.friendly_states.copy(); before=copy.deepcopy((y,red,frame,e.rng.bit_generator.state,e.counters))
    scores=candidate_scores(y,red,C,frame)
    for i,control in enumerate(BLUE_CONTROLS):
        candidate,_,_=advance(y,control,frame,C)
        geom,offense,defense,_=pair_advantages(candidate[None],red,C)
        expected=np.sum((candidate[3]-C.v_min)/(C.v_max-C.v_min)*geom.cos_alpha_i+offense-defense.T)
        assert scores[i]==expected
    for a,b in zip((y,red,frame),before[:3]):np.testing.assert_array_equal(a,b)
    assert e.rng.bit_generator.state==before[3] and e.counters==before[4]


def test_fault_is_transactional_no_transition_reward_kill_or_checkpoint():
    e=AirCombatEnv();e.reset(11);e.state.friendly_frames[0,0,0]=np.nan
    before=copy.deepcopy(e.state);rng=copy.deepcopy(e.rng.bit_generator.state)
    with pytest.raises(SimulationFault,match='joint simulation transaction failed'):
        e.step(np.full((3,3),.5))
    assert e.state.step_count==0 and not e.state.events
    np.testing.assert_array_equal(e.state.friendly_states,before.friendly_states)
    np.testing.assert_array_equal(e.state.adversarial_alive,before.adversarial_alive)
    assert e.rng.bit_generator.state==rng
    with pytest.raises(SimulationFault,match='cannot checkpoint'):e.snapshot()
    with pytest.raises(RuntimeError,match='reset required'):e.step(np.full((3,3),.5))


def test_reference_mode_fail_fast_and_cross_version_snapshot_rejection():
    e=AirCombatEnv(replace(C,integrator='angle_rk4'));e.reset(11)
    e.state.friendly_states[0,4]=math.pi/2
    with pytest.raises(SimulationFault):e.step(np.full((3,3),.5))
    e=AirCombatEnv();e.reset(11);payload=e.snapshot();payload['environment_version']='xu-aircombat-v1.0.0'
    with pytest.raises(ValueError,match='snapshot configuration'):e.restore(payload)


def test_polar_snapshot_exact_following_motion_and_dead_frame_freeze():
    a=AirCombatEnv();a.reset(7)
    a.state.friendly_states[0,4]=math.pi/2
    a.state.friendly_frames[0]=initial_frame(a.state.friendly_states[0])
    a.state.friendly_alive[2]=False
    a._assign()
    b=AirCombatEnv();b.restore(a.snapshot())
    frozen=a.state.friendly_frames[2].copy()
    for _ in range(4):
        x,y=a.step(np.full((3,3),.5)),b.step(np.full((3,3),.5))
        for i in range(4):np.testing.assert_array_equal(x[i],y[i])
        np.testing.assert_array_equal(a.state.friendly_frames,b.state.friendly_frames)
        np.testing.assert_array_equal(a.state.friendly_frames[2],frozen)


def test_worker_fault_cannot_complete_episode_or_supply_ppo_samples():
    from algorithm.common.aircombat_vector import AirCombatVectorEnv
    from algorithm.mappo.trainer import ProtocolMAPPOTrainer
    from algorithm.mappo.config import ProtocolTrainConfig
    trainer=ProtocolMAPPOTrainer(ProtocolTrainConfig(num_envs=1,horizon=2,ppo_epochs=1))
    with AirCombatVectorEnv(replace(C,integrator='angle_rk4'),1) as vector:
        snapshot=vector.snapshot()
        state=snapshot['workers'][0]['state']
        state.friendly_states[0,4]=math.pi/2
        state.friendly_frames[0]=initial_frame(state.friendly_states[0])
        vector.restore(snapshot)
        with pytest.raises(RuntimeError,match='SimulationFault'):trainer.collect(vector)
        assert trainer.updates==trainer.sampled_steps==trainer.training_episode_count==0
        assert vector.episodes.sum()==0 and not trainer.pending_collection['rows']
        assert trainer.pending_collection['sampled']==0 and not trainer.pending_collection['episodes']
        assert vector.trackers[0].steps==0


def test_continuation_config_and_old_checkpoint_cannot_resume_new_physics(tmp_path):
    from algorithm.common.aircombat_vector import AirCombatVectorEnv
    from algorithm.mappo.trainer import ProtocolMAPPOTrainer
    from algorithm.mappo.config import ProtocolTrainConfig
    from algorithm.common.checkpoint import save_protocol_checkpoint,read_protocol_checkpoint,restore_protocol_checkpoint
    from env.aircombat.config import ENVIRONMENT_VERSION
    trainer=ProtocolMAPPOTrainer(ProtocolTrainConfig(num_envs=1,horizon=1,ppo_epochs=1))
    with AirCombatVectorEnv(C,1) as vector:
        path=tmp_path/'new.pt';save_protocol_checkpoint(path,trainer,env_config=C,vector=vector)
        payload=read_protocol_checkpoint(path,environment_version=ENVIRONMENT_VERSION)
        assert payload['env_config']['integrator']=='frame_rk4'
        for cfg in (replace(C,integrator='angle_rk4'),replace(C,pole_cap_ratio=.2)):
            with pytest.raises(ValueError,match='incompatible aircombat'):restore_protocol_checkpoint(payload,trainer,env_config=cfg)
        payload['environment_version']='xu-aircombat-v1.0.0'
        with pytest.raises(ValueError,match='incompatible aircombat'):restore_protocol_checkpoint(payload,trainer,env_config=C)


def test_reports_reject_mixing_historical_and_new_dynamics(tmp_path):
    import json
    from tools.report_aircombat import read_runs
    for seed in (1,2):
        d=tmp_path/'main_3v3'/f'seed_{seed}';d.mkdir(parents=True)
        meta=dict(experiment='main_3v3',training_seed=seed,environment=C.to_dict())
        if seed==1:meta['environment_version']='xu-aircombat-v2.0.0'
        (d/'experiment.json').write_text(json.dumps(meta),encoding='utf-8')
    with pytest.raises(ValueError,match='cannot mix environment versions'):read_runs(tmp_path)


def test_polar_half_rollout_checkpoint_restores_actor_rng_and_ppo_exactly(tmp_path):
    import random
    import torch
    from algorithm.common.aircombat_vector import AirCombatVectorEnv
    from algorithm.mappo.trainer import ProtocolMAPPOTrainer
    from algorithm.mappo.config import ProtocolTrainConfig
    from algorithm.common.checkpoint import save_protocol_checkpoint,read_protocol_checkpoint,restore_protocol_checkpoint
    from env.aircombat.config import ENVIRONMENT_VERSION
    from env.aircombat.observations import observe
    from tests.test_aircombat_training import tensor_equal
    ec=replace(C,max_steps=4)
    cfg=ProtocolTrainConfig(num_envs=2,horizon=4,ppo_epochs=1,minibatch_size=13,checkpoint_interval_episodes=1)
    trainer=ProtocolMAPPOTrainer(cfg)
    with AirCombatVectorEnv(ec,2) as vector:
        snapshot=vector.snapshot()
        for i,worker in enumerate(snapshot['workers']):
            s=worker['state'];s.friendly_states[0,4]=math.pi/2-1e-5
            s.friendly_frames[0]=initial_frame(s.friendly_states[0])
            worker['diagnostics']['min_horizontal_ratio']=1e-5
            fixture=AirCombatEnv(ec);fixture.restore(worker)
            obs,central=observe(fixture.state,ec);info=fixture._info(central,[],False)
            snapshot['workers'][i]=fixture.snapshot()
            snapshot['observations'][i]=obs;snapshot['states'][i]=central
            snapshot['trackers'][i].reset(info)
        vector.restore(snapshot)
        saved=[]
        def checkpoint(t,v):
            if not saved:
                p=tmp_path/'polar_pending.pt';save_protocol_checkpoint(p,t,env_config=ec,vector=v);saved.append(p)
        r1,e1=trainer.collect(vector,checkpoint_callback=checkpoint);m1=trainer.update(r1)
        expected=copy.deepcopy((trainer.actor.state_dict(),trainer.critic.state_dict(),trainer.actor_optimizer.state_dict(),
            trainer.critic_optimizer.state_dict(),vector.snapshot(),random.getstate(),np.random.get_state(),
            torch.get_rng_state(),torch.cuda.get_rng_state_all()))
        payload=read_protocol_checkpoint(saved[0],environment_version=ENVIRONMENT_VERSION)
        assert payload['phase']=='collecting_rollout' and payload['pending_collection']['cursor']==1
        assert payload['vector']['workers'][1]['diagnostics']['continuation_substeps']>0
        clone=ProtocolMAPPOTrainer(cfg);restore_protocol_checkpoint(payload,clone,env_config=ec,vector=vector)
        r2,e2=clone.collect(vector);m2=clone.update(r2)
        tensor_equal((r1,e1,m1),(r2,e2,m2))
        tensor_equal(expected,(clone.actor.state_dict(),clone.critic.state_dict(),clone.actor_optimizer.state_dict(),
            clone.critic_optimizer.state_dict(),vector.snapshot(),random.getstate(),np.random.get_state(),
            torch.get_rng_state(),torch.cuda.get_rng_state_all()))
