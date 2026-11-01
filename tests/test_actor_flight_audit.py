"""Only the new passive audit, not training or historical regression."""
import copy
import gzip
import json
import numpy as np
import pytest
import torch
from algorithm.experiments import configurations
from algorithm.mappo.trainer import ProtocolMAPPOTrainer
from env.aircombat import AirCombatEnv
from env.aircombat.framed_dynamics import initial_frame, frame_rhs, advance
from tools.audit_actor_flight import actor_step, scalar_rhs, motion_probe, episode


@pytest.mark.parametrize('deterministic',[True,False])
def test_hooks_preserve_actions_torch_rng_and_network_weights(deterministic):
    c, tc = configurations('headon_xu_norm_signed_3v3',1)
    trainer = ProtocolMAPPOTrainer(tc)
    env = AirCombatEnv(c); obs, info = env.reset(10010000)
    before = copy.deepcopy(trainer.actor.state_dict())
    torch.manual_seed(993)
    plain = trainer.sample_actions(obs,info['active_mask'],deterministic=deterministic)[0]
    rng = torch.cuda.get_rng_state().clone()
    torch.manual_seed(993)
    captured = actor_step(trainer,obs,info['active_mask'],deterministic)
    np.testing.assert_array_equal(captured['actual_actions'],plain.cpu().numpy())
    assert torch.equal(rng,torch.cuda.get_rng_state())
    assert all(torch.equal(value,before[key]) for key,value in trainer.actor.state_dict().items())
    scales = np.array([1,400,np.pi,np.pi,np.pi,10000,400,np.pi,np.pi,np.pi,10000,10000],np.float32)
    np.testing.assert_array_equal(captured['network_input'],obs.astype(np.float32)/scales)
    if deterministic:
        np.testing.assert_allclose(captured['actual_actions'],captured['beta_mean'],rtol=0,atol=0)
    assert np.isfinite(captured['beta_std']).all()


def test_scalar_rhs_and_probe_are_pure_and_normal_region_equivalent():
    c,_ = configurations('headon_xu_norm_signed_3v3',1)
    state = np.array([1,2,6000,217,.23,-.4]); u = np.array([.7,.2,2.1]); frame=initial_frame(state)
    following,_,_ = advance(state,u,frame,c)
    s0,f0,u0=state.copy(),frame.copy(),u.copy()
    d = motion_probe(state,following,frame,u,c)
    np.testing.assert_allclose(d['same_mu_rhs_difference'],0,rtol=0,atol=1e-14)
    np.testing.assert_allclose(d['paper_rhs'],scalar_rhs(state,u,c.gravity),atol=0,rtol=0)
    assert d['angle_reference_accepted'] and d['angle_rk4_position_error_m']<.001
    np.testing.assert_array_equal(state,s0);np.testing.assert_array_equal(frame,f0);np.testing.assert_array_equal(u,u0)


def test_trace_hooks_and_probes_do_not_change_real_episode(tmp_path):
    from dataclasses import replace
    c,tc=configurations('headon_xu_norm_signed_3v3',1)
    c=replace(c,max_steps=4)
    trainer=ProtocolMAPPOTrainer(tc)
    ep=episode(trainer,c,10010000,'head_on_3v3','untrained',tmp_path/'trace.gz',inspect=True)
    env=AirCombatEnv(c);obs,info=env.reset(10010000,{'scenario':'head_on_3v3'})
    plain=[]
    while not env.done:
        a=trainer.sample_actions(obs,info['active_mask'],deterministic=True)[0].cpu().numpy()
        obs,_,_,_,info=env.step(a);plain.append(info)
    with gzip.open(tmp_path/'trace.gz','rt') as f:rows=[json.loads(line) for line in f]
    assert ep['steps']==4
    for row,actual in zip(rows,plain):
        np.testing.assert_array_equal(row['post_red'],actual['friendly_states'])
        np.testing.assert_array_equal(row['post_blue'],actual['adversarial_states'])
        np.testing.assert_array_equal(row['red_controls'],actual['red_controls'])
        assert row['events']==actual['events']
        assert row['numerical_failure'] is False
    from tools.analyze_actor_flight import reduce_group
    g=reduce_group([ep])
    assert g['trace_steps']==4 and g['sides']['Red']['active_uav_steps']==12
    assert g['failures']==0 and g['flight']['controls_mu_deg_zeta_n']['mean'][2]==pytest.approx(
        np.mean([r['red_controls'][i][2] for r in rows for i in range(3)]))
    assert g['far_posture']['all']['sum']==pytest.approx(sum(sum(r['reward_components']['posture']) for r in rows))
    assert g['policy_chain']['physical_mapping_max_error']==0
    assert g['policy_chain']['action_components_at_trainer_clamp_bound']==0
