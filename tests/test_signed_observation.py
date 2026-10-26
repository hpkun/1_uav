"""Small signed-alpha ablation checks; physics and Xu geometry stay unsigned."""
import copy
from dataclasses import replace
import json
import numpy as np
import pytest
from env.aircombat import AirCombatEnv,AirCombatConfig
from env.aircombat.observations import observe,direction_sign
from env.aircombat.geometry import relative_geometry
from algorithm.experiments import configurations,manifest,export_plan,load_model
from algorithm.aircombat import train,evaluate
from algorithm.common.checkpoint import restore_protocol_checkpoint,read_protocol_checkpoint
from env.aircombat.config import ENVIRONMENT_VERSION
from tests.test_aircombat_training import tensor_equal
from tests.test_experiment_a import options,TRIM

BASE,TC=configurations('headon_xu_norm_3v3',episodes=2500)
SIGNED,SC=configurations('headon_xu_norm_signed_3v3',episodes=2500)


def test_mirrored_signed_angle_and_degenerate_magnitudes_slots():
    results=[]
    for y in (600.,-600.):
        opts=options('A1');opts['adversarial_states'][0,:3]=[1500,y,6200]
        e=AirCombatEnv(SIGNED);o,info=e.reset(7,opts);results.append(o[0,4])
        assert o.shape==(3,12) and info['centralized_state'].shape==(60,)
        np.testing.assert_array_equal(info['centralized_state'].reshape(5,12)[:3],o)
        np.testing.assert_array_equal(o[1:],0)
        np.testing.assert_array_equal(info['centralized_state'].reshape(5,12)[3:],0)
    assert results[0]>0 and results[0]==-results[1]
    assert results[0]==pytest.approx(np.arctan2(np.hypot(600,200),1500))
    for target,gamma in (([0,0,6000],0),([0,0,7000],0),([-1000,0,6000],0),
                         ([1000,-1e-11,6000],0),([1000,-100,6000],np.pi/2)):
        opts=options('A1');opts['adversarial_states'][0,:3]=target;opts['friendly_states'][0,4]=gamma
        e=AirCombatEnv(SIGNED);o,info=e.reset(7,opts)
        raw=relative_geometry(e.state.friendly_states,e.state.adversarial_states).alpha_i[0,0]
        assert direction_sign(e.state.friendly_states[0],e.state.adversarial_states[0])==1.
        assert o[0,4]==raw and np.isfinite(o).all()


@pytest.mark.parametrize('opts',[None,options('A2')])
def test_only_observation_sign_changes_physics_aph_rewards_events_and_rng(opts):
    a,b=AirCombatEnv(BASE),AirCombatEnv(SIGNED);a.reset(100,opts);b.reset(100,opts)
    for _ in range(25):
        x=a.step(TRIM);y=b.step(TRIM)
        ox=x[0].copy();oy=y[0].copy();oy[:,4]=abs(oy[:,4]);np.testing.assert_array_equal(ox,oy)
        tensor_equal(x[1:4],y[1:4])
        for key in ('F','H_a','H_b','events','reward_components','assignments','transition_assignments',
                    'friendly_states','adversarial_states','blue_controls','friendly_alive','adversarial_alive'):
            tensor_equal(x[4][key],y[4][key])
        tensor_equal(a.rng.bit_generator.state,b.rng.bit_generator.state)
        np.testing.assert_array_equal(a.state.friendly_frames,b.state.friendly_frames)
        # Reconstruct the original unsigned observation definition directly.
        g=relative_geometry(a.state.friendly_states,a.state.adversarial_states)
        expected=np.zeros((3,12))
        for i in np.flatnonzero(a.state.friendly_alive):
            r=a.state.friendly_states[i];expected[i,[0,1,2,3,5]]=[1,*r[3:6],r[2]]
            j=a.state.assignments[i]
            if j>=0:expected[i,4:]=[g.alpha_i[i,j],r[2],*a.state.adversarial_states[j,3:6],g.alpha_j[i,j],a.state.adversarial_states[j,2],g.distance[i,j]]
        np.testing.assert_array_equal(x[0],expected)
        if x[2] or x[3]:break
    if opts is not None:assert x[4]['events']  # actual simultaneous Red/Blue kills


def test_single_variable_identity_snapshot_and_paper_exclusion(tmp_path):
    assert SC==TC and replace(SIGNED,observation_encoding='unsigned')==BASE
    assert TC.num_envs==10 and TC.horizon==1500 and TC.evaluation_episodes==200
    assert TC.observation_normalization=='fixed_scale_v1'
    assert not manifest('headon_xu_norm_signed_3v3')['paper_experiment']
    assert len(export_plan(tmp_path/'plan.json')['training_runs'])==37
    with pytest.raises(ValueError):replace(BASE,observation_encoding='unknown')
    for src,dst in ((BASE,SIGNED),(SIGNED,BASE)):
        e=AirCombatEnv(src);e.reset(1000)
        with pytest.raises(ValueError,match='configuration mismatch'):AirCombatEnv(dst).restore(e.snapshot())


def test_cuda_short_train_load_evaluate_and_strict_checkpoint_identity(tmp_path):
    ec=replace(SIGNED,max_steps=2)
    tc=replace(SC,training_episodes=2,num_envs=10,horizon=2,ppo_epochs=1,
               evaluation_episodes=2,checkpoint_interval_episodes=1)
    t=train(ec,tc,tmp_path/'run',console=False)
    clone,loaded,payload=load_model(tmp_path/'run/latest.pt')
    assert loaded==ec and payload['env_config']['observation_encoding']=='signed_alpha_v1'
    tensor_equal(t.actor.state_dict(),clone.actor.state_dict())
    report=evaluate(clone,loaded,[10010000,10010001])
    assert report['completed_episodes']==2 and report['numerical_failures']==0
    original=copy.deepcopy(clone.actor.state_dict())
    with pytest.raises(ValueError,match='incompatible aircombat'):
        restore_protocol_checkpoint(payload,clone,env_config=replace(ec,observation_encoding='unsigned'))
    # A real unsigned historical-style config is also rejected for signed mode.
    unsigned=copy.deepcopy(payload);unsigned['env_config'].pop('observation_encoding')
    with pytest.raises(ValueError,match='incompatible aircombat'):restore_protocol_checkpoint(unsigned,clone,env_config=ec)
    tensor_equal(original,clone.actor.state_dict())
    row=json.loads((tmp_path/'run/train.jsonl').read_text().splitlines()[-1])
    assert row['ppo']['valid_samples']==36 and row['rollout']['numerical_failures']==0
    assert all(np.isfinite(row['ppo'][k]) for k in ('actor_total_loss','critic_loss','entropy'))
    result=dict(workers=10,joint_steps=t.sampled_steps,updates=t.updates,episodes=t.training_episode_count,
        valid_samples=row['ppo']['valid_samples'],evaluation_episodes=2,numerical_failures=0,
        encoding=loaded.observation_encoding,load_exact=True,cross_mode_restore_rejected=True)
    (tmp_path/'acceptance.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('SIGNED_ACCEPTANCE '+json.dumps(result))
