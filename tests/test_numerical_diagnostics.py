import copy
import random

import numpy as np
import pytest
import torch

from algorithm.mappo.config import ProtocolTrainConfig
from algorithm.mappo.diagnostics import diagnose_actor, diagnose_critic, isolated_rng
from algorithm.mappo.trainer import MAPPOTrainer, ProtocolMAPPOTrainer
from tools.diagnose_numerics import synthetic_physical_samples
from tools.numerical_cases import hand_batch, ppo_boundaries, reward_boundaries


def assert_tree_equal(a,b):
    if isinstance(a,torch.Tensor): torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,np.ndarray): np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: assert_tree_equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b): assert_tree_equal(x,y)
    else: assert a==b


def rng_state():
    return random.getstate(),np.random.get_state(),torch.get_rng_state().clone(),torch.cuda.get_rng_state_all()


def test_readonly_diagnostics_preserve_weights_gradients_optimizers_rng_and_forward():
    t=ProtocolMAPPOTrainer(ProtocolTrainConfig(ppo_epochs=1))
    t.update(hand_batch((2,1,0,0,0),t.device),diagnostics=False)
    a,c=synthetic_physical_samples(); a,c=a[:10],c[:2]
    with torch.no_grad():
        before=t.actor.distribution(t.tensor(a)).mean.clone(); values=t.critic(t.tensor(c)).clone()
    state=copy.deepcopy((t.actor.state_dict(),t.critic.state_dict(),t.actor_optimizer.state_dict(),t.critic_optimizer.state_dict()))
    grads=[None if p.grad is None else p.grad.clone() for m in (t.actor,t.critic) for p in m.parameters()]
    rng=rng_state(); modes=(t.actor.training,t.critic.training)
    d=diagnose_actor(t.actor,a,source='synthetic_physical_inputs')
    v=diagnose_critic(t.critic,c,source='synthetic_physical_inputs')
    assert d['status']==v['status']=='ok' and len(d['layers'])==len(v['layers'])==2
    assert d['input'][5]['max']>=500 and d['beta_alpha']['nonfinite_count']==0
    assert all(layer['derivative']['min']>=0 for layer in d['layers'])
    with torch.no_grad():
        torch.testing.assert_close(t.actor.distribution(t.tensor(a)).mean,before,rtol=0,atol=0)
        torch.testing.assert_close(t.critic(t.tensor(c)),values,rtol=0,atol=0)
    assert_tree_equal(state,(t.actor.state_dict(),t.critic.state_dict(),t.actor_optimizer.state_dict(),t.critic_optimizer.state_dict()))
    assert_tree_equal(grads,[p.grad for m in (t.actor,t.critic) for p in m.parameters()])
    assert_tree_equal(rng,rng_state()); assert modes==(t.actor.training,t.critic.training)
    assert not t.actor.network[0]._forward_hooks and not t.critic.network[0]._forward_hooks
    with pytest.raises(ValueError,match='source and critic protocol'):
        diagnose_critic(t.critic,c,source='actual_inspection_environment')
    a[0,1]=np.nan
    assert diagnose_actor(t.actor,a,source='synthetic_physical_inputs')['status']=='nonfinite_input_forward_skipped'


def test_rng_isolation_restores_even_on_exception():
    state=rng_state()
    with pytest.raises(RuntimeError):
        with isolated_rng():
            random.random(); np.random.random(); torch.rand(4); torch.rand(4,device='cuda')
            raise RuntimeError('controlled')
    assert_tree_equal(state,rng_state())


def test_telemetry_does_not_change_optimization_rng_or_existing_metrics():
    c=ProtocolTrainConfig(ppo_epochs=2,minibatch_size=3)
    a,b=ProtocolMAPPOTrainer(c),ProtocolMAPPOTrainer(c)
    batch=hand_batch((4,2,1,0,0),a.device)
    torch.manual_seed(81); ma=a.update(batch); ra=rng_state()
    torch.manual_seed(81); mb=b.update(batch,diagnostics=False); rb=rng_state()
    extra=ma.pop('numerical_diagnostics')
    assert ma==mb
    assert extra['per_uav_valid_samples']==[4,2,1,0,0] and extra['active_uav_count']==3
    assert extra['invalid_samples_in_loss']==0 and extra['beta_parameters']['finite']
    for x,y in ((a.actor,b.actor),(a.critic,b.critic)):
        assert_tree_equal(x.state_dict(),y.state_dict())
    assert_tree_equal(a.actor_optimizer.state_dict(),b.actor_optimizer.state_dict())
    assert_tree_equal(a.critic_optimizer.state_dict(),b.critic_optimizer.state_dict())
    assert_tree_equal(ra,rb)
    assert a.sampled_steps==b.sampled_steps==0


@pytest.fixture(scope='module')
def ppo_cases(): return ppo_boundaries()


@pytest.mark.parametrize('name',['singleton','two_samples','imbalanced','single_uav_minibatches','multi_uav_minibatch','unnormalized_singleton','all_invalid'])
def test_ppo_hand_answers_sample_counts_and_clipping(ppo_cases,name):
    row=next(r for r in ppo_cases['cases'] if r['name']==name)
    for key in row['expected']:
        np.testing.assert_allclose(row['actual'][key],row['expected'][key],rtol=2e-6,atol=2e-7,err_msg=key)
    d=row['metrics']['numerical_diagnostics']
    assert d['per_uav_valid_samples']==row['counts']
    assert d['per_uav_minibatches']==row['expected_participation']
    assert d['active_uav_count']==sum(c>0 for c in row['counts'])
    assert row['metrics']['valid_samples']==sum(row['counts']) and row['sampled_steps']==0
    np.testing.assert_allclose(row['actual_advantage_after'],row['expected_advantage_after'],rtol=1e-7,atol=1e-7)
    assert row['actual_returns']==row['actual_advantage_before']
    batches=row['expected_per_minibatch']
    for column,label in ((3,'actor'),(4,'critic')):
        expected=sum(v[column]>.5 for v in batches)/max(1,len(batches))
        assert d[f'{label}_gradient_clip_fraction']==expected
    if name=='singleton':
        assert d['advantage_before'][0]['mean']==1 and d['advantage_before'][0]['std']==0
        assert d['advantage_after'][0]['mean']==d['advantage_after'][0]['std']==0
        assert row['actual']['policy_loss']==0 and row['actual']['actor_norm_mean']==0
        assert d['singleton_uavs']==[0]
    if name=='two_samples':
        assert d['advantage_before'][0]['mean']==1.5 and d['advantage_before'][0]['std']==.5
        assert d['advantage_after'][0]['mean']==0 and d['advantage_after'][0]['std']==1
    if name=='all_invalid':
        assert row['metrics']['optimizer_steps']==0 and row['updates']==0


def test_real_beta_singleton_keeps_entropy_gradient(ppo_cases):
    real=ppo_cases['real_beta_singleton']
    assert real['policy_loss']==0
    assert real['numerical_diagnostics']['actor_gradient_norm']['max']>0
    assert real['numerical_diagnostics']['beta_parameters']['finite']


def test_clip_telemetry_includes_pytorch_epsilon_at_exact_threshold():
    from algorithm.mappo.diagnostics import UpdateObserver
    t=ProtocolMAPPOTrainer()
    o=UpdateObserver(t.actor,t.critic,1,.5)
    o.minibatch(torch.tensor([0],device=t.device),torch.tensor(.5,device=t.device),torch.tensor(0.,device=t.device))
    assert o.report()['actor_gradient_clip_fraction']==1
    assert o.report()['critic_gradient_clip_fraction']==0


def test_empty_batch_and_rejected_nonfinite_leave_adam_and_rng_unchanged():
    t=ProtocolMAPPOTrainer(ProtocolTrainConfig(ppo_epochs=1))
    t.update(hand_batch((2,1,0,0,0),t.device))
    before=copy.deepcopy((t.actor.state_dict(),t.critic.state_dict(),t.actor_optimizer.state_dict(),t.critic_optimizer.state_dict()))
    rng=rng_state(); updates=t.updates
    result=t.update(hand_batch((0,0,0,0,0),t.device))
    assert result['optimizer_steps']==0 and t.updates==updates
    invalid=hand_batch((1,0,0,0,0),t.device); invalid.rewards[0,0,0]=float('nan')
    with pytest.raises(FloatingPointError): t.update(invalid)
    assert t.last_update_diagnostics['status']=='rejected'
    assert_tree_equal(before,(t.actor.state_dict(),t.critic.state_dict(),t.actor_optimizer.state_dict(),t.critic_optimizer.state_dict()))
    assert_tree_equal(rng,rng_state())


def test_protocol_shape_aliases_are_rejected_before_any_update():
    t=ProtocolMAPPOTrainer()
    b=hand_batch((2,1,0,0,0),t.device)
    b.rewards=b.rewards.transpose(0,1)  # identical number of entries, wrong axes
    with pytest.raises(ValueError,match='axes disagree'): t.update(b)
    b=hand_batch((2,1,0,0,0),t.device)
    with pytest.raises(ValueError,match='90D'): MAPPOTrainer().update(b)
    # Previously the legacy no-valid early return bypassed the width check.
    with pytest.raises(ValueError,match='90D'):
        MAPPOTrainer().update(hand_batch((0,0,0,0,0),t.device))


@pytest.fixture(scope='module')
def reward_cases(): return reward_boundaries()


@pytest.mark.parametrize('name',['normal_timeout','integration_one','integration_all','injected_one','injected_all'])
def test_failure_reward_boundary_exact_float32_and_component_answers(reward_cases,name):
    r=next(c for c in reward_cases['cases'] if c['name']==name)
    np.testing.assert_array_equal(r['per_uav_return'],r['expected_float32_return'])
    assert max(abs(x) for x in r['analytic_error'])<4e-7
    for component in ('completion','separation','terminal'):
        assert r['component_totals'][component]==[0,0,0]
    all_failure=name.endswith('_all')
    assert r['terminated']==all_failure and r['truncated']!=all_failure
    assert r['steps']==(1 if all_failure else 250)
    for i in range(3):
        failed=(i==0 and name!='normal_timeout') or all_failure
        progress=0. if failed and name.startswith('integration') else -.05 if failed else -12.5
        np.testing.assert_allclose(r['component_totals']['progress'][i],progress,rtol=0,atol=1e-10)
        np.testing.assert_allclose(r['component_totals']['efficiency'][i],-.01 if failed else -2.5,rtol=0,atol=1e-12)
    first=r['first_steps'][0]
    assert first['current_mask']==[1,1,1]
    if name!='normal_timeout':
        expected_next=[0,0,0] if all_failure else [0,1,1]
        assert first['next_mask']==expected_next
        assert first['one_step_advantage_v1_next_v2'][0]==first['reward'][0]-1
        assert first['one_step_return_v1_next_v2'][0]==first['reward'][0]


def test_early_failures_have_higher_return_without_changing_reward(reward_cases):
    rows={c['name']:c for c in reward_cases['cases']}
    assert rows['integration_all']['summary']['episode_return']>rows['normal_timeout']['summary']['episode_return']
    assert rows['injected_one']['per_uav_return'][0]>rows['normal_timeout']['per_uav_return'][0]
    assert rows['integration_one']['first_steps'][1]['one_step_return_v1_next_v2'][0]==0
