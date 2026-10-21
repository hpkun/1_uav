import importlib.util
from pathlib import Path
import torch
from env import EnvConfig
from algorithm.mappo.config import TrainConfig
from algorithm.mappo.trainer import MAPPOTrainer
from algorithm.common.vector_env import ParallelVectorEnv


def test_metrics_match_backward_and_pre_change_parameter_updates(monkeypatch,record_property):
    spec=importlib.util.spec_from_file_location('frozen_update_reference',Path(__file__).parent/'fixtures/mappo_update_pre_metrics.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    config=TrainConfig(num_envs=1,horizon=3,ppo_epochs=2,minibatch_size=4,entropy_coef=.07)
    trainer=MAPPOTrainer(config)
    with ParallelVectorEnv(EnvConfig(max_steps=3),1,config.seed) as vector:
        rollout,_=trainer.collect(vector)
    current,reference=MAPPOTrainer(config),MAPPOTrainer(config)
    backward_values=[]; backward=torch.Tensor.backward
    def capture(tensor,*args,**kwargs):
        backward_values.append(tensor.detach().item())
        return backward(tensor,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,'backward',capture)
    torch.manual_seed(173); metrics=current.update(rollout)
    actor_backward=backward_values[::2].copy()
    torch.manual_seed(173); old=module.update(reference,rollout)
    assert abs(metrics['actor_total_loss']-sum(actor_backward)/len(actor_backward))<1e-12
    assert metrics['actor_loss']==metrics['actor_total_loss']
    assert metrics['policy_loss']==old['actor_loss']
    assert abs(metrics['actor_total_loss']-metrics['policy_loss']+config.entropy_coef*metrics['entropy'])<2e-8
    assert abs(metrics['policy_loss']-metrics['actor_total_loss'])>.1
    for current_model,old_model in ((current.actor,reference.actor),(current.critic,reference.critic)):
        for a,b in zip(current_model.parameters(),old_model.parameters()): torch.testing.assert_close(a,b,rtol=0,atol=0)
    for new_optimizer,old_optimizer in ((current.actor_optimizer,reference.actor_optimizer),(current.critic_optimizer,reference.critic_optimizer)):
        for a,b in zip(new_optimizer.state.values(),old_optimizer.state.values()):
            for key in a: torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
    record_property('parameter_update_identical',True)
    record_property('policy_loss',metrics['policy_loss'])
    record_property('actor_total_loss',metrics['actor_total_loss'])
