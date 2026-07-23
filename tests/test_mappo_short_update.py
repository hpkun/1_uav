import numpy as np,torch
from uav_env.algorithms.mappo.networks import SharedActor,CentralizedCritic
from uav_env.algorithms.mappo.rollout_buffer import RolloutBuffer
from uav_env.algorithms.mappo.trainer import MAPPOTrainer
from uav_env.algorithms.mappo.value_normalizer import ValueNormalizer

def test_short_update_changes_finite_parameters():
 c={"actor_lr":5e-4,"critic_lr":5e-4,"normalize_advantages":True,"ppo_epochs":1,"num_mini_batches":1,"clip_param":.2,"value_clip_param":.2,"entropy_coef":.01,"value_loss_coef":1.,"max_grad_norm":10.,"huber_delta":10.};a=SharedActor(3);v=CentralizedCritic(4,1);b=RolloutBuffer(2,1,1,3,4);b.observations[:]=np.random.randn(*b.observations.shape);b.global_states[:]=np.random.randn(*b.global_states.shape);b.actions[:]=1;b.rewards[:]=1;b.advantages[:]=1;b.returns[:]=1
 before=[p.detach().clone() for p in a.parameters()];m=MAPPOTrainer(a,v,c,ValueNormalizer(),torch.device("cpu")).update(b);assert any(not torch.equal(x,p) for x,p in zip(before,a.parameters())) and all(np.isfinite(list(m.values())))

