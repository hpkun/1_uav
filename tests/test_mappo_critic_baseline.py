from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import numpy as np,pytest,torch,yaml
from algorithm.mappo import CentralizedMLPCritic,CentralizedValueCritic,MAPPOTrainer,RolloutBatch
from algorithm.mappo.factory import build_mappo_trainer
from tools.preflight_mappo_critic_baseline_1p5m import validate_configs
ROOT=Path(__file__).resolve().parents[1]
def load(name):return yaml.safe_load((ROOT/name).read_text(encoding="utf-8"))
def configs():return load("configs/mappo_mlp_baseline_1p5m.yaml"),load("configs/mappo_attention_baseline_1p5m.yaml")
def rollout(trainer):
 rng=np.random.default_rng(3);obs=rng.normal(size=(2,1,4,52)).astype("f");alive=np.ones((2,1,4),"f");actions=[];raw=[];logs=[]
 for t in range(2):a,r,l=trainer.act(obs[t],alive[t],return_policy_data=True);actions.append(a);raw.append(r);logs.append(l)
 return RolloutBatch(obs,np.stack(actions),np.stack(raw),np.stack(logs),rng.normal(size=(2,1,4)).astype("f"),np.zeros((2,1),"f"),alive,obs+.01,alive)
def test_mlp_shape_finite_and_dead_focal_zero():
 critic=CentralizedMLPCritic(hidden_dim=32);obs=torch.randn(3,4,52);mask=torch.tensor([[1,1,1,1],[1,0,1,0],[0,0,0,0]],dtype=torch.float32);value=critic(obs,mask)
 assert value.shape==(3,4) and torch.isfinite(value).all() and torch.count_nonzero(value[1,[1,3]])==0 and torch.count_nonzero(value[2])==0
def test_dead_teammate_observation_is_masked():
 critic=CentralizedMLPCritic(hidden_dim=32);obs=torch.randn(2,4,52);mask=torch.ones(2,4);mask[:,2]=0;changed=obs.clone();changed[:,2]=1e8
 assert torch.equal(critic(obs,mask),critic(changed,mask))
def test_shared_critic_parameterization_and_input_dim():
 critic=CentralizedMLPCritic();assert critic.value_network[0].in_features==260 and critic.value_network[-1].out_features==1
 assert len({id(p) for p in critic.parameters()})==len(list(critic.parameters()))
def test_factory_types_default_and_invalid():
 mlp,attn=configs();assert isinstance(build_mappo_trainer(mlp,"cpu",32).critic,CentralizedMLPCritic);assert isinstance(build_mappo_trainer(attn,"cpu",32).critic,CentralizedValueCritic)
 legacy=deepcopy(attn);legacy["network"].pop("critic_type");assert isinstance(build_mappo_trainer(legacy,"cpu",32).critic,CentralizedValueCritic)
 bad=deepcopy(attn);bad["network"]["critic_type"]="bad"
 with pytest.raises(ValueError):build_mappo_trainer(bad,"cpu",32)
def test_same_seed_actor_initialization_bitwise_identical():
 mlp,attn=configs();m=build_mappo_trainer(mlp,"cpu");a=build_mappo_trainer(attn,"cpu")
 assert type(m.actor) is type(a.actor) and all(torch.equal(x,y) for x,y in zip(m.actor.state_dict().values(),a.actor.state_dict().values()))
def test_mlp_update_finite_and_critic_gradient_changes():
 trainer=MAPPOTrainer(hidden_dim=32,critic_type="mlp",ppo_epochs=1,minibatch_size=2);before=deepcopy(next(trainer.critic.parameters()).detach());metrics=trainer.update(rollout(trainer))
 assert all(np.isfinite(x) for x in metrics.values()) and not torch.equal(before,next(trainer.critic.parameters())) and metrics["critic_grad_norm"]>0
def test_checkpoint_type_roundtrip_mismatch_and_legacy_attention(tmp_path):
 attention=MAPPOTrainer(hidden_dim=32);path=tmp_path/"a.pt";attention.save(path);state=torch.load(path,map_location="cpu",weights_only=False);assert state["critic_type"]=="attention";MAPPOTrainer(hidden_dim=32).load(path)
 with pytest.raises(RuntimeError,match="critic_type mismatch"):MAPPOTrainer(hidden_dim=32,critic_type="mlp").load(path)
 state.pop("critic_type");legacy=tmp_path/"legacy.pt";torch.save(state,legacy);MAPPOTrainer(hidden_dim=32).load(legacy)
def test_configs_strictly_matched_and_protocol():
 mlp,attn=configs();assert validate_configs(mlp,attn)
 assert mlp["network"]["observation_dim"]==attn["network"]["observation_dim"]==52
 assert mlp["implementation"]["evaluation_seed_base"]==attn["implementation"]["evaluation_seed_base"]==46_000_000
 assert all("wave" not in str(cfg["network"]).lower() for cfg in (mlp,attn))
 assert not (ROOT/"outputs/mappo_mlp_seed5303_1p5m").exists() and not (ROOT/"outputs/mappo_attention_seed5303_1p5m").exists()
def test_parameter_counts_are_exact_and_actor_equal():
 mlp,attn=configs();m=build_mappo_trainer(mlp,"cpu");a=build_mappo_trainer(attn,"cpu")
 assert sum(p.numel() for p in m.actor.parameters())==sum(p.numel() for p in a.actor.parameters())
 assert sum(p.numel() for p in m.critic.parameters())>0 and sum(p.numel() for p in a.critic.parameters())>0
