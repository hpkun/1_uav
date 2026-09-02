"""Canonical fingerprints and strict formal-checkpoint validation."""
from copy import deepcopy
import hashlib,json
from env.config import ENVIRONMENT_VERSION
from algorithm.common.protocol import config_sha256
from algorithm.mappo.trainer import MAPPO_IMPL_VERSION
from .trainer import MODULAR_MAPPO_IMPL_VERSION
def canonical_sha256(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()
def checkpoint_architecture(trainer):
 return {"actor_class":type(trainer.actor).__name__,"critic_class":type(trainer.critic).__name__,"actor_input_dim":trainer.actor.base_observation_dim+trainer.actor.context_dim,"critic_input_dim":trainer.critic.base_observation_dim+trainer.critic.context_dim,"hidden_dim":256 if trainer.actor.entity_attention_enabled else trainer.actor.backbone[0].out_features,"actor_gru_hidden_dim":trainer.actor.recurrent_hidden_dim,"critic_gru_hidden_dim":trainer.critic.recurrent_hidden_dim,"entity_attention_enabled":trainer.actor.entity_attention_enabled,"entity_dim":trainer.actor.entity_dim if trainer.actor.entity_attention_enabled else 0,"entity_attention_heads":trainer.actor.entity_attention_heads if trainer.actor.entity_attention_enabled else 0}

def validate_modular_checkpoint(state,env_config,algorithm_config,expected_runtime=None):
 if state.get("algorithm")!="modular_mappo":raise RuntimeError("checkpoint algorithm mismatch")
 checkpoint_version=state.get("modular_mappo_impl_version")
 if checkpoint_version!=MODULAR_MAPPO_IMPL_VERSION:raise RuntimeError(f"modular implementation version mismatch: checkpoint={checkpoint_version}, current={MODULAR_MAPPO_IMPL_VERSION}")
 if state.get("baseline_mappo_impl_version")!=MAPPO_IMPL_VERSION:raise RuntimeError("baseline MAPPO implementation version mismatch")
 extra=state.get("extra",{});checks={
  "environment_version":str(env_config.get("environment_version",ENVIRONMENT_VERSION)),
  "environment_variant":str(env_config.get("environment_variant","direct_v2_3")),
  "environment_config_sha256":config_sha256(env_config),
  "algorithm_config_sha256":config_sha256(algorithm_config),
 }
 for key,expected in checks.items():
  if extra.get(key)!=expected:raise RuntimeError(f"checkpoint {key} mismatch: expected {expected!r}, got {extra.get(key)!r}")
 module_hash=canonical_sha256(algorithm_config.get("modules",{}))
 if state.get("module_config_sha256")!=module_hash:raise RuntimeError("checkpoint module config mismatch")
 network=algorithm_config["network"]
 for key,expected in (("observation_dim",int(network["observation_dim"])),("action_dim",int(network["action_dim"])),("num_agents",int(network["num_agents"]))):
  if int(extra.get(key,-1))!=expected:raise RuntimeError(f"checkpoint {key} mismatch")
 if expected_runtime:
  for key,expected in expected_runtime.items():
   if extra.get(key)!=expected:raise RuntimeError(f"checkpoint {key} mismatch")
 if extra.get("network_architecture") is None:raise RuntimeError("checkpoint lacks network architecture")
 pop_enabled=bool(algorithm_config.get("modules",{}).get("popart",{}).get("enabled",False))
 if pop_enabled != bool(state.get("module_config",{}).get("popart",{}).get("enabled",False)):raise RuntimeError("checkpoint PopArt protocol mismatch")
 return True

def _branch_comparable_config(config):
 value=deepcopy(config);value.get("training",{}).pop("total_sampled_steps",None)
 value.get("modules",{}).pop("actor_lr_decay",None)
 return value

def validate_modular_branch(state,env_config,algorithm_config,expected_runtime=None):
 """Allow only an explicit actor_lr_decay intervention at a branch boundary."""
 if state.get("algorithm")!="modular_mappo":raise RuntimeError("branch checkpoint algorithm mismatch")
 if state.get("modular_mappo_impl_version")!=MODULAR_MAPPO_IMPL_VERSION:raise RuntimeError("branch modular implementation version mismatch")
 if state.get("baseline_mappo_impl_version")!=MAPPO_IMPL_VERSION:raise RuntimeError("branch baseline MAPPO implementation version mismatch")
 extra=state.get("extra",{});source_env=extra.get("environment_config");source_algorithm=extra.get("algorithm_config")
 if not isinstance(source_env,dict) or not isinstance(source_algorithm,dict):raise RuntimeError("branch checkpoint lacks self-describing source configs")
 if source_env!=env_config:raise RuntimeError("branch environment config differs from source checkpoint")
 if extra.get("environment_config_sha256")!=config_sha256(env_config):raise RuntimeError("branch environment hash mismatch")
 if _branch_comparable_config(source_algorithm)!=_branch_comparable_config(algorithm_config):raise RuntimeError("branch config differs outside the actor_lr_decay/total_sampled_steps whitelist")
 source_decay=source_algorithm.get("modules",{}).get("actor_lr_decay",{})
 destination_decay=algorithm_config.get("modules",{}).get("actor_lr_decay",{})
 if bool(source_decay.get("enabled",False)) and source_decay!=destination_decay:raise RuntimeError("branch cannot alter an already-enabled actor_lr_decay protocol")
 if expected_runtime:
  for key,expected in expected_runtime.items():
   if extra.get(key)!=expected:raise RuntimeError(f"branch checkpoint {key} mismatch: expected {expected!r}, got {extra.get(key)!r}")
 if extra.get("network_architecture") is None:raise RuntimeError("branch checkpoint lacks network architecture")
 required=("actor","critic","actor_optimizer","critic_optimizer","sampled_steps","vector_steps","module_config_sha256")
 missing=[key for key in required if key not in state]
 if missing:raise RuntimeError("branch checkpoint lacks required state: "+", ".join(missing))
 return {"intervention":"actor_lr_decay" if source_decay!=destination_decay else "fixed_lr_control","source_actor_lr_decay":deepcopy(source_decay),"destination_actor_lr_decay":deepcopy(destination_decay)}

def is_formal_v2_checkpoint(state):
 extra=state.get("extra",{}) if isinstance(state.get("extra",{}),dict) else {}
 required=("environment_version","environment_variant","environment_config_sha256",
           "algorithm_config_sha256","network_architecture","observation_dim",
           "action_dim","num_agents","training_seed","training_gamma",
           "training_num_envs","training_total_sampled_steps","training_smoke")
 return (state.get("algorithm")=="modular_mappo" and
         state.get("modular_mappo_impl_version")==MODULAR_MAPPO_IMPL_VERSION and
         state.get("baseline_mappo_impl_version")==MAPPO_IMPL_VERSION and
         all(key in extra for key in required))

__all__=["canonical_sha256","checkpoint_architecture","validate_modular_checkpoint","validate_modular_branch","is_formal_v2_checkpoint"]
