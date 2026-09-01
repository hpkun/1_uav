"""Canonical fingerprints and strict formal-checkpoint validation."""
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

__all__=["canonical_sha256","checkpoint_architecture","validate_modular_checkpoint","is_formal_v2_checkpoint"]
