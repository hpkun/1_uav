"""Canonical fingerprints and strict formal-checkpoint validation."""
from copy import deepcopy
import hashlib,json
from env.config import ENVIRONMENT_VERSION
from algorithm.common.protocol import config_sha256
from algorithm.mappo.trainer import MAPPO_IMPL_VERSION
from .trainer import MODULAR_MAPPO_IMPL_VERSION
def canonical_sha256(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()
def checkpoint_architecture(trainer):
 mode=trainer.actor.entity_attention_mode if trainer.actor.entity_attention_enabled else "disabled"
 fbmr=mode in {"frozen_base_mean_residual","frozen_base_dual_bounded_mean_residual"};dual=mode=="frozen_base_dual_bounded_mean_residual"
 critic_input_dim=trainer.critic.embedding[0].in_features
 result={"actor_class":type(trainer.actor).__name__,"critic_class":type(trainer.critic).__name__,"actor_input_dim":trainer.actor.backbone[0].in_features if hasattr(trainer.actor,"backbone") else trainer.actor.base_observation_dim+trainer.actor.context_dim,"critic_input_dim":critic_input_dim,"actor_context_dim":trainer.actor.context_dim,"critic_context_dim":trainer.critic.context_dim,"critic_context_injection":trainer.critic.context_injection,"wave_context_encoding":trainer.wave_context.encoding if trainer.wave_context.enabled else "disabled","wave_context_target":trainer.wave_context.target if trainer.wave_context.enabled else "disabled","actor_parameter_count":sum(parameter.numel() for parameter in trainer.actor.parameters()),"critic_parameter_count":sum(parameter.numel() for parameter in trainer.critic.parameters()),"hidden_dim":256 if trainer.actor.entity_attention_enabled else trainer.actor.backbone[0].out_features,"actor_gru_hidden_dim":trainer.actor.recurrent_hidden_dim,"critic_gru_hidden_dim":trainer.critic.recurrent_hidden_dim,"entity_attention_enabled":trainer.actor.entity_attention_enabled,"entity_attention_mode":mode,"entity_dim":trainer.actor.entity_dim if trainer.actor.entity_attention_enabled else 0,"entity_attention_heads":trainer.actor.entity_attention_heads if trainer.actor.entity_attention_enabled else 0,"base_actor_frozen":fbmr,"entity_mean_residual_enabled":fbmr,"max_mean_correction":trainer.actor.max_mean_correction if mode=="frozen_base_mean_residual" else 0.,"dual_bound_enabled":dual,"alpha_abs":trainer.actor.alpha_abs if dual else 0.,"alpha_rel":trainer.actor.alpha_rel if dual else 0.,"log_std_source":"frozen_baseline" if fbmr else "actor_head"}
 if trainer.mission_film.enabled:
  result.update({"mission_film_enabled":True,"mission_film_mode":trainer.mission_film.mode,
   "mission_encoder_hidden_dim":trainer.mission_film.encoder_hidden_dim,"mission_film_alpha":trainer.mission_film.alpha,
   "mission_film_augmented_residual":trainer.mission_film.augmented_residual,
   "mission_film_identity_init":trainer.mission_film.identity_init})
 if trainer.inter_wave_credit.enabled:
  result.update({"inter_wave_credit_enabled":True,"inter_wave_credit_version":trainer.inter_wave_credit.version,
   "iw_critic_class":type(trainer.iw_critic).__name__,"iw_critic_input_dim":56,
   "iw_critic_parameter_count":sum(parameter.numel() for parameter in trainer.iw_critic.parameters()),
   "iw_target_definition":{"wave1":"(clear_wave2 + clear_wave3) / 2","wave2":"clear_wave3","wave3":None},
   "iw_actor_credit":"q_next - q_current (no gamma, stop-gradient)",
   "iw_gradient_fusion":"asymmetric_tactical_preserving_projection"})
 return result

def _validate_embedded_disabled_curriculum_runtime(extra,algorithm_config):
 source=extra.get("environment_config")
 curriculum=extra.get("curriculum_config",algorithm_config.get("modules",{}).get("curriculum",{}))
 if (not bool(curriculum.get("enabled",False)) and isinstance(source,dict)
     and extra.get("current_total_waves") is not None
     and int(extra["current_total_waves"])!=int(source.get("persistent_waves",{}).get("total_waves",1))):
  raise RuntimeError("checkpoint declared/runtime wave mismatch under disabled curriculum")

def validate_modular_checkpoint(state,env_config,algorithm_config,expected_runtime=None):
 if state.get("algorithm")!="modular_mappo":raise RuntimeError("checkpoint algorithm mismatch")
 checkpoint_version=state.get("modular_mappo_impl_version")
 if checkpoint_version!=MODULAR_MAPPO_IMPL_VERSION:raise RuntimeError(f"modular implementation version mismatch: checkpoint={checkpoint_version}, current={MODULAR_MAPPO_IMPL_VERSION}")
 if state.get("baseline_mappo_impl_version")!=MAPPO_IMPL_VERSION:raise RuntimeError("baseline MAPPO implementation version mismatch")
 extra=state.get("extra",{})
 source=extra.get("environment_config")
 curriculum_config=extra.get("curriculum_config",algorithm_config.get("modules",{}).get("curriculum",{}))
 curriculum_enabled=bool(curriculum_config.get("enabled",False))
 # Backward-compatible forensic guard: old checkpoints did not carry runtime
 # hashes, but did carry both the declared environment and the actual wave count.
 _validate_embedded_disabled_curriculum_runtime(extra,algorithm_config)
 new_keys=("declared_environment_config_sha256","declared_total_waves","declared_max_steps",
           "effective_training_environment_config_sha256","effective_training_total_waves","effective_training_max_steps",
           "runtime_environment_config_sha256","runtime_total_waves","runtime_max_steps",
           "evaluation_environment_config_sha256","evaluation_total_waves","evaluation_max_steps",
           "curriculum_enabled")
 if any(key in extra for key in new_keys):
  missing=[key for key in new_keys if key not in extra]
  if missing:raise RuntimeError(f"checkpoint runtime environment provenance incomplete: {missing}")
  if not isinstance(source,dict):raise RuntimeError("checkpoint runtime provenance lacks declared environment_config")
  runtime_config=extra.get("runtime_environment_config")
  if not isinstance(runtime_config,dict):raise RuntimeError("checkpoint runtime provenance lacks runtime_environment_config")
  source_waves=int(source.get("persistent_waves",{}).get("total_waves",1));source_steps=int(source["simulation"]["max_steps"])
  runtime_waves=int(runtime_config.get("persistent_waves",{}).get("total_waves",1));runtime_steps=int(runtime_config["simulation"]["max_steps"])
  strict={"declared_environment_config_sha256":config_sha256(source),"declared_total_waves":source_waves,
          "declared_max_steps":source_steps,"runtime_environment_config_sha256":config_sha256(runtime_config),
          "runtime_total_waves":runtime_waves,"runtime_max_steps":runtime_steps,
          "effective_training_environment_config_sha256":config_sha256(runtime_config),
          "effective_training_total_waves":runtime_waves,"effective_training_max_steps":runtime_steps,
          "evaluation_environment_config_sha256":config_sha256(source),"evaluation_total_waves":source_waves,
          "evaluation_max_steps":source_steps,"curriculum_enabled":curriculum_enabled}
  for key,expected in strict.items():
   if extra.get(key)!=expected:raise RuntimeError(f"checkpoint {key} mismatch: expected {expected!r}, got {extra.get(key)!r}")
  if not curriculum_enabled and config_sha256(runtime_config)!=config_sha256(source):
   raise RuntimeError("checkpoint declared/runtime environment mismatch under disabled curriculum")
 checks={
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
 extra=state.get("extra",{});_validate_embedded_disabled_curriculum_runtime(extra,algorithm_config);source_env=extra.get("environment_config");source_algorithm=extra.get("algorithm_config")
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

def _fbmr_comparable_config(config):
 value=deepcopy(config)
 for key in ("formal_protocol","development_protocol","development_branch"):value.pop(key,None)
 value.get("training",{}).pop("total_sampled_steps",None);value.get("training",{}).pop("actor_learning_rate",None)
 value.get("implementation",{}).pop("evaluation_seed_base",None)
 value.get("modules",{}).pop("actor_lr_decay",None);value.get("modules",{}).pop("entity_attention",None)
 return value

def validate_fbmr_stage2_branch(state,env_config,algorithm_config,expected_runtime=None):
 """Strictly validate one 900k Formal MAPPO source and its Stage-2 intervention."""
 if state.get("algorithm")!="modular_mappo":raise RuntimeError("FBMR source algorithm must be modular_mappo")
 if state.get("modular_mappo_impl_version")!=MODULAR_MAPPO_IMPL_VERSION or state.get("baseline_mappo_impl_version")!=MAPPO_IMPL_VERSION:raise RuntimeError("FBMR source implementation version mismatch")
 extra=state.get("extra",{});_validate_embedded_disabled_curriculum_runtime(extra,algorithm_config);source_env=extra.get("environment_config");source=extra.get("algorithm_config")
 if not isinstance(source_env,dict) or not isinstance(source,dict):raise RuntimeError("FBMR source lacks embedded configs")
 if source_env!=env_config or extra.get("environment_config_sha256")!=config_sha256(env_config):raise RuntimeError("FBMR source environment mismatch")
 if extra.get("algorithm_config_sha256")!=config_sha256(source) or state.get("module_config_sha256")!=canonical_sha256(source.get("modules",{})):raise RuntimeError("FBMR source self-description hash mismatch")
 if extra.get("environment_variant")!="persistent_wave_v2":raise RuntimeError("FBMR source must be persistent_wave_v2")
 if int(state.get("sampled_steps",-1))!=900000:raise RuntimeError("FBMR source sampled_steps must be exactly 900000")
 if int(extra.get("training_num_envs",-1))!=24 or float(extra.get("training_gamma",0))!=.999:raise RuntimeError("FBMR source runtime mismatch")
 if bool(source.get("modules",{}).get("entity_attention",{}).get("enabled",False)):raise RuntimeError("FBMR source must be baseline MAPPO")
 decay=source.get("modules",{}).get("actor_lr_decay",{})
 expected_decay={"enabled":True,"schedule":"delayed_linear","start_step":600000,"end_step":900000,"start_lr":.0003,"end_lr":.0001}
 if decay!=expected_decay:raise RuntimeError("FBMR source actor LR schedule mismatch")
 if abs(float(state["actor_optimizer"]["param_groups"][0]["lr"])-1e-4)>1e-15:raise RuntimeError("FBMR source terminal actor LR is not 1e-4")
 if abs(float(source["training"]["critic_learning_rate"])-3e-4)>1e-15:raise RuntimeError("FBMR source critic LR mismatch")
 required=("actor","critic","actor_optimizer","critic_optimizer","rng_state","sampled_steps","vector_steps","ppo_updates","actor_updates","critic_updates")
 missing=[key for key in required if key not in state]
 if missing:raise RuntimeError("FBMR source lacks continuation state: "+", ".join(missing))
 rng=state.get("rng_state",{});rng_required=("python_random_state","numpy_random_state","torch_cpu_rng_state","torch_cuda_rng_state_all","trainer_permutation_rng_state")
 if state.get("rng_state_available") is not True or any(key not in rng for key in rng_required):raise RuntimeError("FBMR source RNG state is incomplete")
 if not state["actor_optimizer"].get("state") or not state["critic_optimizer"].get("state"):raise RuntimeError("FBMR source optimizer state is incomplete")
 architecture=extra.get("network_architecture",{})
 if architecture.get("entity_attention_enabled") is not False or architecture.get("entity_attention_mode","disabled")!="disabled":raise RuntimeError("FBMR source architecture metadata is not baseline MAPPO")
 if set(state["actor"])!={"backbone.0.weight","backbone.0.bias","backbone.2.weight","backbone.2.bias","mean.weight","mean.bias","log_std.weight","log_std.bias"}:raise RuntimeError("FBMR source actor topology is not baseline MAPPO")
 if _fbmr_comparable_config(source)!=_fbmr_comparable_config(algorithm_config):raise RuntimeError("FBMR Stage-2 config differs outside its strict whitelist")
 branch=algorithm_config.get("development_branch",{});intervention=branch.get("intervention")
 if intervention not in {"mappo_continuation","frozen_base_mean_residual"}:raise RuntimeError("unknown FBMR Stage-2 intervention")
 training=algorithm_config["training"]
 if int(training["total_sampled_steps"])!=1200000 or int(training["num_train_envs"])!=24 or float(training["critic_learning_rate"])!=3e-4:raise RuntimeError("FBMR Stage-2 budget/runtime mismatch")
 if int(algorithm_config["implementation"]["evaluation_seed_base"])!=34000000 or int(training["evaluation_episodes"])!=20 or int(training["evaluation_interval_sampled_steps"])!=100000:raise RuntimeError("FBMR Stage-2 validation protocol mismatch")
 entity=algorithm_config.get("modules",{}).get("entity_attention",{});destination_decay=algorithm_config.get("modules",{}).get("actor_lr_decay",{})
 if intervention=="mappo_continuation":
  if entity.get("enabled",False) or destination_decay!=expected_decay or float(training["actor_learning_rate"])!=3e-4:raise RuntimeError("MAPPO continuation intervention mismatch")
  actor_optimizer_restore=True
 else:
  expected_entity={"enabled":True,"mode":"frozen_base_mean_residual","entity_dim":32,"attention_heads":2,"max_mean_correction":.25}
  if entity!=expected_entity or destination_decay.get("enabled",False) or float(training["actor_learning_rate"])!=1e-4:raise RuntimeError("FBMR intervention mismatch")
  actor_optimizer_restore=False
 if expected_runtime:
  for key,expected in expected_runtime.items():
   if extra.get(key)!=expected:raise RuntimeError(f"FBMR source {key} mismatch")
 return {"intervention":intervention,"source_actor_lr_decay":deepcopy(decay),"destination_actor_lr_decay":deepcopy(destination_decay),"source_sampled_steps":900000,"target_sampled_steps":1200000,"actor_effective_lr":1e-4,"critic_lr":3e-4,"actor_optimizer_restore":actor_optimizer_restore,"critic_optimizer_restore":True,"rng_restore":True}

def _normalized_fbmr_bound_protocol(config):
 value=deepcopy(config)
 branch=value.get("development_branch",{});branch["intervention"]="FROZEN_BASE_BOUND_MODE"
 for key in ("dual_bound_enabled","alpha_abs","alpha_rel","max_mean_correction"):branch.pop(key,None)
 entity=value.get("modules",{}).get("entity_attention",{});entity["mode"]="FROZEN_BASE_BOUND_MODE"
 for key in ("alpha_abs","alpha_rel","max_mean_correction"):entity.pop(key,None)
 validation=value.get("development_protocol",{}).get("validation",{})
 if validation.get("role") in {"FBMR_STAGE2_DEVELOPMENT_VALIDATION","FBMR_V2_DEVELOPMENT_VALIDATION"}:validation["role"]="FBMR_BOUND_DEVELOPMENT_VALIDATION"
 return value

def validate_fbmr_v1_v2_only_bound_diff(v1_config,v2_config):
 """Require the resolved V1/V2 protocols to differ only in their fixed bound."""
 if _normalized_fbmr_bound_protocol(v1_config)!=_normalized_fbmr_bound_protocol(v2_config):
  raise RuntimeError("resolved FBMR V1/V2 configs differ outside the bound intervention")
 return True

def validate_fbmr_v2_stage2_branch(state,env_config,algorithm_config,expected_runtime=None):
 """Strict V2 validator reusing every V1 source invariant."""
 shadow=deepcopy(algorithm_config)
 shadow["modules"]["entity_attention"]={"enabled":True,"mode":"frozen_base_mean_residual","entity_dim":32,"attention_heads":2,"max_mean_correction":.25}
 shadow["development_branch"]["intervention"]="frozen_base_mean_residual"
 shadow.get("development_protocol",{}).get("validation",{})["role"]="FBMR_STAGE2_DEVELOPMENT_VALIDATION"
 result=validate_fbmr_stage2_branch(state,env_config,shadow,expected_runtime)
 entity=algorithm_config.get("modules",{}).get("entity_attention",{})
 expected={"enabled":True,"mode":"frozen_base_dual_bounded_mean_residual","entity_dim":32,"attention_heads":2,"alpha_abs":.25,"alpha_rel":.25}
 branch=algorithm_config.get("development_branch",{})
 if entity!=expected:raise RuntimeError("FBMR V2 entity config mismatch")
 if branch.get("intervention")!="frozen_base_dual_bounded_mean_residual":raise RuntimeError("FBMR V2 branch intervention mismatch")
 if branch.get("source_sampled_steps")!=900000 or branch.get("additional_sampled_steps")!=300000 or branch.get("target_sampled_steps")!=1200000:raise RuntimeError("FBMR V2 branch budget mismatch")
 if branch.get("actor_optimizer_restore") is not False or branch.get("critic_optimizer_restore") is not True or branch.get("rng_restore") is not True:raise RuntimeError("FBMR V2 restore protocol mismatch")
 if branch.get("base_actor_frozen") is not True or branch.get("log_std_source")!="frozen_baseline":raise RuntimeError("FBMR V2 frozen-policy protocol mismatch")
 result.update({"intervention":"frozen_base_dual_bounded_mean_residual","dual_bound_enabled":True,"alpha_abs":.25,"alpha_rel":.25})
 return result

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

__all__=["canonical_sha256","checkpoint_architecture","validate_modular_checkpoint","validate_modular_branch","validate_fbmr_stage2_branch","validate_fbmr_v2_stage2_branch","validate_fbmr_v1_v2_only_bound_diff","is_formal_v2_checkpoint"]
