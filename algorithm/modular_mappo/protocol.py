"""Canonical module/config fingerprints."""
import hashlib,json
def canonical_sha256(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()
def checkpoint_architecture(trainer):
 return {"actor_class":type(trainer.actor).__name__,"critic_class":type(trainer.critic).__name__,"actor_input_dim":trainer.actor.base_observation_dim+trainer.actor.context_dim,"critic_input_dim":trainer.critic.base_observation_dim+trainer.critic.context_dim,"hidden_dim":trainer.actor.backbone[0].out_features,"actor_gru_hidden_dim":trainer.actor.recurrent_hidden_dim,"critic_gru_hidden_dim":trainer.critic.recurrent_hidden_dim}
__all__=["canonical_sha256","checkpoint_architecture"]
