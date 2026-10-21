"""Complete trusted local checkpoints with optimizer, RNG and worker state."""
from pathlib import Path
import random
import numpy as np
import torch
from ..config import ENVIRONMENT_VERSION


def save_checkpoint(path, trainer, env_config, vector=None, best_score=None):
    payload = dict(environment_version=ENVIRONMENT_VERSION, env_config=env_config.to_dict(),
                   train_config=trainer.config.to_dict(), actor=trainer.actor.state_dict(), critic=trainer.critic.state_dict(),
                   actor_optimizer=trainer.actor_optimizer.state_dict(), critic_optimizer=trainer.critic_optimizer.state_dict(),
                   updates=trainer.updates, best_score=best_score, torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if trainer.device.type == 'cuda' else None,
                   numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                   vector=vector.snapshot() if vector else None)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + '.tmp')
    torch.save(payload, temp)
    temp.replace(p)


def read_checkpoint(path):
    # Full resume contains TaskState/NumPy RNG. Only load your own trusted files.
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('environment_version') != ENVIRONMENT_VERSION:
        raise ValueError('checkpoint environment identity mismatch')
    return payload


def restore_checkpoint(payload, trainer, env_config, vector=None):
    if payload['env_config'] != env_config.to_dict() or payload['train_config'] != trainer.config.to_dict():
        raise ValueError('checkpoint configuration mismatch')
    trainer.actor.load_state_dict(payload['actor'])
    trainer.critic.load_state_dict(payload['critic'])
    trainer.actor_optimizer.load_state_dict(payload['actor_optimizer'])
    trainer.critic_optimizer.load_state_dict(payload['critic_optimizer'])
    trainer.updates = payload['updates']
    torch.set_rng_state(payload['torch_rng'])
    if trainer.device.type == 'cuda' and payload['cuda_rng'] is not None:
        torch.cuda.set_rng_state_all(payload['cuda_rng'])
    np.random.set_state(payload['numpy_rng'])
    random.setstate(payload['python_rng'])
    if vector is not None:
        if payload['vector'] is None:
            raise ValueError('checkpoint lacks worker state')
        vector.restore(payload['vector'])
