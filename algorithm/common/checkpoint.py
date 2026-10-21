"""Complete trusted local checkpoints with optimizer, RNG and worker state."""
from pathlib import Path
import random
import numpy as np
import torch
from env.config import ENVIRONMENT_VERSION
from .protocol import ALGORITHM_VERSION, PROTOCOL
import zipfile


def save_checkpoint(path, trainer, env_config, vector=None, best_score=None):
    payload = dict(environment_version=ENVIRONMENT_VERSION, env_config=env_config.to_dict(),
                   algorithm_version=ALGORITHM_VERSION, protocol=PROTOCOL.copy(), sampled_steps=trainer.sampled_steps,
                   training_episode_count=trainer.training_episode_count,
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
    try:
        with zipfile.ZipFile(path) as archive:
            metadata = archive.read(next(n for n in archive.namelist() if n.endswith('/data.pkl')))
            if ENVIRONMENT_VERSION.encode() not in metadata or ALGORITHM_VERSION.encode() not in metadata:
                raise ValueError('incompatible checkpoint: environment/algorithm identity (60D models cannot resume as 90D)')
    except (zipfile.BadZipFile, StopIteration, KeyError) as error:
        raise ValueError('incompatible checkpoint format') from error
    payload = torch.load(path, map_location='cpu', weights_only=False)
    validate_checkpoint(payload)
    return payload


def validate_checkpoint(payload):
    if payload.get('environment_version') != ENVIRONMENT_VERSION or payload.get('algorithm_version') != ALGORITHM_VERSION or payload.get('protocol') != PROTOCOL:
        raise ValueError('incompatible checkpoint environment/algorithm/dimension protocol')
    try:
        dimensions=(payload['critic']['network.0.weight'].shape,payload['critic']['network.4.weight'].shape,
                    payload['actor']['network.0.weight'].shape,payload['actor']['network.4.weight'].shape)
        if dimensions!=((128,90),(5,128),(128,12),(6,128)):
            raise ValueError('incompatible checkpoint actor/critic dimensions')
        for key in ('sampled_steps','updates','training_episode_count'):
            if type(payload[key]) is not int or payload[key]<0:
                raise ValueError(f'incompatible checkpoint {key}')
        for key in ('train_config','env_config','actor_optimizer','critic_optimizer','torch_rng','vector'):
            if key not in payload: raise ValueError(f'incompatible checkpoint missing {key}')
    except (KeyError,AttributeError,TypeError) as error:
        raise ValueError('incompatible checkpoint structure') from error


def restore_checkpoint(payload, trainer, env_config, vector=None):
    validate_checkpoint(payload)
    if payload['env_config'] != env_config.to_dict() or payload['train_config'] != trainer.config.to_dict():
        raise ValueError('checkpoint configuration mismatch')
    trainer.actor.load_state_dict(payload['actor'])
    trainer.critic.load_state_dict(payload['critic'])
    trainer.actor_optimizer.load_state_dict(payload['actor_optimizer'])
    trainer.critic_optimizer.load_state_dict(payload['critic_optimizer'])
    trainer.updates = payload['updates']
    trainer.sampled_steps = payload['sampled_steps']
    trainer.training_episode_count = payload['training_episode_count']
    torch.set_rng_state(payload['torch_rng'])
    if trainer.device.type == 'cuda' and payload['cuda_rng'] is not None:
        torch.cuda.set_rng_state_all(payload['cuda_rng'])
    np.random.set_state(payload['numpy_rng'])
    random.setstate(payload['python_rng'])
    if vector is not None:
        if payload['vector'] is None:
            raise ValueError('checkpoint lacks worker state')
        vector.restore(payload['vector'])
