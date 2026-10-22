"""Complete trusted local checkpoints with optimizer, RNG and worker state."""
from pathlib import Path
import random
import numpy as np
import torch
from env.config import ENVIRONMENT_VERSION
from .protocol import ALGORITHM_VERSION, PROTOCOL
import zipfile
import json


def save_checkpoint(path, trainer, env_config, vector=None, best_score=None):
    if trainer.protocol != PROTOCOL or trainer.algorithm_version != ALGORITHM_VERSION:
        raise ValueError('incompatible checkpoint: inspection trainer protocol required')
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
                raise ValueError('incompatible checkpoint: environment/dynamics/algorithm identity; no automatic migration')
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
    if trainer.protocol != PROTOCOL or trainer.algorithm_version != ALGORITHM_VERSION:
        raise ValueError('incompatible checkpoint: inspection trainer protocol required')
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


def _validate_array_checkpoint(payload):
    from algorithm.mappo.paper_protocol import ALGORITHM_VERSION as identity, PROTOCOL as protocol
    if payload.get('algorithm_version') != identity or payload.get('protocol') != protocol:
        raise ValueError('incompatible checkpoint: independent 12/60 protocol required')
    try:
        if (tuple(payload['actor']['network.0.weight'].shape) != (128, 12)
                or tuple(payload['actor']['network.4.weight'].shape) != (6, 128)
                or tuple(payload['critic']['network.0.weight'].shape) != (128, 60)):
            raise ValueError('incompatible checkpoint: network dimensions')
        for slot in range(5):
            if (tuple(payload['critic'][f'value_heads.{slot}.weight'].shape) != (1, 128)
                    or tuple(payload['critic'][f'value_heads.{slot}.bias'].shape) != (1,)):
                raise ValueError('incompatible checkpoint: individual value heads')
        for name in ('updates', 'sampled_steps', 'training_episode_count'):
            if type(payload[name]) is not int or payload[name] < 0:
                raise ValueError('incompatible checkpoint: counters')
        for name in ('train_config', 'actor_optimizer', 'critic_optimizer', 'torch_rng', 'cuda_rng', 'numpy_rng', 'python_rng'):
            if name not in payload:
                raise ValueError('incompatible checkpoint: missing resume state')
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('incompatible checkpoint: incomplete array protocol') from exc


def save_protocol_checkpoint(path, trainer):
    """Save only the independent generic updater, never an environment/reward.

    A JSON identity inside the torch ZIP is checked before trusted unpickling.
    The five-head/shared-parameter/reduction assumptions are part of identity.
    """
    from algorithm.mappo.paper_protocol import ALGORITHM_VERSION as identity, PROTOCOL as protocol
    if trainer.algorithm_version != identity or trainer.protocol != protocol:
        raise ValueError('incompatible checkpoint: independent 12/60 trainer required')
    payload = dict(algorithm_version=identity, protocol=protocol.copy(), train_config=trainer.config.to_dict(),
                   actor=trainer.actor.state_dict(), critic=trainer.critic.state_dict(),
                   actor_optimizer=trainer.actor_optimizer.state_dict(), critic_optimizer=trainer.critic_optimizer.state_dict(),
                   updates=trainer.updates, sampled_steps=trainer.sampled_steps, training_episode_count=trainer.training_episode_count,
                   torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                   numpy_rng=np.random.get_state(), python_rng=random.getstate())
    _validate_array_checkpoint(payload)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix+'.tmp')
    torch.save(payload, temp)
    with zipfile.ZipFile(temp, 'a') as archive:
        prefix = next(n for n in archive.namelist() if n.endswith('/data.pkl')).rsplit('/', 1)[0]
        archive.writestr(prefix+'/protocol.json', json.dumps(dict(algorithm_version=identity, protocol=protocol)))
    temp.replace(p)


def read_protocol_checkpoint(path):
    from algorithm.mappo.paper_protocol import ALGORITHM_VERSION as identity, PROTOCOL as protocol
    from .runtime import require_cuda
    require_cuda('cuda')
    try:
        with zipfile.ZipFile(path) as archive:
            name = next(n for n in archive.namelist() if n.endswith('/protocol.json'))
            header = json.loads(archive.read(name))
            if header != dict(algorithm_version=identity, protocol=protocol):
                raise ValueError('incompatible checkpoint: array protocol identity')
    except (zipfile.BadZipFile, StopIteration, KeyError, json.JSONDecodeError) as exc:
        raise ValueError('incompatible checkpoint: no 12/60 protocol identity') from exc
    payload = torch.load(path, map_location='cpu', weights_only=False)
    _validate_array_checkpoint(payload)
    return payload


def restore_protocol_checkpoint(payload, trainer):
    _validate_array_checkpoint(payload)
    if trainer.algorithm_version != payload['algorithm_version'] or trainer.protocol != payload['protocol']:
        raise ValueError('incompatible checkpoint: trainer identity')
    if trainer.config.to_dict() != payload['train_config']:
        raise ValueError('checkpoint configuration mismatch')
    trainer.actor.load_state_dict(payload['actor'])
    trainer.critic.load_state_dict(payload['critic'])
    trainer.actor_optimizer.load_state_dict(payload['actor_optimizer'])
    trainer.critic_optimizer.load_state_dict(payload['critic_optimizer'])
    trainer.updates, trainer.sampled_steps = payload['updates'], payload['sampled_steps']
    trainer.training_episode_count = payload['training_episode_count']
    torch.set_rng_state(payload['torch_rng'])
    torch.cuda.set_rng_state_all(payload['cuda_rng'])
    np.random.set_state(payload['numpy_rng'])
    random.setstate(payload['python_rng'])
