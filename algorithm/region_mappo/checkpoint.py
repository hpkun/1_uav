"""Trusted local region checkpoints; rejects inspection identities before unpickling."""
from pathlib import Path
import random
import zipfile
import numpy as np
import torch
from algorithm.common.runtime import require_cuda
from env.region_competition.config import ENVIRONMENT_VERSION
from .config import ALGORITHM_VERSION, PROTOCOL


def save_checkpoint(path,trainer,env_config,vector=None,best_score=None):
    require_cuda(str(trainer.device))
    data=dict(environment_version=ENVIRONMENT_VERSION,algorithm_version=ALGORITHM_VERSION,protocol=PROTOCOL.copy(),
              env_config=env_config.to_dict(),train_config=trainer.config.to_dict(),actor=trainer.actor.state_dict(),critic=trainer.critic.state_dict(),
              actor_optimizer=trainer.actor_optimizer.state_dict(),critic_optimizer=trainer.critic_optimizer.state_dict(),
              sampled_steps=trainer.sampled_steps,updates=trainer.updates,training_episode_count=trainer.training_episode_count,best_score=best_score,
              torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),
              vector=vector.snapshot() if vector else None)
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp'); torch.save(data,temp); temp.replace(path)


def validate(data):
    if data.get('environment_version')!=ENVIRONMENT_VERSION or data.get('algorithm_version')!=ALGORITHM_VERSION or data.get('protocol')!=PROTOCOL:
        raise ValueError('incompatible region checkpoint identity/protocol')
    try:
        dimensions=(data['critic']['network.0.weight'].shape,data['critic']['network.4.weight'].shape,
                    data['actor']['network.0.weight'].shape,data['actor']['network.4.weight'].shape)
        if dimensions!=((128,60),(5,128),(128,12),(6,128)): raise ValueError('incompatible region network dimensions')
        for key in ('sampled_steps','updates','training_episode_count'):
            if type(data[key]) is not int or data[key]<0: raise ValueError('invalid checkpoint counters')
        for key in ('actor_optimizer','critic_optimizer','env_config','train_config','vector','torch_rng','cuda_rng','numpy_rng','python_rng'):
            if key not in data: raise ValueError('incomplete checkpoint: '+key)
    except (KeyError,AttributeError,TypeError) as exc:
        raise ValueError('invalid region checkpoint structure') from exc


def read_checkpoint(path):
    require_cuda('cuda')
    try:
        with zipfile.ZipFile(path) as archive:
            metadata=archive.read(next(n for n in archive.namelist() if n.endswith('/data.pkl')))
            if ENVIRONMENT_VERSION.encode() not in metadata or ALGORITHM_VERSION.encode() not in metadata:
                raise ValueError('incompatible region checkpoint (inspection models are not region models)')
    except (zipfile.BadZipFile,KeyError,StopIteration) as exc:
        raise ValueError('incompatible checkpoint format') from exc
    data=torch.load(path,map_location='cpu',weights_only=False); validate(data); return data


def restore_checkpoint(data,trainer,env_config,vector=None):
    require_cuda(str(trainer.device)); validate(data)
    if data['env_config']!=env_config.to_dict() or data['train_config']!=trainer.config.to_dict(): raise ValueError('checkpoint configuration mismatch')
    trainer.actor.load_state_dict(data['actor']); trainer.critic.load_state_dict(data['critic'])
    trainer.actor_optimizer.load_state_dict(data['actor_optimizer']); trainer.critic_optimizer.load_state_dict(data['critic_optimizer'])
    trainer.sampled_steps=data['sampled_steps']; trainer.updates=data['updates']; trainer.training_episode_count=data['training_episode_count']
    torch.set_rng_state(data['torch_rng']); torch.cuda.set_rng_state_all(data['cuda_rng'])
    np.random.set_state(data['numpy_rng']); random.setstate(data['python_rng'])
    if vector is not None:
        if data['vector'] is None: raise ValueError('checkpoint lacks worker snapshot')
        vector.restore(data['vector'])
