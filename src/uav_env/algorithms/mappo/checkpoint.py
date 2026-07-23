"""Versioned exact MAPPO checkpoint persistence."""

from __future__ import annotations
import random
from pathlib import Path
from typing import Any
import numpy as np
import torch

CHECKPOINT_VERSION=2

def save_checkpoint(path: Path, actor, critic, actor_optimizer, critic_optimizer, normalizer, config: dict[str,Any], environment_steps: int, update_index: int, best_evaluation: dict[str,Any] | None, runner_state: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    torch.save({"version":CHECKPOINT_VERSION,"actor":actor.state_dict(),"critic":critic.state_dict(),"actor_optimizer":actor_optimizer.state_dict(),"critic_optimizer":critic_optimizer.state_dict(),"value_normalizer":normalizer.state_dict(),"config":config,"environment_steps":environment_steps,"update_index":update_index,"best_evaluation":best_evaluation,"runner_state":runner_state,"python_rng":random.getstate(),"numpy_rng":np.random.get_state(),"torch_rng":torch.get_rng_state(),"torch_cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},temporary)
    temporary.replace(path)

def load_checkpoint(path: str|Path, actor, critic=None, actor_optimizer=None, critic_optimizer=None, normalizer=None, actor_only: bool=False, map_location="cpu") -> dict[str,Any]:
    data=torch.load(path,map_location=map_location,weights_only=False)
    try: actor.load_state_dict(data["actor"])
    except RuntimeError as error: raise ValueError(f"Actor dimensions are incompatible: {error}") from error
    if not actor_only:
        if critic is None or normalizer is None: raise ValueError("Full resume requires critic and normalizer")
        critic.load_state_dict(data["critic"]); normalizer.load_state_dict(data["value_normalizer"])
        if actor_optimizer is not None: actor_optimizer.load_state_dict(data["actor_optimizer"])
        if critic_optimizer is not None: critic_optimizer.load_state_dict(data["critic_optimizer"])
        random.setstate(data["python_rng"]); np.random.set_state(data["numpy_rng"]); torch.set_rng_state(data["torch_rng"].cpu())
        if torch.cuda.is_available() and data.get("torch_cuda_rng"): torch.cuda.set_rng_state_all([state.cpu() for state in data["torch_cuda_rng"]])
    return data
