"""Atomic checkpoints. Only load trusted local files."""
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import tempfile
import zipfile

import numpy as np
import torch

from env.interception.config import VERSION, OBS_DIM, STATE_DIM, MAX_PURSUERS, MAX_EVADERS, AXES, matches_saved_config
from .runtime import require_cuda

PROTOCOL = dict(version=VERSION, algorithm="uav-interception-mappo-v4",
                observation=OBS_DIM, state=STATE_DIM, action=AXES,
                pursuers=MAX_PURSUERS, evaders=MAX_EVADERS)


def save(path, model, config, vector, last_log=None, *, run=None):
    environment = asdict(config)
    if run is not None:
        # A resumed historical run keeps its originally committed schema.
        environment = {key: environment[key] for key in run["environment"]}
    payload = dict(protocol=PROTOCOL, environment=environment, training=asdict(model.config),
                   actor=model.actor.state_dict(), critic=model.critic.state_dict(),
                   actor_optimizer=model.actor_optimizer.state_dict(),
                   critic_optimizer=model.critic_optimizer.state_dict(),
                   torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                   numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                   updates=model.updates, steps=model.steps, episodes=model.episodes,
                   vector=vector.snapshot(), last_log=last_log)
    if run is not None:
        payload["research_run"] = run
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            torch.save(payload, stream)
            stream.flush()
            with zipfile.ZipFile(stream.name, "a") as archive:
                root = archive.namelist()[0].split("/")[0]
                archive.writestr(root + "/interception_protocol.json", json.dumps(PROTOCOL))
                if run is not None:
                    archive.writestr(root + "/research_commit.json", json.dumps(
                        dict(run=run, update=model.updates, last_log=last_log), allow_nan=False))
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read(path):
    require_cuda()
    try:
        with zipfile.ZipFile(path) as archive:
            name = next(n for n in archive.namelist() if n.endswith("/interception_protocol.json"))
            header = json.loads(archive.read(name))
    except (zipfile.BadZipFile, StopIteration, ValueError) as exc:
        raise ValueError("incompatible checkpoint: interception protocol header required") from exc
    if header != PROTOCOL:
        raise ValueError("incompatible checkpoint protocol")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("protocol") != PROTOCOL:
        raise ValueError("incompatible checkpoint payload")
    return payload


def load_weights(payload, model):
    model.actor.load_state_dict(payload["actor"])
    model.critic.load_state_dict(payload["critic"])


def restore(payload, model, config, vector):
    if not matches_saved_config(payload["environment"], config) or payload["training"] != asdict(model.config):
        raise ValueError("checkpoint configuration mismatch")
    load_weights(payload, model)
    model.actor_optimizer.load_state_dict(deepcopy(payload["actor_optimizer"]))
    model.critic_optimizer.load_state_dict(deepcopy(payload["critic_optimizer"]))
    model.updates, model.steps, model.episodes = (payload[key] for key in ("updates", "steps", "episodes"))
    vector.restore(payload["vector"])
    torch.set_rng_state(payload["torch_rng"])
    torch.cuda.set_rng_state_all(payload["cuda_rng"])
    np.random.set_state(payload["numpy_rng"])
    random.setstate(payload["python_rng"])
