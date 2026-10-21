"""Training and evaluation entrypoints, with CUDA enforced before sampling."""
import json
import platform
from pathlib import Path
import torch
from .checkpoint import read_checkpoint, restore_checkpoint, save_checkpoint
from .evaluation import evaluate
from .trainer import MAPPOTrainer, require_cuda
from ..config import ENVIRONMENT_VERSION
from ..recording import serializable
from ..vector_env import ParallelVectorEnv
from ..replay import render_replay


def write_json(path, value):
    Path(path).write_text(json.dumps(serializable(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def train(env_config, train_config, output, updates, resume=None):
    require_cuda()
    if updates < 1:
        raise ValueError('updates must be positive')
    output = Path(output)
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f'refusing to overwrite nonempty run: {output}')
    output.mkdir(parents=True, exist_ok=True)
    trainer = MAPPOTrainer(train_config)
    payload = read_checkpoint(resume) if resume else None
    if payload and (payload['env_config'] != env_config.to_dict() or payload['train_config'] != train_config.to_dict()):
        raise ValueError('checkpoint configuration mismatch')
    best = None
    with ParallelVectorEnv(env_config, train_config.num_envs, train_config.seed, train_config.evaluation_seeds) as vector:
        if resume:
            restore_checkpoint(payload, trainer, env_config, vector)
            best = payload['best_score']
        write_json(output / 'config.json', dict(environment_version=ENVIRONMENT_VERSION, environment=env_config.to_dict(),
                   training=train_config.to_dict(), runtime=dict(python=platform.python_version(), torch=torch.__version__,
                   gpu=torch.cuda.get_device_name(), device='cuda')))
        for _ in range(updates):
            rollout, episodes = trainer.collect(vector)
            metrics = trainer.update(rollout)
            evaluation = evaluate(trainer, env_config, train_config.evaluation_seeds)
            score = (evaluation['success_rate'], evaluation['mean_completion_rate'], evaluation['mean_reward'])
            record = dict(update=trainer.updates, metrics=metrics, evaluation=evaluation,
                          rollout_reward_components=trainer.rollout_components,
                          rollout_numerical_counters_max=trainer.rollout_counters,
                          worker_pids=vector.worker_pids, completed_training_episodes=len(episodes),
                          training_episodes=[{k: info[k] for k in ('success', 'episode_length', 'completion_rate', 'numerical_counters')} for info in episodes])
            # Reward component totals over every sampled transition, including partial episodes.
            record['rollout_reward_sum'] = float(rollout.rewards.sum().item())
            record['current_valid_samples'] = int(rollout.current_masks.sum().item())
            record['next_valid_samples'] = int(rollout.next_masks.sum().item())
            with (output / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(serializable(record), allow_nan=False) + '\n')
            if best is None or score > tuple(best):
                best = score
                save_checkpoint(output / 'best.pt', trainer, env_config, vector, best)
            save_checkpoint(output / 'latest.pt', trainer, env_config, vector, best)
            print(json.dumps(dict(**metrics, success_rate=evaluation['success_rate'], completion_rate=evaluation['mean_completion_rate'])), flush=True)
        evaluation = evaluate(trainer, env_config, train_config.evaluation_seeds, output / 'trajectory.json')
        write_json(output / 'evaluation.json', evaluation)
        render_replay(output / 'trajectory.json', output / 'replay.html')
    return trainer, evaluation
