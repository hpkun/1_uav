"""Explicit-budget region training, independent holdout evaluation, complete resume."""
from pathlib import Path
import json
import shutil
import torch
from algorithm.common.runtime import require_cuda
from algorithm.common.serialization import write_json, serializable
from algorithm.mappo.evaluation import evaluation_seeds
from .config import ALGORITHM_VERSION, PROTOCOL
from .trainer import RegionTrainer
from .vector_env import RegionVectorEnv
from .evaluation import evaluate
from .checkpoint import save_checkpoint, read_checkpoint, restore_checkpoint
from .metrics import aggregate


def train(env_config,train_config,output,total_sampled_steps,*,resume=None,device='cuda'):
    require_cuda(device)
    if type(total_sampled_steps) is not int or total_sampled_steps<1: raise ValueError('positive absolute joint budget required')
    output=Path(output); data=read_checkpoint(resume) if resume else None
    if data and (data['env_config']!=env_config.to_dict() or data['train_config']!=train_config.to_dict()): raise ValueError('resume configuration mismatch')
    if data and total_sampled_steps<data['sampled_steps']: raise ValueError('budget precedes checkpoint')
    if output.exists() and any(output.iterdir()):
        if data is None: raise FileExistsError('output must be empty for a new experiment')
        log=output/'training_metrics.jsonl'
        if log.exists():
            steps=[json.loads(line)['sampled_steps'] for line in log.read_text(encoding='utf-8').splitlines() if line.strip()]
            if steps and (max(steps)>data['sampled_steps'] or any(a>=b for a,b in zip(steps,steps[1:]))): raise ValueError('logs are ahead of checkpoint or out of order')
        for role in ('best','latest'):
            path=output/(role+'.pt')
            if path.exists() and read_checkpoint(path)['sampled_steps']>data['sampled_steps']: raise ValueError('output checkpoint is newer than resume')
    output.mkdir(parents=True,exist_ok=True)
    if data and not (output/'best.pt').exists():
        source=Path(resume).parent/'best.pt'
        if source.exists(): shutil.copy2(source,output/'best.pt')
    trainer=RegionTrainer(train_config,device=device); best=data['best_score'] if data else None
    seeds=evaluation_seeds(train_config)
    write_json(output/'run_config.json',dict(environment=env_config.to_dict(),algorithm=train_config.to_dict(),protocol=PROTOCOL,
               algorithm_version=ALGORITHM_VERSION,total_sampled_steps=total_sampled_steps,device=torch.cuda.get_device_name()))
    with RegionVectorEnv(env_config,train_config.num_envs,train_config.seed,seeds) as vector:
        if data: restore_checkpoint(data,trainer,env_config,vector)
        due=(trainer.sampled_steps//train_config.evaluation_interval_sampled_steps+1)*train_config.evaluation_interval_sampled_steps
        episode_due=(trainer.training_episode_count//train_config.checkpoint_interval_episodes+1)*train_config.checkpoint_interval_episodes
        while trainer.sampled_steps<total_sampled_steps:
            budget=min(train_config.horizon*train_config.num_envs,total_sampled_steps-trainer.sampled_steps)
            batch,episodes=trainer.collect(vector,budget); optimization=trainer.update(batch)
            record=dict(sampled_steps=trainer.sampled_steps,update=trainer.updates,episodes=trainer.training_episode_count,
                        rollout=trainer.rollout_metrics,training=aggregate(episodes),optimization=optimization)
            final=trainer.sampled_steps==total_sampled_steps
            if trainer.sampled_steps>=due or final:
                report=evaluate(trainer,env_config,seeds); record['evaluation']=report
                score=(report['mean_relative_score'],report['red_win_rate'],report['mean_episode_return'])
                if best is None or score>tuple(best):
                    best=score; save_checkpoint(output/'best.pt',trainer,env_config,vector,best)
                due=(trainer.sampled_steps//train_config.evaluation_interval_sampled_steps+1)*train_config.evaluation_interval_sampled_steps
            if trainer.training_episode_count>=episode_due or final:
                save_checkpoint(output/'latest.pt',trainer,env_config,vector,best)
                episode_due=(trainer.training_episode_count//train_config.checkpoint_interval_episodes+1)*train_config.checkpoint_interval_episodes
            with (output/'training_metrics.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(serializable(record),allow_nan=False)+'\n')
            print(f'Region steps={trainer.sampled_steps}/{total_sampled_steps} update={trainer.updates} valid={optimization["valid_samples"]}',flush=True)
        save_checkpoint(output/'latest.pt',trainer,env_config,vector,best)
        latest=evaluate(trainer,env_config,seeds)
        best_data=read_checkpoint(output/'best.pt') if (output/'best.pt').exists() else None
        best_report=None
        if best_data:
            best_trainer=RegionTrainer(train_config,device=device); restore_checkpoint(best_data,best_trainer,env_config)
            best_report=evaluate(best_trainer,env_config,seeds)
            restore_checkpoint(read_checkpoint(output/'latest.pt'),trainer,env_config)
        summary=dict(sampled_steps=trainer.sampled_steps,updates=trainer.updates,training_episodes=trainer.training_episode_count,
                     latest=latest,best=best_report,best_sampled_steps=best_data['sampled_steps'] if best_data else None,
                     best_update=best_data['updates'] if best_data else None)
        write_json(output/'run_summary.json',summary)
    return trainer,summary
