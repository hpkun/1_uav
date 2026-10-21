"""Strict sampled-step training schedule; no environment rules live here."""
import csv
import json
import platform
from pathlib import Path
import torch
from env.config import ENVIRONMENT_VERSION
from algorithm.common.checkpoint import read_checkpoint,restore_checkpoint,save_checkpoint
from algorithm.common.metrics import episode_metrics
from algorithm.common.protocol import ALGORITHM_VERSION,PROTOCOL
from algorithm.common.runtime import require_cuda
from algorithm.common.serialization import serializable,write_json
from algorithm.common.vector_env import ParallelVectorEnv
from algorithm.common.progress import ProgressClock,format_training,format_evaluation,format_final
from algorithm.common.resume_logs import check_resume_logs
from .trainer import MAPPOTrainer
from .evaluation import evaluate,evaluation_seeds


def train(env_config,train_config,output,total_sampled_steps,*,resume=None,device='cuda'):
    require_cuda(device)
    if type(total_sampled_steps) is not int or total_sampled_steps<1: raise ValueError('total_sampled_steps must be positive integer')
    output=Path(output)
    if output.exists() and any(output.iterdir()) and not resume: raise FileExistsError('refusing to overwrite nonempty output directory')
    payload=read_checkpoint(resume) if resume else None
    if payload and (payload['env_config']!=env_config.to_dict() or payload['train_config']!=train_config.to_dict()):
        raise ValueError('checkpoint configuration mismatch')
    if payload and total_sampled_steps<payload['sampled_steps']: raise ValueError('new total budget is below checkpoint sampled_steps')
    if payload:
        for warning in check_resume_logs(output,resume,payload,read_checkpoint):
            print(f'[RESUME] {warning}',flush=True)
    output.mkdir(parents=True,exist_ok=True)
    if payload and not (output/'best.pt').exists():
        import shutil
        source=Path(resume).parent/'best.pt'
        if source.exists(): shutil.copy2(source,output/'best.pt')
    trainer=MAPPOTrainer(train_config,device=device)
    best=payload['best_score'] if payload else None
    seeds=evaluation_seeds(train_config)
    write_json(output/'run_config.json',dict(environment_version=ENVIRONMENT_VERSION,algorithm_version=ALGORITHM_VERSION,
        environment=env_config.to_dict(),algorithm=train_config.to_dict(),protocol=PROTOCOL,total_sampled_steps=total_sampled_steps,
        runtime=dict(python=platform.python_version(),torch=torch.__version__,gpu=torch.cuda.get_device_name(),device=str(trainer.device))))
    history_path=output/'evaluation_history.csv'
    if not history_path.exists() or history_path.stat().st_size==0:
        with history_path.open('w',newline='',encoding='utf-8') as f: csv.writer(f).writerow(['sampled_steps','update','checkpoint_role','success_rate','completion_rate','mean_return'])
    with ParallelVectorEnv(env_config,train_config.num_envs,train_config.seed,seeds) as vector:
        if payload: restore_checkpoint(payload,trainer,env_config,vector)
        progress_clock=ProgressClock(trainer.sampled_steps)
        def next_due(interval): return (trainer.sampled_steps//interval+1)*interval
        eval_due=next_due(train_config.evaluation_interval_sampled_steps)
        checkpoint_due=next_due(train_config.checkpoint_interval_sampled_steps)
        console_due=next_due(train_config.console_interval_sampled_steps)
        while trainer.sampled_steps<total_sampled_steps:
            budget=min(train_config.horizon*train_config.num_envs,total_sampled_steps-trainer.sampled_steps)
            rollout,episodes=trainer.collect(vector,int(budget))
            optimization=trainer.update(rollout)
            record=dict(sampled_steps=trainer.sampled_steps,joint_episode_count=trainer.training_episode_count,update=trainer.updates,
                        training=episode_metrics(episodes),rollout=trainer.rollout_metrics,optimization=optimization)
            if trainer.sampled_steps>=eval_due or trainer.sampled_steps==total_sampled_steps:
                report=evaluate(trainer,env_config,seeds)
                score=(report['success_rate'],report['completion_rate'],report['mean_reward'])
                record['independent_evaluation']=report
                best_updated=best is None or score>tuple(best)
                record['best_updated']=best_updated
                with history_path.open('a',newline='',encoding='utf-8') as f:
                    csv.writer(f).writerow([trainer.sampled_steps,trainer.updates,'current',*score])
                if best_updated:
                    best=score
                    save_checkpoint(output/'best.pt',trainer,env_config,vector,best)
                print(format_evaluation(trainer.sampled_steps,report,env_config.num_waypoints,best_updated=best_updated),flush=True)
                eval_due=next_due(train_config.evaluation_interval_sampled_steps)
            if trainer.sampled_steps>=checkpoint_due or trainer.sampled_steps==total_sampled_steps:
                save_checkpoint(output/'latest.pt',trainer,env_config,vector,best)
                checkpoint_due=next_due(train_config.checkpoint_interval_sampled_steps)
            with (output/'training_metrics.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(serializable(record),allow_nan=False)+'\n')
            if trainer.sampled_steps>=console_due or trainer.sampled_steps==total_sampled_steps:
                speed,eta=progress_clock.measure(trainer.sampled_steps,total_sampled_steps)
                print(format_training(record,total_sampled_steps,speed=speed,eta=eta),flush=True)
                console_due=next_due(train_config.console_interval_sampled_steps)
        save_checkpoint(output/'latest.pt',trainer,env_config,vector,best)
        latest=evaluate(trainer,env_config,seeds)
        # Best and latest can be different updates. Evaluate them independently.
        best_path=output/'best.pt'
        if not best_path.exists() and resume:
            candidate=Path(resume).parent/'best.pt'
            if candidate.exists(): best_path=candidate
        best_report=None
        best_payload=None
        if best_path.exists():
            best_trainer=MAPPOTrainer(train_config,device=device)
            best_payload=read_checkpoint(best_path)
            restore_checkpoint(best_payload,best_trainer,env_config)
            best_report=evaluate(best_trainer,env_config,seeds)
            # Creating the best evaluator must not disturb resumed training RNG.
            restore_checkpoint(read_checkpoint(output/'latest.pt'),trainer,env_config)
        summary=dict(environment_version=ENVIRONMENT_VERSION,algorithm_version=ALGORITHM_VERSION,
            sampled_steps=trainer.sampled_steps,updates=trainer.updates,joint_episode_count=trainer.training_episode_count,
            requested_sampled_steps=total_sampled_steps,latest_checkpoint=str((output/'latest.pt').resolve()),
            best_checkpoint=str(best_path.resolve()) if best_payload else None,
            best_checkpoint_sampled_steps=best_payload['sampled_steps'] if best_payload else None,
            best_checkpoint_updates=best_payload['updates'] if best_payload else None,
            latest_evaluation=latest,best_evaluation=best_report,best_score=best)
        write_json(output/'run_summary.json',summary)
        print(format_final(summary,output,env_config.num_waypoints),flush=True)
    return trainer,summary
