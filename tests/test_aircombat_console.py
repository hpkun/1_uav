"""Budget-only entry changes and read-only terminal output on actual CUDA runs."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import random
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from algorithm import experiments
from algorithm.aircombat import train
from algorithm.common.aircombat_progress import TrainingConsole,format_training,format_evaluation
from algorithm.common.aircombat_vector import AirCombatVectorEnv
from algorithm.common.checkpoint import save_protocol_checkpoint,read_protocol_checkpoint
from algorithm.mappo.trainer import ProtocolMAPPOTrainer
from env.aircombat import AirCombatConfig
from env.aircombat.config import ENVIRONMENT_VERSION
from tests.test_aircombat_training import tensor_equal
from tools.report_aircombat import final_stats


@pytest.mark.parametrize('budget',[2500,5000])
def test_budget_changes_only_target_and_retains_formal_protocol(budget):
    env,cfg=experiments.configurations('main_3v3',episodes=budget)
    base_env,base=experiments.configurations('main_3v3')
    assert env==base_env
    diff={k for k,v in cfg.to_dict().items() if v!=base.to_dict()[k]}
    assert diff==({'training_episodes'} if budget==2500 else set())
    assert cfg.training_episodes==budget and cfg.num_envs==10 and cfg.horizon==1500
    assert env.max_steps==250 and cfg.checkpoint_interval_episodes==50 and cfg.evaluation_episodes==200
    assert cfg.worker_seeds()==tuple(range(1000,1010))
    meta=experiments.manifest('main_3v3',episodes=budget)
    assert meta['training_episode_target']==budget and meta['checkpoint_targets'][-1]==budget
    assert meta['training_budget_kind']==('half_budget_diagnostic' if budget==2500 else 'paper_5000')
    assert experiments.configurations('main_3v3')[1].training_episodes==5000


@pytest.mark.parametrize('budget',[None,2500,5000])
def test_cli_passes_budget_and_console_options_without_training(monkeypatch,budget):
    calls=[]
    monkeypatch.setattr(experiments,'run_experiment',lambda *a,**kw:calls.append((a,kw)))
    argv=['train','--experiment','main_3v3','--seed','1','--log-every-updates','7','--quiet']
    if budget is not None:argv+=['--episodes',str(budget)]
    experiments.main(argv)
    assert calls[0][1]['episodes']==budget
    assert calls[0][1]['log_every_updates']==7 and calls[0][1]['console'] is False


def test_budget_metadata_output_roots_and_strict_resume(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    saved=[]
    def fake_train(env,cfg,output,**kwargs):
        saved.append((cfg,output));(output/'train.jsonl').touch()
        experiments.write_json(output/'config.json',dict(training=cfg.to_dict()))
        return SimpleNamespace(config=cfg)
    monkeypatch.setattr(experiments,'train',fake_train)
    monkeypatch.setattr(experiments,'final_evaluation',lambda *args,**kwargs:None)
    experiments.run_experiment('main_3v3',1,episodes=2500,console=False)
    experiments.run_experiment('main_3v3',1,episodes=5000,console=False)
    assert saved[0][1]!=saved[1][1]
    assert str(saved[0][1]).startswith(str(Path('outputs/xu_iv_v2_2500')))
    for cfg,path in saved:
        meta=json.loads((path/'experiment.json').read_text(encoding='utf-8'))
        assert meta['training']['training_episodes']==cfg.training_episodes
        assert json.loads((path/'config.json').read_text(encoding='utf-8'))['training']['training_episodes']==cfg.training_episodes
    with pytest.raises(FileExistsError):
        experiments.run_experiment('main_3v3',1,Path('outputs/xu_iv_v2_2500'),episodes=5000,console=False)
    env,cfg=experiments.configurations('main_3v3',episodes=2500)
    monkeypatch.setattr(experiments,'read_protocol_checkpoint',lambda *a,**kw:dict(env_config=env.to_dict(),train_config=cfg.to_dict()))
    with pytest.raises(ValueError,match='resume experiment configuration mismatch'):
        experiments.run_experiment('main_3v3',1,'resume',episodes=5000,resume='half.pt',console=False)
    for invalid in (0,2499,5001):
        with pytest.raises(ValueError):experiments.configurations('main_3v3',episodes=invalid)
    with pytest.raises(ValueError):experiments.configurations('main_3v3',episodes=2500,smoke=True)


def sample_record(update=5,episodes=None,completed=2):
    return dict(completed_episodes=completed,sampled_steps=123,updates=update,episodes=episodes or [],
        ppo=dict(actor_total_loss=.12345,policy_loss=.12,critic_loss=2.34567,entropy=-.3,
                 approx_kl=.00123,clip_fraction=.125,valid_samples=17))


def episode(outcome,ret,red,blue,win=False):
    return dict(outcome=outcome,episode_return=ret,red_survivors=red,blue_shot_down=blue,win=win)


def test_terminal_metrics_are_real_na_is_not_zero_and_numerical_is_separate():
    episodes=[episode('win',10,3,3,True),episode('numerical_failure',-4,3,0)]
    record=sample_record(episodes=episodes)
    before=copy.deepcopy(record)
    text=format_training(record,2500,episodes,elapsed=12.5,throughput=9.84)
    for token in ('2/2500','Joint steps 123','PPO updates 5','Return 3.000','Train Win Rate 50.0%',
                  'Blue shot down 1.50','Numerical Failure 1 (50.0%)','Actor Total Loss 0.12345',
                  'Policy Loss 0.12000','Critic Loss 2.34567','Entropy -0.30000','Approx KL 0.001230',
                  'Clip Fraction 0.12500','Valid Samples 17','00:00:13','9.8 joint steps/s'):
        assert token in text
    assert record==before and 'episodes":' not in text
    empty=format_training(sample_record(completed=0),5000,[],elapsed=1.,throughput=1.)
    assert 'Train Win Rate N/A' in empty and 'Return N/A' in empty
    assert 'Train Win Rate 0.0%' not in empty
    assert 'Numerical Failure 0 (N/A)' in empty


def test_frequency_window_and_checkpoint_lines_are_independent_of_ppo(capsys):
    display=TrainingConsole(5000,every=5,clock=lambda:10.,started_at=0.)
    for i in range(1,7):
        display.update(sample_record(i,[episode('timeout',8,2,1)] if i==2 else [],completed=1 if i>=2 else 0))
    output=capsys.readouterr().out
    assert output.count('[TRAIN]')==2 # first update and fifth
    assert 'Train Win Rate N/A' in output and 'Recent n=1/100 | Return 8.000' in output
    display.update(sample_record(7,completed=50))
    assert '[TRAIN]' in capsys.readouterr().out
    display.checkpoint(SimpleNamespace(training_episode_count=50,updates=7),Path('episodes_000050.pt'))
    assert '[CKPT] Episodes 50/5000 | PPO updates 7 | File episodes_000050.pt' in capsys.readouterr().out


def test_final_evaluation_reports_source_budget_and_200_episodes(tmp_path,monkeypatch,capsys):
    env,cfg=experiments.configurations('main_3v3',episodes=2500)
    model=SimpleNamespace(config=cfg)
    monkeypatch.setattr(experiments,'load_model',lambda *a:(model,env,dict(training_episode_count=2500)))
    counts=dict(normal=180,timeout=12,numerical_failure=8)
    result=dict(completed_episodes=200,win_rate=.5,mean_red_survivors=2.25,mean_blue_shot_down=2.1,termination_counts=counts)
    def fake_eval(t,e,seeds):
        assert len(seeds)==200 and not set(seeds)&set(cfg.worker_seeds())
        return result.copy()
    monkeypatch.setattr(experiments,'evaluate',fake_eval)
    saved=experiments.final_evaluation(tmp_path/'half.pt',tmp_path/'eval.json')
    text=capsys.readouterr().out
    for token in ('[EVAL]','independent n=200','Win Rate 50.0%','Red alive 2.25',
                  'Blue shot down 2.10','Numerical Failure 8 (4.0%)','Timeout 12 (6.0%)','half.pt'):
        assert token in text
    assert saved['training_episode_target']==2500 and saved['training_budget_kind']=='half_budget_diagnostic'
    runs=[dict(meta=experiments.manifest('main_3v3',s,episodes=2500),
               evaluation=dict(result,completed_training_episodes=2500)) for s in range(1,6)]
    assert final_stats(runs,'main_3v3','win_rate')['status']!='formal_complete'


def test_console_switch_and_frequency_preserve_full_cuda_state(tmp_path,capsys):
    from algorithm.mappo.config import ProtocolTrainConfig
    c=ProtocolTrainConfig(num_envs=2,horizon=2,ppo_epochs=1,minibatch_size=16,
                          training_episodes=3,checkpoint_interval_episodes=1)
    results=[]
    for label,enabled,every in (('quiet',False,5),('frequent',True,1),('sparse',True,5)):
        path=tmp_path/label
        trainer=train(AirCombatConfig(max_steps=3),c,path,console=enabled,log_every_updates=every)
        state=read_protocol_checkpoint(path/'latest.pt',environment_version=ENVIRONMENT_VERSION)
        # worker PID is only printed; worker snapshots are independent of it.
        rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
        records=(path/'train.jsonl').read_bytes()
        results.append((state,rng,records))
        output=capsys.readouterr().out
        if enabled:
            assert '[START]' in output and 'Workers 2 (spawn)' in output and '[DONE]' in output
            assert '[CKPT]' in output and '"episodes": [' not in output
        else:assert not output
        assert trainer.training_episode_count==3
    tensor_equal(results[0],results[1]);tensor_equal(results[0],results[2])


def test_actual_ten_spawn_workers_and_checkpoint_budgets(tmp_path,capsys):
    with AirCombatVectorEnv(AirCombatConfig(),10,training_seed_index=1) as vector:
        assert len(set(vector.worker_pids))==10 and all(p.is_alive() for p in vector.processes)
        for budget in (2500,5000):
            env,cfg=experiments.configurations('main_3v3',episodes=budget)
            trainer=ProtocolMAPPOTrainer(cfg)
            display=TrainingConsole(budget)
            display.start(trainer,vector,env,tmp_path)
            path=tmp_path/f'{budget}.pt'
            save_protocol_checkpoint(path,trainer,env_config=env,vector=vector)
            payload=read_protocol_checkpoint(path,environment_version=ENVIRONMENT_VERSION)
            assert payload['train_config']['training_episodes']==budget
            assert payload['train_config']['horizon']==1500 and len(payload['vector']['workers'])==10
    assert capsys.readouterr().out.count('Workers 10 (spawn)')==2
