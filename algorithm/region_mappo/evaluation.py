from pathlib import Path
import numpy as np
import torch
from env.region_competition import RegionCompetitionEnv
from algorithm.common.runtime import require_cuda
from algorithm.common.serialization import serializable, write_json
from algorithm.mappo.evaluation import ensure_independent
from .metrics import RegionAccumulator, aggregate


def evaluate(trainer,env_config,seeds,*,trace_path=None,red_rule=None,provenance=None):
    require_cuda(str(trainer.device))
    rows=[]
    for index,seed in enumerate(seeds):
        env=RegionCompetitionEnv(env_config); obs,info=env.reset(int(seed))
        tracker=RegionAccumulator(); tracker.reset(info)
        trace=dict(environment_version=env.version,config=env_config.to_dict(),seed=int(seed),
                   provenance=provenance or dict(policy=red_rule or 'mappo_mean',sampled_steps=trainer.sampled_steps,update=trainer.updates),
                   frames=[serializable(info)]) if trace_path and index==0 else None
        while True:
            if red_rule:
                actions=env.policy_actions('red',red_rule); diagnostics=None
            else:
                with torch.no_grad():
                    dist=trainer.actor.distribution(trainer.tensor(np.where(info['red']['active_mask'][:,None],obs,0)))
                    actions=dist.mean.cpu().numpy()
                    diagnostics=dict(selection='mean',alpha=dist.concentration1.cpu().numpy(),beta=dist.concentration0.cpu().numpy(),
                                     log_probability=dist.log_prob(dist.mean).sum(-1).cpu().numpy(),entropy=dist.entropy().sum(-1).cpu().numpy())
            obs,rewards,term,trunc,info=env.step(actions)
            tracker.add(info,rewards)
            if trace is not None:
                frame=serializable(info)
                if diagnostics is not None: frame['policy_diagnostics']=serializable(diagnostics)
                trace['frames'].append(frame)
            if term or trunc: break
        rows.append(tracker.summary(info))
        if trace is not None:
            path=Path(trace_path)
            if path.exists(): raise FileExistsError('refusing to overwrite recorded trajectory')
            write_json(path,trace)
    if not rows: raise ValueError('nonempty independent evaluation seeds required')
    return aggregate(rows)


def check_seeds(seeds,train_config,snapshot):
    if len(set(seeds))!=len(seeds) or not seeds or any(type(s) is not int or s<0 for s in seeds): raise ValueError('invalid evaluation seeds')
    if snapshot is not None: ensure_independent(seeds,train_config.seed,train_config.num_envs,snapshot['episodes'])
