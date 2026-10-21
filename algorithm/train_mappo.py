"""CUDA training CLI. Budgets count joint transitions, including partial batches."""


def main():
    import argparse
    from env.config import EnvConfig
    from algorithm.mappo.config import TrainConfig
    from algorithm.common.checkpoint import read_checkpoint
    from algorithm.mappo.runner import train
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--env-config'); p.add_argument('--algorithm-config')
    p.add_argument('--device',default='cuda'); p.add_argument('--seed',type=int)
    p.add_argument('--num-envs',type=int); p.add_argument('--total-sampled-steps',required=True,type=int)
    p.add_argument('--output-dir',required=True); p.add_argument('--resume')
    a=p.parse_args()
    saved=read_checkpoint(a.resume) if a.resume else None
    ec=EnvConfig.load(a.env_config) if a.env_config else EnvConfig(**saved['env_config']) if saved else EnvConfig()
    tc=TrainConfig.load(a.algorithm_config) if a.algorithm_config else TrainConfig(**saved['train_config']) if saved else TrainConfig()
    if a.seed is not None: tc.seed=a.seed
    if a.num_envs is not None: tc.num_envs=a.num_envs
    tc.__post_init__()
    train(ec,tc,a.output_dir,a.total_sampled_steps,resume=a.resume,device=a.device)


if __name__=='__main__': main()
