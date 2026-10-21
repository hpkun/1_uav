"""python -m coop3d {train,evaluate,replay}."""


def main():
    import argparse
    from pathlib import Path
    from .config import EnvConfig, SCALES
    from .mappo.config import TrainConfig
    parser = argparse.ArgumentParser(description='Coop3D spatial inspection benchmark')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('train')
    p.add_argument('--scale', choices=SCALES)
    p.add_argument('--env-config')
    p.add_argument('--train-config')
    p.add_argument('--output', required=True)
    p.add_argument('--updates', type=int, default=100)
    p.add_argument('--resume')
    p.add_argument('--seed', type=int)
    p = sub.add_parser('evaluate')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seeds', nargs='+', type=int, default=[20000000, 20000001, 20000002])
    p = sub.add_parser('replay')
    p.add_argument('--trace', required=True)
    p.add_argument('--output', required=True)
    a = parser.parse_args()
    if a.command == 'replay':
        from .replay import render_replay
        print(render_replay(a.trace, a.output))
        return
    from .mappo.runner import train, write_json
    if a.command == 'train':
        import yaml
        from .mappo.checkpoint import read_checkpoint
        saved = read_checkpoint(a.resume) if a.resume else None
        env = EnvConfig.load(a.env_config) if a.env_config else EnvConfig(**saved['env_config']) if saved else EnvConfig(scale=a.scale or '3U3P')
        tc = TrainConfig(**yaml.safe_load(Path(a.train_config).read_text(encoding='utf-8'))) if a.train_config else TrainConfig(**saved['train_config']) if saved else TrainConfig()
        if a.scale is not None and env.scale != a.scale:
            raise ValueError('--scale conflicts with saved/environment configuration')
        if a.seed is not None:
            tc.seed = a.seed
            tc.__post_init__()
        train(env, tc, a.output, a.updates, a.resume)
    else:
        from .mappo.checkpoint import read_checkpoint, restore_checkpoint
        from .mappo.trainer import MAPPOTrainer, require_cuda
        from .mappo.evaluation import evaluate
        from .replay import render_replay
        require_cuda()
        payload = read_checkpoint(a.checkpoint)
        env, tc = EnvConfig(**payload['env_config']), TrainConfig(**payload['train_config'])
        # Require seeds outside the training sequence already consumed by workers.
        vector = payload.get('vector')
        if vector is not None:
            for seed in a.seeds:
                offset = seed - tc.seed
                if offset >= 0 and offset // tc.num_envs <= vector['episodes'][offset % tc.num_envs]:
                    raise ValueError('evaluation seed was used for training')
        trainer = MAPPOTrainer(tc)
        restore_checkpoint(payload, trainer, env)
        output = Path(a.output)
        if output.exists() and any(output.iterdir()):
            raise FileExistsError('evaluation output must be empty')
        output.mkdir(parents=True, exist_ok=True)
        report = evaluate(trainer, env, a.seeds, output / 'trajectory.json')
        write_json(output / 'evaluation.json', report)
        render_replay(output / 'trajectory.json', output / 'replay.html')
        print(report)


if __name__ == '__main__':
    main()
