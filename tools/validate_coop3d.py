"""Run the required CUDA acceptance workflow for each scale; writes evidence."""


def main():
    import argparse
    import hashlib
    import json
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    from coop3d import EnvConfig
    from coop3d.mappo.config import TrainConfig
    from coop3d.mappo.runner import train, write_json
    from coop3d.mappo.trainer import MAPPOTrainer, require_cuda
    from coop3d.mappo.checkpoint import read_checkpoint, restore_checkpoint
    from coop3d.mappo.evaluation import evaluate
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='outputs/coop3d_acceptance')
    args = parser.parse_args()
    require_cuda()
    root = Path(args.output)
    results = {}
    for index, scale in enumerate(('3U3P', '5U5P', '3U5P')):
        ec = EnvConfig(scale=scale)
        tc = TrainConfig(seed=1000 + index * 100000, evaluation_seeds=(10000000 + index * 10, 10000001 + index * 10))
        directory = root / scale
        print(f'START {scale}: 10 workers x 250 steps, 10 PPO epochs, CUDA', flush=True)
        trainer, report = train(ec, tc, directory, updates=1)
        payload = read_checkpoint(directory / 'latest.pt')
        clone = MAPPOTrainer(tc)
        restore_checkpoint(payload, clone, ec)
        for a, b in zip(trainer.actor.parameters(), clone.actor.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        holdout = evaluate(clone, ec, [20000000 + index], directory / 'holdout_trajectory.json')
        from coop3d.replay import render_replay
        render_replay(directory / 'holdout_trajectory.json', directory / 'holdout_replay.html')
        metrics = json.loads((directory / 'metrics.jsonl').read_text().splitlines()[-1])
        assert metrics['metrics']['optimizer_steps'] > 0
        assert len(set(metrics['worker_pids'])) == 10
        results[scale] = dict(update=metrics, checkpoint_reload_exact=True, holdout=holdout)
        write_json(root / 'acceptance.json', results)
        print(f'PASS {scale}', flush=True)
    inventory = json.loads(Path('docs/migration_inventory.json').read_text(encoding='utf-8'))
    preserved = []
    for entry in inventory['protected_files']:
        p = Path(entry['path'])
        if hashlib.sha256(p.read_bytes()).hexdigest() != entry['sha256']:
            raise AssertionError(f'protected file changed: {p}')
        preserved.append(entry['path'])
    write_json(root / 'protected_files_verified.json', preserved)
    print(f'PASS: all scales, real PPO, reload, evaluation, replay; {len(preserved)} protected files unchanged', flush=True)


if __name__ == '__main__':
    main()
