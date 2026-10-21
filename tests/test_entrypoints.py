import sys
import pytest


def test_training_entrypoint_config_and_budget(monkeypatch,tmp_path):
    import algorithm.train_mappo as cli
    import algorithm.mappo.runner as runner
    called={}
    def capture(ec,tc,out,budget,**kwargs):
        called.update(env=ec,algorithm=tc,output=out,budget=budget,**kwargs)
    monkeypatch.setattr(runner,'train',capture)
    monkeypatch.setattr(sys,'argv',['train','--env-config','configs/coop3d_3U5P.yaml','--algorithm-config','configs/mappo_coop3d.yaml',
        '--device','cuda','--seed','1234','--num-envs','2','--total-sampled-steps','17','--output-dir',str(tmp_path/'run')])
    cli.main()
    assert called['env'].scale=='3U5P' and called['algorithm'].num_envs==2 and called['algorithm'].seed==1234
    assert called['budget']==17 and called['device']=='cuda'


def test_no_legacy_executable_import_paths():
    import ast
    from pathlib import Path
    assert not Path('coop3d').exists()
    for directory in ('env','algorithm','tools','tests'):
        for p in Path(directory).rglob('*.py'):
            for node in ast.walk(ast.parse(p.read_text(encoding='utf-8'))):
                if isinstance(node,ast.ImportFrom): assert not (node.module or '').startswith('coop3d'),p
                if isinstance(node,ast.Import): assert not any(n.name.startswith('coop3d') for n in node.names),p
