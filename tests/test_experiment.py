import json
import gzip

import pytest

import algorithm.interception as cli
from algorithm.common import experiment
from algorithm.common.interception_checkpoint import read
from algorithm.common.research import file_digest, read_json
from algorithm.mappo.interception import TrainConfig
from test_interception_research import exact


def config_file(tmp_path, **overrides):
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(dict(version=1, name="short", environment=dict(task="S0", dt=1., max_steps=12),
        training=dict(workers=1, horizon=5, epochs=1, minibatch=8),
        total_joint_steps=11, evaluation=dict(interval_joint_steps=6, evaluation_episodes=2),
        test=dict(episodes=2, checkpoint="best")) | overrides), encoding="utf-8")
    return path


@pytest.mark.parametrize("value", ["1M", "1.0", "+1", "1_000", "0", "-2", "1e6"])
def test_integer_cli_budget(value):
    with pytest.raises(SystemExit) as caught:
        cli.main(["train", "--total-steps", value, "--out", "unused"])
    assert caught.value.code == 2


@pytest.mark.parametrize(("overrides", "error"), [
    (dict(version=True), "version"),
    (dict(extra=1), "unknown experiment"),
    (dict(total_joint_steps=5.5), "positive integer"),
    (dict(evaluation=dict(interval_joint_steps=6, evaluation_interval_updates=1)), "not both"),
    (dict(test=dict(seed=1001)), "overlap"),
    (dict(test=dict(checkpoint="best"), evaluation={}), "scheduled validation"),
    (dict(storage=dict(trajectories="selected", trajectory_seeds=[42])), "outside"),
    (dict(environment=dict(task="S0", scale="3v1")), "disagree"),
])
def test_invalid_config_is_rejected_before_creating_output(tmp_path, overrides, error):
    config = config_file(tmp_path, **overrides)
    out = tmp_path / "run"
    with pytest.raises(ValueError, match=error):
        cli.main(["train", "--config", str(config), "--out", str(out)])
    assert not out.exists()


def test_budget_rounding_and_conflicting_legacy_updates(tmp_path):
    training = TrainConfig()
    assert experiment.budget(1_000_000, training) == dict(
        total_joint_steps=1_000_000, aligned_joint_steps=1_000_448, updates=1954)
    config = config_file(tmp_path)
    with pytest.raises(ValueError, match="conflicts"):
        cli.main(["train", "--config", str(config), "--updates", "1", "--out", str(tmp_path/"unused")])


def test_unified_resume_recovery_legacy_parity_and_test_storage(tmp_path, monkeypatch, capsys):
    config = config_file(tmp_path)
    folder, legacy = tmp_path/"unified", tmp_path/"legacy"
    cli.main(["train", "--config", str(config), "--seed", "18", "--total-steps", "12",
              "--out", str(folder), "--until-steps", "5"])
    run = read_json(folder/"config.json")
    assert run["experiment"]["total_joint_steps"] == 12 and run["experiment"]["aligned_joint_steps"] == 15
    assert run["evaluation"]["evaluation_interval_updates"] == 2 and run["training"]["seed"] == 18
    assert read(folder/"latest.pt")["steps"] == 5 and not list(folder.glob("test_*"))
    original = cli.atomic_json

    def interrupt(path, value, **kwargs):
        if path.name == "update_000002.json":
            raise OSError("interrupted after checkpoint")
        original(path, value, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(cli, "atomic_json", interrupt)
        with pytest.raises(OSError, match="after checkpoint"):
            cli.main(["resume", "--out", str(folder)])
    assert not (folder/"update_000002.json").exists()
    cli.main(["resume", "--out", str(folder)])
    assert (folder/"update_000002.json").exists()
    saved = read(folder/"latest.pt")
    assert saved["steps"] == 15 and saved["updates"] == 3
    cli.main(["train", "--env", str(write_config(tmp_path/"env.json", run["environment"])),
              "--training", str(write_config(tmp_path/"training.json", run["training"])),
              "--evaluation-interval-updates", "2", "--evaluation-episodes", "2",
              "--updates", "3", "--out", str(legacy)])
    old = read(legacy/"latest.pt")
    for key in ("actor", "critic", "actor_optimizer", "critic_optimizer", "steps", "episodes",
                "torch_rng", "cuda_rng", "numpy_rng", "python_rng", "vector"):
        exact(saved[key], old[key])
    unified_update = read_json(folder/"update_000003.json")
    legacy_update = read_json(legacy/"update_000003.json")
    for key in ("completed_episodes", "actor_total_loss", "critic_loss", "approx_kl", "valid_samples"):
        exact(unified_update[key], legacy_update[key])
    for key in ("summary", "runs"):
        exact(read_json(folder/"evaluation_000002.json")[key], read_json(legacy/"evaluation_000002.json")[key])
    result = folder/"test_000003.json"
    test = read_json(result)
    assert test["seeds"] == [10010000, 10010001] and len(test["runs"]) == 2
    assert test["model"]["sha256"] == file_digest(folder/"best.pt")
    assert read_json(folder/"best.json")["best_update"] == read_json(legacy/"best.json")["best_update"]
    assert not list(folder.rglob("trajectory_*"))
    assert sorted(p.name for p in folder.glob("*.pt")) == ["best.pt", "latest.pt"]
    before = result.read_bytes()
    cli.main(["resume", "--out", str(folder)])
    assert result.read_bytes() == before and read(folder/"latest.pt")["updates"] == 3
    with pytest.raises(ValueError, match="within"):
        cli.main(["resume", "--out", str(folder), "--extend-total-steps", "16", "--until-steps", "17"])
    assert not list(folder.glob("budget_*.json"))
    cli.main(["resume", "--out", str(folder), "--extend-total-steps", "16"])
    assert result.read_bytes() == before and (folder/"test_000004.json").exists()
    assert read_json(folder/"budget_000001.json")["target"]["aligned_joint_steps"] == 20
    assert read(folder/"latest.pt")["steps"] == 20
    cli.main(["stats", "--out", str(folder), "--json"])
    usage = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert usage == experiment.storage_usage(folder)
    assert usage["files"] < 25 and usage["bytes"] < 4 * 1024**2
    with pytest.raises(FileExistsError):
        cli.main(["train", "--config", str(config), "--out", str(folder)])

    path = str(folder/"latest.pt")
    output, selected = tmp_path/"manual.json", tmp_path/"selected.json"
    base = ["evaluate", path, "--scenario", "random", "--seed", "10020000", "--episodes", "2"]
    cli.main([*base, "--output", str(output)])
    cli.main([*base, "--output", str(selected), "--trajectories", "selected", "--trajectory-seed", "10020001"])
    assert output.read_bytes() == selected.read_bytes()
    assert not output.with_name("manual_trajectories").exists()
    trace, = selected.with_name("selected_trajectories").glob("*.json")
    assert read_json(trace)["seed"] == 10020001
    with pytest.raises(FileExistsError):
        cli.main([*base, "--output", str(output)])
    with pytest.raises(ValueError, match="overlap"):
        cli.main(["evaluate", path, "--seed", "1000", "--output", str(tmp_path/"overlap.json")])
    with pytest.raises(SystemExit):
        cli.main([*base, "--output", str(output), "--out", str(tmp_path/"other")])


def write_config(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_batch_report_and_matching_budget(tmp_path):
    config = config_file(tmp_path, total_joint_steps=2, evaluation={}, test=dict(enabled=False),
                         report=dict(on_complete=True, plots=False))
    folder = tmp_path/"batch"
    command = ["batch", "--config", str(config), "--seeds", "17", "18", "--out", str(folder)]
    cli.main(command)
    report = read_json(folder/"report/summary.json")
    assert report["aggregate"]["critic_loss"][0]["n"] == 2
    assert report["aggregate"]["critic_loss"][0]["std"] is not None
    assert not list(folder.rglob("*.png"))
    original = (folder/"report/summary.json").read_bytes()
    cli.main([*command, "--resume"])
    assert (folder/"report/summary.json").read_bytes() == original
    cli.main(["resume", "--out", str(folder/"seed_17"), "--extend-total-steps", "6"])
    with pytest.raises(ValueError, match="matching"):
        cli.main(["report", "--batch", str(folder), "--out", str(tmp_path/"mismatch"), "--no-plots"])


def test_automatic_selected_trace_and_final_report(tmp_path):
    config = config_file(tmp_path, total_joint_steps=2, evaluation=dict(evaluation_interval_updates=1),
                         test=dict(episodes=2),
                         storage=dict(trajectories="selected", trajectory_seeds=[10010001]),
                         report=dict(on_complete=True, plots=False))
    folder = tmp_path/"selected"
    cli.main(["train", "--config", str(config), "--out", str(folder)])
    final = folder/"test_000001"
    trace, = final.glob("trajectory_*")
    with gzip.open(trace, "rt", encoding="utf-8") as stream:
        assert json.load(stream)["seed"] == 10010001
    data = read_json(final/"summary.json")
    assert data["model"]["sha256"] == file_digest(folder/"latest.pt")
    assert len(data["runs"]) == 2
    report = read_json(folder/"report_000001/summary.json")
    assert report["aggregate"]["final_test"]["all_captured_rate"]["n"] == 1
    assert not list(folder.rglob("*.png"))
    assert (folder/"latest.pt").samefile(folder/"best.pt")
    usage = experiment.storage_usage(folder)
    assert usage["bytes"] - usage["unique_bytes"] == (folder/"latest.pt").stat().st_size
