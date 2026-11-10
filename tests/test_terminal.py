import json
from types import SimpleNamespace

import pytest

from algorithm.common import terminal
from algorithm.common.research import summarize
from algorithm.interception import main, show_storage


@pytest.mark.parametrize(("value", "expected"), [
    (1.23456, "1.23 m"), (0., "0 m"), (None, "N/A"), (float("nan"), "N/A"),
    (float("inf"), "N/A"), (1e308, "1e+308 m"), (-1e-307, "-1e-307 m"),
])
def test_number(value, expected):
    assert terminal.number(value, "m") == expected


def test_rates_and_empty_window():
    assert terminal.rate(1, 3) == "33.3%"
    assert terminal.rate(0, 4) == "0.0%"
    assert terminal.rate(0, 0) == terminal.percent(None) == terminal.percent(1e308) == "N/A"
    window = terminal.interception_results(summarize([]), window=True)
    assert window == "win=0/100 | capture=N/A | protect=N/A | breach=N/A | reward=N/A"
    rows = [dict(outcome=outcome, time=2., reward_parts={}, minimum_separation=1.,
                 episode_return=3., control_effort=0.) for outcome in ("success", "timeout")]
    result = terminal.interception_results(summarize(rows))
    assert "capture=50.0%" in result and "protect=100.0%" in result
    assert "capture_time=2 s" in result


def test_training_line_preserves_small_kl_and_plain_large_integers(capsys):
    row = dict(update=32, planned_updates=3907, joint_steps=16384,
               completed_episodes_total=59, critic_loss=.43249,
               approx_kl=1.2e-12, sampling_speed=1020.)
    terminal.training_progress("Interception", row, total_steps=2000384,
                               window=terminal.interception_results(summarize([]), window=True))
    assert capsys.readouterr().out == (
        "[train] upd=32/3907 (0.8%) | steps=16384/2000384 | ep=59 | win=0/100"
        " | capture=N/A | protect=N/A | breach=N/A | reward=N/A"
        " | critic=0.432 | kl=1.2e-12 | speed=1020 steps/s\n")
    assert terminal.decimal(1e9) == "1000000000"
    assert terminal.decimal(None) == terminal.decimal(float("inf")) == "N/A"
    assert terminal.sampling_speed(None) == "N/A"


def test_short_prefixes_and_summary_fields(capsys, tmp_path):
    for phase in ("START", "TRAIN", "EVAL", "RESUME", "DONE", "PAUSE",
                  "BATCH", "STORAGE", "VALIDATE", "RUN"):
        terminal.emit("Interception", phase, terminal.fields(value=1))
    terminal.training_end("Interception",
                          SimpleNamespace(updates=3907, steps=2000384, episodes=6500),
                          tmp_path / "latest.pt")
    show_storage(dict(files=3920, bytes=120500000, unique_bytes=120500000))
    lines = capsys.readouterr().out.splitlines()
    assert lines[:10] == [f"[{phase.lower()}] value=1" for phase in
                          ("START", "TRAIN", "EVAL", "RESUME", "DONE", "PAUSE",
                           "BATCH", "STORAGE", "VALIDATE", "RUN")]
    assert lines[10] == "[done] upd=3907 | steps=2000384 | ep=6500 | checkpoint=latest.pt"
    assert lines[11] == "[storage] files=3920 | size=120.5 MB | unique=120.5 MB"


def test_cli_failure_is_compact_and_goes_to_stderr(capsys):
    def fail():
        raise RuntimeError("Traceback (most recent call last):\n  internal stack\nValueError: bad state")

    with pytest.raises(SystemExit) as error:
        terminal.run_cli("Interception", fail)
    assert error.value.code == 1
    captured = capsys.readouterr()
    assert not captured.out and captured.err == "[error] message=ValueError: bad state\n"


def test_validate_and_run_offer_legacy_json(tmp_path, capsys):
    main(["validate", "--task", "S0"])
    output = capsys.readouterr().out
    assert "[validate] valid=yes" in output and "actor_obs=(5, 52)" in output
    main(["validate", "--task", "S0", "--json"])
    validation = json.loads(capsys.readouterr().out)
    assert validation["observation_shape"] == [5, 52] and validation["centralized_shape"] == [54]
    assert validation["configuration"]["task"] == "S0" and "pursuers" in validation["initial"]
    human, machine = (tmp_path/name for name in ("human_run", "json_run"))
    main(["run", "--task", "S0", "--out", str(human)])
    assert "[run]" in capsys.readouterr().out
    main(["run", "--task", "S0", "--out", str(machine), "--json"])
    summary = json.loads(capsys.readouterr().out)
    assert summary["episodes"] == 1 and "all_captured_rate" in summary
    for name in ("summary.json", "trajectory_000.json"):
        assert (human/name).read_bytes() == (machine/name).read_bytes()
