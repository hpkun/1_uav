import json

import pytest

from algorithm.common import terminal
from algorithm.common.research import summarize
from algorithm.interception import main


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
    assert window == "Train window n=0/100, no completed episodes yet"
    rows = [dict(outcome=outcome, time=2., reward_parts={}, minimum_separation=1.,
                 episode_return=3., control_effort=0.) for outcome in ("success", "timeout")]
    result = terminal.interception_results(summarize(rows))
    assert "All-target capture 50.0%" in result and "Protection success 100.0%" in result
    assert "Mean all-target capture time 2 s" in result


def test_cli_failure_is_compact_and_goes_to_stderr(capsys):
    def fail():
        raise RuntimeError("Traceback (most recent call last):\n  internal stack\nValueError: bad state")

    with pytest.raises(SystemExit) as error:
        terminal.run_cli("Interception", fail)
    assert error.value.code == 1
    captured = capsys.readouterr()
    assert not captured.out and captured.err == "[Interception ERROR] ValueError: bad state\n"


def test_validate_and_run_offer_legacy_json(tmp_path, capsys):
    main(["validate", "--task", "S0"])
    output = capsys.readouterr().out
    assert "[Interception VALIDATE] Configuration valid" in output and "Actor observations (5, 52)" in output
    main(["validate", "--task", "S0", "--json"])
    validation = json.loads(capsys.readouterr().out)
    assert validation["observation_shape"] == [5, 52] and validation["centralized_shape"] == [54]
    assert validation["configuration"]["task"] == "S0" and "pursuers" in validation["initial"]
    human, machine = (tmp_path/name for name in ("human_run", "json_run"))
    main(["run", "--task", "S0", "--out", str(human)])
    assert "[Interception RUN]" in capsys.readouterr().out
    main(["run", "--task", "S0", "--out", str(machine), "--json"])
    summary = json.loads(capsys.readouterr().out)
    assert summary["episodes"] == 1 and "all_captured_rate" in summary
    for name in ("summary.json", "trajectory_000.json"):
        assert (human/name).read_bytes() == (machine/name).read_bytes()
