from pathlib import Path
import inspect

import numpy as np
import pytest
import torch
import yaml

from uav_combat.diagnostics.action_stability import (
    bank_compensated_actions,
    fresh_actor,
    trim_a1,
    trim_normal_load,
    trim_relative_control,
    vertical_balance,
)
from uav_combat.environment.control import action_to_control
from uav_combat.models import AircraftState


ROOT = Path(__file__).resolve().parents[1]


def level_state() -> AircraftState:
    return AircraftState(0.0, 0.0, -3000.0, 225.0, 0.0, 0.0)


def test_level_unbanked_trim_balance_is_zero():
    assert vertical_balance(0.0, 0.0, 0.0) == pytest.approx(0.0)


def test_level_sixty_degree_bank_is_neutral_under_v12_mapping():
    assert vertical_balance(0.0, 0.0, 1.0) == pytest.approx(0.0)


def test_sixty_degree_bank_trim_load_and_action_are_two_and_quarter():
    assert trim_normal_load(0.0, np.pi / 3.0) == pytest.approx(2.0)
    assert trim_a1(0.0, np.pi / 3.0) == pytest.approx(0.25)


def test_bank_compensated_probe_retains_historical_v11_action_coordinate():
    states = [level_state()] + [AircraftState(0, 0, -3000, 225, 0, 0, False) for _ in range(3)]
    actions = np.zeros((4, 3), dtype=np.float32)
    actions[0, 2] = 1.0
    compensated = bank_compensated_actions(states, actions)
    assert compensated[0, 1] == pytest.approx(0.25)
    historical_nz = 1.0 + 4.0 * compensated[0, 1]
    assert historical_nz * np.cos(np.pi / 3.0) - 1.0 == pytest.approx(0.0, abs=1e-7)


def test_fresh_actor_does_not_load_checkpoint_or_step_optimizer(monkeypatch):
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("checkpoint load forbidden"))
    monkeypatch.setattr(torch.optim.Optimizer, "step", lambda *args, **kwargs: pytest.fail("optimizer step forbidden"))
    actor = fresh_actor(101)
    output = actor.deterministic(torch.zeros(4, 52))
    assert output.shape == (4, 3)


def test_diagnostic_probe_parameters_do_not_enter_active_config():
    config = yaml.safe_load((ROOT / "configs/combat_environment.yaml").read_text(encoding="utf-8"))
    text = repr(config)
    assert "bank_compensated" not in text
    assert "trim_relative" not in text


def test_canonical_action_mapping_is_v12_trim_relative():
    config = yaml.safe_load((ROOT / "configs/combat_environment.yaml").read_text(encoding="utf-8"))
    own = level_state()
    control = action_to_control(own, np.array([0.2, -0.3, 0.4]), config["action"])
    assert control.nx == pytest.approx(0.4)
    assert control.nz == pytest.approx(1.0 / np.cos(np.pi / 3.0 * 0.4) - 0.6)
    assert control.phi == pytest.approx(np.pi / 3.0 * 0.4)


def test_trim_relative_probe_is_diagnostic_only_and_neutral_at_zero_a1():
    control = trim_relative_control(level_state(), np.array([0.0, 0.0, 1.0]))
    assert control.nz == pytest.approx(2.0)
    assert control.nz * np.cos(control.phi) - 1.0 == pytest.approx(0.0)


def test_diagnostic_script_contains_no_training_update_calls():
    source = (ROOT / "scripts/diagnose_action_stability.py").read_text(encoding="utf-8")
    for forbidden in (".update_actor(", ".update_critics(", ".update_targets(", ".optimizer.step("):
        assert forbidden not in source
    assert "torch.load(" not in source
