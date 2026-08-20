from pathlib import Path

import numpy as np
import pytest

from uav_combat.diagnostics.paper_controller_prototype import (
    FeasibleProjectedPController, ModelFeedbackPController,
    command_from_normalized, wrap_angle,
)
from uav_combat.dynamics import PointMassDynamics
from uav_combat.models import AircraftState


ROOT = Path(__file__).resolve().parents[1]


def state(psi=0.0, theta=0.0, speed=225.0):
    return AircraftState(0.0, 0.0, -10_000.0, speed, theta, psi)


def test_zero_error_produces_equilibrium_like_control():
    own = state(theta=0.1)
    desired = command_from_normalized(own, np.zeros(3))
    result = ModelFeedbackPController(2, 2, 2).control(own, desired)
    derivative = PointMassDynamics().derivatives(own, result.command)
    assert derivative[3] == pytest.approx(0.0, abs=1e-12)
    assert derivative[4] == pytest.approx(0.0, abs=1e-12)
    assert derivative[5] == pytest.approx(0.0, abs=1e-12)


def test_heading_error_controls_bank_with_correct_sign():
    own = state()
    controller = ModelFeedbackPController(2, 2, 2)
    positive = controller.control(own, command_from_normalized(own, [0.25, 0, 0]))
    negative = controller.control(own, command_from_normalized(own, [-0.25, 0, 0]))
    assert positive.command.phi > 0.0
    assert negative.command.phi < 0.0


def test_speed_and_pitch_tracking_signs_are_correct():
    own = state()
    controller = ModelFeedbackPController(4, 4, 4)
    faster = controller.control(own, command_from_normalized(own, [0, 0, 0.5]))
    slower = controller.control(own, command_from_normalized(own, [0, 0, -0.5]))
    assert faster.command.nx > 0.0
    assert slower.command.nx < 0.0
    up = controller.control(own, command_from_normalized(own, [0, 0.01, 0]))
    down = controller.control(own, command_from_normalized(own, [0, -0.01, 0]))
    dynamics = PointMassDynamics()
    assert dynamics.derivatives(own, up.command)[4] > 0.0
    assert dynamics.derivatives(own, down.command)[4] < 0.0


def test_outputs_are_finite_and_phi_is_paper_clipped():
    own = state()
    result = ModelFeedbackPController(1, 1, 1).control(
        own, command_from_normalized(own, [1, -1, 1])
    )
    values = [result.raw_nx, result.raw_nz, result.raw_phi,
              result.command.nx, result.command.nz, result.command.phi]
    assert np.all(np.isfinite(values))
    assert abs(result.command.phi) <= np.pi / 2


def test_feasible_projection_avoids_spurious_yaw_for_pitch_down():
    own = state()
    desired = command_from_normalized(own, [0, -1, 0])
    result = FeasibleProjectedPController(2, 2, 2).control(own, desired)
    derivative = PointMassDynamics().derivatives(own, result.command)
    assert result.command.nz == pytest.approx(0.0)
    assert result.command.phi == pytest.approx(0.0)
    assert derivative[4] < 0.0
    assert derivative[5] == pytest.approx(0.0)


def test_wrap_at_pi_is_deterministic():
    assert wrap_angle(np.pi) == pytest.approx(-np.pi)
    assert wrap_angle(-np.pi) == pytest.approx(-np.pi)
    desired = command_from_normalized(state(), [1, 0, 0])
    assert desired.psi == pytest.approx(-np.pi)


def test_controller_prototype_is_not_imported_by_active_runtime():
    active = [*ROOT.joinpath("src/uav_combat/environment").rglob("*.py"),
              *ROOT.joinpath("src/uav_combat/training").rglob("*.py"),
              *ROOT.joinpath("src/uav_combat/madsac").rglob("*.py")]
    for path in active:
        assert "paper_controller_prototype" not in path.read_text(encoding="utf-8")
