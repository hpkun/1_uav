from pathlib import Path

import numpy as np
import pytest

from scripts.diagnose_500k_policy import (
    closing_speed,
    observation_learnability,
    score_components,
)
from uav_combat.models import AircraftState


ROOT = Path(__file__).resolve().parents[1]


def state(x=0.0, v=200.0, psi=0.0):
    return AircraftState(x, 0.0, -3000.0, v, 0.0, psi)


def test_closing_speed_sign_convention():
    own = state(x=0.0, v=200.0)
    approaching = state(x=1000.0, v=100.0)
    separating = state(x=1000.0, v=250.0)
    assert closing_speed(own, approaching) == pytest.approx(100.0)
    assert closing_speed(own, separating) == pytest.approx(-50.0)


def test_score_components_match_canonical_geometry_definitions():
    tail = score_components(state(), state(x=1000.0), 8000.0)
    head_on = score_components(state(), state(x=1000.0, psi=np.pi), 8000.0)
    assert tail["range_score"] == pytest.approx(0.875)
    assert tail["attack_score"] == pytest.approx(1.0)
    assert tail["escape_score"] == pytest.approx(1.0)
    assert tail["product"] == pytest.approx(0.875)
    assert head_on["attack_score"] == pytest.approx(1.0)
    assert head_on["escape_score"] == pytest.approx(0.0)
    assert head_on["product"] == pytest.approx(0.0)


def test_observation_diagnosis_distinguishes_missing_from_derived_features():
    result = observation_learnability()
    assert result["information_missing"] is False
    assert result["information_present_but_not_explicitly_encoded"] is True
    for quantity in ("distance", "closing_rate", "attack_angle", "escape_angle"):
        assert quantity in result


def test_diagnostic_script_is_not_imported_by_active_runtime():
    active_files = [
        *ROOT.joinpath("src").rglob("*.py"),
        ROOT / "scripts/train_madsac.py",
        ROOT / "scripts/evaluate_madsac.py",
    ]
    for path in active_files:
        assert "diagnose_500k_policy" not in path.read_text(encoding="utf-8")
