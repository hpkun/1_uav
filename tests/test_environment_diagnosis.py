from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.diagnose_environment import initial_geometry_diagnosis, weapon_diagnosis
from uav_combat.environment.env import PaperUAVCombatEnv


ROOT = Path(__file__).resolve().parents[1]


def config():
    return yaml.safe_load((ROOT / "configs" / "paper_environment.yaml").read_text(encoding="utf-8"))


def test_diagnostic_observer_does_not_change_rng_or_state_transition():
    cfg = config()
    events = []
    baseline = PaperUAVCombatEnv(cfg)
    observed = PaperUAVCombatEnv(
        cfg, diagnostic_observer=lambda event, payload: events.append((event, payload))
    )
    baseline_observation, baseline_info = baseline.reset(12345)
    observed_observation, observed_info = observed.reset(12345)
    np.testing.assert_array_equal(baseline_observation, observed_observation)
    np.testing.assert_array_equal(baseline_info["red_alive_mask"], observed_info["red_alive_mask"])
    actions = np.zeros((4, 3), dtype=np.float32)
    for _ in range(120):
        base = baseline.step(actions)
        diagnostic = observed.step(actions)
        np.testing.assert_array_equal(base[0], diagnostic[0])
        np.testing.assert_array_equal(base[1], diagnostic[1])
        assert base[2:4] == diagnostic[2:4]
        for key in (
            "red_success", "termination_reason", "red_attack_kills", "blue_attack_kills",
            "red_losses", "red_survivors", "blue_survivors", "episode_length",
        ):
            assert base[4][key] == diagnostic[4][key]
        if base[2] or base[3]:
            break
    assert any(event == "weapon_attempt" for event, _ in events)
    assert any(event == "hit_resolution" for event, _ in events)


def test_weapon_diagnosis_confirms_canonical_four_kilometer_lethality():
    result = weapon_diagnosis(config(), samples=10_000, seed=7)
    critical = [
        row for row in result["rows"]
        if row["distance_m"] == 4000.0 and row["ata_degrees"] == 0.0
    ][0]
    assert critical["threshold_degrees"] == np.rad2deg(np.pi * np.exp(-2.0))
    assert critical["monte_carlo_hit_probability"] > 0.999
    assert result["weapon_model_sample_hit_probability_at_4000m_0deg"] > 0.999
    repeated = result["repeated_attempt_probability_at_4000m"]["ata_ha_30deg"]
    assert repeated["10"] > repeated["5"] > repeated["2"] > repeated["1"]


def test_initial_geometry_diagnosis_measures_eight_kilometer_centers():
    result = initial_geometry_diagnosis(config(), reset_count=10, seed_base=99)
    centers = result["statistics"]["center_distance_m"]
    assert centers["mean"] == 8000.0
    assert centers["std"] < 1e-9
    assert result["head_on_theory"]["time_to_4000m_steps"] == pytest.approx(88.8888888889)
