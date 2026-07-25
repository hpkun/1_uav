import pytest

from scripts.aggregate_mappo_multiseed import summarize


def test_multiseed_statistics_match_hand_calculation():
    result = summarize([1.0, 2.0, 3.0])
    assert result["mean"] == 2.0
    assert result["sample_std"] == 1.0
    assert result["median"] == 2.0
    assert result["min"] == 1.0 and result["max"] == 3.0
    assert result["ci95_low"] == pytest.approx(2.0 - 4.302652729911275/(3**.5))
    assert result["ci95_high"] == pytest.approx(2.0 + 4.302652729911275/(3**.5))
    assert result["confidence_interval_method"] == "student_t"
    assert result["num_training_seeds"] == 3
