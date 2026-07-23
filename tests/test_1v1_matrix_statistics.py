from types import SimpleNamespace

import scripts.evaluate_1v1_matrix as matrix


def test_matrix_fields_probabilities_and_reproducibility(monkeypatch) -> None:
    summary=SimpleNamespace(outcome="draw",timeout=True,red_ground_crash=False,blue_ground_crash=False,collision=False,decision_steps=4,red_damage=0.0,blue_damage=0.0,red_hits=0,blue_hits=0,red_attack_area_steps=0,blue_attack_area_steps=0,cumulative_reward=0.0)
    monkeypatch.setattr(matrix,"run_episode",lambda *args,**kwargs:(None,summary))
    first=matrix.evaluate_matrix(2,10); second=matrix.evaluate_matrix(2,10)
    assert first==second and len(first)==27
    for row in first:
        assert row["red_win_rate"]+row["blue_win_rate"]+row["draw_rate"]==1.0
        assert set(matrix.FIELDS)==set(row)
