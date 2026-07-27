import csv
import json
from pathlib import Path

from scripts.diagnose_3v3_reward_ordering import REWARD_COMPONENTS, paired_differences, summarize, write_outputs


def row(policy: str, seed: int, *, ret: float, attacks: float, hits: float, damage: float, red_survivors: int = 1, timeout: int = 1) -> dict:
    base = {
        "policy_label": policy,
        "checkpoint_path": "",
        "seed": seed,
        "winner": "draw",
        "termination_reason": "timeout" if timeout else "red_eliminated",
        "decision_steps": 400 if timeout else 20,
        "simulation_time": 200.0 if timeout else 10.0,
        "red_survivors": red_survivors,
        "blue_survivors": 3,
        "survivor_difference": red_survivors - 3,
        "team_episode_return": ret,
        "agent_sum_episode_return": ret * 3,
        "mean_per_agent_episode_return": ret,
        "red_attack_attempts": attacks,
        "blue_attack_attempts": 0.0,
        "red_hits": hits,
        "blue_hits": 0.0,
        "red_nominal_damage": damage,
        "blue_nominal_damage": 0.0,
        "red_effective_damage": damage,
        "blue_effective_damage": 0.0,
        "red_overkill_damage": 0.0,
        "blue_overkill_damage": 0.0,
        "red_attack_area_steps": attacks + 1.0,
        "blue_attack_area_steps": 0.0,
        "red_ground_crashes": 0.0,
        "blue_ground_crashes": 0.0,
        "red_collisions": 0.0,
        "blue_collisions": 0.0,
        "timeout": timeout,
        "red_elimination_win": 0,
        "red_timeout_survival_win": 0,
        "draw": 1,
    }
    for component in REWARD_COMPONENTS:
        base[f"{component}_team_total"] = 3.0
        base[f"{component}_per_agent"] = 1.0
    return base


def test_reward_ordering_outputs_paired_bootstrap_and_complete_columns(tmp_path: Path):
    rows = []
    for seed in (10, 11, 12):
        rows.extend(
            [
                row("pursuit", seed, ret=5.0, attacks=2.0, hits=1.0, damage=20.0, red_survivors=2),
                row("straight", seed, ret=1.0, attacks=0.0, hits=0.0, damage=0.0, red_survivors=1),
                row("random", seed, ret=0.0, attacks=0.0, hits=0.0, damage=0.0, red_survivors=1),
                row("learned_actor", seed, ret=1.0, attacks=0.0, hits=0.0, damage=0.0, red_survivors=1),
            ]
        )
    summary = summarize(rows)
    first = paired_differences(rows)
    second = paired_differences(rows)
    assert sorted({item["seed"] for item in rows if item["policy_label"] == "pursuit"}) == [10, 11, 12]
    assert first == second
    assert first["pursuit - straight"]["team_episode_return"]["mean_paired_difference"] == 4.0
    assert summary["diagnosis"]["label"] == "exploration_or_optimization_failure"
    write_outputs(rows, summary, tmp_path)
    csv_path = tmp_path / "reward_ordering_episodes.csv"
    with csv_path.open(newline="", encoding="utf-8") as stream:
        header = next(csv.reader(stream))
    for component in REWARD_COMPONENTS:
        assert f"{component}_team_total" in header
        assert f"{component}_per_agent" in header
    payload = json.loads((tmp_path / "reward_ordering_summary.json").read_text(encoding="utf-8"))
    assert payload["policies"]["pursuit"]["episode_count"] == 3


def test_reward_ordering_without_checkpoint_does_not_claim_exploration_failure():
    rows = []
    for seed in (1, 2):
        rows.extend(
            [
                row("pursuit", seed, ret=5.0, attacks=2.0, hits=1.0, damage=20.0),
                row("straight", seed, ret=1.0, attacks=0.0, hits=0.0, damage=0.0),
                row("random", seed, ret=0.0, attacks=0.0, hits=0.0, damage=0.0),
            ]
        )
    assert summarize(rows)["diagnosis"]["label"] == "insufficient_evidence"
