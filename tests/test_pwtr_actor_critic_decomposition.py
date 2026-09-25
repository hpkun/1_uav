from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from algorithm.modular_mappo.protocol import validate_pwtr_branch
from algorithm.train_modular_mappo import load_config
from tools.analyze_pwtr_actor_critic_decomposition import direction_label, interaction, main_effects

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/diag_mappo_learnability/l3_seed5301/checkpoint_1505280.pt"
NAMES = ("current_actor_only", "current_critic_only", "recent_actor_only", "recent_critic_only")
EXPECTED = {
    "current_actor_only": (True, True, "current", False, False, True, False),
    "current_critic_only": (True, True, "current", False, False, False, True),
    "recent_actor_only": (True, True, "recent_uniform", False, False, True, False),
    "recent_critic_only": (True, True, "recent_uniform", False, False, False, True),
}


def mode(config):
    module = config["modules"]["persistent_wave_trajectory_replay"]
    return (module["fresh_wave_stratification"], module["replay_enabled"], module["replay_source"],
            module["priority_enabled"], module["bridge_enabled"], module["actor_replay"], module["critic_replay"])


def test_four_configs_are_exact_matched_decomposition_protocols():
    configs = {name: load_config(ROOT / f"configs/dev_pwtr_{name}_300k.yaml") for name in NAMES}
    for name, config in configs.items():
        branch = config["development_branch"]
        assert mode(config) == EXPECTED[name]
        assert (branch["source_sampled_steps"], branch["additional_sampled_steps"],
                branch["target_sampled_steps"], config["training"]["total_sampled_steps"]) == (1_505_280, 300_000, 1_805_280, 1_805_280)
        assert branch["actor_optimizer_restore"] and branch["critic_optimizer_restore"] and branch["rng_restore"]
        enabled = sorted(key for key, value in config["modules"].items() if isinstance(value, dict) and value.get("enabled", False))
        assert enabled == ["actor_lr_decay", "persistent_wave_trajectory_replay"]
        module = config["modules"]["persistent_wave_trajectory_replay"]
        assert tuple(module[key] for key in ("sequence_length", "bridge_half_length", "min_segment_length", "partition_capacity", "actor_max_age_updates")) == (128, 64, 32, 32, 2)


@pytest.mark.skipif(not SOURCE.is_file(), reason="formal source checkpoint unavailable")
def test_validator_accepts_exact_modes_and_rejects_one_wrong_switch():
    state = torch.load(SOURCE, map_location="cpu", weights_only=False)
    env = yaml.safe_load((ROOT / "configs/persistent_wave_v2_environment.yaml").read_text(encoding="utf-8"))
    runtime = {"training_seed": 5301, "training_num_envs": 24, "training_smoke": False}
    for name in NAMES:
        config = load_config(ROOT / f"configs/dev_pwtr_{name}_300k.yaml")
        result = validate_pwtr_branch(state, env, config, runtime)
        assert result["intervention"] == f"pwtr_{name}"
    bad = deepcopy(load_config(ROOT / "configs/dev_pwtr_current_actor_only_300k.yaml"))
    bad["modules"]["persistent_wave_trajectory_replay"]["critic_replay"] = True
    with pytest.raises(RuntimeError, match="ablation mode mismatch"):
        validate_pwtr_branch(state, env, bad, runtime)


def test_factorial_interaction_and_main_effect_formulas():
    assert interaction(1.0, 1.2, .9, .8) == pytest.approx(-.3)
    actor, critic = main_effects(1.0, 1.2, .9, .8)
    assert actor == pytest.approx(.05)
    assert critic == pytest.approx(-.25)
    assert direction_label([.1, .2, -.1]) == "POSITIVE"
    assert direction_label([-.1, -.2, .1]) == "NEGATIVE"
    assert direction_label([.1, -.2, 0.0]) == "MIXED"


def test_launcher_is_serial_exact_and_analyzer_does_not_use_45m():
    launcher = (ROOT / "tools/run_pwtr_actor_critic_decomposition_300k.sh").read_text(encoding="utf-8")
    assert "branches=(current_actor_only current_critic_only recent_actor_only recent_critic_only)" in launcher
    assert "for seed in 5301 5302 5303" in launcher
    assert "nohup" not in launcher and " &\n" not in launcher
    analyzer = (ROOT / "tools/analyze_pwtr_actor_critic_decomposition.py").read_text(encoding="utf-8")
    assert "45_000_000" not in analyzer
    assert "evaluate" not in analyzer
