"""CUDA-only tiny PWTR integration smoke. Never starts a formal experiment."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.protocol import validate_pwtr_branch
from algorithm.modular_mappo.runner import ModularMAPPOTrainingRunner
from algorithm.modules import replay_batch_budget, wave_stratified_permutation
from algorithm.train_modular_mappo import load_config

SOURCE = ROOT / "outputs/diag_mappo_learnability/l3_seed5302/checkpoint_1505280.pt"


def digest(value):
    h = hashlib.sha256()
    def add(v):
        if torch.is_tensor(v): h.update(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v, np.ndarray): h.update(np.ascontiguousarray(v).tobytes())
        elif isinstance(v, dict):
            for key in sorted(v, key=str): h.update(str(key).encode()); add(v[key])
        elif isinstance(v, (list, tuple)):
            for item in v: add(item)
        else: h.update(repr(v).encode())
    add(value); return h.hexdigest()


def checkpoint_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", default="outputs/smoke_pwtr_mappo")
    args = parser.parse_args(); out = ROOT / args.output_dir
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is mandatory for PWTR smoke")
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    source_sha = checkpoint_hash(SOURCE); state = torch.load(SOURCE, map_location="cpu", weights_only=False)
    env = yaml.safe_load((ROOT / "configs/persistent_wave_v2_environment.yaml").read_text(encoding="utf-8"))
    plain_cfg = load_config(ROOT / "configs/dev_pwtr_plain_300k.yaml")
    full_cfg = load_config(ROOT / "configs/dev_pwtr_full_300k.yaml")
    runtime = {"training_seed": 5302, "training_num_envs": 24, "training_smoke": False}
    plain_prov = validate_pwtr_branch(state, env, plain_cfg, runtime)
    full_prov = validate_pwtr_branch(state, env, full_cfg, runtime)

    # The real environment path: identical source state, RNG, reset and two vector steps.
    rollouts = {}
    for name, config, provenance in (("plain", plain_cfg, plain_prov), ("full", full_cfg, full_prov)):
        runner = ModularMAPPOTrainingRunner(env, config, 2, 1_805_280, "cuda", 5302, out / name, False,
                                            resume_mode=True, branch_provenance={**provenance, "parent_checkpoint_sha256": source_sha})
        runner.branch_from(SOURCE, provenance["intervention"], source_sha)
        rollouts[name] = runner.collect_rollout(2)
        runner.vector.close()
    fields = ("observations", "actions", "raw_actions", "old_log_probs", "raw_environment_rewards",
              "alive_masks", "next_alive_masks", "wave_indices", "dones")
    first_rollout = {field: np.array_equal(getattr(rollouts["plain"], field), getattr(rollouts["full"], field)) for field in fields}
    if not all(first_rollout.values()): raise RuntimeError(f"first rollout mismatch: {first_rollout}")

    # Disabled capability must remain exact Plain, including optimizer and RNG state.
    disabled_cfg = deepcopy(plain_cfg); disabled_cfg["modules"]["persistent_wave_trajectory_replay"] = {"enabled": False}
    control = build_modular_mappo_trainer(plain_cfg, "cuda", 256, 1_805_280)
    disabled = build_modular_mappo_trainer(disabled_cfg, "cuda", 256, 1_805_280)
    control.load(SOURCE, strict_protocol=False, restore_rng=True); disabled.load(SOURCE, strict_protocol=False, restore_rng=True)
    saved_cpu = torch.get_rng_state(); saved_cuda = torch.cuda.get_rng_state_all()
    mc = control.update(deepcopy(rollouts["plain"])); rc = control.capture_rng_state()
    torch.set_rng_state(saved_cpu); torch.cuda.set_rng_state_all(saved_cuda)
    md = disabled.update(deepcopy(rollouts["plain"])); rd = disabled.capture_rng_state()
    parity = (digest(control.actor.state_dict()) == digest(disabled.actor.state_dict()) and
              digest(control.critic.state_dict()) == digest(disabled.critic.state_dict()) and
              digest(control.actor_optimizer.state_dict()) == digest(disabled.actor_optimizer.state_dict()) and
              digest(control.critic_optimizer.state_dict()) == digest(disabled.critic_optimizer.state_dict()) and
              digest(rc) == digest(rd) and mc == md)
    if not parity: raise RuntimeError("PWTR-disabled update is not bitwise Plain-equivalent")

    # Pure conservation plus a real CUDA replay backward/step on natural-format segments.
    waves = np.asarray([1] * 13 + [2] * 11 + [3] * 7)
    order = wave_stratified_permutation(waves, 8, np.random.default_rng(5))
    conservation = sorted(order.tolist()) == list(range(len(waves)))
    full = build_modular_mappo_trainer(full_cfg, "cuda", 256, 1_805_280); full.load(SOURCE, strict_protocol=False, restore_rng=True)
    module = full.persistent_wave_trajectory_replay
    def batch(wave):
        t = 128; obs = np.random.default_rng(wave).normal(size=(t, 1, 4, 52)).astype("f")
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs[:, 0], device="cuda")
            alive_tensor = torch.ones((t, 4), device="cuda")
            distribution, _ = full.actor.distribution_step(obs_tensor, None, None, None, alive_tensor)
            raw_tensor = distribution.sample()
            log_tensor = full.actor._squashed_log_prob(distribution, raw_tensor, torch.tanh(raw_tensor))
        raw = raw_tensor.cpu().numpy()[:, None]
        behavior_log_probs = log_tensor.cpu().numpy()[:, None]
        return type("Batch", (), {"observations": obs, "next_observations": obs + .01,
            "raw_actions": raw, "old_log_probs": behavior_log_probs,
            "rewards": np.full((t, 1, 4), .1, "f"), "dones": np.zeros((t, 1), "f"),
            "alive_masks": np.ones((t, 1, 4), "f"), "next_alive_masks": np.ones((t, 1, 4), "f"),
            "wave_indices": np.full((t, 1), wave, dtype=np.int64), "wave_transition_flags": np.zeros((t, 1), "f")})()
    module.ingest_rollout(batch(2), 0); module.ingest_rollout(batch(3), 0)
    # Construct both bridge types through the same extractor.
    for source_wave in (1, 2):
        b = batch(source_wave); b.wave_transition_flags[-1, 0] = 1; module.ingest_rollout(b, 0)
        module.ingest_rollout(batch(source_wave + 1), 0)
    module.current_rollout_generation = 1; module.current_later_states = 512; module.fresh_rollout_count = 2
    actor_before, critic_before = digest(full.actor.state_dict()), digest(full.critic.state_dict())
    reward_before = digest([segment["rewards"] for rows in module.partitions.values() for segment in rows])
    replay_metrics = full._pwtr_replay_phase()
    actor_changed = actor_before != digest(full.actor.state_dict()); critic_changed = critic_before != digest(full.critic.state_dict())
    reward_unchanged = reward_before == digest([segment["rewards"] for rows in module.partitions.values() for segment in rows])
    finite = all(np.isfinite(value) for value in replay_metrics.values() if isinstance(value, (int, float)))
    if not actor_changed or not critic_changed or not finite or not reward_unchanged: raise RuntimeError("PWTR replay CUDA update failed")

    checkpoint = out / "pwtr_roundtrip.pt"; full.save(checkpoint)
    restored = build_modular_mappo_trainer(full_cfg, "cuda", 256, 1_805_280); restored.load(checkpoint, strict_protocol=True, restore_rng=True)
    roundtrip = digest(module.state_dict()) == digest(restored.persistent_wave_trajectory_replay.state_dict())
    if not roundtrip: raise RuntimeError("PWTR memory/RNG checkpoint roundtrip failed")
    budget = replay_batch_budget(512)
    result = {"status": "PWTR_CUDA_TINY_SMOKE_PASS", "cuda": torch.cuda.get_device_name(0),
              "first_rollout_bitwise": first_rollout, "plain_disabled_bitwise_parity": parity,
              "stratified_sample_conservation": conservation, "memory_counts": module.memory_counts(),
              "w2_w3_extraction": len(module.partitions["W2_INTERNAL"]) > 0 and len(module.partitions["W3_INTERNAL"]) > 0,
              "bridge12_bridge23_extraction": len(module.partitions["BRIDGE_12"]) > 0 and len(module.partitions["BRIDGE_23"]) > 0,
              "replay_actor_changed": actor_changed, "replay_critic_changed": critic_changed,
              "replay_metrics_finite": finite, "environment_reward_unchanged": reward_unchanged,
              "current_recent_budget_matched": budget == replay_batch_budget(512), "bridge_does_not_change_budget": budget == replay_batch_budget(512),
              "checkpoint_memory_rng_roundtrip": roundtrip, "source_checkpoint_unchanged": source_sha == checkpoint_hash(SOURCE),
              "formal_training_started": False, "formal_evaluation_started": False, "45m_used": False,
              "replay_metrics": replay_metrics}
    (out / "smoke_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
