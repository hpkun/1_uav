# Combat Episode Visualization

This is qualitative visualization only. Recorded episodes never enter formal quantitative metrics, aggregation, checkpoint selection, or holdout evidence. The recommended visualization-only seed is `40000000`; monitoring seeds `29000000..29000019` and formal holdout seeds `30000000..30000199` are rejected.

From the repository root on Ubuntu:

```bash
conda activate uav
python -u tools/record_combat_episode.py \
  --checkpoint outputs/<chosen_run>/latest.pt \
  --env-config configs/persistent_wave_v2_environment.yaml \
  --episode-seed 40000000 --device cuda \
  --output-dir outputs/visualization/ea_wb_seed40000000

python -u tools/render_combat_episode.py \
  --input-dir outputs/visualization/ea_wb_seed40000000 \
  --output outputs/visualization/ea_wb_seed40000000/episode.mp4 \
  --fps 30 --frame-stride 4
```

The recorder loads `state["extra"]["algorithm_config"]`, validates the existing modular checkpoint protocol, reconstructs the trainer through the modular factory, and runs deterministic CUDA inference with the same actor/critic hidden-state lifecycle as the evaluator. It reads truth directly from `env.red` and `env.blue`; NED `z` is rendered as `altitude = -z`.

Output is `episode_trace.npz` plus `metadata.json` with `trace_schema_version=1`. Red uses `[F,4,6]` kinematics; Blue uses `[F,total_waves,4,6]`, with inactive wave slots as NaN and false. Blue trajectories are separated by wave and a visualization-only observer captures the old wave's post-combat, pre-respawn terminal states, preventing a spawn “teleport” line. No attacker-target identity is invented because the environment does not expose an unambiguous mapping.

The renderer is headless Matplotlib Agg, fixed-camera 1280x720 at 30 fps and stride 4 by default. Preview PNG generation does not require ffmpeg. MP4 generation explicitly reports `ffmpeg is required for MP4 rendering` when the system writer is unavailable; it never installs packages or runs an environment.

The renderer infers the method name from enabled modules (with optional metadata override), keeps persistent trajectory/marker/heading artists and updates them incrementally, and displays Red/Blue labels, real-trace death markers, structured hit/kill/loss/wave events, and a final result card. `--event-hold-seconds` defaults to 1.0, `--final-hold-seconds` to 2.0, and `--preview-frame` defaults deterministically to the midpoint. `--trail-length 0` means full trace history; a positive value is a trace-frame window. The recorder's `--wall-timeout-s` defaults to 0 (disabled), so environment termination remains authoritative.

For a faster quick video use `--fps 30 --frame-stride 4 --no-heading`. For a detailed reference-like render use `--fps 20 --frame-stride 1 --event-hold-seconds 1.0 --final-hold-seconds 2.0`. No attacker-target kill identity is fabricated because the environment does not expose an unambiguous mapping.
