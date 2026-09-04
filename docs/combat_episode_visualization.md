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

For an offline, mouse-driven replay, generate a standalone Plotly HTML from the same trace (no checkpoint, environment, or CUDA is loaded):

```bash
python -u tools/render_combat_episode_interactive.py \
  --input-dir outputs/visualization/ea_wb_seed40000000 \
  --output outputs/visualization/ea_wb_seed40000000/episode_interactive.html
```

The HTML embeds Plotly JS and supports rotate, zoom, pan, Play/Pause, Previous/Next/Restart, a time slider, 0.25x-4x speed, trail windows, heading/label/death-marker toggles, Recent Events, a final result panel, and Reset Camera. It is intended to be opened directly from disk and does not use a CDN. MP4 remains fixed-camera and cannot be mouse-rotated.

At each entry-triggered fire attempt, the interactive renderer displays a thick green attacker-to-target line for 0.6 simulation seconds. For schema-v1 traces, these pairs are reconstructed deterministically from the recorded post-motion states, alive masks, weapon envelope, nearest-in-window targeting rule, and per-step fire-attempt counts; generation fails rather than inventing a pair if reconstructed and recorded counts disagree. Death-marker hover text reports `Weapon attack`, `Ground impact`, `Boundary exit`, or `Unknown`, derived from the recorded alive transition and per-step cause counters. This adds no fields to the trace and does not change `trace_schema_version=1`.

Interactive playback uses a wall-clock logical playback clock and a separate `requestAnimationFrame` render scheduler. Slow WebGL updates may skip obsolete visual frames, but they do not slow simulation time: the renderer always catches up directly to the latest logical frame and never queues every missed frame. Plotly mutations are serialized, and trajectory/current-marker/heading/death updates are batched into about five data calls per rendered frame instead of one call per aircraft trace.

Camera interaction is playback-aware. While the user holds and drags on the 3D plot, or while wheel events are active, all Plotly combat-data mutations stop and the currently visible combat traces remain frozen. The Arena, fixed-range axes, grid, and combat traces therefore retain exactly the same geometry while Plotly rotates or zooms the camera. During playback the logical clock continues; releasing the pointer, cancelling the pointer, or stopping wheel input for 200 ms renders only the latest logical frame. While paused, a camera-only interaction does not schedule any combat-data refresh at its start or end. Skipped old frames are never replayed and `uirevision='combat-replay'` preserves the user's camera. The axes explicitly disable autorange, `aspectmode='cube'` gives the X/Y/Z axes equal on-screen lengths without changing their numeric ranges, the scene uses orbit dragging, and scroll zoom is enabled. Pause and manual seek retain their normal semantics, and changing speed re-anchors the clock without a time jump.

The renderer infers the method name from enabled modules (with optional metadata override), keeps persistent trajectory/marker/heading artists and updates them incrementally, and displays Red/Blue labels, real-trace death markers, structured hit/kill/loss/wave events, and a final result card. `--event-hold-seconds` defaults to 1.0, `--final-hold-seconds` to 2.0, and `--preview-frame` defaults deterministically to the midpoint. `--trail-length 0` means full trace history; a positive value is a trace-frame window. The recorder's `--wall-timeout-s` defaults to 0 (disabled), so environment termination remains authoritative.

For a faster quick video use `--fps 30 --frame-stride 4 --no-heading`. For a detailed reference-like render use `--fps 20 --frame-stride 1 --event-hold-seconds 1.0 --final-hold-seconds 2.0`. No attacker-target kill identity is fabricated because the environment does not expose an unambiguous mapping.
