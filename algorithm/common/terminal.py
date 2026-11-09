"""Plain-text experiment logs. Formatting never drives the experiment."""
import math
import sys


def number(value, unit="", *, style=".3g"):
    if value is None or not math.isfinite(value):
        return "N/A"
    return format(value, style) + (f" {unit}" if unit else "")


def percent(value):
    return number(value, style=".1%") if value is not None and 0 <= value <= 1 else "N/A"


def rate(count, total):
    return percent(count / total if total else None)


def quantity(value, unit):
    return f"{value:,} {unit}{'' if value == 1 else 's'}"


def seed_span(seeds):
    if not seeds:
        return "N/A"
    if len(seeds) == 1:
        return str(seeds[0])
    if all(seed == seeds[0] + index for index, seed in enumerate(seeds)):
        return f"{seeds[0]}-{seeds[-1]}"
    return ",".join(map(str, seeds))


def emit(experiment, phase, *sections):
    message = " | ".join(section for section in sections if section)
    print(f"[{experiment} {phase}] {message}", file=sys.stderr if phase == "ERROR" else sys.stdout, flush=True)


def run_cli(experiment, command):
    try:
        command()
    except KeyboardInterrupt:
        emit(experiment, "ERROR", "Interrupted")
        raise SystemExit(130) from None
    except (OSError, ValueError, RuntimeError, FloatingPointError) as exc:
        lines = str(exc).strip().splitlines()
        detail = lines[-1] if lines and lines[0].startswith("Traceback") else " ".join(lines)
        emit(experiment, "ERROR", detail or type(exc).__name__)
        raise SystemExit(1) from None


def evaluation_setup(interval, seeds=(), *, scenario, actions, final=False):
    if not interval:
        return "Periodic evaluation off"
    schedule = f"every {quantity(interval, 'update')}" + (" plus final update" if final else "")
    return (f"Periodic evaluation {schedule}, n={len(seeds):,}, seeds={seed_span(seeds)}, "
            f"scenario={scenario}, actions={actions}")


def training_start(experiment, scenario, training, *, start_update, planned_updates,
                   evaluation, folder, resume=False, stop_steps=None):
    wave = training.workers * training.horizon
    budget = f"Budget {quantity(planned_updates, 'update')} / {quantity(planned_updates * wave, 'joint step')}"
    if stop_steps is not None and stop_steps < planned_updates * wave:
        budget += f", stop at {stop_steps:,} joint steps"
    emit(experiment, "RESUME" if resume else "START", scenario, f"Train seed {training.seed}",
         budget, f"From update {start_update:,} / {start_update * wave:,} joint steps" if resume else None,
         f"Sampling {quantity(training.workers, 'worker')} x {quantity(training.horizon, 'step')}",
         evaluation, f"Output {folder}")


def training_progress(experiment, record, *, total_steps, window=None):
    update, planned = record["update"], record["planned_updates"]
    diagnostics = [f"Critic loss {number(record.get('critic_loss'))}",
                   f"Approx KL {number(record.get('approx_kl'))}"]
    if "sampling_speed" in record:
        diagnostics.append(f"Sampling {number(record['sampling_speed'], 'joint steps/s')}")
    emit(experiment, "TRAIN", f"Update {update:,}/{planned:,} ({rate(update, planned)})",
         f"Joint steps {record['joint_steps']:,}/{total_steps:,}",
         f"Completed episodes {record['completed_episodes_total']:,}", window, ", ".join(diagnostics))


def training_end(experiment, model, checkpoint, *, paused=False):
    emit(experiment, "PAUSE" if paused else "DONE", f"Update {model.updates:,}",
         f"Joint steps {model.steps:,}", f"Completed episodes {model.episodes:,}", f"Checkpoint {checkpoint}")


def interception_results(summary, *, window=False):
    n = summary["episodes"]
    if window and not n:
        return "Train window n=0/100, no completed episodes yet"
    label = f"Train window n={n}/100" if window else f"n={quantity(n, 'episode')}"
    fields = [label, f"All-target capture {percent(summary['all_captured_rate'])}",
              f"Protection success {percent(summary['protection_success_rate'])}",
              f"Breach {percent(summary['breach_rate'])}",
              f"Mean return {number(summary['mean_episode_return'])}"]
    if not window:
        fields.append(f"Mean all-target capture time {number(summary['mean_capture_time_s'], 's')}")
    if summary["numerical_failure_count"]:
        fields.append(f"Simulation failures {summary['numerical_failure_count']:,}")
    return ", ".join(fields)


def tracking_results(summary):
    fields = [f"n={quantity(summary['episodes'], 'episode')}", f"Tracking success {percent(summary['success_rate'])}",
              f"Mean position error {number(summary['mean_position_error'], 'm')}",
              f"Mean return {number(summary['team_return'])}"]
    if summary["outcomes"]["numerical_failure"]:
        fields.append(f"Numerical failures {summary['outcomes']['numerical_failure']:,}")
    return ", ".join(fields)
