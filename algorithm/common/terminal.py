"""Human-readable experiment output; this module never drives the experiment."""
import math
from pathlib import Path
import sys


def number(value, unit="", *, style=".3g"):
    if value is None or not math.isfinite(value):
        return "N/A"
    return format(value, style) + (f" {unit}" if unit else "")


def percent(value):
    return number(value, style=".1%") if value is not None and 0 <= value <= 1 else "N/A"


def rate(count, total):
    return percent(count / total if total else None)


def decimal(value):
    """Display a measured speed without scientific notation or comma grouping."""
    if value is None or not math.isfinite(value):
        return "N/A"
    if abs(value) >= 1:
        return format(value, ".0f")
    return format(value, ".3f").rstrip("0").rstrip(".") or "0"


def sampling_speed(value):
    shown = decimal(value)
    return shown + " steps/s" if shown != "N/A" else shown


def fields(**values):
    return " | ".join(f"{key}={value}" for key, value in values.items() if value is not None)


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
    print(f"[{phase.lower()}] {message}", file=sys.stderr if phase.upper() == "ERROR" else sys.stdout, flush=True)


def run_cli(experiment, command, *, debug=False):
    try:
        command()
    except KeyboardInterrupt:
        emit(experiment, "ERROR", fields(message="Interrupted"))
        raise SystemExit(130) from None
    except Exception as exc:
        if debug:
            raise
        lines = str(exc).strip().splitlines()
        detail = lines[-1] if lines and lines[0].startswith("Traceback") else " ".join(lines)
        emit(experiment, "ERROR", fields(message=detail or type(exc).__name__))
        raise SystemExit(1) from None


def evaluation_setup(interval, seeds=(), *, scenario, actions, final=False):
    return fields(eval_every=interval or "off", eval_ep=len(seeds) if interval else 0,
                  eval_seeds=seed_span(seeds) if interval else None,
                  eval_scenario=scenario if interval else None,
                  eval_actions=actions if interval else None,
                  eval_final="yes" if final and interval else None)


def training_start(experiment, scenario, training, *, start_update, planned_updates,
                   evaluation, folder, resume=False, stop_steps=None, requested_steps=None):
    wave = training.workers * training.horizon
    emit(experiment, "RESUME" if resume else "START", scenario,
         fields(seed=training.seed, upd=planned_updates, steps=planned_updates * wave,
                requested=requested_steps, from_upd=start_update if resume else None,
                from_steps=start_update * wave if resume else None,
                stop_steps=stop_steps if stop_steps is not None and stop_steps < planned_updates * wave else None,
                workers=training.workers, horizon=training.horizon),
         evaluation, fields(out=folder))


def training_progress(experiment, record, *, total_steps, window=None):
    update, planned = record["update"], record["planned_updates"]
    emit(experiment, "TRAIN", fields(upd=f"{update}/{planned} ({rate(update, planned)})",
         steps=f"{record['joint_steps']}/{total_steps}", ep=record["completed_episodes_total"]),
         window, fields(critic=number(record.get("critic_loss")),
                        kl=number(record.get("approx_kl")),
                        speed=sampling_speed(record["sampling_speed"])
                        if "sampling_speed" in record else None))


def training_end(experiment, model, checkpoint, *, paused=False):
    emit(experiment, "PAUSE" if paused else "DONE",
         fields(upd=model.updates, steps=model.steps, ep=model.episodes,
                checkpoint=Path(checkpoint).name))


def interception_results(summary, *, window=False):
    n = summary["episodes"]
    return fields(**({"win": f"{n}/100"} if window else {"ep": n}),
                  capture=percent(summary["all_captured_rate"]) if n else "N/A",
                  protect=percent(summary["protection_success_rate"]) if n else "N/A",
                  breach=percent(summary["breach_rate"]) if n else "N/A",
                  reward=number(summary["mean_episode_return"]) if n else "N/A",
                  **({"capture_time": number(summary["mean_capture_time_s"], "s")}
                     if not window and summary["mean_capture_time_s"] is not None else {}),
                  failures=summary["numerical_failure_count"] or None)


def tracking_results(summary):
    return fields(ep=summary["episodes"], success=percent(summary["success_rate"]),
                  pos_err=number(summary["mean_position_error"], "m"),
                  reward=number(summary["team_return"]),
                  failures=summary["outcomes"]["numerical_failure"] or None)
