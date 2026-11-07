"""Research journal and derived views. This module does not import Torch."""
import csv
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile

from .serialization import serializable

OUTCOMES = ("success", "breach", "collision", "timeout", "simulation_failure")


@dataclass(frozen=True)
class EvaluationProtocol:
    evaluation_interval_updates: int = 0
    evaluation_episodes: int = 5
    evaluation_seed_base: int = 1000
    evaluation_scenario: str = "random"
    evaluation_deterministic: bool = True

    def __post_init__(self):
        for name, minimum in (("evaluation_interval_updates", 0), ("evaluation_episodes", 1),
                              ("evaluation_seed_base", 0)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"invalid {name}")
        if self.evaluation_scenario not in ("fixed", "random") or type(self.evaluation_deterministic) is not bool:
            raise ValueError("invalid evaluation scenario or deterministic flag")

    @property
    def seeds(self):
        return list(range(self.evaluation_seed_base, self.evaluation_seed_base + self.evaluation_episodes))


def encoded(value):
    return json.dumps(serializable(value), sort_keys=True, allow_nan=False, ensure_ascii=False)


def digest(value):
    return hashlib.sha256(encoded(value).encode("utf-8")).hexdigest()


def file_digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def atomic_bytes(path, data):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            os.replace(temporary, path)
        finally:
            stream.close()
            temporary.unlink(missing_ok=True)


def atomic_json(path, value):
    atomic_bytes(path, (encoded(value) + "\n").encode("utf-8"))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarize(rows):
    from statistics import mean, median
    total = len(rows)
    outcomes = {name: sum(row["outcome"] == name for row in rows) for name in OUTCOMES}
    times = [row["time"] for row in rows]
    successful = [row["time"] for row in rows if row["outcome"] == "success"]
    targets = sorted({target for row in rows for target in row.get("target_ids", [])})
    target_stats = {}
    for target in targets:
        trials = [row for row in rows if target in row["target_ids"]]
        captures = sum(target in row["captured_ids"] for row in trials)
        target_stats[str(target)] = dict(captures=captures, trials=len(trials), rate=captures / len(trials))
    parts = sorted({key for row in rows for key in row["reward_parts"]})
    separations = [row["minimum_separation"] for row in rows]
    partial = [row for row in rows if row.get("captured_ids") and
               len(row["captured_ids"]) < len(row["target_ids"])]
    return dict(episodes=total, outcomes=outcomes, **{
                    name + "_rate": outcomes[name] / total if total else None
                    for name in OUTCOMES[:4]},
                numerical_failure_count=outcomes["simulation_failure"],
                all_captured_rate=outcomes["success"] / total if total else None,
                partial_capture_episodes=len(partial),
                per_target_capture=target_stats,
                capture_time_samples=len(successful),
                mean_capture_time_s=mean(successful) if successful else None,
                median_capture_time_s=median(successful) if successful else None,
                median_episode_time_s=median(times) if times else None,
                minimum_separation_m=min(separations, default=None),
                mean_minimum_separation_m=mean(separations) if separations else None,
                mean_episode_return=mean(row["episode_return"] for row in rows) if total else None,
                mean_control_effort=mean(row["control_effort"] for row in rows) if total else None,
                mean_reward_components={key: mean(row["reward_parts"].get(key, 0.) for row in rows)
                                        for key in parts})


def commit_header(path):
    try:
        with zipfile.ZipFile(path) as archive:
            key = next(k for k in archive.namelist() if k.endswith("/research_commit.json"))
            return json.loads(archive.read(key))
    except (zipfile.BadZipFile, StopIteration) as exc:
        raise ValueError("research commit metadata required. Start training in a new run directory.") from exc


def journal(folder, head, *, repair=False):
    run, count, final = head["run"], head["update"], head["last_log"]
    paths = sorted(folder.glob("update_*.json"))
    expected = [folder / f"update_{i:06}.json" for i in range(1, count + 1)]
    missing_final = bool(expected and paths == expected[:-1])
    if paths != expected and not (missing_final and repair):
        raise ValueError("update journal gap or newer records than checkpoint")
    rows = [read_json(path) for path in paths] + ([final] if missing_final else [])
    previous, episodes = None, 0
    wave = run["training"]["workers"] * run["training"]["horizon"]
    for i, row in enumerate(rows, 1):
        episodes += len(row["completed_episodes"])
        if (row["update"], row["joint_steps"], row["completed_episodes_total"], row["previous"], row["run_id"]) != (
            i, i * wave, episodes, previous, run["id"]
        ):
            raise ValueError("inconsistent training journal")
        previous = digest(row)
    if rows and rows[-1] != final:
        raise ValueError("latest journal record differs from checkpoint")
    if missing_final:
        atomic_json(expected[-1], final)
    return rows


def evaluation_records(folder, run, count):
    protocol = EvaluationProtocol(**run["evaluation"])
    interval = protocol.evaluation_interval_updates
    paths = sorted(folder.glob("evaluation_*.json"))
    rows = []
    for path in paths:
        item = read_json(path)
        update = item["update"]
        if (not interval or not 0 < update <= count or update % interval or
                path.name != f"evaluation_{update:06}.json" or item["run_id"] != run["id"] or
                item["joint_steps"] != update * run["training"]["workers"] * run["training"]["horizon"] or
                item["protocol"] != asdict(protocol) or item["seeds"] != protocol.seeds or
                [row["seed"] for row in item["runs"]] != protocol.seeds or
                item["summary"] != summarize(item["runs"])):
            raise ValueError("inconsistent evaluation commit")
        rows.append(item)
    expected = list(range(interval, count, interval)) if interval else []
    if [row["update"] for row in rows if row["update"] < count] != expected:
        raise ValueError("missing historical evaluation commit")
    return rows


def score(row):
    stats = row["summary"]
    capture = stats["mean_capture_time_s"]
    return (-stats["success_rate"], stats["breach_rate"], stats["collision_rate"],
            float("inf") if capture is None else capture, row["update"])


def publish_best(folder, evaluations, update):
    best = folder / "best.pt"
    metadata = folder / "best.json"
    if not evaluations:
        if best.exists() or metadata.exists():
            raise ValueError("unverifiable Best files")
        return
    winner = min(evaluations, key=score)
    known = {item["checkpoint_sha256"] for item in evaluations}
    current = file_digest(best) if best.exists() else None
    if current is not None and current not in known:
        raise ValueError("Best checkpoint differs from evaluated models")
    wanted = winner["checkpoint_sha256"]
    if current != wanted:
        latest = folder / "latest.pt"
        if winner["update"] != update or file_digest(latest) != wanted:
            raise ValueError("cannot recover historical Best weights")
        atomic_bytes(best, latest.read_bytes())
    stats = winner["summary"]
    atomic_json(metadata, dict(best_update=winner["update"], best_joint_steps=winner["joint_steps"],
                              best_score={key: stats[key] for key in ("success_rate", "breach_rate",
                                          "collision_rate", "mean_capture_time_s")},
                              evaluation_protocol=winner["protocol"],
                              checkpoint_source=winner["checkpoint_source"], checkpoint_sha256=wanted))


def metric_rows(record):
    training = {key: value for key, value in record.items()
                if key not in ("completed_episodes", "previous")}
    episodes = [row | dict(update=record["update"], run_id=record["run_id"], scenario="random",
                          episode_length=row["steps"], physical_time=row["time"],
                          reward_components=row["reward_parts"])
                for row in record["completed_episodes"]]
    return training, episodes


def write_views(folder, records, *, append=False):
    metrics, episodes = [], []
    for row in records:
        metric, finished = metric_rows(row)
        metrics.append(metric)
        episodes.extend(finished)
    for name, data in (("training_metrics.jsonl", metrics), ("episode_metrics.jsonl", episodes)):
        path = folder / name
        content = "".join(encoded(row) + "\n" for row in data)
        if append:
            with path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
        else:
            atomic_bytes(path, content.encode("utf-8"))


def evaluation_csv(folder, rows):
    fields = ["update", "joint_steps", "scenario", "seeds", "success_rate", "breach_rate",
              "collision_rate", "timeout_rate", "mean_capture_time_s", "mean_episode_return"]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        values = row["summary"] | row
        values["scenario"] = row["protocol"]["evaluation_scenario"]
        values["seeds"] = encoded(row["seeds"])
        writer.writerow({key: values[key] for key in fields})
    atomic_bytes(folder / "evaluation_history.csv", buffer.getvalue().encode("utf-8"))


def terminal_line(record, recent):
    window = summarize(recent[-100:])

    def show(value, style=".3f"):
        return "N/A" if value is None else format(value, style)

    return (f"[TRAIN] update {record['update']}/{record['planned_updates']} | "
            f"joint steps {record['joint_steps']} | episodes {record['completed_episodes_total']} | "
            f"window n={window['episodes']}/100 success {show(window['success_rate'], '.1%')} "
            f"breach {show(window['breach_rate'], '.1%')} collision {show(window['collision_rate'], '.1%')} "
            f"timeout {show(window['timeout_rate'], '.1%')} return {show(window['mean_episode_return'])} "
            f"capture {show(window['mean_capture_time_s'])} s (n={window['capture_time_samples']}) | "
            f"actor {show(record['actor_total_loss'])} critic {show(record['critic_loss'])} "
            f"entropy {show(record['entropy'])} KL {show(record['approx_kl'], '.3g')} "
            f"clip {show(record['clip_fraction'], '.1%')} valid {record['valid_samples']} | "
            f"{record['sampling_speed']:.1f} joint steps/s")
