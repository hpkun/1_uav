"""Side-effect-free, standard-library console formatting for recorded metrics."""
import math
import time


def finite_number(value):
    try:
        result=float(value) if value is not None else None
    except (TypeError,ValueError,OverflowError):
        return None
    return result if result is not None and math.isfinite(result) else None


def number(value,digits=3,*,signed=False,group=False):
    value=finite_number(value)
    if value is None: return 'N/A'
    return format(value,('+' if signed else '')+(',' if group else '')+f'.{digits}f')


def percent(value,digits=1):
    value=finite_number(value)
    return 'N/A' if value is None else number(value*100,digits)+'%'


def duration(seconds):
    seconds=finite_number(seconds)
    if seconds is None or seconds<0: return 'N/A'
    total=math.ceil(seconds)
    hours,remainder=divmod(total,3600); minutes,seconds=divmod(remainder,60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def loss_values(optimization):
    """Old v2 actor_loss was policy loss; never invent an unrecorded total."""
    values=dict(optimization) if isinstance(optimization,dict) else {}
    legacy='policy_loss' not in values and 'actor_total_loss' not in values and 'actor_loss' in values
    values['policy_loss']=values.get('policy_loss',values.get('actor_loss') if legacy else None)
    values['actor_total_loss']=values.get('actor_total_loss',None)
    return values,legacy


class ProgressClock:
    """Session wall throughput (including evaluation), excluding old resumed steps."""
    def __init__(self,start_steps,clock=time.monotonic):
        self.clock=clock; self.start_steps=start_steps; self.start_time=clock()

    def measure(self,steps,total):
        elapsed=self.clock()-self.start_time
        speed=(steps-self.start_steps)/elapsed if elapsed>0 and steps>self.start_steps else None
        eta=max(0,total-steps)/speed if speed else None
        return speed,eta


def format_training(record,total,*,speed=None,eta=None):
    training=record.get('training') or {}; rollout=record.get('rollout') or {}
    opt,_=loss_values(record.get('optimization'))
    steps=record.get('sampled_steps'); total_value=finite_number(total)
    fraction=finite_number(steps)/total_value if finite_number(steps) is not None and total_value and total_value>0 else None
    # Exactly two logical lines, with stable labels and no ANSI/cursor controls.
    first=(f"[TRAIN] {number(steps,0,group=True)}/{number(total,0,group=True)} ({percent(fraction)})"
        f" | Update {number(record.get('update'),0)} | Episodes {number(record.get('joint_episode_count'),0,group=True)}"
        f" | Speed {number(speed,0,group=True)} joint steps/s | ETA {duration(eta)}"
        f" | Train: Success {percent(training.get('success_rate'))} | Completion {percent(training.get('completion_rate'))}"
        f" | Return {number(training.get('mean_episode_return'),3)}")
    second=(f"        Task: Distance {number(rollout.get('mean_target_distance'),0,group=True)} m"
        f" | Distance reduction {number(rollout.get('target_distance_reduction'),0,signed=True,group=True)} m"
        f" | PPO: Policy {number(opt.get('policy_loss'),4)} | Actor total {number(opt.get('actor_total_loss'),4)}"
        f" | Critic {number(opt.get('critic_loss'),4)} | KL {number(opt.get('approx_kl'),5)}"
        f" | Clip {percent(opt.get('clip_fraction'),2)} | Entropy {number(opt.get('entropy'),2)}"
        f" | Flight: Speed clipping {percent(rollout.get('speed_clip_rate'))} | Gamma clipping {percent(rollout.get('gamma_clip_rate'))}")
    return first+'\n'+second


def episode_count(report):
    if not isinstance(report,dict): return None
    rows=report.get('episodes')
    return len(rows) if isinstance(rows,list) else finite_number(report.get('joint_episode_count'))


def format_evaluation(steps,report,total_waypoints,*,best_updated,role='current'):
    report=report or {}; n=episode_count(report)
    rows=report.get('episodes')
    k=sum(bool(r['success']) for r in rows) if isinstance(rows,list) and all(isinstance(r,dict) and 'success' in r for r in rows) else None
    return (f"[EVAL] Steps {number(steps,0,group=True)} | Episodes n={number(n,0)} | Model {role}\n"
        f"       Success {percent(report.get('success_rate'))} ({number(k,0)}/{number(n,0)})"
        f" | Completion {percent(report.get('completion_rate',report.get('mean_completion_rate')))}"
        f" | Completed waypoints {number(report.get('mean_completed_waypoints'),1)}/{number(total_waypoints,0)}\n"
        f"       Mean Return {number(report.get('mean_reward',report.get('mean_episode_return')),3)}"
        f" | Mean Episode Length {number(report.get('mean_episode_length'),1)}"
        f" | {'BEST UPDATED' if best_updated else 'BEST UNCHANGED'}")


def format_model(label,report,total_waypoints):
    report=report or {}
    return (f"{label}: n={number(episode_count(report),0)} | Success {percent(report.get('success_rate'))}"
        f" | Completion {percent(report.get('completion_rate',report.get('mean_completion_rate')))}"
        f" | Completed {number(report.get('mean_completed_waypoints'),1)}/{number(total_waypoints,0)}"
        f" | Return {number(report.get('mean_reward',report.get('mean_episode_return')),3)}")


def format_final(summary,run_dir,total_waypoints):
    from pathlib import Path
    run_dir=Path(run_dir).resolve()
    paths=[f"best.pt: {summary.get('best_checkpoint') or 'N/A'}",f"latest.pt: {summary.get('latest_checkpoint') or 'N/A'}"]
    paths.extend(f'{name}: {run_dir/name}' for name in ('training_metrics.jsonl','evaluation_history.csv','run_summary.json'))
    return '\n'.join([f"[DONE] Sampled steps {number(summary.get('sampled_steps'),0,group=True)}"
        f" | PPO updates {number(summary.get('updates'),0)} | Training episodes {number(summary.get('joint_episode_count'),0)}",
        '       '+format_model('BEST',summary.get('best_evaluation'),total_waypoints),
        '       '+format_model('LATEST',summary.get('latest_evaluation'),total_waypoints),
        *('       '+p for p in paths)])
