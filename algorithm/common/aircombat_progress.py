"""Read-only terminal summaries for air-combat records; no simulator/Torch calls."""
from collections import deque
import time
from .progress import number,percent,duration


def episode_summary(episodes):
    n=len(episodes)
    def mean(key):return sum(e[key] for e in episodes)/n if n else None
    counts=dict(normal=sum(e.get('outcome') in ('win','red_eliminated','mutual_elimination') for e in episodes),
                timeout=sum(e.get('outcome')=='timeout' for e in episodes),
                numerical_failure=sum(e.get('outcome')=='numerical_failure' for e in episodes))
    return dict(n=n,return_mean=mean('episode_return'),win_rate=mean('win'),
                red=mean('red_survivors'),blue=mean('blue_shot_down'),counts=counts)


def format_endings(counts,n):
    return ' | '.join(f'{label} {counts[key]} ({percent(counts[key]/n if n else None)})'
                    for key,label in (('normal','Normal'),('timeout','Timeout'),('numerical_failure','Numerical Failure')))


def format_training(record,total,episodes,*,elapsed,throughput):
    s=episode_summary(episodes);p=record['ppo'];completed=record['completed_episodes']
    return (
        f"[TRAIN] Episodes {completed}/{total} ({percent(completed/total)})"
        f" | Joint steps {record['sampled_steps']} | PPO updates {record['updates']}"
        f" | Elapsed {duration(elapsed)} | Throughput {number(throughput,1)} joint steps/s (session)\n"
        f"        Recent n={s['n']}/100 | Return {number(s['return_mean'])} | Train Win Rate {percent(s['win_rate'])}"
        f" | Red alive {number(s['red'],2)} | Blue shot down {number(s['blue'],2)}\n"
        f"        Ends(recent): {format_endings(s['counts'],s['n'])}\n"
        f"        PPO: Actor Total Loss {number(p.get('actor_total_loss'),5)} | Policy Loss {number(p.get('policy_loss'),5)}"
        f" | Critic Loss {number(p.get('critic_loss'),5)} | Entropy {number(p.get('entropy'),5)}"
        f" | Approx KL {number(p.get('approx_kl'),6)} | Clip Fraction {number(p.get('clip_fraction'),5)}"
        f" | Valid Samples {number(p.get('valid_samples'),0)}")


def format_evaluation(report,checkpoint):
    counts=report['termination_counts'];n=report['completed_episodes']
    return (f"[EVAL] independent n={n} | Win Rate {percent(report['win_rate'])}"
            f" | Red alive {number(report['mean_red_survivors'],2)} | Blue shot down {number(report['mean_blue_shot_down'],2)}"
            f" | {format_endings(counts,n)} | Model {checkpoint}")


class TrainingConsole:
    def __init__(self,total,*,enabled=True,every=5,start_steps=0,history=(),started_at=None,clock=time.monotonic):
        if type(every) is not int or every<1:raise ValueError('log_every_updates must be a positive integer')
        self.total,self.enabled,self.every=total,enabled,every
        self.clock=clock;self.started_at=clock() if started_at is None else started_at
        self.start_steps=start_steps;self.recent=deque(maxlen=100)
        self.counts=dict(normal=0,timeout=0,numerical_failure=0);self.known_episodes=0
        self.last_bucket=0;self.first=True
        self.add(history)

    def emit(self,line):
        if self.enabled:print(line,flush=True)

    def add(self,episodes):
        self.recent.extend(episodes)
        s=episode_summary(episodes);self.known_episodes+=s['n']
        for key in self.counts:self.counts[key]+=s['counts'][key]

    def start(self,trainer,vector,env_config,output,*,resume=None):
        self.last_bucket=trainer.training_episode_count//50
        self.emit(f"[START] scale={env_config.scale} | seed={trainer.config.training_seed_index}"
                  f" | Episodes {trainer.training_episode_count}/{self.total}"
                  f" | Workers {len(vector.worker_pids)} (spawn) | GPU {trainer.device}"
                  f" | horizon={trainer.config.horizon} | max_steps={env_config.max_steps}"
                  f" | integrator={env_config.integrator}"
                  f" | init={env_config.initial_scenario}"
                  f" | input_norm={trainer.config.observation_normalization}"
                  f" | posture={env_config.posture_reward_mode} | gate_speed={env_config.posture_gate_speed:g}m/s"
                  + (f" | reward_radius={env_config.madsac_battlefield_radius:g}m" if env_config.posture_reward_mode=='madsac_inspired' else '')
                  +
                  f" | log_every_updates={self.every} | output={output}"
                  + (f" | resume={resume}" if resume else ''))

    def update(self,record):
        self.add(record['episodes'])
        bucket=record['completed_episodes']//50
        due=self.first or record['updates']%self.every==0 or bucket>self.last_bucket or record['completed_episodes']>=self.total
        self.first=False;self.last_bucket=bucket
        if due and self.enabled:
            elapsed=max(0.,self.clock()-self.started_at)
            speed=(record['sampled_steps']-self.start_steps)/elapsed if elapsed>0 else None
            self.emit(format_training(record,self.total,list(self.recent),elapsed=elapsed,throughput=speed))

    def checkpoint(self,trainer,path):
        self.emit(f"[CKPT] Episodes {trainer.training_episode_count}/{self.total}"
                  f" | PPO updates {trainer.updates} | File {path.name}")

    def done(self,trainer):
        elapsed=max(0.,self.clock()-self.started_at)
        self.emit(f"[DONE] training Episodes {trainer.training_episode_count}/{self.total}"
                  f" | Joint steps {trainer.sampled_steps} | PPO updates {trainer.updates}"
                  f" | Elapsed {duration(elapsed)} | Ends(recorded): {format_endings(self.counts,self.known_episodes)}")
