"""Save separate MAPPO training/evaluation metric figures."""
from __future__ import annotations
import argparse,csv
from pathlib import Path
import matplotlib.pyplot as plt
def plot(path:Path,x:str,y:str,out:Path)->None:
    with path.open(encoding="utf-8") as f: rows=list(csv.DictReader(f))
    if not rows or y not in rows[0]: return
    fig=plt.figure(); plt.plot([float(r[x]) for r in rows],[float(r[y]) for r in rows]); plt.xlabel(x); plt.ylabel(y); plt.tight_layout(); fig.savefig(out,dpi=160); plt.close(fig)
def plot_actions(path:Path,out:Path)->None:
    with path.open(encoding="utf-8") as f: rows=list(csv.DictReader(f))
    if not rows: return
    fig=plt.figure()
    x=[float(r["environment_steps"]) for r in rows]
    for index in range(15): plt.plot(x,[float(r[f"action_{index}_frequency"]) for r in rows],label=str(index))
    plt.xlabel("environment_steps");plt.ylabel("action frequency");plt.legend(ncol=3,fontsize=7);plt.tight_layout();fig.savefig(out,dpi=160);plt.close(fig)
def main()->None:
    p=argparse.ArgumentParser();p.add_argument("run_dir_positional",nargs="?");p.add_argument("--run-dir");a=p.parse_args();run_dir=a.run_dir or a.run_dir_positional
    if not run_dir:p.error("a run directory is required")
    root=Path(run_dir);out=root/"plots";out.mkdir(exist_ok=True)
    for name in ("rollout_return_mean","policy_loss","value_loss","entropy","observation_saturation_mean","approx_kl","clip_fraction"): plot(root/"metrics.csv","environment_steps",name,out/f"{name}.png")
    for name in ("red_win_rate","blue_win_rate","draw_rate","timeout_rate","red_crash_rate","blue_crash_rate","mean_episode_steps"): plot(root/"evaluations.csv","environment_steps",name,out/f"{name}.png")
    plot_actions(root/"evaluations.csv",out/"action_distribution.png")
if __name__=="__main__":main()
