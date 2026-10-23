"""Tables VI-X evaluation traces and static Fig 13-17 style plots."""
import argparse
from pathlib import Path
import textwrap
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from algorithm.experiments import load_model,write_json
from env.aircombat import AirCombatEnv
from env.aircombat.scenarios import TABLES
from algorithm.common.aircombat_vector import CombatEpisodeAccumulator
from tools.report_aircombat import save_figure
from env.aircombat.geometry import pair_advantages


def record(checkpoint,scenario,output,*,seed=10000000):
    trainer,config,payload=load_model(checkpoint)
    if scenario=='random_training':table=None
    else:
        table,n,rows=TABLES[scenario]
        if (config.num_uavs,config.num_adversaries)!=(n,len(rows)-n):raise ValueError('checkpoint/scenario scale mismatch')
    if seed in trainer.config.worker_seeds():raise ValueError('trajectory evaluation seed overlaps training')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    env=AirCombatEnv(config);obs,info=env.reset(seed,{'scenario':scenario})
    tracker=CombatEpisodeAccumulator();tracker.reset(info)
    def frame(info):
        return dict(step=info['episode_length'],friendly_states=info['friendly_states'].tolist(),
                    adversarial_states=info['adversarial_states'].tolist(),friendly_alive=info['friendly_alive'].tolist(),
                    adversarial_alive=info['adversarial_alive'].tolist(),assignments=info['assignments'].tolist())
    frames=[frame(info)];events=[];changes=[]
    while True:
        previous=info['assignments'].copy()
        actions,_=trainer.sample_actions(obs,info['active_mask'],deterministic=True)
        obs,r,term,trunc,info=env.step(actions.cpu().numpy())
        tracker.add(info,r)
        current=frame(info)
        # Read the same post-motion geometry and pre-death alive mask used by
        # resolve_attacks; do not interpret APH assignments as weapon triggers.
        _,red_attack,blue_attack,_=pair_advantages(env.state.friendly_states,env.state.adversarial_states,config)
        prior=frames[-1]
        live=np.array(prior['friendly_alive'])[:,None]&np.array(prior['adversarial_alive'])[None,:]
        current.update(red_actions=actions.cpu().numpy().tolist(),blue_controls=info['blue_controls'].tolist(),
                       red_controls=info['red_controls'].tolist(),
                       red_attack_eligible=((red_attack>0)&live).tolist(),
                       blue_attack_eligible=((blue_attack>0)&live.T).tolist(),
                       numerical_diagnostics=info.get('numerical_diagnostics',{}),
                       reward=r.tolist(),reward_components={k:v.tolist() for k,v in info['reward_components'].items()},
                       risk_terms={k:v.tolist() for k,v in info['risk_terms'].items()},
                       numerical_failure=info['numerical_failure'],failures=info['failures'])
        frames.append(current);events.extend(info['events'])
        for i in np.flatnonzero(previous!=info['assignments']):
            changes.append(dict(step=info['episode_length'],friendly_id=int(i),previous_target=int(previous[i]),target=int(info['assignments'][i])))
        if term or trunc:break
    data=dict(scenario=scenario,table=table,seed=seed,checkpoint=str(Path(checkpoint).resolve()),
              checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
              environment_version=env.version,updates=payload['updates'],sampled_steps=payload['sampled_steps'],
              checkpoint_phase=payload.get('phase'),policy='deterministic Beta mean',
              completed_training_episodes=payload['training_episode_count'],training_seed=trainer.config.training_seed_index,
              environment=config.to_dict(),frames=frames,events=events,assignment_changes=changes,summary=tracker.summary(info))
    write_json(output/'trace.json',data);plot_trace(data,output)
    return data


def plot_trace(data,output):
    output=Path(output);frames=data['frames'];steps=np.array([f['step'] for f in frames])
    r=np.array([f['friendly_states'] for f in frames]);b=np.array([f['adversarial_states'] for f in frames])
    fig=plt.figure(figsize=(7,5));ax=fig.add_subplot(projection='3d')
    flat,plane=plt.subplots(figsize=(6,4.5))
    for side,states,style,color in (('Red',r,'-','tab:red'),('Blue',b,'--','tab:blue')):
        for i in range(states.shape[1]):
            live=np.array([f['friendly_alive' if side=='Red' else 'adversarial_alive'][i] for f in frames])
            # Include the physical death endpoint, exclude later frozen slots.
            stop=np.flatnonzero(~live);end=int(stop[0])+1 if len(stop) else len(states)
            p=states[:end,i,:3]
            ax.plot(*p.T,style,color=color,alpha=.7,label=side if i==0 else None)
            plane.plot(p[:,0],p[:,1],style,color=color,alpha=.7,label=side if i==0 else None)
            ax.text(*p[0],f'{side[0]}{i}',fontsize=7)
    for event in data['events']:
        victim=r if event['attacker_side']=='Blue' else b
        p=victim[event['step'],event['target_id'],:3]
        ax.scatter(*p,marker='x',color='black');plane.scatter(p[0],p[1],marker='x',color='black')
        plane.annotate(str(event['step']),p[:2],fontsize=7)
    title=f"{data['scenario']} — local checkpoint, trained {data['completed_training_episodes']} episodes"
    ax.set(xlabel='x (m)',ylabel='y (m)',zlabel='z (m)',title=title);ax.legend()
    plane.set(xlabel='x (m)',ylabel='y (m)',title=title);plane.axis('equal');plane.legend()
    save_figure(fig,output/'trajectory_3d');save_figure(flat,output/'trajectory_xy')
    fig,ax=plt.subplots(figsize=(9,3.5));assign=np.array([f['assignments'] for f in frames])
    for i in range(assign.shape[1]):ax.step(steps,assign[:,i],where='post',label=f'Red {i}')
    for event in data['events']:
        ax.axvline(event['step'],color='black',alpha=.3,lw=.8)
    if data['events']:
        caption='Events: '+'; '.join(f"t={e['step']} {e['attacker_side']} {e['attacker_id']} -> {'Blue' if e['attacker_side']=='Red' else 'Red'} {e['target_id']}" for e in data['events'])
        ax.set_title(title+'\n'+'\n'.join(textwrap.wrap(caption,110)),fontsize=9)
    else:ax.set_title(title)
    ax.set(xlabel='Simulation decision step',ylabel='Target ID (-1 = none)');ax.legend(loc='upper right',fontsize=8)
    save_figure(fig,output/'assignment_events')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('checkpoint');p.add_argument('--scenario',choices=(*TABLES,'random_training'),required=True)
    p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=10000000)
    p.add_argument('--video',action='store_true');p.add_argument('--fps',type=int,default=8)
    a=p.parse_args();output=Path(a.output)
    if output.exists() and any(output.iterdir()):raise FileExistsError('use an empty recording directory')
    data=record(a.checkpoint,a.scenario,a.output,seed=a.seed)
    if a.video:
        from tools.aircombat_video import render_video,export_details
        write_json(output/'video_info.json',render_video(data,output/'replay.mp4',fps=a.fps))
        export_details(data,output)


if __name__=='__main__':main()
