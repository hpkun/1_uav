"""Offline MP4 and physical-event report from the existing combat trace format."""
import csv
import json
import math
import subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from env.aircombat.geometry import relative_geometry


def arrays(data):
    f=data['frames']
    return np.array([x['friendly_states'] for x in f]),np.array([x['adversarial_states'] for x in f])


def render_video(data,path,fps=8):
    if not 1<=fps<=60:raise ValueError('fps must be 1..60')
    r,b=arrays(data);frames=data['frames'];dt=data['environment']['dt']
    fig=plt.figure(figsize=(14,7.5));ax=fig.add_subplot(121,projection='3d');xy=fig.add_subplot(122)
    allp=np.concatenate((r.reshape(-1,6),b.reshape(-1,6)))[:,:3]
    low=allp.min(0)-300;high=allp.max(0)+300
    ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),zlim=(low[2],high[2]),xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)')
    ax.set_box_aspect(high-low);ax.view_init(elev=24,azim=-58)
    xy.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),xlabel='X (m)',ylabel='Y (m)',title='Top view: orange = attack eligible, yellow = kill')
    xy.set_aspect('equal',adjustable='box');xy.grid(alpha=.25)
    artists=[]
    for side,states,color,style in (('Red',r,'tab:red','-'),('Blue',b,'tab:blue','--')):
        for i in range(states.shape[1]):
            line,=ax.plot([],[],[],style,color=color,alpha=.75,label=f'{side[0]}{i}')
            dot,=ax.plot([],[],[],'o',color=color,markersize=6)
            flat,=xy.plot([],[],style,color=color,alpha=.7)
            point,=xy.plot([],[],'o',color=color,markersize=6)
            label=xy.text(0,0,f'{side[0]}{i}',fontsize=9)
            artists.append((side,i,states,line,dot,flat,point,label))
    ax.legend(loc='upper left',fontsize=8,ncol=2)
    note=fig.text(.05,.02,'',fontsize=10);temp=[]
    fig.subplots_adjust(left=.02,right=.98,top=.85,bottom=.12,wspace=.12)
    encoders=subprocess.run([FFMpegWriter.bin_path(),'-encoders'],capture_output=True,text=True,check=True).stdout
    codec='libx264' if 'libx264' in encoders else 'libopenh264'
    if codec not in encoders:raise RuntimeError('FFmpeg requires an installed H.264 encoder')
    writer=FFMpegWriter(fps=fps,codec=codec,bitrate=3500,extra_args=['-pix_fmt','yuv420p','-movflags','+faststart'],
        metadata={'title':'APH-MAPPO final 3v3 actual simulator replay'})
    with writer.saving(fig,str(path),dpi=100):
        for k,f in enumerate(frames):
            for obj in temp:obj.remove()
            temp=[]
            for side,i,states,line,dot,flat,point,label in artists:
                key='friendly_alive' if side=='Red' else 'adversarial_alive'
                dead=[j for j in range(k+1) if not frames[j][key][i]]
                end=dead[0] if dead else k
                p=states[:end+1,i,:3];last=p[-1]
                line.set_data_3d(*p.T);flat.set_data(p[:,0],p[:,1]);dot.set_data_3d(*last[:,None]);point.set_data([last[0]],[last[1]])
                alive=f[key][i];dot.set_marker('o' if alive else 'x');point.set_marker('o' if alive else 'x')
                label.set_position((last[0]+45,last[1]+45));label.set_text(f'{side[0]}{i}'+('' if alive else ' X'))
            if k:
                for side,key,aa,bb in (('Red','red_attack_eligible',r,b),('Blue','blue_attack_eligible',b,r)):
                    for i,j in np.argwhere(f.get(key,[])):
                        p,q=aa[k,i,:3],bb[k,j,:3]
                        temp+=ax.plot(*np.array([p,q]).T,color='orange',lw=1,alpha=.6)
                        temp+=xy.plot([p[0],q[0]],[p[1],q[1]],color='orange',lw=1,alpha=.6)
            events=[e for e in data['events'] if e['step']==f['step']]
            for e in events:
                aa,bb=(r,b) if e['attacker_side']=='Red' else (b,r)
                p,q=aa[k,e['attacker_id'],:3],bb[k,e['target_id'],:3]
                temp+=ax.plot(*np.array([p,q]).T,color='gold',lw=3)
                temp+=xy.plot([p[0],q[0]],[p[1],q[1]],color='gold',lw=3)
            fig.suptitle(f"APH-MAPPO FINAL | {data['completed_training_episodes']} training episodes | seed {data['seed']}\n"
                f"step {k}/{len(frames)-1} | simulation {k*dt:.2f}s | Red {sum(f['friendly_alive'])} / Blue {sum(f['adversarial_alive'])} alive | playback {fps*dt:g}x",fontsize=13)
            recent=[e for e in data['events'] if 0<=k-e['step']<=8]
            note.set_text('Events: '+('; '.join(f"step {e['step']}: {e['attacker_side']} {e['attacker_id']} -> {'Blue' if e['attacker_side']=='Red' else 'Red'} {e['target_id']} killed" for e in recent) or 'none')+
                (' | '+data['summary']['outcome'] if k==len(frames)-1 else ''))
            writer.grab_frame()
    plt.close(fig)
    return dict(width=1400,height=750,fps=fps,frames=len(frames),codec=codec,duration_seconds=len(frames)/fps,
        simulated_duration_seconds=(len(frames)-1)*dt,playback_speed=fps*dt)


def geometry(a,b):
    g=relative_geometry(np.array(a)[None,:],np.array(b)[None,:])
    return dict(distance_m=float(g.distance[0,0]),attacker_angle_deg=float(np.degrees(g.alpha_i[0,0])),
        target_los_angle_deg=float(np.degrees(g.alpha_j[0,0])))


def export_details(data,output):
    output=Path(output);frames=data['frames'];dt=data['environment']['dt'];r,b=arrays(data)
    def write_csv(name,rows):
        if not rows:
            (output/name).write_text('No events\n',encoding='utf-8-sig');return
        with (output/name).open('w',encoding='utf-8-sig',newline='') as out:
            writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    aircraft=[];trajectory=[]
    for side,states,key in (('Red',r,'friendly_alive'),('Blue',b,'adversarial_alive')):
        for i in range(states.shape[1]):
            dead=[k for k,f in enumerate(frames) if not f[key][i]];stop=dead[0] if dead else len(frames)-1
            path=states[:stop+1,i]
            row=dict(side=side,id=i,survived=not bool(dead),last_step=stop,last_time_s=stop*dt,
                start_xyz_m=path[0,:3].tolist(),end_xyz_m=path[-1,:3].tolist(),
                path_length_m=float(np.linalg.norm(np.diff(path[:,:3],axis=0),axis=1).sum()),
                displacement_m=float(np.linalg.norm(path[-1,:3]-path[0,:3])),
                speed_min_mps=float(path[:,3].min()),speed_max_mps=float(path[:,3].max()),
                altitude_min_m=float(path[:,2].min()),altitude_max_m=float(path[:,2].max()),
                gamma_min_deg=float(np.degrees(path[:,4].min())),gamma_max_deg=float(np.degrees(path[:,4].max())),
                start_v_gamma_psi=path[0,3:].tolist(),end_v_gamma_psi=path[-1,3:].tolist())
            aircraft.append(row)
            for k,f in enumerate(frames):
                s=states[k,i]
                trajectory.append(dict(step=k,time_s=k*dt,side=side,id=i,alive=f[key][i],
                    x_m=s[0],y_m=s[1],z_m=s[2],speed_mps=s[3],gamma_rad=s[4],psi_rad=s[5],
                    assigned_target=f['assignments'][i] if side=='Red' else ''))
    kills=[]
    for e in data['events']:
        aa,bb=(r,b) if e['attacker_side']=='Red' else (b,r);k=e['step'];a=aa[k,e['attacker_id']];target=bb[k,e['target_id']]
        kills.append(dict(**e,time_s=k*dt,attacker_xyz_m=a[:3].tolist(),target_xyz_m=target[:3].tolist(),
            attacker_v_gamma_psi=a[3:].tolist(),target_v_gamma_psi=target[3:].tolist(),**geometry(a,target)))
    windows=[]
    for side,key,aa,bb in (('Red','red_attack_eligible',r,b),('Blue','blue_attack_eligible',b,r)):
        for i in range(aa.shape[1]):
            for j in range(bb.shape[1]):
                indices=[k for k in range(1,len(frames)) if frames[k][key][i][j]]
                for group in np.split(indices,np.flatnonzero(np.diff(indices)>1)+1):
                    if not len(group):continue
                    start,end=int(group[0]),int(group[-1]);a=aa[start,i];target=bb[start,j]
                    windows.append(dict(side=side,attacker_id=i,target_id=j,start_step=start,end_step=end,
                        start_time_s=start*dt,end_time_s=end*dt,eligible_samples=len(group),eligible_duration_s=len(group)*dt,
                        start_attacker_xyz_m=a[:3].tolist(),start_target_xyz_m=target[:3].tolist(),
                        end_attacker_xyz_m=aa[end,i,:3].tolist(),end_target_xyz_m=bb[end,j,:3].tolist(),**geometry(a,target)))
    windows.sort(key=lambda x:(x['start_step'],x['side'],x['attacker_id'],x['target_id']))
    closest=[]
    for i in range(r.shape[1]):
        for j in range(b.shape[1]):
            samples=[k for k in range(len(frames)) if frames[max(0,k-1)]['friendly_alive'][i] and frames[max(0,k-1)]['adversarial_alive'][j]]
            k=min(samples,key=lambda k:np.linalg.norm(r[k,i,:3]-b[k,j,:3]))
            closest.append(dict(red_id=i,blue_id=j,step=k,time_s=k*dt,red_xyz_m=r[k,i,:3].tolist(),blue_xyz_m=b[k,j,:3].tolist(),distance_m=float(np.linalg.norm(r[k,i,:3]-b[k,j,:3]))))
    report=dict(checkpoint=data['checkpoint'],checkpoint_sha256=data['checkpoint_sha256'],
        training_episodes=data['completed_training_episodes'],updates=data['updates'],sampled_steps=data['sampled_steps'],
        environment_version=data['environment_version'],scenario=data['scenario'],seed=data['seed'],policy=data['policy'],
        dt=dt,simulation_duration_s=(len(frames)-1)*dt,summary=data['summary'],aircraft=aircraft,kills=kills,
        attack_windows=windows,closest_approaches=closest,assignment_changes=data['assignment_changes'],
        semantics='No trigger/missile model. Attack windows are actual post-motion eligibility samples using pre-death alive masks; kills require dwell. IDs zero-based; xyz metres; gamma/psi raw unfolded/wrapped coordinates.')
    (output/'battle_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    write_csv('trajectory.csv',trajectory);write_csv('kills.csv',kills);write_csv('attack_windows.csv',windows);write_csv('aircraft.csv',aircraft);write_csv('closest_approaches.csv',closest)
    return report


def verify_trace(data):
    """Replay recorded actions in an independent environment, without a policy."""
    import hashlib
    from algorithm.common.runtime import require_cuda
    from env.aircombat import AirCombatEnv,AirCombatConfig
    from algorithm.common.aircombat_vector import CombatEpisodeAccumulator
    require_cuda('cuda')
    env=AirCombatEnv(AirCombatConfig(**data['environment']))
    obs,info=env.reset(data['seed'],{'scenario':data['scenario']})
    tracker=CombatEpisodeAccumulator();tracker.reset(info);events=[]
    for f in data['frames'][1:]:
        obs,reward,term,trunc,info=env.step(f['red_actions']);tracker.add(info,reward)
        for key in ('friendly_states','adversarial_states','friendly_alive','adversarial_alive','assignments','red_controls','blue_controls'):
            np.testing.assert_array_equal(f[key],info[key])
        np.testing.assert_array_equal(f['reward'],reward)
        for key,value in f['reward_components'].items():np.testing.assert_array_equal(value,info['reward_components'][key])
        events.extend(info['events'])
    assert events==data['events'] and tracker.summary(info)==data['summary'] and (term or trunc)
    digest=hashlib.sha256(Path(data['checkpoint']).read_bytes()).hexdigest()
    assert digest==data['checkpoint_sha256']
    return dict(verified_steps=len(data['frames'])-1,states_controls_masks_rewards_assignments_exact=True,
                events_and_summary_exact=True,model_sha256_unchanged=digest)


def write_report(report,output):
    """Human-readable companion; all numbers originate in archived trace data."""
    output=Path(output);probe=json.loads((output/'video_probe.json').read_text())
    video=probe['streams'][0];s=report['summary'];dt=report['dt']
    from fractions import Fraction
    fps=float(Fraction(video['avg_frame_rate']));speed=fps*dt
    xyz=lambda p:'('+', '.join(f'{v:.2f}' for v in p)+')'
    text=['# Final模型3v3对局录像报告','',
        f"模型：`{report['checkpoint']}`。实际训练{report['training_episodes']}回合、{report['updates']}次PPO更新、{report['sampled_steps']}个joint steps。",
        f"环境：{report['environment_version']}；场景：{report['scenario']}；独立种子：{report['seed']}；Red策略：{report['policy']}。Blue沿用aggregate-sum、p=0.8。",
        f"模型SHA256：`{report['checkpoint_sha256']}`。只读加载，没有重新训练或修改模型。",'',
        '## 录像与结果','',
        f"`replay.mp4`：{video['width']}×{video['height']}，H.264，{video['avg_frame_rate']} fps，{video['nb_frames']}帧，{float(video['duration']):.3f}秒，{int(probe['format']['size']):,}字节，无音轨。每个仿真采样帧对应一个视频帧；包含初始帧；无虚构插值。",
        f"仿真{report['simulation_duration_s']:.2f}秒、{s['episode_length']}步，dt={dt}秒。视频{speed:g}倍速，仿真时刻t对应视频约t/{speed:g}秒；影片末帧额外占{1/fps:.3f}秒。",
        f"本局结局：{s['outcome']}；Red存活{s['red_survivors']}架、Blue存活{sum(a['side']=='Blue' for a in report['aircraft'])-s['blue_shot_down']}架；Red击落{s['blue_shot_down']}架；数值异常{s['numerical_failures']}次。Red获胜标志为{s['win']}。",
        f"平均每架Red累计Return={s['episode_return']:.8f}。五项奖励累计（按Red槽平均）："+', '.join(f'{k}={v:.8f}' for k,v in s['reward_components_total'].items())+'。',
        '画面左侧为真实比例三维轨迹，右侧为俯视图。Red实线、Blue虚线；圆点为活机，X为阵亡终点；橙线为当前采样攻击条件成立，黄线为击落事件。阵亡后的轨迹不继续延长。','',
        '## 双方逐机轨迹','',
        '坐标为(x,y,z)，单位m；ID从0开始。阵亡飞机的终点是击落时刻位置，其余为仿真最后时刻位置。航程为逐步三维折线长度，排除阵亡后冻结槽。','',
        '| 飞机 | 起点(m) | 终点(m) | 航程(km) | 状态/最后时刻 |',
        '|---|---|---|---:|---|']
    for a in report['aircraft']:
        text.append(f"| {a['side'][0]}{a['id']} | {xyz(a['start_xyz_m'])} | {xyz(a['end_xyz_m'])} | {a['path_length_m']/1000:.3f} | {'存活' if a['survived'] else '阵亡'}，{a['last_time_s']:.2f}s |")
    text+=['','| 飞机 | 速率范围(m/s) | 高度范围(m) | 展开gamma范围(度) |','|---|---:|---:|---:|']
    for a in report['aircraft']:
        text.append(f"| {a['side'][0]}{a['id']} | {a['speed_min_mps']:.2f}–{a['speed_max_mps']:.2f} | {a['altitude_min_m']:.2f}–{a['altitude_max_m']:.2f} | {a['gamma_min_deg']:.2f}–{a['gamma_max_deg']:.2f} |")
    text+=['','gamma为环境原始展开角度，可越过±90°、180°；不能直接按主值航迹角解释。实际速度方向可由v、gamma、psi重建。全过程每0.25秒的原始六维状态、存活和Red目标见`trajectory.csv`/`trace.json`。','',
        '## 开火条件与击落位置','',
        '本环境没有扳机动作、弹药量或导弹飞行。这里的“开火”指采样时刻攻击条件成立：距离≤2000m、攻击方夹角≤45°、目标相对同一LOS夹角<90°，且双方该步运动前存活。连续两个采样条件成立才击落；资格窗口和击落不是两种独立的真实弹药事件。',
        f"本局共{len(report['attack_windows'])}段攻击资格窗口，Red {sum(w['side']=='Red' for w in report['attack_windows'])}段、Blue {sum(w['side']=='Blue' for w in report['attack_windows'])}段。以下窗口起止为采样时刻，2个样本×dt的记账时长不代表已精确确定步内首次满足条件的连续时间。"]
    for w in report['attack_windows']:
        target='Blue' if w['side']=='Red' else 'Red'
        text+=['',f"### {w['side']} {w['attacker_id']} → {target} {w['target_id']}",
            f"资格窗口：第{w['start_step']}–{w['end_step']}步，仿真{w['start_time_s']:.2f}–{w['end_time_s']:.2f}s；首次满足时距离{w['distance_m']:.2f}m，攻击夹角{w['attacker_angle_deg']:.2f}°，目标LOS夹角{w['target_los_angle_deg']:.2f}°。",
            f"窗口开始双方位置：攻击方{xyz(w['start_attacker_xyz_m'])}，目标{xyz(w['start_target_xyz_m'])}。"]
        event=next((e for e in report['kills'] if e['step']==w['end_step'] and e['attacker_side']==w['side'] and e['attacker_id']==w['attacker_id'] and e['target_id']==w['target_id']),None)
        if event:
            text +=[f"击落：第{event['step']}步，仿真{event['time_s']:.2f}s，视频约{event['time_s']/speed:.2f}s；攻击方{xyz(event['attacker_xyz_m'])}，目标{xyz(event['target_xyz_m'])}。",
                f"击落距离{event['distance_m']:.2f}m，攻击夹角{event['attacker_angle_deg']:.2f}°，目标LOS夹角{event['target_los_angle_deg']:.2f}°。攻击方速率{event['attacker_v_gamma_psi'][0]:.2f}m/s，目标速率{event['target_v_gamma_psi'][0]:.2f}m/s。"]
    text+=['','## 最近交会与目标分配','','最近距离仅在双方还参与战斗的采样端点统计，包含击落步，排除阵亡后冻结位置造成的虚假接近。距离近不等于满足射击锥和目标方向条件。','',
        '| Red / Blue | 最小距离(m) | 步数 | 仿真时间(s) |','|---|---:|---:|---:|']
    for a in report['closest_approaches']:text.append(f"| R{a['red_id']} / B{a['blue_id']} | {a['distance_m']:.2f} | {a['step']} | {a['time_s']:.2f} |")
    trace=json.loads((output/'trace.json').read_text());initial=trace['frames'][0]['assignments']
    text+=['',f"初始Red分配为{initial}；共{len(report['assignment_changes'])}条单机目标变化记录（同一步双机互换按两条计）。APH分配不是开火开关，武器检查仍针对所有存活敌机。",'',
        '| 步数 | 仿真时间(s) | Red | 目标变化 |','|---|---:|---|---|']
    for c in report['assignment_changes']:text.append(f"| {c['step']} | {c['step']*dt:.2f} | R{c['friendly_id']} | {c['previous_target']} → {c['target']} |")
    text+=['','-1表示无分配；阵亡时变为-1不是另一次攻击事件。','',
        '## 数值与可复核性','',json.dumps(s.get('numerical_diagnostics',{}),ensure_ascii=False),
        '极区进入/成功延续不是数值失败、开火或击落。`verification.json`独立按记录动作重放，核对双方状态、控制、存活、奖励分量、目标分配及事件精确一致，并复核模型哈希。',
        '本录像选择固定独立种子，未按胜负挑选。它只是单局样本，不能把超时结果当成模型总体胜率。原训练随附200局独立评估记录另在训练目录`final_evaluation.json`；本次没有重新训练、重新选best或大规模评估。','',
        '文件索引：`replay.mp4`、`trajectory_3d.png`、`trajectory_xy.png`、`assignment_events.png`；原始`trace.json`；逐步`trajectory.csv`；`kills.csv`、`attack_windows.csv`、`aircraft.csv`、`closest_approaches.csv`；结构化`battle_report.json`；编码核验`video_probe.json`。']
    (output/'录像报告.md').write_text('\n'.join(text)+'\n',encoding='utf-8')


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trace',required=True);p.add_argument('--output',required=True)
    p.add_argument('--fps',type=int,default=8);p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    data=json.loads(Path(a.trace).read_text(encoding='utf-8'))
    if a.verify_only:
        (output/'verification.json').write_text(json.dumps(verify_trace(data),indent=2),encoding='utf-8');return
    if (output/'replay.mp4').exists():raise FileExistsError('video already exists')
    info=render_video(data,output/'replay.mp4',a.fps)
    (output/'video_info.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    export_details(data,output)


if __name__=='__main__':main()
