"""Offline reductions and scientific plots of audit_actor_flight real records."""
import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from algorithm.common.serialization import write_json
from env.dynamics import velocity_vector, wrap_angle
from env.aircombat.framed_dynamics import initial_frame
from tools.diagnose_headon import longest_run


def describe(values):
    a = np.asarray(values,float)
    if not a.size: return dict(n=0)
    return dict(n=len(a),min=np.min(a,axis=0).tolist(),mean=np.mean(a,axis=0).tolist(),
                std=np.std(a,axis=0).tolist(),p05=np.quantile(a,.05,axis=0).tolist(),
                p50=np.quantile(a,.5,axis=0).tolist(),p95=np.quantile(a,.95,axis=0).tolist(),max=np.max(a,axis=0).tolist())


def read_rows(path):
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def continuous_run(rows, predicate, dt=.25):
    current = {}; best = dict(steps=0,duration_s=0.)
    for r in rows:
        key=(r['seed'],r['id'])
        previous=current.get(key,(-2,0))
        length=(previous[1]+1 if r['step']==previous[0]+1 else 1) if predicate(r) else 0
        current[key]=(r['step'],length)
        if length>best['steps']:
            best=dict(steps=length,duration_s=length*dt,seed=r['seed'],id=r['id'],start_step=r['step']-length+1,end_step=r['step'])
    return best


def reduce_group(episodes):
    flights=[]; sides={'Red':[],'Blue':[]}; postures=[]; probes=[]; entries=[]; ranges=[]
    summary=[]; saturation=[]; critic=[]; actual_actions=[]; masked=0; stream_count=0
    beta_alpha=[];beta_beta=[];beta_std=[];mapping_error=0.;first_chain=None
    for ep in episodes:
        rows=read_rows(ep['trace']);previous_range=None;previous_controls={};seen_polar=set()
        for row in rows:
            if 'failure' in row: continue
            stream_count+=1
            rs,bs=np.array(row['post_red']),np.array(row['post_blue'])
            pre=np.array(row['pre_red']); preblue=np.array(row['pre_blue'])
            live=np.array(row['pre_red_alive'],bool); bl=np.array(row['pre_blue_alive'],bool)
            mask=live[:,None]&bl[None,:]
            dist=np.array(row['red_distance']);ai=np.array(row['red_alpha_i_deg']);aj=np.array(row['red_alpha_j_deg'])
            in_range=mask&(dist>1e-9)&(dist<=2000)
            ranges.extend(ai[in_range].tolist())
            if previous_range is None:previous_range=np.zeros_like(in_range)
            entries.extend(ai[in_range&~previous_range].tolist());previous_range=in_range
            bydiag={(d['side'],d['id']):d for d in row['numerical_step']}
            for side,old,new,alive,frames,controls in (
                ('Red',pre,rs,live,np.array(row['pre_red_frames']),np.array(row['red_controls'])),
                ('Blue',preblue,bs,bl,np.array(row['pre_blue_frames']),np.array(row['blue_controls']))):
                for i in np.flatnonzero(alive):
                    before,after=old[i],new[i];v0,v1=velocity_vector(before),velocity_vector(after)
                    h0,h1=np.linalg.norm(v0[:2])/before[3],np.linalg.norm(v1[:2])/after[3]
                    pitch0,pitch1=np.arctan2(v0[2],np.linalg.norm(v0[:2])),np.arctan2(v1[2],np.linalg.norm(v1[:2]))
                    yawchange=float(wrap_angle(np.arctan2(v1[1],v1[0])-np.arctan2(v0[1],v0[0])))
                    mu,zeta,n=controls[i];diag=bydiag[(side,int(i))]
                    nominal=initial_frame(before)
                    gauge=float(np.arctan2(frames[i,:,1]@nominal[:,2],frames[i,:,1]@nominal[:,1]))
                    item=dict(seed=ep['seed'],id=int(i),step=row['step'],mu=float(np.degrees(mu)),zeta=zeta,normal=n,
                        gamma=float(np.degrees(pitch1)),raw_gamma=float(np.degrees(after[4])),speed=float(after[3]),height=float(after[2]),
                        height_change=float(after[2]-ep['initial']['red' if side=='Red' else 'blue'][i][2]),
                        dz=float(after[2]-before[2]),gamma_rate=float(np.degrees(pitch1-pitch0)/.25),
                        yaw_rate=float(np.degrees(yawchange)/.25),velocity_jump=float(np.linalg.norm(v1-v0)),
                        horizontal_ratio=h1,gauge_deg=float(np.degrees(gauge)),
                        pole=diag['continuation_substeps']>0,endpoint_pole=h1<.1,
                        **{k:v for k,v in diag.items() if k not in ('side','id','control')})
                    key=(side,int(i))
                    item['reference_phase']='polar_step' if item['pole'] else 'after_pole' if key in seen_polar else 'before_first_pole'
                    if item['pole']:seen_polar.add(key)
                    previous=previous_controls.get((side,int(i)))
                    item['action_change']=np.abs(controls[i]-previous).tolist() if previous is not None else None
                    previous_controls[(side,int(i))]=controls[i]
                    item.update(turn=None,target=None,bearing_deg=None,target_elevation_deg=None,c_self=None,c_rel=None)
                    if side=='Red':
                        j=row['pre_assignments'][i];item['target']=j
                        if j>=0 and bl[j]:
                            los=preblue[j,:3]-before[:3];d0=np.linalg.norm(los);d1=np.linalg.norm(bs[j,:3]-after[:3])
                            item.update(distance=float(d1),alpha_i_deg=float(ai[i,j]),alpha_j_deg=float(aj[i,j]),
                                c_self=float(np.dot(after[:3]-before[:3],los/d0)/.25) if d0>1e-9 else 0.,c_rel=float((d0-d1)/.25),
                                target_elevation_deg=float(np.degrees(np.arctan2(los[2],np.linalg.norm(los[:2])))))
                            if min(h0,h1)>=.1 and np.linalg.norm(los[:2])>1e-9:
                                error=np.arctan2(v0[0]*los[1]-v0[1]*los[0],np.dot(v0[:2],los[:2]))
                                remaining=float(wrap_angle(error-yawchange))
                                item['bearing_deg']=float(np.degrees(error))
                                improvement=abs(error)-abs(remaining)
                                item['turn']='toward' if improvement>1e-9 else 'away' if improvement< -1e-9 else 'neutral'
                            # d gamma/dt for canonical physical elevation is frame-dependent.
                            u,a,b=frames[i].T
                            projection=np.cos(mu)*a[2]+np.sin(mu)*b[2]
                            item['vertical_acceleration_margin']=float(n*projection/h0-h0) if h0>1e-9 else None
                            item['aligned_chart']=abs(gauge)<np.radians(.1) and before[4]>-np.pi/2 and before[4]<np.pi/2 and h0>=.2
                            item['bank_sign_turn_agreement']=(np.sign(np.sin(mu))==np.sign(yawchange)) if item['aligned_chart'] and abs(mu)>1e-5 and n>1e-5 and abs(yawchange)>1e-8 else None
                            item['phase']='far' if d1>2000 else 'in_range'
                        flights.append(item)
                    sides[side].append(item)
            postures.extend(row['posture_samples'])
            for p in row.get('motion_probes',[]):probes.append(dict(seed=ep['seed'],step=row['step'],**p))
            if 'actor_tanh_saturation_per_slot' in row:
                saturation.extend(np.array(row['actor_tanh_saturation_per_slot'])[live].tolist())
                critic.append(row['critic_tanh_saturation'])
                actions=np.array(row['actor']['actual_actions']);actual_actions.extend(actions[live].tolist())
                beta_alpha.extend(np.array(row['actor']['beta_alpha'])[live].tolist());beta_beta.extend(np.array(row['actor']['beta_beta'])[live].tolist());beta_std.extend(np.array(row['actor']['beta_std'])[live].tolist())
                mapped=actions[live]*[2*np.pi,3,8]+[-np.pi,-1,0]
                mapping_error=max(mapping_error,float(np.max(abs(mapped-np.array(row['red_controls'])[live]))))
                if first_chain is None:
                    first_chain=dict(seed=row['seed'],step=row['step'],raw_observation=row['raw_observation'],
                        actor=row['actor'],physical_controls=row['red_controls'],current_state=row['pre_red'],next_state=row['post_red'],
                        targets_before=row['pre_assignments'],targets_after=row['post_assignments'])
                masked+=int((~live).sum())
        summary.append({k:ep[k] for k in ('seed','steps','outcome','red_survivors','blue_survivors','win','first_steps','counts','assignment_switches','events')})
    def side_summary(samples):
        return dict(active_uav_steps=len(samples),states={key:describe([s[key] for s in samples]) for key in
            ('speed','gamma','raw_gamma','gamma_rate','yaw_rate','height','height_change','velocity_jump','gauge_deg')},
            gamma_abs_exceeds_fraction={str(limit):float(np.mean([abs(s['gamma'])>limit for s in samples])) for limit in (45,60,75,84.26,89)},
            raw_gamma_abs_exceeds_fraction={str(limit):float(np.mean([abs(s['raw_gamma'])>limit for s in samples])) for limit in (45,60,75,84.26,89,90)},
            pole_episode_fraction=len({s['seed'] for s in samples if s['pole']})/len(episodes),
            pole_step_fraction=float(np.mean([s['pole'] for s in samples])),
            pole_entries=sum(s['pole_entries'] for s in samples),continuation_substeps=sum(s['continuation_substeps'] for s in samples),
            speed_clip_count=sum(s['speed_clips'] for s in samples),speed_clip_fraction=float(np.mean([s['speed_clips']>0 for s in samples])),
            min_horizontal_ratio=min(s['min_horizontal_ratio'] for s in samples),
            max_reference_axis_error_rad=max(s.get('max_reference_axis_error',0) for s in samples),
            max_frame_spin_rate_rad_s=max(s.get('max_frame_spin_rate',0) for s in samples),
            max_frame_orthogonality_error=max(s['frame_orthogonality_error'] for s in samples),
            gauge_abs_deg=describe([abs(s['gauge_deg']) for s in samples]),
            gauge_abs_over_1deg_fraction=float(np.mean([abs(s['gauge_deg'])>1 for s in samples])),
            gauge_abs_over_90deg_fraction=float(np.mean([abs(s['gauge_deg'])>90 for s in samples])),
            outside_cap_gauge_over_90_count=sum(s['min_horizontal_ratio']>=.1 and abs(s['gauge_deg'])>90 for s in samples),
            unclipped_velocity_increment_bound_ratio=describe([s['velocity_jump']/(.25*9.81*(abs(s['zeta'])+s['normal']+1)) for s in samples if not s['speed_clips']]),
            reference_phases={phase:dict(n=len(segment),
                gauge_abs_deg=describe([abs(s['gauge_deg']) for s in segment]),
                physical_gamma_deg=describe([s['gamma'] for s in segment]),
                gamma_rate_deg_s=describe([s['gamma_rate'] for s in segment]),
                controls_mu_deg_zeta_n=describe([[s['mu'],s['zeta'],s['normal']] for s in segment]))
                for phase in ('before_first_pole','polar_step','after_pole')
                for segment in [[s for s in samples if s['reference_phase']==phase]]},
            longest_polar_run=continuous_run(samples,lambda s:s['pole']),
            longest_climb=continuous_run(samples,lambda s:s['dz']>.001),longest_dive=continuous_run(samples,lambda s:s['dz']<-.001))
    def flight_summary(samples):
        controls=np.array([[s['mu'],s['zeta'],s['normal']] for s in samples])
        turns={k:sum(s['turn']==k for s in samples) for k in ('toward','away','neutral')}
        classified=turns['toward']+turns['away'];vm=[s['vertical_acceleration_margin'] for s in samples if s.get('vertical_acceleration_margin') is not None]
        target=[s for s in samples if s['c_rel'] is not None]
        norm=(controls*np.array([np.pi/180,1,1])+[np.pi,1,0])/[2*np.pi,3,8]
        return dict(n=len(samples),controls_mu_deg_zeta_n=describe(controls),
            bound_fraction=np.mean((norm<=.01)|(norm>=.99),axis=0).tolist(),
            bank_over_60_fraction=float(np.mean(abs(controls[:,0])>60)),bank_over_90_fraction=float(np.mean(abs(controls[:,0])>90)),
            near_max_n_small_bank_fraction=float(np.mean((controls[:,2]>=7.6)&(abs(controls[:,0])<=10))),
            high_normal_fraction=float(np.mean(controls[:,2]>6)),
            climbing_fraction=float(np.mean([s['dz']>.001 for s in samples])),
            diving_fraction=float(np.mean([s['dz']<-.001 for s in samples])),
            normal_above_local_equilibrium_fraction=float(np.mean(np.array(vm)>0)) if vm else None,
            controls_step_absolute_change=describe([s['action_change'] for s in samples if s['action_change'] is not None]),
            turn_counts=turns,turn_toward_fraction=turns['toward']/classified if classified else None,
            bank_sign_agreement_fraction=float(np.mean([s['bank_sign_turn_agreement'] for s in samples if s.get('bank_sign_turn_agreement') is not None])) if any(s.get('bank_sign_turn_agreement') is not None for s in samples) else None,
            receding_fraction=float(np.mean([s['c_rel']< -1e-6 for s in target])) if target else None,
            self_receding_fraction=float(np.mean([s['c_self']< -1e-6 for s in target])) if target else None,
            longest_high_normal=continuous_run(samples,lambda s:s['normal']>6),
            longest_bank_over_60=continuous_run(samples,lambda s:abs(s['mu'])>60),
            bearing_groups={name:describe([[s['mu'],s['zeta'],s['normal'],s['yaw_rate']] for s in samples
                if s['bearing_deg'] is not None and (s['bearing_deg']>1 if name=='left' else s['bearing_deg']< -1)]) for name in ('left','right')})
    far=[p for p in postures if p['band']=='far']
    def posture_stats(samples):
        p=np.array([s['posture'] for s in samples]);positive=p[p>0]
        return dict(n=len(p),mean=float(p.mean()) if len(p) else None,sum=float(p.sum()),positive_n=len(positive),positive_sum=float(positive.sum()))
    positive=sum(max(p['posture'],0) for p in far)
    receding=[p for p in far if p['c_rel']< -1e-6];self_receding=[p for p in receding if p['c_self']< -1e-6]
    farstats=dict(all=posture_stats(far),approaching=posture_stats([p for p in far if p['c_rel']>1e-6]),
        receding=posture_stats(receding),self_receding=posture_stats(self_receding),
        self_closing_relative_receding=posture_stats([p for p in receding if p['c_self']>1e-6]),
        receding_positive_share= sum(max(p['posture'],0) for p in receding)/positive if positive else None,
        longest_receding_positive=longest_run(postures,lambda p:p['band']=='far' and p['c_rel']< -1e-6 and p['posture']>0),
        longest_self_receding_positive=longest_run(postures,lambda p:p['band']=='far' and p['c_rel']< -1e-6 and p['c_self']< -1e-6 and p['posture']>0))
    aligned=[p for p in probes if p['safe_angle_step'] and abs(p['gauge_angle_rad'])<1e-6 and 'angle_rk4_difference' in p]
    nonaligned=[p for p in probes if p['safe_angle_step'] and abs(p['gauge_angle_rad'])>=np.radians(1) and 'angle_rk4_difference' in p]
    def probe_summary(ps):
        return dict(n=len(ps),position_error_m=describe([p['angle_rk4_position_error_m'] for p in ps if 'angle_rk4_position_error_m' in p]),
            state_error=describe([np.abs(p['angle_rk4_difference']) for p in ps if 'angle_rk4_difference' in p]),
            same_mu_rhs_difference=describe([np.abs(p['same_mu_rhs_difference']) for p in ps if 'same_mu_rhs_difference' in p]),
            chart_reset_velocity_difference_m_s=describe([p['counterfactual_chart_reset_velocity_difference_m_s'] for p in ps]))
    cf=sorted(probes,key=lambda p:p['counterfactual_chart_reset_velocity_difference_m_s'],reverse=True)
    return dict(episodes=len(episodes),wins=sum(ep['win'] for ep in episodes),win_rate=float(np.mean([ep['win'] for ep in episodes])),
        mean_steps=float(np.mean([ep['steps'] for ep in episodes])),mean_red_survivors=float(np.mean([ep['red_survivors'] for ep in episodes])),
        mean_blue_kills=float(np.mean([3-ep['blue_survivors'] for ep in episodes])),
        timeouts=sum(ep['outcome']=='timeout' for ep in episodes),mutual_eliminations=sum(ep['outcome']=='mutual_elimination' for ep in episodes),
        failures=sum(ep['failure'] is not None for ep in episodes),trace_steps=stream_count,
        flight=flight_summary(flights),per_red={str(i):flight_summary([s for s in flights if s['id']==i]) for i in range(3)},
        phases={name:flight_summary([s for s in flights if s.get('phase')==name]) for name in ('far','in_range') if any(s.get('phase')==name for s in flights)},
        flight_first40=flight_summary([s for s in flights if s['step']<=40]),sides={side:side_summary(ss) for side,ss in sides.items()},
        in_range_alpha_deg=describe(ranges),range_entry_alpha_deg=describe(entries),far_posture=farstats,
        tanh_saturation=dict(actor_fraction=float(np.mean(saturation)) if saturation else None,critic_fraction=float(np.mean(critic)) if critic else None),
        policy_chain=dict(actual_normalized_actions=describe(actual_actions),beta_alpha=describe(beta_alpha),beta_beta=describe(beta_beta),beta_std=describe(beta_std),
            physical_mapping_max_error=mapping_error,first_transition=first_chain,
            action_components_at_trainer_clamp_bound=int(np.sum((np.asarray(actual_actions)<=1e-6)|(np.asarray(actual_actions)>=1-1e-6))) if actual_actions else None,
            clamp_note='Strictly interior executed values prove the [epsilon,1-epsilon] clamp was inactive; pre-clamp stochastic draws were not separately recorded.'),
        motion_probes=dict(total=len(probes),aligned_normal=probe_summary(aligned),misaligned_outside_cap=probe_summary(nonaligned),
            all=probe_summary(probes),largest_chart_reset_counterfactuals=cf[:5]),
        weapon={side:{key:sum(e['counts'][side][key] for e in episodes) for key in episodes[0]['counts'][side]} for side in ('Red','Blue')},
        first_times_s={side:{key:describe([e['first_steps'][side][key]*.25 for e in episodes if e['first_steps'][side][key] is not None])
                         for key in ('range','own_cone','eligible','kill')} for side in ('Red','Blue')},
        episode_details=summary)


def save_figure(fig,path):
    fig.savefig(path,dpi=155,bbox_inches='tight');plt.close(fig)


def plot_episode(ep,folder,source):
    rows=[r for r in read_rows(ep['trace']) if 'step' in r]
    r=np.array([rows[0]['pre_red']]+[x['post_red'] for x in rows]);b=np.array([rows[0]['pre_blue']]+[x['post_blue'] for x in rows])
    rm=np.array([rows[0]['pre_red_alive']]+[x['post_red_alive'] for x in rows]);bm=np.array([rows[0]['pre_blue_alive']]+[x['post_blue_alive'] for x in rows])
    t=np.arange(len(rows)+1)*.25;tt=t[1:];model=ep['model'];scene=ep['scenario'];seed=ep['seed']
    stem=f'{model}_{scene}_{seed}'
    title=f'{model} | {scene} | seed={seed} | dt=0.25 s\n{source}'
    colors=['#be2b2b','#e97922','#e8b027','#2455b4','#378ab8','#683ab5']
    fig=plt.figure(figsize=(12,5));ax=fig.add_subplot(121,projection='3d');xy=fig.add_subplot(122)
    for k,(states,mask,side) in enumerate(((r,rm,'R'),(b,bm,'B'))):
        for i in range(3):
            # Include the death endpoint and exclude frozen post-death coordinates.
            end=np.flatnonzero(~mask[:,i]);last=int(end[0]) if len(end) else len(states)-1
            s=states[:last+1,i];color=colors[3*k+i]
            ax.plot(s[:,0],s[:,1],s[:,2],color=color,label=f'{side}{i}')
            ax.scatter(*s[0,:3],color=color,marker='o',s=18);ax.scatter(*s[-1,:3],color=color,marker='x',s=30)
            xy.plot(s[:,0],s[:,1],color=color,label=f'{side}{i}');xy.scatter(*s[0,:2],color=color,marker='o',s=18);xy.scatter(*s[-1,:2],color=color,marker='x',s=30)
    for e in ep['events']:
        point=e['target_position'];xy.scatter(point[0],point[1],marker='*',color='black',s=65)
        xy.annotate(f'{e["step"]*.25:.2f}s',point[:2],fontsize=7)
    ax.set(xlabel='x (m)',ylabel='y (m)',zlabel='z (m)');xy.set(xlabel='x (m)',ylabel='y (m)',title='o=start, x=end/death, *=kill location')
    xy.axis('equal');xy.grid(alpha=.25);ax.legend(fontsize=8);fig.suptitle(title,fontsize=9);fig.tight_layout()
    save_figure(fig,folder/f'{stem}_trajectory.png')
    controls=np.array([x['red_controls'] for x in rows]);live=np.array([x['pre_red_alive'] for x in rows])
    fig,axs=plt.subplots(4,2,figsize=(12,11),sharex=True)
    units=[('bank mu (deg)',180/np.pi),('tangential load zeta',1),('normal load n',1)]
    for i in range(3):
        for j,(label,scale) in enumerate(units):
            axs[j,0].plot(tt,np.where(live[:,i],controls[:,i,j]*scale,np.nan),color=colors[i],label=f'R{i}')
        velocity=velocity_vector(r[:,i]);gamma=np.degrees(np.arctan2(velocity[:,2],np.linalg.norm(velocity[:,:2],axis=1)))
        for j,values in enumerate((r[:,i,3],gamma,r[:,i,2])):
            active=np.r_[True,live[:,i]];axs[j,1].plot(t,np.where(active,values,np.nan),color=colors[i],label=f'R{i}')
        yaw=np.degrees(np.unwrap(np.arctan2(velocity[:,1],velocity[:,0])))
        axs[3,0].plot(t,np.where(active,yaw,np.nan),color=colors[i],label=f'R{i}')
        gauge=[]
        for row in rows:
            frame=np.array(row['pre_red_frames'][i]);nominal=initial_frame(row['pre_red'][i])
            gauge.append(np.degrees(np.arctan2(frame[:,1]@nominal[:,2],frame[:,1]@nominal[:,1])))
        axs[3,1].plot(tt,np.where(live[:,i],gauge,np.nan),color=colors[i],label=f'R{i}')
    blue_live=np.array([x['pre_blue_alive'] for x in rows])
    for i in range(3):
        velocity=velocity_vector(b[:,i]);pitch=np.degrees(np.arctan2(velocity[:,2],np.linalg.norm(velocity[:,:2],axis=1)))
        active=np.r_[True,blue_live[:,i]]
        for j,values in enumerate((b[:,i,3],pitch,b[:,i,2])):
            axs[j,1].plot(t,np.where(active,values,np.nan),color=colors[3+i],ls='--',lw=1,alpha=.7,label=f'B{i}')
        yaw=np.degrees(np.unwrap(np.arctan2(velocity[:,1],velocity[:,0])))
        axs[3,0].plot(t,np.where(active,yaw,np.nan),color=colors[3+i],ls='--',lw=1,alpha=.7,label=f'B{i}')
    for j,(label,_) in enumerate(units):axs[j,0].set_ylabel(label)
    for j,label in enumerate(('speed (m/s)','physical gamma (deg)','altitude (m)')):axs[j,1].set_ylabel(label)
    axs[3,0].set_ylabel('physical heading unwrapped (deg)');axs[3,1].set_ylabel('local-axis gauge (deg)')
    for ax in axs.flat:ax.grid(alpha=.25)
    for row in rows:
        if any(x['side']=='Red' and x['continuation_substeps'] for x in row['numerical_step']):
            for ax in axs.flat:ax.axvspan(row['time_s']-.25,row['time_s'],color='purple',alpha=.025)
    axs[0,0].legend();axs[0,1].legend(ncol=2,fontsize=7);axs[-1,0].set_xlabel('time (s)');axs[-1,1].set_xlabel('time (s)')
    fig.suptitle(title+'\nPurple shading: actual polar-continuation steps; gamma is physical elevation',fontsize=9);fig.tight_layout()
    save_figure(fig,folder/f'{stem}_controls_states.png')
    fig,axs=plt.subplots(3,1,figsize=(11,8),sharex=True)
    for i in range(3):
        ds=[];angles=[];aspect=[];closure=[];self_closure=[]
        for row in rows:
            j=row['pre_assignments'][i]
            if j<0 or not row['pre_red_alive'][i]:
                for values in (ds,angles,aspect,closure,self_closure):values.append(np.nan)
                continue
            d1=row['red_distance'][i][j];d0=np.linalg.norm(np.array(row['pre_blue'][j][:3])-row['pre_red'][i][:3])
            ds.append(d1);angles.append(row['red_alpha_i_deg'][i][j]);aspect.append(row['red_alpha_j_deg'][i][j]);closure.append((d0-d1)/.25)
            los=np.array(row['pre_blue'][j][:3])-row['pre_red'][i][:3]
            self_closure.append(np.dot(np.array(row['post_red'][i][:3])-row['pre_red'][i][:3],los/d0)/.25 if d0>1e-9 else 0.)
        for ax,values in zip(axs,(ds,angles,closure)):ax.plot(tt,values,color=colors[i],label=f'R{i}')
        axs[1].plot(tt,aspect,color=colors[i],ls=':',alpha=.65,label=f'R{i} target aspect')
        axs[2].plot(tt,self_closure,color=colors[i],ls=':',alpha=.65,label=f'R{i} self closure')
    axs[0].axhline(2000,color='black',ls='--',lw=1);axs[1].axhline(45,color='black',ls='--',lw=1);axs[2].axhline(0,color='black',lw=1)
    axs[1].axhline(90,color='gray',ls=':',lw=1)
    for ax in axs:ax.grid(alpha=.25);ax.legend(fontsize=8)
    axs[0].set_ylabel('assigned target distance (m)');axs[1].set_ylabel('own alpha_i (deg)');axs[2].set_ylabel('relative closure (m/s)');axs[2].set_xlabel('time (s)')
    fig.suptitle(title+'\nUses PRE-action assigned target, POST-motion geometry; target switches may jump',fontsize=9);fig.tight_layout()
    save_figure(fig,folder/f'{stem}_relative.png')


def comparison_plot(groups,folder):
    models=list(dict.fromkeys(g['model'] for g in groups));fig,axs=plt.subplots(2,3,figsize=(14,8))
    fields=[('win_rate','Win rate (%)',100),('mean_blue_kills','Blue kills / episode',1),('mean_red_survivors','Red survivors / episode',1),
            ('normal','Mean normal load n',1),('gamma','Mean physical gamma (deg)',1),('pole','Red polar UAV-steps (%)',100)]
    for ax,(key,label,mult) in zip(axs.flat,fields):
        for offset,scene in ((-.19,'head_on'),(.19,'head_on_3v3')):
            values=[]
            for model in models:
                g=next(x for x in groups if x['model']==model and x['scenario']==scene)
                value=(g['flight']['controls_mu_deg_zeta_n']['mean'][2] if key=='normal' else g['sides']['Red']['states']['gamma']['mean'] if key=='gamma'
                       else g['sides']['Red']['pole_step_fraction'] if key=='pole' else g[key]);values.append(value*mult)
            ax.bar(np.arange(len(models))+offset,values,width=.38,label=scene)
        ax.set_ylabel(label);ax.set_xticks(range(len(models)),models,rotation=25,ha='right',fontsize=8);ax.grid(axis='y',alpha=.2)
    axs[0,0].legend(fontsize=8);fig.suptitle('Paired real episodes: 20 seeds 10010000-10010019 per scenario; deterministic Beta mean\nRaw model was trained with random headings; others with head_on. Different rewards: no Return ranking.',fontsize=10)
    fig.tight_layout();save_figure(fig,folder/'models_comparison.png')


def reference_axis_plot(source,folder):
    from env.aircombat import AirCombatConfig
    from env.aircombat.framed_dynamics import frame_rhs
    from tools.audit_actor_flight import scalar_rhs
    folder=Path(folder)
    path=Path(source)/'signed/head_on_3v3_10010003.jsonl.gz'
    if not path.exists():return
    rows=read_rows(path);i=1;t=[];mu=[];effective=[];gauge=[];gamma=[];paper=[];framed=[];difference=[]
    c=AirCombatConfig()
    for row in rows:
        if 'step' not in row or not row['pre_red_alive'][i]:continue
        state=np.array(row['pre_red'][i]);frame=np.array(row['pre_red_frames'][i]);control=np.array(row['red_controls'][i])
        nominal=initial_frame(state);delta=np.arctan2(frame[:,1]@nominal[:,2],frame[:,1]@nominal[:,1]);h=np.linalg.norm(frame[:2,0])
        fd=frame_rhs(np.r_[state[:4],frame.ravel()],control,c);du=fd[4:].reshape(3,3)[:,0]
        t.append((row['step']-1)*.25);mu.append(np.degrees(control[0]));effective.append(np.degrees(wrap_angle(control[0]+delta)));gauge.append(np.degrees(delta))
        gamma.append(np.degrees(np.arctan2(frame[2,0],h)))
        paper.append(np.degrees(scalar_rhs(state,control,c.gravity)[5]) if h>.15 else np.nan)
        framed.append(np.degrees(float(du@nominal[:,2])/np.cos(state[4])) if h>.15 else np.nan)
        p=next((p for p in row['motion_probes'] if p['side']=='Red' and p['id']==i),None)
        difference.append(p['counterfactual_chart_reset_velocity_difference_m_s'] if p else np.nan)
    fig,axs=plt.subplots(4,1,figsize=(12,10),sharex=True)
    axs[0].plot(t,mu,label='commanded mu');axs[0].plot(t,effective,label='mu + saved-axis gauge',ls='--');axs[0].plot(t,gauge,label='saved-axis gauge',ls=':',alpha=.65);axs[0].set_ylabel('angle (deg)')
    axs[1].plot(t,gamma,label='physical gamma');axs[1].axhline(84.2608295,color='purple',ls='--');axs[1].set_ylabel('gamma (deg)')
    axs[2].plot(t,paper,label='Eq.(2), same exported state and mu');axs[2].plot(t,framed,label='actual saved-frame instantaneous derivative',ls='--');axs[2].set_ylabel('exported yaw derivative (deg/s)')
    axs[3].scatter(t,difference,s=12,label='one-step chart-reset CF (copies only)');axs[3].set_ylabel('velocity difference (m/s)');axs[3].set_xlabel('pre-action time (s)')
    for ax in axs:
        ax.axvline(31.25,color='red',lw=.8,ls='--');ax.grid(alpha=.25);ax.legend(fontsize=8,loc='best')
    for row in rows:
        if any(d['side']=='Red' and d['id']==i and d['continuation_substeps'] for d in row.get('numerical_step',[])):
            for ax in axs:ax.axvspan((row['step']-1)*.25,row['step']*.25,color='purple',alpha=.08)
    fig.suptitle('Signed trained Actor | Table VII | seed=10010003 | R1 | actual dt=0.25 s\noutputs/headon_norm_signed_2500/headon_xu_norm_signed_3v3/seed_1/latest.pt\nRed marker = step126 start; CF is not ground truth and is never fed back; yaw derivative masked near vertical',fontsize=9)
    fig.tight_layout();save_figure(fig,folder/'reference_axis_reversal_signed_seed10010003.png')


def sampled_examples(episodes):
    sampled=[]
    for ep in episodes:
        if not ep.get('diagnostic_only'):continue
        actions=[];means=[];stds=[];first=None
        for row in read_rows(ep['trace']):
            if 'actor' not in row:continue
            mask=np.array(row['pre_red_alive'],bool);actor=row['actor']
            actions.extend(np.array(actor['actual_actions'])[mask].tolist())
            means.extend(np.array(actor['beta_mean'])[mask].tolist());stds.extend(np.array(actor['beta_std'])[mask].tolist())
            if first is None:first=dict(raw_observation=row['raw_observation'],actor=actor,physical_controls=row['red_controls'])
        a=np.array(actions);m=np.array(means)
        sampled.append(dict(model=ep['model'],seed=ep['seed'],selection='fresh stochastic evaluation, NOT historical training',
            first_transition=first,steps=ep['steps'],outcome=ep['outcome'],actual_action_stats=describe(a),beta_std=describe(stds),
            absolute_sample_minus_mean=describe(abs(a-m)),
            components_at_trainer_clamp_bound=int(np.sum((a<=1e-6)|(a>=1-1e-6)))))
    return sampled


def target_kinematics(episodes,path):
    """Explicit per-Red LOS telemetry derived ONLY from the saved true endpoints."""
    count=0
    with gzip.open(path,'xt',encoding='utf-8',compresslevel=3) as stream:
        for ep in episodes:
            for row in read_rows(ep['trace']):
                if 'step' not in row:continue
                for i,alive in enumerate(row['pre_red_alive']):
                    if not alive:continue
                    j=row['pre_assignments'][i]
                    s0,s1=np.array(row['pre_red'][i]),np.array(row['post_red'][i])
                    frame=np.array(row['pre_red_frames'][i]);nominal=initial_frame(s0)
                    item=dict(model=ep['model'],scenario=ep['scenario'],seed=ep['seed'],step=row['step'],time_s=row['time_s'],
                        selection=ep['selection'],diagnostic_only=ep.get('diagnostic_only',False),red_id=i,target_before=j,
                        target_after=row['post_assignments'][i],red_alive_after=row['post_red_alive'][i],
                        physical_control=row['red_controls'][i],pre_state=s0.tolist(),post_state=s1.tolist(),
                        saved_reference_gauge_deg=float(np.degrees(np.arctan2(frame[:,1]@nominal[:,2],frame[:,1]@nominal[:,1]))),
                        actual_posture=row['reward_components']['posture'][i],target_geometry=None)
                    if j>=0 and row['pre_blue_alive'][j]:
                        b0,b1=np.array(row['pre_blue'][j]),np.array(row['post_blue'][j]);los0=b0[:3]-s0[:3];los1=b1[:3]-s1[:3]
                        d0,d1=np.linalg.norm(los0),np.linalg.norm(los1)
                        bearings=[]
                        for state,los in ((s0,los0),(s1,los1)):
                            v=velocity_vector(state);degenerate=min(np.linalg.norm(v[:2]),np.linalg.norm(los[:2]))<=1e-9
                            bearings.append(dict(horizontal_bearing_deg=None if degenerate else float(np.degrees(np.arctan2(v[0]*los[1]-v[1]*los[0],np.dot(v[:2],los[:2])))),
                                horizontal_degenerate=bool(degenerate),los_elevation_deg=float(np.degrees(np.arctan2(los[2],np.linalg.norm(los[:2]))))))
                        item['target_geometry']=dict(pre_target_state=b0.tolist(),post_target_state=b1.tolist(),target_alive_after=row['post_blue_alive'][j],
                            d0_m=float(d0),d1_m=float(d1),c_rel_m_s=float((d0-d1)/.25),
                            c_self_m_s=float(np.dot(s1[:3]-s0[:3],los0/d0)/.25) if d0>1e-9 else 0.,
                            before=bearings[0],after=bearings[1],alpha_i_after_deg=row['red_alpha_i_deg'][i][j],
                            alpha_j_after_deg=row['red_alpha_j_deg'][i][j],eligible_after=row['red_eligible'][i][j],
                            dwell_before_death=row['red_dwell_at_resolution'][i][j])
                    stream.write(json.dumps(item,allow_nan=False,separators=(',',':'))+'\n');count+=1
    return count


def analyze(source,output):
    source=Path(source);out=Path(output);out.mkdir(parents=True,exist_ok=False);figures=out/'figures';figures.mkdir()
    episodes=json.loads((source/'episodes.json').read_text());groups=[]
    models=list(dict.fromkeys(ep['model'] for ep in episodes))
    for model in models:
        for scenario in ('head_on','head_on_3v3'):
            chosen=[ep for ep in episodes if ep['model']==model and ep['scenario']==scenario and not ep.get('diagnostic_only')]
            if not chosen:continue
            print(f'[REDUCE] {model} {scenario} n={len(chosen)}',flush=True)
            result=reduce_group(chosen);result.update(model=model,scenario=scenario);groups.append(result)
            write_json(out/f'{model}_{scenario}.json',result)
            # Complete first-seed episode plus a measured polar/high-altitude example if different.
            if scenario=='head_on_3v3' or model=='signed_gated':
                checkpoint='no trained model: reference control/initial weights' if model in ('trim','untrained') else str(Path(next(m['checkpoint'] for m in json.loads((source/'inventory.json').read_text())['models'] if m['label']==model)).as_posix())
                plot_episode(chosen[0],figures,checkpoint)
                if scenario=='head_on_3v3' and model=='signed':
                    worst=max(chosen,key=lambda ep:ep['numerical_diagnostics']['continuation_substeps'])
                    if worst['seed']!=chosen[0]['seed']:plot_episode(worst,figures,checkpoint+' | largest continuation count in this 20-seed batch')
                    case=next(ep for ep in chosen if ep['seed']==10010003)
                    if case['seed'] not in (chosen[0]['seed'],worst['seed']):
                        plot_episode(case,figures,checkpoint+' | reference-axis reversal case R1, step 126')
    write_json(out/'summary.json',groups);comparison_plot(groups,figures);reference_axis_plot(source,figures)
    # A common-start check is stronger than assuming identical seed labels suffice.
    matches=[]
    for scenario in ('head_on','head_on_3v3'):
        for seed in range(10010000,10010020):
            chosen=[ep for ep in episodes if ep['scenario']==scenario and ep['seed']==seed and not ep.get('diagnostic_only')]
            first=chosen[0]['initial']
            matches.append(dict(scenario=scenario,seed=seed,models=len(chosen),exact_initial_state_match=all(ep['initial']==first for ep in chosen)))
    write_json(out/'paired_initial_state_checks.json',matches)
    write_json(out/'sampled_policy_examples.json',sampled_examples(episodes))
    target_kinematics(episodes,out/'assigned_target_kinematics.jsonl.gz')
    with (out/'episodes.csv').open('x',newline='',encoding='utf-8-sig') as f:
        keys=['model','scenario','seed','selection','diagnostic_only','steps','outcome','red_survivors','blue_survivors','win','assignment_switches']
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for ep in episodes:writer.writerow({key:ep.get(key,False) for key in keys})
    print(f'[ANALYSIS_DONE] groups={len(groups)} figures={len(list(figures.glob("*.png")))}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source');p.add_argument('--output',required=True)
    a=p.parse_args();analyze(a.source,a.output)
