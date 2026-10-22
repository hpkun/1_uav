import numpy as np
from env.region_competition.reward import COMPONENTS


class RegionAccumulator:
    def reset(self,info):
        self.n=len(info['states']); self.steps=0
        self.parts={k:0. for k in COMPONENTS}; self.returns=np.zeros(self.n)
        self.flight={t:dict(steps=np.zeros(self.n,dtype=int),path=np.zeros(self.n),speed=np.zeros(self.n),
                           gamma=[],dwell=np.zeros(self.n),safe=0,cross_safe=0,
                           numerical=dict.fromkeys(('speed_clips','gamma_clips','nonfinite_states'),0)) for t in ('red','blue')}

    def add(self,info,reward):
        self.steps+=1; self.returns+=reward
        for k in self.parts: self.parts[k]+=float(np.sum(info['red']['reward_components'][k]))
        for t in self.flight:
            f=self.flight[t]; row=info[t]; valid=row['current_active_mask']; ids=np.flatnonzero(valid)
            f['steps']+=valid
            f['path'][ids]+=np.linalg.norm(row['states'][ids,:3]-row['previous_states'][ids,:3],axis=1)
            f['speed'][ids]+=row['states'][ids,3]
            f['gamma'].extend(row['states'][ids,4].tolist())
            f['dwell']+=np.asarray(row['dwell']).sum(1)
            for k in f['numerical']: f['numerical'][k]+=int(np.sum(row['numerical_events'][k]))
            active=np.flatnonzero(row['active_mask'])
            for i in active:
                friends=active[active!=i]
                other='blue' if t=='red' else 'red'; opponents=info[other]['states'][info[other]['active_mask'],:3]
                if len(friends) and np.any(np.linalg.norm(row['states'][friends,:3]-row['states'][i,:3],axis=1)<info['safe_distance']): f['safe']+=1
                if len(opponents) and np.any(np.linalg.norm(opponents-row['states'][i,:3],axis=1)<info['safe_distance']): f['cross_safe']+=1

    def summary(self,info):
        result=dict(seed=info.get('episode_seed'),episode_length=self.steps,red_score=info['scores']['red'],blue_score=info['scores']['blue'],
                    relative_score=info['scores']['red']-info['scores']['blue'],winner=info['winner'],
                    episode_return=float(self.returns.mean()),reward_components={k:v/self.n for k,v in self.parts.items()},
                    terminated=info['terminated'],truncated=info['truncated'],blue_policy=info['blue_policy'])
        for t,f in self.flight.items():
            count=max(1,int(f['steps'].sum()))
            result[t]=dict(active_flight_steps=f['steps'].tolist(),path_m=f['path'].tolist(),
                           mean_speed_m_s=float(f['speed'].sum()/count),per_uav_mean_speed_m_s=(f['speed']/np.maximum(1,f['steps'])).tolist(),
                           gamma_quantiles_rad=np.quantile(f['gamma'],[0,.25,.5,.75,1]).tolist() if f['gamma'] else None,
                           dwell_seconds=f['dwell'].tolist(),mean_dwell_seconds=float(f['dwell'].mean()),
                           friendly_separation_violation_rate=f['safe']/count,cross_team_proximity_rate=f['cross_safe']/count,
                           numerical_events=f['numerical'].copy(),numerical_rates={k:v/count for k,v in f['numerical'].items()})
        return result


def aggregate(rows):
    if not rows: return dict(episodes=0)
    result=dict(episodes=len(rows),per_episode=rows)
    for key in ('red_score','blue_score','relative_score','episode_return','episode_length'):
        result['mean_'+key]=float(np.mean([r[key] for r in rows]))
    result['red_win_rate']=sum(r['winner']=='red' for r in rows)/len(rows)
    result['draw_rate']=sum(r['winner']=='draw' for r in rows)/len(rows)
    for t in ('red','blue'):
        result[t]={k:float(np.mean([r[t][k] for r in rows])) for k in ('mean_speed_m_s','mean_dwell_seconds','friendly_separation_violation_rate','cross_team_proximity_rate')}
        result[t]['mean_numerical_rates']={k:float(np.mean([r[t]['numerical_rates'][k] for r in rows])) for k in ('speed_clips','gamma_clips','nonfinite_states')}
        result[t]['mean_path_m']=float(np.mean([sum(r[t]['path_m']) for r in rows]))
    return result
