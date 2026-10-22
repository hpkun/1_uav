"""Interactive standalone Red/Blue replay using only the recorded trajectory."""
import json
from pathlib import Path
import numpy as np
import plotly.graph_objects as go


def render_replay(trajectory_path,output_path):
    source=Path(trajectory_path); out=Path(output_path)
    if out.resolve()==source.resolve() or out.exists(): raise FileExistsError('replay output must be new')
    data=json.loads(source.read_text(encoding='utf-8')); frames=data['frames']
    radius=data['config']['region_radius']
    paths={t:np.asarray([f[t]['states'] for f in frames]) for t in ('red','blue')}
    theta=np.linspace(0,2*np.pi,33); phi=np.linspace(0,np.pi,17)
    xx=radius*np.outer(np.cos(theta),np.sin(phi)); yy=radius*np.outer(np.sin(theta),np.sin(phi)); zz=radius*np.outer(np.ones_like(theta),np.cos(phi))

    def traces(k):
        result=[]
        for team,color,dash in [('red','#d62828','solid'),('blue','#1764c0','dash')]:
            for i in range(paths[team].shape[1]):
                s=paths[team][:k+1,i,:3]
                result.append(go.Scatter3d(x=s[:,0],y=s[:,1],z=s[:,2],mode='lines+markers',name=f'{team} UAV {i}',
                    line=dict(color=color,width=4,dash=dash),marker=dict(size=2,color=color),
                    hovertemplate=f'{team} {i}<br>x=%{{x:.1f}}m<br>y=%{{y:.1f}}m<br>z=%{{z:.1f}}m<extra></extra>'))
        for j,c in enumerate(frames[k]['regions']):
            result.append(go.Surface(x=xx+c[0],y=yy+c[1],z=zz+c[2],opacity=.12,showscale=False,
                                     colorscale=[[0,'#45a46b'],[1,'#45a46b']],name=f'Region {j}',hoverinfo='skip'))
        return result

    def title(k):
        f=frames[k]
        return f'动态区域竞争 | step {f["step"]} | Red {f["scores"]["red"]:.3f}s / Blue {f["scores"]["blue"]:.3f}s'
    all_positions=np.concatenate([p[:,:,:3].reshape(-1,3) for p in paths.values()]+[np.asarray([f['regions'] for f in frames]).reshape(-1,3)])
    low=all_positions.min(0)-radius; high=all_positions.max(0)+radius
    layout=dict(title=title(0),scene=dict(xaxis=dict(title='x (m)',range=[low[0],high[0]]),
                yaxis=dict(title='y (m)',range=[low[1],high[1]]),zaxis=dict(title='z (m), up',range=[low[2],high[2]]),aspectmode='data'),
                updatemenus=[dict(type='buttons',buttons=[dict(label='播放',method='animate',args=[None,dict(frame=dict(duration=70,redraw=True),fromcurrent=True)]),
                    dict(label='暂停',method='animate',args=[[None],dict(mode='immediate',frame=dict(duration=0,redraw=False))])])],
                sliders=[dict(currentvalue=dict(prefix='step '),steps=[dict(label=str(f['step']),method='animate',
                    args=[[str(k)],dict(mode='immediate',frame=dict(duration=0,redraw=True),transition=dict(duration=0))]) for k,f in enumerate(frames)])])
    figure=go.Figure(data=traces(0),layout=layout,frames=[go.Frame(name=str(k),data=traces(k),layout=go.Layout(title=title(k))) for k in range(len(frames))])
    figure.write_html(str(out),include_plotlyjs=True,auto_play=False,full_html=True)
    return out
