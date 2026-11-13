"""HTML and MP4 export from one recorded ideal-interception trajectory."""
import argparse
from pathlib import Path
import shutil
import subprocess

import numpy as np

from .ideal_interception_timeline import Timeline
from .interception_replay import video_times


def _coordinates(point,view):
    return point if view=="3d" else point[:2] if view=="top" else (point[0],point[2])


def render_html(source,output,view="3d"):
    import plotly.graph_objects as go
    timeline=Timeline(source)
    times=timeline.key_times
    rows=[timeline.at(t) for t in times]
    first=rows[0]["entities"]
    colors={e["id"]:("#d64040" if e["role"]=="defender" else "#2378bf") for e in first}
    is3d=view=="3d"
    def line(points,**options):
        coords=np.asarray(points,float)
        if not len(coords):
            coords=np.empty((0,3))
        if is3d:
            return go.Scatter3d(x=coords[:,0],y=coords[:,1],z=coords[:,2],**options)
        x,y=(coords[:,0],coords[:,1]) if view=="top" else (coords[:,0],coords[:,2])
        return go.Scatter(x=x,y=y,**options)
    scene=timeline.data["scene"]
    center=np.asarray(scene["protected_center"],float)
    radius=scene["protected_radius"]
    theta=np.linspace(0,2*np.pi,64)
    data=[]
    if is3d:
        angle,height=np.meshgrid(theta,[center[2],center[2]+scene["protected_height"]])
        data.append(go.Surface(x=center[0]+radius*np.cos(angle),y=center[1]+radius*np.sin(angle),
            z=height,opacity=.18,showscale=False,name="Protected region"))
    elif view=="top":
        data.append(line(np.column_stack((center[0]+radius*np.cos(theta),
             center[1]+radius*np.sin(theta),np.zeros_like(theta))),mode="lines",
             name="Protected region",line=dict(color="#55a868",width=2)))
    else:
        data.append(line([[center[0]-radius,0,center[2]],
                          [center[0]+radius,0,center[2]],
                          [center[0]+radius,0,center[2]+scene["protected_height"]],
                          [center[0]-radius,0,center[2]+scene["protected_height"]],
                          [center[0]-radius,0,center[2]]],mode="lines",name="Protected region",
                         line=dict(color="#55a868",width=2)))
    for j,reference in enumerate(timeline.references):
        data.append(line(reference[0],mode="lines",name=f"I{j} reference",
                         line=dict(color="#2378bf",dash="dash",width=3),opacity=.55))
    static=len(data)
    def dynamic(k):
        row=rows[k]
        traces=[]
        for entity in row["entities"]:
            member=entity["id"]
            history=np.array([sample["entities"][[e["id"] for e in sample["entities"]].index(member)]["position"]
                              for sample in rows[:k+1]])
            color=colors[member]
            traces.append(line(history,mode="lines",name=f"{entity['name']} actual",
                               line=dict(color=color,width=3)))
            traces.append(line([entity["position"]] if entity["active"] else [],mode="markers",
                               name=entity["name"],marker=dict(color=color,size=7)))
            velocity=np.asarray(entity["velocity"])
            arrow=[entity["position"],(np.asarray(entity["position"])+velocity*.15).tolist()]
            traces.append(line(arrow if entity["active"] else [],mode="lines",
                               name=f"{entity['name']} velocity",showlegend=False,
                               line=dict(color=color,width=4)))
        events=[e for e in row["events"] if e["positions"]]
        points=[np.mean(e["positions"],axis=0) for e in events]
        traces.append(line(points,mode="markers",name="Geometric events",
                           text=[f"{e['kind']} @ {e['time']:.6f} s" for e in events],
                           marker=dict(color="#e69500",size=8,symbol="x")))
        return traces
    data.extend(dynamic(0))
    frames=[go.Frame(data=dynamic(k),traces=list(range(static,len(data))),
                     name=str(k),layout=go.Layout(title=f"t={t:.6f} s | {rows[k]['outcome']}"))
            for k,t in enumerate(times)]
    fig=go.Figure(data=data,frames=frames)
    sliders=[dict(steps=[dict(method="animate",label=f"{t:.3f}",args=[[str(k)],
        dict(mode="immediate",frame=dict(duration=0,redraw=True),transition=dict(duration=0))])
        for k,t in enumerate(times)],currentvalue=dict(prefix="Time (s): "))]
    fig.update_layout(title=f"Ideal interception | {timeline.data['scenario_id'][:12]}",
        sliders=sliders,updatemenus=[dict(type="buttons",buttons=[
            dict(label="Play",method="animate",args=[None,dict(frame=dict(duration=100,redraw=True),
                transition=dict(duration=0),fromcurrent=True)]),
            dict(label="Pause",method="animate",args=[[None],dict(frame=dict(duration=0,redraw=False),
                mode="immediate")])])],height=700)
    if is3d:
        fig.update_layout(scene=dict(aspectmode="data",xaxis_title="x (m)",yaxis_title="y (m)",
                                     zaxis_title="z (m)"))
    else:
        fig.update_xaxes(title_text="x (m)",range=[scene["lower"][0],scene["upper"][0]])
        axis=1 if view=="top" else 2
        fig.update_yaxes(title_text=("y (m)" if view=="top" else "z (m)"),
                         range=[scene["lower"][axis],scene["upper"][axis]])
    fig.write_html(output,include_plotlyjs=True,full_html=True,auto_open=False)


def render_mp4(source,output,view="3d",fps=30):
    ffmpeg=shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("FFmpeg required for MP4 export")
    encoders=subprocess.check_output([ffmpeg,"-hide_banner","-encoders"],text=True).splitlines()
    names={line.split()[1] for line in encoders if len(line.split())>2 and line.startswith(" V")}
    codec=next((name for name in ("libx264","libopenh264","mpeg4") if name in names),None)
    if codec is None:
        raise RuntimeError("FFmpeg needs an H.264 or MPEG-4 video encoder")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    matplotlib.rcParams["animation.ffmpeg_path"]=ffmpeg
    timeline=Timeline(source)
    scene=timeline.data["scene"]
    history=[timeline.at(t) for t in timeline.key_times]
    fig=plt.figure(figsize=(12.8,7.2),dpi=100)
    axis=fig.add_subplot(projection="3d" if view=="3d" else None)
    colors={"defender":"#d64040","intruder":"#2378bf"}
    output=Path(output)
    with FFMpegWriter(fps=fps,codec=codec,extra_args=["-pix_fmt","yuv420p"]).saving(fig,str(output),100) as writer:
        for time in video_times(timeline.duration,fps):
            axis.clear()
            row=timeline.at(time)
            for j,reference in enumerate(timeline.references):
                coords=np.asarray([_coordinates(point,view) for point in reference[0]])
                if view=="3d":
                    axis.plot(*coords.T,linestyle="--",color=colors["intruder"],alpha=.4)
                else:
                    axis.plot(*coords.T,linestyle="--",color=colors["intruder"],alpha=.4)
            center=np.asarray(scene["protected_center"])
            radius=scene["protected_radius"]
            theta=np.linspace(0,2*np.pi,64)
            if view=="top":
                axis.plot(center[0]+radius*np.cos(theta),center[1]+radius*np.sin(theta),color="#55a868")
            elif view=="altitude":
                axis.plot([center[0]-radius,center[0]+radius,center[0]+radius,center[0]-radius,center[0]-radius],
                          [center[2],center[2],center[2]+scene["protected_height"],
                           center[2]+scene["protected_height"],center[2]],color="#55a868")
            else:
                axis.plot(center[0]+radius*np.cos(theta),center[1]+radius*np.sin(theta),
                          np.full_like(theta,center[2]),color="#55a868")
            for entity in row["entities"]:
                point=_coordinates(entity["position"],view)
                prior=[sample["entities"][[e["id"] for e in sample["entities"]].index(entity["id"])]["position"]
                       for sample in history if sample["time"]<time]
                trail=np.asarray([_coordinates(p,view) for p in prior+[entity["position"]]])
                axis.plot(*trail.T,color=colors[entity["role"]],linewidth=1.5)
                if entity["active"]:
                    axis.scatter(*point,color=colors[entity["role"]],s=45)
                    axis.text(*point,entity["name"])
            for event in row["events"]:
                if event["positions"]:
                    point=_coordinates(np.mean(event["positions"],axis=0),view)
                    axis.scatter(*point,color="#e69500",marker="x",s=80)
            axis.set_xlim(*[scene["lower"][0],scene["upper"][0]])
            axis.set_ylim(*[scene["lower"][1 if view=="top" or view=="3d" else 2],
                             scene["upper"][1 if view=="top" or view=="3d" else 2]])
            if view=="3d":
                axis.set_zlim(scene["lower"][2],scene["upper"][2])
                axis.set_box_aspect(np.subtract(scene["upper"],scene["lower"]))
            axis.set_title(f"{time:.3f} s | {row['outcome']} | {timeline.data['scenario_id'][:12]}")
            writer.grab_frame()
    plt.close(fig)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--view",choices=("3d","top","altitude"),default="3d")
    parser.add_argument("--fps",type=int,default=30)
    args=parser.parse_args(argv)
    if args.output.suffix==".mp4":
        render_mp4(args.trace,args.output,args.view,args.fps)
    elif args.output.suffix==".html":
        render_html(args.trace,args.output,args.view)
    else:
        parser.error("output must end in .html or .mp4")


if __name__=="__main__":
    main()
