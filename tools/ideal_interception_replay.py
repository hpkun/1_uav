"""HTML and MP4 export from one recorded ideal-interception trajectory."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from bisect import bisect_right

import numpy as np

from .ideal_interception_timeline import Timeline


def _coordinates(point,view):
    return point if view=="3d" else point[:2] if view=="top" else (point[0],point[2])


def video_times(duration,fps):
    if not math.isfinite(duration) or duration<0 or fps<=0:
        raise ValueError("finite duration and positive FPS required")
    count=max(1,math.ceil(duration*fps))
    times=[k/fps for k in range(count)]
    times[-1]=duration
    return times


def protected_cylinder_geometry(scene,angles=64,levels=8):
    center=np.asarray(scene["protected_center"],float)
    radius=float(scene["protected_radius"])
    theta=np.linspace(0,2*np.pi,angles)
    angle,height=np.meshgrid(theta,np.linspace(center[2],center[2]+scene["protected_height"],levels))
    return dict(x=center[0]+radius*np.cos(angle),y=center[1]+radius*np.sin(angle),
                z=height,center=center)


def render_html(source,output,view="3d",overwrite=False):
    import plotly.graph_objects as go
    timeline=Timeline(source)
    times=sorted(set(timeline.key_times+video_times(timeline.duration,30)))
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
    cylinder=protected_cylinder_geometry(scene)
    center=cylinder["center"]
    radius=scene["protected_radius"]
    theta=np.linspace(0,2*np.pi,64)
    data=[]
    if is3d:
        data.append(go.Surface(x=cylinder["x"],y=cylinder["y"],z=cylinder["z"],
                               opacity=.18,showscale=False,name="Protected region"))
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
    label={"3d":"3D Overview","top":"XY Top View","altitude":"XZ Side View"}[view]
    fig.update_layout(title=f"{label} | {timeline.data['scenario_id'][:12]}",height=700)
    if is3d:
        fig.update_layout(scene=dict(aspectmode="data",xaxis_title="x (m)",yaxis_title="y (m)",
                                     zaxis_title="z (m)"))
    else:
        fig.update_xaxes(title_text="x (m)",range=[scene["lower"][0],scene["upper"][0]])
        axis=1 if view=="top" else 2
        fig.update_yaxes(title_text=("y (m)" if view=="top" else "z (m)"),
                         range=[scene["lower"][axis],scene["upper"][axis]])
    times_json=json.dumps(times,allow_nan=False)
    keys_json=json.dumps(timeline.key_times,allow_nan=False)
    script="""
const plot=document.getElementById('ideal-plot');
const times=TIMES, keys=KEYS, duration=times[times.length-1];
const clock=document.getElementById('clock'), stamp=document.getElementById('stamp');
let current=0, running=false, origin=0, started=0, shown=-1, rate=1;
clock.max=String(duration);
function indexAt(values,t) {
  let lo=0,hi=values.length;
  while(lo+1<hi){const mid=(lo+hi)>>1;if(values[mid]<=t)lo=mid;else hi=mid;}
  return lo;
}
function show(t){
  current=Math.min(duration,Math.max(0,t));
  clock.value=String(current);stamp.textContent=current.toFixed(3)+' s';
  const k=indexAt(times,current);
  if(k!==shown){shown=k;Plotly.animate(plot,[String(k)],{mode:'immediate',
    frame:{duration:0,redraw:true},transition:{duration:0}});}
}
function tick(now){
  if(!running)return;
  show(origin+(now-started)*rate/1000);
  if(current>=duration){running=false;return;}
  requestAnimationFrame(tick);
}
function pause(){
  if(running)show(origin+(performance.now()-started)*rate/1000);
  running=false;
}
document.getElementById('play').onclick=()=>{
  if(running)return;if(current>=duration)show(0);
  origin=current;started=performance.now();running=true;requestAnimationFrame(tick);
};
document.getElementById('pause').onclick=pause;
document.getElementById('rate').onchange=()=>{const resume=running;if(resume)pause();rate=Number(document.getElementById('rate').value);if(resume){origin=current;started=performance.now();running=true;requestAnimationFrame(tick);}};
clock.oninput=()=>{pause();show(Number(clock.value));};
document.getElementById('previous').onclick=()=>{pause();show(keys[Math.max(0,indexAt(keys,current-1e-9))]);};
document.getElementById('next').onclick=()=>{pause();show(keys[Math.min(keys.length-1,indexAt(keys,current+1e-9)+1)]);};
show(0);
""".replace("TIMES",times_json).replace("KEYS",keys_json)
    graph=fig.to_html(full_html=False,include_plotlyjs=True,div_id="ideal-plot")
    html=("<!doctype html><meta charset='utf-8'><title>Ideal interception replay</title>"
          "<style>body{font:15px system-ui;margin:16px}#clock{width:45vw}</style>"
          "<button id='play'>Play</button><button id='pause'>Pause</button>"
          "<button id='previous'>Previous keyframe</button><button id='next'>Next keyframe</button>"
          "<select id='rate'><option value='.5'>0.5×</option><option value='1' selected>1×</option>"
          "<option value='2'>2×</option><option value='4'>4×</option></select>"
          f"<input id='clock' type='range' min='0' max='{timeline.duration}' step='0.001' value='0'>"
          "<span id='stamp'></span>"+graph+"<script>"+script+"</script>")
    _atomic_text(output,html,overwrite)


def _publish(temporary,output,overwrite):
    if overwrite:
        os.replace(temporary,output)
    else:
        os.link(temporary,output)


def _atomic_text(output,content,overwrite=False):
    output=Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    try:
        with tempfile.NamedTemporaryFile("w",encoding="utf-8",dir=output.parent,
                                         suffix=".html",delete=False) as stream:
            temporary=Path(stream.name)
            stream.write(content)
        _publish(temporary,output,overwrite)
    finally:
        if "temporary" in locals():
            temporary.unlink(missing_ok=True)


def _render_mp4_to_path(source,output,view="3d",fps=30):
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
    cylinder=protected_cylinder_geometry(scene)
    center=cylinder["center"]
    radius=scene["protected_radius"]
    theta=np.linspace(0,2*np.pi,64)
    if view=="3d":
        axis.plot_surface(cylinder["x"],cylinder["y"],cylinder["z"],color="#55a868",alpha=.15)
        for level in (0,-1):
            axis.plot(cylinder["x"][level],cylinder["y"][level],cylinder["z"][level],color="#55a868")
    elif view=="top":
        axis.plot(center[0]+radius*np.cos(theta),center[1]+radius*np.sin(theta),color="#55a868")
    else:
        axis.plot([center[0]-radius,center[0]+radius,center[0]+radius,center[0]-radius,center[0]-radius],
                  [center[2],center[2],center[2]+scene["protected_height"],
                   center[2]+scene["protected_height"],center[2]],color="#55a868")
    for reference in timeline.references:
        coords=np.asarray([_coordinates(point,view) for point in reference[0]])
        axis.plot(*coords.T,linestyle="--",color=colors["intruder"],alpha=.4)
    axis.set_xlim(scene["lower"][0],scene["upper"][0])
    axis.set_ylim(scene["lower"][1 if view!="altitude" else 2],
                  scene["upper"][1 if view!="altitude" else 2])
    if view=="3d":
        axis.set_zlim(scene["lower"][2],scene["upper"][2])
        axis.set_box_aspect(np.subtract(scene["upper"],scene["lower"]))
    first=history[0]["entities"]
    history_times=timeline.key_times
    positions={entity["id"]:np.asarray([_coordinates(next(e for e in row["entities"]
                 if e["id"]==entity["id"])["position"],view) for row in history]) for entity in first}
    artists={}
    for entity in first:
        color=colors[entity["role"]]
        trail=axis.plot([],[],*([[]] if view=="3d" else []),color=color,linewidth=1.5)[0]
        point=axis.plot([],[],*([[]] if view=="3d" else []),marker="o",linestyle="none",color=color)[0]
        artists[entity["id"]]=(trail,point)
    event_artists=[]
    for event in timeline.events:
        if event["kind"] not in ("interception_contact","friendly_collision","intruder_collision",
                                 "region_breach","boundary_exit","path_complete") or not event["positions"]:
            continue
        point=_coordinates(np.mean(event["positions"],axis=0),view)
        artist=axis.plot(*([v] for v in point),marker="x",linestyle="none",color="#e69500")[0]
        artist.set_visible(False)
        event_artists.append((event["time"],artist))
    def set_artist(artist,coordinates):
        if view=="3d":
            artist.set_data_3d(*coordinates.T)
        else:
            artist.set_data(*coordinates.T)
    try:
        with FFMpegWriter(fps=fps,codec=codec,extra_args=["-pix_fmt","yuv420p"]).saving(fig,str(output),100) as writer:
            for time in video_times(timeline.duration,fps):
                row=timeline.at(time)
                end=bisect_right(history_times,time)
                for entity in row["entities"]:
                    trail,point=artists[entity["id"]]
                    current=np.asarray(_coordinates(entity["position"],view))
                    path=np.vstack((positions[entity["id"]][:end],current))
                    set_artist(trail,path)
                    set_artist(point,current[None,:])
                    point.set_visible(entity["active"])
                for event_time,artist in event_artists:
                    artist.set_visible(event_time<=time)
                axis.set_title(f"{time:.3f} s | {row['outcome']} | {timeline.data['scenario_id'][:12]}")
                writer.grab_frame()
    finally:
        plt.close(fig)


def render_mp4(source,output,view="3d",fps=30,overwrite=False):
    output=Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    with tempfile.TemporaryDirectory(dir=output.parent) as folder:
        temporary=Path(folder)/"video.mp4"
        _render_mp4_to_path(source,temporary,view,fps)
        if not temporary.is_file() or temporary.stat().st_size==0:
            raise RuntimeError("FFmpeg did not produce a valid video file")
        _publish(temporary,output,overwrite)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--view",choices=("3d","top","altitude"),default="3d")
    parser.add_argument("--fps",type=int,default=30)
    parser.add_argument("--overwrite",action="store_true")
    args=parser.parse_args(argv)
    if args.output.suffix==".mp4":
        render_mp4(args.trace,args.output,args.view,args.fps,args.overwrite)
    elif args.output.suffix==".html":
        render_html(args.trace,args.output,args.view,args.overwrite)
    else:
        parser.error("output must end in .html or .mp4")


if __name__=="__main__":
    main()
