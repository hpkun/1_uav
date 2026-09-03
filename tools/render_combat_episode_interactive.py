"""Create a standalone, offline Plotly 3D combat replay."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from tools.combat_visualization import extract_death_frames, extract_events, infer_method_display_name, read_trace, blue_losses_at_frame, recent_events

def _safe(value):
    if isinstance(value, np.ndarray): return [_safe(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)): return [_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)): return None
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.bool_,)): return bool(value)
    return value

def render(trace_path: Path, metadata_path: Path, output: Path, playback_fps: float = 20,
           frame_stride: int = 1, initial_trail: str = "full", title: str = "Combat Episode Replay"):
    trace = read_trace(trace_path); metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata.get("trace_schema_version", 1)) != 1: raise ValueError("unsupported trace schema")
    stride = max(1, int(frame_stride)); rendered = list(range(0, len(trace["steps"]), stride))
    if rendered[-1] != len(trace["steps"]) - 1: rendered.append(len(trace["steps"]) - 1)
    deaths = extract_death_frames(trace); events = metadata.get("events") or extract_events(trace)
    payload = {"red": _safe(trace["red_kinematics"]), "red_alive": _safe(trace["red_alive"]),
               "blue": _safe(trace["blue_kinematics"]), "blue_alive": _safe(trace["blue_alive"]),
               "steps": _safe(trace["steps"]), "time": _safe(trace["time_s"]),
               "active_wave": _safe(trace["active_wave"]), "waves_cleared": _safe(trace["waves_cleared"]),
               "rendered": rendered, "deaths": deaths, "events": events, "initial_trail": initial_trail,
               "metadata": {"method": infer_method_display_name(metadata), "checkpoint": metadata.get("checkpoint_sampled_steps", "?"),
                "training_seed": metadata.get("checkpoint_training_seed", "?"), "episode_seed": metadata.get("episode_seed", "?"),
                "total_waves": metadata.get("total_waves", 3), "arena_radius": metadata.get("arena_radius", 5000),
                "success": bool(metadata.get("red_success")), "termination": metadata.get("termination_reason", ""),
                "final_red": metadata.get("final_red_survivors", 0), "return": metadata.get("episode_return", 0), "sim_time": metadata.get("simulation_time_s", 0)}}
    try:
        from plotly.offline import get_plotlyjs
        plotly_js = get_plotlyjs()
    except ImportError as exc:
        raise RuntimeError("plotly is required for interactive rendering; install plotly>=5.18") from exc
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    html = """<!doctype html><html><head><meta charset='utf-8'><title>__TITLE__</title>
<style>body{margin:0;font:14px Arial;color:#222}#bar{padding:8px;background:#f4f6f8;display:flex;gap:6px;align-items:center;flex-wrap:wrap}button,select{padding:5px}#plot{height:calc(100vh - 110px)}#info{padding:5px 10px;white-space:pre-wrap}label{margin-left:8px}</style></head><body>
<div id='bar'><button id='play'>Play</button><button id='prev'>Previous Frame</button><button id='next'>Next Frame</button><button id='restart'>Restart</button><select id='speed'><option value='.25'>0.25x</option><option value='.5'>0.5x</option><option value='1' selected>1x</option><option value='2'>2x</option><option value='4'>4x</option></select><select id='trail'><option value='0'>Full trail</option><option value='5'>5s trail</option><option value='10'>10s trail</option><option value='20'>20s trail</option></select><button id='camera'>Reset Camera</button><label><input id='heads' type='checkbox' checked> headings</label><label><input id='labels' type='checkbox' checked> labels</label><label><input id='markers' type='checkbox' checked> death markers</label><input id='slider' type='range' min='0' max='__MAX__' value='0' step='1' style='min-width:240px'></div><div id='plot'></div><div id='info'></div>
<script>__PLOTLY__</script><script type='application/json' id='payload'>__PAYLOAD__</script><script>
const P=JSON.parse(document.getElementById('payload').textContent), N=P.rendered.length, plot=document.getElementById('plot'); let ri=0, timer=null;
const colors=['#d62728','#1f77b4']; const traces=[]; const names=[];
function xyz(a){return [a[0],a[1],-a[2]]} function valid(v){return v&&v[0]!==null}
for(let a=0;a<4;a++){traces.push({x:[],y:[],z:[],mode:'lines',type:'scatter3d',line:{color:colors[0],width:3},name:'Red '+(a+1)});names.push('red'+a)}
for(let w=0;w<P.blue[0].length;w++)for(let a=0;a<4;a++){traces.push({x:[],y:[],z:[],mode:'lines',type:'scatter3d',line:{color:colors[1],width:2,dash:['solid','dash','dot'][w%3]},name:'Blue W'+(w+1)+' '+(a+1)});names.push('blue'+w+a)}
const currentRed=traces.length; traces.push({x:[],y:[],z:[],mode:'markers+text',text:[],type:'scatter3d',marker:{size:6,color:colors[0]},name:'Current Red'}); const currentBlue=traces.length; traces.push({x:[],y:[],z:[],mode:'markers+text',text:[],type:'scatter3d',marker:{size:6,color:colors[1]},name:'Current Blue'});
const redHeads=traces.length; traces.push({x:[],y:[],z:[],mode:'lines',type:'scatter3d',line:{color:colors[0],width:3},name:'Red headings',showlegend:false}); const blueHeads=traces.length; traces.push({x:[],y:[],z:[],mode:'lines',type:'scatter3d',line:{color:colors[1],width:3},name:'Blue headings',showlegend:false});
const deathTrace=traces.length; traces.push({x:P.deaths.map(d=>d.position[0]),y:P.deaths.map(d=>d.position[1]),z:P.deaths.map(d=>-d.position[2]),mode:'markers',type:'scatter3d',marker:{symbol:'x',size:7,color:P.deaths.map(d=>d.side==='red'?colors[0]:colors[1])},name:'Deaths',visible:true});
const r=P.metadata.arena_radius, ang=Array.from({length:100},(_,i)=>2*Math.PI*i/99); traces.push({x:ang.map(t=>r*Math.cos(t)),y:ang.map(t=>r*Math.sin(t)),z:ang.map(()=>0),mode:'lines',type:'scatter3d',line:{color:'#999',width:1},name:'Arena'});
Plotly.newPlot(plot,traces,{title:'__TITLE__',uirevision:'combat-replay',scene:{aspectmode:'cube',xaxis:{title:'X / m'},yaxis:{title:'Y / m'},zaxis:{title:'Altitude / m'}},margin:{l:0,r:0,t:40,b:0},showlegend:true},{responsive:true,displaylogo:false});
function frame(){return P.rendered[ri]} function path(d,start,f){let q=d.slice(start,f+1).filter(valid);return {x:q.map(v=>v[0]),y:q.map(v=>v[1]),z:q.map(v=>-v[2])}} function heads(states){let x=[],y=[],z=[];states.forEach(s=>{if(valid(s)){let L=250,ct=Math.cos(s[4]);x.push(s[0],s[0]+L*ct*Math.cos(s[5]),null);y.push(s[1],s[1]+L*ct*Math.sin(s[5]),null);z.push(-s[2],-s[2]-L*Math.sin(s[4]),null)}});return{x,y,z}}
function update(){let f=frame(), active=Math.max(0,Math.min(P.blue[0].length-1,P.active_wave[f]-1)), trail=Number(document.getElementById('trail').value), windowFrames=trail?Math.max(1,Math.round(trail/Math.max(.001,(P.time[1]-P.time[0])))):0, start=windowFrames?Math.max(0,f-windowFrames):0; for(let a=0;a<4;a++){let q=path(P.red.map(v=>v[a]),start,f);Plotly.restyle(plot,{x:[q.x],y:[q.y],z:[q.z]},[a])} let ti=4;for(let w=0;w<P.blue[0].length;w++)for(let a=0;a<4;a++){let q=path(P.blue.map(v=>v[w][a]),start,f);Plotly.restyle(plot,{x:[q.x],y:[q.y],z:[q.z]},[ti++])} let rs=[],rx=[],ry=[],rz=[],rt=[],bs=[],bx=[],by=[],bz=[],bt=[];for(let a=0;a<4;a++){let s=P.red[f][a];if(P.red_alive[f][a]&&valid(s)){rs.push(s);rx.push(s[0]);ry.push(s[1]);rz.push(-s[2]);rt.push('R'+(a+1))}}for(let a=0;a<4;a++){let s=P.blue[f][active][a];if(P.blue_alive[f][active][a]&&valid(s)){bs.push(s);bx.push(s[0]);by.push(s[1]);bz.push(-s[2]);bt.push('B'+(a+1))}}let showLabels=document.getElementById('labels').checked;Plotly.restyle(plot,{x:[rx],y:[ry],z:[rz],text:[showLabels?rt:[]]},[currentRed]);Plotly.restyle(plot,{x:[bx],y:[by],z:[bz],text:[showLabels?bt:[]]},[currentBlue]);let rh=heads(rs),bh=heads(bs),showHeads=document.getElementById('heads').checked;Plotly.restyle(plot,{x:[rh.x],y:[rh.y],z:[rh.z],visible:showHeads},[redHeads]);Plotly.restyle(plot,{x:[bh.x],y:[bh.y],z:[bh.z],visible:showHeads},[blueHeads]);Plotly.restyle(plot,{x:[P.deaths.filter(d=>d.frame<=f).map(d=>d.position[0])],y:[P.deaths.filter(d=>d.frame<=f).map(d=>d.position[1])],z:[P.deaths.filter(d=>d.frame<=f).map(d=>-d.position[2])]},[deathTrace]);document.getElementById('slider').value=ri;let losses=P.deaths.filter(d=>d.side==='blue'&&d.frame<=f).length;let ev=P.events.filter(e=>e.trace_frame<=f).slice(-5).map(e=>(e.type||'').replaceAll('_',' ')+(e.count?' +'+e.count:''));document.getElementById('info').textContent=`${P.metadata.method} | Checkpoint ${P.metadata.checkpoint} | Training seed ${P.metadata.training_seed} | Episode seed ${P.metadata.episode_seed}\nSim time ${Number(P.time[f]).toFixed(1)} s | Step ${P.steps[f]} | Wave ${P.active_wave[f]} | Waves cleared ${P.waves_cleared[f]} | Red alive ${P.red_alive[f].filter(Boolean).length}/4 | Blue alive ${P.blue_alive[f][active].filter(Boolean).length}/4 | Current Total Blue Losses ${losses}\nRecent Events: ${ev.join(' | ')||'none'}${ri===N-1?'\nFINAL: '+(P.metadata.success?'MISSION SUCCESS':'EPISODE COMPLETE')+' | Red survivors '+P.metadata.final_red+' | Return '+Number(P.metadata.return).toFixed(2):''}`;}
document.getElementById('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;document.getElementById('play').textContent='Play'}else{document.getElementById('play').textContent='Pause';timer=setInterval(()=>{ri=(ri+1)%N;update()},1000/(Number(__FPS__)*Number(document.getElementById('speed').value)))}}; document.getElementById('slider').oninput=e=>{ri=Number(e.target.value);update()}; document.getElementById('prev').onclick=()=>{ri=Math.max(0,ri-1);update()}; document.getElementById('next').onclick=()=>{ri=Math.min(N-1,ri+1);update()}; document.getElementById('restart').onclick=()=>{ri=0;update()}; document.getElementById('camera').onclick=()=>Plotly.relayout(plot,{'scene.camera':null}); document.getElementById('heads').onchange=update; document.getElementById('labels').onchange=update; document.getElementById('markers').onchange=e=>Plotly.restyle(plot,{visible:e.target.checked},[deathTrace]);document.getElementById('trail').onchange=update;if(P.initial_trail!=='full')document.getElementById('trail').value=String(P.initial_trail).replace('s','');update();
</script></body></html>"""
    html = html.replace("__TITLE__", str(title).replace("'", "&#39;")).replace("__PLOTLY__", plotly_js).replace("__PAYLOAD__", payload_json).replace("__MAX__", str(len(rendered)-1)).replace("__FPS__", str(float(playback_fps)))
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(html, encoding="utf-8")

def main():
    p=argparse.ArgumentParser(); p.add_argument('--input-dir'); p.add_argument('--trace'); p.add_argument('--metadata'); p.add_argument('--output'); p.add_argument('--playback-fps',type=float,default=20); p.add_argument('--frame-stride',type=int,default=1); p.add_argument('--initial-trail',default='full'); p.add_argument('--title',default='Combat Episode Replay'); a=p.parse_args(); d=Path(a.input_dir) if a.input_dir else None; trace=Path(a.trace) if a.trace else d/'episode_trace.npz'; meta=Path(a.metadata) if a.metadata else d/'metadata.json'; out=Path(a.output) if a.output else d/'episode_interactive.html'; render(trace,meta,out,a.playback_fps,a.frame_stride,a.initial_trail,a.title)
if __name__=='__main__': main()
