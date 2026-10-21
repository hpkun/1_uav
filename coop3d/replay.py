"""Self-contained interactive Plotly replay from recorded frame data."""
import json
from pathlib import Path
from plotly.offline import get_plotlyjs


def render_replay(trace_path, output_path):
    data = json.loads(Path(trace_path).read_text(encoding='utf-8'))
    if not data.get('frames'):
        raise ValueError('empty trajectory')
    template = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>Coop3D · 三维巡检回放</title><style>
body{margin:0;background:#101827;color:#edf2ff;font:15px system-ui}header{padding:18px 24px;border-bottom:1px solid #30405a}
h1{margin:0;font-size:23px}small{color:#a8bad7}.layout{display:grid;grid-template-columns:1fr 340px;height:77vh}
#plot{min-width:0}aside{padding:20px;background:#162237;overflow:auto}pre{white-space:pre-wrap;font-size:13px;line-height:1.65}
footer{display:flex;align-items:center;gap:18px;padding:18px 24px}input{flex:1}button,select{background:#29476b;color:white;border:1px solid #6686ac;border-radius:6px;padding:8px}
</style><header><h1>Coop3D · 三维巡检任务</h1><small id="subtitle"></small></header>
<div class="layout"><div id="plot"></div><aside><h3 id="status"></h3><label>UAV 状态查询 <select id="uav"></select></label><pre id="details"></pre><h3>本步事件</h3><pre id="events"></pre></aside></div>
<footer><button id="play">播放</button><input id="time" type="range" min="0" value="0"><span id="clock"></span></footer>
<script>__PLOTLY__</script><script>
const DATA=__DATA__, F=DATA.frames, N=F[0].states.length;
const COLORS=['#63dcfa','#ffbe73','#c2a5ff','#74e2ad','#ff8fab'];
const slider=document.getElementById('time'), select=document.getElementById('uav');
slider.max=F.length-1; for(let i=0;i<N;i++)select.add(new Option('UAV '+i,i));
document.getElementById('subtitle').textContent=DATA.environment_version+' · '+DATA.config.scale+' · seed '+DATA.seed+' · 数据来自实际环境记录';
let index=0,timer=null;
const all=F.flatMap(f=>f.states.map(s=>s.slice(0,3))).concat(F[0].waypoints);
function range(k){const a=all.map(p=>p[k]);const lo=Math.min(...a),hi=Math.max(...a),pad=Math.max(300,(hi-lo)*.08);return [lo-pad,hi+pad]}
const layout={paper_bgcolor:'#101827',plot_bgcolor:'#101827',font:{color:'#dde8ff'},margin:{l:0,r:0,t:20,b:0},uirevision:'camera',
scene:{aspectmode:'data',xaxis:{title:'X (m)',range:range(0)},yaxis:{title:'Y (m)',range:range(1)},zaxis:{title:'Z (m) ↑',range:range(2)},bgcolor:'#101827'},legend:{orientation:'h'}};
function update(){const f=F[index],tr=[];
 for(let i=0;i<N;i++){const path=F.slice(0,index+1).map(x=>x.states[i]);
 tr.push({type:'scatter3d',mode:'lines',name:'UAV '+i+' 轨迹',x:path.map(s=>s[0]),y:path.map(s=>s[1]),z:path.map(s=>s[2]),line:{color:COLORS[i],width:4}});
 const s=f.states[i];tr.push({type:'scatter3d',mode:'markers',showlegend:false,x:[s[0]],y:[s[1]],z:[s[2]],marker:{size:6,color:f.active_mask[i]?COLORS[i]:'#637084'},text:['UAV '+i+'<br>v='+s[3].toFixed(2)+' m/s<br>γ='+s[4].toFixed(4)+'<br>ψ='+s[5].toFixed(4)+'<br>目标='+f.assignments[i]],hoverinfo:'text'});
 const j=f.assignments[i];if(j>=0){const p=f.waypoints[j];tr.push({type:'scatter3d',mode:'lines',showlegend:false,x:[s[0],p[0]],y:[s[1],p[1]],z:[s[2],p[2]],line:{color:COLORS[i],width:2,dash:'dash'},hoverinfo:'skip'})}}
 tr.push({type:'scatter3d',mode:'markers+text',name:'巡检航路点',x:f.waypoints.map(p=>p[0]),y:f.waypoints.map(p=>p[1]),z:f.waypoints.map(p=>p[2]),text:f.completed.map((c,j)=>'P'+j+(c?' ✓':'')),textposition:'top center',marker:{size:8,symbol:'diamond',color:f.completed.map(c=>c?'#74e2ad':'#ffbe73')},hovertemplate:'%{text}<br>X=%{x}<br>Y=%{y}<br>Z=%{z}<extra></extra>'});
 Plotly.react('plot',tr,layout,{responsive:true,displaylogo:false});
 const i=+select.value,s=f.states[i],r=f.reward_components||{};
 document.getElementById('status').textContent='完成 '+f.completed_waypoints+'/'+f.total_waypoints+(f.success?' · 成功':'');
 document.getElementById('details').textContent='有效: '+!!f.active_mask[i]+'\n位置 (m): '+s.slice(0,3).map(x=>x.toFixed(1)).join(', ')+'\n速度: '+s[3].toFixed(2)+' m/s\nγ / ψ (rad): '+s.slice(4).map(x=>x.toFixed(4)).join(' / ')+'\n分配目标: '+f.assignments[i]+'\n控制 μ, ζ, ζ̂: '+(f.physical_actions?f.physical_actions[i].map(x=>x.toFixed(3)).join(', '):'初始状态')+'\n友机距离 (m): '+f.pairwise_distances[i].map(x=>x.toFixed(1)).join(', ')+'\n\nReward\n'+Object.entries(r).map(([k,v])=>k+': '+v[i].toFixed(5)).join('\n')+'\n总和: '+(f.rewards?f.rewards[i].toFixed(5):'0');
 document.getElementById('events').textContent=JSON.stringify({completion:f.events,failures:f.failures||[]},null,2);
 document.getElementById('clock').textContent='步 '+f.episode_length+' / '+(F.length-1)+' · '+(f.episode_length*DATA.config.dt).toFixed(2)+' s';slider.value=index;
}
slider.oninput=()=>{index=+slider.value;update()};select.onchange=update;
document.getElementById('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;document.getElementById('play').textContent='播放'}else{if(index===F.length-1)index=0;document.getElementById('play').textContent='暂停';timer=setInterval(()=>{if(index<F.length-1){index++;update()}else{clearInterval(timer);timer=null;document.getElementById('play').textContent='播放'}},100)}};update();
</script></html>'''
    content = template.replace('__PLOTLY__', get_plotlyjs()).replace('__DATA__', json.dumps(data, ensure_ascii=False, allow_nan=False).replace('</', '<\\/'))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding='utf-8')
    return output
