/* Exercise the offline Plotly player in a real Chromium browser via CDP. */
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {mkdtemp, readFile, rm, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join, dirname, resolve} from 'node:path';
import {pathToFileURL} from 'node:url';

const [browser, html, screenshot] = process.argv.slice(2);
assert(browser && html, 'usage: node tests/check_interception_browser.mjs BROWSER REPLAY.html [SCREENSHOT.png]');
const replay = JSON.parse((await readFile(html, 'utf8')).match(/const DATA = (.*);\r?\n/)[1]);
const parent = resolve(tmpdir());
const profile = await mkdtemp(join(parent, 'interception-browser-'));
const child = spawn(browser, ['--headless=new', '--no-first-run', '--disable-background-networking',
  '--enable-unsafe-swiftshader', '--use-angle=swiftshader', '--remote-debugging-port=0',
  '--window-size=1440,1100',
  '--user-data-dir=' + profile, 'about:blank'], {stdio:'ignore', windowsHide:true});
const wait = ms => new Promise(r => setTimeout(r, ms));
async function until(fn) {
  for (let i=0; i<160; ++i) {
    const value = await fn();
    if (value) return value;
    await wait(100);
  }
  throw new Error('Browser did not reach expected state');
}
let ws;
try {
  const port = await until(async () => {
    try { return (await readFile(join(profile, 'DevToolsActivePort'), 'utf8')).split('\n')[0]; }
    catch (e) { if (e.code !== 'ENOENT') throw e; }
  });
  const tabs = await (await fetch('http://localhost:'+port+'/json/list')).json();
  ws = new WebSocket(tabs.find(t=>t.type==='page').webSocketDebuggerUrl);
  await new Promise((ok, fail) => {ws.onopen=ok; ws.onerror=fail;});
  const calls = new Map(), errors = [];
  let id = 0;
  ws.onmessage = event => {
    const data = JSON.parse(event.data);
    if (data.method==='Runtime.exceptionThrown') errors.push(data.params.exceptionDetails.text);
    if (data.id) {
      const [ok, fail] = calls.get(data.id);
      calls.delete(data.id);
      if (data.error) fail(new Error(data.error.message)); else ok(data.result);
    }
  };
  const send = (method, params={}) => new Promise((ok, fail) => {
    calls.set(++id,[ok,fail]);
    ws.send(JSON.stringify({id,method,params}));
  });
  const js = async expression => {
    const result = await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});
    if (result.exceptionDetails) throw new Error(result.exceptionDetails.exception?.description ??
                                                 result.exceptionDetails.text);
    return result.result.value;
  };
  await send('Runtime.enable');
  await send('Page.navigate',{url:pathToFileURL(resolve(html)).href});
  await until(() => js("document.querySelector('#time')?.textContent"));
  const slider = "document.querySelector('#clock')";
  const time = "document.querySelector('#time').textContent";
  const plot = "document.querySelector('.plotly-graph-div')";
  const panel = "document.querySelector('#details').textContent";
  const rendered = () => js("Number("+plot+".dataset.frame)");
  const max = Number(await js(slider + ".max"));
  assert.equal(max+1,replay.keys.length);
  const seek = k => js(slider+".value="+k+";"+slider+".dispatchEvent(new Event('input'))");
  const expectKey = k => until(async () => (await rendered()) === replay.keys[k]);
  const click = id => js("document.querySelector('#"+id+"').click()");
  await click('next'); await expectKey(1);
  await click('prev'); await expectKey(0);
  // Exercise real mouse interaction on the WebGL scene.
  const camera = () => js("JSON.stringify("+plot+"._fullLayout.scene._scene.getCamera())");
  await js(plot+".scrollIntoView({block:'center'})");
  const rect = await js("(()=>{const canvas="+plot+
    ".querySelector('canvas');const r=canvas.getBoundingClientRect();return [r.x+r.width*.45,r.y+r.height*.55]})()");
  const [x,y] = rect;
  const hit = await js("document.elementFromPoint("+x+","+y+")?.outerHTML.slice(0,250)");
  assert(await js(plot+".contains(document.elementFromPoint("+x+","+y+"))"), hit);
  const oldCamera = await camera();
  await js("window.replayMouse=[]; for (const type of ['mousedown','mousemove','mouseup','wheel']) "+
           "document.addEventListener(type,e=>{replayMouse.push([type,e.target.tagName,e.buttons,e.clientX,e.clientY]);},true)");
  await send('Input.dispatchMouseEvent',{type:'mouseMoved',x,y});
  await send('Input.dispatchMouseEvent',{type:'mousePressed',x,y,button:'left',buttons:1,clickCount:1});
  for (let k=1;k<=5;k++) {
    await send('Input.dispatchMouseEvent',{type:'mouseMoved',x:x+k*10,y:y+k*4,buttons:1});
    await js("new Promise(ok=>requestAnimationFrame(ok))");
  }
  await send('Input.dispatchMouseEvent',{type:'mouseReleased',x:x+50,y:y+20,button:'left',clickCount:1});
  try { await until(async () => await camera() !== oldCamera); }
  catch (error) { throw new Error("Camera drag did not change the scene at "+x+","+y+" on "+hit+
    " events="+JSON.stringify(await js("replayMouse"))+" mode="+
    await js(plot+"._fullLayout.scene.dragmode"),{cause:error}); }
  const rotated = await camera();
  await send('Input.dispatchMouseEvent',{type:'mouseWheel',x,y,deltaX:0,deltaY:-120});
  await until(async () => await camera() !== rotated);
  // Wheel interpolation must settle before taking a reference camera.
  let previous, stable = 0;
  const zoomed = await until(async () => {
    const current = await camera();
    stable = current === previous ? stable+1 : 0;
    previous = current;
    return stable >= 3 && current;
  });
  const markers = async k => {
    await seek(k); await expectKey(k);
    const stamp = replay.frames[replay.keys[k]].time;
    const expected = replay.events.filter(e => e.time <= stamp);
    for (const [axis,key] of ['x','y','z'].entries())
      assert.deepEqual(await js(plot+".data.at(-1)."+key),expected.map(e=>e.position[axis]));
    assert.deepEqual(await js(plot+".data.at(-1).marker.symbol"),expected.map(e=>e.plotly_marker));
    assert.deepEqual(await js(plot+".data.at(-1).marker.color"),expected.map(e=>e.color));
    const actual = await js(plot+".data.at(-3).text");
    const entities = replay.frames[replay.keys[k]].entities;
    assert.deepEqual(actual,entities.filter(e=>!e.captured).map(e=>e.name));
    if (entities.some(e=>e.captured)) assert.match(await js(panel),/captured \/ inactive/);
  };
  for (const event of replay.events) {
    const k = replay.keys.findIndex(i => replay.frames[i].time === event.time);
    if (k>0) await markers(k-1);
    await markers(k);
  }
  await markers(max);
  const terminal = await js(time);
  assert(terminal.startsWith(replay.frames.at(-1).time.toFixed(3)+' s'));
  // Send many scrubs in one turn and verify that the last request wins.
  const lastScrub = Math.min(2,max);
  await js("for(const k of [0,"+max+",1,"+max+",0,"+lastScrub+"]){"+slider+
           ".value=k;"+slider+".dispatchEvent(new Event('input'))}");
  await expectKey(lastScrub);
  assert.match(await js(panel),/normalized policy action/);
  await seek(0); await expectKey(0);
  for (const speed of ['.5','1','2','4']) {
    await js("document.querySelector('#rate').value='"+speed+
             "';document.querySelector('#rate').dispatchEvent(new Event('change'))");
    await click('restart');
    await until(async () => (await rendered()) > 0);
    await click('pause');
    // Flush the latest requested frame before testing the paused state.
    await js("new Promise(ok=>requestAnimationFrame(()=>requestAnimationFrame(ok)))");
  }
  await seek(0); await expectKey(0);
  await click('play');
  await until(async () => (await rendered()) > 0);
  await click('pause');
  await wait(200);
  const paused = await js(time);
  await wait(250);
  assert.equal(await js(time),paused);
  await click('play');
  await until(async () => (await js(time))===terminal);
  await click('play');
  await until(async () => (await rendered()) < replay.frames.length-1);
  await click('pause');
  await markers(max);
  const finalCamera = JSON.parse(await camera());
  const expectedCamera = JSON.parse(zoomed);
  for (const key of ['eye','up','center'])
    for (const axis of ['x','y','z'])
      assert(Math.abs(finalCamera[key][axis]-expectedCamera[key][axis]) < 1e-9,
        JSON.stringify({rotated,zoomed,finalCamera,key,axis}));
  assert.deepEqual(errors,[]);
  if (screenshot) {
    await js(`Plotly.relayout(${plot}, {'scene.camera.eye':{x:2.2,y:2.2,z:1.7}}).then(() => true)`);
    await js("window.scrollTo(0,0)");
    const captured = await send('Page.captureScreenshot', {format:'png'});
    await writeFile(screenshot, Buffer.from(captured.data,'base64'), {flag:'wx'});
  }
  console.log(JSON.stringify({frames:replay.frames.length, keyframes:max+1, terminal,
    markerEvents:replay.events.length, cameraPreserved:true, rotated:true, zoomed:true,
    rates:[.5,1,2,4], rapidScrub:true, pausedResume:true, errors}));
} finally {
  ws?.close();
  if (child.exitCode === null) child.kill();
  await wait(500);
  if (dirname(profile) === parent) await rm(profile,{recursive:true,force:true,maxRetries:5,retryDelay:200});
}
