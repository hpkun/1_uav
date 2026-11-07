/* Exercise the offline Plotly player in a real Chromium browser via CDP. */
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {mkdtemp, readFile, rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join, dirname, resolve} from 'node:path';
import {pathToFileURL} from 'node:url';

const [browser, html] = process.argv.slice(2);
assert(browser && html, 'usage: node tests/check_interception_browser.mjs BROWSER REPLAY.html');
const replay = JSON.parse((await readFile(html, 'utf8')).match(/const DATA = (.*);\r?\n/)[1]);
const parent = resolve(tmpdir());
const profile = await mkdtemp(join(parent, 'interception-browser-'));
const child = spawn(browser, ['--headless=new', '--no-first-run', '--disable-background-networking',
  '--enable-unsafe-swiftshader', '--use-angle=swiftshader', '--remote-debugging-port=0',
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
  const original = await js(time);
  const max = await js(slider + ".max");
  assert.equal(Number(max)+1,replay.frames.length);
  await js("document.querySelector('#next').click()");
  assert.match(await js(time), /step 1 /);
  await js("document.querySelector('#prev').click()");
  assert.equal(await js(time),original);
  await js(`Plotly.relayout(${plot}, {'scene.camera.eye': {x:1.7,y:0.4,z:1.2}}).then(() => true)`);
  const camera = await js(`JSON.stringify(${plot}.layout.scene.camera)`);
  await js(`${slider}.value=${max}; ${slider}.dispatchEvent(new Event('input'))`);
  const terminal = await js(time);
  assert(terminal.startsWith(replay.frames.at(-1).time.toFixed(3)+' s'));
  assert.match(await js("document.querySelector('#details').textContent"), /Active evaders: 0/);
  assert.match(await js("document.querySelector('#details').textContent"), /capture · /);
  assert.match(await js("document.querySelector('#details').textContent"), /Dwell \(s\)/);
  await until(() => js(`${plot}.data.at(-1).x.length > 0`));
  const positions = replay.events.flatMap(e => e.positions);
  for (const [axis,key] of ['x','y','z'].entries())
    assert.deepEqual(await js(`${plot}.data.at(-1).${key}`),positions.map(p=>p[axis]));
  assert.equal(await js(`JSON.stringify(${plot}.layout.scene.camera)`),camera);
  await js("document.querySelector('#rate').value='4'; document.querySelector('#rate').dispatchEvent(new Event('change'))");
  await js("document.querySelector('#restart').click()");
  await wait(350);
  await js("document.querySelector('#pause').click()");
  const paused = await js(time);
  assert.notEqual(paused, original);
  assert.notEqual(paused, terminal);
  await wait(300);
  assert.equal(await js(time),paused);
  await js("document.querySelector('#play').click()");
  await until(async () => (await js(time))===terminal);
  assert.equal(await js(`JSON.stringify(${plot}.layout.scene.camera)`),camera);
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({frames:Number(max)+1, terminal, pause:paused, cameraPreserved:true,
    playbackControls:'play, pause, restart, seek, step both directions, 4x', errors}));
} finally {
  ws?.close();
  if (child.exitCode === null) child.kill();
  await wait(500);
  if (dirname(profile) === parent) await rm(profile,{recursive:true,force:true,maxRetries:5,retryDelay:200});
}
