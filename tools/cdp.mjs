// Minimal Chrome DevTools Protocol driver for visual checks.
// Usage: node tools/cdp.mjs <url> <width> <height> <outfile> [hash]
import { spawn } from 'node:child_process';
import fs from 'node:fs';

const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const [url, width = '390', height = '900', out = '/tmp/cdp.png', hash = '#games'] = process.argv.slice(2);
const port = 9400 + (process.pid % 300);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const chrome = spawn(
  CHROME,
  [
    '--headless=new',
    '--disable-gpu',
    '--hide-scrollbars',
    '--no-first-run',
    '--no-default-browser-check',
    `--remote-debugging-port=${port}`,
    `--user-data-dir=/tmp/cdp-profile-${process.pid}`,
    `--window-size=${width},${height}`,
    'about:blank',
  ],
  { stdio: 'ignore' }
);

async function findTarget() {
  for (let i = 0; i < 100; i += 1) {
    try {
      const res = await fetch(`http://127.0.0.1:${port}/json/list`);
      const targets = await res.json();
      const page = targets.find((t) => t.type === 'page');
      if (page?.webSocketDebuggerUrl) return page.webSocketDebuggerUrl;
    } catch {
      /* not up yet */
    }
    await sleep(120);
  }
  throw new Error('Chrome DevTools target not found');
}

async function main() {
  const ws = new WebSocket(await findTarget());
  await new Promise((resolve, reject) => {
    ws.addEventListener('open', resolve, { once: true });
    ws.addEventListener('error', reject, { once: true });
  });

  let nextId = 0;
  const pending = new Map();
  ws.addEventListener('message', (event) => {
    const data = JSON.parse(event.data);
    if (data.id && pending.has(data.id)) {
      const { resolve, reject } = pending.get(data.id);
      pending.delete(data.id);
      if (data.error) reject(new Error(JSON.stringify(data.error)));
      else resolve(data.result);
    }
  });
  const send = (method, params = {}) => {
    const id = ++nextId;
    ws.send(JSON.stringify({ id, method, params }));
    return new Promise((resolve, reject) => {
      pending.set(id, { resolve, reject });
      setTimeout(() => {
        if (pending.has(id)) {
          pending.delete(id);
          reject(new Error(`timeout ${method}`));
        }
      }, 30000);
    });
  };

  await send('Page.enable');
  await send('Emulation.setDeviceMetricsOverride', {
    width: Number(width),
    height: Number(height),
    deviceScaleFactor: 1,
    mobile: false,
  });
  await send('Page.navigate', { url: `${url}${hash}` });
  await sleep(4000);

  if (process.env.THEME) {
    await send('Runtime.evaluate', {
      expression: `localStorage.setItem('nfl_theme', '${process.env.THEME}'); location.reload();`,
    });
    await sleep(3500);
  }

  const info = await send('Runtime.evaluate', {
    expression: `JSON.stringify({
      innerWidth: window.innerWidth,
      scrollWidth: document.documentElement.scrollWidth,
      overflow: [...document.querySelectorAll('body *')]
        .filter((n) => n.getBoundingClientRect().right > window.innerWidth + 1)
        .slice(0, 15)
        .map((n) => ({ tag: n.tagName, cls: String(n.className).slice(0, 50), right: Math.round(n.getBoundingClientRect().right), w: Math.round(n.getBoundingClientRect().width) }))
    })`,
    returnByValue: true,
  });
  console.log(info.result.value);

  const shot = await send('Page.captureScreenshot', {
    format: 'png',
    captureBeyondViewport: process.env.FULL === '1',
    fromSurface: true,
  });
  fs.writeFileSync(out, Buffer.from(shot.data, 'base64'));
  console.log('wrote', out);
  ws.close();
}

try {
  await main();
} finally {
  chrome.kill('SIGKILL');
  fs.rmSync(`/tmp/cdp-profile-${process.pid}`, { recursive: true, force: true });
}
