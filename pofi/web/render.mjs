// Renders a Pofi 3D plan (JSON written by pofi/render3d.py) to H.264 segments with headless Chromium.
//
//   node render.mjs --plan plan.json --out DIR [--workers 3] [--from 0] [--to N]
//   node render.mjs --plan plan.json --thumb out.jpg
//   node render.mjs --plan plan.json --shots DIR --list 0,120,450      (single frames for checking)
//   node render.mjs --check                                             (can the browser start here?)
//
// Each worker is its own browser (WebGL runs in one GPU process per browser), renders a contiguous
// frame range and pipes JPEG screenshots into ffmpeg. Prints one JSON line {"segments":[...]} at the end.
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { existsSync, mkdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..', '..');
const ARGS = ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist',
  '--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--font-render-hinting=none'];

function opts() {
  const o = { workers: 1, from: 0, to: null };
  const a = process.argv.slice(2);
  for (let i = 0; i < a.length; i += 2) o[a[i].replace(/^--/, '')] = a[i + 1];
  o.workers = Math.max(1, Number(o.workers)); o.from = Number(o.from); if (o.to !== null) o.to = Number(o.to);
  return o;
}

const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json',
  '.ttf': 'font/ttf', '.webp': 'image/webp', '.png': 'image/png' };
// POFI_EMOJI_DIR comes from pofi/render3d.py, so a moved cache (POFI_CACHE) is served too
const EMOJI_DIR = process.env.POFI_EMOJI_DIR || path.join(ROOT, '.cache', 'emoji3d');
const MOUNTS = [['/three/', path.join(HERE, 'node_modules', 'three')], ['/lib/', path.join(HERE, 'node_modules', 'polygon-clipping', 'dist')],
  ['/fonts/', path.join(ROOT, 'assets', 'fonts')], ['/emoji/', EMOJI_DIR], ['/', HERE]];
const FRAME_TIMEOUT_MS = 120000;   // a stalled browser must fail fast so the caller can fall back to 2D

function withTimeout(p, what) {
  let timer;
  return Promise.race([p, new Promise((_, rej) => { timer = setTimeout(() => rej(new Error(`${what}: ${FRAME_TIMEOUT_MS / 1000} sn içinde yanıt yok`)), FRAME_TIMEOUT_MS); })])
    .finally(() => clearTimeout(timer));
}

function serve(port = 0) {
  const server = createServer(async (req, res) => {
    let url = decodeURIComponent(new URL(req.url, 'http://x').pathname);
    if (url === '/') url = '/index.html';
    for (const [prefix, dir] of MOUNTS) {
      if (!url.startsWith(prefix)) continue;
      const file = path.resolve(dir, '.' + url.slice(prefix.length - 1));
      if (!file.startsWith(dir)) break;
      try {
        const body = await readFile(file);
        res.writeHead(200, { 'Content-Type': TYPES[path.extname(file)] || 'application/octet-stream' });
        res.end(body);
      } catch { res.writeHead(404); res.end(); }
      return;
    }
    res.writeHead(404); res.end();
  });
  return new Promise(r => server.listen(port, '127.0.0.1', () => r(server)));
}

async function openPage(port, plan) {
  const browser = await chromium.launch({ args: ARGS });
  const page = await browser.newPage({ viewport: { width: plan.width, height: plan.height }, deviceScaleFactor: 1 });
  const errors = [];
  page.on('pageerror', e => errors.push(String(e && e.stack || e)));
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  await page.goto(`http://127.0.0.1:${port}/`);
  await page.waitForFunction(() => !!window.pofi, null, { timeout: 60000 });
  await page.evaluate(p => window.pofi.init(p), plan);
  if (errors.length) throw new Error('sayfa hatası: ' + errors.join('\n'));
  const cdp = await page.context().newCDPSession(page);
  return { browser, page, cdp, errors };
}

async function shot(cdp, quality = 93) {
  const r = await cdp.send('Page.captureScreenshot', { format: 'jpeg', quality, optimizeForSpeed: true });
  return Buffer.from(r.data, 'base64');
}

async function renderRange(port, plan, a, b, out, onFrame) {
  const { browser, page, cdp, errors } = await openPage(port, plan);
  const ff = spawn('ffmpeg', ['-v', 'error', '-y', '-f', 'image2pipe', '-vcodec', 'mjpeg', '-framerate', String(plan.fps),
    '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-tune', 'animation', '-pix_fmt', 'yuv420p',
    '-g', String(plan.fps * 2), out], { stdio: ['pipe', 'inherit', 'inherit'] });
  const done = new Promise((res, rej) => ff.on('close', c => (c === 0 ? res() : rej(new Error(`ffmpeg ${out} çıkış kodu ${c}`)))));
  done.catch(() => {});   // awaited below; a page error must not be masked by ffmpeg exiting on empty input
  try {
    for (let fi = a; fi < b; fi++) {
      await withTimeout(page.evaluate(i => window.pofi.renderFrame(i), fi), `kare ${fi}`);
      if (errors.length) throw new Error('sayfa hatası: ' + errors.join('\n'));
      const jpg = await withTimeout(shot(cdp), `kare ${fi} görüntüsü`);
      if (!ff.stdin.write(jpg)) await new Promise(r => ff.stdin.once('drain', r));
      onFrame();
    }
  } finally {
    ff.stdin.end();
    await browser.close();
  }
  await done;
  return out;
}

async function main() {
  const o = opts();
  if (o.serve) { const sv = await serve(Number(o.serve)); console.log('http://127.0.0.1:' + sv.address().port); return; }
  if ('check' in o) {
    const browser = await chromium.launch({ args: ARGS });
    const page = await browser.newPage();
    const ok = await page.evaluate(() => !!document.createElement('canvas').getContext('webgl2'));
    await browser.close();
    console.log(JSON.stringify({ ok }));
    if (!ok) process.exitCode = 1;
    return;
  }
  const plan = JSON.parse(await readFile(o.plan, 'utf8'));
  const server = await serve();
  const port = server.address().port;
  try {
    if (o.thumb) {
      const { browser, page, cdp } = await openPage(port, plan);
      await page.evaluate(() => window.pofi.renderThumb());
      const { writeFile } = await import('node:fs/promises');
      await writeFile(o.thumb, await shot(cdp, 95));
      await browser.close();
      console.log(JSON.stringify({ thumb: o.thumb }));
      return;
    }
    if (o.shots) {
      if (!existsSync(o.shots)) mkdirSync(o.shots, { recursive: true });
      const { browser, page, cdp, errors } = await openPage(port, plan);
      const { writeFile } = await import('node:fs/promises');
      const t0 = Date.now();
      const list = String(o.list).split(',').map(Number);
      for (const fi of list) {
        await page.evaluate(i => window.pofi.renderFrame(i), fi);
        if (errors.length) throw new Error('sayfa hatası: ' + errors.join('\n'));
        await writeFile(path.join(o.shots, `f${String(fi).padStart(6, '0')}.jpg`), await shot(cdp, 92));
      }
      await browser.close();
      console.log(JSON.stringify({ shots: list.length, ms_per_frame: (Date.now() - t0) / list.length }));
      return;
    }
    if (!existsSync(o.out)) mkdirSync(o.out, { recursive: true });
    const total = plan.frames;
    const to = o.to === null ? total : Math.min(total, o.to);
    const n = to - o.from;
    const per = Math.ceil(n / o.workers);
    let doneFrames = 0, lastPct = -1;
    const t0 = Date.now();
    const tick = () => {
      doneFrames++;
      const pct = Math.floor(100 * doneFrames / n);
      if (pct !== lastPct && pct % 5 === 0) {
        lastPct = pct;
        const el = (Date.now() - t0) / 1000;
        const eta = doneFrames ? el / doneFrames * (n - doneFrames) : 0;
        process.stderr.write(`[3d] %${pct}  ${doneFrames}/${n} kare  ${(1000 * el / doneFrames).toFixed(0)} ms/kare  kalan ~${Math.round(eta / 60)} dk\n`);
      }
    };
    const jobs = [];
    for (let k = 0; k < o.workers; k++) {
      const a = o.from + k * per, b = Math.min(to, a + per);
      if (a >= b) break;
      jobs.push(renderRange(port, plan, a, b, path.join(o.out, `seg3d_${String(k).padStart(3, '0')}.mp4`), tick));
    }
    const segments = await Promise.all(jobs);
    console.log(JSON.stringify({ segments, seconds: (Date.now() - t0) / 1000 }));
  } finally {
    server.close();
  }
}

main().catch(e => { console.error(e && e.stack || e); process.exit(1); });
