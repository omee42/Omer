// Pofi 3D renderer: builds a small 3D world with three.js and draws one frame per call.
// The page is driven by render.mjs (headless Chromium); every animation is a pure function of time,
// so any worker can render any frame range and the chunks join seamlessly.
import * as THREE from 'three';
import { TTFLoader } from 'three/addons/loaders/TTFLoader.js';
import { Font } from 'three/addons/loaders/FontLoader.js';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';

// ------------------------------------------------------------------ helpers
const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const easeOutBack = (x, s = 1.70158) => { x = clamp(x) - 1; return x * x * ((s + 1) * x + s) + 1; };
const easeInOut = x => { x = clamp(x); return x * x * (3 - 2 * x); };
const easeOutCubic = x => 1 - Math.pow(1 - clamp(x), 3);
const prog = (t, a, d) => (d > 0 ? clamp((t - a) / d) : (t >= a ? 1 : 0));
const pop = (t, a, d = 0.45) => (t < a ? 0 : easeOutBack(prog(t, a, d), 2.2));
const wob = (t, ph = 0, amp = 0.05, sp = 1.3) => amp * Math.sin(t * sp + ph);
const bob = (t, ph = 0, amp = 1, sp = 2.2) => amp * Math.sin(t * sp + ph);
const hash = (n) => { const x = Math.sin(n * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); };
const col = c => new THREE.Color(c);
const mixHex = (a, b, k) => col(a).lerp(col(b), k);
const isEmojiChar = ch => /\p{Extended_Pictographic}|‍|️/u.test(ch);
const stripEmoji = s => [...String(s)].filter(c => !isEmojiChar(c)).join('').replace(/\s{2,}/g, ' ').trim();
const trUpper = s => String(s).toLocaleUpperCase('tr-TR');
// Contrast maths on sRGB hex strings, the same formulas as pofi/gfx.py. THREE.Color holds linear values,
// so it must not be used for these checks.
const hexRgb = c => { const h = String(c).replace('#', ''); return [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16)); };
const rgbHex = a => '#' + a.map(v => Math.round(clamp(v, 0, 255)).toString(16).padStart(2, '0')).join('');
const srgbLum = c => { const f = v => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
  const [r, g, b] = hexRgb(c); return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b); };
const contrast = (a, b) => { const [x, y] = [srgbLum(a), srgbLum(b)].sort((p, q) => q - p); return (x + 0.05) / (y + 0.05); };
const darken = (c, k) => rgbHex(hexRgb(c).map(v => v * (1 - k)));
const inkOn = (bg, dark) => (contrast('#ffffff', bg) >= 3 ? '#ffffff' : dark);          // gfx.ink
const readableOnWhite = (c, ratio = 3) => { for (let i = 0; i < 12 && contrast(c, '#ffffff') < ratio; i++) c = darken(c, 0.12); return c; };

let P, W, H, S, portrait, renderer, scene, camera, camRef, font3d, ui, irisEl;
const texCache = new Map();
const scenes = [];
let world, mascot, captionEl, captionText = null, chipEl;

// ------------------------------------------------------------------ layout (same regions as the 2D engine)
function layout() {
  if (portrait) {
    return { content: [40 * S, 190 * S, W - 80 * S, 1130 * S], mascot: [W / 2, H - 46 * S, 380 * S],
             caption: [W / 2, 1400 * S, W - 90 * S], chip: [W / 2, 95 * S] };   // caption clear of the owl's tufts
  }
  return { content: [445 * S, 40 * S, W - 480 * S, 850 * S], mascot: [232 * S, H - 36 * S, 400 * S],
           caption: [445 * S + (W - 480 * S) / 2, H - 98 * S, 1340 * S], chip: [215 * S, 70 * S] };
}
let L;

const _v = new THREE.Vector3();
function at(px, py, z = 0) { // screen pixel -> world point on plane z (reference camera)
  _v.set(px / W * 2 - 1, -(py / H * 2 - 1), 0.5).unproject(camRef);
  const dir = _v.sub(camRef.position).normalize();
  const k = (z - camRef.position.z) / dir.z;
  return camRef.position.clone().add(dir.multiplyScalar(k));
}
function upp(z = 0) { // world units per pixel at plane z
  const d = camRef.position.z - z;
  return 2 * d * Math.tan(THREE.MathUtils.degToRad(camRef.fov / 2)) / H;
}
function placeAt(obj, px, py, z = 0) { obj.position.copy(at(px, py, z)); }

// ------------------------------------------------------------------ materials
const matCache = new Map();
// Glossy toy plastic. A clearcoat layer would look a little richer but doubles the shading cost on the
// software GPU, so the gloss comes from a low roughness instead.
function plastic(color, opts = {}) {
  const key = `p|${color}|${JSON.stringify(opts)}`;
  if (!matCache.has(key)) {
    const { clearcoat, clearcoatRoughness, ...rest } = opts;
    const gloss = clearcoat === undefined ? 0.6 : clearcoat;
    if (typeof rest.emissive === 'string') rest.emissive = col(rest.emissive);
    matCache.set(key, new THREE.MeshStandardMaterial({ color: col(color), roughness: 0.42 - 0.16 * gloss, metalness: 0, ...rest }));
  }
  return matCache.get(key);
}
function matte(color, opts = {}) {
  const key = `m|${color}|${JSON.stringify(opts)}`;
  if (!matCache.has(key)) matCache.set(key, new THREE.MeshStandardMaterial({ color: col(color), roughness: 0.85, ...opts }));
  return matCache.get(key);
}
function shadowy(mesh, cast = true, receive = false) { mesh.castShadow = cast; mesh.receiveShadow = receive; return mesh; }
// three compiles an opaque material with alpha forced to 1 and does not recompile when `transparent` flips,
// so a fade would work or not depending on the frame that first drew the object. Recompile on every flip:
// the program then depends only on the current state, and any frame renders the same on a cold page.
function setOpacity(m, op) {
  if (m.userData.baseTransparent === undefined) m.userData.baseTransparent = m.transparent;
  const tr = op < 0.999 || m.userData.baseTransparent;
  if (m.transparent !== tr) { m.transparent = tr; m.needsUpdate = true; }
  m.opacity = op;
}

// ------------------------------------------------------------------ textures
function loadTex(url) {
  if (texCache.has(url)) return texCache.get(url);
  const p = new Promise((res, rej) => new THREE.TextureLoader().load(url, t => {
    t.colorSpace = THREE.SRGBColorSpace; res(t); }, undefined, rej));
  texCache.set(url, p);
  return p;
}
const texReady = new Map();
async function preloadTextures(urls) {
  for (const u of urls) texReady.set(u, await loadTex(u));
}

function canvasTex(w, h, draw) {
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  draw(c.getContext('2d'), w, h);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function turkishFlagTexture(w = 1200) {
  // statutory proportions on a 12 x 8 grid (as in common reference SVGs)
  const h = Math.round(w * 8 / 12), u = w / 12;
  return canvasTex(w, h, (g) => {
    g.fillStyle = '#E30A17'; g.fillRect(0, 0, w, h);
    g.fillStyle = '#fff'; g.beginPath(); g.arc(4.25 * u, 4 * u, 2 * u, 0, Math.PI * 2); g.fill();
    g.fillStyle = '#E30A17'; g.beginPath(); g.arc(4.75 * u, 4 * u, 1.6 * u, 0, Math.PI * 2); g.fill();
    g.fillStyle = '#fff'; g.beginPath();
    const R = u, r = R * Math.cos(2 * Math.PI / 5) / Math.cos(Math.PI / 5);
    for (let i = 0; i < 10; i++) {
      const a = Math.PI + Math.PI * i / 5, rr = i % 2 ? r : R;
      const x = 6.8333 * u + rr * Math.cos(a), y = 4 * u + rr * Math.sin(a);
      i ? g.lineTo(x, y) : g.moveTo(x, y);
    }
    g.closePath(); g.fill();
  });
}

function radialTex(inner, outer, size = 256) {
  return canvasTex(size, size, (g, w, h) => {
    const gr = g.createRadialGradient(w / 2, h / 2, 0, w / 2, h / 2, w / 2);
    gr.addColorStop(0, inner); gr.addColorStop(1, outer);
    g.fillStyle = gr; g.fillRect(0, 0, w, h);
  });
}

// ------------------------------------------------------------------ 3D text
const textGeoCache = new Map();
function glyphAdvance(ch, size) {
  const g = font3d.data.glyphs[ch] || font3d.data.glyphs['?'];
  return (g ? g.ha : font3d.data.resolution * 0.5) * size / font3d.data.resolution;
}
function measure3d(str, size) { let w = 0; for (const ch of str) w += glyphAdvance(ch, size); return w; }
function wrap3d(text, size, maxW) {
  const words = String(text).split(/\s+/).filter(Boolean);
  const lines = []; let cur = '';
  for (const w of words) {
    const cand = cur ? cur + ' ' + w : w;
    if (cur && measure3d(cand, size) > maxW) { lines.push(cur); cur = w; } else cur = cand;
  }
  if (cur) lines.push(cur);
  return lines;
}
// Glyph outlines -> shapes with the font's non-zero fill rule. A contour inside another one is a hole when
// it runs the other way round, and more of the same solid when it runs the same way (overlapping parts such
// as the cedilla of Ş). three's own TextGeometry decides by the first contour's direction, which turns this
// font's "D" (its hole comes first) into a filled letter with a second letter on top.
function pointInPoly(p, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const a = poly[i], b = poly[j];
    if ((a.y > p.y) !== (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x) inside = !inside;
  }
  return inside;
}
function glyphShapes(ch, size) {
  const data = font3d.data, glyph = data.glyphs[ch] || data.glyphs['?'];
  if (!glyph || !glyph.o) return [];
  const k = size / data.resolution, o = glyph.o.split(' ');
  const sp = new THREE.ShapePath();
  for (let i = 0; i < o.length;) {
    const a = o[i++];
    if (a === 'm') sp.moveTo(o[i++] * k, o[i++] * k);
    else if (a === 'l') sp.lineTo(o[i++] * k, o[i++] * k);
    else if (a === 'q') { const x = o[i++] * k, y = o[i++] * k, x1 = o[i++] * k, y1 = o[i++] * k; sp.quadraticCurveTo(x1, y1, x, y); }
    else if (a === 'b') { const x = o[i++] * k, y = o[i++] * k, x1 = o[i++] * k, y1 = o[i++] * k, x2 = o[i++] * k, y2 = o[i++] * k; sp.bezierCurveTo(x1, y1, x2, y2, x, y); }
  }
  const polys = sp.subPaths.map(p => p.getPoints(5)).filter(p => p.length > 2);
  const signed = polys.map(p => THREE.ShapeUtils.area(p)), size_ = signed.map(Math.abs);
  const order = polys.map((_, i) => i).sort((a, b) => size_[b] - size_[a]);
  const kind = new Array(polys.length), owner = new Array(polys.length);   // owner: index of the solid a hole cuts
  const shapes = new Map();
  for (const i of order) {
    let parent = -1;
    for (const j of order) {
      if (j === i || size_[j] <= size_[i] || kind[j] === undefined) continue;
      if (pointInPoly(polys[i][0], polys[j]) && (parent < 0 || size_[j] < size_[parent])) parent = j;
    }
    if (parent < 0) kind[i] = 'solid';
    else if (Math.sign(signed[i]) !== Math.sign(signed[parent])) kind[i] = kind[parent] === 'solid' ? 'hole' : 'solid';
    else kind[i] = kind[parent];
    if (kind[i] === 'solid') shapes.set(i, new THREE.Shape(polys[i]));
    else { owner[i] = kind[parent] === 'solid' ? parent : owner[parent]; shapes.get(owner[i]).holes.push(new THREE.Path(polys[i])); }
  }
  // Most letters of this font are built from overlapping strokes (E is five boxes). Extruded one by one they
  // leave seams where the parts cross, so merge them into one outline: union of the solids minus the holes.
  const solids = polys.filter((_, i) => kind[i] === 'solid'), holes = polys.filter((_, i) => kind[i] === 'hole');
  if (solids.length > 1 && window.polygonClipping) {
    try {
      const ring = p => [p.map(v => [v.x, v.y])];
      let mp = window.polygonClipping.union(...solids.map(ring));
      if (holes.length) mp = window.polygonClipping.difference(mp, ...holes.map(ring));
      const toV = r => r.slice(0, -1).map(([x, y]) => new THREE.Vector2(x, y));
      return mp.map(([outer, ...inner]) => { const sh = new THREE.Shape(toV(outer)); sh.holes = inner.map(r => new THREE.Path(toV(r))); return sh; });
    } catch (e) { /* fall back to the overlapping shapes */ }
  }
  return [...shapes.values()];
}
function charGeo(ch, size, depth) {
  const key = `${ch}|${size.toFixed(4)}|${depth.toFixed(4)}`;
  if (!textGeoCache.has(key)) {
    const g = new THREE.ExtrudeGeometry(glyphShapes(ch, size), { depth, bevelEnabled: true,
      bevelThickness: depth * 0.28, bevelSize: size * 0.035, bevelSegments: 2 });
    textGeoCache.set(key, g);
  }
  return textGeoCache.get(key);
}
// Text built from per-letter meshes so letters can bounce in one by one.
function text3d(text, { heightPx, maxWpx = 1e9, maxHpx = 1e9, color = '#ffffff', side = null, z = 0, glow = 0.16 }) {
  const u = upp(z);
  let size = heightPx * u;
  let lines;
  for (let i = 0; i < 30; i++) {
    lines = wrap3d(text, size, maxWpx * u);
    const widest = Math.max(...lines.map(l => measure3d(l, size)));
    const tall = lines.length * size * 1.25;
    if ((widest <= maxWpx * u && tall <= maxHpx * u) || size < 12 * u) break;
    size *= 0.92;
  }
  const depth = size * 0.32;
  // a little self-light keeps the letter faces as bright as the colour asks for, whatever way they turn
  const front = plastic(color, { roughness: 0.3, clearcoat: 0.9, emissive: color, emissiveIntensity: glow });
  const sideMat = plastic(side || '#' + mixHex(color, '#000000', 0.35).getHexString(), { roughness: 0.5 });
  const group = new THREE.Group();
  const letters = [];
  lines.forEach((line, li) => {
    const lw = measure3d(line, size);
    let x = -lw / 2;
    const y = ((lines.length - 1) / 2 - li) * size * 1.25 - size * 0.36;
    for (const ch of line) {
      const adv = glyphAdvance(ch, size);
      if (ch !== ' ') {
        const m = shadowy(new THREE.Mesh(charGeo(ch, size, depth), [front, sideMat]));
        const holder = new THREE.Group();
        holder.position.set(x, y, -depth / 2);
        holder.add(m);
        group.add(holder);
        letters.push({ holder, base: holder.position.clone() });
      }
      x += adv;
    }
  });
  group.userData = { letters, size, lines };
  return group;
}
function animateLetters(group, t, start, { stagger = 0.045, dur = 0.5, wave = 0.0 } = {}) {
  const { letters, size } = group.userData;
  letters.forEach((l, i) => {
    const k = pop(t, start + i * stagger, dur);
    l.holder.scale.setScalar(Math.max(0.001, k));
    l.holder.position.y = l.base.y + (1 - Math.min(1, k)) * size * 1.2 + (wave ? Math.sin(t * 4 - i * 0.5) * size * wave : 0);
  });
}

// ------------------------------------------------------------------ visuals
function emojiUrl(e) { return P.emoji[e]; }
function visualKey(v) { return typeof v === 'string' ? 'e:' + v : JSON.stringify(v); }

const SHAPES = {
  'daire': s => { s.absarc(0, 0, 0.5, 0, Math.PI * 2, false); },
  'kare': s => roundRect(s, -0.45, -0.45, 0.9, 0.9, 0.08),
  'dikdörtgen': s => roundRect(s, -0.5, -0.28, 1.0, 0.56, 0.06),
  'üçgen': s => { s.moveTo(0, 0.5); s.lineTo(0.52, -0.42); s.lineTo(-0.52, -0.42); s.closePath(); },
  'yıldız': s => starPath(s, 5, 0.52, 0.22),
  'kalp': s => { s.moveTo(0, -0.45); s.bezierCurveTo(-0.6, -0.05, -0.55, 0.45, -0.25, 0.45); s.bezierCurveTo(-0.08, 0.45, 0, 0.32, 0, 0.22);
                 s.bezierCurveTo(0, 0.32, 0.08, 0.45, 0.25, 0.45); s.bezierCurveTo(0.55, 0.45, 0.6, -0.05, 0, -0.45); },
  'oval': s => { s.absellipse(0, 0, 0.52, 0.34, 0, Math.PI * 2, false, 0); },
  'baklava': s => { s.moveTo(0, 0.52); s.lineTo(0.36, 0); s.lineTo(0, -0.52); s.lineTo(-0.36, 0); s.closePath(); },
  'beşgen': s => polyPath(s, 5, 0.52),
  'altıgen': s => polyPath(s, 6, 0.5, 0),
  'yarım daire': s => { s.moveTo(-0.5, -0.2); s.absarc(0, -0.2, 0.5, Math.PI, 0, true); s.lineTo(-0.5, -0.2); },
  'hilal': s => { s.absarc(0, 0, 0.5, Math.PI * 0.32, Math.PI * 1.68, false);
                  s.absarc(0.17, 0, 0.4, Math.PI * 1.75, Math.PI * 0.25, true); },
};
const SHAPE_ALIASES = { circle: 'daire', square: 'kare', triangle: 'üçgen', rectangle: 'dikdörtgen', star: 'yıldız', heart: 'kalp',
  ellipse: 'oval', diamond: 'baklava', pentagon: 'beşgen', hexagon: 'altıgen', semicircle: 'yarım daire', crescent: 'hilal' };
function roundRect(s, x, y, w, h, r) {
  s.moveTo(x + r, y); s.lineTo(x + w - r, y); s.quadraticCurveTo(x + w, y, x + w, y + r); s.lineTo(x + w, y + h - r);
  s.quadraticCurveTo(x + w, y + h, x + w - r, y + h); s.lineTo(x + r, y + h); s.quadraticCurveTo(x, y + h, x, y + h - r);
  s.lineTo(x, y + r); s.quadraticCurveTo(x, y, x + r, y);
}
function starPath(s, n, ro, ri) {
  for (let i = 0; i < n * 2; i++) {
    const a = Math.PI / 2 + Math.PI * i / n, r = i % 2 ? ri : ro;
    i ? s.lineTo(r * Math.cos(a), r * Math.sin(a)) : s.moveTo(r * Math.cos(a), r * Math.sin(a));
  }
  s.closePath();
}
function polyPath(s, n, r, rot = Math.PI / 2) {
  for (let i = 0; i < n; i++) {
    const a = rot + Math.PI * 2 * i / n;
    i ? s.lineTo(r * Math.cos(a), r * Math.sin(a)) : s.moveTo(r * Math.cos(a), r * Math.sin(a));
  }
  s.closePath();
}
const shapeGeoCache = new Map();
function shapeGeo(kind) {
  kind = SHAPE_ALIASES[kind] || kind;
  if (!shapeGeoCache.has(kind)) {
    const s = new THREE.Shape(); (SHAPES[kind] || SHAPES['daire'])(s);
    const g = new THREE.ExtrudeGeometry(s, { depth: 0.16, bevelEnabled: true, bevelThickness: 0.05, bevelSize: 0.035,
      bevelSegments: 4, curveSegments: 48 });
    g.center();
    shapeGeoCache.set(kind, g);
  }
  return shapeGeoCache.get(kind);
}

let flagTex = null;
// A visual is a Group with unit height ~1 (scaled by caller); update(t) animates flags.
function makeVisual(v) {
  const g = new THREE.Group();
  g.userData.update = null;
  let aspect = 1;
  if (typeof v === 'string' || (v && v.emoji)) {
    const e = typeof v === 'string' ? v : v.emoji;
    const tex = texReady.get(emojiUrl(e));
    aspect = tex ? tex.image.height / tex.image.width : 1;
    const mat = new THREE.MeshBasicMaterial({ map: tex, transparent: true, alphaTest: 0.04, depthWrite: true, toneMapped: false });
    const m = new THREE.Mesh(new THREE.PlaneGeometry(1, aspect), mat);
    m.customDepthMaterial = new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking, map: tex, alphaTest: 0.5 });
    m.castShadow = true;
    g.add(m);
  } else if (v.shape) {
    const m = shadowy(new THREE.Mesh(shapeGeo(v.shape), plastic(v.color || '#FF5A5F', { clearcoat: 1, roughness: 0.28 })));
    m.scale.setScalar(0.92);
    g.add(m);
  } else if (v.flag) {
    flagTex = flagTex || turkishFlagTexture();
    const geo = new THREE.PlaneGeometry(1, 2 / 3, 40, 24);
    const base = geo.attributes.position.array.slice();
    const m = shadowy(new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ map: flagTex, roughness: 0.75, side: THREE.DoubleSide })));
    g.add(m);
    aspect = 2 / 3;
    g.userData.update = (t) => {
      const p = geo.attributes.position.array;
      for (let i = 0; i < p.length; i += 3) {
        const x = base[i] + 0.5, y = base[i + 1];
        p[i + 2] = Math.sin(x * 6 - t * 5) * 0.045 * x + Math.sin(y * 5 + t * 3) * 0.01 * x;
        p[i + 1] = y - Math.sin(x * 6 - t * 5) * 0.012 * x;
      }
      geo.attributes.position.needsUpdate = true;
      geo.computeVertexNormals();
    };
  } else if (v.symbol) {
    const tg = text3d(String(v.symbol), { heightPx: 300, color: v.color || '#7C6CF2', maxWpx: 1e6 });
    // fit into a unit box (like an emoji picture), centred, so "09.05" is as wide as one picture
    const box = new THREE.Box3().setFromObject(tg), c = box.getCenter(new THREE.Vector3());
    const bw = box.max.x - box.min.x || 1, bh = box.max.y - box.min.y || 1, k = 1 / Math.max(bw, bh);
    tg.position.set(-c.x * k, -c.y * k, -c.z * k); tg.scale.setScalar(k);
    g.add(tg);
    aspect = bh / bw;
  }
  g.userData.aspect = aspect;
  // own materials, so fading one picture (a wrong quiz answer, an uncounted item) never fades its twins
  if (!(typeof v === 'string' || (v && v.emoji))) {
    g.traverse(o => { if (o.material) o.material = Array.isArray(o.material) ? o.material.map(m => m.clone()) : o.material.clone(); });
  }
  return g;
}
// Place a visual with a given on-screen width in pixels.
function sizeVisual(g, widthPx, z = 0) {
  const a = g.userData.aspect || 1;
  const wU = widthPx * upp(z);
  const k = a >= 1 ? wU / a * a : wU;   // unit-width planes: scale by width
  g.userData.baseScale = k;
  g.scale.setScalar(k);
}

// ------------------------------------------------------------------ cards, coins, stars
const cardGeoCache = new Map();
function card3d(wPx, hPx, z = 0, color = '#ffffff', alpha = 1) {
  const u = upp(z), w = wPx * u, h = hPx * u;
  const key = `${w.toFixed(3)}|${h.toFixed(3)}`;
  if (!cardGeoCache.has(key)) cardGeoCache.set(key, new RoundedBoxGeometry(w, h, Math.min(w, h) * 0.08, 5, Math.min(w, h) * 0.12));
  const mat = new THREE.MeshStandardMaterial({ color: col(color), roughness: 0.5, transparent: alpha < 1, opacity: alpha,
    emissive: col(CARD_GLOW), emissiveIntensity: CARD_BASE });
  // Cards are drawn after the world and do not write depth; the pictures mounted on them are drawn after
  // the cards (see Scene.vis). A picture that swings or a card that tilts or spins therefore never slices
  // through the other, while both are still hidden correctly behind Pofi, who is drawn first.
  mat.depthWrite = false;
  const m = shadowy(new THREE.Mesh(cardGeoCache.get(key), mat), true, true);
  m.renderOrder = 1;
  m.userData.depth = Math.min(w, h) * 0.08;
  return m;
}
// Cards keep a faint warm self-light so they read as white paper, and glow in a colour when highlighted.
const CARD_GLOW = '#fff6ea', CARD_BASE = 0.22;
function cardGlow(card, color, k) {
  const m = card.material;
  if (!color || k <= 0) { m.emissive.set(CARD_GLOW); m.emissiveIntensity = CARD_BASE; return; }
  m.emissive.copy(col(CARD_GLOW).lerp(col(color), Math.min(1, k * 3))); m.emissiveIntensity = CARD_BASE + k;
}
function coin3d(text, color, sizePx, z = 0.2) {
  const u = upp(z), r = sizePx * u / 2;
  const g = new THREE.Group();
  const body = shadowy(new THREE.Mesh(new THREE.CylinderGeometry(r, r, r * 0.38, 48), [
    plastic(color, { roughness: 0.3 }), plastic(color, { roughness: 0.3 }), plastic(color, { roughness: 0.3 })]));
  body.rotation.x = Math.PI / 2;
  g.add(body);
  const rim = new THREE.Mesh(new THREE.TorusGeometry(r * 0.98, r * 0.08, 12, 48), plastic('#' + mixHex(color, '#ffffff', 0.45).getHexString()));
  rim.position.z = r * 0.17; g.add(rim);
  const ink = inkOn(color, P.theme.dark);
  const tx = text3d(String(text), { heightPx: sizePx * 0.62, color: ink, z, maxWpx: sizePx * 0.8 });
  tx.position.z = r * 0.2; g.add(tx);
  return g;
}
function star3d(color = '#FFC93C') {
  const m = shadowy(new THREE.Mesh(shapeGeo('yıldız'), plastic(color, { clearcoat: 1, roughness: 0.25, emissive: col(color), emissiveIntensity: 0.12 })));
  return m;
}

// ------------------------------------------------------------------ DOM overlay
function el(html, cls = '', style = {}) {
  const d = document.createElement('div');
  d.className = 'el ' + cls; d.innerHTML = html; Object.assign(d.style, style);
  d.style.opacity = '0';
  ui.appendChild(d);
  return d;
}
function place(d, x, y, s = 1, o = 1, rot = 0) {
  if (o <= 0.01 || s <= 0.01) { d.style.opacity = '0'; return; }
  d.style.opacity = String(o);
  d.style.transform = `translate(${x}px, ${y}px) translate(-50%, -50%) scale(${s}) rotate(${rot}deg)`;
}
function fitText(d, maxW, maxH, startPx, minPx = 18) {
  d.style.maxWidth = maxW + 'px';
  let fs = startPx;
  d.style.fontSize = fs + 'px';
  for (let i = 0; i < 40 && (d.scrollWidth > maxW + 1 || d.scrollHeight > maxH + 1) && fs > minPx; i++) {
    fs *= 0.93; d.style.fontSize = fs + 'px';
  }
  return fs;
}
const esc = s => String(s).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

// ------------------------------------------------------------------ world
const SEASON_PALETTES = {
  winter: { ground: '#eef4fb', ground2: '#d8e6f5', tree: ['#2e7d5b', '#3f8f6c'], snow: true, sky: ['#b9d9f2', '#eaf4fb'] },
  spring: { ground: '#9be07a', ground2: '#7fcf62', tree: ['#58b947', '#f6a6c9', '#78c850'], flowers: true, sky: ['#8fd3ff', '#e9f7ff'] },
  summer: { ground: '#86d45f', ground2: '#6cc14a', tree: ['#43a84a', '#5cbf4a'], flowers: true, sky: ['#6cc6ff', '#e3f6ff'] },
  autumn: { ground: '#b7d36a', ground2: '#a2c45a', tree: ['#f2994a', '#e86a3a', '#f6c443', '#7fb94c'], leaves: true, sky: ['#9fd4f5', '#fff1dc'] },
};
const INDOOR = new Set(['harfler', 'dil', 'sayilar', 'matematik', 'sekiller', 'sanat', 'muzik', 'duygular', 'degerler', 'gorgu',
  'evimiz', 'gunluk-yasam', 'saglik', 'beslenme', 'vucut', 'deneyler', 'bilim']);

function seasonOf(date) {
  const m = Number(String(date).slice(5, 7));
  if (m === 12 || m <= 2) return 'winter';
  if (m <= 5) return 'spring';
  if (m <= 8) return 'summer';
  return 'autumn';
}

// Sky and wall are a CSS gradient behind the transparent WebGL canvas: a full-screen textured quad costs
// as much as a whole object layer on the software GPU, the browser's compositor draws it for free.
function setBackdrop(top, bottom, glow = null) {
  const st = document.getElementById('stage');
  st.style.background = (glow ? `radial-gradient(ellipse at ${glow[0]} ${glow[1]}, ${glow[2]} 0%, rgba(255,255,255,0) 38%), ` : '')
    + `linear-gradient(to bottom, ${top} 0%, ${bottom} 100%)`;
}
function skyBackground(top, bottom) {
  return canvasTex(16, 512, (g, w, h) => {
    const gr = g.createLinearGradient(0, 0, 0, h); gr.addColorStop(0, top); gr.addColorStop(1, bottom);
    g.fillStyle = gr; g.fillRect(0, 0, w, h);
  });
}

function buildWorld() {
  const ep = P.episode;
  const cat = ep.category;
  const kind = cat === 'uzay' ? 'space' : (INDOOR.has(cat) ? 'indoor' : 'outdoor');
  const season = seasonOf(ep.date);
  const pal = SEASON_PALETTES[season];
  const th = P.theme;
  const wg = new THREE.Group(); scene.add(wg);
  const anim = [];
  // ground plane height: the mascot's feet touch it
  const feet = at(L.mascot[0], L.mascot[1], 1.0);
  const gy = feet.y;

  // lights
  const hemi = new THREE.HemisphereLight('#ffffff', kind === 'space' ? '#30264f' : '#a7c98d', kind === 'space' ? 0.9 : 1.15);
  scene.add(hemi);
  const sun = new THREE.DirectionalLight('#fff3df', kind === 'space' ? 1.6 : 2.4);
  sun.position.set(6, 14, 10); sun.castShadow = true;
  sun.shadow.mapSize.set(P.Q.shadowSize, P.Q.shadowSize);
  Object.assign(sun.shadow.camera, { left: -16, right: 16, top: 14, bottom: -10, near: 1, far: 50 });
  sun.shadow.radius = 5; sun.shadow.bias = -0.0004; sun.shadow.normalBias = 0.02; sun.shadow.intensity = 0.6;
  scene.add(sun); scene.add(sun.target);
  const fill = new THREE.DirectionalLight('#dfe8ff', 0.55); fill.position.set(-8, 4, 10); scene.add(fill);

  if (kind === 'space') {
    setBackdrop('#120c2e', '#3b2a7a');
    scene.fog = new THREE.Fog('#2a1f5c', 30, 90);
    const starGeo = new THREE.BufferGeometry(); const pts = [];
    for (let i = 0; i < 900; i++) pts.push((hash(i) - 0.5) * 140, (hash(i + 99) - 0.2) * 70, -30 - hash(i + 7) * 30);
    starGeo.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
    const stars = new THREE.Points(starGeo, new THREE.PointsMaterial({ color: '#ffffff', size: 0.35, sizeAttenuation: true, fog: false }));
    wg.add(stars);
    anim.push(t => { stars.material.size = 0.3 + 0.06 * Math.sin(t * 2); });
    const moon = shadowy(new THREE.Mesh(new THREE.SphereGeometry(60, 96, 64), matte('#b9b4d6', { roughness: 0.95 })), false, true);
    moon.position.set(0, gy - 60, -10); wg.add(moon);
    for (let i = 0; i < 24; i++) {
      const c = new THREE.Mesh(new THREE.CircleGeometry(0.4 + hash(i) * 1.1, 32), matte('#9d97c4'));
      const ang = (hash(i + 3) - 0.5) * 0.9, ang2 = (hash(i + 5) - 0.5) * 0.5;
      c.position.set(Math.sin(ang) * 60.05, gy - 60 + Math.cos(ang) * Math.cos(ang2) * 60.05, -10 + Math.sin(ang2) * 60.05);
      c.lookAt(moon.position); c.rotateY(Math.PI); wg.add(c);
    }
    const planet = shadowy(new THREE.Mesh(new THREE.SphereGeometry(3.2, 64, 48), plastic('#ff9f5a', { roughness: 0.6, clearcoat: 0.2 })));
    planet.position.set(14, 8, -28); wg.add(planet);
    const ring = new THREE.Mesh(new THREE.TorusGeometry(5, 0.35, 8, 96), plastic('#ffd27a'));
    ring.position.copy(planet.position); ring.rotation.set(1.2, 0.3, 0.2); wg.add(ring);
    const p2 = new THREE.Mesh(new THREE.SphereGeometry(1.6, 48, 32), plastic('#5ec8ff', { roughness: 0.5 }));
    p2.position.set(-16, 10, -30); wg.add(p2);
    anim.push(t => { planet.rotation.y = t * 0.1; ring.rotation.z = 0.2 + t * 0.03; });
  } else if (kind === 'indoor') {
    const wall = mixHex(th.bg[0], '#ffffff', 0.25);
    setBackdrop('#' + wall.getHexString(), '#' + mixHex(th.bg[1], '#ffffff', 0.2).getHexString(), ['50%', '30%', 'rgba(255,255,255,0.55)']);
    scene.fog = new THREE.Fog('#' + wall.getHexString(), 40, 120);
    const floorTex = canvasTex(512, 512, (g, w, h) => {
      g.fillStyle = '#e8c49a'; g.fillRect(0, 0, w, h);
      for (let i = 0; i < 8; i++) { g.fillStyle = i % 2 ? '#e2bb8f' : '#edcda6'; g.fillRect(0, i * h / 8, w, h / 8 - 3); }
    });
    floorTex.wrapS = floorTex.wrapT = THREE.RepeatWrapping; floorTex.repeat.set(6, 6);
    const backZ = -14;
    floorTex.repeat.set(6, 2);
    const floor = shadowy(new THREE.Mesh(new THREE.PlaneGeometry(140, 40), new THREE.MeshLambertMaterial({ map: floorTex })), false, true);
    floor.rotation.x = -Math.PI / 2; floor.position.set(0, gy, backZ + 20); wg.add(floor);
    // skirting + rug + window + shelf with toy blocks
    const skirt = new THREE.Mesh(new THREE.BoxGeometry(140, 0.6, 0.3), matte('#ffffff'));
    skirt.position.set(0, gy + 0.3, backZ + 0.15); wg.add(skirt);
    const rug = shadowy(new THREE.Mesh(new THREE.CircleGeometry(9, 64), matte('#' + mixHex(th.accent, '#ffffff', 0.55).getHexString())), false, true);
    rug.rotation.x = -Math.PI / 2; rug.position.set(1, gy + 0.02, -1); wg.add(rug);
    const win = new THREE.Group();
    const glass = new THREE.Mesh(new THREE.PlaneGeometry(7, 5), new THREE.MeshBasicMaterial({ map: skyBackground('#8fd3ff', '#e9f7ff') }));
    win.add(glass);
    const fr = matte('#ffffff');
    for (const [w2, h2, x2, y2] of [[7.6, 0.35, 0, 2.6], [7.6, 0.35, 0, -2.6], [0.35, 5.5, 3.6, 0], [0.35, 5.5, -3.6, 0], [0.2, 5, 0, 0], [7, 0.2, 0, 0]]) {
      const b = shadowy(new THREE.Mesh(new THREE.BoxGeometry(w2, h2, 0.3), fr)); b.position.set(x2, y2, 0.1); win.add(b);
    }
    win.position.copy(portrait ? at(W * 0.22, 330 * S, backZ + 0.05) : at(220 * S, 330 * S, backZ + 0.05));
    if (portrait) win.scale.setScalar(0.8);
    wg.add(win);
    const cloudy = new THREE.Mesh(new THREE.SphereGeometry(0.8, 24, 16), matte('#ffffff')); cloudy.position.set(-1.5, 0.8, 0.05);
    cloudy.scale.set(1.6, 0.8, 0.1); win.add(cloudy);
    // floor toys
    const ball = shadowy(new THREE.Mesh(new THREE.SphereGeometry(1.1, 48, 32), plastic('#ff6f91')), true, true);
    ball.position.set(13, gy + 1.1, -6); wg.add(ball);
    const stack = [['#3ec1d3', 1.7], ['#ffc93c', 1.4], ['#5ed67a', 1.1]];
    stack.forEach(([c, r], i) => { const d = shadowy(new THREE.Mesh(new THREE.TorusGeometry(r * 0.7, r * 0.28, 16, 48), plastic(c)), true, true);
      d.rotation.x = Math.PI / 2; d.position.set(-14, gy + 0.4 + i * 0.75, -7); wg.add(d); });
  } else {
    setBackdrop(pal.sky[0], pal.sky[1], portrait ? ['80%', '8%', 'rgba(255,250,215,0.9)'] : ['84%', '10%', 'rgba(255,250,215,0.9)']);
    scene.fog = new THREE.Fog(pal.sky[1], 35, 110);
    const groundTex = radialTex(pal.ground, pal.ground2, 512);
    const ground = shadowy(new THREE.Mesh(new THREE.PlaneGeometry(260, 260), new THREE.MeshLambertMaterial({ map: groundTex })), false, true);
    ground.rotation.x = -Math.PI / 2; ground.position.y = gy; wg.add(ground);
    // rolling hills on the horizon
    for (let i = 0; i < 7; i++) {
      const r = 14 + hash(i) * 12;
      const hill = new THREE.Mesh(new THREE.SphereGeometry(r, 32, 12, 0, Math.PI * 2, 0, Math.PI / 2), new THREE.MeshLambertMaterial({ color: i % 2 ? pal.ground2 : pal.ground }));
      hill.position.set(-70 + i * 24 + hash(i + 2) * 8, gy - r * 0.3, -60 - hash(i + 4) * 20); hill.scale.y = 0.55; wg.add(hill);
    }
    // trees
    const treeSpots = [[-13, -9], [-17, -15], [12, -10], [17, -16], [-8, -20], [6, -22], [22, -12], [-22, -12], [26, -24], [-27, -22]];
    treeSpots.forEach(([x, z], i) => {
      const t = new THREE.Group();
      const trunk = shadowy(new THREE.Mesh(new THREE.CylinderGeometry(0.28, 0.4, 2.6, 12), matte('#8a5a3b')), false, false);
      trunk.position.y = 1.3; t.add(trunk);
      if (season === 'winter') {
        for (let k = 0; k < 3; k++) {
          const cone = new THREE.Mesh(new THREE.ConeGeometry(2.0 - k * 0.5, 2.2, 20), matte(pal.tree[k % 2]));
          cone.position.y = 2.6 + k * 1.1; t.add(cone);
          const cap = new THREE.Mesh(new THREE.ConeGeometry(1.2 - k * 0.32, 0.9, 20), matte('#ffffff'));
          cap.position.y = 3.3 + k * 1.1; t.add(cap);
        }
      } else {
        const c = pal.tree[i % pal.tree.length];
        for (let k = 0; k < 3; k++) {
          const s = new THREE.Mesh(new THREE.SphereGeometry(1.5 - k * 0.2, 24, 16), plastic(c, { roughness: 0.7, clearcoat: 0.1 }));
          s.position.set((k - 1) * 0.9, 3.2 + (k === 1 ? 0.9 : 0), (hash(i + k) - 0.5)); t.add(s);
        }
      }
      t.position.set(x, gy, z); t.scale.setScalar(1 + hash(i + 9) * 0.5); wg.add(t);
      anim.push(tt => { t.rotation.z = Math.sin(tt * 0.8 + i) * 0.015; });
    });
    if (pal.flowers) {
      const fc = ['#ff6f91', '#ffd23f', '#ffffff', '#b28dff'];
      for (let i = 0; i < 40; i++) {
        const f = new THREE.Mesh(new THREE.SphereGeometry(0.16, 12, 8), matte(fc[i % 4]));
        f.position.set((hash(i) - 0.5) * 40, gy + 0.12, -3 - hash(i + 50) * 14); wg.add(f);
      }
    }
    // clouds
    for (let i = 0; i < 7; i++) {
      const cl = new THREE.Group();
      for (let k = 0; k < 5; k++) {
        const s = new THREE.Mesh(new THREE.SphereGeometry(1.2 + hash(i * 7 + k) * 0.9, 24, 16), matte('#ffffff', { roughness: 1, emissive: col('#ffffff'), emissiveIntensity: 0.25 }));
        s.position.set(k * 1.3 - 2.6, Math.sin(k) * 0.4, hash(k) * 0.6); cl.add(s);
      }
      const x0 = -50 + i * 16, y0 = gy + 16 + hash(i + 1) * 9, z0 = -45 - hash(i + 3) * 15;
      cl.position.set(x0, y0, z0); wg.add(cl);
      anim.push(t => { cl.position.x = ((x0 + t * 0.35 + 60) % 120) - 60; });
    }
    // sun disc
    const sunDisc = new THREE.Mesh(new THREE.CircleGeometry(4, 48), new THREE.MeshBasicMaterial({ color: '#fff6c8', fog: false }));
    sunDisc.position.copy(at(W * (portrait ? 0.8 : 0.84), H * (portrait ? 0.08 : 0.1), -80)); wg.add(sunDisc);
  }

  // festive bunting (national days, not on remembrance days): one garland along the top edge, behind everything
  if ((cat === 'ulusal-bayram' || cat === 'ozel-gun') && ep.mood !== 'calm') {
    flagTex = flagTex || turkishFlagTexture(600);
    const z = -8, left = at(-40, 8, z), right = at(W + 40, 8, z), mid = at(W / 2, portrait ? 120 * S : 92 * S, z);
    const sagDepth = left.y - mid.y;
    const yOn = x => { const k = (x - mid.x) / ((right.x - left.x) / 2); return mid.y + sagDepth * k * k; };
    const pts = []; for (let i = 0; i <= 40; i++) { const x = left.x + (right.x - left.x) * i / 40; pts.push(new THREE.Vector3(x, yOn(x), z)); }
    wg.add(new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 80, 0.035, 6), matte('#ffffff')));
    const fw = 62 * S * upp(z), geo = new THREE.PlaneGeometry(fw, fw * 2 / 3); geo.translate(0, -fw / 3, 0);
    const mat = new THREE.MeshStandardMaterial({ map: flagTex, side: THREE.DoubleSide, roughness: 0.8 });
    const n = portrait ? 9 : 15;
    for (let i = 0; i < n; i++) {
      const x = left.x + (right.x - left.x) * (i + 0.5) / n;
      const fl = new THREE.Mesh(geo, mat); fl.position.set(x, yOn(x), z); wg.add(fl);
      anim.push(t => { fl.rotation.x = Math.sin(t * 2.2 + i * 0.8) * 0.22; fl.rotation.z = Math.sin(t * 1.6 + i) * 0.05; });
    }
  }

  // ambient particles: falling leaves / snow / sparkles
  if (kind === 'outdoor' && (pal.leaves || pal.snow)) {
    const n = pal.snow ? 120 : 40;
    const geo = pal.snow ? new THREE.SphereGeometry(0.07, 8, 6) : new THREE.PlaneGeometry(0.3, 0.2);
    const mat = pal.snow ? matte('#ffffff', { emissive: col('#ffffff'), emissiveIntensity: 0.3 })
      : new THREE.MeshStandardMaterial({ color: '#f08a3c', side: THREE.DoubleSide });
    const inst = new THREE.InstancedMesh(geo, mat, n);
    const dummy = new THREE.Object3D();
    wg.add(inst);
    anim.push(t => {
      for (let i = 0; i < n; i++) {
        const sp = 0.6 + hash(i) * 0.8;
        const y = gy + 18 - ((t * sp + hash(i + 3) * 18) % 18);
        dummy.position.set((hash(i + 1) - 0.5) * 34 + Math.sin(t + i) * 0.6, y, -2 - hash(i + 2) * 16);
        dummy.rotation.set(t * sp * 2 + i, t + i, 0);
        dummy.updateMatrix(); inst.setMatrixAt(i, dummy.matrix);
      }
      inst.instanceMatrix.needsUpdate = true;
    });
  }
  return { group: wg, update: t => anim.forEach(f => f(t)), groundY: gy, kind };
}

// ------------------------------------------------------------------ mascot: Pofi the owl
const OWL = { body: '#8E7CF8', belly: '#FFE6C4', disc: '#C3B8FF', beak: '#FFA53A', pupil: '#2D2548' };
function buildMascot() {
  const root = new THREE.Group(); scene.add(root);
  const rig = new THREE.Group(); root.add(rig);
  const fluffy = (c) => new THREE.MeshPhysicalMaterial({ color: col(c), roughness: 0.62, sheen: 1, sheenRoughness: 0.5,
    sheenColor: mixHex(c, '#ffffff', 0.5) });
  const body = shadowy(new THREE.Mesh(new THREE.SphereGeometry(1, 48, 32), fluffy(OWL.body)));
  body.scale.set(0.98, 1.1, 0.9); body.position.y = 1.15; rig.add(body);
  const belly = new THREE.Mesh(new THREE.SphereGeometry(0.66, 36, 24), fluffy(OWL.belly));
  belly.scale.set(1, 1.05, 0.55); belly.position.set(0, 0.82, 0.55); rig.add(belly);
  // feather scallops on the belly
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < (r === 1 ? 2 : 3); c++) {
      const a = new THREE.Mesh(new THREE.TorusGeometry(0.09, 0.018, 6, 12, Math.PI), matte('#e8c79f'));
      const x = (r === 1 ? (c - 0.5) : (c - 1)) * 0.2;
      a.position.set(x, 1.05 - r * 0.2, 0.9 - r * 0.03); a.rotation.z = Math.PI; rig.add(a);
    }
  }
  const head = new THREE.Group(); head.position.y = 1.55; rig.add(head);
  for (const sx of [-1, 1]) {
    const disc = new THREE.Mesh(new THREE.SphereGeometry(0.43, 32, 20), fluffy(OWL.disc));
    disc.scale.set(1, 0.95, 0.42); disc.position.set(sx * 0.36, 0.05, 0.66); head.add(disc);
    const tuft = shadowy(new THREE.Mesh(new THREE.ConeGeometry(0.17, 0.55, 20), fluffy('#6f5ce0')));
    tuft.position.set(sx * 0.55, 0.62, 0.05); tuft.rotation.z = -sx * 0.35; head.add(tuft);
    const cheek = new THREE.Mesh(new THREE.SphereGeometry(0.13, 24, 16), new THREE.MeshStandardMaterial({ color: '#ff8fb5', transparent: true, opacity: 0.55, roughness: 0.6 }));
    cheek.scale.set(1.3, 0.8, 0.3); cheek.position.set(sx * 0.68, -0.22, 0.72); head.add(cheek);
  }
  const eyes = [];
  for (const sx of [-1, 1]) {
    const eye = new THREE.Group(); eye.position.set(sx * 0.35, 0.07, 0.8); head.add(eye);
    const white = new THREE.Mesh(new THREE.SphereGeometry(0.29, 32, 20), new THREE.MeshStandardMaterial({ color: '#ffffff', roughness: 0.18 }));
    white.scale.z = 0.55; eye.add(white);
    const pupil = new THREE.Mesh(new THREE.SphereGeometry(0.18, 28, 18), new THREE.MeshStandardMaterial({ color: OWL.pupil, roughness: 0.12 }));
    pupil.scale.z = 0.5; pupil.position.set(-sx * 0.03, -0.02, 0.1); eye.add(pupil);
    const hl = new THREE.Mesh(new THREE.SphereGeometry(0.055, 16, 12), new THREE.MeshBasicMaterial({ color: '#ffffff' }));
    hl.position.set(-sx * 0.03 - 0.06, 0.06, 0.2); eye.add(hl);
    const hl2 = new THREE.Mesh(new THREE.SphereGeometry(0.025, 12, 8), new THREE.MeshBasicMaterial({ color: '#ffffff' }));
    hl2.position.set(-sx * 0.03 + 0.06, -0.06, 0.2); eye.add(hl2);
    // eyelid: a shell that closes downward
    const lid = new THREE.Mesh(new THREE.SphereGeometry(0.305, 28, 14, 0, Math.PI * 2, 0, Math.PI / 2), fluffy(OWL.disc));
    lid.scale.z = 0.6; lid.rotation.x = -Math.PI / 2; eye.add(lid);
    // happy closed-eye arc
    const arc = new THREE.Mesh(new THREE.TorusGeometry(0.17, 0.035, 10, 24, Math.PI), new THREE.MeshBasicMaterial({ color: OWL.pupil }));
    arc.position.set(0, -0.05, 0.19); arc.visible = false; eye.add(arc);
    eyes.push({ eye, white, pupil, hl, hl2, lid, arc });
  }
  const beakTop = shadowy(new THREE.Mesh(new THREE.ConeGeometry(0.13, 0.3, 24), plastic(OWL.beak, { roughness: 0.35 })));
  beakTop.rotation.x = Math.PI; beakTop.position.set(0, -0.17, 0.9); head.add(beakTop);
  const mouth = new THREE.Mesh(new THREE.SphereGeometry(0.11, 24, 16), new THREE.MeshStandardMaterial({ color: '#7a1e40', roughness: 0.6 }));
  mouth.scale.set(1, 0.6, 0.4); mouth.position.set(0, -0.3, 0.86); head.add(mouth);
  const jaw = new THREE.Group(); jaw.position.set(0, -0.27, 0.86); head.add(jaw);
  const beakBot = new THREE.Mesh(new THREE.ConeGeometry(0.09, 0.16, 20), plastic('#f08a2a'));
  beakBot.rotation.x = Math.PI; beakBot.position.set(0, -0.06, 0.02); jaw.add(beakBot);
  const wings = [];
  for (const sx of [-1, 1]) {
    const pivot = new THREE.Group(); pivot.position.set(sx * 0.9, 1.4, 0.12); rig.add(pivot);
    const w = shadowy(new THREE.Mesh(new THREE.SphereGeometry(0.5, 24, 16), fluffy('#7b68ea')));
    w.scale.set(0.36, 1.0, 0.7); w.position.set(sx * 0.1, -0.42, 0); pivot.add(w);
    wings.push({ pivot, sx });
  }
  for (const sx of [-1, 1]) {
    for (let k = -1; k <= 1; k++) {
      const toe = shadowy(new THREE.Mesh(new THREE.SphereGeometry(0.1, 12, 8), plastic(OWL.beak)));
      toe.scale.set(1, 0.6, 1.4); toe.position.set(sx * 0.32 + k * 0.11, 0.06, 0.42); rig.add(toe);
    }
  }
  return { root, rig, head, eyes, jaw, mouth, wings, body };
}

function updateMascot(t, fi, info) {
  const m = mascot;
  const [mx, my, mh] = L.mascot;
  const big = info.big && !portrait ? 1.12 : 1.0;
  const z = 1.0;
  // the owl model is ~2.3 units tall
  const scale = mh * big * upp(z) / 2.45;   // model height including the ear tufts
  const base = at(mx, my, z);
  const jumpH = (info.jump || 0) * (portrait ? 45 : 70) * S * upp(z);   // 2D jump height; lower under the Shorts caption
  m.root.position.set(base.x, base.y + jumpH + Math.sin(t * 2.6) * 0.03 * scale, base.z);
  m.root.scale.setScalar(scale);
  const squash = 1 + 0.03 * Math.sin(t * 2.6 * 2);
  m.rig.scale.set(1 / Math.sqrt(squash), squash, 1 / Math.sqrt(squash));
  m.rig.rotation.y = portrait ? 0 : 0.28 + Math.sin(t * 0.7) * 0.06; // turned toward the content
  m.head.rotation.z = Math.sin(t * 1.1) * 0.06;
  m.head.rotation.x = -0.05 + Math.sin(t * 0.9) * 0.04;
  const env = P.mouth[fi] || 0;
  const open = Math.min(1, env * 1.2);
  m.jaw.position.y = -0.27 - open * 0.11;
  m.jaw.rotation.x = open * 0.5;
  m.mouth.scale.y = 0.2 + open * 1.1;
  // blink
  const cycle = 3.4 + 0.9 * Math.sin(Math.floor(t / 3.4) * 1.7);
  const ph = t % cycle;
  const blink = ph < 0.18 ? Math.sin(ph / 0.18 * Math.PI) : 0;
  const happy = !!info.happy;
  m.eyes.forEach(e => {
    e.lid.rotation.x = -Math.PI / 2 + blink * (Math.PI / 2) * 0.98;
    e.lid.visible = !happy;
    e.white.visible = e.pupil.visible = e.hl.visible = e.hl2.visible = !happy;
    e.arc.visible = happy;
  });
  m.wings.forEach(({ pivot, sx }) => {
    // positive z-rotation swings the right wing's tip outwards, negative the left one's
    let a = sx * (0.14 + 0.04 * Math.sin(t * 2.6));
    if (info.waving && sx > 0) a = 2.2 + Math.sin(t * 9) * 0.4;
    if (info.jump) a = sx * (0.5 + 0.9 * info.jump);
    pivot.rotation.z = a;
  });
}

// ------------------------------------------------------------------ confetti
function makeConfetti(seed, originPx, z = 0.5) {
  const n = 110;
  const geo = new THREE.PlaneGeometry(0.16, 0.1);
  const cols = ['#FF5A5F', '#FFC93C', '#3EC1D3', '#7C6CF2', '#5ED67A', '#FF8BC2'];
  const mat = new THREE.MeshStandardMaterial({ side: THREE.DoubleSide, roughness: 0.6 });
  const inst = new THREE.InstancedMesh(geo, mat, n);
  for (let i = 0; i < n; i++) inst.setColorAt(i, col(cols[i % cols.length]));
  const o = at(originPx[0], originPx[1], z);
  const dummy = new THREE.Object3D();
  const k = upp(z) * 1080 / (portrait ? 1.6 : 1.0);
  inst.userData.update = (t) => {
    const vis = t >= 0 && t < 2.6;
    inst.visible = vis; if (!vis) return;
    for (let i = 0; i < n; i++) {
      const a = (hash(seed + i) - 0.5) * Math.PI * 1.4 + Math.PI / 2;
      const sp = (6 + hash(seed + i * 3) * 7) * k / 9;
      const drag = 1 - Math.exp(-1.3 * t);
      const x = o.x + Math.cos(a) * sp * drag / 1.3 * 1.4;
      const y = o.y + Math.sin(a) * sp * drag / 1.3 - 4.5 * k / 9 * t * t;
      dummy.position.set(x, y, o.z + (hash(seed + i * 7) - 0.5) * 2);
      dummy.rotation.set(t * 6 + i, t * 4 + i * 2, t * 5);
      const s = clamp((2.6 - t) / 0.6) * (0.8 + hash(i) * 0.8) * k / 9 * 1.6;
      dummy.scale.setScalar(Math.max(0.001, s));
      dummy.updateMatrix(); inst.setMatrixAt(i, dummy.matrix);
    }
    inst.instanceMatrix.needsUpdate = true;
  };
  return inst;
}

// ------------------------------------------------------------------ scenes
class Scene {
  constructor(d, index) {
    this.d = d; this.s = d.spec; this.index = index; this.T = d.T; this.start = d.start; this.dur = d.dur;
    this.th = P.theme;
    this.g = new THREE.Group(); this.g.visible = false; scene.add(this.g);
    this.doms = [];
    this.anims = [];
    this.built = false;
  }
  t0(tag, def = 1e9) { return this.T[tag] ? this.T[tag][0] : def; }
  t1(tag, def = 1e9) { return this.T[tag] ? this.T[tag][1] : def; }
  active(tag, t) { const r = this.T[tag]; return !!r && r[0] <= t && t < r[1] + 0.15; }
  pulse(t, tags) {
    let best = 0;
    for (const tag of tags) { const a = this.t0(tag); if (a <= t && t < a + 0.4) best = Math.max(best, Math.sin(Math.PI * (t - a) / 0.4)); }
    return 1 + 0.07 * best;
  }
  lineTags(prefix = 'l', key = 'lines') { return (this.s[key] || []).map((_, i) => prefix + i); }
  dom(html, cls, style) { const d = el(html, cls, style); this.doms.push(d); return d; }
  vis(v, px, py, wPx, z = 0) {
    const g = makeVisual(v); sizeVisual(g, wPx, z); placeAt(g, px, py, z); this.g.add(g);
    g.userData.home = g.position.clone(); g.userData.z = z;
    g.traverse(o => { o.renderOrder = 2; });   // on top of any card it sits on (card3d)
    return g;
  }
  setVis(g, t, s, rot = 0, dy = 0, opacity = 1) {
    const k = g.userData.baseScale * Math.max(0.0001, s);
    g.scale.setScalar(k); g.visible = s > 0.02;
    g.rotation.z = rot;
    g.position.copy(g.userData.home); g.position.y += dy * upp(g.userData.z);
    if (g.userData.update) g.userData.update(t);
    if (opacity < 1 || g.userData.faded) {
      g.userData.faded = opacity < 1;
      g.traverse(o => { if (o.material) (Array.isArray(o.material) ? o.material : [o.material]).forEach(m => setOpacity(m, opacity)); });
    }
  }
  titleDom(text, maxW, maxH, sizePx = 96, color = '#ffffff') {
    const d = this.dom(esc(text), 'title', { color, webkitTextStroke: `${Math.round(10 * S)}px ${this.th.dark}`,
      paintOrder: 'stroke fill', textShadow: `0 ${Math.round(8 * S)}px 0 ${this.th.dark}, 0 ${Math.round(14 * S)}px 24px rgba(30,20,60,.35)` });
    fitText(d, maxW, maxH, sizePx * S);
    return d;
  }
  label(text, maxW, maxH, sizePx = 52) {
    const d = this.dom(esc(text), 'body', { color: this.th.dark });
    fitText(d, maxW, maxH, sizePx * S);
    return d;
  }
  build() {}
  update(t) {}
  mascot(t) { return { waving: false, happy: false, jump: 0, big: false }; }
  show(on) { this.g.visible = on; if (!on) this.doms.forEach(d => (d.style.opacity = '0')); }
}

function contentBox() { return L.content; }

class Intro extends Scene {
  build() {
    const [x, y, w, h] = contentBox();
    this.title = text3d(this.s.title, { heightPx: 150 * S, maxWpx: w * 0.94, maxHpx: h * 0.36, color: '#ffffff', side: this.th.dark });
    placeAt(this.title, x + w / 2, y + h * 0.24, 0.3); this.g.add(this.title);
    if (this.s.subtitle) {
      // a light accent (yellow on the space theme) needs a dark outline, a strong one a white outline
      const ring = contrast(this.th.accent, '#ffffff') >= 2 ? '#ffffff' : this.th.dark;
      this.sub = this.dom(esc(this.s.subtitle), 'title', { color: this.th.accent,
        webkitTextStroke: `${Math.round(9 * S)}px ${ring}`, paintOrder: 'stroke fill', textShadow: `0 6px 18px rgba(30,20,60,.3)` });
      fitText(this.sub, w * 0.8, h * 0.13, 84 * S);
    }
    const vs = (this.s.visuals || []).slice(0, 5); const n = vs.length;
    // with 4-5 pictures the row reaches the left edge, where the big waving owl's wing would cover the first one
    const lm = !portrait && n >= 4 ? w * 0.1 : 0, rw = w - lm;
    this.vs = vs.map((v, i) => {
      const fx = (i + 0.5) / n;
      const size = Math.min(rw / (n + 0.6), h * 0.3);
      const cx = x + lm + rw * (0.5 + (fx - 0.5) * Math.min(1, 0.28 * n));
      const cy = y + h * 0.77 - Math.sin(Math.PI * fx) * h * 0.04;
      return this.vis(v, cx, cy, size);
    });
  }
  update(t) {
    const [x, y, w, h] = contentBox();
    animateLetters(this.title, t, 0.2, { stagger: 0.05, dur: 0.55, wave: t > 1.6 ? 0.04 : 0 });
    this.title.rotation.y = Math.sin(t * 0.8) * 0.12;
    this.title.rotation.x = -0.08 + Math.sin(t * 0.6) * 0.03;
    if (this.sub) place(this.sub, x + w / 2, y + h * 0.45, pop(t, 0.9, 0.5));
    this.vs.forEach((g, i) => this.setVis(g, t, pop(t, 0.75 + 0.14 * i), wob(t, i, 0.09), bob(t, i * 1.3, 10 * S)));
  }
  mascot(t) { return { waving: t < this.t1('l0', 2.5), happy: t < 0.6, jump: 0, big: true }; }
}

class Outro extends Scene {
  build() {
    const [x, y, w, h] = contentBox();
    this.title = text3d(this.s.title || 'Görüşmek üzere!', { heightPx: 140 * S, maxWpx: w * 0.95, maxHpx: h * 0.28, color: '#ffffff', side: this.th.dark });
    placeAt(this.title, x + w / 2, y + h * 0.18, 0.3); this.g.add(this.title);
    this.stars = [];
    for (let i = 0; i < 6; i++) {
      const s = star3d(); const px = x + w * (i + 0.5) / 6, py = y + h * (0.42 + 0.06 * Math.sin(i * 2.1));
      placeAt(s, px, py, 0); s.userData.home = s.position.clone(); s.userData.k = 70 * S * upp(0);
      this.g.add(s); this.stars.push(s);
    }
    const nxt = this.s.next;
    if (nxt) {
      const cw = w * 0.7, ch = h * 0.36, cx = x + w / 2, cy = y + h * 0.74;
      this.card = card3d(cw, ch, -0.1); placeAt(this.card, cx, cy, -0.1); this.g.add(this.card);
      this.card.userData.home = this.card.position.clone();
      const v = nxt.visual;
      const vsz = ch * (portrait ? 0.5 : 0.66);
      if (v) { this.nv = this.vis(v, cx - cw / 2 + vsz * 0.72, cy, vsz, 0.15); }
      const tx0 = v ? vsz * 1.45 : 0, avail = v ? cw - tx0 - 40 * S : cw * 0.9;
      this.lab = this.dom('Sıradaki bölüm', 'body', { color: readableOnWhite(this.th.accent), fontSize: Math.round(40 * S) + 'px' });
      this.tx = this.label(nxt.text, avail, ch * 0.5, 72);
      this.txPos = [v ? cx - cw / 2 + tx0 + avail / 2 : cx, cy];
    } else {
      const vs = (this.s.visuals || []).slice(0, 3); const n = vs.length;
      this.vs = vs.map((v, i) => this.vis(v, x + w * (0.5 + ((i + 0.5) / n - 0.5) * Math.min(1, 0.3 * n)), y + h * 0.72, Math.min(w / (n + 0.6), h * 0.28)));
    }
  }
  update(t) {
    animateLetters(this.title, t, 0.2, { stagger: 0.04, wave: t > 1.4 ? 0.035 : 0 });
    this.title.rotation.y = Math.sin(t * 0.8) * 0.1;
    this.stars.forEach((s, i) => {
      const k = s.userData.k * pop(t, 0.4 + 0.1 * i) * (0.85 + 0.2 * Math.sin(t * 3 + i * 1.7));
      s.scale.setScalar(Math.max(0.0001, k)); s.rotation.y = t * 1.5 + i; s.position.y = s.userData.home.y + Math.sin(t * 2 + i) * 0.08;
    });
    if (this.card) {
      const lines = this.s.lines || [];
      const appear = lines.length > 1 ? this.t0('l1', 1.2) : 1.2;
      const sc = pop(t, appear, 0.5);
      this.card.scale.setScalar(Math.max(0.0001, sc)); this.card.visible = sc > 0.02;
      this.card.rotation.y = (1 - Math.min(1, sc)) * 0.8 + Math.sin(t) * 0.04;
      if (this.nv) this.setVis(this.nv, t, sc, wob(t, 0, 0.06), bob(t, 0, 5 * S));
      const [tx, ty] = this.txPos; const ch = contentBox()[3] * 0.36;
      place(this.lab, tx, ty - ch * 0.22, sc); place(this.tx, tx, ty + ch * 0.12, sc);
    }
    if (this.vs) this.vs.forEach((g, i) => this.setVis(g, t, pop(t, 0.6 + 0.14 * i), wob(t, i, 0.08), bob(t, i * 1.3, 10 * S)));
  }
  mascot(t) { return { waving: true, happy: t > this.dur - 2.4, jump: 0, big: true }; }
}

class Talk extends Scene {
  build() {
    const [x, y, w, h] = contentBox();
    const vs = (this.s.visuals || []).slice(0, 3), n = vs.length;
    const areaH = h * (this.s.text ? 0.66 : 0.9);
    const size = n === 1 ? Math.min(w * 0.5, areaH * 0.78) : n === 2 ? Math.min(w * 0.38, areaH * 0.7) : Math.min(w * 0.28, areaH * 0.64);
    this.vs = vs.map((v, i) => this.vis(v, x + w * (i + 0.5) / n, y + areaH * 0.52, size));
    if (this.s.text) {
      this.txt = this.dom(esc(this.s.text), 'body card', { color: this.th.dark, padding: `${Math.round(22 * S)}px ${Math.round(44 * S)}px` });
      fitText(this.txt, w * 0.86, h * 0.26, 70 * S);
    }
  }
  update(t) {
    const [x, y, w, h] = contentBox();
    const p = this.pulse(t, this.lineTags());
    this.vs.forEach((g, i) => { this.setVis(g, t, pop(t, 0.15 + 0.12 * i) * p, wob(t, i * 2, 0.06), bob(t, i * 1.7, 9 * S)); g.rotation.y = Math.sin(t * 0.9 + i) * 0.25; });
    if (this.txt) place(this.txt, x + w / 2, y + h * 0.84, pop(t, Math.min(this.t0('l0', 0.5) + 0.35, 1.0)));
  }
}

class Item extends Scene {
  build() {
    const [x, y, w, h] = contentBox();
    const size = Math.min(w * 0.44, h * 0.52);
    this.cx = x + w / 2; this.cy = y + h * 0.38;
    this.color = this.s.color || this.th.accent;
    // glowing podium under the object
    const u = upp(0);
    this.podium = shadowy(new THREE.Mesh(new THREE.CylinderGeometry(size * 0.45 * u, size * 0.52 * u, size * 0.12 * u, 64),
      plastic(this.color, { clearcoat: 1, roughness: 0.25, emissive: col(this.color), emissiveIntensity: 0.15 })), true, true);
    placeAt(this.podium, this.cx, this.cy + size * 0.55, -0.3); this.g.add(this.podium);
    this.glow = new THREE.Mesh(new THREE.CircleGeometry(size * 0.75 * u, 64), new THREE.MeshBasicMaterial({
      map: radialTex('rgba(255,255,255,0.9)', 'rgba(255,255,255,0)'), color: col(this.color), transparent: true, depthWrite: false }));
    placeAt(this.glow, this.cx, this.cy, -0.8); this.g.add(this.glow);
    this.v = this.vis(this.s.visual, this.cx, this.cy, size);
    this.word = text3d(trUpper(this.s.word), { heightPx: 150 * S, maxWpx: w * 0.9, maxHpx: h * 0.24, color: this.color, side: '#' + mixHex(this.color, '#000', 0.4).getHexString() });
    placeAt(this.word, this.cx, y + h * 0.88, 0.4); this.g.add(this.word);
  }
  update(t) {
    const p = this.pulse(t, this.lineTags());
    const sc = pop(t, 0.15);
    this.setVis(this.v, t, sc * p, wob(t, 0, 0.07), bob(t, 0, 10 * S));
    this.v.rotation.y = Math.sin(t * 1.2) * 0.35;
    const ps = pop(t, 0.05, 0.5);
    this.podium.scale.setScalar(Math.max(0.0001, ps)); this.podium.rotation.y = t * 0.6;
    this.glow.scale.setScalar(Math.max(0.0001, ps * (1 + 0.04 * Math.sin(t * 2.4))));
    const appear = this.t0('l0', 0.5) + 0.45;
    animateLetters(this.word, t, appear, { stagger: 0.06, wave: t > appear + 1 ? 0.03 : 0 });
    this.word.rotation.y = Math.sin(t * 0.7) * 0.1;
  }
}

class Fact extends Scene {
  build() {
    const [x, y, w, h] = contentBox();
    const cw = w * 0.94, ch = h * 0.8; this.cx = x + w / 2; this.cy = y + h * 0.53;
    this.card = card3d(cw, ch, -0.4); placeAt(this.card, this.cx, this.cy, -0.4); this.g.add(this.card);
    this.head = this.dom(`<img src="${P.emoji['💡']}" style="width:1.15em;height:1.15em;vertical-align:-0.22em;margin-right:0.25em">${esc(this.s.title || 'Biliyor muydun?')}`, 'title pill',
      { background: this.th.accent, color: inkOn(this.th.accent, this.th.dark), padding: `${Math.round(6 * S)}px ${Math.round(40 * S)}px`,
        boxShadow: '0 8px 20px rgba(40,25,80,.25)', fontSize: Math.round(62 * S) + 'px' });
    fitText(this.head, cw * 0.9, ch * 0.16, 62 * S, 30 * S);
    this.headPos = [this.cx, this.cy - ch / 2 + 34 * S];
    if (portrait) {
      this.txt = this.label(this.s.text, cw * 0.86, ch * 0.34, 80); this.txtPos = [this.cx, this.cy + ch * 0.28];
      this.v = this.vis(this.s.visual, this.cx, this.cy - ch * 0.12, ch * 0.42, 0.3);
    } else {
      this.txt = this.label(this.s.text, cw * 0.5, ch * 0.62, 86); this.txtPos = [this.cx + cw * 0.2, this.cy + ch * 0.05];
      this.v = this.vis(this.s.visual, this.cx - cw * 0.26, this.cy + ch * 0.08, Math.min(ch * 0.58, cw * 0.36), 0.3);
    }
  }
  update(t) {
    const sc = pop(t, 0.15, 0.55);
    this.card.scale.setScalar(Math.max(0.0001, sc)); this.card.visible = sc > 0.02;
    this.card.rotation.x = (1 - Math.min(sc, 1)) * 0.6;
    this.card.rotation.y = Math.sin(t * 0.5) * 0.03;
    place(this.head, this.headPos[0], this.headPos[1], pop(t, 0.35));
    place(this.txt, this.txtPos[0], this.txtPos[1], sc > 0.9 ? pop(t, 0.55) : 0);
    this.setVis(this.v, t, sc > 0.9 ? pop(t, 0.45) * this.pulse(t, this.lineTags()) : 0, wob(t, 0, 0.06), bob(t, 0, 8 * S));
    this.v.rotation.y = Math.sin(t) * 0.3;
  }
}

function gridCells(n, area, port, maxCols) {
  let [x, y, w, h] = area;
  x += w * 0.025; w *= 0.95;
  let cols = port ? (n === 1 ? 1 : 2) : (n <= 4 ? n : 3);
  if (maxCols) cols = Math.min(cols, maxCols);
  const rows = Math.ceil(n / cols), cw = w / cols, ch = h / rows;
  const out = [];
  for (let i = 0; i < n; i++) {
    const r = Math.floor(i / cols), c = i % cols;
    const inRow = r < rows - 1 ? cols : n - cols * (rows - 1);
    const x0 = x + (w - inRow * cw) / 2;
    out.push([x0 + (c + 0.5) * cw, y + (r + 0.5) * ch, cw, ch]);
  }
  return out;
}

function uniformLabels(scene, texts, maxW, maxH, sizePx) {
  const ds = texts.map(tx => scene.dom(esc(tx), 'body', { color: scene.th.dark }));
  const sizes = ds.map(d => fitText(d, maxW, maxH, sizePx * S)).sort((a, b) => a - b);
  const common = sizes.length > 2 ? sizes[1] : sizes[0];
  ds.forEach(d => fitText(d, maxW, maxH, common));
  return ds;
}

class Gallery extends Scene {
  get revealAll() { return false; }
  build() {
    const [x, y, w, h] = contentBox();
    this.items = this.s.items;
    const th = this.s.title ? h * 0.16 : 0;
    if (this.s.title) { this.ttl = this.titleDom(this.s.title, w * 0.9, th, 96); this.ttlPos = [x + w / 2, y + th / 2]; }
    const cells = gridCells(this.items.length, [x, y + th, w, h - th], portrait);
    const cw0 = cells[0][2] * 0.88, chh = Math.min(cells[0][3] * 0.9, cw0 * 1.4);
    const labelH = Math.min(chh * 0.22, 130 * S);
    this.labels = uniformLabels(this, this.items.map(i => i.label), cw0 * 0.86, labelH, 58);
    this.cells = cells.map(([cx, cy, cw, ch], i) => {
      const card = card3d(cw0, chh, -0.25); placeAt(card, cx, cy, -0.25); this.g.add(card);
      card.userData.home = card.position.clone();
      const vs = Math.min(cw0 * 0.74, (chh - labelH) * 0.8);
      const v = this.vis(this.items[i].visual, cx, cy - labelH * 0.5, vs, 0.15);
      return { card, v, cx, cy, labelPos: [cx, cy + chh / 2 - labelH * 0.62] };
    });
  }
  appear(i) { return this.revealAll ? 0.15 + 0.14 * i : this.t0('i' + i, 0.3); }
  update(t) {
    if (this.ttl) place(this.ttl, this.ttlPos[0], this.ttlPos[1], pop(t, 0.1));
    this.cells.forEach((c, i) => {
      const sc = pop(t, this.appear(i));
      const hl = this.active('i' + i, t);
      const k = hl ? 1.08 : 1.0;
      c.card.visible = sc > 0.02;
      c.card.scale.setScalar(Math.max(0.0001, sc * k));
      c.card.position.copy(c.card.userData.home); c.card.position.z += hl ? 0.5 : 0;
      c.card.rotation.y = (1 - Math.min(sc, 1)) * 1.2 + Math.sin(t * 0.8 + i) * 0.04;
      cardGlow(c.card, hl ? this.th.accent : null, 0.2);
      this.setVis(c.v, t, sc * k * this.pulse(t, ['i' + i]), wob(t, i, hl ? 0.08 : 0.04), bob(t, i * 1.1, 6 * S));
      c.v.position.z += hl ? 0.5 : 0;
      place(this.labels[i], c.labelPos[0], c.labelPos[1], sc * k);
    });
  }
}
class Recap extends Gallery { get revealAll() { return true; } }

class Count extends Scene {
  build() {
    const [x, y, w, h] = contentBox();
    const n = this.s.n; this.n = n;
    const top = h * 0.2;
    const cols = portrait ? Math.min(n, n <= 9 ? 3 : 4) : (n <= 5 ? n : n <= 10 ? Math.ceil(n / 2) : 5);
    const rows = Math.ceil(n / cols);
    const band = portrait && this.s.label ? h * 0.12 : 0;
    const aw = w, ah = h - top - band, cw = aw / cols, ch = ah / rows;
    const size = Math.min(cw, ch) * 0.74;
    this.top = top; this.band = band;
    this.items = [];
    for (let i = 0; i < n; i++) {
      const r = Math.floor(i / cols), c = i % cols, inRow = r < rows - 1 ? cols : n - cols * (rows - 1);
      const cx = x + (aw - inRow * cw) / 2 + (c + 0.5) * cw, cy = y + top + (r + 0.5) * ch;
      const v = this.vis(this.s.visual, cx, cy, size);
      const badge = coin3d(String(i + 1), this.th.accent, size * 0.36, 0.4);
      placeAt(badge, cx + size * 0.36, cy + size * 0.3, 0.4); badge.userData.home = badge.position.clone(); this.g.add(badge);
      this.items.push({ v, badge, cx, cy });
    }
    this.bigs = {};
    this.bigPos = [x + w / 2, y + top * 0.5];
    this.bigSize = Math.min(190 * S, top * 0.95);
    if (this.s.label) {
      const txt = `${n} ${trUpper(this.s.label)}`;
      this.lab = text3d(txt, { heightPx: 100 * S, maxWpx: w * (portrait ? 0.8 : 0.34), maxHpx: (band || top) * 0.85, color: '#ffffff', side: this.th.dark });
      placeAt(this.lab, portrait ? x + w / 2 : x + w * 0.83, portrait ? y + h - band / 2 : y + top * 0.5, 0.4); this.g.add(this.lab);
    }
  }
  big(k) {
    if (!this.bigs[k]) { const c = coin3d(String(k), this.th.accent2, this.bigSize, 0.5); placeAt(c, this.bigPos[0], this.bigPos[1], 0.5); this.g.add(c); this.bigs[k] = c; }
    return this.bigs[k];
  }
  update(t) {
    const done = this.t0('done');
    let current = 0;
    for (let i = 1; i <= this.n; i++) if (t >= this.t0('c' + i)) current = i;
    this.items.forEach((it, i) => {
      const ct = this.t0('c' + (i + 1));
      const counted = t >= ct;
      let jump = 0;
      if (t > done) jump = Math.abs(Math.sin((t - done) * 5 - i * 0.5)) * 22 * S * clamp((done + 2.5 - t) / 1.0);
      if (counted) {
        // drop in from above with a bounce
        const k = prog(t, ct, 0.45);
        const fall = (1 - easeOutBack(k, 2.6)) * 120 * S;
        this.setVis(it.v, t, 1, wob(t, i, 0.05), fall + jump + bob(t, i, 4 * S));
        const bs = pop(t, ct + 0.1, 0.35);
        it.badge.visible = bs > 0.02; it.badge.scale.setScalar(Math.max(0.0001, bs));
        it.badge.position.copy(it.badge.userData.home); it.badge.position.y += jump * upp(0.4);
        it.badge.rotation.y = Math.sin(t * 2 + i) * 0.3;
      } else {
        this.setVis(it.v, t, pop(t, 0.1 + 0.05 * i), 0, 0, 0.3);
        it.badge.visible = false;
      }
    });
    Object.values(this.bigs).forEach(c => (c.visible = false));
    if (current) {
      const c = this.big(current); c.visible = true;
      const sc = pop(t, this.t0('c' + current), 0.35); c.scale.setScalar(Math.max(0.0001, sc)); c.rotation.y = (1 - Math.min(1, sc)) * 3 + Math.sin(t * 1.5) * 0.2;
    }
    if (this.lab) { this.lab.visible = t > done; animateLetters(this.lab, t, done, { stagger: 0.04 }); }
  }
}

class Letter extends Scene {
  build() {
    const [x, y, w, h] = contentBox();
    const up = trUpper(this.s.letter);
    const lo = up === 'I' ? 'ı' : up === 'İ' ? 'i' : up.toLocaleLowerCase('tr-TR');
    this.up = up;
    const lz = portrait ? [x, y, w, h * 0.36] : [x, y, w * 0.42, h];
    const wz = portrait ? [x, y + h * 0.38, w, h * 0.62] : [x + w * 0.44, y, w * 0.56, h];
    const [lx, ly, lw, lh] = lz;
    const big = Math.min(lw * 0.6, lh * 0.75);
    this.U = text3d(up, { heightPx: big, color: this.th.accent, z: 0.2, maxWpx: lw * 0.55 });
    this.l = text3d(lo, { heightPx: big * 0.75, color: this.th.accent2, z: 0.2, maxWpx: lw * 0.4 });
    placeAt(this.U, lx + lw * 0.32, ly + lh * 0.5, 0.2); placeAt(this.l, lx + lw * 0.76, ly + lh * 0.56, 0.2);
    this.g.add(this.U); this.g.add(this.l);
    const [wx, wy, ww, wh] = wz;
    const words = this.s.words, n = words.length, rh = wh / Math.max(n, 3), y0 = wy + (wh - rh * n) / 2;
    this.rows = words.map((wd, i) => {
      const cy = y0 + (i + 0.5) * rh;
      const card = card3d(ww * 0.94, rh * 0.84, -0.2); placeAt(card, wx + ww / 2, cy, -0.2); this.g.add(card);
      card.userData.home = card.position.clone();
      const vs = rh * 0.7;
      const v = this.vis(wd.visual, wx + ww * 0.06 + vs * 0.6, cy, vs, 0.1);
      const word = wd.word; const k = Math.max(0, trUpper(word).indexOf(up));
      const html = esc(word.slice(0, k)) + `<span style="color:${readableOnWhite(this.th.accent)}">${esc(word.slice(k, k + 1))}</span>` + esc(word.slice(k + 1));
      const d = this.dom(html, 'title', { color: this.th.dark });
      fitText(d, ww * 0.6, rh * 0.62, Math.min(rh * 0.55, 92 * S));
      return { card, v, d, pos: [wx + ww * 0.06 + vs * 1.25, cy] };
    });
  }
  update(t) {
    const p = this.pulse(t, this.lineTags());
    const sc = pop(t, 0.15, 0.6);
    this.U.scale.setScalar(Math.max(0.0001, sc * p)); this.U.rotation.y = Math.sin(t * 0.9) * 0.35; this.U.rotation.z = wob(t, 0, 0.04);
    const sl = pop(t, 0.45) * p; this.l.scale.setScalar(Math.max(0.0001, sl)); this.l.rotation.y = Math.sin(t * 0.9 + 1.5) * 0.35;
    this.rows.forEach((r, i) => {
      const a = this.t0('w' + i), s = pop(t, a), hl = this.active('w' + i, t), k = hl ? 1.06 : 1;
      r.card.visible = s > 0.02; r.card.scale.setScalar(Math.max(0.0001, s * k));
      cardGlow(r.card, hl ? this.th.accent : null, 0.18);
      r.card.rotation.x = (1 - Math.min(s, 1)) * 1.2;
      this.setVis(r.v, t, s * k, wob(t, i, 0.05), bob(t, i, 4 * S));
      r.d.style.left = '0px';
      place(r.d, r.pos[0] + r.d.offsetWidth / 2, r.pos[1], s * k);
    });
  }
}

class Compare extends Scene {
  build() {
    const [x, y, w, h] = contentBox();
    const th = this.s.title ? h * 0.16 : 0;
    if (this.s.title) { this.ttl = this.titleDom(this.s.title, w * 0.9, th, 96); this.ttlPos = [x + w / 2, y + th / 2]; }
    const boxes = portrait
      ? [[x + w / 2, y + th + (h - th) * 0.25, w * 0.9, (h - th) * 0.46], [x + w / 2, y + th + (h - th) * 0.75, w * 0.9, (h - th) * 0.46]]
      : [[x + w * 0.25, y + th + (h - th) / 2, w * 0.46, (h - th) * 0.9], [x + w * 0.75, y + th + (h - th) / 2, w * 0.46, (h - th) * 0.9]];
    const labs = uniformLabels(this, [this.s.left.label, this.s.right.label], boxes[0][2] * 0.8, boxes[0][3] * 0.16, 64);
    this.sides = ['left', 'right'].map((side, i) => {
      const [cx, cy, cw, ch] = boxes[i], it = this.s[side];
      const card = card3d(cw, ch, -0.3); placeAt(card, cx, cy, -0.3); this.g.add(card);
      const vs = Math.min(cw * 0.62, ch * 0.62) * Number(it.scale || 1);
      const v = this.vis(it.visual, cx, cy - ch * 0.08, vs, 0.1);
      return { card, v, lab: labs[i], labPos: [cx, cy + ch * 0.37], tag: i ? 'R' : 'L' };
    });
  }
  update(t) {
    if (this.ttl) place(this.ttl, this.ttlPos[0], this.ttlPos[1], pop(t, 0.1));
    this.sides.forEach((sd, i) => {
      const s = pop(t, this.t0(sd.tag)), hl = this.active(sd.tag, t), k = hl ? 1.05 : 1;
      sd.card.visible = s > 0.02; sd.card.scale.setScalar(Math.max(0.0001, s * k));
      sd.card.rotation.y = (1 - Math.min(s, 1)) * (i ? -1.2 : 1.2) + Math.sin(t * 0.7 + i) * 0.03;
      cardGlow(sd.card, hl ? this.th.accent : null, 0.18);
      this.setVis(sd.v, t, s * k * this.pulse(t, [sd.tag]), wob(t, i, 0.05), bob(t, i, 6 * S));
      sd.v.rotation.y = Math.sin(t + i) * 0.3;
      place(sd.lab, sd.labPos[0], sd.labPos[1], s * k);
    });
  }
}

class Quiz extends Scene {
  build() {
    const [x, y, w, h] = contentBox();
    const qh = h * 0.2; this.qh = qh;
    let pillCol = this.th.accent2;
    for (let i = 0; i < 8 && contrast('#ffffff', pillCol) < 4; i++) pillCol = darken(pillCol, 0.12);   // white question text needs 4:1
    this.q = this.dom(esc(this.s.question), 'title pill', { background: pillCol, color: '#fff', padding: `${Math.round(10 * S)}px ${Math.round(44 * S)}px`,
      boxShadow: '0 10px 24px rgba(40,25,80,.25)' });
    fitText(this.q, w * 0.86, qh * 0.85, 90 * S);
    this.qPos = [x + w / 2, y + qh / 2];
    this.opts = this.s.options; this.answer = this.s.answer;
    const cells = gridCells(this.opts.length, [x, y + qh, w, h - qh], portrait, 4);
    const side = Math.min(cells[0][2] * 0.86, cells[0][3] * 0.9);
    const labs = uniformLabels(this, this.opts.map(o => o.label), side * 0.86, side * 0.17, 54);
    this.cards = cells.map(([cx, cy], i) => {
      const card = card3d(side, side, -0.3); placeAt(card, cx, cy, -0.3); this.g.add(card); card.userData.home = card.position.clone();
      const v = this.vis(this.opts[i].visual, cx, cy - side * 0.08, side * 0.62, 0.1);
      return { card, v, lab: labs[i], cx, cy, side };
    });
    const c = this.cards[this.answer];
    this.check = this.vis('✅', c.cx + side * 0.42, c.cy - side * 0.42, side * 0.3, 0.6);
    this.coins = {};
    if (this.s.mood !== 'calm') { this.confetti = makeConfetti(1000 + this.index, [c.cx, c.cy]); this.g.add(this.confetti); }
    this.ring = this.dom('<svg viewBox="0 0 100 100" width="100%" height="100%"><circle cx="50" cy="50" r="44" fill="none" stroke="rgba(255,255,255,.85)" stroke-width="9"/>' +
      `<circle class="arc" cx="50" cy="50" r="44" fill="none" stroke="${this.th.accent2}" stroke-width="9" stroke-linecap="round" transform="rotate(-90 50 50)" stroke-dasharray="276.5" stroke-dashoffset="0"/></svg>`, '',
      { width: Math.round(Math.min(220 * S, qh * 1.2)) + 'px', height: Math.round(Math.min(220 * S, qh * 1.2)) + 'px' });
  }
  coin(k) {
    if (!this.coins[k]) { const c = coin3d(String(k), this.th.accent, Math.min(170 * S, this.qh * 0.9), 0.5); placeAt(c, this.qPos[0], this.qPos[1], 0.5); this.g.add(c); this.coins[k] = c; }
    return this.coins[k];
  }
  update(t) {
    const reveal = this.t0('reveal');
    const counting = this.t0('cd3') <= t && t < this.t1('cd1');
    place(this.q, this.qPos[0], this.qPos[1], pop(t, 0.05), counting ? 0 : 1);
    this.cards.forEach((c, i) => {
      const sc = pop(t, 0.45 + 0.22 * i);
      let k = 1, op = 1, spin = 0;
      const correct = i === this.answer;
      if (t >= reveal) {
        const pr = prog(t, reveal, 0.6);
        if (correct) { k = 1 + 0.14 * easeOutBack(pr, 2.5); spin = easeInOut(pr) * Math.PI * 2; }
        else { k = 1 - 0.1 * easeInOut(pr); op = 1 - 0.62 * easeInOut(pr); }
      }
      c.card.visible = sc > 0.02; c.card.scale.setScalar(Math.max(0.0001, sc * k));
      c.card.rotation.y = spin + (1 - Math.min(sc, 1)) * 1.4;
      c.card.rotation.x = (!correct && t >= reveal) ? easeInOut(prog(t, reveal, 0.6)) * 0.35 : 0;
      c.card.position.copy(c.card.userData.home); c.card.position.z += (correct && t >= reveal) ? 0.6 : 0;
      cardGlow(c.card, correct && t >= reveal ? '#33C96B' : null, 0.4 * easeInOut(prog(t, reveal, 0.4)));
      setOpacity(c.card.material, op);
      this.setVis(c.v, t, sc * k, wob(t, i, 0.05), bob(t, i, 5 * S), op);
      c.v.rotation.y = Math.sin(spin) * 0.35;
      c.v.position.z += (correct && t >= reveal) ? 0.7 : 0;
      place(c.lab, c.cx, c.cy + c.side * 0.36, sc * k, op);
    });
    this.setVis(this.check, t, t >= reveal ? pop(t, reveal + 0.35) : 0, 0, 0);
    Object.values(this.coins).forEach(c => (c.visible = false));
    let anyCd = false;
    for (const kk of [3, 2, 1]) {
      const r = this.T['cd' + kk];
      if (r && r[0] <= t && t < r[1]) {
        anyCd = true;
        const c = this.coin(kk); c.visible = true;
        const s = pop(t, r[0], 0.3); c.scale.setScalar(Math.max(0.0001, s)); c.rotation.y = (1 - Math.min(1, s)) * 4;
        place(this.ring, this.qPos[0], this.qPos[1], 1);
        this.ring.querySelector('.arc').setAttribute('stroke-dashoffset', String(276.5 * (t - r[0]) / (r[1] - r[0])));
      }
    }
    if (!anyCd) this.ring.style.opacity = '0';
    if (this.confetti) this.confetti.userData.update(t - reveal);
  }
  mascot(t) {
    const rv = this.t0('reveal');
    const calm = this.s.mood === 'calm';
    return { waving: false, happy: rv <= t && t < rv + 1.6, jump: (!calm && rv <= t && t < rv + 0.6) ? Math.sin(Math.PI * clamp((t - rv) / 0.6)) : 0, big: false };
  }
}

class Chant extends Scene {
  build() {
    const [x, y, w, h] = contentBox();
    if (this.s.title) { this.ttl = this.titleDom(this.s.title, w * 0.8, h * 0.13, 80); this.ttlPos = [x + w / 2, y + h * 0.07]; }
    this.lines = this.s.items.map((it, i) => {
      const v = it.visual || this.s.visual;
      const tx = text3d(it.line, { heightPx: 118 * S, maxWpx: w * 0.94, maxHpx: h * 0.3, color: '#ffffff', side: this.th.dark });
      placeAt(tx, x + w / 2, y + h * (v ? 0.8 : 0.5), 0.3); tx.visible = false; this.g.add(tx);
      const vg = v ? this.vis(v, x + w / 2, y + h * 0.36, h * 0.4) : null;
      if (vg) vg.visible = false;
      return { tx, vg };
    });
    this.note = this.vis('🎵', x + w / 2, y + h * 0.45, 160 * S);
  }
  update(t) {
    if (this.ttl) place(this.ttl, this.ttlPos[0], this.ttlPos[1], pop(t, 0.1));
    let cur = -1;
    this.lines.forEach((_, i) => { if (t >= this.t0('c' + i)) cur = i; });
    this.setVis(this.note, t, cur < 0 ? 1 : 0, wob(t, 0, 0.1), bob(t, 0, 12 * S));
    this.lines.forEach((ln, i) => {
      const on = i === cur;
      ln.tx.visible = on; if (ln.vg) ln.vg.visible = on;
      if (!on) return;
      const a = this.t0('c' + i), beat = Math.abs(Math.sin((t - a) * 4.2));
      animateLetters(ln.tx, t, a, { stagger: 0.025, dur: 0.4, wave: 0.05 });
      ln.tx.rotation.y = Math.sin(t * 0.8) * 0.1;
      if (ln.vg) { this.setVis(ln.vg, t, pop(t, a, 0.35), wob(t, i, 0.1), beat * 20 * S); ln.vg.rotation.y = Math.sin(t * 1.4) * 0.4; }
    });
  }
}

class Story extends Scene {
  build() {
    const [x, y, w, h] = contentBox();
    this.panels = this.s.panels;
    const u = upp(-0.6);
    // puppet-theatre stage: wooden floor strip, red curtains and a valance
    const pw = w * 0.98, ph = h * 0.94, cx = x + w / 2, cy = y + h / 2;
    const stage = new THREE.Group(); this.g.add(stage); this.stage = stage;
    const back = new THREE.Mesh(new THREE.PlaneGeometry(pw * u, ph * u), new THREE.MeshStandardMaterial({ map: radialTex('#fff8ec', '#f3dcc0'), roughness: 0.9 }));
    placeAt(back, cx, cy, -1.2); stage.add(back);
    const floor = shadowy(new THREE.Mesh(new THREE.BoxGeometry(pw * u, 0.35, 2.2), matte('#c98a52')), false, true);
    floor.position.copy(at(cx, cy + ph * 0.42, -0.4)); stage.add(floor);
    const curtainMat = new THREE.MeshPhysicalMaterial({ color: '#c8102e', roughness: 0.55, sheen: 1, sheenColor: col('#ff8a9a') });
    for (const sx of [-1, 1]) {
      for (let k = 0; k < 5; k++) {
        const c = shadowy(new THREE.Mesh(new THREE.CylinderGeometry(0.28, 0.32, ph * u * 1.02, 16), curtainMat));
        c.position.copy(at(cx + sx * (pw / 2 - (k + 0.5) * 22 * S), cy, -0.9 + k * 0.05)); stage.add(c);
      }
    }
    const val = shadowy(new THREE.Mesh(new THREE.BoxGeometry(pw * u * 1.04, ph * u * 0.12, 0.5), curtainMat));
    val.position.copy(at(cx, y + ph * 0.04, -0.7)); stage.add(val);
    const gold = new THREE.Mesh(new THREE.BoxGeometry(pw * u * 1.04, 0.12, 0.55), plastic('#ffc93c'));
    gold.position.copy(at(cx, y + ph * 0.1, -0.68)); stage.add(gold);
    {
      this.ttl = text3d(this.s.title || 'Bir varmış, bir yokmuş...', { heightPx: 100 * S, maxWpx: w * 0.8, maxHpx: h * 0.3, color: '#ffffff', side: this.th.dark });
      placeAt(this.ttl, cx, cy, 0.2); this.g.add(this.ttl);
    }
    this.pw = pw; this.ph = ph; this.cx = cx; this.cy = cy;
    this.panelObjs = this.panels.map((p, j) => {
      const vs = p.visuals || []; const n = vs.length;
      const hasCap = !!p.caption; const areaH = ph * (hasCap ? 0.68 : 0.86);
      const size = Math.min(pw * 0.7 / (n + 0.5), areaH * (n === 1 ? 0.82 : 0.6));
      const grp = new THREE.Group(); this.g.add(grp);
      const items = vs.map((v, i) => {
        const g = makeVisual(v); sizeVisual(g, size, 0); placeAt(g, cx - pw * 0.35 + pw * 0.7 * (i + 0.5) / n, y + ph * 0.06 + areaH * 0.58, 0);
        g.userData.home = g.position.clone(); g.userData.z = 0; grp.add(g); return g;
      });
      let cap = null;
      if (hasCap) { cap = this.dom(esc(p.caption), 'body card', { color: this.th.dark, padding: `${Math.round(12 * S)}px ${Math.round(28 * S)}px` }); fitText(cap, pw * 0.7, ph * 0.16, 56 * S); }
      return { grp, items, cap };
    });
  }
  update(t) {
    let cur = -1;
    this.panels.forEach((_, j) => { if (t >= this.t0('p' + j)) cur = j; });
    if (this.ttl) { const on = cur < 0; this.ttl.visible = on; if (on) { animateLetters(this.ttl, t, 0.1, { stagger: 0.03 }); } }
    const pr = cur >= 0 ? easeOutCubic(prog(t, this.t0('p' + cur), 0.45)) : 0;
    const shift = W * upp(0);
    this.panelObjs.forEach((po, j) => {
      let dx = null;
      if (j === cur) dx = (1 - pr);
      else if (j === cur - 1 && pr < 1) dx = -pr;
      po.grp.visible = dx !== null;
      if (po.cap) place(po.cap, this.cx + (dx ?? 0) * W, this.cy + this.ph * 0.33, dx === null ? 0 : 1);
      if (dx === null) return;
      po.grp.position.x = dx * shift;
      po.items.forEach((g, i) => { g.position.copy(g.userData.home); g.position.y += bob(t, i * 1.4, 8 * S) * upp(0); g.rotation.z = wob(t, i, 0.06); g.rotation.y = Math.sin(t + i) * 0.25; if (g.userData.update) g.userData.update(t); });
    });
  }
}

class MathScene extends Scene {
  build() {
    const [x, y, w, h] = contentBox();
    const a = this.s.a, b = this.s.b, op = this.s.op || '+';
    this.a = a; this.b = b; this.op = op; this.res = op === '+' ? a + b : a - b;
    let slots, gh;
    if (portrait) {
      const r1 = y + h * 0.28, r2 = y + h * 0.74;
      slots = { A: [x + w * 0.2, r1, w * 0.36], op: [x + w * 0.5, r1, w * 0.18], B: [x + w * 0.8, r1, w * 0.36], eq: [x + w * 0.22, r2, w * 0.2], R: [x + w * 0.62, r2, w * 0.55] };
      gh = h * 0.36;
    } else {
      const r = y + h * 0.42;
      slots = { A: [x + w * 0.13, r, w * 0.25], op: [x + w * 0.31, r, w * 0.1], B: [x + w * 0.49, r, w * 0.25], eq: [x + w * 0.67, r, w * 0.1], R: [x + w * 0.86, r, w * 0.27] };
      gh = h * 0.56;
    }
    const grid = c => { const cols = c === 1 ? 1 : c <= 4 ? 2 : c <= 9 ? 3 : 4; const rows = Math.ceil(c / cols); return [cols, rows]; };
    const cellOf = (c, gw) => { const [cols, rows] = grid(c); return Math.min(gw / cols, gh / rows); };
    const cells = [['A', a], ['B', b], ['R', this.res]].filter(([, c]) => c > 0).map(([k, c]) => cellOf(c, slots[k][2]));
    const shared = cells.length ? Math.min(...cells) : 100;
    this.groups = {};
    for (const [k, count, tag, crossed] of [['A', a, 'a', false], ['B', b, 'b', op === '-'], ['R', this.res, 'res', false]]) {
      const [sx, sy] = slots[k]; const items = [];
      if (count === 0) {
        const z = text3d('0', { heightPx: gh * 0.4, color: '#ffffff', side: this.th.dark }); placeAt(z, sx, sy, 0.2); this.g.add(z);
        items.push({ zero: z });
      } else {
        const [cols, rows] = grid(count);
        for (let i = 0; i < count; i++) {
          const r = Math.floor(i / cols), c = i % cols, inRow = r < rows - 1 ? cols : count - cols * (rows - 1);
          const px = sx - inRow * shared / 2 + (c + 0.5) * shared, py = sy - rows * shared / 2 + (r + 0.5) * shared;
          const v = this.vis(this.s.visual, px, py, shared * 0.84);
          const xm = crossed ? this.vis('❌', px + shared * 0.21, py - shared * 0.21, shared * 0.42, 0.3) : null;
          items.push({ v, xm });
        }
      }
      const half = count ? grid(count)[1] * shared / 2 : gh * 0.2;
      const coin = coin3d(String(count), this.th.accent2, 110 * S, 0.4); placeAt(coin, sx, sy + half + 70 * S, 0.4); this.g.add(coin);
      this.groups[k] = { items, tag, coin, count };
    }
    this.syms = [];
    for (const [k, sym, tag] of [['op', op === '+' ? '+' : '−', 'op'], ['eq', '=', 'eq']]) {
      const [sx, sy] = slots[k];
      const tx = text3d(sym, { heightPx: 150 * S, color: this.th.accent, side: '#' + mixHex(this.th.accent, '#000', 0.4).getHexString() });
      placeAt(tx, sx, sy, 0.2); this.g.add(tx); this.syms.push({ tx, tag });
    }
  }
  update(t) {
    for (const k of ['A', 'B', 'R']) {
      const gr = this.groups[k], appear = this.t0(gr.tag);
      gr.items.forEach((it, i) => {
        if (it.zero) { const s = pop(t, appear); it.zero.scale.setScalar(Math.max(0.0001, s)); it.zero.visible = s > 0.02; return; }
        const s = pop(t, appear + Math.min(0.06, 0.6 / gr.count) * i, 0.4);
        this.setVis(it.v, t, s, wob(t, i, 0.05), bob(t, i, 3 * S));
        if (it.xm) this.setVis(it.xm, t, s > 0.5 ? s : 0);
      });
      const cs = pop(t, appear + 0.2); gr.coin.visible = cs > 0.02; gr.coin.scale.setScalar(Math.max(0.0001, cs)); gr.coin.rotation.y = Math.sin(t * 1.3) * 0.3;
    }
    this.syms.forEach(({ tx, tag }) => { const s = pop(t, this.t0(tag)); tx.visible = s > 0.02; tx.scale.setScalar(Math.max(0.0001, s)); tx.rotation.y = Math.sin(t) * 0.3; });
  }
}

class Action extends Scene {
  build() {
    const [x, y, w, h] = contentBox();
    this.v = this.vis(this.s.visual, x + w / 2, y + h * 0.4, h * 0.46);
    this.tx = text3d(trUpper(this.s.text), { heightPx: 150 * S, maxWpx: w * 0.8, maxHpx: h * 0.22, color: this.th.accent, side: '#' + mixHex(this.th.accent, '#000', 0.4).getHexString() });
    placeAt(this.tx, x + w / 2, y + h * 0.84, 0.3); this.g.add(this.tx);
    this.coins = {};
    this.bar = this.dom(`<div style="width:100%;height:100%;border-radius:999px;background:rgba(255,255,255,.75);overflow:hidden">
      <div class="fill" style="height:100%;width:100%;border-radius:999px;background:${this.th.accent2}"></div></div>`, '', { width: Math.round(w * 0.6) + 'px', height: Math.round(22 * S) + 'px' });
    this.barPos = [x + w / 2, y + h * 0.99];
    this.coinPos = [x + w - 100 * S, y + 100 * S];
  }
  update(t) {
    const [a, b] = this.T.go || [1e9, 1e9];
    const going = a <= t && t < b;
    const jump = going ? Math.abs(Math.sin((t - a) * Math.PI / 0.6)) * 60 * S : 0;
    this.setVis(this.v, t, pop(t, 0.1), wob(t, 0, going ? 0.12 : 0.05), jump + bob(t, 0, 6 * S));
    this.v.rotation.y = going ? Math.sin((t - a) * 4) * 0.5 : Math.sin(t) * 0.2;
    animateLetters(this.tx, t, 0.3, { stagger: 0.05, wave: going ? 0.06 : 0.02 });
    Object.values(this.coins).forEach(c => (c.visible = false));
    if (going) {
      const left = Math.max(1, Math.ceil(b - t));
      if (!this.coins[left]) { const c = coin3d(String(left), this.th.accent2, 150 * S, 0.5); placeAt(c, this.coinPos[0], this.coinPos[1], 0.5); this.g.add(c); this.coins[left] = c; }
      const c = this.coins[left]; c.visible = true; const s = pop(t, b - left, 0.3); c.scale.setScalar(Math.max(0.0001, s)); c.rotation.y = (1 - Math.min(1, s)) * 4;
      place(this.bar, this.barPos[0], this.barPos[1], 1);
      this.bar.querySelector('.fill').style.width = (100 * (1 - (t - a) / (b - a))) + '%';
    } else this.bar.style.opacity = '0';
  }
  mascot(t) {
    const [a, b] = this.T.go || [1e9, 1e9];
    return { waving: a <= t && t < b, happy: a <= t && t < b + 1, jump: (a <= t && t < b) ? Math.abs(Math.sin((t - a) * Math.PI / 0.6)) : 0, big: false };
  }
}

const SCENES = { intro: Intro, outro: Outro, talk: Talk, item: Item, fact: Fact, gallery: Gallery, recap: Recap, count: Count,
  letter: Letter, compare: Compare, quiz: Quiz, chant: Chant, story: Story, math: MathScene, action: Action };

// ------------------------------------------------------------------ frame loop
function sceneAt(t) {
  let i = 0;
  for (let k = 0; k < scenes.length; k++) if (scenes[k].start <= t) i = k;
  return i;
}

function updateCaption(t) {
  if (P.show_captions === false) { captionEl.style.opacity = '0'; return; }
  let cur = null;
  for (const c of P.captions) { if (c[0] <= t && t <= c[1] + 0.25) cur = c; if (c[0] > t) break; }
  if (!cur) { captionEl.style.opacity = '0'; return; }
  const [a, b, text] = cur;
  if (captionText !== text) {
    captionEl.innerHTML = esc(stripEmoji(text)); captionText = text;
    fitText(captionEl, L.caption[2], 140 * S, 50 * S, 26 * S);
  }
  const o = clamp((t - a) / 0.12) * clamp((b + 0.25 - t) / 0.15);
  place(captionEl, L.caption[0], L.caption[1], 1, o);
}

function updateIris(i, sc, lt) {
  const TR = 0.32;
  const R = Math.hypot(W, H) / 2 + 10;
  let r = null, cover = true;
  if (i < scenes.length - 1 && lt > sc.dur - TR) { r = R * easeInOut((lt - (sc.dur - TR)) / TR); cover = true; }
  else if (i > 0 && lt < TR) { r = R * easeInOut(lt / TR); cover = false; }
  if (r === null) { irisEl.style.display = 'none'; return; }
  irisEl.style.display = 'block';
  irisEl.style.background = P.theme.accent;
  irisEl.style.clipPath = cover ? `circle(${r}px at 50% 50%)` : 'none';
  irisEl.style.webkitMaskImage = cover ? 'none' : `radial-gradient(circle at 50% 50%, transparent ${r}px, #000 ${r + 1}px)`;
  irisEl.style.maskImage = irisEl.style.webkitMaskImage;
}

function renderFrame(fi) {
  const t = fi / P.fps;
  const i = sceneAt(t);
  const sc = scenes[i];
  if (!sc.built) { sc.build(); sc.built = true; }
  scenes.forEach((s, k) => { if (k !== i && s.g.visible) s.show(false); });
  sc.show(true);
  const lt = t - sc.start;
  sc.update(lt);
  world.update(t);
  updateMascot(t, fi, sc.mascot(lt));
  updateCaption(t);
  updateIris(i, sc, lt);
  // gentle camera drift for depth (pure translation, so the layout stays put on screen)
  const dx = Math.sin(t * 0.21) * 0.12, dy = Math.sin(t * 0.17 + 1) * 0.06;
  camera.position.set(camRef.position.x + dx, camRef.position.y + dy, camRef.position.z);
  camera.quaternion.copy(camRef.quaternion);
  const shiftPx = -dx / upp(0), shiftPy = dy / upp(0);
  ui.style.transform = `translate(${shiftPx.toFixed(2)}px, ${shiftPy.toFixed(2)}px)`;
  captionEl.style.marginLeft = (-shiftPx).toFixed(2) + 'px'; captionEl.style.marginTop = (-shiftPy).toFixed(2) + 'px';
  chipEl.style.marginLeft = (-shiftPx).toFixed(2) + 'px'; chipEl.style.marginTop = (-shiftPy).toFixed(2) + 'px';
  renderer.render(scene, camera);
  return true;
}

// ------------------------------------------------------------------ thumbnail
// YouTube cover: the episode's world, a big 3D title on the left, the topic's pictures on the right and
// Pofi waving in the corner. The plan has no scenes, only `thumb: {text, visuals}`.
function renderThumb() {
  const T = P.thumb || { text: P.episode.title, visuals: [] };
  const k = W / 1280;
  const stage = document.getElementById('stage');
  stage.style.background = 'repeating-conic-gradient(from 0deg at 66% 48%, rgba(255,255,255,0.2) 0deg 10deg, rgba(255,255,255,0) 10deg 20deg), '
    + stage.style.background;
  const g = new THREE.Group(); scene.add(g);
  const title = text3d(trUpper(T.text), { heightPx: 150 * k, maxWpx: W * 0.52, maxHpx: H * 0.5, color: '#ffffff', side: P.theme.dark, z: 0.6, glow: 0.45 });
  placeAt(title, W * 0.32, H * 0.31, 0.6); title.rotation.set(-0.06, 0.2, 0.025); g.add(title);
  animateLetters(title, 10, 0, {});
  const vs = (T.visuals || []).slice(0, 3), n = vs.length;
  const sizes = { 1: [430], 2: [360, 260], 3: [330, 240, 210] }[n] || [];
  const spots = { 1: [[0.74, 0.47]], 2: [[0.71, 0.43], [0.89, 0.69]], 3: [[0.71, 0.41], [0.89, 0.69], [0.89, 0.23]] }[n] || [];
  vs.forEach((v, i) => {
    const m = makeVisual(v); sizeVisual(m, sizes[i] * k, 0.3); placeAt(m, W * spots[i][0], H * spots[i][1], 0.3 - i * 0.2);
    m.rotation.set(0, -0.25 + i * 0.2, i ? -0.1 : 0.05); if (m.userData.update) m.userData.update(1.2); g.add(m);
  });
  L.mascot = [W * 0.12, H + 40 * k, H * 0.42];
  updateMascot(3.0, 0, { waving: true, happy: false, jump: 0, big: false });
  mascot.rig.rotation.y = 0.35;
  world.update(3.0);
  const band = el(esc(P.episode.channel || ''), 'title pill', { background: P.theme.accent, color: inkOn(P.theme.accent, P.theme.dark),
    padding: `${Math.round(6 * k)}px ${Math.round(34 * k)}px`, fontSize: Math.round(34 * k) + 'px', boxShadow: '0 8px 20px rgba(40,25,80,.3)' });
  place(band, W * 0.69, H - 52 * k, 1, 1);
  chipEl.style.opacity = '0'; captionEl.style.opacity = '0'; irisEl.style.display = 'none';
  camera.position.copy(camRef.position); camera.quaternion.copy(camRef.quaternion);
  renderer.render(scene, camera);
  return true;
}

// ------------------------------------------------------------------ init
async function init(plan) {
  P = plan; W = plan.width; H = plan.height; S = Math.min(W, H) / 1080; portrait = H > W; L = layout();
  const stage = document.getElementById('stage');
  stage.style.width = W + 'px'; stage.style.height = H + 'px';
  ui = document.getElementById('ui'); ui.style.width = W + 'px'; ui.style.height = H + 'px';
  irisEl = document.getElementById('iris');
  const Q = Object.assign({ antialias: true, shadow: 'pcfsoft', shadowSize: 2048, scale: 1 }, plan.render || {});
  P.Q = Q;
  renderer = new THREE.WebGLRenderer({ antialias: Q.antialias, preserveDrawingBuffer: true, alpha: true });
  renderer.setClearColor(0x000000, 0);
  renderer.setPixelRatio(Q.scale); renderer.setSize(W, H);
  renderer.shadowMap.enabled = Q.shadow !== 'none';
  renderer.shadowMap.type = { pcfsoft: THREE.PCFSoftShadowMap, pcf: THREE.PCFShadowMap, basic: THREE.BasicShadowMap, vsm: THREE.VSMShadowMap }[Q.shadow] ?? THREE.PCFShadowMap;
  renderer.toneMapping = THREE.NeutralToneMapping; renderer.toneMappingExposure = 1.0;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  stage.insertBefore(renderer.domElement, ui);
  scene = new THREE.Scene();
  camRef = new THREE.PerspectiveCamera(portrait ? 42 : 30, W / H, 0.1, 400);
  camRef.position.set(0, 1.2, portrait ? 17 : 18); camRef.lookAt(0, 0.2, 0); camRef.updateMatrixWorld(); camRef.updateProjectionMatrix();
  camera = camRef.clone();
  await document.fonts.load(`800 40px Baloo`); await document.fonts.load(`900 40px Nunito`); await document.fonts.ready;
  const fontData = await new Promise((res, rej) => new TTFLoader().load('/fonts/Baloo2-ExtraBold.ttf', res, undefined, rej));
  font3d = new Font(fontData);
  await preloadTextures(Object.values(P.emoji));
  world = buildWorld();
  mascot = buildMascot();
  captionEl = el('', 'body card', { color: P.theme.dark, padding: `${Math.round(10 * S)}px ${Math.round(30 * S)}px`, zIndex: 5 });
  chipEl = el(esc(P.episode.title), 'title pill', { background: 'rgba(255,255,255,.88)', color: P.theme.dark, fontWeight: 600,
    padding: `${Math.round(6 * S)}px ${Math.round(26 * S)}px`, boxShadow: '0 6px 16px rgba(40,25,80,.18)' });
  fitText(chipEl, portrait ? 900 * S : 400 * S, 100 * S, 38 * S, 26 * S);
  place(chipEl, L.chip[0], L.chip[1], 1, 1);
  P.scenes.forEach((d, i) => scenes.push(new (SCENES[d.type])(d, i)));
  window.__ready = true;
  return true;
}

window.pofi = { init, renderFrame, renderThumb, debug: () => ({ THREE, renderer, scene, camera, world, mascot, scenes }) };
