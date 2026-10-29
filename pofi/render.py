"""Builds the timeline of an episode, renders frames in parallel and muxes the final MP4."""
import copy
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import tempfile
import zlib
from bisect import bisect_right
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import audio, gfx
from .config import load_config
from .gfx import blit, clamp, fit_text, rgb, rounded_card, text_sprite
from .mascot import MascotAnimator
from .scenes import SCENES, Layout, Say, Sfx, Wait
from .schema import get_visual_list

THEMES = [
    {"name": "şeker", "bg": ("#FFF0C2", "#FFC2DC"), "accent": "#FF4F8B", "accent2": "#7C5CFA", "dark": "#3D2A7A"},
    {"name": "gökyüzü", "bg": ("#D4F1FF", "#B9C9FF"), "accent": "#1E9BF0", "accent2": "#FF7A45", "dark": "#1D2F6F"},
    {"name": "orman", "bg": ("#E4FBD3", "#B7EBC9"), "accent": "#2FAE5F", "accent2": "#FF8A3D", "dark": "#1F4D3A"},
    {"name": "gün batımı", "bg": ("#FFE3C7", "#FFB4A2"), "accent": "#FF6B3D", "accent2": "#6C4BE0", "dark": "#5A2142"},
    {"name": "lavanta", "bg": ("#F0E6FF", "#CFC2FF"), "accent": "#8A5CF6", "accent2": "#FF5C93", "dark": "#35206E"},
    {"name": "deniz", "bg": ("#D2FAF6", "#A9E4F7"), "accent": "#0FA3B1", "accent2": "#FF6F59", "dark": "#0D3B4F"},
    {"name": "uzay", "bg": ("#2B2D6E", "#5B3E9E"), "accent": "#FFC93C", "accent2": "#FF5CA8", "dark": "#1A1440"},
    {"name": "limon", "bg": ("#FFF9C4", "#FFE08A"), "accent": "#F59E0B", "accent2": "#10B981", "dark": "#5B3A00"},
    {"name": "bayram", "bg": ("#FFE5E5", "#FFC9C9"), "accent": "#E30A17", "accent2": "#1E63D6", "dark": "#5B0A12"},
]
CATEGORY_THEME = {
    "sayilar": 1, "matematik": 1, "harfler": 0, "dil": 0, "renkler": 0, "sekiller": 4, "sanat": 4,
    "hayvanlar": 2, "doga": 2, "bitkiler": 2, "mevsimler": 3, "hava": 1, "bilim": 5, "deneyler": 5,
    "uzay": 6, "vucut": 3, "saglik": 7, "beslenme": 7, "duygular": 4, "degerler": 3, "gorgu": 3,
    "meslekler": 5, "tasitlar": 1, "trafik": 1, "muzik": 0, "ozel-gun": 8, "ulusal-bayram": 8,
    "gunluk-yasam": 7, "evimiz": 7, "dunya": 5, "kultur": 8,
}


def theme_for(ep):
    if isinstance(ep.get("theme"), dict):
        base = dict(THEMES[0])
        base.update(ep["theme"])
        return base
    if isinstance(ep.get("theme"), str):
        for th in THEMES:
            if th["name"] == ep["theme"]:
                return th
    idx = CATEGORY_THEME.get(ep.get("category"))
    if idx is None:
        idx = zlib.crc32(ep["id"].encode()) % len(THEMES)
    return THEMES[idx]


class Ctx:
    def __init__(self, cfg, ep, W, H, fps):
        self.cfg = cfg
        self.ep = ep
        self.W, self.H, self.fps = W, H, fps
        self.layout = Layout(W, H)
        self.theme = theme_for(ep)
        self.cache = {}
        self.mascot_name = cfg["kanal"]["maskot"]
        self.channel = cfg["kanal"]["ad"]

    def fmt(self, s):
        return str(s).replace("{maskot}", self.mascot_name).replace("{kanal}", self.channel)


# ---------------------------------------------------------------- background


class Background:
    def __init__(self, ctx):
        self.ctx = ctx
        W, H = ctx.W, ctx.H
        th = ctx.theme
        base = gfx.vertical_gradient((W, H), th["bg"][0], th["bg"][1]).convert("RGBA")
        rng = np.random.default_rng(zlib.crc32(ctx.ep["id"].encode()))
        blobs = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(blobs)
        dark = th["name"] == "uzay"
        for _ in range(9):
            r = rng.uniform(0.08, 0.22) * max(W, H)
            cx, cy = rng.uniform(0, W), rng.uniform(0, H)
            a = int(rng.uniform(30, 60))
            d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(255, 255, 255, a))
        blobs = blobs.filter(ImageFilter.GaussianBlur(max(W, H) * 0.03))
        base.alpha_composite(blobs)
        if dark:
            st = ImageDraw.Draw(base)
            for _ in range(140):
                x, y = rng.uniform(0, W), rng.uniform(0, H)
                r = rng.uniform(1, 3.2) * ctx.layout.s
                st.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 255, int(rng.uniform(120, 230))))
        self.base = base
        # small floating shapes
        self.floaters = []
        kinds = ["circle", "star", "circle", "heart"]
        for i in range(12):
            size = rng.uniform(18, 46) * ctx.layout.s
            self.floaters.append({
                "x": rng.uniform(0, W), "y": rng.uniform(0, H), "size": size,
                "speed": rng.uniform(12, 30) * ctx.layout.s, "phase": rng.uniform(0, 6.28),
                "kind": kinds[i % len(kinds)],
            })
        self.sprites = {}

    def _floater(self, kind, size):
        key = (kind, int(size))
        if key not in self.sprites:
            S = int(size) * 2
            m = Image.new("L", (S, S), 0)
            d = ImageDraw.Draw(m)
            if kind == "circle":
                d.ellipse((2, 2, S - 2, S - 2), fill=110)
            elif kind == "star":
                d.polygon(gfx._star(S / 2, S / 2, S * 0.48, S * 0.2), fill=130)
            else:
                d.ellipse((S * 0.05, S * 0.12, S * 0.52, S * 0.58), fill=120)
                d.ellipse((S * 0.48, S * 0.12, S * 0.95, S * 0.58), fill=120)
                d.polygon([(S * 0.07, S * 0.42), (S * 0.93, S * 0.42), (S * 0.5, S * 0.92)], fill=120)
            m = m.resize((int(size), int(size)), Image.LANCZOS)
            im = Image.new("RGBA", m.size, (255, 255, 255, 0))
            im.putalpha(m)
            self.sprites[key] = im
        return self.sprites[key]

    def frame(self, t):
        canvas = self.base.copy()
        H = self.ctx.H
        for f in self.floaters:
            y = (f["y"] - f["speed"] * t) % (H + 100) - 50
            x = f["x"] + 18 * math.sin(t * 0.7 + f["phase"])
            blit(canvas, self._floater(f["kind"], f["size"]), x, y)
        return canvas


# ---------------------------------------------------------------- timeline


class Timeline:
    def __init__(self, cfg, ep, W, H, fps, tts):
        self.ctx = Ctx(cfg, ep, W, H, fps)
        self.fps = fps
        gap_default = float(cfg["ses"].get("satir_arasi", 0.45))
        self.scenes = [SCENES[s["type"]](s, ep, self.ctx, i) for i, s in enumerate(ep["scenes"])]
        self.voice, self.sfx, self.captions = [], [], []
        t = 0.0
        for i, sc in enumerate(self.scenes):
            sc.start = t
            lt = sc.lead
            for st in sc.steps():
                if isinstance(st, Sfx):
                    when = st.at if st.at is not None else lt + st.offset
                    self.sfx.append((t + when, st.name, st.gain))
                elif isinstance(st, Wait):
                    sc.T[st.tag] = (lt, lt + st.seconds)
                    lt += st.seconds
                elif isinstance(st, Say):
                    clip = tts.say(st.text)
                    d = len(clip) / audio.SR
                    sc.T[st.tag] = (lt, lt + d)
                    self.voice.append((t + lt, clip))
                    if st.sfx:
                        self.sfx.append((t + lt, st.sfx, 0.55))
                    if st.caption:
                        self.captions.append((t + lt, t + lt + d, st.text))
                    lt += d + (gap_default if st.gap is None else st.gap)
            sc.dur = lt + sc.tail
            if i > 0:
                self.sfx.append((t - 0.22, "whoosh", 0.45))
            t += sc.dur
        self.duration = t
        self.starts = [s.start for s in self.scenes]
        self.n_frames = int(math.ceil(self.duration * fps))

    def scene_at(self, t):
        i = max(0, bisect_right(self.starts, t) - 1)
        return i, self.scenes[i]

    def chapters(self):
        """YouTube chapters: (seconds, title) — only if YouTube's rules are satisfied."""
        out = []
        for sc in self.scenes:
            title = sc.chapter()
            if title and (not out or out[-1][1] != title):
                out.append((sc.start, title))
        if not out or out[0][0] > 0.01:
            out.insert(0, (0.0, "Giriş"))
        merged = [out[0]]
        for s, title in out[1:]:
            if s - merged[-1][0] >= 10:
                merged.append((s, title))
        while len(merged) > 1 and self.duration - merged[-1][0] < 10:
            merged.pop()
        return merged if len(merged) >= 3 else []


# ---------------------------------------------------------------- frame compositing


class FrameRenderer:
    TR = 0.32  # transition half-length

    def __init__(self, tl: Timeline, mouth_env, captions_on=True):
        self.tl = tl
        self.ctx = tl.ctx
        self.bg = Background(self.ctx)
        self.anim = MascotAnimator(mouth_env, tl.fps)
        self.captions_on = captions_on
        self.cap_starts = [c[0] for c in tl.captions]
        self._chip = None

    def chip(self):
        if self._chip is None:
            s = self.ctx.layout.s
            ep = self.ctx.ep
            txt = text_sprite(self.ctx.fmt(ep["title"]), "title2", int(40 * s), rgb(self.ctx.theme["dark"]),
                              shadow=False)
            w = min(txt.width + 50 * s, 400 * s if not self.ctx.layout.portrait else 900 * s)
            if txt.width > w - 40 * s:
                txt = fit_text(self.ctx.fmt(ep["title"]), "title2", int(w - 40 * s), int(100 * s), int(40 * s),
                               min_size=int(32 * s), fill=rgb(self.ctx.theme["dark"]), shadow=False)
            ph = max(66 * s, txt.height + 12 * s)  # grow for a two-line title instead of squeezing it
            pill = rounded_card(int(w), int(ph), int(min(33 * s, ph / 2)), (255, 255, 255), 215).copy()
            blit(pill, txt, pill.width / 2, pill.height / 2 - 6 * s)
            self._chip = pill
        return self._chip

    def caption_at(self, t):
        i = bisect_right(self.cap_starts, t) - 1
        if i < 0:
            return None
        a, b, text = self.tl.captions[i]
        if t <= b + 0.25:
            return i, a, b, text
        return None

    def draw_caption(self, canvas, t):
        c = self.caption_at(t)
        if not c:
            return
        i, a, b, text = c
        s = self.ctx.layout.s
        cx, cy, maxw = self.ctx.layout.caption
        key = ("cap", i)
        spr = self.ctx.cache.get(key)
        if spr is None:
            txt = fit_text(self.ctx.fmt(gfx_strip(text)), "body", int(maxw - 70 * s), int(130 * s), int(50 * s),
                           min_size=26, fill=rgb(self.ctx.theme["dark"]), shadow=False)
            card = rounded_card(int(min(maxw, txt.width + 60 * s)), int(txt.height + 22 * s), int(30 * s),
                                (255, 255, 255), 225, shadow=True).copy()
            blit(card, txt, card.width / 2, card.height / 2 - 8 * s)
            self.ctx.cache[key] = spr = card
        op = clamp((t - a) / 0.12) * clamp((b + 0.25 - t) / 0.15)
        blit(canvas, spr, cx, cy, opacity=op)

    def draw_mascot(self, canvas, scene, lt, t):
        info = scene.mascot(lt)
        L = self.ctx.layout
        x, bottom, h = L.mascot
        if info.get("big") and not L.portrait:
            h *= 1.12
        h = int(h)
        jump = info.get("jump", 0) * 70 * L.s
        y = bottom - jump + 5 * L.s * math.sin(t * 2.6)
        sh_w = h * 0.62 * (1 - 0.3 * info.get("jump", 0))
        key = ("shadow", int(sh_w))
        sh = self.ctx.cache.get(key)
        if sh is None:
            S = Image.new("L", (int(sh_w) + 40, int(sh_w * 0.22) + 40), 0)
            ImageDraw.Draw(S).ellipse((20, 20, 20 + sh_w, 20 + sh_w * 0.22), fill=80)
            S = S.filter(ImageFilter.GaussianBlur(8))
            sh = Image.new("RGBA", S.size, (40, 25, 80, 0))
            sh.putalpha(S)
            self.ctx.cache[key] = sh
        blit(canvas, sh, x, bottom - 8 * L.s)
        spr = self.anim.pose(t, h, info.get("waving", False), info.get("happy", False))
        blit(canvas, spr, x, y + 6 * L.s, anchor="bottom")

    def draw_transition(self, canvas, i, scene, lt):
        W, H = self.ctx.W, self.ctx.H
        R = math.hypot(W, H) / 2 + 10
        col = rgb(self.ctx.theme["accent"]) + (255,)
        if i < len(self.tl.scenes) - 1 and lt > scene.dur - self.TR:
            k = gfx.ease_in_out((lt - (scene.dur - self.TR)) / self.TR)
            r = R * k
            d = ImageDraw.Draw(canvas)
            d.ellipse((W / 2 - r, H / 2 - r, W / 2 + r, H / 2 + r), fill=col)
        elif i > 0 and lt < self.TR:
            k = gfx.ease_in_out(lt / self.TR)
            r = R * k
            mask = Image.new("L", (W, H), 255)
            ImageDraw.Draw(mask).ellipse((W / 2 - r, H / 2 - r, W / 2 + r, H / 2 + r), fill=0)
            over = Image.new("RGBA", (W, H), col)
            over.putalpha(mask)
            canvas.alpha_composite(over)

    def render(self, fi):
        t = fi / self.tl.fps
        i, scene = self.tl.scene_at(t)
        lt = t - scene.start
        canvas = self.bg.frame(t)
        L = self.ctx.layout
        chip = self.chip()
        blit(canvas, chip, L.chip[0], L.chip[1] - 33 * L.s + (chip.height - 36 * L.s) / 2)
        scene.draw(canvas, lt)
        self.draw_mascot(canvas, scene, lt, t)
        if self.captions_on:
            self.draw_caption(canvas, t)
        self.draw_transition(canvas, i, scene, lt)
        return canvas.convert("RGB")


def gfx_strip(text):
    from .text import strip_emoji
    return strip_emoji(text)


# ---------------------------------------------------------------- encoding

_WORKER = {}


def _encode_chunk(args):
    a, b, out = args
    fr = _WORKER["renderer"]
    W, H, fps = fr.ctx.W, fr.ctx.H, fr.tl.fps
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "19",
           "-tune", "animation", "-pix_fmt", "yuv420p", "-g", str(fps * 2), str(out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fi in range(a, b):
        p.stdin.write(fr.render(fi).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg segment {out} failed")
    return out


def mouth_envelope(voice, fps, n_frames):
    hop = audio.SR / fps
    env = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        a = int(i * hop)
        seg = voice[a: a + int(hop)]
        if len(seg):
            env[i] = float(np.sqrt(np.mean(seg ** 2)))
    peak = np.percentile(env[env > 1e-3], 90) if np.any(env > 1e-3) else 1.0
    env = np.clip(env / max(peak, 1e-6), 0, 1)
    env[env < 0.12] = 0
    # light smoothing so the beak doesn't flicker
    out = env.copy()
    for i in range(1, len(env)):
        out[i] = max(env[i], out[i - 1] * 0.55)
    return out


def srt_time(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(tl, path):
    from .text import strip_emoji
    rows = []
    for i, (a, b, text) in enumerate(tl.captions, 1):
        rows.append(f"{i}\n{srt_time(a)} --> {srt_time(b + 0.2)}\n{strip_emoji(tl.ctx.fmt(text))}\n")
    Path(path).write_text("\n".join(rows), encoding="utf-8")


def render_episode(ep, out_path, cfg=None, fmt="long", workers=None, tts=None, preview_only=False):
    """Render an episode dict to out_path (.mp4). Returns the Timeline."""
    from .tts import TTS

    cfg = cfg or load_config()
    if fmt == "short":
        W, H = cfg["shorts"]["genislik"], cfg["shorts"]["yukseklik"]
    else:
        W, H = cfg["video"]["genislik"], cfg["video"]["yukseklik"]
    fps = int(cfg["video"]["fps"])
    tts = tts or TTS(cfg)
    gfx._TRANSFORM_CACHE.clear()  # sprite keys are only unique within one render
    tl = Timeline(cfg, ep, W, H, fps, tts)
    seed = zlib.crc32(str(ep["id"]).encode())
    music = audio.make_music(tl.duration, seed, ep.get("mood", "happy"))
    mixed, voice = audio.mix(tl.duration, tl.voice, tl.sfx, music, float(cfg["video"]["muzik_seviyesi"]),
                             float(cfg["video"]["efekt_seviyesi"]))
    env = mouth_envelope(voice, fps, tl.n_frames)
    renderer = FrameRenderer(tl, env, bool(cfg["video"]["altyazi_goster"]))
    if preview_only:
        return tl, renderer
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="pofi-"))
    try:
        raw_wav, norm_wav = tmp / "mix.wav", tmp / "mix_norm.wav"
        audio.write_wav(raw_wav, mixed)
        audio.loudnorm(raw_wav, norm_wav, float(cfg["video"]["hedef_loudness"]))
        workers = workers or max(1, (os.cpu_count() or 2))
        n = tl.n_frames
        chunk = max(fps * 4, math.ceil(n / (workers * 3)))
        jobs = [(a, min(n, a + chunk), tmp / f"seg_{k:04d}.mp4") for k, a in enumerate(range(0, n, chunk))]
        _WORKER["renderer"] = renderer
        print(f"[render] {ep['id']} ({fmt}) {tl.duration:.1f} sn, {n} kare, {len(jobs)} parça, {workers} işçi")
        if workers > 1:
            ctx = mp.get_context("fork")
            with ctx.Pool(workers) as pool:
                segs = pool.map(_encode_chunk, jobs, chunksize=1)
        else:
            segs = [_encode_chunk(j) for j in jobs]
        lst = tmp / "list.txt"
        lst.write_text("".join(f"file '{p}'\n" for p in segs))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                        "-i", str(norm_wav), "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
                        "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(out_path)], check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return tl


# ---------------------------------------------------------------- shorts


def make_short(ep, max_seconds=58):
    """A vertical Shorts cut: hook intro, one or two quizzes (or items) and a short outro."""
    if ep.get("shorts"):
        scenes = copy.deepcopy(ep["shorts"])
    else:
        src = ep["scenes"]
        intro = copy.deepcopy(src[0])
        intro["lines"] = (intro.get("lines") or [])[:1] or [f"Merhaba! Bugün {ep['title']}!"]
        picks = [s for s in src if s["type"] == "quiz"][:2]
        if not picks:
            picks = [s for s in src if s["type"] in ("item", "fact", "count", "compare")][:2]
        scenes = [intro] + copy.deepcopy(picks) + [{
            "type": "outro", "title": "Tam bölüm kanalda!",
            "lines": ["Tam bölümü kanalımızda izleyebilirsin. Hoşça kal!"],
            "emoji": get_visual_list(src[0])[:3],
        }]
    short = copy.deepcopy(ep)
    short["id"] = ep["id"] + "-short"
    short["scenes"] = scenes
    return short
