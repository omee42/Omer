"""Drawing primitives: easing, text sprites, cards, shapes, flags and compositing."""
import math
from functools import lru_cache

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from .assets import emoji_image, font

# ---------------------------------------------------------------- easing


def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def ease_out_cubic(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def ease_in_out(x):
    x = clamp(x)
    return 3 * x * x - 2 * x * x * x


def ease_out_back(x, s=1.70158):
    x = clamp(x)
    x -= 1
    return x * x * ((s + 1) * x + s) + 1


def ease_out_elastic(x):
    x = clamp(x)
    if x in (0, 1):
        return x
    return 2 ** (-10 * x) * math.sin((x * 10 - 0.75) * (2 * math.pi / 3)) + 1


def progress(t, start, dur):
    return clamp((t - start) / dur) if dur > 0 else float(t >= start)


def pop_scale(t, start, dur=0.45):
    """0 -> overshoot -> 1 pop-in curve."""
    if t < start:
        return 0.0
    return ease_out_back(progress(t, start, dur), 2.2)


# ---------------------------------------------------------------- colour


def rgb(c):
    if isinstance(c, (tuple, list)):
        return tuple(int(v) for v in c[:3])
    c = c.lstrip("#")
    return tuple(int(c[i: i + 2], 16) for i in (0, 2, 4))


def mix(a, b, k):
    a, b = rgb(a), rgb(b)
    return tuple(int(a[i] + (b[i] - a[i]) * k) for i in range(3))


def lighten(c, k):
    return mix(c, (255, 255, 255), k)


def darken(c, k):
    return mix(c, (0, 0, 0), k)


# ---------------------------------------------------------------- compositing


def blit(canvas: Image.Image, sprite: Image.Image, x: float, y: float, anchor="center", opacity=1.0):
    """Alpha-composite sprite onto canvas; (x, y) is the anchor point. Clips at edges."""
    if sprite is None or opacity <= 0.01:
        return
    w, h = sprite.size
    if anchor == "center":
        x0, y0 = int(round(x - w / 2)), int(round(y - h / 2))
    elif anchor == "bottom":
        x0, y0 = int(round(x - w / 2)), int(round(y - h))
    else:  # top-left
        x0, y0 = int(round(x)), int(round(y))
    cw, ch = canvas.size
    sx0, sy0 = max(0, -x0), max(0, -y0)
    sx1, sy1 = min(w, cw - x0), min(h, ch - y0)
    if sx1 <= sx0 or sy1 <= sy0:
        return
    part = sprite if (sx0, sy0, sx1, sy1) == (0, 0, w, h) else sprite.crop((sx0, sy0, sx1, sy1))
    if opacity < 0.999:
        part = part.copy()
        a = part.getchannel("A").point(lambda v: int(v * opacity))
        part.putalpha(a)
    canvas.alpha_composite(part, (x0 + sx0, y0 + sy0))


_TRANSFORM_CACHE = {}


def transformed(key: str, base: Image.Image, width: float, angle: float = 0.0) -> Image.Image:
    """Scaled/rotated copy of a sprite, cached on quantised size/angle."""
    wq = max(2, int(round(width / 2.0)) * 2)
    aq = int(round(angle)) if abs(angle) >= 0.5 else 0
    ck = (key, wq, aq)
    hit = _TRANSFORM_CACHE.get(ck)
    if hit is not None:
        return hit
    hq = max(2, int(round(base.height * wq / base.width)))
    im = base.resize((wq, hq), Image.LANCZOS) if (wq, hq) != base.size else base
    if aq:
        im = im.rotate(aq, resample=Image.BICUBIC, expand=True)
    if len(_TRANSFORM_CACHE) > 1500:
        _TRANSFORM_CACHE.clear()
    _TRANSFORM_CACHE[ck] = im
    return im


# ---------------------------------------------------------------- text


@lru_cache(maxsize=64)
def _line_metrics(kind, size):
    f = font(kind, size)
    top = f.getbbox("ÂÇĞİÖŞÜÁbdfhklt")[1]
    bottom = f.getbbox("gjpqyşçğ")[3]
    return top, bottom


@lru_cache(maxsize=2048)
def text_sprite(text, kind="title", size=96, fill=(255, 255, 255), stroke=0, stroke_fill=(40, 30, 70),
                shadow=True, align="center", line_gap=0.12):
    f = font(kind, size)
    lines = text.split("\n")
    pad = stroke + max(6, size // 10)
    boxes = [f.getbbox(l or " ", stroke_width=stroke) for l in lines]
    top, bottom = _line_metrics(kind, size)
    glyph_h = bottom - top
    line_h = int(glyph_h * (1 + line_gap))
    width = int(max(b[2] - b[0] for b in boxes) + 2 * pad)
    height = int(line_h * (len(lines) - 1) + glyph_h + 2 * pad)
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for i, (l, b) in enumerate(zip(lines, boxes)):
        lw = b[2] - b[0]
        x = (width - lw) / 2 - b[0] if align == "center" else pad - b[0]
        y = pad + i * line_h - top
        d.text((x, y), l, font=f, fill=rgb(fill), stroke_width=stroke, stroke_fill=rgb(stroke_fill))
    if shadow:
        alpha = img.getchannel("A")
        sh = Image.new("RGBA", img.size, (30, 20, 60, 0))
        sh.putalpha(alpha.filter(ImageFilter.GaussianBlur(max(2, size // 18))).point(lambda v: int(v * 0.45)))
        out = Image.new("RGBA", (width, height + size // 12), (0, 0, 0, 0))
        out.alpha_composite(sh, (0, size // 12))
        out.alpha_composite(img, (0, 0))
        return out
    return img


def _wrap_px(words, f, max_w):
    lines, cur = [], ""
    for w in words:
        cand = (cur + " " + w).strip()
        if cur and f.getlength(cand) > max_w:
            lines.append(cur)
            cur = w
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return lines


def wrap_balanced(words, f, max_w):
    """Greedy pixel wrap, then shrink the width while keeping the line count so lines are even."""
    lines = _wrap_px(words, f, max_w)
    if len(lines) <= 1:
        return lines
    lo, hi = max(f.getlength(w) for w in words), max_w
    for _ in range(14):
        mid = (lo + hi) / 2
        if len(_wrap_px(words, f, mid)) <= len(lines):
            hi = mid
        else:
            lo = mid
    return _wrap_px(words, f, hi)


def _has_orphan(lines):
    if len(lines) < 2:
        return False
    return any(len(l.split()) == 1 and len(l.strip(".,!?'")) <= 4 for l in lines)


def fit_text(text, kind, max_w, max_h, size, min_size=28, max_chars=None, **kw):
    """Largest text sprite that fits inside max_w x max_h, wrapping evenly and avoiding orphan words."""
    return fit_text_sized(text, kind, max_w, max_h, size, min_size, max_chars, **kw)[0]


def uniform_texts(texts, kind, max_w, max_h, size, **kw):
    """Fit several labels at one shared font size so a row of cards looks consistent."""
    common = min(fit_text_sized(t, kind, max_w, max_h, size, **kw)[1] for t in texts)
    return [fit_text_sized(t, kind, max_w, max_h, common, **kw)[0] for t in texts]


def fit_text_sized(text, kind, max_w, max_h, size, min_size=28, max_chars=None, **kw):
    words = text.split()
    best, best_score, best_size = None, -1, size
    while True:
        f = font(kind, size)
        pad = kw.get("stroke", 0) + max(6, size // 10)
        avail = max_w - 2 * pad
        if max_chars:
            avail = min(avail, f.getlength("a" * max_chars))
        if "\n" in text:
            lines = text.split("\n")
        else:
            lines = wrap_balanced(words, f, max(10, avail))
        spr = text_sprite("\n".join(lines), kind, size, **kw)
        fits = spr.width <= max_w and spr.height <= max_h
        if fits:
            score = size * (0.8 if _has_orphan(lines) else 1.0)
            if score > best_score:
                best, best_score, best_size = spr, score, size
            if not _has_orphan(lines) or size * 0.92 < best_score * 0.8:
                return best, best_size
        if size <= min_size:
            return (best, best_size) if best else (spr, size)
        size = max(min_size, int(size * 0.92))


# ---------------------------------------------------------------- shapes and cards


@lru_cache(maxsize=256)
def rounded_card(w, h, radius=40, fill=(255, 255, 255), alpha=245, border=None, border_w=0, shadow=True):
    w, h = int(w), int(h)
    s = 18 if shadow else 0
    img = Image.new("RGBA", (w + 2 * s, h + 2 * s + s // 2), (0, 0, 0, 0))
    if shadow:
        sh = Image.new("L", img.size, 0)
        ImageDraw.Draw(sh).rounded_rectangle((s, s + s // 2, s + w, s + h + s // 2), radius, fill=70)
        sh = sh.filter(ImageFilter.GaussianBlur(s * 0.6))
        dark = Image.new("RGBA", img.size, (40, 25, 80, 0))
        dark.putalpha(sh)
        img.alpha_composite(dark)
    body = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(body)
    d.rounded_rectangle((s, s, s + w, s + h), radius, fill=rgb(fill) + (alpha,),
                        outline=rgb(border) + (255,) if border else None, width=border_w)
    img.alpha_composite(body)
    return img


def radial_gradient(size, inner, outer, cx=0.38, cy=0.32, r=0.85, mid=None):
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xx / w) - cx) ** 2 + ((yy / h) - cy) ** 2) / r
    d = np.clip(d, 0, 1)[..., None]
    a = np.array(rgb(inner), np.float32)
    b = np.array(rgb(outer), np.float32)
    if mid is None:
        arr = a + (b - a) * d
    else:
        m = np.array(rgb(mid), np.float32)
        k = np.clip(d / 0.55, 0, 1)
        k2 = np.clip((d - 0.55) / 0.45, 0, 1)
        arr = np.where(d < 0.55, a + (m - a) * k, m + (b - m) * k2)
    return Image.fromarray(arr.astype(np.uint8), "RGB")


def vertical_gradient(size, top, bottom):
    w, h = size
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    a = np.array(rgb(top), np.float32)
    b = np.array(rgb(bottom), np.float32)
    arr = np.broadcast_to(a + (b - a) * t, (h, w, 3))
    return Image.fromarray(arr.astype(np.uint8), "RGB")


def shaded(mask: Image.Image, color, highlight=0.45, shade=0.35):
    """Fill a mask with a soft 3D-looking gradient in the Fluent emoji spirit."""
    g = radial_gradient(mask.size, lighten(color, highlight * 0.6), darken(color, shade * 0.7), 0.35, 0.28, 1.05,
                        mid=color)
    out = g.convert("RGBA")
    out.putalpha(mask)
    # glossy highlight blob
    w, h = mask.size
    hl = Image.new("L", mask.size, 0)
    ImageDraw.Draw(hl).ellipse((w * 0.18, h * 0.10, w * 0.52, h * 0.36), fill=120)
    hl = ImageChops.multiply(hl.filter(ImageFilter.GaussianBlur(w * 0.05)), mask)
    white = Image.new("RGBA", mask.size, (255, 255, 255, 0))
    white.putalpha(hl)
    out.alpha_composite(white)
    return out


def _poly(n, cx, cy, r, rot=-math.pi / 2):
    return [(cx + r * math.cos(rot + 2 * math.pi * i / n), cy + r * math.sin(rot + 2 * math.pi * i / n)) for i in range(n)]


def _star(cx, cy, r_out, r_in, points=5, rot=-math.pi / 2):
    pts = []
    for i in range(points * 2):
        r = r_out if i % 2 == 0 else r_in
        a = rot + math.pi * i / points
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


SHAPES = ["daire", "kare", "üçgen", "dikdörtgen", "yıldız", "kalp", "oval", "baklava", "beşgen", "altıgen",
          "yarım daire", "hilal"]
SHAPE_ALIASES = {"circle": "daire", "square": "kare", "triangle": "üçgen", "rectangle": "dikdörtgen",
                 "star": "yıldız", "heart": "kalp", "ellipse": "oval", "diamond": "baklava",
                 "pentagon": "beşgen", "hexagon": "altıgen", "semicircle": "yarım daire", "crescent": "hilal"}


@lru_cache(maxsize=64)
def shape_sprite(kind: str, color="#FF5A5F", size=512):
    kind = SHAPE_ALIASES.get(kind, kind)
    ss = 2
    S = size * ss
    m = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(m)
    p = S * 0.08
    if kind == "daire":
        d.ellipse((p, p, S - p, S - p), fill=255)
    elif kind == "kare":
        d.rounded_rectangle((p * 1.3, p * 1.3, S - p * 1.3, S - p * 1.3), S * 0.06, fill=255)
    elif kind == "dikdörtgen":
        d.rounded_rectangle((p, S * 0.24, S - p, S * 0.76), S * 0.05, fill=255)
    elif kind == "üçgen":
        d.polygon([(S / 2, p), (S - p, S - p * 1.4), (p, S - p * 1.4)], fill=255)
    elif kind == "yıldız":
        d.polygon(_star(S / 2, S * 0.54, S * 0.46, S * 0.2), fill=255)
    elif kind == "kalp":
        r = S * 0.22
        d.ellipse((S * 0.5 - 2 * r, S * 0.18, S * 0.5, S * 0.18 + 2 * r), fill=255)
        d.ellipse((S * 0.5, S * 0.18, S * 0.5 + 2 * r, S * 0.18 + 2 * r), fill=255)
        d.polygon([(S * 0.5 - 2 * r * 0.98, S * 0.18 + r * 1.25), (S * 0.5 + 2 * r * 0.98, S * 0.18 + r * 1.25),
                   (S * 0.5, S * 0.9)], fill=255)
    elif kind == "oval":
        d.ellipse((p, S * 0.2, S - p, S * 0.8), fill=255)
    elif kind == "baklava":
        d.polygon([(S / 2, p), (S - p * 2, S / 2), (S / 2, S - p), (p * 2, S / 2)], fill=255)
    elif kind == "beşgen":
        d.polygon(_poly(5, S / 2, S * 0.53, S * 0.45), fill=255)
    elif kind == "altıgen":
        d.polygon(_poly(6, S / 2, S / 2, S * 0.45, 0), fill=255)
    elif kind == "yarım daire":
        d.pieslice((p, S * 0.25, S - p, S * 1.15 - p), 180, 360, fill=255)
    elif kind == "hilal":
        d.ellipse((p, p, S - p, S - p), fill=255)
        d.ellipse((S * 0.3, p * 0.6, S - p * 0.2, S - p * 1.4), fill=0)
    else:
        raise KeyError(f"Bilinmeyen şekil: {kind}")
    m = m.resize((size, size), Image.LANCZOS)
    out = Image.new("RGBA", (size, size + size // 10), (0, 0, 0, 0))
    sh = Image.new("RGBA", out.size, (40, 25, 80, 0))
    sm = Image.new("L", out.size, 0)
    sm.paste(m.point(lambda v: v * 0.35), (0, size // 14))
    sh.putalpha(sm.filter(ImageFilter.GaussianBlur(size / 40)))
    out.alpha_composite(sh)
    out.alpha_composite(shaded(m, color), (0, 0))
    return out


@lru_cache(maxsize=8)
def flag_tr(width=600):
    """Turkish flag with the statutory proportions (12 x 8 grid, as in common reference SVGs)."""
    ss = 3
    u = width * ss / 12.0  # one grid unit; the flag is 12 x 8 units
    L, G = int(12 * u), int(8 * u)
    red, white = (227, 10, 23), (255, 255, 255)
    img = Image.new("RGB", (L, G), red)
    d = ImageDraw.Draw(img)
    cy = 4 * u
    d.ellipse((4.25 * u - 2 * u, cy - 2 * u, 4.25 * u + 2 * u, cy + 2 * u), fill=white)
    d.ellipse((4.75 * u - 1.6 * u, cy - 1.6 * u, 4.75 * u + 1.6 * u, cy + 1.6 * u), fill=red)
    r_out = 1.0 * u
    r_in = r_out * math.cos(2 * math.pi / 5) / math.cos(math.pi / 5)
    d.polygon(_star(6.8333 * u, cy, r_out, r_in, 5, math.pi), fill=white)  # one point faces the crescent
    img = img.resize((L // ss, G // ss), Image.LANCZOS).convert("RGBA")
    w, h = img.size
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, w - 1, h - 1), int(h * 0.05), fill=255)
    img.putalpha(mask)
    out = Image.new("RGBA", (w + 30, h + 40), (0, 0, 0, 0))
    sh = Image.new("RGBA", out.size, (40, 25, 80, 0))
    sm = Image.new("L", out.size, 0)
    sm.paste(mask.point(lambda v: v * 0.35), (15, 28))
    sh.putalpha(sm.filter(ImageFilter.GaussianBlur(10)))
    out.alpha_composite(sh)
    out.alpha_composite(img, (15, 10))
    return out


@lru_cache(maxsize=128)
def number_badge(text, color="#FF7A59", size=120):
    ss = 2
    S = size * ss
    m = Image.new("L", (S, S), 0)
    ImageDraw.Draw(m).ellipse((4, 4, S - 4, S - 4), fill=255)
    m = m.resize((size, size), Image.LANCZOS)
    b = shaded(m, color, 0.35, 0.25)
    t = text_sprite(text, "title", int(size * 0.62), (255, 255, 255), stroke=0, shadow=False)
    b.alpha_composite(t, ((size - t.width) // 2, (size - t.height) // 2 + int(size * 0.04)))
    return b


# ---------------------------------------------------------------- visuals


def visual_key(v) -> str:
    if isinstance(v, str):
        return "e:" + v
    if "emoji" in v:
        return "e:" + v["emoji"]
    if "shape" in v:
        return f"s:{v['shape']}:{v.get('color', '#FF5A5F')}"
    if "flag" in v:
        return "f:" + v["flag"]
    if "symbol" in v:
        return f"t:{v['symbol']}:{v.get('color', '#7C6CF2')}"
    raise ValueError(f"Görsel anlaşılamadı: {v}")


@lru_cache(maxsize=512)
def _visual_base(key: str) -> Image.Image:
    kind, _, rest = key.partition(":")
    if kind == "e":
        im = emoji_image(rest)
        # add a soft contact shadow so emoji don't float awkwardly
        out = Image.new("RGBA", (im.width, im.height + 24), (0, 0, 0, 0))
        sh = Image.new("RGBA", out.size, (40, 25, 80, 0))
        sm = Image.new("L", out.size, 0)
        sm.paste(im.getchannel("A").point(lambda v: v * 0.3), (0, 14))
        sh.putalpha(sm.filter(ImageFilter.GaussianBlur(7)))
        out.alpha_composite(sh)
        out.alpha_composite(im)
        return out
    if kind == "s":
        shape, _, color = rest.rpartition(":")
        return shape_sprite(shape, color)
    if kind == "f":
        if rest.lower() in ("tr", "türkiye", "turkiye"):
            return flag_tr(600)
        raise KeyError(f"Bilinmeyen bayrak: {rest}")
    if kind == "t":
        txt, _, color = rest.rpartition(":")
        size = 300 if len(txt) <= 2 else 220 if len(txt) <= 4 else 150
        spr = text_sprite(txt, "title", size, lighten(color, 0.05), stroke=max(6, size // 18),
                          stroke_fill=(255, 255, 255), shadow=True)
        return spr
    raise KeyError(key)


def visual(v, width, angle=0.0):
    key = visual_key(v)
    base = _visual_base(key)
    return transformed(key, base, width, angle)


def visual_aspect(v):
    b = _visual_base(visual_key(v))
    return b.height / b.width


# ---------------------------------------------------------------- particles


class Confetti:
    COLORS = ["#FF5A5F", "#FFC93C", "#3EC1D3", "#7C6CF2", "#5ED67A", "#FF8BC2"]

    def __init__(self, seed, W, H, n=90, origin=None):
        rng = np.random.default_rng(seed)
        self.W, self.H = W, H
        ox, oy = origin or (W / 2, H * 0.45)
        ang = rng.uniform(-math.pi, 0, n)
        spd = rng.uniform(700, 1500, n) * (H / 1080)
        self.p0 = np.stack([np.full(n, ox), np.full(n, oy)], 1)
        self.v0 = np.stack([np.cos(ang) * spd, np.sin(ang) * spd], 1)
        self.rot = rng.uniform(0, 360, n)
        self.spin = rng.uniform(-400, 400, n)
        self.col = [self.COLORS[i % len(self.COLORS)] for i in range(n)]
        self.size = rng.uniform(14, 26, n) * (H / 1080)
        self.kind = rng.integers(0, 2, n)

    def draw(self, canvas, t, dur=2.4):
        if t < 0 or t > dur:
            return
        layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        g = 1600 * (self.H / 1080)
        drag = np.exp(-1.2 * t)
        pos = self.p0 + self.v0 * (1 - drag) / 1.2 + np.array([0, 0.5 * g * t * t * 0.6])
        fade = clamp((dur - t) / 0.6)
        for i, (x, y) in enumerate(pos):
            s = self.size[i]
            a = math.radians(self.rot[i] + self.spin[i] * t)
            c = rgb(self.col[i]) + (int(255 * fade),)
            if self.kind[i] == 0:
                pts = [(x + s * math.cos(a + k * math.pi / 2) * (1 if k % 2 == 0 else 0.5),
                        y + s * math.sin(a + k * math.pi / 2) * (1 if k % 2 == 0 else 0.5)) for k in range(4)]
                d.polygon(pts, fill=c)
            else:
                d.ellipse((x - s / 2, y - s / 2, x + s / 2, y + s / 2), fill=c)
        canvas.alpha_composite(layer)
