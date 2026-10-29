"""Scene types. Each scene turns its JSON spec into timed steps and draws itself per frame.

Times inside a scene are local (0 = scene start). `self.T[tag] = (start, end)` is filled
by the scheduler for every Say/Wait step, so drawing code can sync visuals to speech.
"""
import math
from dataclasses import dataclass
from typing import Optional

from PIL import Image, ImageDraw, ImageFilter

from . import gfx
from .gfx import (Confetti, blit, clamp, ease_in_out, ease_out_back, ease_out_cubic, fit_text, lighten,
                  number_badge, pop_scale, progress, rgb, rounded_card, text_sprite, transformed, uniform_texts)
from .schema import get_visual, get_visual_list
from .text import number_to_words, tr_capitalize, tr_upper


@dataclass
class Say:
    text: str
    tag: str = ""
    sfx: Optional[str] = None
    gap: Optional[float] = None
    caption: bool = True


@dataclass
class Wait:
    seconds: float
    tag: str = ""


@dataclass
class Sfx:
    name: str
    gain: float = 1.0
    offset: float = 0.0


class Layout:
    """Screen regions for landscape (YouTube) and portrait (Shorts) videos."""

    def __init__(self, W, H):
        self.W, self.H = W, H
        self.portrait = H > W
        s = min(W, H) / 1080.0
        self.s = s
        if self.portrait:
            self.content = (40 * s, 190 * s, W - 80 * s, 1130 * s)
            self.mascot = (W / 2, H - 46 * s, 370 * s)
            self.caption = (W / 2, 1418 * s, W - 90 * s)
            self.chip = (W / 2, 95 * s)
        else:
            self.content = (445 * s, 40 * s, W - 480 * s, 850 * s)
            self.mascot = (232 * s, H - 36 * s, 400 * s)
            self.caption = (445 * s + (W - 480 * s) / 2, H - 98 * s, 1340 * s)
            self.chip = (215 * s, 70 * s)

    @property
    def cx(self):
        x, y, w, h = self.content
        return x + w / 2


def wobble(t, phase=0.0, amp=3.0, speed=1.3):
    return amp * math.sin(t * speed + phase)


def bob(t, phase=0.0, amp=8.0, speed=2.2):
    return amp * math.sin(t * speed + phase)


class Scene:
    lead = 0.55
    tail = 0.55
    chapter_default = None

    def __init__(self, spec, ep, ctx, index):
        self.spec = spec
        self.ep = ep
        self.ctx = ctx
        self.L = ctx.layout
        self.s = ctx.layout.s
        self.index = index
        self.T = {}
        self.start = 0.0
        self.dur = 0.0
        self.theme = ctx.theme
        self.init()

    # -- helpers ---------------------------------------------------------
    def init(self):
        pass

    def fmt(self, text):
        return self.ctx.fmt(text)

    def say_lines(self, lines, prefix, sfx=None):
        return [Say(self.fmt(l), f"{prefix}{i}", sfx if i == 0 else None) for i, l in enumerate(lines or [])]

    def t0(self, tag, default=1e9):
        return self.T.get(tag, (default, default))[0]

    def t1(self, tag, default=1e9):
        return self.T.get(tag, (default, default))[1]

    def active(self, tag, t):
        a, b = self.T.get(tag, (None, None))
        return a is not None and a <= t < b + 0.15

    def pulse(self, t, tags):
        """Small scale bump at the start of any of the given lines."""
        best = 0.0
        for tag in tags:
            a = self.t0(tag)
            if a <= t < a + 0.4:
                best = max(best, math.sin(math.pi * (t - a) / 0.4))
        return 1.0 + 0.07 * best

    def put(self, canvas, v, cx, cy, width, scale=1.0, angle=0.0, opacity=1.0):
        if scale <= 0.02:
            return
        spr = gfx.visual(v, width * scale, angle)
        blit(canvas, spr, cx, cy, opacity=opacity)

    def put_sprite(self, canvas, key, spr, cx, cy, scale=1.0, angle=0.0, opacity=1.0):
        if scale <= 0.02:
            return
        if abs(scale - 1) > 0.01 or abs(angle) >= 0.5:
            spr = transformed(key, spr, spr.width * scale, angle)
        blit(canvas, spr, cx, cy, opacity=opacity)

    def title_text(self, text, max_w, max_h, size=120, color=(255, 255, 255)):
        return fit_text(text, "title", int(max_w), int(max_h), int(size * self.s), stroke=int(11 * self.s),
                        stroke_fill=rgb(self.theme["dark"]), fill=color)

    def body_text(self, text, max_w, max_h, size=64, color=None):
        return fit_text(text, "body", int(max_w), int(max_h), int(size * self.s),
                        fill=rgb(color or self.theme["dark"]), shadow=False)

    def labels(self, key, texts, max_w, max_h, size=58):
        cache = self.ctx.cache
        k = ("labels", self.index, key)
        if k not in cache:
            cache[k] = uniform_texts([self.fmt(x) for x in texts], "body", int(max_w), int(max_h),
                                     int(size * self.s), fill=rgb(self.theme["dark"]), shadow=False)
        return cache[k]

    def card(self, w, h, alpha=235, border=None):
        return rounded_card(int(w), int(h), int(36 * self.s), (255, 255, 255), alpha,
                            border=border, border_w=int(8 * self.s) if border else 0)

    # -- to override -----------------------------------------------------
    def steps(self):
        return self.say_lines(self.spec.get("lines"), "l")

    def draw(self, canvas, t):
        pass

    def mascot(self, t):
        return {"waving": False, "happy": False, "jump": 0.0, "big": False}

    def chapter(self):
        return self.spec.get("chapter") or self.chapter_default


# ---------------------------------------------------------------- intro / outro


class Intro(Scene):
    chapter_default = "Giriş"

    def init(self):
        self.visuals = get_visual_list(self.spec)[:5]

    def steps(self):
        st = [Sfx("tada", 0.9, 0.15)]
        for i in range(len(self.visuals)):
            st.append(Sfx("pop", 0.5, 0.75 + 0.14 * i))
        return st + self.say_lines(self.spec.get("lines"), "l")

    def draw(self, canvas, t):
        x, y, w, h = self.L.content
        title = self.title_text(self.fmt(self.spec["title"]), w * 0.96, h * 0.36, 150)
        sc = pop_scale(t, 0.2, 0.6)
        ang = wobble(t, 0, 2.0, 1.4) if t > 0.8 else 0
        self.put_sprite(canvas, "intro-title", title, x + w / 2, y + h * 0.24, sc, ang)
        sub = self.spec.get("subtitle")
        if sub:
            sp = fit_text(self.fmt(sub), "title", int(w * 0.8), int(h * 0.13), int(84 * self.s),
                          fill=rgb(self.theme["accent"]), stroke=int(9 * self.s), stroke_fill=(255, 255, 255))
            self.put_sprite(canvas, "intro-sub", sp, x + w / 2, y + h * 0.45, pop_scale(t, 0.65, 0.5))
        n = len(self.visuals)
        if n:
            size = min(w / (n + 0.6), h * 0.3)
            for i, v in enumerate(self.visuals):
                fx = (i + 0.5) / n
                cx = x + w * (0.5 + (fx - 0.5) * min(1.0, 0.28 * n))
                cy = y + h * 0.77 - math.sin(math.pi * fx) * h * 0.04 + bob(t, i * 1.3, 10 * self.s)
                self.put(canvas, v, cx, cy, size, pop_scale(t, 0.75 + 0.14 * i), wobble(t, i, 5))

    def mascot(self, t):
        return {"waving": t < self.t1("l0", 2.5), "happy": t < 0.6, "jump": 0.0, "big": True}


class Outro(Scene):
    chapter_default = "Kapanış"
    tail = 1.8

    def steps(self):
        return [Sfx("sparkle", 0.8, 0.1)] + self.say_lines(self.spec.get("lines"), "l")

    def draw(self, canvas, t):
        x, y, w, h = self.L.content
        title = self.title_text(self.fmt(self.spec.get("title", "Görüşmek üzere!")), w * 0.95, h * 0.3, 140)
        self.put_sprite(canvas, "outro-title", title, x + w / 2, y + h * 0.2, pop_scale(t, 0.2, 0.6),
                        wobble(t, 0, 2))
        for i in range(6):
            fx = (i + 0.5) / 6
            sx = x + w * fx
            sy = y + h * (0.42 + 0.06 * math.sin(i * 2.1))
            tw = 0.75 + 0.25 * math.sin(t * 3 + i * 1.7)
            self.put(canvas, "⭐", sx, sy, 70 * self.s * tw, pop_scale(t, 0.4 + 0.1 * i), wobble(t, i, 12))
        nxt = self.spec.get("next")
        if nxt:
            lines = self.spec.get("lines") or []
            appear = self.t0("l1", 1.2) if len(lines) > 1 else 1.2
            cw, ch = w * 0.7, h * 0.36
            sc = pop_scale(t, appear, 0.5)
            if sc > 0.02:
                card = self.card(cw, ch)
                cx, cy = x + w / 2, y + h * 0.74
                v = get_visual(nxt)
                layer = card.copy()
                lab = text_sprite("Sıradaki bölüm", "body", int(40 * self.s), rgb(self.theme["accent"]), shadow=False)
                tx = self.body_text(self.fmt(nxt["text"]), cw * (0.58 if v else 0.9), ch * 0.5, 72)
                off = (layer.width - cw) / 2
                if v:
                    vs = gfx.visual(v, ch * 0.66)
                    blit(layer, vs, off + ch * 0.48, layer.height / 2 - 6)
                    blit(layer, lab, off + ch * 0.95 + lab.width / 2, off + ch * 0.28)
                    blit(layer, tx, off + ch * 0.95 + tx.width / 2, off + ch * 0.62)
                else:
                    blit(layer, lab, layer.width / 2, off + ch * 0.28)
                    blit(layer, tx, layer.width / 2, off + ch * 0.62)
                self.put_sprite(canvas, f"outro-next-{self.index}", layer, cx, cy + bob(t, 0, 5 * self.s), sc)

    def mascot(self, t):
        return {"waving": True, "happy": t > self.dur - 1.6, "jump": 0.0, "big": True}


# ---------------------------------------------------------------- talk / item / fact


class Talk(Scene):
    def init(self):
        self.visuals = get_visual_list(self.spec)[:3]

    def steps(self):
        st = [Sfx("sparkle", 0.5, 0.15)] if self.visuals else []
        return st + self.say_lines(self.spec.get("lines"), "l")

    def draw(self, canvas, t):
        x, y, w, h = self.L.content
        n = len(self.visuals)
        text = self.spec.get("text")
        area_h = h * (0.66 if text else 0.9)
        tags = [f"l{i}" for i in range(len(self.spec.get("lines", [])))]
        pulse = self.pulse(t, tags)
        if n:
            size = {1: min(w * 0.5, area_h * 0.78), 2: min(w * 0.38, area_h * 0.7), 3: min(w * 0.28, area_h * 0.64)}[n]
            for i, v in enumerate(self.visuals):
                cx = x + w * (i + 0.5) / n
                cy = y + area_h * 0.52 + bob(t, i * 1.7, 9 * self.s)
                self.put(canvas, v, cx, cy, size, pop_scale(t, 0.15 + 0.12 * i) * pulse, wobble(t, i * 2, 3))
        if text:
            sp = self.body_text(self.fmt(text), w * 0.84, h * 0.22, 70)
            card = self.card(min(w, sp.width + 90 * self.s), sp.height + 50 * self.s)
            layer = card.copy()
            blit(layer, sp, layer.width / 2, layer.height / 2 - 8 * self.s)
            appear = min(self.t0("l0", 0.5) + 0.35, 1.0)
            self.put_sprite(canvas, f"talk-{self.index}", layer, x + w / 2, y + h * 0.84, pop_scale(t, appear))


class Item(Scene):
    def init(self):
        self.v = get_visual(self.spec)
        self.color = self.spec.get("color") or self.theme["accent"]

    def steps(self):
        return [Sfx("sparkle", 0.6, 0.15)] + self.say_lines(self.spec.get("lines"), "l")

    def _halo(self, size):
        key = (self.color, int(size))
        cache = self.ctx.cache.setdefault("halo", {})
        if key not in cache:
            S = int(size)
            m = Image.new("L", (S, S), 0)
            ImageDraw.Draw(m).ellipse((S * 0.1, S * 0.1, S * 0.9, S * 0.9), fill=255)
            m = m.filter(ImageFilter.GaussianBlur(S * 0.05))
            im = Image.new("RGBA", (S, S), lighten(self.color, 0.35) + (0,))
            im.putalpha(m.point(lambda v: int(v * 0.85)))
            cache[key] = im
        return cache[key]

    def draw(self, canvas, t):
        x, y, w, h = self.L.content
        size = min(w * 0.44, h * 0.56)
        cx, cy = x + w / 2, y + h * 0.4
        tags = [f"l{i}" for i in range(len(self.spec.get("lines", [])))]
        halo = self._halo(size * 1.45)
        hs = pop_scale(t, 0.05, 0.5) * (1 + 0.03 * math.sin(t * 2.4))
        self.put_sprite(canvas, f"halo-{self.color}", halo, cx, cy, hs)
        self.put(canvas, self.v, cx, cy + bob(t, 0, 10 * self.s), size,
                 pop_scale(t, 0.15) * self.pulse(t, tags), wobble(t, 0, 4))
        word = tr_upper(self.fmt(self.spec["word"]))
        sp = fit_text(word, "title", int(w * 0.9), int(h * 0.26), int(150 * self.s), fill=rgb(self.color),
                      stroke=int(14 * self.s), stroke_fill=(255, 255, 255))
        appear = self.t0("l0", 0.5) + 0.45
        self.put_sprite(canvas, f"word-{self.index}", sp, cx, y + h * 0.86, pop_scale(t, appear, 0.5),
                        wobble(t, 1, 1.5) if t > appear + 0.5 else 0)


class Fact(Scene):
    def steps(self):
        return [Sfx("sparkle", 0.7, 0.2)] + self.say_lines(self.spec.get("lines"), "l")

    def draw(self, canvas, t):
        x, y, w, h = self.L.content
        cw, ch = w * 0.94, h * 0.8
        key = f"fact-{self.index}"
        cache = self.ctx.cache
        if key not in cache:
            card = self.card(cw, ch).copy()
            off = (card.width - cw) / 2
            head = text_sprite(self.fmt(self.spec.get("title", "Biliyor muydun?")), "title", int(66 * self.s),
                               (255, 255, 255), shadow=False)
            pill_w, pill_h = head.width + 150 * self.s, head.height + 10 * self.s
            pill = rounded_card(int(pill_w), int(pill_h), int(pill_h / 2), rgb(self.theme["accent"]), 255, shadow=False)
            bulb = gfx.visual("💡", pill_h * 0.95)
            blit(pill, bulb, 30 * self.s + pill_h * 0.5, pill.height / 2)
            blit(pill, head, pill.width / 2 + pill_h * 0.4, pill.height / 2 + 4 * self.s)
            blit(card, pill, card.width / 2, off + pill.height / 2 + 22 * self.s)
            if self.L.portrait:
                tx = self.body_text(self.fmt(self.spec["text"]), cw * 0.86, ch * 0.34, 70)
                blit(card, tx, card.width / 2, off + ch * 0.78)
            else:
                tx = self.body_text(self.fmt(self.spec["text"]), cw * 0.5, ch * 0.62, 72)
                blit(card, tx, off + cw * 0.70, off + ch * 0.58)
            cache[key] = card
        card = cache[key]
        cx, cy = x + w / 2, y + h * 0.53
        sc = pop_scale(t, 0.15, 0.55)
        self.put_sprite(canvas, key, card, cx, cy, sc)
        if sc > 0.9:
            if self.L.portrait:
                vx, vy, vs = cx, cy - ch * 0.12, ch * 0.42
            else:
                vx, vy, vs = cx - cw * 0.26, cy + ch * 0.08, min(ch * 0.58, cw * 0.36)
            tags = [f"l{i}" for i in range(len(self.spec.get("lines", [])))]
            self.put(canvas, get_visual(self.spec), vx, vy + bob(t, 0, 8 * self.s), vs,
                     pop_scale(t, 0.45) * self.pulse(t, tags), wobble(t, 0, 4))


# ---------------------------------------------------------------- grids


def grid_cells(n, area, portrait, max_cols=None):
    x, y, w, h = area
    if portrait:
        cols = 1 if n == 1 else 2 if n <= 4 else 2
        if n in (5, 6):
            cols = 2
    else:
        cols = n if n <= 4 else 3
    if max_cols:
        cols = min(cols, max_cols)
    rows = math.ceil(n / cols)
    cw, ch = w / cols, h / rows
    cells = []
    for i in range(n):
        r, c = divmod(i, cols)
        in_row = cols if r < rows - 1 else n - cols * (rows - 1)
        x0 = x + (w - in_row * cw) / 2
        cells.append((x0 + (c + 0.5) * cw, y + (r + 0.5) * ch, cw, ch))
    return cells


class Gallery(Scene):
    reveal_all = False

    def init(self):
        self.items = self.spec["items"]

    def steps(self):
        st = []
        if self.reveal_all:
            for i in range(len(self.items)):
                st.append(Sfx("pop", 0.5, 0.15 + 0.14 * i))
        st += self.say_lines(self.spec.get("lines"), "l")
        for i, it in enumerate(self.items):
            if it.get("line"):
                st.append(Say(self.fmt(it["line"]), f"i{i}", None if self.reveal_all else "pop"))
        return st + self.say_lines(self.spec.get("after"), "a")

    def appear(self, i):
        if self.reveal_all:
            return 0.15 + 0.14 * i
        return self.t0(f"i{i}", 0.3)

    def draw(self, canvas, t):
        x, y, w, h = self.L.content
        title = self.spec.get("title")
        th = h * 0.16 if title else 0
        if title:
            sp = self.title_text(self.fmt(title), w * 0.9, th, 96)
            self.put_sprite(canvas, f"gt-{self.index}", sp, x + w / 2, y + th / 2, pop_scale(t, 0.1))
        cells = grid_cells(len(self.items), (x, y + th, w, h - th), self.L.portrait)
        for i, (it, (cx, cy, cw, ch)) in enumerate(zip(self.items, cells)):
            a = self.appear(i)
            sc = pop_scale(t, a)
            if sc <= 0.02:
                continue
            hl = self.active(f"i{i}", t)
            k = 1.08 if hl else 1.0
            card_w = cw * 0.88
            card_h = min(ch * 0.9, card_w * 1.4)
            card = self.card(card_w, card_h, alpha=215 if hl else 165,
                             border=self.theme["accent"] if hl else None)
            self.put_sprite(canvas, f"gc-{self.index}-{i}-{hl}", card, cx, cy, sc * k)
            label_h = min(card_h * 0.22, 130 * self.s)
            vs = min(card_w * 0.74, (card_h - label_h) * 0.8)
            self.put(canvas, get_visual(it), cx, cy - label_h * 0.5 + bob(t, i * 1.1, 6 * self.s),
                     vs, sc * k * (self.pulse(t, [f"i{i}"])), wobble(t, i, 4 if hl else 2))
            lab = self.labels("g", [x["label"] for x in self.items], card_w * 0.86, label_h, 58)[i]
            self.put_sprite(canvas, f"gl-{self.index}-{i}", lab, cx, cy + card_h / 2 - label_h * 0.62, sc * k)


class Recap(Gallery):
    reveal_all = True
    chapter_default = "Tekrar"


class Count(Scene):
    def init(self):
        self.n = int(self.spec["n"])
        self.v = get_visual(self.spec)

    def steps(self):
        st = self.say_lines(self.spec.get("lines"), "l") + [Wait(0.2, "start")]
        for i in range(1, self.n + 1):
            st.append(Say(tr_capitalize(number_to_words(i)) + "!", f"c{i}", "pop", gap=0.2))
        st += [Sfx("tada", 0.8), Wait(0.5, "done")]
        return st + self.say_lines(self.spec.get("after"), "a")

    def draw(self, canvas, t):
        x, y, w, h = self.L.content
        top = h * 0.2
        n = self.n
        if self.L.portrait:
            cols = min(n, 3 if n <= 9 else 4)
        else:
            cols = n if n <= 5 else math.ceil(n / 2) if n <= 10 else 5
        rows = math.ceil(n / cols)
        aw, ah = w, h - top
        cw, ch = aw / cols, ah / rows
        size = min(cw, ch) * 0.74
        current = 0
        for i in range(1, n + 1):
            if t >= self.t0(f"c{i}"):
                current = i
        done = self.t0("done")
        for i in range(n):
            r, c = divmod(i, cols)
            in_row = cols if r < rows - 1 else n - cols * (rows - 1)
            x0 = x + (aw - in_row * cw) / 2
            cx = x0 + (c + 0.5) * cw
            cy = y + top + (r + 0.5) * ch
            counted = t >= self.t0(f"c{i + 1}")
            jump = 0.0
            if t > done:
                jump = -abs(math.sin((t - done) * 5 - i * 0.5)) * 22 * self.s * clamp((done + 2.5 - t) / 1.0)
            if counted:
                sc = pop_scale(t, self.t0(f"c{i + 1}"), 0.4)
                self.put(canvas, self.v, cx, cy + jump + bob(t, i, 4 * self.s), size, sc,
                         wobble(t, i, 3))
                b = number_badge(str(i + 1), self.theme["accent"], int(size * 0.36))
                self.put_sprite(canvas, f"cb-{i}-{int(size)}", b, cx + size * 0.36, cy + size * 0.3 + jump, sc)
            else:
                self.put(canvas, self.v, cx, cy, size, pop_scale(t, 0.1 + 0.05 * i), 0, opacity=0.32)
        if current:
            big = number_badge(str(current), self.theme["accent2"], int(min(190 * self.s, top * 0.95)))
            sc = pop_scale(t, self.t0(f"c{current}"), 0.35)
            self.put_sprite(canvas, f"cbig-{current}", big, x + w / 2, y + top * 0.5, sc)
        label = self.spec.get("label")
        if label and t > done:
            sp = self.title_text(f"{n} {tr_upper(self.fmt(label))}", w * 0.34, top * 0.9, 110)
            lx = x + w * 0.83 if not self.L.portrait else x + w / 2
            ly = y + top * 0.5 if not self.L.portrait else y + h * 0.97
            self.put_sprite(canvas, f"clab-{self.index}", sp, lx, ly, pop_scale(t, done))


# ---------------------------------------------------------------- letters / compare


class Letter(Scene):
    def init(self):
        self.letter = tr_upper(self.spec["letter"])
        from .text import tr_lower
        self.lower = tr_lower(self.letter)
        self.words = self.spec["words"]

    def steps(self):
        st = [Sfx("sparkle", 0.7, 0.15)] + self.say_lines(self.spec.get("lines"), "l")
        for i, wd in enumerate(self.words):
            st.append(Say(self.fmt(wd["line"]), f"w{i}", "pop"))
        return st + self.say_lines(self.spec.get("after"), "a")

    def draw(self, canvas, t):
        x, y, w, h = self.L.content
        if self.L.portrait:
            lz = (x, y, w, h * 0.36)
            wz = (x, y + h * 0.38, w, h * 0.62)
        else:
            lz = (x, y, w * 0.42, h)
            wz = (x + w * 0.44, y, w * 0.56, h)
        lx, ly, lw, lh = lz
        size = int(min(lw * 0.6, lh * 0.75) / self.s)
        up = text_sprite(self.letter, "title", int(size * self.s), rgb(self.theme["accent"]),
                         stroke=int(16 * self.s), stroke_fill=(255, 255, 255))
        lo = text_sprite(self.lower, "title", int(size * 0.75 * self.s), rgb(self.theme["accent2"]),
                         stroke=int(14 * self.s), stroke_fill=(255, 255, 255))
        sc = pop_scale(t, 0.15, 0.6)
        tags = [f"l{i}" for i in range(len(self.spec.get("lines", [])))]
        p = self.pulse(t, tags)
        self.put_sprite(canvas, f"lu-{self.letter}-{size}", up, lx + lw * 0.34, ly + lh * 0.5, sc * p, wobble(t, 0, 3))
        self.put_sprite(canvas, f"ll-{self.letter}-{size}", lo, lx + lw * 0.76, ly + lh * 0.58, pop_scale(t, 0.45) * p,
                        wobble(t, 1.5, 3))
        wx, wy, ww, wh = wz
        n = len(self.words)
        rh = wh / max(n, 3)
        y0 = wy + (wh - rh * n) / 2
        for i, wd in enumerate(self.words):
            a = self.t0(f"w{i}")
            sc = pop_scale(t, a)
            if sc <= 0.02:
                continue
            hl = self.active(f"w{i}", t)
            k = 1.06 if hl else 1.0
            cy = y0 + (i + 0.5) * rh
            card = self.card(ww * 0.94, rh * 0.84, alpha=225 if hl else 175,
                             border=self.theme["accent"] if hl else None)
            self.put_sprite(canvas, f"lc-{self.index}-{i}-{hl}", card, wx + ww / 2, cy, sc * k)
            vs = rh * 0.7
            self.put(canvas, get_visual(wd), wx + ww * 0.06 + vs * 0.6, cy + bob(t, i, 4 * self.s), vs, sc * k,
                     wobble(t, i, 3))
            word = self.fmt(wd["word"])
            fs = int(min(rh * 0.55, 92 * self.s))
            while fs > 30 and gfx.font("title", fs).getlength(word) > ww * 0.6:
                fs -= 4
            whole = text_sprite(word, "title", fs, rgb(self.theme["dark"]), stroke=int(6 * self.s),
                                stroke_fill=(255, 255, 255), align="left")
            first = text_sprite(word[0], "title", fs, rgb(self.theme["accent"]), stroke=int(6 * self.s),
                                stroke_fill=(255, 255, 255), align="left")
            combo = whole.copy()
            combo.alpha_composite(first, (0, 0))
            tx = wx + ww * 0.06 + vs * 1.25 + combo.width / 2
            self.put_sprite(canvas, f"lw-{self.index}-{i}", combo, tx, cy, sc * k)


class Compare(Scene):
    def steps(self):
        L, R = self.spec["left"], self.spec["right"]
        st = self.say_lines(self.spec.get("lines"), "l")
        st += [Say(self.fmt(L["line"]), "L", "pop"), Say(self.fmt(R["line"]), "R", "pop")]
        return st + self.say_lines(self.spec.get("after"), "a")

    def draw(self, canvas, t):
        x, y, w, h = self.L.content
        title = self.spec.get("title")
        th = h * 0.16 if title else 0
        if title:
            sp = self.title_text(self.fmt(title), w * 0.9, th, 96)
            self.put_sprite(canvas, f"cmt-{self.index}", sp, x + w / 2, y + th / 2, pop_scale(t, 0.1))
        if self.L.portrait:
            boxes = [(x + w / 2, y + th + (h - th) * 0.25, w * 0.9, (h - th) * 0.46),
                     (x + w / 2, y + th + (h - th) * 0.75, w * 0.9, (h - th) * 0.46)]
        else:
            boxes = [(x + w * 0.25, y + th + (h - th) / 2, w * 0.46, (h - th) * 0.9),
                     (x + w * 0.75, y + th + (h - th) / 2, w * 0.46, (h - th) * 0.9)]
        for side, tag, (cx, cy, cw, ch) in zip(("left", "right"), ("L", "R"), boxes):
            it = self.spec[side]
            sc = pop_scale(t, self.t0(tag))
            if sc <= 0.02:
                continue
            hl = self.active(tag, t)
            k = 1.05 if hl else 1.0
            card = self.card(cw, ch, alpha=225 if hl else 180, border=self.theme["accent"] if hl else None)
            self.put_sprite(canvas, f"cm-{self.index}-{side}-{hl}", card, cx, cy, sc * k)
            vs = min(cw * 0.62, ch * 0.62) * float(it.get("scale", 1.0))
            self.put(canvas, get_visual(it), cx, cy - ch * 0.08 + bob(t, side == "left", 6 * self.s), vs,
                     sc * k * self.pulse(t, [tag]), wobble(t, side == "left", 3))
            lab = self.labels("c", [self.spec["left"]["label"], self.spec["right"]["label"]], cw * 0.8,
                              ch * 0.16, 64)[0 if side == "left" else 1]
            self.put_sprite(canvas, f"cml-{self.index}-{side}", lab, cx, cy + ch * 0.37, sc * k)


# ---------------------------------------------------------------- quiz


class Quiz(Scene):
    chapter_default = "Bilmece"

    def init(self):
        self.opts = self.spec["options"]
        self.answer = int(self.spec["answer"])
        self.confetti = None

    def steps(self):
        lines = self.spec.get("lines") or [self.spec["question"]]
        st = [Sfx("ding", 0.6, 0.05)]
        for i in range(len(self.opts)):
            st.append(Sfx("pop", 0.45, 0.45 + 0.22 * i))
        st += self.say_lines(lines, "l")
        st.append(Wait(0.25, "think"))
        for k in (3, 2, 1):
            st += [Sfx("tick" if k > 1 else "tock", 0.9), Wait(1.0, f"cd{k}")]
        st += [Sfx("correct", 1.0), Wait(0.3, "reveal")]
        after = self.spec.get("after") or [f"Doğru cevap: {self.opts[self.answer]['label']}!"]
        return st + self.say_lines(after, "a")

    def draw(self, canvas, t):
        x, y, w, h = self.L.content
        s = self.s
        qh = h * 0.2
        q = fit_text(self.fmt(self.spec["question"]), "title", int(w * 0.8), int(qh * 0.8), int(90 * s),
                     fill=(255, 255, 255), stroke=0, shadow=False)
        key = f"qpill-{self.index}"
        cache = self.ctx.cache
        if key not in cache:
            pill = rounded_card(int(q.width + 80 * s), int(q.height + 24 * s), int(40 * s),
                                rgb(self.theme["accent2"]), 255)
            pill = pill.copy()
            blit(pill, q, pill.width / 2, pill.height / 2 - 4 * s)
            cache[key] = pill
        counting = self.t0("cd3") <= t < self.t1("cd1")
        pill_op = 0.0 if (counting and self.L.portrait) else 1.0
        self.put_sprite(canvas, key, cache[key], x + w / 2, y + qh / 2, pop_scale(t, 0.05), opacity=pill_op)
        reveal = self.t0("reveal")
        n = len(self.opts)
        area = (x, y + qh, w, h - qh)
        if self.L.portrait:
            cells = grid_cells(n, area, True)
        else:
            cells = grid_cells(n, area, False, max_cols=4)
        for i, (op, (cx, cy, cw, ch)) in enumerate(zip(self.opts, cells)):
            sc = pop_scale(t, 0.45 + 0.22 * i)
            if sc <= 0.02:
                continue
            k, op_a = 1.0, 1.0
            correct = i == self.answer
            if t >= reveal:
                pr = progress(t, reveal, 0.5)
                if correct:
                    k = 1 + 0.14 * ease_out_back(pr, 2.5)
                else:
                    k = 1 - 0.1 * ease_in_out(pr)
                    op_a = 1 - 0.62 * ease_in_out(pr)
            side = min(cw * 0.86, ch * 0.9)
            border = "#33C96B" if (t >= reveal and correct) else None
            card = self.card(side, side, alpha=235, border=border)
            self.put_sprite(canvas, f"qc-{self.index}-{border}-{int(side)}", card, cx, cy, sc * k, opacity=op_a)
            vs = side * 0.62
            self.put(canvas, get_visual(op), cx, cy - side * 0.08 + bob(t, i, 5 * s), vs, sc * k,
                     wobble(t, i, 3), opacity=op_a)
            lab = self.labels("q", [o["label"] for o in self.opts], side * 0.86, side * 0.17, 54)[i]
            self.put_sprite(canvas, f"ql-{self.index}-{i}", lab, cx, cy + side * 0.36, sc * k, opacity=op_a)
            if t >= reveal and correct:
                self.put(canvas, "✅", cx + side * 0.42 * k, cy - side * 0.42 * k, side * 0.3,
                         pop_scale(t, reveal + 0.1), 0)
                if self.confetti is None:
                    self.confetti = Confetti(1000 + self.index, self.L.W, self.L.H, origin=(cx, cy))
        # countdown
        for kk in (3, 2, 1):
            a, b = self.T.get(f"cd{kk}", (None, None))
            if a is not None and a <= t < b:
                badge = number_badge(str(kk), self.theme["accent"], int(170 * s))
                bx, by = (x + w - 100 * s, y + qh / 2) if not self.L.portrait else (x + w / 2, y + qh / 2)
                self._ring(canvas, bx, by, 110 * s, (t - a) / (b - a))
                self.put_sprite(canvas, f"cd-{kk}", badge, bx, by, pop_scale(t, a, 0.3))
        if self.confetti is not None and t >= reveal:
            self.confetti.draw(canvas, t - reveal)

    def _ring(self, canvas, cx, cy, r, frac):
        d = ImageDraw.Draw(canvas)
        d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(255, 255, 255, 200), width=int(14 * self.s))
        d.arc((cx - r, cy - r, cx + r, cy + r), -90, -90 + 360 * (1 - frac), fill=rgb(self.theme["accent2"]),
              width=int(14 * self.s))

    def mascot(self, t):
        rv = self.t0("reveal")
        happy = rv <= t < rv + 1.6
        jump = math.sin(math.pi * clamp((t - rv) / 0.6)) if rv <= t < rv + 0.6 else 0
        return {"waving": False, "happy": happy, "jump": jump, "big": False}


# ---------------------------------------------------------------- chant / story


class Chant(Scene):
    chapter_default = "Tekerleme"

    def init(self):
        self.items = self.spec["items"]
        self.default_v = get_visual(self.spec) if any(k in self.spec for k in ("emoji", "visual", "shape")) else None

    def steps(self):
        st = self.say_lines(self.spec.get("lines"), "l")
        for i, it in enumerate(self.items):
            st.append(Say(self.fmt(it["line"]), f"c{i}", "pop" if i == 0 else None, gap=0.3))
        return st + self.say_lines(self.spec.get("after"), "a")

    def draw(self, canvas, t):
        x, y, w, h = self.L.content
        title = self.spec.get("title")
        if title:
            sp = self.title_text(self.fmt(title), w * 0.8, h * 0.13, 80)
            self.put_sprite(canvas, f"cht-{self.index}", sp, x + w / 2, y + h * 0.07, pop_scale(t, 0.1))
        cur = None
        for i in range(len(self.items)):
            if t >= self.t0(f"c{i}"):
                cur = i
        if cur is None:
            note = gfx.visual("🎵", 160 * self.s)
            blit(canvas, note, x + w / 2, y + h * 0.45 + bob(t, 0, 12 * self.s))
            return
        it = self.items[cur]
        a = self.t0(f"c{cur}")
        v = get_visual(it) or self.default_v
        beat = abs(math.sin((t - a) * 4.2))
        if v:
            self.put(canvas, v, x + w / 2, y + h * 0.36 - beat * 20 * self.s, h * 0.4, pop_scale(t, a, 0.35),
                     wobble(t, cur, 6))
        sp = fit_text(self.fmt(it["line"]), "title", int(w * 0.94), int(h * 0.3), int(118 * self.s),
                      fill=(255, 255, 255), stroke=int(11 * self.s), stroke_fill=rgb(self.theme["dark"]))
        self.put_sprite(canvas, f"chl-{self.index}-{cur}", sp, x + w / 2, y + h * (0.8 if v else 0.5),
                        pop_scale(t, a, 0.35))


class Story(Scene):
    chapter_default = "Hikâye"

    def init(self):
        self.panels = self.spec["panels"]

    def steps(self):
        st = self.say_lines(self.spec.get("lines"), "l")
        for j, p in enumerate(self.panels):
            st += [Sfx("whoosh", 0.6), Wait(0.4, f"p{j}")]
            st += self.say_lines(p.get("lines"), f"p{j}l")
        return st + self.say_lines(self.spec.get("after"), "a")

    def _page(self, j, pw, ph):
        key = f"page-{self.index}-{j}"
        cache = self.ctx.cache
        if key not in cache:
            page = rounded_card(int(pw), int(ph), int(40 * self.s), (255, 249, 236), 250,
                                border=lighten(self.theme["accent"], 0.3), border_w=int(10 * self.s)).copy()
            cap = self.panels[j].get("caption")
            if cap:
                off = (page.width - pw) / 2
                tx = self.body_text(self.fmt(cap), pw * 0.86, ph * 0.18, 64)
                blit(page, tx, page.width / 2, off + ph * 0.86)
            cache[key] = page
        return cache[key]

    def draw(self, canvas, t):
        x, y, w, h = self.L.content
        pw, ph = w * 0.96, h * 0.94
        cur = -1
        for j in range(len(self.panels)):
            if t >= self.t0(f"p{j}") - 0.4:
                cur = j
        if cur < 0:
            sp = self.title_text(self.fmt(self.spec.get("title", "Bir varmış, bir yokmuş...")), w * 0.9, h * 0.3, 110)
            self.put_sprite(canvas, f"st-{self.index}", sp, x + w / 2, y + h / 2, pop_scale(t, 0.1))
            return
        a = self.t0(f"p{cur}") - 0.4
        pr = ease_out_cubic(progress(t, a, 0.5))
        for j, dx in ((cur - 1, -pr), (cur, 1 - pr)):
            if j < 0:
                continue
            ox = dx * (w + 200 * self.s)
            page = self._page(j, pw, ph)
            cx, cy = x + w / 2 + ox, y + h / 2
            blit(canvas, page, cx, cy)
            vs = get_visual_list(self.panels[j])
            n = len(vs)
            has_cap = bool(self.panels[j].get("caption"))
            area_h = ph * (0.72 if has_cap else 0.9)
            size = min(pw / (n + 0.5), area_h * (0.75 if n == 1 else 0.62))
            for i, v in enumerate(vs):
                vx = cx - pw / 2 + pw * (i + 0.5) / n
                vy = cy - ph / 2 + area_h * 0.55 + bob(t, i * 1.4, 8 * self.s)
                self.put(canvas, v, vx, vy, size, 1.0, wobble(t, i, 4))


# ---------------------------------------------------------------- math / action


class Math(Scene):
    def init(self):
        self.a, self.b = int(self.spec["a"]), int(self.spec["b"])
        self.op = self.spec.get("op", "+")
        self.res = self.a + self.b if self.op == "+" else self.a - self.b
        self.v = get_visual(self.spec)

    def steps(self):
        st = self.say_lines(self.spec.get("lines"), "l")
        st += [Say(tr_capitalize(number_to_words(self.a)), "a", "pop", gap=0.3),
               Say("artı" if self.op == "+" else "eksi", "op", None, gap=0.3),
               Say(number_to_words(self.b), "b", "pop", gap=0.3),
               Say("eşittir", "eq", None, gap=0.3),
               Say(number_to_words(self.res) + "!", "res", "tada", gap=0.5)]
        return st + self.say_lines(self.spec.get("after"), "a")

    def _group(self, canvas, t, count, cx, cy, gw, gh, appear, crossed=False, opacity=1.0):
        if count == 0:
            sp = self.title_text("0", gw * 0.6, gh * 0.4, 140)
            self.put_sprite(canvas, "zero", sp, cx, cy, pop_scale(t, appear))
            return
        cols = 1 if count == 1 else 2 if count <= 4 else 3 if count <= 9 else 4
        rows = math.ceil(count / cols)
        cell = min(gw / cols, gh / rows)
        size = cell * 0.84
        for i in range(count):
            r, c = divmod(i, cols)
            in_row = cols if r < rows - 1 else count - cols * (rows - 1)
            px = cx - in_row * cell / 2 + (c + 0.5) * cell
            py = cy - rows * cell / 2 + (r + 0.5) * cell
            sc = pop_scale(t, appear + 0.06 * i, 0.4)
            self.put(canvas, self.v, px, py + bob(t, i, 3 * self.s), size, sc, wobble(t, i, 3), opacity=opacity)
            if crossed and sc > 0.5:
                self.put(canvas, "❌", px + size * 0.25, py - size * 0.25, size * 0.5, sc)

    def draw(self, canvas, t):
        x, y, w, h = self.L.content
        s = self.s
        if self.L.portrait:
            row1, row2 = y + h * 0.28, y + h * 0.74
            slots = {"A": (x + w * 0.2, row1, w * 0.36), "op": (x + w * 0.5, row1, w * 0.18),
                     "B": (x + w * 0.8, row1, w * 0.36), "eq": (x + w * 0.22, row2, w * 0.2),
                     "R": (x + w * 0.62, row2, w * 0.55)}
            gh = h * 0.36
        else:
            row = y + h * 0.42
            slots = {"A": (x + w * 0.13, row, w * 0.25), "op": (x + w * 0.31, row, w * 0.1),
                     "B": (x + w * 0.49, row, w * 0.25), "eq": (x + w * 0.67, row, w * 0.1),
                     "R": (x + w * 0.86, row, w * 0.27)}
            gh = h * 0.56
        ax, ay, aw = slots["A"]
        self._group(canvas, t, self.a, ax, ay, aw, gh, self.t0("a"))
        bx, by, bw = slots["B"]
        self._group(canvas, t, self.b, bx, by, bw, gh, self.t0("b"), crossed=(self.op == "-"))
        rx, ry, rw = slots["R"]
        self._group(canvas, t, self.res, rx, ry, rw, gh, self.t0("res"))
        for slot, sym, tag in (("op", "+" if self.op == "+" else "−", "op"), ("eq", "=", "eq")):
            sx, sy, sw = slots[slot]
            sp = text_sprite(sym, "title", int(150 * s), rgb(self.theme["accent"]), stroke=int(12 * s),
                             stroke_fill=(255, 255, 255))
            self.put_sprite(canvas, f"msym-{sym}", sp, sx, sy, pop_scale(t, self.t0(tag)))
        for slot, val, tag in (("A", self.a, "a"), ("B", self.b, "b"), ("R", self.res, "res")):
            sx, sy, sw = slots[slot]
            badge = number_badge(str(val), self.theme["accent2"], int(110 * s))
            self.put_sprite(canvas, f"mb-{val}", badge, sx, sy + gh / 2 + 70 * s, pop_scale(t, self.t0(tag) + 0.2))


class Action(Scene):
    chapter_default = "Hareket zamanı"

    def steps(self):
        sec = float(self.spec.get("seconds", 5))
        st = self.say_lines(self.spec.get("lines"), "l") + [Sfx("boing", 0.8)]
        for k in range(int(sec)):
            st.append(Sfx("tick", 0.5, k + 1.0))
        st += [Wait(sec, "go"), Sfx("tada", 0.8)]
        after = self.spec.get("after") or ["Harika! Çok güzel yaptın!"]
        return st + self.say_lines(after, "a")

    def draw(self, canvas, t):
        x, y, w, h = self.L.content
        a, b = self.T.get("go", (1e9, 1e9))
        going = a <= t < b
        jump = -abs(math.sin((t - a) * math.pi / 0.6)) * 60 * self.s if going else 0
        self.put(canvas, get_visual(self.spec), x + w / 2, y + h * 0.4 + jump + bob(t, 0, 6 * self.s), h * 0.46,
                 pop_scale(t, 0.1), wobble(t, 0, 6 if going else 3))
        sp = fit_text(tr_upper(self.fmt(self.spec["text"])), "title", int(w * 0.8), int(h * 0.22), int(150 * self.s),
                      fill=rgb(self.theme["accent"]), stroke=int(14 * self.s), stroke_fill=(255, 255, 255))
        self.put_sprite(canvas, f"act-{self.index}", sp, x + w / 2, y + h * 0.84, pop_scale(t, 0.3),
                        wobble(t, 0, 3) if going else 0)
        if going:
            left = max(1, math.ceil(b - t))
            badge = number_badge(str(left), self.theme["accent2"], int(150 * self.s))
            self.put_sprite(canvas, f"actn-{left}", badge, x + w - 100 * self.s, y + 100 * self.s,
                            pop_scale(t, b - left, 0.3))
            d = ImageDraw.Draw(canvas)
            bw, bh = w * 0.6, 22 * self.s
            bx, by = x + (w - bw) / 2, y + h * 0.99
            d.rounded_rectangle((bx, by - bh / 2, bx + bw, by + bh / 2), bh / 2, fill=(255, 255, 255, 170))
            d.rounded_rectangle((bx, by - bh / 2, bx + bw * (1 - (t - a) / (b - a)), by + bh / 2), bh / 2,
                                fill=rgb(self.theme["accent2"]))

    def mascot(self, t):
        a, b = self.T.get("go", (1e9, 1e9))
        jump = abs(math.sin((t - a) * math.pi / 0.6)) if a <= t < b else 0
        return {"waving": a <= t < b, "happy": a <= t < b + 1, "jump": jump, "big": False}


SCENES = {
    "intro": Intro, "outro": Outro, "talk": Talk, "item": Item, "fact": Fact, "gallery": Gallery,
    "recap": Recap, "count": Count, "letter": Letter, "compare": Compare, "quiz": Quiz, "chant": Chant,
    "story": Story, "math": Math, "action": Action,
}
