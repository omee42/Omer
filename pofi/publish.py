"""Thumbnail and YouTube metadata for a rendered episode."""
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from . import gfx
from .gfx import blit, fit_text, rgb
from .mascot import owl
from .render import theme_for
from .schema import get_visual, get_visual_list
from .text import tr_upper

CREDITS = (
    "Emoji çizimleri: Microsoft Fluent Emoji (MIT lisansı).\n"
    "Müzik, ses efektleri ve animasyon: kanalımız tarafından üretilmiştir."
)


def _main_visuals(ep):
    th = ep["youtube"].get("thumbnail") or {}
    if th.get("emoji"):
        return th["emoji"][:3]
    vs = get_visual_list(ep["scenes"][0])
    if vs:
        return vs[:3]
    for s in ep["scenes"]:
        v = get_visual(s) if s["type"] in ("item", "fact", "count") else None
        if v:
            return [v]
    return ["⭐"]


def thumbnail(ep, path, cfg, W=1280, H=720):
    text = (ep["youtube"].get("thumbnail") or {}).get("text") or ep["title"]
    if str(cfg["video"].get("motor", "2d")).lower() == "3d":
        from . import render3d
        try:
            return render3d.render_thumbnail(ep, cfg, text, _main_visuals(ep), path)
        except Exception as e:
            print(f"[kapak] UYARI: 3D kapak çizilemedi ({e}); 2D kapak kullanılıyor")
    return thumbnail_2d(ep, path, cfg, W, H)


def thumbnail_2d(ep, path, cfg, W=1280, H=720):
    th = theme_for(ep)
    img = gfx.radial_gradient((W, H), gfx.lighten(th["bg"][0], 0.2), th["bg"][1], 0.5, 0.4, 0.9).convert("RGBA")
    # sunburst rays for energy
    rays = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(rays)
    cx, cy = W * 0.66, H * 0.48
    for k in range(18):
        a0 = k * 2 * math.pi / 18
        a1 = a0 + math.pi / 18
        d.polygon([(cx, cy), (cx + 1600 * math.cos(a0), cy + 1600 * math.sin(a0)),
                   (cx + 1600 * math.cos(a1), cy + 1600 * math.sin(a1))], fill=45)
    white = Image.new("RGBA", (W, H), (255, 255, 255, 0))
    white.putalpha(rays.filter(ImageFilter.GaussianBlur(3)))
    img.alpha_composite(white)
    vs = _main_visuals(ep)
    n = len(vs)
    sizes = {1: [450], 2: [370, 270], 3: [340, 250, 220]}[n]
    spots = {1: [(0.73, 0.47)], 2: [(0.70, 0.42), (0.88, 0.68)], 3: [(0.70, 0.40), (0.89, 0.68), (0.89, 0.22)]}[n]
    for v, size, (fx, fy) in reversed(list(zip(vs, sizes, spots))):
        blit(img, gfx.visual(v, size, -6 if fx > 0.8 else 4), W * fx, H * fy)
    text = (ep["youtube"].get("thumbnail") or {}).get("text") or ep["title"]
    words = tr_upper(text)
    sp = fit_text(words, "title", int(W * 0.47), int(H * 0.6), 230, min_size=60, max_chars=11,
                  fill=(255, 255, 255), stroke=16, stroke_fill=rgb(th["dark"]))
    blit(img, sp, W * 0.27, H * 0.38)
    blit(img, owl(300, 3, "open", 1), 175, H + 8, anchor="bottom")
    band = gfx.rounded_card(int(W * 0.34), 64, 32, rgb(th["accent"]), 255)
    lab = gfx.text_sprite(cfg["kanal"]["ad"], "title2", 34, gfx.ink(th["accent"], th["dark"]), shadow=False)
    band = band.copy()
    blit(band, lab, band.width / 2, band.height / 2 - 4)
    blit(img, band, W * 0.69, H - 56)
    img.convert("RGB").save(path, quality=92)


def _fmt_time(sec):
    sec = int(sec)
    return f"{sec // 60}:{sec % 60:02d}"


def metadata(ep, tl, cfg, short=False):
    yt = ep["youtube"]
    title = yt["title"] if not short else (yt.get("short_title") or yt["title"])[:90]
    if short and "#shorts" not in title.lower():
        title = (title[:88] + " #shorts") if len(title) > 88 else title + " #shorts"
    desc = [yt["description"].strip(), ""]
    chapters = [] if short else tl.chapters()
    if chapters:
        desc.append("Bölümler:")
        desc += [f"{_fmt_time(s)} {name}" for s, name in chapters]
        desc.append("")
    desc.append(f"👨‍👩‍👧 Bu video {cfg['kanal']['yas_grubu']} grubundaki çocuklar için hazırlanmıştır. "
                "Ebeveyn eşliğinde izlenmesi önerilir.")
    if cfg["kanal"].get("web"):
        desc.append(cfg["kanal"]["web"])
    desc += ["", CREDITS]
    tags = []
    for t in list(yt.get("tags", [])) + list(cfg["youtube"].get("etiketler", [])):
        if t not in tags and sum(len(x) + 2 for x in tags) + len(t) < 480:
            tags.append(t)
    return {
        "title": title,
        "description": "\n".join(desc)[:4900],
        "tags": tags,
        "categoryId": str(cfg["youtube"].get("kategori", 27)),
        "defaultLanguage": "tr",
        "defaultAudioLanguage": "tr",
        "madeForKids": bool(cfg["youtube"].get("cocuklar_icin", True)),
        "privacyStatus": cfg["youtube"].get("gizlilik", "private"),
        "duration_seconds": round(tl.duration, 1),
    }


def write_metadata(meta, json_path, txt_path):
    Path(json_path).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    txt = (f"BAŞLIK:\n{meta['title']}\n\nAÇIKLAMA:\n{meta['description']}\n\nETİKETLER:\n{', '.join(meta['tags'])}\n\n"
           f"KİTLE: {'Evet, çocuklara özel' if meta['madeForKids'] else 'Hayır'}\nSÜRE: {_fmt_time(meta['duration_seconds'])}\n")
    Path(txt_path).write_text(txt, encoding="utf-8")
