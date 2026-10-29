"""3D video frames: exports the timeline as a JSON plan and renders it with three.js in headless Chromium.

The Python side keeps doing everything that decides *what* happens and *when* (voice, timing tags, captions,
music); pofi/web/app.js only draws. See pofi/web/render.mjs for the browser driver.
"""
import copy
import json
import shutil
import subprocess
from pathlib import Path

from .assets import emoji_index, emoji_key
from .render import theme_for
from .schema import get_visual, get_visual_list

WEB = Path(__file__).resolve().parent / "web"
FIXED_EMOJI = ["✅", "🎵", "❌", "💡", "⭐"]


class Render3DError(RuntimeError):
    pass


def available():
    """Why 3D can't run here, or None when it can."""
    if not shutil.which("node"):
        return "node bulunamadı"
    if not (WEB / "node_modules" / "three").exists() or not (WEB / "node_modules" / "playwright").exists():
        return "pofi/web içinde 'npm ci' çalıştırılmamış"
    return None


def _fmt_all(obj, fmt):
    if isinstance(obj, str):
        return fmt(obj)
    if isinstance(obj, list):
        return [_fmt_all(x, fmt) for x in obj]
    if isinstance(obj, dict):
        return {k: _fmt_all(v, fmt) for k, v in obj.items()}
    return obj


def _spec(sc, ep):
    """The scene spec with {maskot} filled in and every picture under one predictable key."""
    s = _fmt_all(copy.deepcopy(sc.spec), sc.fmt)
    raw = sc.spec
    kind = raw["type"]
    if kind in ("item", "count", "fact", "math", "action", "chant"):
        s["visual"] = get_visual(raw)
    if kind in ("intro", "talk", "outro"):
        s["visuals"] = get_visual_list(raw)
    if kind in ("gallery", "recap"):
        for it, r in zip(s["items"], raw["items"]):
            it["visual"] = get_visual(r)
    if kind == "quiz":
        for it, r in zip(s["options"], raw["options"]):
            it["visual"] = get_visual(r)
        s["mood"] = ep.get("mood", "happy")
    if kind == "letter":
        for it, r in zip(s["words"], raw["words"]):
            it["visual"] = get_visual(r)
    if kind == "compare":
        for side in ("left", "right"):
            s[side]["visual"] = get_visual(raw[side])
    if kind == "chant":
        for it, r in zip(s["items"], raw["items"]):
            it["visual"] = get_visual(r)
    if kind == "story":
        for it, r in zip(s["panels"], raw["panels"]):
            it["visuals"] = get_visual_list(r)
    if kind == "outro" and raw.get("next"):
        s["next"]["visual"] = get_visual(raw["next"])
    return s


def _emoji_in(obj, out):
    if isinstance(obj, str):
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("visual", "visuals"):
                for x in (v if isinstance(v, list) else [v]):
                    if isinstance(x, str):
                        out.add(x)
                    elif isinstance(x, dict) and isinstance(x.get("emoji"), str):
                        out.add(x["emoji"])
            else:
                _emoji_in(v, out)
    elif isinstance(obj, list):
        for x in obj:
            _emoji_in(x, out)


def _emoji_map(used):
    index = emoji_index()
    out = {}
    for e in sorted(used):
        p = index.get(emoji_key(e))
        if p is None:
            raise Render3DError(f"emoji çizimi yok: {e!r}")
        out[e] = "/emoji/" + p.name
    return out


def _theme(ep):
    th = theme_for(ep)
    return {"bg": list(th["bg"]), "accent": th["accent"], "accent2": th["accent2"], "dark": th["dark"], "name": th["name"]}


def _episode(ep, cfg):
    fmt = lambda s: str(s).replace("{maskot}", cfg["kanal"]["maskot"]).replace("{kanal}", cfg["kanal"]["ad"])
    return {"title": fmt(ep["title"]), "category": ep.get("category", ""), "date": str(ep.get("date", "")),
            "mood": ep.get("mood", "happy"), "channel": cfg["kanal"]["ad"]}, fmt


def build_plan(tl, mouth_env):
    ctx = tl.ctx
    ep = ctx.ep
    scenes = []
    used = set(FIXED_EMOJI)
    for sc in tl.scenes:
        spec = _spec(sc, ep)
        _emoji_in(spec, used)
        scenes.append({"type": spec["type"], "spec": spec, "start": round(sc.start, 4), "dur": round(sc.dur, 4),
                       "T": {k: [round(a, 4), round(b, 4)] for k, (a, b) in sc.T.items()}})
    return {
        "width": ctx.W, "height": ctx.H, "fps": tl.fps, "frames": tl.n_frames,
        "theme": _theme(ep),
        "episode": _episode(ep, ctx.cfg)[0],
        "emoji": _emoji_map(used),
        "mouth": [round(float(x), 3) for x in mouth_env],
        "captions": [[round(a, 3), round(b, 3), ctx.fmt(text)] for a, b, text in tl.captions],
        "show_captions": bool(ctx.cfg["video"].get("altyazi_goster", True)),
        "scenes": scenes,
    }


def _run(args, timeout):
    why = available()
    if why:
        raise Render3DError(why)
    p = subprocess.run(["node", str(WEB / "render.mjs")] + args, stdout=subprocess.PIPE, timeout=timeout, text=True)
    if p.returncode != 0:
        raise Render3DError(f"3D çizim başarısız (çıkış kodu {p.returncode})")
    lines = [l for l in p.stdout.splitlines() if l.startswith("{")]
    if not lines:
        raise Render3DError("3D çizici sonuç bildirmedi")
    return json.loads(lines[-1])


def render_segments(tl, mouth_env, tmp, workers=3, frames=None, timeout=4 * 3600):
    """Render all frames (or the range `frames=(a, b)`) to MP4 segments inside tmp; returns their paths."""
    tmp = Path(tmp)
    plan_path = tmp / "plan3d.json"
    plan_path.write_text(json.dumps(build_plan(tl, mouth_env), ensure_ascii=False), encoding="utf-8")
    args = ["--plan", str(plan_path), "--out", str(tmp), "--workers", str(workers)]
    if frames:
        args += ["--from", str(frames[0]), "--to", str(frames[1])]
    res = _run(args, timeout)
    print(f"[3d] {len(res['segments'])} parça {res['seconds'] / 60:.1f} dakikada çizildi")
    return [Path(s) for s in res["segments"]]


def thumbnail_plan(ep, cfg, text, visuals, W=1280, H=720):
    episode, fmt = _episode(ep, cfg)
    used = set(FIXED_EMOJI)
    _emoji_in({"visuals": visuals}, used)
    return {"width": W, "height": H, "fps": 30, "frames": 1, "theme": _theme(ep), "episode": episode,
            "emoji": _emoji_map(used), "mouth": [0], "captions": [], "show_captions": False, "scenes": [],
            "thumb": {"text": fmt(text), "visuals": visuals}}


def render_thumbnail(ep, cfg, text, visuals, out_jpg, timeout=600):
    """YouTube cover rendered in the episode's 3D world."""
    import tempfile

    with tempfile.TemporaryDirectory(prefix="pofi-thumb-") as td:
        plan_path = Path(td) / "plan.json"
        plan_path.write_text(json.dumps(thumbnail_plan(ep, cfg, text, visuals), ensure_ascii=False), encoding="utf-8")
        _run(["--plan", str(plan_path), "--thumb", str(out_jpg)], timeout)
    return Path(out_jpg)
