"""Command line: python -m pofi <komut>

  setup                      Emoji çizimlerini ve sesi indirir
  validate [dosyalar...]     Senaryoları kontrol eder (varsayılan: bolumler/*.json)
  preview DOSYA              Her sahneden kareler içeren hızlı bir önizleme (PNG) üretir
  render DOSYA               Videoyu, Shorts'u, kapağı, altyazıyı ve YouTube metnini üretir
  gunluk [--tarih T]         O günün bölümünü bulur ve üretir (GitHub Actions bunu kullanır)
  takvim [--gun N]           Yaklaşan günleri ve senaryosu hazır olanları listeler
"""
import argparse
import datetime as dt
import json
import sys
import zlib
from pathlib import Path

from .config import ROOT, load_config

EPISODES = ROOT / "bolumler"
SPARE = EPISODES / "yedek"
OUT = ROOT / "cikti"


def _today_tr():
    return (dt.datetime.utcnow() + dt.timedelta(hours=3)).date()


def find_episode(date: str):
    hits = sorted(EPISODES.glob(f"{date}*.json"))
    if hits:
        return hits[0], False
    spares = sorted(SPARE.glob("*.json"))
    if not spares:
        return None, False
    return spares[zlib.crc32(date.encode()) % len(spares)], True


def cmd_setup(args):
    from .assets import ensure_emoji, ensure_voice
    cfg = load_config()
    ensure_emoji()
    if cfg["ses"].get("kalite_kontrol", True) and int(cfg["ses"].get("deneme", 3)) > 1:
        from .asr import ensure_asr
        ensure_asr()
    for spec in (cfg["ses"]["motor"], cfg["ses"]["yedek"]):
        kind, _, name = spec.partition(":")
        if kind == "piper":
            ensure_voice(name)
    print("Hazır.")


def cmd_validate(args):
    from .schema import validate
    files = [Path(f) for f in args.files] or sorted(EPISODES.glob("*.json")) + sorted(SPARE.glob("*.json"))
    bad = 0
    ids = {}
    for f in files:
        try:
            ep = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"✗ {f.name}: JSON hatası: {e}")
            bad += 1
            continue
        errs = validate(ep)
        if not errs and f.parent == EPISODES and not f.name.startswith(ep["date"]):
            errs.append(f"dosya adı '{ep['date']}' ile başlamalı")
        if ep.get("id") in ids:
            errs.append(f"id '{ep['id']}' {ids[ep['id']]} dosyasında da kullanılmış")
        ids[ep.get("id")] = f.name
        if errs:
            bad += 1
            print(f"✗ {f.name}")
            for e in errs:
                print(f"    - {e}")
        else:
            print(f"✓ {f.name}")
    print(f"\n{len(files) - bad}/{len(files)} senaryo geçerli.")
    return 1 if bad else 0


class EstimateTTS:
    """Silent stand-in for quick previews: duration estimated from text length."""

    def __init__(self):
        self.report = {}

    def say(self, text):
        import numpy as np
        from .audio import SR
        from .text import normalize_for_tts
        n = len(normalize_for_tts(text))
        return np.zeros(int((0.3 + n / 17.0) * SR), dtype=np.float32)


def cmd_preview(args):
    from PIL import Image
    from .render import make_short, render_episode
    from .schema import load_episode
    ep = load_episode(args.file)
    if args.format == "short":
        ep = make_short(ep)
    tts = EstimateTTS() if args.hizli else None
    tl, fr = render_episode(ep, None, fmt=args.format, preview_only=True, tts=tts)
    frames = []
    for sc in tl.scenes:
        marks = sorted(set([0.35 * sc.dur, 0.7 * sc.dur, max(0.0, sc.dur - 0.5)]))
        for tag in ("reveal", "done"):
            if tag in sc.T:
                marks.append(sc.T[tag][0] + 0.7)
        for m in marks[:3]:
            frames.append((sc.start + m, sc.spec["type"]))
    thumbs = []
    tw = 640 if args.format != "short" else 300
    for t, kind in frames:
        im = fr.render(int(t * tl.fps)).resize((tw, int(tw * tl.ctx.H / tl.ctx.W)))
        thumbs.append(im)
    cols = 3 if args.format != "short" else 6
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumbs[0].width, rows * thumbs[0].height), (30, 30, 30))
    for i, im in enumerate(thumbs):
        sheet.paste(im, ((i % cols) * im.width, (i // cols) * im.height))
    out = Path(args.out or OUT / ep["id"] / f"onizleme{'-short' if args.format == 'short' else ''}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    if args.frames:
        for t in args.frames:
            fr.render(int(float(t) * tl.fps)).save(out.with_name(f"kare-{float(t):06.2f}.png"))
    print(f"Önizleme: {out}  (süre {tl.duration:.1f} sn, {len(tl.scenes)} sahne)")
    for sc in tl.scenes:
        print(f"  {sc.start:6.1f}s  {sc.dur:5.1f}s  {sc.spec['type']}")
    return 0


def produce(path: Path, out_dir: Path, short=True, workers=None):
    from .publish import metadata, thumbnail, write_metadata
    from .render import make_short, render_episode, write_srt
    from .schema import load_episode
    from .tts import TTS
    cfg = load_config()
    ep = load_episode(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    tts = TTS(cfg)
    tl = render_episode(ep, out_dir / "video.mp4", cfg, "long", workers, tts)
    write_srt(tl, out_dir / "altyazi.srt")
    write_voice_report(tts, out_dir / "ses_raporu.txt")
    thumbnail(ep, out_dir / "kapak.jpg", cfg)
    write_metadata(metadata(ep, tl, cfg), out_dir / "youtube.json", out_dir / "youtube.txt")
    if short and cfg["shorts"]["aktif"]:
        sep = make_short(ep)
        stl = render_episode(sep, out_dir / "shorts.mp4", cfg, "short", workers, tts)
        if stl.duration > 60 and not ep.get("shorts") and len(sep["scenes"]) > 3:
            sep["scenes"] = sep["scenes"][:2] + sep["scenes"][-1:]
            stl = render_episode(sep, out_dir / "shorts.mp4", cfg, "short", workers, tts)
        write_metadata(metadata(ep, stl, cfg, short=True), out_dir / "shorts.json", out_dir / "shorts.txt")
    print(f"Bitti → {out_dir}")
    return tl


def write_voice_report(tts, path):
    if not tts.report:
        return
    rows = sorted(tts.report.items(), key=lambda kv: kv[1]["score"])
    weak = [(t, r) for t, r in rows if r["score"] < 0.7]
    out = [f"Seslendirme kalite raporu — {len(rows)} cümle, {len(weak)} tanesi kontrol edilmeli.",
           "Puan: Whisper'ın cümleyi ne kadar doğru duyduğu (1.0 = tamamen doğru).", ""]
    for t, r in weak:
        out.append(f"[{r['score']:.2f}] {t}\n        duyulan: {r['heard']}")
    Path(path).write_text("\n".join(out) + "\n", encoding="utf-8")


def cmd_render(args):
    p = Path(args.file)
    ep = json.loads(p.read_text(encoding="utf-8"))
    produce(p, Path(args.out or OUT / ep["id"]), short=not args.no_short, workers=args.workers)
    return 0


def cmd_daily(args):
    date = args.tarih or _today_tr().isoformat()
    path, spare = find_episode(date)
    if path is None:
        print(f"{date} için senaryo yok ve yedek bölüm bulunamadı.", file=sys.stderr)
        return 2
    if spare:
        print(f"⚠ {date} için senaryo yok; yedek bölüm kullanılıyor: {path.name}")
    out = Path(args.out or OUT / date)
    produce(path, out, short=not args.no_short, workers=args.workers)
    (out / "bolum.txt").write_text(f"{date}\n{path.relative_to(ROOT)}\n{'yedek' if spare else 'planli'}\n")
    return 0


def cmd_calendar(args):
    import yaml
    cal_path = ROOT / "takvim.yaml"
    cal = yaml.safe_load(cal_path.read_text(encoding="utf-8")) if cal_path.exists() else {}
    days = {str(d["tarih"]): d for d in cal.get("gunler", [])}
    start = dt.date.fromisoformat(args.baslangic) if args.baslangic else _today_tr()
    missing = 0
    for k in range(args.gun):
        d = (start + dt.timedelta(days=k)).isoformat()
        has = bool(sorted(EPISODES.glob(f"{d}*.json")))
        missing += not has
        info = days.get(d, {})
        print(f"{'✓' if has else '·'} {d}  {info.get('kategori', '?'):14s} {info.get('konu', '')}")
    print(f"\n{args.gun - missing}/{args.gun} gün için senaryo hazır.")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pofi", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup")
    v = sub.add_parser("validate")
    v.add_argument("files", nargs="*")
    p = sub.add_parser("preview")
    p.add_argument("file")
    p.add_argument("--out")
    p.add_argument("--format", choices=["long", "short"], default="long")
    p.add_argument("--frames", nargs="*", help="ayrıca kaydedilecek saniyeler")
    p.add_argument("--hizli", action="store_true", help="seslendirme yapmadan, tahmini sürelerle hızlı önizleme")
    r = sub.add_parser("render")
    r.add_argument("file")
    r.add_argument("--out")
    r.add_argument("--no-short", action="store_true")
    r.add_argument("--workers", type=int)
    g = sub.add_parser("gunluk")
    g.add_argument("--tarih")
    g.add_argument("--out")
    g.add_argument("--no-short", action="store_true")
    g.add_argument("--workers", type=int)
    c = sub.add_parser("takvim")
    c.add_argument("--gun", type=int, default=30)
    c.add_argument("--baslangic")
    args = ap.parse_args(argv)
    fn = {"setup": cmd_setup, "validate": cmd_validate, "preview": cmd_preview, "render": cmd_render,
          "gunluk": cmd_daily, "takvim": cmd_calendar}[args.cmd]
    return fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
