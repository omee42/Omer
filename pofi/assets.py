"""Downloads and caches third-party assets (emoji art, voices) and loads fonts.

Emoji art: Microsoft Fluent Emoji 3D (MIT), packaged on npm as @lobehub/fluent-emoji-3d.
Voices: Piper Turkish voices (CC0) re-packaged in the sherpa-onnx GitHub releases.
"""
import io
import shutil
import tarfile
import time
import urllib.request
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageFont

from .config import CACHE, ROOT

FONT_DIR = ROOT / "assets" / "fonts"
FONTS = {
    "title": FONT_DIR / "Baloo2-ExtraBold.ttf",
    "title2": FONT_DIR / "Baloo2-SemiBold.ttf",
    "body": FONT_DIR / "Nunito-Black.ttf",
    "body2": FONT_DIR / "Nunito-Bold.ttf",
}

EMOJI_URL = "https://registry.npmjs.org/@lobehub/fluent-emoji-3d/-/fluent-emoji-3d-1.1.0.tgz"
EMOJI_DIR = CACHE / "emoji3d"

VOICE_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-tr_TR-{name}-medium.tar.bz2"
VOICE_DIR = CACHE / "voices"
FREE_VOICES = {"fahrettin", "fettah"}  # CC0; 'dfki' is CC BY-NC-SA and must not be used commercially


def _download(url: str, tries: int = 4) -> bytes:
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "pofi-video/1.0"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except Exception as e:
            if k == tries - 1:
                raise
            print(f"[assets] indirme hatası ({e}); {2 ** (k + 1)} sn sonra tekrar denenecek")
            time.sleep(2 ** (k + 1))


def _fresh(marker: Path, source: str) -> bool:
    """True when the cached asset was downloaded from the same source URL."""
    return marker.exists() and marker.read_text().strip() == source


def ensure_emoji() -> Path:
    marker = EMOJI_DIR / ".ok"
    if _fresh(marker, EMOJI_URL):
        return EMOJI_DIR
    print("[assets] Fluent Emoji 3D indiriliyor...")
    data = _download(EMOJI_URL)
    tmp = EMOJI_DIR.with_suffix(".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        for m in tf.getmembers():
            if m.isfile() and m.name.startswith("package/assets/") and m.name.endswith(".webp"):
                (tmp / Path(m.name).name).write_bytes(tf.extractfile(m).read())
    shutil.rmtree(EMOJI_DIR, ignore_errors=True)
    tmp.rename(EMOJI_DIR)
    marker.write_text(EMOJI_URL)
    return EMOJI_DIR


def ensure_voice(name: str) -> Path:
    if name not in FREE_VOICES:
        raise ValueError(f"'{name}' sesi kullanılamaz. Ticari kullanıma uygun sesler: {sorted(FREE_VOICES)}")
    d = VOICE_DIR / name
    onnx = d / f"tr_TR-{name}-medium.onnx"
    url = VOICE_URL.format(name=name)
    if _fresh(d / ".ok", url) and onnx.exists():
        return onnx
    print(f"[assets] '{name}' sesi indiriliyor...")
    data = _download(url)
    tmp = d.with_suffix(".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:bz2") as tf:
        for m in tf.getmembers():
            base = Path(m.name).name
            if m.isfile() and (base.endswith(".onnx") or base.endswith(".onnx.json") or base == "MODEL_CARD"):
                (tmp / base).write_bytes(tf.extractfile(m).read())
    if not (tmp / onnx.name).exists() or not (tmp / (onnx.name + ".json")).exists():
        raise RuntimeError(f"'{name}' ses paketi eksik dosya içeriyor")
    (tmp / ".ok").write_text(url)
    shutil.rmtree(d, ignore_errors=True)
    tmp.rename(d)
    return onnx


@lru_cache(maxsize=None)
def font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONTS[kind]), size)


def _norm_key(seq: str) -> str:
    parts = [p for p in seq.replace("_", "-").lower().split("-") if p and p != "fe0f"]
    return "-".join(parts)


@lru_cache(maxsize=1)
def emoji_index() -> dict:
    d = ensure_emoji()
    return {_norm_key(p.stem): p for p in d.glob("*.webp")}


def emoji_key(emoji: str) -> str:
    return _norm_key("-".join(f"{ord(c):x}" for c in emoji))


def has_emoji(emoji: str) -> bool:
    return emoji_key(emoji) in emoji_index()


@lru_cache(maxsize=512)
def emoji_image(emoji: str) -> Image.Image:
    path = emoji_index().get(emoji_key(emoji))
    if path is None:
        raise KeyError(f"Emoji bulunamadı: {emoji!r} ({emoji_key(emoji)})")
    return Image.open(path).convert("RGBA")
