"""Episode file format (JSON) and its validation.

An episode is a list of scenes; each scene type is documented in docs/SENARYO_REHBERI.md.
"""
import json
import re
import unicodedata
from pathlib import Path

from .gfx import SHAPE_ALIASES, SHAPES
from .text import is_emoji_char

SCENE_TYPES = {
    "intro", "talk", "item", "gallery", "recap", "count", "letter", "compare", "fact",
    "quiz", "chant", "story", "math", "action", "outro",
}
MOODS = {"happy", "calm", "energetic"}
MAX_LINE = 170
HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


class EpisodeError(ValueError):
    pass


def get_visual(obj, key="visual"):
    """Normalise the different ways a picture can be written in a scene."""
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj
    if key in obj:
        v = obj[key]
        return v if isinstance(v, (str, dict)) else None
    if "emoji" in obj and isinstance(obj["emoji"], (str, dict)):
        return obj["emoji"]
    if "shape" in obj:
        return {"shape": obj["shape"], "color": obj.get("color", "#FF5A5F")}
    if "flag" in obj:
        return {"flag": obj["flag"]}
    if "symbol" in obj:
        return {"symbol": str(obj["symbol"]), "color": obj.get("color", "#7C6CF2")}
    return None


def get_visual_list(obj, key="emoji"):
    v = obj.get(key) if isinstance(obj, dict) else None
    if v is None and isinstance(obj, dict):
        v = obj.get("visuals")
    if v is None:
        single = get_visual(obj) if isinstance(obj, dict) else None
        return [single] if single else []
    if isinstance(v, (str, dict)):
        return [v]
    return list(v)


def _check_visual(v, where, errors):
    from .assets import has_emoji

    if v is None:
        errors.append(f"{where}: görsel eksik (emoji / shape / flag / symbol)")
        return
    if isinstance(v, str):
        if not has_emoji(v):
            errors.append(f"{where}: '{v}' emojisi çizim setinde yok, başka bir emoji seçin")
        return
    if "emoji" in v:
        _check_visual(v["emoji"], where, errors)
    elif "shape" in v:
        if SHAPE_ALIASES.get(v["shape"], v["shape"]) not in SHAPES:
            errors.append(f"{where}: bilinmeyen şekil '{v['shape']}'. Seçenekler: {', '.join(SHAPES)}")
        if "color" in v and not HEX.match(str(v["color"])):
            errors.append(f"{where}: renk #RRGGBB biçiminde olmalı")
    elif "flag" in v:
        if str(v["flag"]).lower() not in ("tr", "türkiye", "turkiye"):
            errors.append(f"{where}: şimdilik yalnızca 'tr' bayrağı çizilebiliyor")
    elif "symbol" in v:
        if not (1 <= len(str(v["symbol"])) <= 6):
            errors.append(f"{where}: symbol 1-6 karakter olmalı")
        if any(is_emoji_char(c) for c in str(v["symbol"])):
            errors.append(f"{where}: symbol içinde emoji olamaz; emoji için \"emoji\" alanını kullanın")
        if "color" in v and not HEX.match(str(v["color"])):
            errors.append(f"{where}: renk #RRGGBB biçiminde olmalı")
    else:
        errors.append(f"{where}: görsel anlaşılamadı: {v}")


def _check_text(value, where, errors, max_len=None, required=True):
    """On-screen text: must be a string, without emoji (the fonts cannot draw them)."""
    if value is None or value == "":
        if required:
            errors.append(f"{where}: gerekli")
        return False
    if not isinstance(value, str):
        errors.append(f"{where}: metin (\"...\") olmalı")
        return False
    if any(is_emoji_char(c) for c in value):
        errors.append(f"{where}: ekranda yazan metinde emoji olamaz (yazı tipi çizemez); emojiyi görsel alanına koyun")
    if max_len and len(value) > max_len:
        errors.append(f"{where}: en fazla {max_len} karakter olmalı ({len(value)} var)")
    return True


def _check_lines(lines, where, errors, required=False, max_n=8):
    if lines is None:
        if required:
            errors.append(f"{where}: 'lines' gerekli")
        return
    if not isinstance(lines, list) or not all(isinstance(x, str) for x in lines):
        errors.append(f"{where}: cümleler metin listesi olmalı")
        return
    if required and not lines:
        errors.append(f"{where}: en az bir cümle gerekli")
    if len(lines) > max_n:
        errors.append(f"{where}: en fazla {max_n} cümle olabilir ({len(lines)} var)")
    for i, l in enumerate(lines):
        if not l.strip():
            errors.append(f"{where}[{i}]: boş cümle")
        if len(l) > MAX_LINE:
            errors.append(f"{where}[{i}]: cümle çok uzun ({len(l)} > {MAX_LINE} karakter); bölün")


def _check_items(items, where, errors, need_line, min_n=1, max_n=6, need_label=True):
    if not isinstance(items, list) or not (min_n <= len(items) <= max_n):
        errors.append(f"{where}: {min_n}-{max_n} öğe olmalı")
        return
    for i, it in enumerate(items):
        w = f"{where}[{i}]"
        if not isinstance(it, dict):
            errors.append(f"{w}: nesne olmalı")
            continue
        _check_visual(get_visual(it), w, errors)
        if need_label or "label" in it:
            _check_text(it.get("label"), f"{w}.label", errors, 22)
        if need_line or it.get("line"):
            if not isinstance(it.get("line"), str) or not it["line"].strip():
                errors.append(f"{w}: 'line' (söylenecek cümle) gerekli")
            elif len(it["line"]) > MAX_LINE:
                errors.append(f"{w}: 'line' çok uzun")


def validate(ep: dict) -> list:
    errors = []
    if not isinstance(ep, dict):
        return ["dosya bir JSON nesnesi ({...}) olmalı"]
    for key in ("id", "date", "title", "category", "scenes", "youtube"):
        if key not in ep:
            errors.append(f"'{key}' alanı eksik")
    if errors:
        return errors
    if not isinstance(ep["id"], str) or not re.match(r"^[a-z0-9-]+$", ep["id"]):
        errors.append("'id' küçük harf, rakam ve '-' içeren bir metin olmalı (ör. 2026-10-30-sonbahar)")
    if not isinstance(ep["date"], str) or not re.match(r"^\d{4}-\d{2}-\d{2}$", ep["date"]):
        errors.append("'date' YYYY-AA-GG biçiminde olmalı")
    _check_text(ep["title"], "title", errors, 40)
    if ep.get("mood", "happy") not in MOODS:
        errors.append(f"'mood' şunlardan biri olmalı: {sorted(MOODS)}")
    theme = ep.get("theme")
    if theme is not None:
        from .render import THEMES
        names = [t["name"] for t in THEMES]
        if isinstance(theme, str):
            if theme not in names:
                errors.append(f"'theme' şunlardan biri olmalı: {', '.join(names)}")
        elif isinstance(theme, dict):
            for k, v in theme.items():
                vals = v if (k == "bg" and isinstance(v, list) and len(v) == 2) else [v]
                if k not in ("bg", "accent", "accent2", "dark") or not all(HEX.match(str(x)) for x in vals):
                    errors.append(f"'theme.{k}' geçersiz (bg: iki renk listesi; accent/accent2/dark: #RRGGBB)")
        else:
            errors.append("'theme' bir tema adı ya da renk nesnesi olmalı")
    yt = ep["youtube"]
    if not isinstance(yt, dict) or not yt.get("title") or not yt.get("description"):
        errors.append("youtube.title ve youtube.description gerekli")
    else:
        if len(yt["title"]) > 100:
            errors.append("youtube.title en fazla 100 karakter olmalı")
        if len(yt.get("short_title") or "") > 90:
            errors.append("youtube.short_title en fazla 90 karakter olmalı")
        if re.search(r"[<>]", yt["title"] + yt["description"] + (yt.get("short_title") or "")):
            errors.append("YouTube başlık/açıklamasında < veya > kullanılamaz")
        th = yt.get("thumbnail") or {}
        if th.get("text"):
            _check_text(th.get("text"), "youtube.thumbnail.text", errors, 18)
        for i, v in enumerate(th.get("emoji", [])):
            _check_visual(v, f"youtube.thumbnail.emoji[{i}]", errors)
        if not isinstance(yt.get("tags", []), list):
            errors.append("youtube.tags bir liste olmalı")
    _check_scenes(ep["scenes"], "scenes", errors)
    if ep.get("shorts") is not None:
        _check_scenes(ep["shorts"], "shorts", errors)
    return errors


def _check_scenes(scenes, prefix, errors):
    if not isinstance(scenes, list) or len(scenes) < 3:
        errors.append(f"{prefix}: en az 3 sahne gerekli")
        return
    if not isinstance(scenes[0], dict) or scenes[0].get("type") != "intro":
        errors.append(f"{prefix}: ilk sahne 'intro' olmalı")
    if not isinstance(scenes[-1], dict) or scenes[-1].get("type") != "outro":
        errors.append(f"{prefix}: son sahne 'outro' olmalı")
    for i, s in enumerate(scenes):
        w = f"{prefix}[{i}]({s.get('type') if isinstance(s, dict) else '?'})"
        if not isinstance(s, dict):
            errors.append(f"{w}: sahne bir nesne olmalı")
            continue
        try:
            _check_scene(s, w, errors)
        except (TypeError, AttributeError, ValueError, KeyError) as e:
            errors.append(f"{w}: alan türü hatalı ({e}); rehberdeki biçime bakın")


def _check_scene(s, w, errors):
    t = s.get("type")
    if t not in SCENE_TYPES:
        errors.append(f"{w}: bilinmeyen sahne türü. Seçenekler: {', '.join(sorted(SCENE_TYPES))}")
        return
    _check_lines(s.get("lines"), w + ".lines", errors,
                 required=t in ("intro", "talk", "item", "fact", "outro", "action"))
    _check_lines(s.get("after"), w + ".after", errors)
    if "title" in s and t != "intro":
        _check_text(s["title"], w + ".title", errors, 40)
    if t == "intro":
        _check_text(s.get("title"), w + ".title", errors, 40)
        if s.get("subtitle"):
            _check_text(s["subtitle"], w + ".subtitle", errors, 40)
        vs = get_visual_list(s)
        if len(vs) > 5:
            errors.append(f"{w}: en fazla 5 görsel")
        for j, v in enumerate(vs):
            _check_visual(v, f"{w}.emoji[{j}]", errors)
    elif t == "talk":
        for j, v in enumerate(get_visual_list(s)):
            _check_visual(v, f"{w}.emoji[{j}]", errors)
        if len(get_visual_list(s)) > 3:
            errors.append(f"{w}: en fazla 3 görsel")
        if s.get("text"):
            _check_text(s["text"], w + ".text", errors, 60)
    elif t == "item":
        _check_visual(get_visual(s), w, errors)
        _check_text(s.get("word"), w + ".word", errors, 16)
        if "color" in s and not HEX.match(str(s["color"])):
            errors.append(f"{w}: 'color' #RRGGBB olmalı")
    elif t in ("gallery", "recap"):
        _check_items(s.get("items"), w + ".items", errors, need_line=(t == "gallery"), max_n=6)
    elif t == "count":
        _check_visual(get_visual(s), w, errors)
        n = s.get("n")
        if not isinstance(n, int) or isinstance(n, bool) or not (1 <= n <= 20):
            errors.append(f"{w}: 'n' 1-20 arasında tam sayı olmalı")
        if "label" in s:
            _check_text(s["label"], w + ".label", errors, 16)
    elif t == "letter":
        from .text import TR_ALPHABET, tr_upper
        letter = s.get("letter")
        if not isinstance(letter, str) or tr_upper(letter) not in TR_ALPHABET:
            errors.append(f"{w}: 'letter' Türk alfabesinden tek harf olmalı")
            return
        words = s.get("words")
        if not isinstance(words, list) or not (1 <= len(words) <= 4):
            errors.append(f"{w}: 1-4 kelime olmalı")
            return
        L = tr_upper(letter)
        for j, wd in enumerate(words):
            ww = f"{w}.words[{j}]"
            if not isinstance(wd, dict):
                errors.append(f"{ww}: {{\"word\", \"emoji\", \"line\"}} nesnesi olmalı")
                continue
            _check_visual(get_visual(wd), ww, errors)
            if not _check_text(wd.get("word"), ww + ".word", errors, 16):
                continue
            if not isinstance(wd.get("line"), str) or not wd["line"].strip():
                errors.append(f"{ww}: 'line' gerekli")
            up = tr_upper(wd["word"])
            if L == "Ğ":
                if L not in up:
                    errors.append(f"{ww}: '{wd['word']}' içinde 'ğ' yok (Ğ ile başlayan kelime olmadığı için içinde geçmesi yeterli)")
            elif not up.startswith(L):
                errors.append(f"{ww}: '{wd['word']}' '{letter}' harfiyle başlamıyor")
    elif t == "compare":
        for side in ("left", "right"):
            it = s.get(side)
            if not isinstance(it, dict):
                errors.append(f"{w}: '{side}' gerekli")
                continue
            _check_visual(get_visual(it), f"{w}.{side}", errors)
            _check_text(it.get("label"), f"{w}.{side}.label", errors, 22)
            if not isinstance(it.get("line"), str) or not it["line"].strip():
                errors.append(f"{w}.{side}: 'line' gerekli")
            if "scale" in it:
                sc = it["scale"]
                if not isinstance(sc, (int, float)) or isinstance(sc, bool) or not (0.2 <= sc <= 1.0):
                    errors.append(f"{w}.{side}: 'scale' 0.2 ile 1.0 arasında bir sayı olmalı (küçük olan tarafı küçültür)")
    elif t == "fact":
        _check_visual(get_visual(s), w, errors)
        _check_text(s.get("text"), w + ".text", errors, 90)
    elif t == "quiz":
        _check_text(s.get("question"), w + ".question", errors, 50)
        _check_items(s.get("options"), w + ".options", errors, need_line=False, min_n=2, max_n=4)
        a = s.get("answer")
        if not isinstance(a, int) or isinstance(a, bool) or not (0 <= a < len(s.get("options") or [])):
            errors.append(f"{w}: 'answer' doğru seçeneğin sırası olmalı (0'dan başlar)")
    elif t == "chant":
        if get_visual(s) is not None:
            _check_visual(get_visual(s), w, errors)
        items = s.get("items")
        if not isinstance(items, list) or not (1 <= len(items) <= 8):
            errors.append(f"{w}: 1-8 satır olmalı")
            return
        for j, it in enumerate(items):
            if not isinstance(it, dict):
                errors.append(f"{w}.items[{j}]: {{\"line\": ...}} nesnesi olmalı")
                continue
            _check_text(it.get("line"), f"{w}.items[{j}].line", errors, 60)
            if get_visual(it) is not None:
                _check_visual(get_visual(it), f"{w}.items[{j}]", errors)
    elif t == "story":
        panels = s.get("panels")
        if not isinstance(panels, list) or not (1 <= len(panels) <= 6):
            errors.append(f"{w}: 1-6 sayfa olmalı")
            return
        for j, p in enumerate(panels):
            if not isinstance(p, dict):
                errors.append(f"{w}.panels[{j}]: nesne olmalı")
                continue
            vs = get_visual_list(p)
            if not (1 <= len(vs) <= 3):
                errors.append(f"{w}.panels[{j}]: 1-3 görsel olmalı")
            for k, v in enumerate(vs):
                _check_visual(v, f"{w}.panels[{j}].emoji[{k}]", errors)
            _check_lines(p.get("lines"), f"{w}.panels[{j}].lines", errors, required=True, max_n=4)
            if p.get("caption"):
                _check_text(p["caption"], f"{w}.panels[{j}].caption", errors, 50)
    elif t == "math":
        _check_visual(get_visual(s), w, errors)
        a, b, op = s.get("a"), s.get("b"), s.get("op", "+")
        if op not in ("+", "-"):
            errors.append(f"{w}: 'op' '+' veya '-' olmalı")
        if not all(isinstance(x, int) and not isinstance(x, bool) for x in (a, b)) or not (0 <= a <= 10 and 0 <= b <= 10):
            errors.append(f"{w}: 'a' ve 'b' 0-10 arası tam sayı olmalı")
        elif op == "-" and b > a:
            errors.append(f"{w}: çıkarmada sonuç eksi olamaz")
    elif t == "action":
        _check_visual(get_visual(s), w, errors)
        _check_text(s.get("text"), w + ".text", errors, 18)
        sec = s.get("seconds", 5)
        if not isinstance(sec, (int, float)) or isinstance(sec, bool) or not (2 <= sec <= 12):
            errors.append(f"{w}: 'seconds' 2-12 arası olmalı")
    elif t == "outro":
        nxt = s.get("next")
        if nxt is not None:
            if not isinstance(nxt, dict):
                errors.append(f"{w}: 'next' {{'text': ..., 'emoji': ...}} biçiminde olmalı")
            else:
                _check_text(nxt.get("text"), f"{w}.next.text", errors, 30)
                if get_visual(nxt) is not None:
                    _check_visual(get_visual(nxt), f"{w}.next", errors)


def _nfc(x):
    if isinstance(x, str):
        return unicodedata.normalize("NFC", x)
    if isinstance(x, list):
        return [_nfc(v) for v in x]
    if isinstance(x, dict):
        return {k: _nfc(v) for k, v in x.items()}
    return x


def load_episode(path) -> dict:
    path = Path(path)
    try:
        ep = _nfc(json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as e:
        raise EpisodeError(f"{path.name}: JSON hatası: {e}") from e
    try:
        errs = validate(ep)
    except Exception as e:
        raise EpisodeError(f"{path.name}: beklenmeyen alan türü: {e}") from e
    if errs:
        raise EpisodeError(f"{path.name}:\n  - " + "\n  - ".join(errs))
    return ep
