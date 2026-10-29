"""Episode file format (JSON) and its validation.

An episode is a list of scenes; each scene type is documented in docs/SENARYO_REHBERI.md.
"""
import json
import re
from pathlib import Path

from .gfx import SHAPE_ALIASES, SHAPES

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
    if "emoji" in obj and isinstance(obj["emoji"], str):
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
    else:
        errors.append(f"{where}: görsel anlaşılamadı: {v}")


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
        if need_label and not str(it.get("label", "")).strip():
            errors.append(f"{w}: 'label' gerekli")
        if len(str(it.get("label", ""))) > 22:
            errors.append(f"{w}: 'label' en fazla 22 karakter olmalı")
        if need_line and not str(it.get("line", "")).strip():
            errors.append(f"{w}: 'line' (söylenecek cümle) gerekli")
        if len(str(it.get("line", ""))) > MAX_LINE:
            errors.append(f"{w}: 'line' çok uzun")


def validate(ep: dict) -> list:
    errors = []
    for key in ("id", "date", "title", "category", "scenes", "youtube"):
        if key not in ep:
            errors.append(f"'{key}' alanı eksik")
    if errors:
        return errors
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", ep["date"]):
        errors.append("'date' YYYY-AA-GG biçiminde olmalı")
    if len(ep["title"]) > 40:
        errors.append("'title' en fazla 40 karakter olmalı (ekranda büyük yazılıyor)")
    if ep.get("mood", "happy") not in MOODS:
        errors.append(f"'mood' şunlardan biri olmalı: {sorted(MOODS)}")
    yt = ep["youtube"]
    if not isinstance(yt, dict) or not yt.get("title") or not yt.get("description"):
        errors.append("youtube.title ve youtube.description gerekli")
    else:
        if len(yt["title"]) > 100:
            errors.append("youtube.title en fazla 100 karakter olmalı")
        if re.search(r"[<>]", yt["title"] + yt["description"]):
            errors.append("YouTube başlık/açıklamasında < veya > kullanılamaz")
        th = yt.get("thumbnail", {})
        if th and len(th.get("text", "")) > 18:
            errors.append("youtube.thumbnail.text en fazla 18 karakter olmalı")
        for i, v in enumerate(th.get("emoji", []) if th else []):
            _check_visual(v, f"youtube.thumbnail.emoji[{i}]", errors)
    scenes = ep["scenes"]
    if not isinstance(scenes, list) or len(scenes) < 3:
        errors.append("en az 3 sahne gerekli")
        return errors
    if scenes[0].get("type") != "intro":
        errors.append("ilk sahne 'intro' olmalı")
    if scenes[-1].get("type") != "outro":
        errors.append("son sahne 'outro' olmalı")
    for i, s in enumerate(scenes):
        w = f"scenes[{i}]({s.get('type')})"
        t = s.get("type")
        if t not in SCENE_TYPES:
            errors.append(f"{w}: bilinmeyen sahne türü. Seçenekler: {', '.join(sorted(SCENE_TYPES))}")
            continue
        _check_lines(s.get("lines"), w + ".lines", errors,
                     required=t in ("intro", "talk", "item", "fact", "outro", "action"))
        _check_lines(s.get("after"), w + ".after", errors)
        if t == "intro":
            if not s.get("title"):
                errors.append(f"{w}: 'title' gerekli")
            for j, v in enumerate(get_visual_list(s)[:6]):
                _check_visual(v, f"{w}.emoji[{j}]", errors)
        elif t == "talk":
            for j, v in enumerate(get_visual_list(s)):
                _check_visual(v, f"{w}.emoji[{j}]", errors)
            if len(get_visual_list(s)) > 3:
                errors.append(f"{w}: en fazla 3 görsel")
            if len(s.get("text", "")) > 60:
                errors.append(f"{w}: 'text' en fazla 60 karakter")
        elif t == "item":
            _check_visual(get_visual(s), w, errors)
            if not s.get("word") or len(s["word"]) > 16:
                errors.append(f"{w}: 'word' gerekli ve en fazla 16 karakter")
            if "color" in s and not HEX.match(str(s["color"])):
                errors.append(f"{w}: 'color' #RRGGBB olmalı")
        elif t in ("gallery", "recap"):
            _check_items(s.get("items"), w + ".items", errors, need_line=(t == "gallery"), max_n=6)
        elif t == "count":
            _check_visual(get_visual(s), w, errors)
            n = s.get("n")
            if not isinstance(n, int) or not (1 <= n <= 20):
                errors.append(f"{w}: 'n' 1-20 arasında tam sayı olmalı")
        elif t == "letter":
            letter = str(s.get("letter", ""))
            from .text import TR_ALPHABET, tr_upper
            if tr_upper(letter) not in TR_ALPHABET:
                errors.append(f"{w}: 'letter' Türk alfabesinden tek harf olmalı")
            words = s.get("words", [])
            if not (1 <= len(words) <= 4):
                errors.append(f"{w}: 1-4 kelime olmalı")
            for j, wd in enumerate(words):
                _check_visual(get_visual(wd), f"{w}.words[{j}]", errors)
                if not wd.get("word") or not wd.get("line"):
                    errors.append(f"{w}.words[{j}]: 'word' ve 'line' gerekli")
                elif tr_upper(wd["word"][:1]) != tr_upper(letter):
                    errors.append(f"{w}.words[{j}]: '{wd['word']}' '{letter}' harfiyle başlamıyor")
        elif t == "compare":
            for side in ("left", "right"):
                it = s.get(side)
                if not isinstance(it, dict):
                    errors.append(f"{w}: '{side}' gerekli")
                    continue
                _check_visual(get_visual(it), f"{w}.{side}", errors)
                if not it.get("label") or not it.get("line"):
                    errors.append(f"{w}.{side}: 'label' ve 'line' gerekli")
        elif t == "fact":
            _check_visual(get_visual(s), w, errors)
            if not s.get("text") or len(s["text"]) > 90:
                errors.append(f"{w}: 'text' gerekli ve en fazla 90 karakter")
        elif t == "quiz":
            if not s.get("question") or len(s["question"]) > 50:
                errors.append(f"{w}: 'question' gerekli ve en fazla 50 karakter")
            _check_items(s.get("options"), w + ".options", errors, need_line=False, min_n=2, max_n=4)
            a = s.get("answer")
            if not isinstance(a, int) or not (0 <= a < len(s.get("options", []))):
                errors.append(f"{w}: 'answer' doğru seçeneğin sırası olmalı (0'dan başlar)")
        elif t == "chant":
            items = s.get("items", [])
            if not (1 <= len(items) <= 8):
                errors.append(f"{w}: 1-8 satır olmalı")
            for j, it in enumerate(items):
                if not it.get("line"):
                    errors.append(f"{w}.items[{j}]: 'line' gerekli")
                elif len(it["line"]) > 60:
                    errors.append(f"{w}.items[{j}]: tekerleme satırı en fazla 60 karakter")
                if get_visual(it) is not None:
                    _check_visual(get_visual(it), f"{w}.items[{j}]", errors)
        elif t == "story":
            panels = s.get("panels", [])
            if not (1 <= len(panels) <= 6):
                errors.append(f"{w}: 1-6 sayfa olmalı")
            for j, p in enumerate(panels):
                vs = get_visual_list(p)
                if not (1 <= len(vs) <= 3):
                    errors.append(f"{w}.panels[{j}]: 1-3 görsel olmalı")
                for k, v in enumerate(vs):
                    _check_visual(v, f"{w}.panels[{j}].emoji[{k}]", errors)
                _check_lines(p.get("lines"), f"{w}.panels[{j}].lines", errors, required=True, max_n=4)
                if len(p.get("caption", "")) > 50:
                    errors.append(f"{w}.panels[{j}]: 'caption' en fazla 50 karakter")
        elif t == "math":
            _check_visual(get_visual(s), w, errors)
            a, b, op = s.get("a"), s.get("b"), s.get("op", "+")
            if op not in ("+", "-"):
                errors.append(f"{w}: 'op' '+' veya '-' olmalı")
            if not all(isinstance(x, int) for x in (a, b)) or not (0 <= a <= 10 and 0 <= b <= 10):
                errors.append(f"{w}: 'a' ve 'b' 0-10 arası tam sayı olmalı")
            elif op == "-" and b > a:
                errors.append(f"{w}: çıkarmada sonuç eksi olamaz")
        elif t == "action":
            _check_visual(get_visual(s), w, errors)
            if not s.get("text") or len(s["text"]) > 18:
                errors.append(f"{w}: 'text' gerekli ve en fazla 18 karakter (ör. 'ZIPLA!')")
            sec = s.get("seconds", 5)
            if not isinstance(sec, (int, float)) or not (2 <= sec <= 12):
                errors.append(f"{w}: 'seconds' 2-12 arası olmalı")
        elif t == "outro":
            nxt = s.get("next")
            if nxt is not None:
                if not isinstance(nxt, dict) or not nxt.get("text") or len(nxt["text"]) > 30:
                    errors.append(f"{w}: 'next' {{'text': ..., 'emoji': ...}} biçiminde, metin en fazla 30 karakter")
                elif get_visual(nxt) is not None:
                    _check_visual(get_visual(nxt), f"{w}.next", errors)
    return errors


def load_episode(path) -> dict:
    path = Path(path)
    try:
        ep = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise EpisodeError(f"{path.name}: JSON hatası: {e}") from e
    errs = validate(ep)
    if errs:
        raise EpisodeError(f"{path.name}:\n  - " + "\n  - ".join(errs))
    return ep
