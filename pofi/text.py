"""Turkish text helpers: casing, number words and TTS normalisation."""
import re
import unicodedata

_UPPER_MAP = str.maketrans({"i": "İ", "ı": "I"})
_LOWER_MAP = str.maketrans({"I": "ı", "İ": "i"})

TR_ALPHABET = list("ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ")


def tr_upper(s: str) -> str:
    return unicodedata.normalize("NFC", s).translate(_UPPER_MAP).upper()


def tr_lower(s: str) -> str:
    return unicodedata.normalize("NFC", s).translate(_LOWER_MAP).lower()


def tr_capitalize(s: str) -> str:
    return tr_upper(s[:1]) + s[1:] if s else s


_ONES = ["", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
_TENS = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"]


def _below_thousand(n: int) -> list:
    words = []
    h, rest = divmod(n, 100)
    if h:
        if h > 1:
            words.append(_ONES[h])
        words.append("yüz")
    t, o = divmod(rest, 10)
    if t:
        words.append(_TENS[t])
    if o:
        words.append(_ONES[o])
    return words


def number_to_words(n: int) -> str:
    """Spell an integer in Turkish (e.g. 1923 -> 'bin dokuz yüz yirmi üç')."""
    if n == 0:
        return "sıfır"
    if n < 0:
        return "eksi " + number_to_words(-n)
    words = []
    for value, name in ((10**12, "trilyon"), (10**9, "milyar"), (10**6, "milyon"), (1000, "bin")):
        q, n = divmod(n, value)
        if q:
            if not (value == 1000 and q == 1):
                words.extend(_below_thousand(q) if q < 1000 else [number_to_words(q)])
            words.append(name)
    words.extend(_below_thousand(n))
    return " ".join(words)


_VOWELS = "aeıioöuü"


def ordinal_words(n: int) -> str:
    """Turkish ordinal: 1 -> 'birinci', 3 -> 'üçüncü', 6 -> 'altıncı'."""
    w = number_to_words(n)
    if w.endswith("dört"):  # dört -> dördüncü (t softens before a vowel)
        return w[:-1] + "düncü"
    last_vowel = next((c for c in reversed(w) if c in _VOWELS), "e")
    v = {"a": "ı", "ı": "ı", "e": "i", "i": "i", "o": "u", "u": "u", "ö": "ü", "ü": "ü"}[last_vowel]
    if w[-1] in _VOWELS:
        return w + "nc" + v
    return w + v + "nc" + v


_NOT_EMOJI = set("°℃℉№")


def is_emoji_char(ch: str) -> bool:
    if ch in _NOT_EMOJI:
        return False
    cp = ord(ch)
    if cp in (0x200D, 0xFE0F, 0xFE0E, 0x20E3):
        return True
    if 0x1F000 <= cp <= 0x1FAFF or 0x2600 <= cp <= 0x27BF or 0x2B00 <= cp <= 0x2BFF:
        return True
    if 0xE0020 <= cp <= 0xE007F:
        return True
    return unicodedata.category(ch) == "So"


def strip_emoji(s: str) -> str:
    return re.sub(r"\s{2,}", " ", "".join(c for c in s if not is_emoji_char(c))).strip()


def _num_suffix(n: int, suffix: str) -> str:
    w = number_to_words(n)
    suffix = tr_lower(suffix)
    if w.endswith("dört") and suffix[:1] in _VOWELS:  # 4'e -> dörde, 4'ü -> dördü
        w = w[:-1] + "d"
    return w + suffix


def _decimal(m):
    whole, frac = m.group(1), m.group(2)
    if frac == "5":
        return f"{whole} buçuk"
    zeros = len(frac) - len(frac.lstrip("0"))
    return f"{whole} virgül " + "sıfır " * zeros + (frac.lstrip("0") or "")


_MATH = [
    (re.compile(r"(\d)\s*\+\s*(\d)"), r"\1 artı \2"),
    (re.compile(r"(\d)\s*[-−–]\s*(\d)"), r"\1 eksi \2"),
    (re.compile(r"(\d)\s*[x×*]\s*(\d)"), r"\1 çarpı \2"),
    (re.compile(r"(\d)\s*[÷/]\s*(\d)"), r"\1 bölü \2"),
    (re.compile(r"(\d)\s*=\s*(\d)"), r"\1 eşittir \2"),
    (re.compile(r"(\d)\s*=\s*\?"), r"\1 eşittir kaç"),
]
_SUFFIX = r"['’]([^\W\d_]+)"


def normalize_for_tts(text: str, lexicon: dict = None) -> str:
    """Make narration text safe and natural for the speech engine (numbers, symbols, emoji)."""
    s = unicodedata.normalize("NFC", text)
    s = s.replace("½", " yarım ").replace("¼", " çeyrek ")
    s = re.sub(r"\s*°(?:\s*[CF](?![^\W\d_]))?", " derece", s)       # 100 °C -> 100 derece
    s = strip_emoji(s)
    s = re.sub(r"(?<![\d.,])(\d{1,3}(?:\.\d{3})+)(?![\d,]|\.\d)", lambda m: m.group(1).replace(".", ""), s)
    s = re.sub(r"(?<![\d,.])(\d+),(\d+)(?![\d,])", _decimal, s)            # 3,5 -> 3 buçuk
    s = re.sub(r"\b(\d{1,2}):(\d{2})\b", lambda m: m.group(1) if m.group(2) == "00"
               else f"{m.group(1)} {int(m.group(2))}", s)                # 10:30 -> 10 30
    s = re.sub(r"\b(\d+)\.(?=\s+[a-zçğıöşü])", lambda m: ordinal_words(int(m.group(1))), s)  # 3. sınıf
    if "=" in s or "−" in s:  # an equation: spell out the operators
        s = re.sub(r"(\d|\w)\s+[-−–]\s+(?=\d)", r"\1 eksi ", s)
        s = re.sub(r"(\d)[-−–](\d)", r"\1 eksi \2", s)
        s = re.sub(r"\s\+\s", " artı ", s)
        s = re.sub(r"\s=\s", " eşittir ", s)
    else:  # 3-7 yaş -> 3 ile 7 yaş
        s = re.sub(r"(?<![\d-])(\d+)[-–](\d+)(?![\d-])", r"\1 ile \2", s)
    for pat, rep in _MATH:
        while pat.search(s):
            s = pat.sub(rep, s)
    s = re.sub(r"(?<![\w])[-−](?=\d)", "eksi ", s)                          # -5 derece
    s = re.sub(r"%\s*(\d+)", lambda m: "yüzde " + m.group(1), s)
    s = re.sub(r"(\d+)\s*%", lambda m: "yüzde " + m.group(1), s)
    s = re.sub(r"(\d+)" + _SUFFIX, lambda m: _num_suffix(int(m.group(1)), m.group(2)), s)
    s = re.sub(r"\d+", lambda m: number_to_words(int(m.group(0))), s)
    s = s.replace("…", "...").replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    # "Buna kızdım." dedi -> Buna kızdım, dedi (no sentence break inside the quote)
    s = re.sub(r"[.!?]+[\"»]\s+(?=[a-zçğıöşü])", ", ", s)
    s = re.sub(r"[\"«»]", "", s)
    s = re.sub(r"\s*[-–—]\s+", ", ", s)
    for word, say in (lexicon or {}).items():
        s = re.sub(rf"\b{re.escape(word)}\b", lambda m, say=say: say, s)
    return re.sub(r"\s{2,}", " ", s).strip(" ,")


def wrap_words(words, max_chars):
    lines, cur = [], ""
    for w in words:
        cand = (cur + " " + w).strip()
        if len(cand) > max_chars and cur:
            lines.append(cur)
            cur = w
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return lines
