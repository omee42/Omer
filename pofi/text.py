"""Turkish text helpers: casing, number words and TTS normalisation."""
import re
import unicodedata

_UPPER_MAP = str.maketrans({"i": "İ", "ı": "I"})
_LOWER_MAP = str.maketrans({"I": "ı", "İ": "i"})

TR_ALPHABET = list("ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ")


def tr_upper(s: str) -> str:
    return s.translate(_UPPER_MAP).upper()


def tr_lower(s: str) -> str:
    return s.translate(_LOWER_MAP).lower()


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
    for value, name in ((10**9, "milyar"), (10**6, "milyon"), (1000, "bin")):
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
    last_vowel = next((c for c in reversed(w) if c in _VOWELS), "e")
    v = {"a": "ı", "ı": "ı", "e": "i", "i": "i", "o": "u", "u": "u", "ö": "ü", "ü": "ü"}[last_vowel]
    if w[-1] in _VOWELS:
        return w + "nc" + v
    return w + v + "nc" + v


def is_emoji_char(ch: str) -> bool:
    cp = ord(ch)
    if cp in (0x200D, 0xFE0F, 0x20E3):
        return True
    if 0x1F000 <= cp <= 0x1FAFF or 0x2600 <= cp <= 0x27BF or 0x2B00 <= cp <= 0x2BFF:
        return True
    if 0xE0020 <= cp <= 0xE007F:
        return True
    return unicodedata.category(ch) == "So"


def strip_emoji(s: str) -> str:
    return re.sub(r"\s{2,}", " ", "".join(c for c in s if not is_emoji_char(c))).strip()


_MATH = [
    (re.compile(r"(\d)\s*\+\s*(\d)"), r"\1 artı \2"),
    (re.compile(r"(\d)\s*[-−–]\s*(\d)"), r"\1 eksi \2"),
    (re.compile(r"(\d)\s*[x×*]\s*(\d)"), r"\1 çarpı \2"),
    (re.compile(r"(\d)\s*[÷/]\s*(\d)"), r"\1 bölü \2"),
    (re.compile(r"(\d)\s*=\s*(\d)"), r"\1 eşittir \2"),
    (re.compile(r"(\d)\s*=\s*\?"), r"\1 eşittir kaç"),
]


def normalize_for_tts(text: str, lexicon: dict = None) -> str:
    """Make narration text safe and natural for the speech engine."""
    s = strip_emoji(text)
    for pat, rep in _MATH:
        while pat.search(s):
            s = pat.sub(rep, s)
    s = re.sub(r"%\s*(\d+)", lambda m: "yüzde " + m.group(1), s)
    # 5'e -> beşe, 10'dan -> ondan (apostrophe suffixes are glued to the word)
    s = re.sub(r"(\d+)['’]([a-zçğıöşü]+)", lambda m: number_to_words(int(m.group(1))) + m.group(2), s)
    s = re.sub(r"\d+", lambda m: number_to_words(int(m.group(0))), s)
    s = s.replace("…", "...").replace("“", '"').replace("”", '"').replace("’", "'")
    s = re.sub(r"[\"«»]", "", s)
    s = re.sub(r"\s*[-–—]\s+", ", ", s)
    for word, say in (lexicon or {}).items():
        s = re.sub(rf"\b{re.escape(word)}\b", say, s)
    return re.sub(r"\s{2,}", " ", s).strip()


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
