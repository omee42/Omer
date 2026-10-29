"""Loads config.yaml with defaults so a partial file still works."""
import copy
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path(os.environ.get("POFI_CACHE", ROOT / ".cache"))

DEFAULTS = {
    "kanal": {"ad": "Pofi ile Öğreniyorum", "maskot": "Pofi", "yas_grubu": "3-7 yaş", "web": ""},
    "ses": {"motor": "piper:fettah", "yedek": "piper:fettah", "hiz": 1.12, "perde": 0,
            "satir_arasi": 0.45, "deneme": 3, "kalite_kontrol": True, "sozluk": {}},
    "video": {"genislik": 1920, "yukseklik": 1080, "fps": 30, "altyazi_goster": True,
              "muzik_seviyesi": 0.16, "efekt_seviyesi": 0.55, "hedef_loudness": -14},
    "shorts": {"aktif": True, "genislik": 1080, "yukseklik": 1920},
    "youtube": {"otomatik_yukle": False, "gizlilik": "private", "yayin_saati": "17:00", "kategori": 27,
                "cocuklar_icin": True, "etiketler": []},
}


def _merge(base, extra):
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path=None):
    path = Path(path or ROOT / "config.yaml")
    data = {}
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return _merge(DEFAULTS, data)
