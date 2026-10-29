"""Speech-recognition judge used to pick the clearest of several narration takes.

Piper voices are slightly random on every run, so we synthesise a few takes per line and
keep the one Whisper understands best. Model: Whisper small (int8) from the sherpa-onnx releases.
"""
import difflib
import io
import re
import shutil
import tarfile
import urllib.request

import numpy as np

from .config import CACHE
from .text import number_to_words, tr_lower

ASR_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-small.tar.bz2"
ASR_DIR = CACHE / "asr" / "whisper-small"
FILES = ("small-encoder.int8.onnx", "small-decoder.int8.onnx", "small-tokens.txt")


def ensure_asr():
    if all((ASR_DIR / f).exists() for f in FILES):
        return ASR_DIR
    print("[asr] Whisper modeli indiriliyor (~640 MB, bir kez)...")
    tmp = ASR_DIR.with_suffix(".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    req = urllib.request.Request(ASR_URL, headers={"User-Agent": "pofi-video/1.0"})
    with urllib.request.urlopen(req, timeout=900) as r:
        with tarfile.open(fileobj=r, mode="r|bz2") as tf:
            for m in tf:
                name = m.name.rsplit("/", 1)[-1]
                if m.isfile() and name in FILES:
                    (tmp / name).write_bytes(tf.extractfile(m).read())
    shutil.rmtree(ASR_DIR, ignore_errors=True)
    tmp.rename(ASR_DIR)
    return ASR_DIR


def _norm(s: str) -> list:
    s = re.sub(r"\d+", lambda m: number_to_words(int(m.group(0))), s)
    s = tr_lower(s).replace("â", "a").replace("î", "i").replace("û", "u")
    s = re.sub(r"[^a-zçğıöşü ]", " ", s)
    return s.split()


def similarity(ref: str, hyp: str) -> float:
    return difflib.SequenceMatcher(None, _norm(ref), _norm(hyp)).ratio()


class Judge:
    def __init__(self):
        import sherpa_onnx

        d = ensure_asr()
        self.rec = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=str(d / FILES[0]), decoder=str(d / FILES[1]), tokens=str(d / FILES[2]),
            language="tr", task="transcribe", num_threads=4)

    def transcribe(self, audio48k: np.ndarray) -> str:
        from scipy.signal import resample_poly

        a16 = resample_poly(audio48k, 1, 3).astype(np.float32)
        a16 = np.concatenate([np.zeros(4000, np.float32), a16, np.zeros(8000, np.float32)])
        s = self.rec.create_stream()
        s.accept_waveform(16000, a16)
        self.rec.decode_stream(s)
        return s.result.text.strip()

    def score(self, audio48k: np.ndarray, text: str):
        hyp = self.transcribe(audio48k)
        return similarity(text, hyp), hyp
