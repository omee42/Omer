"""Text-to-speech with on-disk caching. Every clip is returned as mono float32 at audio.SR."""
import asyncio
import hashlib
import json
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

from .audio import SR, trim_silence
from .config import CACHE
from .text import normalize_for_tts

TTS_CACHE = CACHE / "tts"


def _decode_to_array(path: Path, pitch: float = 0.0) -> np.ndarray:
    """Decode any audio file to mono float32 @ SR via ffmpeg (optional pitch shift)."""
    filters = []
    if pitch:
        filters.append(f"rubberband=pitch={2 ** (pitch / 12):.5f}:formant=preserved")
    cmd = ["ffmpeg", "-v", "error", "-i", str(path)]
    if filters:
        cmd += ["-af", ",".join(filters)]
    cmd += ["-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    out = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


class PiperEngine:
    def __init__(self, voice: str, speed: float):
        from piper import PiperVoice, SynthesisConfig
        from .assets import ensure_voice

        onnx = ensure_voice(voice)
        self.voice = PiperVoice.load(str(onnx), config_path=str(onnx.with_suffix(".onnx.json")))
        self.syn = SynthesisConfig(length_scale=speed)
        self.id = f"piper:{voice}:{speed}"

    def synth_file(self, text: str, out: Path):
        with wave.open(str(out), "wb") as wf:
            self.voice.synthesize_wav(text, wf, syn_config=self.syn)


class EdgeEngine:
    """Microsoft neural voices through the edge-tts package (needs internet)."""

    def __init__(self, voice: str, speed: float):
        import edge_tts  # noqa: F401  (fail early when the package is missing)

        self.voice = voice
        pct = int(round((1.0 / speed - 1.0) * 100))
        self.rate = f"{pct:+d}%"
        self.id = f"edge:{voice}:{self.rate}"

    def synth_file(self, text: str, out: Path):
        import edge_tts

        async def run():
            await edge_tts.Communicate(text, self.voice, rate=self.rate).save(str(out))

        asyncio.run(run())
        if out.stat().st_size < 1000:
            raise RuntimeError("edge-tts boş ses döndürdü")


def _make_engine(spec: str, speed: float):
    kind, _, name = spec.partition(":")
    if kind == "piper":
        return PiperEngine(name, speed)
    if kind == "edge":
        return EdgeEngine(name, speed)
    raise ValueError(f"Bilinmeyen ses motoru: {spec}")


class TTS:
    def __init__(self, cfg: dict):
        s = cfg["ses"]
        self.lexicon = s.get("sozluk") or {}
        self.pitch = float(s.get("perde") or 0)
        speed = float(s.get("hiz") or 1.0)
        try:
            self.engine = _make_engine(s["motor"], speed)
            if isinstance(self.engine, EdgeEngine):
                self._probe()
        except Exception as e:  # network blocked, package missing, ...
            print(f"[tts] '{s['motor']}' kullanılamadı ({e}); yedek ses: {s['yedek']}")
            self.engine = _make_engine(s["yedek"], speed)
        TTS_CACHE.mkdir(parents=True, exist_ok=True)
        self.takes = int(s.get("deneme", 3)) if isinstance(self.engine, PiperEngine) else 1
        self.judge = None
        if self.takes > 1 and s.get("kalite_kontrol", True):
            try:
                from .asr import Judge
                self.judge = Judge()
            except Exception as e:
                print(f"[tts] kalite kontrol (Whisper) kullanılamadı: {e}; her cümle tek seferde üretilecek")
        if self.judge is None:
            self.takes = 1
        self.report = {}

    def _probe(self):
        with tempfile.TemporaryDirectory() as td:
            self.engine.synth_file("Merhaba.", Path(td) / "p.mp3")

    def _synth(self, spoken: str) -> np.ndarray:
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td) / ("clip.mp3" if isinstance(self.engine, EdgeEngine) else "clip.wav")
            self.engine.synth_file(spoken, raw)
            audio = _decode_to_array(raw, self.pitch)
        return trim_silence(audio)

    def say(self, text: str) -> np.ndarray:
        spoken = normalize_for_tts(text, self.lexicon)
        if not spoken:
            return np.zeros(int(0.3 * SR), dtype=np.float32)
        key = hashlib.sha1(f"{self.engine.id}|{self.pitch}|{self.takes}|{spoken}".encode()).hexdigest()[:20]
        cached = TTS_CACHE / f"{key}.f32"
        meta = TTS_CACHE / f"{key}.json"
        if cached.exists():
            if meta.exists():
                self.report[spoken] = json.loads(meta.read_text(encoding="utf-8"))
            return np.fromfile(cached, dtype=np.float32)
        best, best_score, best_hyp = None, -1.0, ""
        judged = self.judge is not None and len(spoken.split()) > 2  # Whisper is unreliable on 1-2 words
        for _ in range(self.takes if judged else 1):
            audio = self._synth(spoken)
            if not judged:
                best = audio
                break
            score, hyp = self.judge.score(audio, spoken)
            if score > best_score:
                best, best_score, best_hyp = audio, score, hyp
            if score >= 0.96:
                break
        best.astype(np.float32).tofile(cached)
        if judged:
            info = {"score": round(best_score, 3), "heard": best_hyp}
            meta.write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
            self.report[spoken] = info
        return best
