"""Text-to-speech with on-disk caching. Every clip is returned as mono float32 at audio.SR."""
import asyncio
import hashlib
import http.client
import json
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path
from xml.sax.saxutils import escape

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


class AzureEngine:
    """Azure AI Speech neural voices through the official REST API.

    Needs the AZURE_SPEECH_KEY and AZURE_SPEECH_REGION environment variables (GitHub secrets).
    The free F0 tier covers 500 000 characters a month; one episode uses about 3 000. F0 also allows only
    20 requests a minute, so requests are spaced `gap` seconds apart (an episode has ~60 lines: ~3 minutes).
    """

    def __init__(self, voice: str, rate: str = "-8%", pitch: str = "+3%", gap: float = 3.1):
        self.key = os.environ.get("AZURE_SPEECH_KEY", "").strip()
        region = os.environ.get("AZURE_SPEECH_REGION", "").strip().lower().replace(" ", "")
        if not self.key or not region:
            raise RuntimeError("AZURE_SPEECH_KEY ve AZURE_SPEECH_REGION tanımlı değil")
        self.url = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"
        self.voice, self.rate, self.pitch = voice, rate, pitch
        self.gap = max(0.0, float(gap))
        self._last = 0.0
        self.id = f"azure:{voice}:{rate}:{pitch}"

    def ssml(self, text: str) -> str:
        return ("<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='tr-TR'>"
                f"<voice name='{self.voice}'><prosody rate='{self.rate}' pitch='{self.pitch}'>"
                f"{escape(text)}</prosody></voice></speak>")

    def synth_file(self, text: str, out: Path):
        req = urllib.request.Request(self.url, data=self.ssml(text).encode("utf-8"), method="POST", headers={
            "Ocp-Apim-Subscription-Key": self.key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "riff-48khz-16bit-mono-pcm",
            "User-Agent": "pofi-video",
        })
        # 429 (rate limit) is waited out up to 8 times (Retry-After, at least 15 s each); other transient
        # errors get 5 tries. Errors while reading the response (dropped connection, TLS reset) count too.
        tries, limited = 0, 0
        while True:
            wait = self._last + self.gap - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
                break
            except urllib.error.HTTPError as e:
                if e.code in (401, 403):
                    raise RuntimeError("Azure anahtarı ya da bölge (region) hatalı") from e
                if e.code == 429 and limited < 8:
                    limited += 1
                    try:
                        delay = float(e.headers.get("Retry-After") or 0)
                    except (TypeError, ValueError):
                        delay = 0
                    time.sleep(min(60.0, max(delay, 15.0)))
                    continue
                tries += 1
                if e.code not in (408, 500, 502, 503, 504) or tries >= 5:
                    raise
            except (urllib.error.URLError, TimeoutError, http.client.HTTPException, OSError):
                tries += 1
                if tries >= 5:
                    raise
            time.sleep(2 ** tries)
        if len(data) < 1000:
            raise RuntimeError("Azure boş ses döndürdü")
        out.write_bytes(data)


class VoiceSwitched(RuntimeError):
    """The main voice failed in the middle of an episode; the caller re-runs the episode with the backup."""


def _make_engine(spec: str, cfg_ses: dict):
    kind, _, name = spec.partition(":")
    speed = float(cfg_ses.get("hiz") or 1.0)
    if kind == "piper":
        return PiperEngine(name, speed)
    if kind == "edge":
        return EdgeEngine(name, speed)
    if kind == "azure":
        return AzureEngine(name or "tr-TR-EmelNeural", str(cfg_ses.get("azure_hiz", "-8%")),
                           str(cfg_ses.get("azure_perde", "+3%")), float(cfg_ses.get("azure_aralik", 3.1)))
    raise ValueError(f"Bilinmeyen ses motoru: {spec}")


class TTS:
    def __init__(self, cfg: dict):
        s = cfg["ses"]
        self.lexicon = s.get("sozluk") or {}
        self.cfg_ses = s
        self.on_backup = False
        try:
            self.engine = _make_engine(s["motor"], s)
            if not isinstance(self.engine, PiperEngine):
                self._probe()
        except Exception as e:  # no key, network blocked, package missing, ...
            print(f"[tts] '{s['motor']}' kullanılamadı ({e}); yedek ses: {s['yedek']}")
            self.engine = _make_engine(s["yedek"], s)
            self.on_backup = True
        # rubberband pitch shifting is only for the offline voices; Azure sets its pitch in SSML
        self.pitch = float(s.get("perde") or 0) if isinstance(self.engine, PiperEngine) else 0.0
        print(f"[tts] ses: {self.engine.id}")
        TTS_CACHE.mkdir(parents=True, exist_ok=True)
        # Piper sometimes swallows a syllable, so it gets several takes; cloud voices are steady and
        # get one take that is still scored for the voice report.
        self.takes = int(s.get("deneme", 3)) if isinstance(self.engine, PiperEngine) else 1
        self.judge = None
        if s.get("kalite_kontrol", True):
            try:
                from .asr import Judge
                self.judge = Judge()
            except Exception as e:
                print(f"[tts] kalite kontrol (Whisper) kullanılamadı: {e}; her cümle tek seferde üretilecek")
        if self.judge is None:
            self.takes = 1
        self.report = {}

    def use_backup(self):
        self.engine = _make_engine(self.cfg_ses["yedek"], self.cfg_ses)
        self.on_backup = True
        self.pitch = float(self.cfg_ses.get("perde") or 0) if isinstance(self.engine, PiperEngine) else 0.0
        self.takes = int(self.cfg_ses.get("deneme", 3)) if isinstance(self.engine, PiperEngine) and self.judge else 1
        self.report = {}

    def _probe(self):
        with tempfile.TemporaryDirectory() as td:
            self.engine.synth_file("Merhaba.", Path(td) / "p.mp3")

    def _synth(self, spoken: str) -> np.ndarray:
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td) / ("clip.mp3" if isinstance(self.engine, EdgeEngine) else "clip.wav")
            try:
                self.engine.synth_file(spoken, raw)
            except Exception as e:
                if self.on_backup or isinstance(self.engine, PiperEngine):
                    raise
                print(f"[tts] '{self.engine.id}' bölüm ortasında yanıt vermedi ({e}); bölüm yedek sesle baştan seslendirilecek")
                self.use_backup()
                raise VoiceSwitched(str(e)) from e
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
            audio = np.fromfile(cached, dtype=np.float32)
            if meta.exists():
                self.report[spoken] = json.loads(meta.read_text(encoding="utf-8"))
            elif self.judge is not None and len(spoken.split()) > 2:
                # made while Whisper was unavailable (one-take cloud voices share the cache key): score it now
                score, hyp = self.judge.score(audio, spoken)
                info = {"score": round(score, 3), "heard": hyp}
                meta.write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
                self.report[spoken] = info
            return audio
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
