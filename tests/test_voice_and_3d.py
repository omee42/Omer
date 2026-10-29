import io
import json
import wave

import pytest

from pofi import tts as tts_mod
from pofi.config import load_config


def _wav_bytes(seconds=0.5, sr=48000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x00\x10" * int(seconds * sr))
    return buf.getvalue()


class _Resp:
    def __init__(self, data):
        self.data = data

    def read(self):
        return self.data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_azure_needs_secrets(monkeypatch):
    monkeypatch.delenv("AZURE_SPEECH_KEY", raising=False)
    monkeypatch.delenv("AZURE_SPEECH_REGION", raising=False)
    with pytest.raises(RuntimeError):
        tts_mod.AzureEngine("tr-TR-EmelNeural")


def test_azure_request(monkeypatch, tmp_path):
    monkeypatch.setenv("AZURE_SPEECH_KEY", "abc")
    monkeypatch.setenv("AZURE_SPEECH_REGION", "West Europe")
    seen = {}

    def fake_urlopen(req, timeout=0):
        seen["url"] = req.full_url
        seen["headers"] = dict(req.header_items())
        seen["body"] = req.data.decode("utf-8")
        return _Resp(_wav_bytes())

    monkeypatch.setattr(tts_mod.urllib.request, "urlopen", fake_urlopen)
    eng = tts_mod.AzureEngine("tr-TR-EmelNeural", "-8%", "+3%")
    out = tmp_path / "a.wav"
    eng.synth_file("Elma & armut <tatlı>", out)
    assert seen["url"] == "https://westeurope.tts.speech.microsoft.com/cognitiveservices/v1"
    assert seen["headers"]["Ocp-apim-subscription-key"] == "abc"
    assert seen["headers"]["X-microsoft-outputformat"] == "riff-48khz-16bit-mono-pcm"
    assert "Elma &amp; armut &lt;tatlı&gt;" in seen["body"]
    assert "rate='-8%'" in seen["body"] and "pitch='+3%'" in seen["body"]
    assert "tr-TR-EmelNeural" in seen["body"]
    assert out.stat().st_size > 1000


def test_azure_bad_key_is_clear(monkeypatch, tmp_path):
    monkeypatch.setenv("AZURE_SPEECH_KEY", "abc")
    monkeypatch.setenv("AZURE_SPEECH_REGION", "westeurope")

    def fake_urlopen(req, timeout=0):
        raise tts_mod.urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr(tts_mod.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="anahtar"):
        tts_mod.AzureEngine("tr-TR-EmelNeural").synth_file("Merhaba", tmp_path / "x.wav")


def test_plan_has_every_picture():
    pytest.importorskip("numpy")
    from pofi import render, render3d
    from pofi.schema import load_episode

    class FakeTTS:
        def say(self, text):
            import numpy as np
            return np.zeros(int(0.05 * len(text) * 48000), dtype=np.float32)

    cfg = load_config()
    ep = load_episode("bolumler/2026-10-29-cumhuriyet-bayrami.json")
    tl = render.Timeline(cfg, ep, 1920, 1080, 30, FakeTTS())
    plan = render3d.build_plan(tl, [0.0] * tl.n_frames)
    json.dumps(plan, ensure_ascii=False)
    assert plan["frames"] == tl.n_frames and len(plan["mouth"]) == tl.n_frames
    assert len(plan["scenes"]) == len(ep["scenes"])
    seen = set()
    render3d._emoji_in([s["spec"] for s in plan["scenes"]], seen)
    assert seen and seen <= set(plan["emoji"])
    quiz = next(s for s in plan["scenes"] if s["type"] == "quiz")
    assert quiz["spec"]["mood"] == ep.get("mood", "happy")
    assert all(o["visual"] for o in quiz["spec"]["options"])
    assert "{maskot}" not in json.dumps(plan, ensure_ascii=False)
