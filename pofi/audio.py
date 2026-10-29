"""Procedural audio: sound effects, background music and final mix.

Everything is synthesised from scratch with numpy, so there are no licensing
questions about music or sound effects on the channel.
"""
import json
import subprocess
import zlib
import wave
from pathlib import Path

import numpy as np

SR = 48000


def trim_silence(a: np.ndarray, thresh: float = 0.004, pad: float = 0.03, tail: float = 0.12) -> np.ndarray:
    """Trim leading/trailing silence but keep soft word endings (e.g. a final 'z' or 's')."""
    idx = np.nonzero(np.abs(a) > thresh)[0]
    if len(idx) == 0:
        return a
    return a[max(0, idx[0] - int(pad * SR)): min(len(a), idx[-1] + int(tail * SR))]


def _t(dur):
    return np.arange(int(dur * SR)) / SR


def _env(n, attack=0.005, decay=0.3):
    t = np.arange(n) / SR
    a = np.clip(t / max(attack, 1e-4), 0, 1)
    return a * np.exp(-t / max(decay, 1e-4))


def _midi(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def _noise(n, rng):
    return rng.uniform(-1, 1, n)


def _lowpass(x, cutoff):
    # one-pole lowpass, vectorised enough for short effects
    from scipy.signal import lfilter
    a = np.exp(-2 * np.pi * cutoff / SR)
    return lfilter([1 - a], [1, -a], x)


def _bandpass(x, lo, hi):
    from scipy.signal import butter, sosfilt
    sos = butter(2, [lo, hi], btype="band", fs=SR, output="sos")
    return sosfilt(sos, x)


def _highpass(x, cutoff):
    from scipy.signal import butter, sosfilt
    sos = butter(2, cutoff, btype="high", fs=SR, output="sos")
    return sosfilt(sos, x)


def bell(freq, dur=1.0, decay=0.35, partials=((1, 1.0), (2.0, 0.35), (2.76, 0.25), (5.4, 0.08))):
    t = _t(dur)
    out = np.zeros_like(t)
    for ratio, amp in partials:
        out += amp * np.sin(2 * np.pi * freq * ratio * t) * np.exp(-t / (decay / ratio ** 0.5))
    return out * np.clip(t / 0.002, 0, 1)


def pluck(freq, dur=0.9, bright=1.0):
    t = _t(dur)
    out = np.zeros_like(t)
    for h in range(1, 9):
        amp = (1.0 / h) * (0.9 ** h) * bright ** (h - 1)
        out += amp * np.sin(2 * np.pi * freq * h * t + h) * np.exp(-t * (2.5 + 1.6 * h))
    return out * np.clip(t / 0.003, 0, 1)


def _sfx(name: str) -> np.ndarray:
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    if name == "pop":
        t = _t(0.12)
        f = 950 * np.exp(-t * 18) + 280
        x = np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(len(t), 0.002, 0.045)
        return x * 0.9
    if name == "ding":
        return bell(1318.5, 1.2, 0.45) * 0.55
    if name == "correct":
        a = bell(1046.5, 0.9, 0.3)
        b = bell(1568.0, 1.3, 0.45)
        out = np.zeros(int(1.5 * SR))
        out[: len(a)] += a
        o = int(0.13 * SR)
        out[o: o + len(b)] += b
        return out * 0.5
    if name == "tada":
        notes = [72, 76, 79, 84]
        out = np.zeros(int(2.0 * SR))
        for i, m in enumerate(notes):
            n = bell(_midi(m), 1.6, 0.5)
            o = int(i * 0.075 * SR)
            out[o: o + len(n)] += n[: len(out) - o] * (0.6 if i < 3 else 0.8)
        return out * 0.4
    if name == "sparkle":
        out = np.zeros(int(1.0 * SR))
        for i, m in enumerate([88, 91, 95, 100, 96]):
            n = bell(_midi(m), 0.6, 0.15)
            o = int(i * 0.05 * SR)
            out[o: o + len(n)] += n[: len(out) - o] * 0.3
        return out * 0.5
    if name == "whoosh":
        n = int(0.45 * SR)
        x = _noise(n, rng)
        env = np.sin(np.linspace(0, np.pi, n)) ** 2
        lo = _bandpass(x, 300, 1200)
        hi = _bandpass(x, 1200, 4000)
        mix = np.linspace(0, 1, n)
        return (lo * (1 - mix) + hi * mix) * env * 0.9
    if name in ("tick", "tock"):
        t = _t(0.06)
        f = 1500 if name == "tick" else 1100
        return np.sin(2 * np.pi * f * t) * _env(len(t), 0.001, 0.012) * 0.7
    if name == "boing":
        t = _t(0.5)
        f = 220 + 120 * np.sin(2 * np.pi * 9 * t) * np.exp(-t * 4) + 80 * t
        return np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(len(t), 0.003, 0.22) * 0.6
    if name == "clap":
        out = np.zeros(int(0.35 * SR))
        for o in (0.0, 0.012, 0.024):
            n = int(0.12 * SR)
            b = _bandpass(_noise(n, rng), 900, 3500) * _env(n, 0.001, 0.03)
            s = int(o * SR)
            out[s: s + n] += b
        return out * 0.7
    if name == "drum":
        t = _t(0.3)
        f = 120 * np.exp(-t * 18) + 48
        return np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(len(t), 0.002, 0.12)
    raise KeyError(name)


_SFX_CACHE = {}


def sfx(name: str) -> np.ndarray:
    if name not in _SFX_CACHE:
        _SFX_CACHE[name] = _sfx(name).astype(np.float32)
    return _SFX_CACHE[name]


# ---------------------------------------------------------------- music

PROGRESSIONS = [
    [0, 7, 9, 5],    # I V vi IV
    [0, 5, 7, 0],    # I IV V I
    [0, 9, 5, 7],    # I vi IV V
    [0, 5, 0, 7],    # I IV I V
    [9, 5, 0, 7],    # vi IV I V
]


def _chord(root_midi, degree):
    minor = degree in (2, 4, 9)
    base = root_midi + degree
    return [base, base + (3 if minor else 4), base + 7]


def make_music(duration: float, seed: int = 0, mood: str = "happy") -> np.ndarray:
    """Return stereo (n, 2) cheerful background music of the requested length."""
    rng = np.random.default_rng(seed)
    if mood == "calm":
        bpm = int(rng.integers(68, 78))
    elif mood == "energetic":
        bpm = int(rng.integers(118, 128))
    else:
        bpm = int(rng.integers(98, 110))
    root = 60 + int(rng.choice([0, 2, 5, 7, -3]))
    prog = PROGRESSIONS[int(rng.integers(len(PROGRESSIONS)))]
    beat = 60.0 / bpm
    bars = 8
    loop_len = bars * 4 * beat
    n = int(loop_len * SR)
    L = np.zeros(n)
    R = np.zeros(n)

    def add(sig, start, gain, pan=0.0):
        s = int(start * SR) % n
        sig = sig * gain
        e = min(n, s + len(sig))
        L[s:e] += sig[: e - s] * (1 - max(0, pan))
        R[s:e] += sig[: e - s] * (1 + min(0, pan))
        rest = len(sig) - (e - s)
        if rest > 0:  # wrap the tail around so the loop is seamless
            L[:rest] += sig[e - s:] * (1 - max(0, pan))
            R[:rest] += sig[e - s:] * (1 + min(0, pan))

    melody_rhythms = [
        [0, 1, 2, 3], [0, 1.5, 2, 3], [0, 0.5, 1, 2, 3], [0, 2, 2.5, 3], [0, 1, 1.5, 2.5],
    ]
    rhythm = melody_rhythms[int(rng.integers(len(melody_rhythms)))]
    for bar in range(bars):
        chord = _chord(root, prog[bar % 4])
        t0 = bar * 4 * beat
        # bass
        bass_pattern = [0, 2] if mood != "energetic" else [0, 1, 2, 3]
        for b in bass_pattern:
            add(pluck(_midi(chord[0] - 24), 1.2 * beat, 0.6), t0 + b * beat, 0.55)
        # strummed chord
        strums = [0, 1.5, 2, 3] if mood != "calm" else [0, 2]
        for b in strums:
            for i, m in enumerate(chord):
                add(pluck(_midi(m), 1.6 * beat), t0 + b * beat + i * 0.012, 0.16 if b in (0, 2) else 0.1,
                    pan=-0.3 + 0.3 * i)
        # music-box melody from chord tones
        if mood != "energetic" or bar % 2 == 0:
            for k, b in enumerate(rhythm):
                tone = chord[int(rng.integers(3))] + 12 * (1 + (k == len(rhythm) - 1 and bar % 4 == 3))
                add(bell(_midi(tone), 1.0, 0.28 if mood != "calm" else 0.5), t0 + b * beat,
                    0.12 if mood != "calm" else 0.16, pan=0.25)
        # percussion
        if mood != "calm":
            for b in range(4):
                if b in (0, 2):
                    add(sfx("drum").astype(float), t0 + b * beat, 0.35)
                if b in (1, 3):
                    add(sfx("clap").astype(float), t0 + b * beat, 0.08)
            for e in range(8):
                m = int(0.03 * SR)
                shaker = _highpass(rng.uniform(-1, 1, m), 5000) * _env(m, 0.002, 0.012)
                add(shaker, t0 + e * beat / 2, 0.09 if e % 2 else 0.05, pan=0.4)

    loop = np.stack([L, R], axis=1)
    loop /= max(1e-6, np.abs(loop).max())
    reps = int(np.ceil(duration / loop_len)) + 1
    music = np.tile(loop, (reps, 1))[: int(duration * SR)]
    fade_in = min(len(music), int(0.8 * SR))
    music[:fade_in] *= np.linspace(0, 1, fade_in)[:, None]
    fade_out = min(len(music), int(2.5 * SR))
    music[-fade_out:] *= np.linspace(1, 0, fade_out)[:, None]
    return music.astype(np.float32)


# ---------------------------------------------------------------- mixing

def envelope(x: np.ndarray, attack=0.02, release=0.35) -> np.ndarray:
    from scipy.signal import lfilter
    rect = np.abs(x)
    a = np.exp(-1.0 / (release * SR))
    env = lfilter([1 - a], [1, -a], rect)
    peak = np.percentile(env[env > 1e-4], 95) if np.any(env > 1e-4) else 1.0
    return np.clip(env / max(peak, 1e-6), 0, 1)


def mix(duration: float, voice_events, sfx_events, music: np.ndarray, music_level: float, sfx_level: float):
    """voice_events: [(t, mono)], sfx_events: [(t, name, gain)]. Returns (stereo, voice_track)."""
    n = int(duration * SR)
    voice = np.zeros(n, dtype=np.float32)
    for t, clip in voice_events:
        s = int(t * SR)
        e = min(n, s + len(clip))
        voice[s:e] += clip[: e - s]
    peak = np.abs(voice).max()
    if peak > 0:
        voice *= 0.89 / peak
    fx = np.zeros(n, dtype=np.float32)
    for t, name, gain in sfx_events:
        clip = sfx(name)
        s = int(t * SR)
        e = min(n, s + len(clip))
        if s < n:
            fx[s:e] += clip[: e - s] * gain
    duck = 1.0 - 0.55 * envelope(voice)
    m = music[:n] if len(music) >= n else np.pad(music, ((0, n - len(music)), (0, 0)))
    out = m * (music_level * duck)[:, None]
    out += voice[:, None]
    out += (fx * sfx_level)[:, None]
    out = np.tanh(out * 1.1) / np.tanh(1.1)
    return out.astype(np.float32), voice


def write_wav(path: Path, stereo: np.ndarray):
    data = (np.clip(stereo, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(data.tobytes())


def loudnorm(src: Path, dst: Path, target=-14.0):
    """Two-pass EBU R128 normalisation (YouTube plays at about -14 LUFS)."""
    first = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(src), "-af",
         f"loudnorm=I={target}:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"],
        capture_output=True, text=True,
    ).stderr
    blob = first[first.rfind("{"): first.rfind("}") + 1]
    m = json.loads(blob)
    af = (f"loudnorm=I={target}:TP=-1.5:LRA=11:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
          f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-af", af, "-ar", str(SR), str(dst)], check=True)
