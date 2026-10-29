import json
from pathlib import Path

from pofi.schema import validate

ROOT = Path(__file__).resolve().parent.parent


def test_example_episode_is_valid():
    ep = json.loads((ROOT / "bolumler" / "2026-10-29-cumhuriyet-bayrami.json").read_text(encoding="utf-8"))
    assert validate(ep) == []


def test_catches_errors():
    ep = json.loads((ROOT / "bolumler" / "2026-10-29-cumhuriyet-bayrami.json").read_text(encoding="utf-8"))
    ep["scenes"][0]["type"] = "talk"
    ep["scenes"].append({"type": "quiz", "question": "?", "options": [{"emoji": "🍎", "label": "Elma"}], "answer": 3})
    errs = validate(ep)
    assert any("intro" in e for e in errs)
    assert any("answer" in e for e in errs)
    assert any("outro" in e for e in errs)
