from __future__ import annotations

from pathlib import Path

import pytest

from voice_svc.synthesis import EngineParams, build_engine_request


def test_mock_only_routes_to_mock():
    name, req = build_engine_request(
        EngineParams(
            text="hi",
            engine="xtts-v2",
            language="en",
            baseline_speed=1.0,
            fatigue_level="rested",
            piper_voice=None,
        ),
        sample_path=Path("/tmp/x.wav"),
        mock_only=True,
    )
    assert name == "mock"
    assert req.text == "hi"
    assert req.voice_id is None


def test_xtts_uses_sample_path_and_prosody():
    name, req = build_engine_request(
        EngineParams(
            text="hello",
            engine="xtts-v2",
            language="en",
            baseline_speed=1.0,
            fatigue_level="tired",
            piper_voice=None,
        ),
        sample_path=Path("/samples/abc.wav"),
        mock_only=False,
    )
    assert name == "xtts-v2"
    assert req.voice_id == "/samples/abc.wav"
    # tired profile: speed 0.92, temperature 0.70 (see prosody.XTTS_PROSODY)
    assert req.params["speed"] == 1.0 * 0.92
    assert req.params["temperature"] == 0.70


def test_piper_uses_voice_id_and_length_scale():
    name, req = build_engine_request(
        EngineParams(
            text="hello",
            engine="piper",
            language="en",
            baseline_speed=1.0,
            fatigue_level="rested",
            piper_voice="en_GB-alba-medium",
        ),
        sample_path=None,
        mock_only=False,
    )
    assert name == "piper"
    assert req.voice_id == "en_GB-alba-medium"
    assert req.params["length_scale"] == 1.0  # baseline 1/1.0 * rested 1.00


def test_unknown_engine_raises():
    with pytest.raises(ValueError, match="unsupported engine"):
        build_engine_request(
            EngineParams(
                text="hi",
                engine="bogus",
                language="en",
                baseline_speed=1.0,
                fatigue_level="rested",
                piper_voice=None,
            ),
            sample_path=None,
            mock_only=False,
        )
