"""End-to-end probes against the running container in mock mode.

Real uvicorn + FastAPI + MockEngine over loopback. No GPU, no weights.
Started with VOICE_SVC_MOCK_ONLY=true, so every request routes to the mock
engine regardless of the requested engine; the request still goes through the
real Pydantic validation and streaming/WAV-framing code paths."""

from __future__ import annotations

import io
import struct
import wave

import httpx
import pytest

pytestmark = pytest.mark.integration


def test_health(http: httpx.Client) -> None:
    r = http.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["mock_only"] is True
    assert body["engines"] == ["mock"]


def test_voices_lists_baked_piper_voices(http: httpx.Client) -> None:
    r = http.get("/voices")
    assert r.status_code == 200
    body = r.json()
    assert body["engines"] == ["mock"]
    # The image bakes three piper voices into /voices.
    assert "en_GB-alba-medium" in body["piper_voices"]
    assert "en_GB-northern_english_male-medium" in body["piper_voices"]
    assert "en_US-libritts_r-medium" in body["piper_voices"]


def test_synthesize_pcm_carries_rate(http: httpx.Client) -> None:
    r = http.post(
        "/synthesize",
        json={
            "text": "hello there",
            "engine": "piper",
            "piper_voice": "en_GB-alba-medium",
            "response_format": "pcm",
        },
    )
    assert r.status_code == 200, r.text
    assert "rate=24000" in r.headers["content-type"]  # MockEngine rate
    assert len(r.content) > 0 and len(r.content) % 2 == 0


def test_synthesize_wav_has_riff_header(http: httpx.Client) -> None:
    r = http.post(
        "/synthesize",
        json={
            "text": "hello there",
            "engine": "piper",
            "piper_voice": "en_GB-alba-medium",
            "response_format": "wav",
        },
    )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("audio/wav")
    with wave.open(io.BytesIO(r.content), "rb") as wf:
        assert wf.getframerate() == 24000
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
    # Sanity on the raw RIFF magic too.
    assert r.content[:4] == b"RIFF" and r.content[8:12] == b"WAVE"
    assert struct.unpack("<H", r.content[20:22])[0] == 1  # PCM


def test_xtts_without_sample_is_rejected(http: httpx.Client) -> None:
    # Validation runs even in mock mode: engine xtts-v2 requires a sample.
    r = http.post("/synthesize", json={"text": "hi", "engine": "xtts-v2"})
    assert r.status_code == 422
