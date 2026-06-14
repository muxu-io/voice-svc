"""End-to-end probes against the running container with a real GPU.

`--gpus all` and no MOCK_ONLY override, so synthesis runs through the real
engines: Piper (baked voices) and XTTS-v2 (Coqui fork, ~2 GB weights pulled at
first boot into the mounted TTS cache). Selected by `pytest -m gpu`;
auto-skips without docker / image / NVIDIA host.

XTTS-v2 needs a reference clip to clone; the test synthesizes a short tone WAV
in-process and ships it as the base64 sample. The point is a smoke of the real
inference path, not voice quality."""

from __future__ import annotations

import base64
import io
import math
import struct
import wave

import httpx
import pytest

pytestmark = pytest.mark.gpu


def _tone_wav_b64(seconds: float = 3.0, rate: int = 24000, freq: float = 140.0) -> str:
    """A mono 16-bit PCM WAV of a quiet sine tone, base64-encoded. Valid audio
    for XTTS to derive conditioning latents from (content, not quality, matters
    for a smoke test)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        frames = bytearray()
        for n in range(int(seconds * rate)):
            sample = int(0.2 * 32767 * math.sin(2 * math.pi * freq * n / rate))
            frames += struct.pack("<h", sample)
        wf.writeframes(bytes(frames))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_health_reports_real_engines_loaded(gpu_http: httpx.Client) -> None:
    r = gpu_http.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["mock_only"] is False
    assert "piper" in body["engines"]
    assert "xtts-v2" in body["engines"], f"xtts failed to load: {body.get('engines_failed')}"


def test_piper_real_synthesis_returns_wav(gpu_http: httpx.Client) -> None:
    r = gpu_http.post(
        "/synthesize",
        json={
            "text": "The quick brown fox jumps over the lazy dog.",
            "engine": "piper",
            "piper_voice": "en_GB-alba-medium",
            "response_format": "wav",
        },
    )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("audio/wav")
    with wave.open(io.BytesIO(r.content), "rb") as wf:
        assert wf.getnframes() > 0


def test_xtts_real_synthesis_returns_wav(gpu_http: httpx.Client) -> None:
    r = gpu_http.post(
        "/synthesize",
        json={
            "text": "Hello from the cloned voice.",
            "engine": "xtts-v2",
            "language": "en",
            "sample_audio_b64": _tone_wav_b64(),
            "response_format": "wav",
        },
    )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("audio/wav")
    with wave.open(io.BytesIO(r.content), "rb") as wf:
        assert wf.getnframes() > 0
