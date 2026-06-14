from __future__ import annotations

import base64
import struct

import pytest
from httpx import ASGITransport, AsyncClient

from voice_svc.app import build_app
from voice_svc.config import VoiceSvcConfig
from voice_svc.engine_registry import EngineRegistry
from voice_svc.engines.mock import MockEngine

_SAMPLE_B64 = base64.b64encode(b"RIFFmock-wav").decode("ascii")


def _build(tmp_path):
    cfg = VoiceSvcConfig(mock_only=True, sample_cache_dir=tmp_path / "samples")
    registry = EngineRegistry()
    registry.register("mock", MockEngine())
    return build_app(cfg, registry)


def _xtts_body(text="hello world", **over):
    body = {
        "text": text,
        "engine": "xtts-v2",
        "language": "en",
        "baseline_speed": 1.0,
        "fatigue_level": "rested",
        "sample_audio_b64": _SAMPLE_B64,
        "response_format": "pcm",
    }
    body.update(over)
    return body


@pytest.mark.asyncio
async def test_synthesize_returns_pcm_audio(tmp_path):
    app = _build(tmp_path)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/synthesize", json=_xtts_body())
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("audio/")
        assert len(r.content) > 0 and len(r.content) % 2 == 0


@pytest.mark.asyncio
async def test_synthesize_pcm_content_type_carries_rate(tmp_path):
    app = _build(tmp_path)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/synthesize", json=_xtts_body(text="hello"))
        assert "rate=24000" in r.headers["content-type"]


@pytest.mark.asyncio
async def test_synthesize_piper_needs_no_sample(tmp_path):
    app = _build(tmp_path)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post(
            "/synthesize",
            json={
                "text": "hello",
                "engine": "piper",
                "piper_voice": "en_GB-alba-medium",
            },
        )
        assert r.status_code == 200


@pytest.mark.asyncio
async def test_xtts_without_sample_returns_422(tmp_path):
    app = _build(tmp_path)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post(
            "/synthesize",
            json={"text": "hi", "engine": "xtts-v2"},  # missing sample_audio_b64
        )
        assert r.status_code == 422


@pytest.mark.asyncio
async def test_piper_without_voice_returns_422(tmp_path):
    app = _build(tmp_path)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post(
            "/synthesize",
            json={"text": "hi", "engine": "piper"},  # missing piper_voice
        )
        assert r.status_code == 422


@pytest.mark.asyncio
async def test_synthesize_with_alignment_returns_json(tmp_path):
    app = _build(tmp_path)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/synthesize", json=_xtts_body(text="ab", include_alignment=True))
        assert r.status_code == 200
        body = r.json()
        assert "audio_base64" in body
        assert isinstance(body["alignment"], list) and len(body["alignment"]) >= 2
        assert body["sample_rate"] == 24000


@pytest.mark.asyncio
async def test_synthesize_wav_has_riff_header(tmp_path):
    app = _build(tmp_path)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/synthesize", json=_xtts_body(text="hello", response_format="wav"))
        assert r.status_code == 200
        body = r.content
        assert body[:4] == b"RIFF" and body[8:12] == b"WAVE"
        fmt = struct.unpack("<IHHIIHH", body[16:36])
        assert fmt[1] == 1 and fmt[2] == 1 and fmt[3] == 24000 and fmt[6] == 16
