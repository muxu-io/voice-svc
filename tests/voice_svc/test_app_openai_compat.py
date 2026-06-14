from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from voice_svc.app import build_app
from voice_svc.config import VoiceSvcConfig
from voice_svc.engine_registry import EngineRegistry
from voice_svc.engines.mock import MockEngine


def _build(tmp_path):
    cfg = VoiceSvcConfig(mock_only=True, sample_cache_dir=tmp_path / "s")
    registry = EngineRegistry()
    registry.register("mock", MockEngine())
    return build_app(cfg, registry)


@pytest.mark.asyncio
async def test_openai_speech_returns_audio(tmp_path):
    app = _build(tmp_path)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post(
            "/v1/audio/speech",
            json={"input": "hello", "voice": "en_GB-alba-medium", "response_format": "wav"},
        )
        assert r.status_code == 200
        assert r.content[:4] == b"RIFF"
