from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from voice_svc.app import build_app
from voice_svc.config import VoiceSvcConfig
from voice_svc.engine_registry import EngineRegistry
from voice_svc.engines.mock import MockEngine


def _build(tmp_path):
    voices_dir = tmp_path / "voices"
    voices_dir.mkdir()
    (voices_dir / "en_GB-alba-medium.onnx").write_bytes(b"x")
    (voices_dir / "en_US-libritts_r-medium.onnx").write_bytes(b"x")
    cfg = VoiceSvcConfig(mock_only=True, piper_model_dir=str(voices_dir))
    registry = EngineRegistry()
    registry.register("mock", MockEngine())
    return build_app(cfg, registry)


@pytest.mark.asyncio
async def test_health(tmp_path):
    app = _build(tmp_path)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["engines"] == ["mock"]
        assert body["mock_only"] is True


@pytest.mark.asyncio
async def test_voices_lists_engines_and_piper_voices(tmp_path):
    app = _build(tmp_path)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/voices")
        assert r.status_code == 200
        body = r.json()
        assert body["engines"] == ["mock"]
        assert sorted(body["piper_voices"]) == [
            "en_GB-alba-medium",
            "en_US-libritts_r-medium",
        ]
