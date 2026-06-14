from unittest.mock import AsyncMock, MagicMock

import pytest

from voice_svc.engines.base import EngineRequest
from voice_svc.engines.piper import PiperEngine


@pytest.mark.asyncio
async def test_piper_invokes_subprocess_with_voice_id_and_length_scale(monkeypatch):
    captured: dict = {}

    async def fake_create_subprocess_exec(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        proc = MagicMock()
        proc.stdin = AsyncMock()
        proc.stdin.write = MagicMock()
        proc.stdin.drain = AsyncMock()
        proc.stdin.close = MagicMock()
        proc.wait = AsyncMock(return_value=0)

        chunk1 = b"\x00\x00" * 240
        chunk2 = b"\x00\x00" * 240
        proc.stdout = AsyncMock()
        proc.stdout.read = AsyncMock(side_effect=[chunk1, chunk2, b""])

        return proc

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_create_subprocess_exec)

    engine = PiperEngine(piper_bin="piper", model_dir="/voices")
    req = EngineRequest(
        text="hi there",
        voice_id="en_GB-alba-medium",
        params={"length_scale": 1.10},
    )
    chunks = [c async for c in engine.synthesize(req)]
    assert b"".join(chunks)
    assert "piper" in captured["args"]
    flat_args = " ".join(str(a) for a in captured["args"])
    assert "en_GB-alba-medium" in flat_args
    assert "1.10" in flat_args or "1.1" in flat_args
