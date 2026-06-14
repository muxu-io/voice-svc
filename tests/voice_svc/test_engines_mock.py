import pytest

from voice_svc.engines.base import EngineAdapter, EngineRequest
from voice_svc.engines.mock import MockEngine


@pytest.mark.asyncio
async def test_mock_engine_returns_pcm_chunks():
    engine = MockEngine()
    req = EngineRequest(text="hello world", language="en", params={})
    chunks: list[bytes] = []
    async for chunk in engine.synthesize(req):
        chunks.append(chunk)
    assert len(chunks) > 0
    audio = b"".join(chunks)
    # 16-bit PCM, even byte count
    assert len(audio) % 2 == 0
    # MockEngine produces N samples per text character (~deterministic).
    expected_min_samples = len(req.text) * 100
    assert len(audio) // 2 >= expected_min_samples


@pytest.mark.asyncio
async def test_mock_engine_with_alignment_returns_phoneme_frames():
    engine = MockEngine()
    req = EngineRequest(text="ab", language="en", params={})
    audio: list[bytes] = []
    alignment: list[dict] = []
    async for kind, payload in engine.synthesize_with_alignment(req):
        if kind == "audio":
            audio.append(payload)
        else:
            alignment.append(payload)
    assert audio
    assert alignment
    assert all("phoneme" in f and "start_ms" in f and "end_ms" in f for f in alignment)


def test_engine_adapter_is_a_protocol():
    """MockEngine must satisfy the EngineAdapter Protocol structurally."""
    assert isinstance(MockEngine(), EngineAdapter)


def test_engines_declare_sample_rate():
    """Every engine declares the PCM sample rate it emits. The synthesis
    layer relies on this to set Content-Type and write WAV headers."""
    from voice_svc.engines.piper import PiperEngine
    from voice_svc.engines.xtts import XTTSEngine

    assert MockEngine().sample_rate == 24000
    # PiperEngine doesn't load anything at construction time — the binary is
    # only invoked during synthesize() — so we can instantiate freely.
    assert PiperEngine().sample_rate == 22050
    # XTTSEngine takes a model arg; pass None (we don't synthesize here, just
    # check the rate).
    assert XTTSEngine(model=None).sample_rate == 24000
    assert XTTSEngine(model=None, sample_rate=22050).sample_rate == 22050
