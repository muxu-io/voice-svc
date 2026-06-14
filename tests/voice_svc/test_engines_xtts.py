from unittest.mock import MagicMock

import numpy as np
import pytest

from voice_svc.engines.base import EngineRequest
from voice_svc.engines.xtts import XTTSEngine


@pytest.mark.asyncio
async def test_xtts_invokes_inference_stream_with_speed_and_temperature():
    fake_model = MagicMock()
    fake_model.get_conditioning_latents = MagicMock(
        return_value=("fake_gpt_latent", "fake_speaker_embedding")
    )
    fake_model.inference_stream = MagicMock(
        return_value=iter(
            [
                np.zeros(2400, dtype=np.float32),
                np.zeros(2400, dtype=np.float32),
            ]
        )
    )

    engine = XTTSEngine(model=fake_model, sample_rate=24000)
    req = EngineRequest(
        text="hello",
        language="en",
        voice_id="/state/alice/media/voice-sample.wav",
        params={"speed": 0.92, "temperature": 0.70},
    )
    chunks = [c async for c in engine.synthesize(req)]
    audio = b"".join(chunks)
    assert audio
    assert len(audio) % 2 == 0

    fake_model.get_conditioning_latents.assert_called_once_with(
        audio_path="/state/alice/media/voice-sample.wav"
    )
    fake_model.inference_stream.assert_called_once()
    call_kwargs = fake_model.inference_stream.call_args.kwargs
    assert call_kwargs["language"] == "en"
    assert call_kwargs["speed"] == pytest.approx(0.92)
    assert call_kwargs["temperature"] == pytest.approx(0.70)
    assert call_kwargs["gpt_cond_latent"] == "fake_gpt_latent"
    assert call_kwargs["speaker_embedding"] == "fake_speaker_embedding"


@pytest.mark.asyncio
async def test_xtts_caches_conditioning_latents_per_voice():
    """Latents are deterministic from the WAV; recompute would waste a forward
    pass on every sentence. Cache by voice_id."""
    fake_model = MagicMock()
    fake_model.get_conditioning_latents = MagicMock(
        return_value=("fake_gpt_latent", "fake_speaker_embedding")
    )
    fake_model.inference_stream = MagicMock(return_value=iter([np.zeros(1200, dtype=np.float32)]))

    engine = XTTSEngine(model=fake_model, sample_rate=24000)
    req = EngineRequest(
        text="hello",
        language="en",
        voice_id="/state/alice/media/voice-sample.wav",
    )
    [c async for c in engine.synthesize(req)]
    fake_model.inference_stream.return_value = iter([np.zeros(1200, dtype=np.float32)])
    [c async for c in engine.synthesize(req)]

    # Two synthesize calls, but only one latent computation.
    fake_model.get_conditioning_latents.assert_called_once()
