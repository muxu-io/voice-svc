"""XTTS-v2 engine adapter. Coqui TTS's XTTS v2 model, used when an authored
voice-sample.wav exists for the persona. Conditions on the sample for voice
identity; supports `speed` and `temperature` knobs that map to the prosody
profile defined in voice_svc.prosody.

The real model loads in the container at startup. Tests inject a mock with
`inference_stream` returning an iterable of float32 numpy arrays."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Literal

import numpy as np

from voice_svc.engines.base import (
    AlignmentFrame,
    EngineAdapter,
    EngineRequest,
)


class XTTSEngine(EngineAdapter):
    name = "xtts-v2"

    def __init__(self, model: Any, sample_rate: int = 24000):
        self._model = model
        self.sample_rate = sample_rate
        # Keep the legacy underscore-prefixed name for any code that may
        # still refer to it (none in-tree, but it's a public attribute name
        # in the test that injects the engine directly).
        self._sample_rate = sample_rate
        # Per-voice latent cache. `Xtts.inference_stream` does not accept
        # `speaker_wav` directly; the reference clip must first be encoded
        # via `get_conditioning_latents`. Latents are deterministic from the
        # WAV so we cache them by path to avoid re-encoding on every sentence.
        self._latents: dict[str, tuple[Any, Any]] = {}

    async def synthesize(self, req: EngineRequest) -> AsyncIterator[bytes]:
        if not req.voice_id:
            raise ValueError("XTTSEngine requires voice_id (path to sample wav)")
        speed = float(req.params.get("speed", 1.0))
        temperature = float(req.params.get("temperature", 0.75))

        gpt_cond_latent, speaker_embedding = self._get_latents(req.voice_id)

        chunk_iter = self._model.inference_stream(
            text=req.text,
            language=req.language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            speed=speed,
            temperature=temperature,
        )

        for arr in chunk_iter:
            if isinstance(arr, np.ndarray):
                pcm = np.clip(arr * 32767.0, -32768, 32767).astype(np.int16).tobytes()
            else:
                pcm = (
                    np.clip(arr.detach().cpu().numpy() * 32767.0, -32768, 32767)
                    .astype(np.int16)
                    .tobytes()
                )
            yield pcm

    async def synthesize_with_alignment(
        self, req: EngineRequest
    ) -> AsyncIterator[tuple[Literal["audio", "alignment"], bytes | AlignmentFrame]]:
        async for chunk in self.synthesize(req):
            yield "audio", chunk

    def _get_latents(self, voice_id: str) -> tuple[Any, Any]:
        """Return cached (gpt_cond_latent, speaker_embedding) for the given
        reference WAV, computing them on first miss."""
        cached = self._latents.get(voice_id)
        if cached is not None:
            return cached
        gpt_cond_latent, speaker_embedding = self._model.get_conditioning_latents(
            audio_path=voice_id
        )
        self._latents[voice_id] = (gpt_cond_latent, speaker_embedding)
        return gpt_cond_latent, speaker_embedding
