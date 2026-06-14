"""Deterministic mock engine for tests. Produces silence (PCM zeros) shaped
by the text length, plus synthetic alignment frames that map characters to
phonemes one-to-one."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Literal

from voice_svc.engines.base import (
    AlignmentFrame,
    EngineAdapter,
    EngineRequest,
)

_SAMPLE_RATE = 24000
_BYTES_PER_SAMPLE = 2  # 16-bit
_SAMPLES_PER_CHAR = 100  # ~4ms per char at 24kHz; deterministic
_CHUNK_SAMPLES = 480  # 20ms chunks


class MockEngine(EngineAdapter):
    name = "mock"
    sample_rate = _SAMPLE_RATE

    async def synthesize(self, req: EngineRequest) -> AsyncIterator[bytes]:
        total_samples = max(_CHUNK_SAMPLES, len(req.text) * _SAMPLES_PER_CHAR)
        chunk_bytes = b"\x00\x00" * _CHUNK_SAMPLES
        emitted = 0
        while emitted < total_samples:
            yield chunk_bytes
            emitted += _CHUNK_SAMPLES

    async def synthesize_with_alignment(
        self, req: EngineRequest
    ) -> AsyncIterator[tuple[Literal["audio", "alignment"], bytes | AlignmentFrame]]:
        ms_per_char = (_SAMPLES_PER_CHAR * 1000) // _SAMPLE_RATE
        cursor_ms = 0
        for ch in req.text:
            yield "alignment", {
                "phoneme": ch,
                "start_ms": cursor_ms,
                "end_ms": cursor_ms + ms_per_char,
            }
            cursor_ms += ms_per_char
        async for chunk in self.synthesize(req):
            yield "audio", chunk
