"""Engine adapter protocol. Each TTS engine (XTTS-v2, Piper, mock) implements
this. The adapter is the seam at which engines are substitutable; v1 ships
with XTTS-v2 + Piper, future migrations may replace XTTS with OpenVoice and
Piper with Kokoro without changing the adapter contract."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable


@dataclass(frozen=True)
class EngineRequest:
    text: str
    language: str = "en"
    voice_id: str | None = None  # Piper voice id, or XTTS reference path
    params: dict = field(default_factory=dict)  # engine-specific knobs


AlignmentFrame = dict  # {"phoneme": str, "start_ms": int, "end_ms": int}


@runtime_checkable
class EngineAdapter(Protocol):
    name: str
    sample_rate: int  # PCM sample rate the engine emits (Hz, mono int16)

    async def synthesize(self, req: EngineRequest) -> AsyncIterator[bytes]:
        """Yield PCM 16-bit mono audio chunks at `self.sample_rate`."""
        ...

    async def synthesize_with_alignment(
        self, req: EngineRequest
    ) -> AsyncIterator[tuple[Literal["audio", "alignment"], bytes | AlignmentFrame]]:
        """Interleaved audio + alignment stream. May yield no alignment frames
        if the engine does not support them; consumers should treat alignment
        as best-effort."""
        ...
