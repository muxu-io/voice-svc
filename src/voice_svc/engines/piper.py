"""Piper engine adapter. Invokes the `piper` binary as a subprocess; pipes
text in via stdin, streams raw 16-bit PCM out via stdout. The container's
Dockerfile installs piper-tts and the voice model files."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Literal

from voice_svc.engines.base import (
    AlignmentFrame,
    EngineAdapter,
    EngineRequest,
)

_CHUNK_BYTES = 4096


class PiperEngine(EngineAdapter):
    name = "piper"
    # Default Piper voice models (medium tier) emit 22050 Hz mono int16.
    # Each .onnx.json declares its sample_rate exactly; for now we hardcode
    # the common case and treat per-voice rates as a future refinement.
    sample_rate = 22050

    def __init__(self, piper_bin: str = "piper", model_dir: str = "/voices"):
        self._piper_bin = piper_bin
        self._model_dir = model_dir

    async def synthesize(self, req: EngineRequest) -> AsyncIterator[bytes]:
        if not req.voice_id:
            raise ValueError("PiperEngine requires voice_id")
        length_scale = float(req.params.get("length_scale", 1.0))
        model_path = str(Path(self._model_dir) / f"{req.voice_id}.onnx")

        proc = await asyncio.create_subprocess_exec(
            self._piper_bin,
            "--model",
            model_path,
            "--output-raw",
            "--length_scale",
            f"{length_scale}",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        if proc.stdin is not None:
            proc.stdin.write(req.text.encode("utf-8"))
            await proc.stdin.drain()
            proc.stdin.close()

        assert proc.stdout is not None
        while True:
            chunk = await proc.stdout.read(_CHUNK_BYTES)
            if not chunk:
                break
            yield chunk
        await proc.wait()

    async def synthesize_with_alignment(
        self, req: EngineRequest
    ) -> AsyncIterator[tuple[Literal["audio", "alignment"], bytes | AlignmentFrame]]:
        async for chunk in self.synthesize(req):
            yield "audio", chunk
