"""FastAPI app factory. Stateless: every request carries a fully-resolved
voice spec (engine, language, prosody inputs, and either a piper voice id or a
base64 XTTS reference sample). No persona/state knowledge lives here."""

from __future__ import annotations

import base64
import struct
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, model_validator

from voice_svc.config import VoiceSvcConfig
from voice_svc.engine_registry import EngineRegistry
from voice_svc.samples import materialize_sample
from voice_svc.synthesis import EngineParams, build_engine_request


def _mime_for_format(fmt: str, sample_rate: int) -> str:
    return {
        "pcm": f"audio/L16; rate={sample_rate}",
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "opus": "audio/ogg",
    }.get(fmt.lower(), "application/octet-stream")


def _wav_header(sample_rate: int, num_samples: int, channels: int = 1, bits: int = 16) -> bytes:
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    data_size = num_samples * channels * bits // 8
    return (
        b"RIFF"
        + struct.pack("<I", 36 + data_size)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<IHHIIHH", 16, 1, channels, sample_rate, byte_rate, block_align, bits)
        + b"data"
        + struct.pack("<I", data_size)
    )


class SynthesizeRequest(BaseModel):
    text: str
    engine: str  # "xtts-v2" | "piper"
    language: str = "en"
    baseline_speed: float = 1.0
    fatigue_level: str = "rested"
    piper_voice: str | None = None
    sample_audio_b64: str | None = None
    response_format: str = "pcm"
    include_alignment: bool = False

    @model_validator(mode="after")
    def _check_engine_fields(self) -> SynthesizeRequest:
        if self.engine == "xtts-v2" and not self.sample_audio_b64:
            raise ValueError("engine 'xtts-v2' requires sample_audio_b64")
        if self.engine == "piper" and not self.piper_voice:
            raise ValueError("engine 'piper' requires piper_voice")
        if self.engine not in ("xtts-v2", "piper"):
            raise ValueError(f"unknown engine: {self.engine!r}")
        return self


class OpenAISpeechRequest(BaseModel):
    model: str = "tts-1"
    input: str
    voice: str  # interpreted as a piper voice id
    response_format: str = "mp3"
    speed: float = 1.0


def build_app(
    cfg: VoiceSvcConfig,
    registry: EngineRegistry,
    engines_failed: list[dict] | None = None,
) -> FastAPI:
    app = FastAPI(title="voice-svc")
    failed = list(engines_failed) if engines_failed else []

    def _piper_voices() -> list[str]:
        d = Path(cfg.piper_model_dir)
        if not d.exists():
            return []
        return sorted(p.stem for p in d.glob("*.onnx"))

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "engines": registry.names(),
            "engines_failed": failed,
            "mock_only": cfg.mock_only,
        }

    @app.get("/voices")
    async def voices() -> dict:
        return {"engines": registry.names(), "piper_voices": _piper_voices()}

    def _resolve_sample(req: SynthesizeRequest) -> Path | None:
        if cfg.mock_only or req.engine != "xtts-v2":
            return None
        try:
            return materialize_sample(req.sample_audio_b64, cfg.sample_cache_dir)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    async def _render(spec: EngineParams, sample_path: Path | None, fmt: str):
        engine_name, eng_req = build_engine_request(spec, sample_path, cfg.mock_only)
        engine = registry.get(engine_name)
        rate = engine.sample_rate
        media_type = _mime_for_format(fmt, rate)

        if fmt.lower() == "wav":
            audio = bytearray()
            async for chunk in engine.synthesize(eng_req):
                audio.extend(chunk)
            body = _wav_header(rate, len(audio) // 2) + bytes(audio)
            return StreamingResponse(iter([body]), media_type=media_type)

        async def gen():
            async for chunk in engine.synthesize(eng_req):
                yield chunk

        return StreamingResponse(gen(), media_type=media_type)

    @app.post("/synthesize")
    async def synthesize(req: SynthesizeRequest):
        sample_path = _resolve_sample(req)
        spec = EngineParams(
            text=req.text,
            engine=req.engine,
            language=req.language,
            baseline_speed=req.baseline_speed,
            fatigue_level=req.fatigue_level,
            piper_voice=req.piper_voice,
        )

        if not req.include_alignment:
            return await _render(spec, sample_path, req.response_format)

        engine_name, eng_req = build_engine_request(spec, sample_path, cfg.mock_only)
        engine = registry.get(engine_name)
        rate = engine.sample_rate
        audio_buf = bytearray()
        alignment: list[dict] = []
        async for kind, payload in engine.synthesize_with_alignment(eng_req):
            if kind == "audio":
                audio_buf.extend(payload)
            else:
                alignment.append(payload)
        return {
            "audio_base64": base64.b64encode(bytes(audio_buf)).decode("ascii"),
            "audio_format": req.response_format,
            "sample_rate": rate,
            "alignment": alignment,
        }

    @app.post("/v1/audio/speech")
    async def openai_speech(req: OpenAISpeechRequest):
        spec = EngineParams(
            text=req.input,
            engine="piper",
            language="en",
            baseline_speed=req.speed,
            fatigue_level="rested",
            piper_voice=req.voice,
        )
        return await _render(spec, None, req.response_format)

    app.state.config = cfg
    app.state.registry = registry
    return app
