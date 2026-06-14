"""Synthesis orchestration: map a resolved voice spec + fatigue level to an
engine request. No persona knowledge — the caller resolves the spec and sends
it on the wire (see persona.voice_spec). Prosody (fatigue + baseline_speed →
engine knobs) stays server-side and engine-specific."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from voice_svc.engines.base import EngineRequest
from voice_svc.fatigue_level import FatigueLevel
from voice_svc.prosody import merge_piper_params, merge_xtts_params


@dataclass(frozen=True)
class EngineParams:
    text: str
    engine: str
    language: str
    baseline_speed: float
    fatigue_level: str
    piper_voice: str | None


def build_engine_request(
    spec: EngineParams,
    sample_path: Path | None,
    mock_only: bool = False,
) -> tuple[str, EngineRequest]:
    """Returns (engine_name, EngineRequest). engine_name routes through the
    registry; in mock_only mode every request resolves to the mock engine."""
    if mock_only:
        return "mock", EngineRequest(
            text=spec.text,
            language=spec.language,
            voice_id=None,
            params={},
        )

    fatigue = FatigueLevel(spec.fatigue_level)

    if spec.engine == "xtts-v2":
        params = merge_xtts_params(baseline_speed=spec.baseline_speed, fatigue_level=fatigue)
        return "xtts-v2", EngineRequest(
            text=spec.text,
            language=spec.language,
            voice_id=str(sample_path) if sample_path else None,
            params=params,
        )

    if spec.engine == "piper":
        params = merge_piper_params(baseline_speed=spec.baseline_speed, fatigue_level=fatigue)
        return "piper", EngineRequest(
            text=spec.text,
            language=spec.language,
            voice_id=spec.piper_voice,
            params=params,
        )

    raise ValueError(f"unsupported engine: {spec.engine!r}")
