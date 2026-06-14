"""Fatigue → engine prosody mapping. Server-side, engine-specific.

Numbers are starting points; tuning happens during validation (spec § 10),
not in this module. Per-persona overrides are a future feature."""

from __future__ import annotations

from voice_svc.fatigue_level import FatigueLevel

XTTS_PROSODY: dict[FatigueLevel, dict] = {
    FatigueLevel.RESTED: {"speed": 1.00, "temperature": 0.75},
    FatigueLevel.TIRED: {"speed": 0.92, "temperature": 0.70},
    FatigueLevel.EXHAUSTED: {"speed": 0.85, "temperature": 0.65},
}

PIPER_PROSODY: dict[FatigueLevel, dict] = {
    FatigueLevel.RESTED: {"length_scale": 1.00},
    FatigueLevel.TIRED: {"length_scale": 1.10},
    FatigueLevel.EXHAUSTED: {"length_scale": 1.20},
}


def merge_xtts_params(baseline_speed: float, fatigue_level: FatigueLevel) -> dict:
    profile = XTTS_PROSODY[fatigue_level]
    return {
        "speed": baseline_speed * profile["speed"],
        "temperature": profile["temperature"],
    }


def merge_piper_params(baseline_speed: float, fatigue_level: FatigueLevel) -> dict:
    """Piper's length_scale is inverse-rate: >1 = slower. The persona's
    baseline_speed (>1 = faster) maps to baseline_length_scale = 1/speed,
    then the profile multiplies through."""
    profile = PIPER_PROSODY[fatigue_level]
    baseline_ls = 1.0 / baseline_speed if baseline_speed > 0 else 1.0
    return {"length_scale": baseline_ls * profile["length_scale"]}
