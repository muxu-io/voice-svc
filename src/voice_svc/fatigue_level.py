"""Local fatigue enum. voice-svc is a standalone service: it must not import
`persona`. The three levels mirror persona.fatigue.FatigueLevel by value so
the wire field (`fatigue_level: "rested"|"tired"|"exhausted"`) constructs the
enum directly. The caller derives the level; the service only maps it to
engine prosody (see prosody.py)."""

from __future__ import annotations

from enum import StrEnum


class FatigueLevel(StrEnum):
    RESTED = "rested"
    TIRED = "tired"
    EXHAUSTED = "exhausted"
