from __future__ import annotations

import pytest

from voice_svc.fatigue_level import FatigueLevel


def test_values_match_wire_strings():
    assert FatigueLevel.RESTED == "rested"
    assert FatigueLevel.TIRED == "tired"
    assert FatigueLevel.EXHAUSTED == "exhausted"


def test_constructible_from_wire_string():
    assert FatigueLevel("rested") is FatigueLevel.RESTED
    assert FatigueLevel("tired") is FatigueLevel.TIRED
    assert FatigueLevel("exhausted") is FatigueLevel.EXHAUSTED


def test_unknown_string_raises():
    with pytest.raises(ValueError):
        FatigueLevel("sleepy")
