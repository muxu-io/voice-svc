import pytest

from voice_svc.fatigue_level import FatigueLevel
from voice_svc.prosody import (
    PIPER_PROSODY,
    XTTS_PROSODY,
    merge_piper_params,
    merge_xtts_params,
)


def test_xtts_profiles_are_strictly_decreasing_in_speed():
    assert (
        XTTS_PROSODY[FatigueLevel.RESTED]["speed"]
        > XTTS_PROSODY[FatigueLevel.TIRED]["speed"]
        > XTTS_PROSODY[FatigueLevel.EXHAUSTED]["speed"]
    )


def test_piper_length_scales_are_strictly_increasing():
    """length_scale > 1 means slower in Piper."""
    assert (
        PIPER_PROSODY[FatigueLevel.RESTED]["length_scale"]
        < PIPER_PROSODY[FatigueLevel.TIRED]["length_scale"]
        < PIPER_PROSODY[FatigueLevel.EXHAUSTED]["length_scale"]
    )


def test_merge_xtts_multiplies_baseline_speed_through_profile():
    merged = merge_xtts_params(baseline_speed=1.10, fatigue_level=FatigueLevel.EXHAUSTED)
    assert merged["speed"] == pytest.approx(1.10 * XTTS_PROSODY[FatigueLevel.EXHAUSTED]["speed"])
    assert "temperature" in merged


def test_merge_piper_multiplies_baseline_inverse_speed_through_length_scale():
    """baseline_speed < 1 (slower) → length_scale > 1; multiplied by profile."""
    merged = merge_piper_params(baseline_speed=0.90, fatigue_level=FatigueLevel.RESTED)
    expected_baseline_ls = 1.0 / 0.90
    expected = expected_baseline_ls * PIPER_PROSODY[FatigueLevel.RESTED]["length_scale"]
    assert merged["length_scale"] == pytest.approx(expected)
