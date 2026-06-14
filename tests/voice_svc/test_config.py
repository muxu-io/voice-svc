from __future__ import annotations

from pathlib import Path

from voice_svc.config import VoiceSvcConfig


def test_defaults_when_env_unset(monkeypatch):
    for var in (
        "VOICE_SVC_MOCK_ONLY",
        "VOICE_SVC_PIPER_BIN",
        "VOICE_SVC_PIPER_MODEL_DIR",
        "VOICE_SVC_PORT",
        "VOICE_SVC_SAMPLE_CACHE_DIR",
    ):
        monkeypatch.delenv(var, raising=False)

    cfg = VoiceSvcConfig.from_env()

    assert cfg.mock_only is False
    assert cfg.piper_bin == "piper"
    assert cfg.piper_model_dir == "/voices"
    assert cfg.port == 7000
    assert cfg.sample_cache_dir == Path("/tmp/voice-samples")
    assert not hasattr(cfg, "base_dir")


def test_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("VOICE_SVC_MOCK_ONLY", "true")
    monkeypatch.setenv("VOICE_SVC_PORT", "9999")
    monkeypatch.setenv("VOICE_SVC_SAMPLE_CACHE_DIR", str(tmp_path / "s"))

    cfg = VoiceSvcConfig.from_env()

    assert cfg.mock_only is True
    assert cfg.port == 9999
    assert cfg.sample_cache_dir == tmp_path / "s"
