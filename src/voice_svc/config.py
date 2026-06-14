"""Voice-svc configuration. Read from env in production; constructed
explicitly in tests. The service is stateless — it holds no persona/state
base dir; callers send a fully-resolved voice spec per request."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VoiceSvcConfig:
    mock_only: bool = False
    piper_bin: str = "piper"
    piper_model_dir: str = "/voices"
    port: int = 7000
    sample_cache_dir: Path = Path("/tmp/voice-samples")

    @classmethod
    def from_env(cls) -> VoiceSvcConfig:
        return cls(
            mock_only=os.environ.get("VOICE_SVC_MOCK_ONLY", "").lower() == "true",
            piper_bin=os.environ.get("VOICE_SVC_PIPER_BIN", "piper"),
            piper_model_dir=os.environ.get("VOICE_SVC_PIPER_MODEL_DIR", "/voices"),
            port=int(os.environ.get("VOICE_SVC_PORT", "7000")),
            sample_cache_dir=Path(
                os.environ.get("VOICE_SVC_SAMPLE_CACHE_DIR", "/tmp/voice-samples")
            ),
        )
