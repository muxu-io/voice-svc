"""Run voice-svc with `python -m voice_svc`. Loads engines from config,
binds the FastAPI app, and serves with uvicorn.

Engine loading is best-effort: if a heavy engine library is missing or its
model fails to load, the service continues with the engines that did load.
The first request that needs a missing engine returns 503."""

from __future__ import annotations

import logging
import sys

import uvicorn

from voice_svc.app import build_app
from voice_svc.config import VoiceSvcConfig
from voice_svc.engine_registry import EngineRegistry
from voice_svc.engines.mock import MockEngine


def _load_engines(cfg: VoiceSvcConfig) -> tuple[EngineRegistry, list[dict]]:
    """Load all engines that should be available for this config.

    Returns (registry, failures). `failures` is a list of
    `{"name": <engine>, "error": <message>}` for any engine that failed to
    load. When `cfg.mock_only=False`, failures are logged at ERROR level (not
    WARNING) — a missing real engine is a deployment problem worth shouting
    about, not a soft hint."""
    log = logging.getLogger("voice_svc.bootstrap")
    registry = EngineRegistry()
    failures: list[dict] = []

    if cfg.mock_only:
        registry.register("mock", MockEngine())
        return registry, failures

    fail_log = log.error  # real-runtime: missing engines are loud failures

    # Piper
    try:
        from voice_svc.engines.piper import PiperEngine

        registry.register(
            "piper",
            PiperEngine(piper_bin=cfg.piper_bin, model_dir=cfg.piper_model_dir),
        )
        log.info("registered piper engine")
    except Exception as e:  # noqa: BLE001
        fail_log("piper engine not available: %s", e)
        failures.append({"name": "piper", "error": str(e)})

    # XTTS-v2
    try:
        from TTS.api import TTS  # type: ignore[import-not-found]

        from voice_svc.engines.xtts import XTTSEngine

        model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
        registry.register("xtts-v2", XTTSEngine(model=model.synthesizer.tts_model))
        log.info("registered xtts-v2 engine")
    except Exception as e:  # noqa: BLE001
        fail_log("xtts-v2 engine not available: %s", e)
        failures.append({"name": "xtts-v2", "error": str(e)})

    return registry, failures


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    cfg = VoiceSvcConfig.from_env()
    registry, failures = _load_engines(cfg)
    app = build_app(cfg, registry, engines_failed=failures)
    uvicorn.run(app, host="0.0.0.0", port=cfg.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
