"""Engine registry. Holds engine instances for the lifetime of the service so
they stay warm across requests."""

from __future__ import annotations

from voice_svc.engines.base import EngineAdapter


class EngineRegistry:
    def __init__(self):
        self._engines: dict[str, EngineAdapter] = {}

    def register(self, name: str, engine: EngineAdapter) -> None:
        self._engines[name] = engine

    def get(self, name: str) -> EngineAdapter:
        if name not in self._engines:
            raise KeyError(f"engine {name!r} not registered")
        return self._engines[name]

    def names(self) -> list[str]:
        return list(self._engines.keys())
