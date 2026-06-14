import pytest

from voice_svc.engine_registry import EngineRegistry
from voice_svc.engines.mock import MockEngine


@pytest.mark.asyncio
async def test_registry_returns_registered_engine_by_name():
    reg = EngineRegistry()
    engine = MockEngine()
    reg.register("mock", engine)
    assert reg.get("mock") is engine


def test_registry_get_unknown_engine_raises():
    reg = EngineRegistry()
    with pytest.raises(KeyError):
        reg.get("nonexistent")


def test_registry_listing_engines_returns_registered_names():
    reg = EngineRegistry()
    reg.register("mock", MockEngine())
    assert set(reg.names()) == {"mock"}
