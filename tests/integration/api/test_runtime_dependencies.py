import asyncio

import pytest

from backend import server


def test_validate_runtime_dependencies_passes_when_asr_disabled(monkeypatch):
    monkeypatch.setitem(server.WEB_CONFIG, "asr_enabled", False)
    monkeypatch.setattr(server, "_websocket_runtime_backend", lambda: "")

    asyncio.run(server.validate_runtime_dependencies())


def test_validate_runtime_dependencies_fails_without_websocket_backend(monkeypatch):
    monkeypatch.setitem(server.WEB_CONFIG, "asr_enabled", True)
    monkeypatch.setattr(server, "_websocket_runtime_backend", lambda: "")

    with pytest.raises(RuntimeError, match="websocket runtime backend"):
        asyncio.run(server.validate_runtime_dependencies())


def test_validate_runtime_dependencies_passes_with_websocket_backend(monkeypatch):
    monkeypatch.setitem(server.WEB_CONFIG, "asr_enabled", True)
    monkeypatch.setattr(server, "_websocket_runtime_backend", lambda: "websockets")

    asyncio.run(server.validate_runtime_dependencies())
