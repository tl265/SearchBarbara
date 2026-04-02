import asyncio
import base64
from types import SimpleNamespace

from backend import server
from backend.infra.asr import AsrResult


class FakeWebSocket:
    def __init__(self, messages):
        self.messages = list(messages)
        self.sent = []
        self.cookies = {}
        self.headers = {}
        self.accepted = False
        self.closed = None

    async def accept(self):
        self.accepted = True

    async def close(self, code=1000):
        self.closed = code

    async def receive_json(self):
        if not self.messages:
            raise server.WebSocketDisconnect()
        return self.messages.pop(0)

    async def send_json(self, payload):
        self.sent.append(payload)


def _enable_test_asr(monkeypatch):
    monkeypatch.setitem(server.WEB_CONFIG, "asr_enabled", True)
    monkeypatch.setattr(
        server,
        "_current_user_from_websocket",
        lambda _websocket: SimpleNamespace(
            user_id="test-user",
            email="test@example.com",
            session_expires_at=0,
        ),
    )


def test_asr_stream_emits_partial_and_final(monkeypatch):
    _enable_test_asr(monkeypatch)
    calls = {"count": 0}

    async def fake_transcribe_asr_snapshot(audio_bytes, *, mime_type="audio/webm", language=""):
        calls["count"] += 1
        text = "partial transcript" if calls["count"] == 1 else "final transcript"
        return AsrResult(text=text, duration_ms=123, audio_bytes=len(audio_bytes))

    monkeypatch.setattr(server, "_transcribe_asr_snapshot", fake_transcribe_asr_snapshot)

    payload = base64.b64encode(b"fake-webm-audio").decode("ascii")
    websocket = FakeWebSocket(
        [
            {
                "type": "asr.start",
                "session_id": "draft-session",
                "mime_type": "audio/webm",
                "language": "en",
            },
            {
                "type": "asr.audio_chunk",
                "session_id": "draft-session",
                "seq": 1,
                "audio": payload,
                "replace_buffer": True,
            },
            {
                "type": "asr.stop",
                "reason": "user_stop",
            },
        ]
    )

    asyncio.run(server.asr_stream(websocket))

    assert websocket.accepted is True
    assert websocket.sent[0]["type"] == "asr.ready"
    assert websocket.sent[1]["type"] == "asr.partial"
    assert websocket.sent[1]["text"] == "partial transcript"
    assert websocket.sent[2]["type"] == "asr.final"
    assert websocket.sent[2]["text"] == "final transcript"
    assert websocket.sent[3]["type"] == "asr.done"
    assert calls["count"] == 2


def test_asr_stream_rejects_unauthorized_socket(monkeypatch):
    monkeypatch.setitem(server.WEB_CONFIG, "asr_enabled", True)
    monkeypatch.setattr(server, "_current_user_from_websocket", lambda _websocket: None)
    websocket = FakeWebSocket([])

    asyncio.run(server.asr_stream(websocket))

    assert websocket.accepted is False
    assert websocket.closed == 4401
