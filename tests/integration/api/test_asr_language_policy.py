from backend.infra.asr import AsrResult, OpenAIAsrTranscriber


class _FakeTranscriptions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return "can you hear me"


class _FakeAudio:
    def __init__(self):
        self.transcriptions = _FakeTranscriptions()


class _FakeClient:
    def __init__(self):
        self.audio = _FakeAudio()


def test_transcriber_uses_language_auto_detect_when_language_is_empty():
    fake_client = _FakeClient()
    transcriber = OpenAIAsrTranscriber(model="gpt-4o-transcribe")
    transcriber._client = fake_client

    result = transcriber.transcribe_snapshot(
        b"fake-audio",
        mime_type="audio/webm",
        language="",
    )

    assert isinstance(result, AsrResult)
    assert result.text == "can you hear me"
    assert fake_client.audio.transcriptions.calls[0]["language"] is None
