from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI


class AsrError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = str(code or "provider_error")
        self.message = str(message or "ASR request failed.")


@dataclass
class AsrResult:
    text: str
    duration_ms: int
    audio_bytes: int


class OpenAIAsrTranscriber:
    def __init__(
        self,
        *,
        model: str,
        timeout_sec: float = 30.0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.model = str(model or "gpt-4o-transcribe").strip() or "gpt-4o-transcribe"
        self.timeout_sec = max(5.0, float(timeout_sec or 30.0))
        self._log = logger or logging.getLogger("searchbarbara.asr")
        self._client: Optional[OpenAI] = None

    def _client_or_create(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(timeout=self.timeout_sec)
        return self._client

    def transcribe_snapshot(
        self,
        audio_bytes: bytes,
        *,
        mime_type: str = "audio/webm",
        language: str = "",
    ) -> AsrResult:
        payload = bytes(audio_bytes or b"")
        if not payload:
            raise AsrError("no_speech", "No audio payload was received.")
        filename = "dictation.webm"
        if "wav" in str(mime_type or "").lower():
            filename = "dictation.wav"
        started = time.perf_counter()
        try:
            rsp = self._client_or_create().audio.transcriptions.create(
                file=(filename, io.BytesIO(payload), mime_type or "audio/webm"),
                model=self.model,
                response_format="text",
                language=language or None,
            )
        except Exception as exc:
            self._log.warning("ASR transcription failed: %s", exc)
            raise AsrError("provider_error", "Transcription request failed.") from exc
        duration_ms = max(0, int(round((time.perf_counter() - started) * 1000)))
        text = str(rsp or "").strip()
        return AsrResult(text=text, duration_ms=duration_ms, audio_bytes=len(payload))
