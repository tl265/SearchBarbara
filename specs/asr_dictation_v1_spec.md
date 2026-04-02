# SearchBarbara Spec: ASR Dictation for Chat Box (v1.0 Locked)

**Date:** April 2, 2026  
**Owner:** Product + Frontend + Backend  
**Status:** Approved for implementation  
**Primary ASR Engine:** OpenAI `gpt-4o-transcribe`  
**Fallback ASR Engine:** None in v1 (no auto-fallback)

---

## 1) Summary

Add ChatGPT-like voice dictation to the main prompt box so users can speak prompts instead of typing.  
The system streams audio to OpenAI ASR, shows interim transcript updates, and commits final transcript into the existing `#task` textarea.

---

## 2) Goals

1. Provide one-tap microphone dictation near the chat input.
2. Deliver high transcription accuracy with low latency.
3. Preserve existing run flow (user still manually clicks **Start**).
4. Ensure privacy-safe and accessible behavior.
5. Ship with automated tests and CI quality gates.

---

## 3) Non-Goals (v1)

- No TTS or voice playback.
- No full voice conversation loop.
- No persistent raw audio storage.
- No diarization UI.
- No punctuation verbalization guidance UX.
- No auto-fallback to mini model.

---

## 4) Existing Integration Points (Repository)

- **Template/UI:** `frontend/templates/index_v2.html`
  - Existing prompt input: `textarea#task`
  - Existing action area near `#runBtn`
- **Frontend logic:** `frontend/static/app_v2.js`
  - Input handling, char count, run actions
- **Styling:** `frontend/static/styles_v2.css`
  - Button states and prompt-dock styles
- **Backend integration area:** app server routes/auth modules
  - e.g., `app/main.py`, `backend/server.py`, auth infra modules

---

## 5) User Experience

## 5.1 Core flow
1. User taps mic button (`#asrBtn`).
2. Browser requests microphone permission (if needed).
3. App starts listening and streaming audio.
4. User sees live partial transcript.
5. Dictation stops when:
   - user taps mic again, or
   - **silence exceeds 5 seconds**.
6. Final transcript is appended to `#task`.
7. User can edit and click **Start** manually.

## 5.2 UX rules
- Dictation must **not erase existing text**.
- Insert at cursor position; fallback append at end.
- Respect existing `maxlength=140` on `#task`.
- Keep character counter accurate.
- No auto-submit in v1.
- Unsupported browser: mic button disabled + explanatory tooltip/status.

---

## 6) Functional Requirements

### FR-1: ASR control
- Add mic toggle button near prompt controls:
  - idle
  - listening
  - processing
  - disabled/unsupported

### FR-2: Streaming transcript
- Display interim transcript during speech.
- Commit final transcript to textarea value only.

### FR-3: Silence auto-stop
- While listening, if silence duration >5s:
  - stop stream
  - finalize buffered transcript
  - return UI to idle

### FR-4: Error handling
Handle and message:
- permission denied
- no microphone
- silence/no speech
- network/provider failure
- auth/session errors

### FR-5: Session safety
- If session changes during dictation, stop and clean up.
- No transcript leakage across sessions.

### FR-6: Feature flag
- Controlled via config flag (e.g., `APP_CONFIG.asr_enabled`).

### FR-7: Language behavior
- No manual language selector in v1.
- Use model-level auto language detection with `gpt-4o-transcribe`.

---

## 7) Technical Design

## 7.1 Frontend capture + state

Use:
- `navigator.mediaDevices.getUserMedia({ audio: true })`
- `MediaRecorder` for chunked audio
- WebSocket streaming to backend endpoint

Client ASR state:
- `supported: boolean`
- `enabled: boolean`
- `isListening: boolean`
- `isProcessing: boolean`
- `permissionState: 'unknown'|'granted'|'denied'`
- `partialText: string`
- `finalBuffer: string`
- `lastSpeechAt: number | null`
- `silenceTimeoutMs: 5000`
- `localeHint: string | null` (optional metadata only)
- `lastError: string | null`
- `ws: WebSocket | null`
- `mediaRecorder: MediaRecorder | null`
- `stream: MediaStream | null`

## 7.2 Backend ASR gateway

### Endpoint (preferred)
- `WS /api/asr/stream`

### Optional fallback endpoint
- `POST /api/asr/transcribe` (not required for v1 launch)

Server responsibilities:
- authenticate request/session
- receive audio chunks
- stream to OpenAI `gpt-4o-transcribe`
- return partial/final transcript events
- enforce timeout/rate limits
- avoid logging sensitive payload content

## 7.3 Event protocol (proposed)

### Client → Server
- `asr.start`
  - `{ session_id, sample_rate, model: "gpt-4o-transcribe" }`
- `asr.audio_chunk`
  - `{ seq, audio }` (binary or base64 payload)
- `asr.stop`
  - `{ reason: "user_stop" | "silence_timeout" | "session_switch" | "error" }`

### Server → Client
- `asr.ready`
- `asr.partial`
  - `{ text, seq, confidence? }`
- `asr.final`
  - `{ text, utterance_id, confidence? }`
- `asr.error`
  - `{ code, message }`
- `asr.done`

## 7.4 Model policy (locked)
- Always use `gpt-4o-transcribe` in v1.
- No mini auto-routing in v1.

---

## 8) Security & Privacy

1. Require authenticated user/session for ASR endpoint.
2. Do not persist raw audio by default.
3. Do not log transcript content in plaintext.
4. Telemetry may store metadata only:
   - transcript length
   - latency
   - error codes
   - model
5. Inline user notice:
   - “Audio is sent for transcription and not retained by SearchBarbara by default.”

---

## 9) Accessibility

- Mic button has dynamic `aria-label`:
  - “Start voice dictation”
  - “Stop voice dictation”
- Toggle semantics via `aria-pressed`.
- `#asrStatus` uses `aria-live="polite"`.
- Keyboard interactions: Tab + Enter/Space supported.
- Visible focus ring and contrast-compliant state colors.

---

## 10) Observability & Metrics

Track:
- `asr_start`
- `asr_stop`
- `asr_silence_autostop`
- `asr_partial_received`
- `asr_final_received`
- `asr_error`
- `asr_unsupported`

Dimensions:
- browser/device
- locale hint (if collected)
- latency bucket
- permission state
- model (`gpt-4o-transcribe`)

Never include raw audio or transcript text in analytics.

---

## 11) Performance Targets

- p50 start → first partial: **< 800ms**
- p50 stop → final transcript: **< 1.5s**
- start→final success rate: **> 95%** (excluding explicit permission denies)

---

## 12) Rollout Plan

1. Implement behind `asr_enabled` flag.
2. Internal cohort rollout.
3. Validate error rate, silence-stop reliability, and latency.
4. Gradually expand to all users.
5. Rollback path: disable feature flag immediately.

---

## 13) Detailed Test Plan (Mandatory)

## 13.1 Frontend unit tests

1. **ASR supported init**
   - APIs present + flag on → mic enabled idle state.

2. **ASR unsupported init**
   - APIs absent → mic disabled + support message shown.

3. **Start listening transition**
   - Click mic in idle → listening state + aria updates.

4. **Stop listening transition**
   - Click mic while listening → processing then idle.

5. **Partial transcript behavior**
   - Sequential partials update live view without duplication.

6. **Final transcript commit**
   - Final event appends to `#task` at cursor/end.

7. **Typed + dictated merge**
   - Existing text preserved, dictated text added correctly.

8. **Max length enforcement**
   - Input exceeding 140 is safely truncated; counter remains correct.

9. **Permission denied mapping**
   - Deny event shows expected message and returns idle state.

10. **Silence auto-stop**
   - >5s silence triggers stop with reason `silence_timeout` and finalization.

11. **Session switch cleanup**
   - Active dictation stops; stream closes; no leaked text.

## 13.2 Backend integration tests

1. **Auth enforcement**
   - Unauthenticated socket rejected.

2. **Handshake success**
   - Valid `asr.start` returns `asr.ready`.

3. **Chunk validation**
   - Invalid chunk payload returns structured `asr.error`.

4. **Normal stop flow**
   - `asr.stop` emits final (if available) then done.

5. **Provider timeout mapping**
   - Upstream timeout maps to stable client error code.

6. **Model lock verification**
   - Requests route to `gpt-4o-transcribe` in v1.

7. **Sensitive logging guard**
   - Logs contain metadata only; no transcript/audio text.

## 13.3 End-to-end tests (mocked mic + mocked ASR backend)

1. Happy path dictation.
2. Permission-denied path.
3. Rapid toggle start/stop.
4. Typed + dictated mixed input.
5. Session switch during listening.
6. Silence >5s auto-stop behavior.
7. Accessibility keyboard and live-region smoke.

## 13.4 Non-functional tests

1. Partial/final latency benchmark against targets.
2. Long utterance stability (e.g., 60 seconds).
3. Resource cleanup after repeated dictation sessions.

## 13.5 CI merge gates

Required green checks:
- frontend unit tests
- backend integration tests
- ASR E2E smoke tests
- lint/type checks
- privacy logging guard checks

---

## 14) Acceptance Criteria

1. User can dictate into chat box using mic button.
2. Interim transcript appears while speaking.
3. Final transcript inserts into `#task` without deleting existing text.
4. Silence >5s auto-stops and finalizes safely.
5. `maxlength=140` and char counter remain correct.
6. Accessibility requirements are met.
7. No regression in existing prompt submission flow.
8. Automated test suites are present and passing in CI.

---

## 15) Definition of Done

Feature is done only when:
1. End-to-end ASR dictation works in supported browsers.
2. Error/permission/silence paths are stable and user-friendly.
3. Privacy and logging safeguards are verified.
4. CI test gates are green.
5. Feature flag supports safe rollout and immediate rollback.

---

## 16) Open Questions

None for v1.0 lock.  
(Previously open items are now resolved and incorporated.)

---

## 17) References

- OpenAI Speech-to-Text guide:
  https://platform.openai.com/docs/guides/speech-to-text
- OpenAI model docs:
  - https://platform.openai.com/docs/models/gpt-4o-transcribe
  - https://platform.openai.com/docs/models/gpt-4o-mini-transcribe
