# Session Management V1 Specification (Multi-User Ready, Single-User Runtime)

## 1) Summary

Implement persistent, ChatGPT-like session management so users can close the browser and return later, with a session list UI and run restoration.

V1 behavior:
- `1 session = 1 run`
- single-user runtime (no auth)
- multi-user-ready schema (`owner_id` reserved)
- keep-forever retention with manual delete
- core ops: open, rename, delete
- auto-reconnect SSE for active runs when reopened

Future-proofing:
- add pause-ready backend state hooks (without implementing manual node editing yet)

---

## 2) Product Scope (Locked)

### Included
- Persistent session index stored on disk
- Session list/read/update/delete APIs
- Session list UI (title + status + updated time)
- URL/session restore behavior
- Startup reconciliation and recovery
- Backward compatibility with existing `/api/runs/*` endpoints

### Not included in V1
- Authentication/authorization
- Multi-run threads per session
- Manual node edit/assert/pause UI
- Workflow control engine for rerun-from-modified-node logic

---

## 3) Core Model

### Session identity
- `session_id` equals `run_id` in V1
- One run maps to one session card

### Session metadata (index-level)
- `session_id: str`
- `owner_id: Optional[str]` (null for now)
- `title: str`
- `status: str` (`queued|running|completed|failed|aborted`)
- `execution_state: str` (`idle|running|paused|completed|failed|aborted`)
- `created_at: iso8601`
- `updated_at: iso8601`
- `last_checkpoint_at: Optional[iso8601]`
- `state_file_path: str`
- `report_file_path: Optional[str]`
- `has_manual_edits: bool` (default `false`)

### Future hooks (state-level)
- `manual_edit_log: []`
- `manual_assertions: {}`

---

## 4) Persistence Design

### 4.1 New index file
- Path: `runs/sessions_index.json`
- Contains lightweight metadata for all sessions

### 4.2 Atomic writes
- Persist index using temp-file + rename to avoid corruption

### 4.3 Startup reconciliation
On startup:
1. Load `runs/sessions_index.json` if valid.
2. Rehydrate in-memory run/session state from indexed `state_file_path` files.
3. If index entry exists but state file missing, mark session failed/unavailable.
4. Backfill missing index entries from discovered `runs/state_web_*.json` files.

### 4.4 Event-driven index updates
Any run status transition or report-generation transition updates index summary fields and flushes to disk.

---

## 5) API Design

## 5.1 New endpoints

### `GET /api/sessions`
Returns session summaries sorted by `updated_at desc`.

Response item fields:
- `session_id`
- `title`
- `status`
- `execution_state`
- `updated_at`
- `created_at`
- `owner_id` (nullable)

### `GET /api/sessions/{session_id}`
Returns full snapshot for selected session (equivalent to run snapshot payload + session fields as needed).

### `PATCH /api/sessions/{session_id}`
Request:
```json
{ "title": "new title" }
```
Rules:
- trim whitespace
- reject empty title
- enforce max length (e.g., 200 chars)

### `DELETE /api/sessions/{session_id}`
Rules:
- reject delete for active running session (`409`) unless aborted first
- remove session from index
- remove state file
- keep report file by default in V1 (future config can change this)

## 5.2 Existing endpoints kept
- `POST /api/runs`
- `GET /api/runs/{run_id}`
- `GET /api/runs/{run_id}/events`
- report/abort endpoints

Compatibility contract:
- existing clients continue working
- new UI may prefer session endpoints

---

## 6) Run Creation Integration

On `POST /api/runs`:
1. Create run as today.
2. Create a session index entry immediately:
   - `session_id = run_id`
   - `title = trimmed task (capped)`
   - initial statuses and timestamps
3. Persist index before worker execution starts.

---

## 7) Pause-Ready Hook Design (No pause workflow yet)

Introduce execution-state plumbing now:
- add `execution_state` in session index and transitions
- centralize transition updates in one helper
- reserve `manual_edit_log` and `manual_assertions` fields in persisted state

No pause endpoint behavior is required in this iteration; this is schema and transition scaffolding only.

---

## 8) UI V1 Design

### 8.1 Session list panel
Show sessions with:
- title
- status badge
- updated timestamp

### 8.2 Session actions
Per row:
- open
- rename
- delete (with confirmation)

### 8.3 Restore behavior
- URL supports session selection (`session_id`), with fallback compatibility to `run_id`.
- On page load:
  1. fetch session list
  2. open URL-selected session if present
  3. otherwise open most recent session

### 8.4 Active session reconnect
If selected session is running, auto-connect SSE stream and continue live updates.

---

## 9) Data Model / Type Changes

## `app/models.py`
Add:
- `SessionSummary`
- `SessionListResponse`
- `PatchSessionRequest`

Optionally include in snapshot response:
- `title`
- `execution_state`

## `app/run_manager.py`
Add methods:
- `list_sessions()`
- `get_session(session_id)`
- `rename_session(session_id, title)`
- `delete_session(session_id)`
- index load/save/reconcile helpers
- centralized execution-state transition updater

## `app/main.py`
Add session endpoints and wire to run manager.

---

## 10) Edge Cases / Failure Modes

- Corrupt index JSON: recover with empty index + backfill from state files.
- Missing state file for indexed session: mark as failed/unavailable, keep row visible.
- Rename to duplicate title: allowed.
- Delete running session: reject with `409` and message to abort first.
- SSE reconnect on reopened running session: auto-subscribe.

---

## 11) Testing Plan

### Backend
1. Create run -> session appears in `GET /api/sessions`.
2. Restart server -> sessions persist.
3. Rename session -> reflected in list and detail.
4. Delete completed session -> removed from list, state file removed.
5. Delete running session -> `409`.
6. Corrupt index -> graceful recovery + backfill.
7. Reopen running session -> snapshot + live event stream.

### Frontend
1. Session list renders and sorts by updated time.
2. Open session updates URL and displayed run.
3. Rename/delete UI updates list and active selection correctly.
4. Browser close/reopen restores latest or URL-selected session.

### Regression
- Existing `/api/runs/*` flows remain functional.
- Report generation/download behavior unchanged.

---

## 12) Rollout

1. Add backend index + endpoints + reconciliation.
2. Keep dual API support (`runs` and `sessions`).
3. Update UI to use sessions panel and open behavior.
4. Validate restart/recovery and running-session reconnect.

---

## 13) Assumptions and Defaults

- Single-user deployment in V1 (`owner_id = null`).
- Session retention is manual-delete only.
- Session model is one-run-per-session for now.
- Report files are retained by default when deleting sessions.
- Pause/edit workflow logic is explicitly deferred; only scaffolding is included.
