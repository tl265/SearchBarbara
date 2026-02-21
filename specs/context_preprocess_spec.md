# Spec: Context Management UI + Incremental Context Preprocessing (Single Prompt)

Version: **context_digest.v1.4.1**  
Owner: Research Agent Platform  
Scope: **frontend UI + backend contracts + invariants**. Detailed extraction logic lives in the prompt file.

---

## 0) Prompt location (required wiring)

This spec intentionally does **not** encode parsing taxonomy; it references a single prompt file as the source of truth.

**Prompt files (must be loaded by runtime):**
- `context_preprocess_common.system.txt`
- `context_preprocess_per_file.system.txt`
- `context_preprocess_aggregate.system.txt`

Recommended runtime config:
- `CONTEXT_PREPROCESS_PROMPT_PATH_COMMON = "<repo>/prompts/context_preprocess_common.system.txt"`
- `CONTEXT_PREPROCESS_PROMPT_PATH_PER_FILE = "<repo>/prompts/context_preprocess_per_file.system.txt"`
- `CONTEXT_PREPROCESS_PROMPT_PATH_AGGREGATE = "<repo>/prompts/context_preprocess_aggregate.system.txt"`
- `CONTEXT_PREPROCESS_PROMPT_VERSION = "v1.4"`

The backend must record `prompt_version` alongside each per-file digest cache key.

---

## 1) Design principle (thin spec)

- This **spec** defines: UI requirements, API contracts, diff/caching rules, required JSON invariants, provenance rules, and acceptance tests.
- The **prompt file** defines: extraction taxonomy (facts/analyses/cases/…), reliability scoring heuristics, dedup rules, and aggregation behavior.

**Source of truth for parsing behavior = prompt files**  
Files: `context_preprocess_common.system.txt`, `context_preprocess_per_file.system.txt`, and `context_preprocess_aggregate.system.txt`

No web browsing; no external retrieval.

---

## 2) Goals

1) Users can **upload / replace / delete** context files for a research session.
2) After any change, the system automatically refreshes the **parsed context digest(s)** and shows them in the UI.
3) Save time & cost: do **incremental parsing** by detecting differences and skipping work when nothing changed.

---

## 3) Concepts & data model

### 3.1 ContextSet (per session)
- `context_set_id` (string)
- `session_id` (string)
- `revision` (int, monotonically increasing)
- `updated_at` (timestamp)
- `files[]` (ContextFile)
- `aggregate_digest_status` (`ready|stale|parsing|error`)
- `aggregate_digest_ref` (pointer/id to stored digest JSON)

### 3.2 ContextFile
- `file_id` (string; stable internal id)
- `filename` (string; recommended immutable after upload if used in citations)
- `mime_type` (string)
- `size_bytes` (int)
- `uploaded_at` (timestamp)
- `content_hash` (sha256 raw bytes)
- `extracted_text_hash` (sha256 extracted text)
- `chunking_version` (string)
- `spans_hash` (sha256 over ordered `(span_id,text)` pairs)
- `prompt_version` (string; version of the active context preprocess prompt file)
- `digest_status` (`ready|stale|parsing|error`)
- `digest_hash` (sha256 digest JSON)
- `digest_ref` (pointer/id to stored per-file digest JSON)

> If you must support file renames, prefer citations based on `file_id` rather than `filename`, or run a controlled “relabel citations” migration.

---

## 4) Input envelopes (single prompt, two tasks)

The same prompt file MUST support two invocation tasks (decided by the caller).

### 4.1 Task A: PER_FILE_DIGEST (from spans)
Caller provides exactly one file’s spans:

```
TASK: PER_FILE_DIGEST
FILE: <filename>
FORMAT: <format>
SOURCE_SPANS:
[S1] <text...>
[S2] <text...>
...
END_FILE
```

### 4.2 Task B: AGGREGATE_DIGEST (from per-file digests)
Caller provides a manifest + per-file digest JSON objects (no raw spans):

```
TASK: AGGREGATE_DIGEST
MANIFEST:
- filename: ...
  format: ...
- filename: ...
  format: ...

FILE_DIGESTS:
<DOC1_JSON>
<DOC2_JSON>
...
END_FILE_DIGESTS
```

> Aggregate MUST be produced from per-file digests to avoid reparsing unchanged files.

---

## 5) Output invariants (JSON)

### 5.1 Required keys (always)
The model MUST return valid JSON with:
- `schema_version` (string)
- `mode` = `"single"` or `"batch"`
- `summary` (string)
- `facts` (array)
- `uncertainties` (array)

Every `facts[i]` must include:
- `claim` (string)
- `sources` (non-empty array of SourceRef strings)

### 5.2 SourceRef format (provenance)
Each `sources[]` entry MUST be:
- `"<filename>::<span_id>"`

Optional readability suffix:
- `"<filename>::<span_id>::<locator_or_hint>"`

All SourceRefs must correspond to provided spans in Task A digests; in Task B, sources must be copied/merged from input digests.

### 5.3 Task-specific output
- Task A output: `mode="single"` and includes:
  - `document: { filename, format }`
- Task B output: `mode="batch"` and includes:
  - `document: { filename: "__BATCH__", format: "mixed" }`
  - `batch.files[]` from manifest
  - optional `documents[]` (pass-through or compacted per-file digests)

---

## 6) Incremental processing & diff detection (skip unchanged)

### 6.1 Per-file cache key
A per-file digest is uniquely determined by:
- `extracted_text_hash`
- `chunking_version`
- `prompt_version`

If all are unchanged, reuse cached digest (do not re-run Task A).

### 6.2 When to re-run PER_FILE_DIGEST
Re-run Task A only if:
- new file uploaded
- file replaced and `content_hash` changes
- extraction changes → `extracted_text_hash` changes
- chunker changes → `chunking_version` or `spans_hash` changes
- prompt changes → `prompt_version` changes

### 6.3 When to re-run AGGREGATE_DIGEST
Re-run Task B if:
- any file added/deleted
- any per-file digest_hash changes
- prompt_version changes (caller decides; recommended: tie to the same prompt file version)

---

## 7) Backend API contracts (minimal)

### 7.1 Read
- `GET /sessions/{session_id}/context` → ContextSet manifest (revision, files, statuses)
- `GET /sessions/{session_id}/context/files/{file_id}` → file metadata + preview URL (optional)
- `GET /sessions/{session_id}/context/files/{file_id}/digest` → per-file digest JSON
- `GET /sessions/{session_id}/context/digest` → aggregate digest JSON

### 7.2 Mutate (Upload / Replace / Delete)
All mutate endpoints:
- MUST be idempotent via `Idempotency-Key`
- SHOULD use optimistic concurrency via `If-Match: <revision>`

- `POST /sessions/{session_id}/context/files` (upload 1+ files)
- `PUT /sessions/{session_id}/context/files/{file_id}` (replace content)
- `DELETE /sessions/{session_id}/context/files/{file_id}` (delete)

Server behavior on mutate:
- increments `context_set.revision`
- computes hashes
- determines diff: `new|changed|unchanged|deleted`
- updates digest_status and triggers parsing workflow accordingly

### 7.3 Status updates
Either:
- SSE/WebSocket: `GET /sessions/{session_id}/context/stream`
- or polling `GET /sessions/{session_id}/context` while any status is `parsing|stale`

Statuses: `ready|parsing|stale|error` with optional error payload.

---

## 8) Frontend UI spec

### 8.1 Information architecture
Add a “Context” tab inside each research session.

**Left pane: Files**
- file list rows: filename, size, updated time, status chip (Ready/Parsing/Stale/Error/Unchanged)
- actions: Replace, Delete, Download, View digest
- primary CTA: Upload files
- toggle: Auto-parse on change (default ON)

**Right pane: Parsed context**
- tabs: Aggregate Digest (default), Selected File Digest, Raw Preview (optional)
- diff banner: “No changes detected—using cached digest” OR “+N new, ~M changed, -K deleted—parsing…”

### 8.2 Required interactions
- Upload (multi-select): shows progress; then updates list + triggers parse
- Replace: if hashes unchanged, show “Unchanged”; if changed, reparse only that file
- Delete: removes file; recompute aggregate only
- Retry parse: for error states
- Source click: jumps to Raw Preview and highlights the referenced span_id (if available)

---

## 9) Parsing workflow (reference)

1) On mutate, server computes diff + updates statuses.
2) For each `new|changed|stale` file:
   - extract → chunk → compute cache key
   - if cache miss: call LLM with **Task A** using `context_preprocess_per_file.system.txt`
   - validate JSON invariants
3) After per-file updates, if ContextSet membership or any digest changed:
   - call LLM with **Task B** using `context_preprocess_aggregate.system.txt`
   - validate JSON invariants
4) UI updates via stream/polling.

---

## 10) Acceptance tests

1) Upload identical file twice → server reports `unchanged`; no Task A call; aggregate unchanged.
2) Replace one file → only that file Task A; then Task B.
3) Delete one file → no Task A; Task B recomputed.
4) Multi-file upload → Task A only for new/changed; Task B once.
5) UI always shows digests consistent with latest `revision`.
6) All facts have non-empty sources with valid SourceRefs.
