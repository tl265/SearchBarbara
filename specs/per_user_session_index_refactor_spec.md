# Per-User Session Index + Locking Refactor Spec

## Summary
Refactor session metadata persistence from one global `runs/sessions_index.json` into per-user index files to reduce cross-user contention and large-file rewrites, while shrinking the scope of the current global runtime lock.
Keep Auth0-based ownership semantics unchanged; this is a storage/runtime concurrency refactor, not an auth refactor.

## Goals
1. Remove cross-user hot contention on global session index read/write paths.
2. Preserve current API behavior and response shapes.
3. Keep legacy ownerless sessions readable under `AUTH_ALLOW_LEGACY`.
4. Minimize lock hold time and avoid global lock during disk I/O.
5. Maintain crash safety and backward compatibility during migration.

## Non-Goals
1. No auth model changes (Auth0 flow unchanged).
2. No frontend API contract changes.
3. No changes to authoritative per-run checkpoint format (`state_web_<run_id>.json`) beyond optional metadata backfill if needed.

## Current-State Constraints (from code)
- Global index path: `runs/sessions_index.json`.
- Single writer thread + single `_index_write_lock` for all users.
- Session listing can fall back to index file when `_lock` is contended.
- `_sessions` in-memory cache is global and includes all owners.
- `_lock` currently guards many operations beyond tiny registry swaps.

## Design Overview

### 1) Index Sharding Model
Replace single global file with owner-scoped files:

- User-owned index: `runs/index/users/<owner_hash>.json`
- Legacy ownerless index: `runs/index/legacy.json` (for `owner_id=None`)
- Optional tiny manifest: `runs/index/manifest.json` for schema/version flags only.

`owner_hash`:
- Deterministic hash of `owner_id` (e.g., sha256 truncated), not raw user ID in filename.
- File payload still includes `owner_id` inside entries for validation/debug.

### 2) Index Payload Schema
Per index file payload:
- `schema_version`
- migration flags (if needed per file)
- `updated_at`
- `sessions: []` (only for that owner shard; or legacy shard)

Session item format remains same as current `SessionSummary`-compatible projection:
- `session_id`, `owner_id`, lifecycle fields, timestamps, report metadata, etc.

### 3) In-Memory Layout
Keep global in-memory maps but partition index cache by owner key:

- `_sessions_by_owner: Dict[str, Dict[session_id, session_item]]`
- owner key convention:
  - authenticated user: exact `owner_id`
  - legacy shard: `__legacy__`

Keep `_runs` keyed by `run_id` (authoritative runtime objects) unchanged.

### 4) Locking Model
Introduce explicit lock separation:

- `registry_lock` (existing `_lock`, narrowed use):
  - only for in-memory registry map insert/remove/lookups and pointer swaps.
  - no disk I/O under this lock.
- `session_lock(run_id)`:
  - authoritative for a run/session mutations.
- `owner_index_lock(owner_key)`:
  - per-owner index writer synchronization (or striped lock by owner hash).
- keep session lock ordering deterministic for multi-lock ops:
  - default order: `registry_lock` (short) -> `session_lock` -> release `registry_lock` ASAP.
  - cross-session ops lock by sorted key order.

### 5) Writer/Flush Strategy
Replace single global index writer pending payload with owner-scoped queueing:

- `_index_flush_pending_by_owner[owner_key] = latest_payload`
- Writer loop drains owners; writes each shard independently.
- Coalesce writes per owner (latest wins).
- Optional per-owner min flush interval (same current 2s concept) to throttle churn.

### 6) Read Paths
- `list_sessions(owner_id=...)`:
  - fast path: read in-memory owner shard snapshot under short `registry_lock`.
  - fallback path on lock contention: load only that owner’s index file (not all users).
  - if `AUTH_ALLOW_LEGACY` and owner set: merge user shard + `legacy.json`.
- `get_snapshot(run_id, owner_id)`:
  - unchanged ownership checks.
  - fallback loads by run state file remains authoritative.
  - once loaded, update corresponding owner shard cache and schedule owner-shard flush.

### 7) Legacy Compatibility / Migration
Startup bootstrap:
1. If old `runs/sessions_index.json` exists:
   - load once.
   - partition entries by owner into new shard files.
   - write shard files atomically.
   - preserve old file as `.bak` until migration completion marker.
2. Set migration marker in manifest to avoid repeat repartition.
3. During transition, read fallback order:
   - new owner shard(s) first
   - old global file only if new shards missing and migration not finalized.

No API downtime required.

### 8) Error Handling & Recovery
- If shard write fails:
  - keep in-memory state authoritative.
  - log owner key + error.
  - retry on next scheduled flush.
- If shard file corrupted:
  - return empty shard for listing fallback.
  - recover from in-memory cache and next flush.
- Keep current stale-nonterminal normalization in fallback list path (or equivalent), but scoped to shard load.

## Interfaces / Public Contract Impact
No external HTTP API changes expected.

Internal additions:
- owner-shard path helpers:
  - `_owner_index_key(owner_id) -> str`
  - `_owner_index_path(owner_key) -> Path`
- owner-scoped load/save:
  - `_load_sessions_index_for_owner(owner_key)`
  - `_save_sessions_index_for_owner_locked(owner_key, force=False)`
- owner-scoped writer queue/locks.

## Implementation Phases

### Phase 1: Plumbing + Dual-Write
1. Add owner-shard helpers + directories.
2. Build owner-scoped in-memory session cache.
3. On every session projection update, enqueue flush for owner shard.
4. Keep existing global index writer for safety (temporary dual-write).

Acceptance:
- APIs unchanged.
- Per-owner shard files created and updated.
- Existing behavior preserved.

### Phase 2: Read Switch
1. Switch `list_sessions` fallback to owner-shard reads.
2. Switch bootstrap to load owner shards first.
3. Keep global index fallback only for migration window.

Acceptance:
- No full global index read in normal path.
- Owner A listing unaffected by owner B churn.

### Phase 3: Migration Finalization
1. One-time repartition of old global index into owner shards.
2. Mark migration complete in manifest.
3. Remove normal dependency on global index; keep optional backup.

Acceptance:
- System starts correctly with only owner shards.
- Legacy ownerless sessions still visible under `AUTH_ALLOW_LEGACY`.

### Phase 4: Lock Scope Tightening
1. Audit `with self._lock` blocks:
   - remove disk I/O and heavy recompute from inside.
2. Keep only registry map operations under global lock.
3. Validate lock ordering and deadlock safety.

Acceptance:
- lower lock hold time.
- no lock timeout fallback spikes in normal load.

## Test Cases

### Functional
1. User A creates/updates/deletes sessions; only A shard changes.
2. User B simultaneously active; no writes to A shard from B operations.
3. `list_sessions(owner=A)` returns only A (+ legacy if allowed).
4. `AUTH_ALLOW_LEGACY=false`: ownerless sessions hidden.
5. `AUTH_ALLOW_LEGACY=true`: ownerless sessions visible to authenticated user as before.
6. `get_snapshot(run_id)` ownership enforcement unchanged.

### Concurrency
1. High-frequency events for User A do not delay User B list calls.
2. Parallel run updates for multiple users produce valid shard files (atomic writes).
3. No deadlock under concurrent session ops + workspace bind + report generation.

### Migration
1. Startup with only old global index migrates to shards.
2. Startup with partial shard files + old global index recovers deterministically.
3. Corrupted one shard file does not break other users.

### Performance/Observability
1. Measure lock hold times before/after (`manager_lock_held_ms` trends down).
2. List fallback rate under contention decreases.
3. Index write volume per file reduced (smaller payloads).

## Rollout Plan
1. Ship behind feature flag: `SESSIONS_INDEX_SHARDING_ENABLED=true`.
2. Stage in dev/staging with synthetic concurrent users.
3. Enable in prod with dual-write first.
4. After validation window, disable global index read/write path.
5. Keep one release rollback path:
   - feature flag off -> revert to global index behavior.

## Assumptions / Defaults
1. Keep `state_web_<run_id>.json` as authoritative source of truth.
2. Use hashed owner IDs in filenames by default.
3. Preserve `legacy` shard for ownerless sessions.
4. No external API schema changes.
5. Maintain current session summary fields unchanged.
