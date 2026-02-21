import json
import copy
import hashlib
import os
import threading
import uuid
from datetime import datetime, timezone
from glob import glob
from pathlib import Path
from queue import Queue
import time
from typing import Any, Dict, List, Optional

from app.models import RunState, SessionSummary
from deep_research_agent import (
    DeepResearchAgent,
    SYSTEM_REPORT,
    SubQuestionFinding,
    UsageTracker,
    slugify_for_filename,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class RunManager:
    _DEFAULT_HEARTBEAT_INTERVAL_SEC = 8.0
    _SESSIONS_INDEX_SCHEMA_VERSION = 1
    _MAX_EVENTS_PER_RUN = 2000
    _IDEMPOTENCY_TTL_CREATE_RUN_SEC = 600
    _IDEMPOTENCY_TTL_REPORT_SEC = 300

    def __init__(self, heartbeat_interval_sec: float = _DEFAULT_HEARTBEAT_INTERVAL_SEC) -> None:
        self._runs: Dict[str, RunState] = {}
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._subscribers: Dict[str, List[Queue]] = {}
        self._session_locks: Dict[str, threading.RLock] = {}
        self._session_locks_guard = threading.Lock()
        self._cancel_flags: Dict[str, bool] = {}
        self._pause_flags: Dict[str, bool] = {}
        self._active_report_runs: set[str] = set()
        self._idempotency_lock = threading.Lock()
        self._idempotency_records: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._index_write_lock = threading.Lock()
        self._index_flush_event = threading.Event()
        self._index_flush_seq = 0
        self._index_flush_pending: Optional[tuple[int, Dict[str, Any]]] = None
        self._index_writer_thread = threading.Thread(
            target=self._sessions_index_writer_loop,
            daemon=True,
        )
        self._index_writer_thread.start()
        self._lock_debug = str(os.getenv("RUN_MANAGER_LOCK_DEBUG", "0")).strip() in {"1", "true", "yes"}
        self._lock_wait_warn_ms = float(os.getenv("RUN_MANAGER_LOCK_WAIT_WARN_MS", "120"))
        self._lock_hold_warn_ms = float(os.getenv("RUN_MANAGER_LOCK_HOLD_WARN_MS", "150"))
        self._lock_owner_section: str = ""
        self._lock_owner_thread_id: int = 0
        self._lock_owner_acquired_at: float = 0.0
        self._heartbeat_interval_sec = max(1.0, float(heartbeat_interval_sec))
        self._index_flush_min_interval_sec = 2.0
        self._last_index_flush_ts = 0.0
        self._runs_dir = Path("runs")
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        self._sessions_index_path = self._runs_dir / "sessions_index.json"
        self._sessions_index_meta: Dict[str, Any] = {
            "schema_version": self._SESSIONS_INDEX_SCHEMA_VERSION,
            "title_migration_v1_done": False,
            "execution_state_migration_v1_done": False,
        }
        self._bootstrap_sessions_from_disk()

    def create_run(
        self,
        task: str,
        max_depth: int,
        max_rounds: int,
        results_per_query: int,
        model: str,
        report_model: str,
    ) -> str:
        run_id = str(uuid.uuid4())
        now = _now()
        state = RunState(
            run_id=run_id,
            session_id=run_id,
            owner_id=None,
            title=self._default_title_from_task(task),
            status="queued",
            version=1,
            execution_state="idle",
            created_at=now,
            updated_at=now,
            task=task,
            max_depth=max_depth,
            max_rounds=max_rounds,
            results_per_query=results_per_query,
            model=model,
            report_model=report_model,
            state_file_path=None,
            tree={
                "task": task,
                "plan": {"sub_questions": [], "success_criteria": []},
                "max_depth": max_depth,
                "rounds": [],
                "final_sufficiency": None,
                "report_status": "pending",
                "report_phase": "",
                "report_mode": "",
                "report_phase_updated_at": "",
                "report_error": "",
            },
        )
        with self._lock:
            self._refresh_lifecycle_fields_locked(state)
            self._runs[run_id] = state
            self._session_locks.setdefault(run_id, threading.RLock())
            self._subscribers[run_id] = []
            self._cancel_flags[run_id] = False
            self._pause_flags[run_id] = False
            self._sync_session_from_state_locked(state)
            self._save_sessions_index_locked(force=True)

        t = threading.Thread(
            target=self._execute_run,
            args=(run_id,),
            daemon=True,
        )
        t.start()
        return run_id

    def _idempotency_record_id(self, scope: str, key: str) -> str:
        return f"{str(scope or '').strip()}::{str(key or '').strip()}"

    def _idempotency_make_hash(self, payload: Dict[str, Any]) -> str:
        try:
            serialized = json.dumps(
                payload if isinstance(payload, dict) else {},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                default=str,
            )
        except Exception:
            serialized = "{}"
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _idempotency_gc_locked(self, now_ts: Optional[float] = None) -> None:
        now = float(now_ts if now_ts is not None else time.time())
        stale = [
            rid
            for rid, rec in self._idempotency_records.items()
            if float(rec.get("expires_at_ts", 0.0) or 0.0) <= now
        ]
        for rid in stale:
            self._idempotency_records.pop(rid, None)

    def idempotency_begin(
        self, scope: str, key: str, payload: Dict[str, Any], ttl_sec: int
    ) -> Dict[str, Any]:
        clean_key = str(key or "").strip()
        if not clean_key:
            return {"state": "new"}
        clean_scope = str(scope or "").strip()
        now = time.time()
        req_hash = self._idempotency_make_hash(payload)
        rid = self._idempotency_record_id(clean_scope, clean_key)
        with self._idempotency_lock:
            self._idempotency_gc_locked(now_ts=now)
            existing = self._idempotency_records.get(rid)
            if existing is None:
                self._idempotency_records[rid] = {
                    "scope": clean_scope,
                    "key": clean_key,
                    "request_hash": req_hash,
                    "status": "in_progress",
                    "created_at_ts": now,
                    "expires_at_ts": now + max(1, int(ttl_sec)),
                    "response_payload": None,
                }
                print(
                    f"[idempotency] scope={clean_scope} state=new key={clean_key[:12]}"
                )
                return {"state": "new"}
            if str(existing.get("request_hash", "")) != req_hash:
                print(
                    f"[idempotency] scope={clean_scope} state=mismatch key={clean_key[:12]}"
                )
                return {"state": "mismatch"}
            if str(existing.get("status", "")) == "completed":
                print(
                    f"[idempotency] scope={clean_scope} state=replay key={clean_key[:12]}"
                )
                return {
                    "state": "replay",
                    "payload": copy.deepcopy(existing.get("response_payload") or {}),
                }
            print(
                f"[idempotency] scope={clean_scope} state=in_progress key={clean_key[:12]}"
            )
            return {"state": "in_progress"}

    def idempotency_complete(
        self, scope: str, key: str, payload: Dict[str, Any], ttl_sec: int
    ) -> None:
        clean_key = str(key or "").strip()
        clean_scope = str(scope or "").strip()
        if not clean_key:
            return
        now = time.time()
        rid = self._idempotency_record_id(clean_scope, clean_key)
        with self._idempotency_lock:
            self._idempotency_gc_locked(now_ts=now)
            rec = self._idempotency_records.get(rid)
            if rec is None:
                return
            rec["status"] = "completed"
            rec["response_payload"] = copy.deepcopy(
                payload if isinstance(payload, dict) else {}
            )
            rec["expires_at_ts"] = now + max(1, int(ttl_sec))

    def idempotency_clear_in_progress(self, scope: str, key: str) -> None:
        clean_key = str(key or "").strip()
        clean_scope = str(scope or "").strip()
        if not clean_key:
            return
        rid = self._idempotency_record_id(clean_scope, clean_key)
        with self._idempotency_lock:
            rec = self._idempotency_records.get(rid)
            if rec and str(rec.get("status", "")) == "in_progress":
                self._idempotency_records.pop(rid, None)

    def _session_lock_for(self, run_id: str) -> threading.RLock:
        rid = str(run_id or "").strip()
        with self._session_locks_guard:
            return self._session_locks.setdefault(rid, threading.RLock())

    def _derive_research_state_locked(self, state: RunState) -> str:
        es = str(state.execution_state or "").strip().lower()
        st = str(state.status or "").strip().lower()
        if es == "paused":
            return "paused"
        if st in {"queued", "running"} or es == "running":
            return "running"
        return "terminal"

    def _derive_report_state_locked(self, state: RunState) -> str:
        tree = state.tree if isinstance(state.tree, dict) else {}
        status = str(tree.get("report_status", "") or "").strip().lower()
        phase = str(tree.get("report_phase", "") or "").strip().lower()
        if status == "running" or phase == "started":
            return "generating"
        if status in {"completed", "partial_ready"}:
            return "ready"
        if status == "failed":
            return "failed"
        return "idle"

    def _derive_terminal_reason_locked(self, state: RunState) -> str:
        research_state = self._derive_research_state_locked(state)
        if research_state != "terminal":
            return "none"
        err = str(state.error or "").strip()
        if err == "Run aborted by user":
            return "aborted"
        if err:
            return "failed"
        if str(state.status or "").strip().lower() == "completed":
            return "completed"
        return "none"

    def _refresh_lifecycle_fields_locked(self, state: RunState) -> None:
        state.research_state = self._derive_research_state_locked(state)
        state.report_state = self._derive_report_state_locked(state)
        state.terminal_reason = self._derive_terminal_reason_locked(state)

    def _next_state_version_locked(self, state: RunState) -> int:
        cur = max(1, int(getattr(state, "version", 1) or 1))
        state.version = cur + 1
        return state.version

    def _version_conflict_payload_locked(
        self, state: RunState, expected_version: Optional[int]
    ) -> Optional[Dict[str, Any]]:
        if expected_version is None:
            return None
        try:
            expected = int(expected_version)
        except Exception:
            expected = -1
        current = max(1, int(getattr(state, "version", 1) or 1))
        if expected == current:
            return None
        return {
            "error": "State version conflict. Refresh session and retry.",
            "error_code": "conflict_state_version",
            "expected_version": expected,
            "current_version": current,
        }

    def _ensure_run_loaded_locked(self, run_id: str) -> Optional[RunState]:
        state = self._runs.get(run_id)
        if state:
            return state
        session_item = self._sessions.get(run_id)
        if not session_item:
            loaded = self._load_sessions_index()
            for item in loaded.get("sessions", []):
                if isinstance(item, dict) and str(item.get("session_id", "")).strip() == run_id:
                    session_item = item
                    self._sessions[run_id] = item
                    break
        if not session_item:
            # Treat authoritative state checkpoint as source of truth even when
            # sessions_index is missing/stale.
            fallback_path = str(self._runs_dir / f"state_web_{run_id}.json")
            if Path(fallback_path).exists():
                session_item = {
                    "session_id": run_id,
                    "state_file_path": fallback_path,
                    "title": "Recovered session",
                    "owner_id": None,
                }
                self._sessions[run_id] = session_item
            else:
                return None
        state_path = str(session_item.get("state_file_path", "")).strip()
        state = self._load_run_state_from_files(run_id, state_path, session_item)
        if not state:
            return None
        recovered = self._recover_stale_report_generation_on_load_locked(state, run_id=run_id)
        self._runs[run_id] = state
        self._session_locks.setdefault(run_id, threading.RLock())
        self._subscribers.setdefault(run_id, [])
        self._cancel_flags.setdefault(run_id, False)
        self._pause_flags.setdefault(run_id, False)
        self._refresh_lifecycle_fields_locked(state)
        self._sync_session_from_state_locked(state)
        if recovered:
            try:
                self._persist_run_state_locked(state)
            except Exception:
                pass
            self._save_sessions_index_locked(force=True)
        return state

    def _ensure_run_loaded_session_locked(self, run_id: str) -> Optional[RunState]:
        state = self._runs.get(run_id)
        if state:
            return state

        session_item: Optional[Dict[str, Any]] = None
        with self._lock:
            state = self._runs.get(run_id)
            if state:
                return state
            item = self._sessions.get(run_id)
            if isinstance(item, dict):
                session_item = dict(item)

        if not session_item:
            loaded = self._load_sessions_index()
            for item in loaded.get("sessions", []):
                if isinstance(item, dict) and str(item.get("session_id", "")).strip() == run_id:
                    session_item = item
                    break
        if not session_item:
            fallback_path = str(self._runs_dir / f"state_web_{run_id}.json")
            if Path(fallback_path).exists():
                session_item = {
                    "session_id": run_id,
                    "state_file_path": fallback_path,
                    "title": "Recovered session",
                    "owner_id": None,
                }
            else:
                return None

        state_path = str(session_item.get("state_file_path", "")).strip()
        loaded_state = self._load_run_state_from_files(run_id, state_path, session_item)
        if not loaded_state:
            return None
        recovered = self._recover_stale_report_generation_on_load_locked(
            loaded_state, run_id=run_id
        )

        with self._lock:
            existing = self._runs.get(run_id)
            if existing:
                return existing
            self._runs[run_id] = loaded_state
            self._session_locks.setdefault(run_id, threading.RLock())
            self._subscribers.setdefault(run_id, [])
            self._cancel_flags.setdefault(run_id, False)
            self._pause_flags.setdefault(run_id, False)
            if run_id not in self._sessions:
                self._sessions[run_id] = dict(session_item)
            self._refresh_lifecycle_fields_locked(loaded_state)
            self._sync_session_from_state_locked(loaded_state)
            if recovered:
                self._save_sessions_index_locked(force=True)

        if recovered:
            try:
                self._persist_run_state_locked(loaded_state)
            except Exception:
                pass
        return loaded_state

    def _recover_stale_report_generation_on_load_locked(
        self, state: RunState, run_id: Optional[str] = None
    ) -> bool:
        # If a session is loaded from disk with report_state=generating, there is no active
        # in-memory report worker attached to this process for that historical task.
        # Recover the stuck marker to a terminal report state so users can trigger report again.
        rid = str(run_id or state.run_id or state.session_id or "").strip()
        if rid and rid in self._active_report_runs:
            return False
        if self._derive_report_state_locked(state) != "generating":
            return False
        if self._derive_research_state_locked(state) == "running":
            return False
        tree = state.tree if isinstance(state.tree, dict) else {}
        state.tree = tree
        mode = str(tree.get("report_mode", "") or "").strip()
        existing_err = str(tree.get("report_error", "") or "").strip()
        tree["report_status"] = "failed"
        tree["report_phase"] = "failed"
        tree["report_mode"] = mode
        tree["report_error"] = (
            existing_err
            or "Recovered stale report-generation state (no active report worker)."
        )
        tree["report_phase_updated_at"] = _now().isoformat()
        return True

    def _is_lock_locked(self, lock_obj: Any) -> bool:
        try:
            fn = getattr(lock_obj, "locked", None)
            if callable(fn):
                return bool(fn())
        except Exception:
            pass
        try:
            acquired = lock_obj.acquire(blocking=False)
        except Exception:
            return False
        if acquired:
            try:
                lock_obj.release()
            except Exception:
                pass
            return False
        return True

    def _lock_debug_for_session(self, run_id: str) -> Dict[str, Any]:
        rid = str(run_id or "").strip()
        with self._session_locks_guard:
            session_lock = self._session_locks.get(rid)
        return {
            "manager_lock": "locked" if self._is_lock_locked(self._lock) else "free",
            "manager_lock_owner_section": self._lock_owner_section or "",
            "manager_lock_owner_thread_id": int(self._lock_owner_thread_id or 0),
            "manager_lock_held_ms": (
                max(0.0, (time.perf_counter() - float(self._lock_owner_acquired_at or 0.0)) * 1000.0)
                if self._is_lock_locked(self._lock) and float(self._lock_owner_acquired_at or 0.0) > 0.0
                else 0.0
            ),
            "session_lock": (
                "missing"
                if session_lock is None
                else ("locked" if self._is_lock_locked(session_lock) else "free")
            ),
            "index_write_lock": "locked"
            if self._is_lock_locked(self._index_write_lock)
            else "free",
        }

    def list_sessions(self, include_lock_debug: bool = False) -> List[SessionSummary]:
        items: List[Dict[str, Any]]
        from_disk_fallback = False
        acquired, acquired_at = self._lock_acquire("list_sessions", timeout=0.5)
        if acquired:
            try:
                items = list(self._sessions.values())
            finally:
                self._lock_release("list_sessions", acquired_at)
        else:
            # Fallback path: avoid hanging API calls when lock is contended.
            loaded = self._load_sessions_index()
            raw_items = loaded.get("sessions", [])
            items = [it for it in raw_items if isinstance(it, dict)]
            from_disk_fallback = True

        # Disk fallback may contain stale non-terminal statuses from interrupted runs.
        # Normalize for listing so UI does not show "running" without a live worker.
        if from_disk_fallback:
            normalized: List[Dict[str, Any]] = []
            for it in items:
                fixed = dict(it)
                st = str(fixed.get("status", "")).strip().lower()
                if st in {"queued", "running"}:
                    fixed["status"] = "failed"
                    fixed["execution_state"] = "failed"
                normalized.append(fixed)
            items = normalized
        items.sort(
            key=lambda it: str(it.get("updated_at", "")),
            reverse=True,
        )
        out: List[SessionSummary] = []
        for it in items:
            try:
                summary = SessionSummary.model_validate(it)
                if include_lock_debug:
                    summary.lock_debug = self._lock_debug_for_session(summary.session_id)
                out.append(summary)
            except Exception:
                continue
        return out

    def rename_session(self, session_id: str, title: str) -> Optional[SessionSummary]:
        clean = " ".join(str(title or "").split()).strip()
        if not clean:
            return None
        if len(clean) > 200:
            clean = clean[:200]
        with self._lock:
            item = self._sessions.get(session_id)
            if not item:
                return None
            item["title"] = clean
            item["updated_at"] = _now().isoformat()
            state = self._runs.get(session_id)
            if state:
                state.title = clean
                state.updated_at = _now()
            self._save_sessions_index_locked(force=True)
            try:
                return SessionSummary.model_validate(item)
            except Exception:
                return None

    def delete_session(self, session_id: str) -> Optional[str]:
        with self._lock:
            state = self._runs.get(session_id)
            if state:
                status = str(getattr(state, "status", "") or "").strip().lower()
                execution_state = str(
                    getattr(state, "execution_state", "") or ""
                ).strip().lower()
                report_state = str(
                    getattr(state, "report_state", "") or ""
                ).strip().lower()
                report_worker_active = str(session_id or "").strip() in self._active_report_runs
                if (
                    status in {"queued", "running"}
                    or execution_state in {"running", "paused"}
                    or report_state == "generating"
                    or report_worker_active
                ):
                    return "conflict_running"
            sess = self._sessions.get(session_id)
            if not sess:
                return None
            state_file_path = str(sess.get("state_file_path", "")).strip()
            self._sessions.pop(session_id, None)
            self._runs.pop(session_id, None)
            self._session_locks.pop(session_id, None)
            self._subscribers.pop(session_id, None)
            self._cancel_flags.pop(session_id, None)
            self._pause_flags.pop(session_id, None)
            self._save_sessions_index_locked(force=True)
        if state_file_path:
            try:
                path = Path(state_file_path)
                if path.exists():
                    path.unlink()
            except Exception:
                pass
        return "deleted"

    def abort_run(
        self, run_id: str, expected_version: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        session_lock = self._session_lock_for(run_id)
        should_emit = False
        out_version = 1
        with session_lock:
            state = self._ensure_run_loaded_session_locked(run_id)
            if not state:
                return None
            conflict = self._version_conflict_payload_locked(state, expected_version)
            if conflict:
                return conflict
            # Allow cancellation during startup race: queued + idle still has a live worker
            # that has not emitted run_started yet.
            can_abort = (
                state.status in ("queued", "running")
                or state.execution_state in ("running", "paused")
            )
            if not can_abort:
                return {"status": state.status, "version": state.version}
            state.updated_at = _now()
            self._refresh_lifecycle_fields_locked(state)
            self._next_state_version_locked(state)
            out_version = max(1, int(state.version or 1))
            should_emit = True
            with self._lock:
                self._cancel_flags[run_id] = True
                self._pause_flags[run_id] = False
                self._sync_session_from_state_locked(state)
                self._save_sessions_index_locked(force=True)
        if should_emit:
            self._publish_event(
                run_id,
                {
                    "run_id": run_id,
                    "timestamp": _now().isoformat(),
                    "event_type": "run_abort_requested",
                    "payload": {},
                },
            )
            self._publish_event(
                run_id,
                {
                    "run_id": run_id,
                    "timestamp": _now().isoformat(),
                    "event_type": "abort_requested",
                    "payload": {},
                },
            )
        if should_emit:
            return {"status": "aborting", "version": out_version}
        return {"status": "aborting"}

    def pause_run(
        self, run_id: str, expected_version: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        session_lock = self._session_lock_for(run_id)
        with session_lock:
            state = self._ensure_run_loaded_session_locked(run_id)
            if not state:
                return None
            conflict = self._version_conflict_payload_locked(state, expected_version)
            if conflict:
                return conflict
            if state.execution_state != "running":
                return {"status": state.execution_state, "version": state.version}
            state.execution_state = "paused"
            state.updated_at = _now()
            self._refresh_lifecycle_fields_locked(state)
            self._next_state_version_locked(state)
            with self._lock:
                self._pause_flags[run_id] = True
                self._sync_session_from_state_locked(state)
                self._save_sessions_index_locked(force=True)
        return {"status": "paused", "version": max(1, int(state.version or 1))}

    def resume_run(
        self, run_id: str, expected_version: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        session_lock = self._session_lock_for(run_id)
        with session_lock:
            state = self._ensure_run_loaded_session_locked(run_id)
            if not state:
                return None
            conflict = self._version_conflict_payload_locked(state, expected_version)
            if conflict:
                return conflict
            if state.execution_state != "paused":
                return {"status": state.execution_state, "version": state.version}
            state.execution_state = "running"
            state.updated_at = _now()
            self._refresh_lifecycle_fields_locked(state)
            self._next_state_version_locked(state)
            with self._lock:
                self._pause_flags[run_id] = False
                self._sync_session_from_state_locked(state)
                self._save_sessions_index_locked(force=True)
        return {"status": "running", "version": max(1, int(state.version or 1))}

    def generate_partial_report(
        self, run_id: str, expected_version: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        session_lock = self._session_lock_for(run_id)
        snapshot: Optional[RunState] = None
        report_mode = "partial"
        start_persist_error = ""
        acquired = session_lock.acquire(timeout=2.0)
        if not acquired:
            return {
                "error": "Session is busy. Please retry report generation in a moment.",
                "error_code": "busy",
            }
        try:
            state = self._ensure_run_loaded_session_locked(run_id)
            if not state:
                return None
            conflict = self._version_conflict_payload_locked(state, expected_version)
            if conflict:
                return conflict
            self._refresh_lifecycle_fields_locked(state)
            if state.report_state == "generating":
                # Defensive stale-state recovery: if no active in-process report
                # worker is registered for this run, treat lingering "generating"
                # as stale and recover instead of hard-blocking with 409.
                recovered = self._recover_stale_report_generation_on_load_locked(
                    state, run_id=run_id
                )
                if recovered:
                    self._refresh_lifecycle_fields_locked(state)
                    try:
                        self._persist_run_state_locked(state)
                    except Exception:
                        pass
                    with self._lock:
                        self._sync_session_from_state_locked(state)
                        self._save_sessions_index_locked(force=True)
                else:
                    return {
                        "error": "Report generation is already in progress.",
                        "error_code": "conflict_report_running",
                    }
            if state.research_state not in {"paused", "terminal"}:
                return {
                    "error": (
                        "Report generation is only allowed when research is paused "
                        "or has reached a terminal state."
                    ),
                    "error_code": "invalid_state",
                }
            report_mode = (
                "partial" if state.research_state == "paused" else "final_from_snapshot"
            )
            self._set_report_phase_locked(
                state,
                report_status="running",
                report_phase="started",
                report_mode=report_mode,
                report_error="",
            )
            self._next_state_version_locked(state)
            self._active_report_runs.add(run_id)
            self._refresh_lifecycle_fields_locked(state)
            snapshot = RunState.model_validate(self._snapshot_payload_locked(state))
            try:
                self._persist_run_state_locked(state)
            except Exception as exc:
                start_persist_error = (
                    "Failed to persist run checkpoint when report generation started: "
                    f"{exc}"
                )
            with self._lock:
                self._sync_session_from_state_locked(state)
                self._save_sessions_index_locked(force=True)
        finally:
            session_lock.release()
        if not snapshot:
            with self._lock:
                self._active_report_runs.discard(run_id)
            return None
        if start_persist_error:
            with self._lock:
                self._active_report_runs.discard(run_id)
            self._publish_event(
                run_id,
                {
                    "run_id": run_id,
                    "timestamp": _now().isoformat(),
                    "event_type": "report_generation_failed",
                    "payload": {"mode": report_mode, "error": start_persist_error},
                },
            )
            return {"error": start_persist_error, "error_code": "persist_failed"}
        base_usage = self._best_effort_usage_snapshot(snapshot)
        self._publish_event(
            run_id,
            {
                "run_id": run_id,
                "timestamp": _now().isoformat(),
                "event_type": "report_generation_started",
                "payload": {"mode": report_mode},
            },
        )
        report_heartbeat_stop = threading.Event()

        def report_heartbeat_worker() -> None:
            while not report_heartbeat_stop.wait(self._heartbeat_interval_sec):
                self._publish_event(
                    run_id,
                    {
                        "run_id": run_id,
                        "timestamp": _now().isoformat(),
                        "event_type": "report_heartbeat",
                        "payload": {
                            "phase": "writing_report",
                            "last_event_type": "report_generation_started",
                            "phase_updated_at": _now().isoformat(),
                        },
                    },
                )

        report_heartbeat_thread = threading.Thread(
            target=report_heartbeat_worker, daemon=True
        )
        report_heartbeat_thread.start()
        try:
            report_text, partial_usage = self._build_partial_report_with_agent(
                snapshot, report_mode
            )
        except Exception:
            with self._lock:
                self._active_report_runs.discard(run_id)
            raise
        finally:
            report_heartbeat_stop.set()
            report_heartbeat_thread.join(timeout=1.0)
        if not report_text.strip():
            with self._lock:
                self._active_report_runs.discard(run_id)
            self._publish_event(
                run_id,
                {
                    "run_id": run_id,
                    "timestamp": _now().isoformat(),
                    "event_type": "report_generation_failed",
                    "payload": {"mode": report_mode},
                },
            )
            return {
                "error": "Partial report generation failed (report agent unavailable).",
                "error_code": "generation_failed",
            }
        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        slug = slugify_for_filename(snapshot.task)
        report_path = reports_dir / f"report_{slug}_{ts}.md"
        try:
            report_path.write_text(report_text, encoding="utf-8")
        except Exception:
            with self._lock:
                self._active_report_runs.discard(run_id)
            raise

        persist_error = ""
        with session_lock:
            state = self._ensure_run_loaded_session_locked(run_id)
            if not state:
                self._active_report_runs.discard(run_id)
                return None
            prev_report_versions = copy.deepcopy(
                state.report_versions if isinstance(state.report_versions, list) else []
            )
            prev_current_idx = state.current_report_version_index
            prev_report_text = state.report_text
            prev_report_file_path = state.report_file_path
            prev_token_usage = copy.deepcopy(state.token_usage)
            prev_report_status = (
                state.tree.get("report_status", "pending")
                if isinstance(state.tree, dict)
                else "pending"
            )
            prev_report_phase = (
                state.tree.get("report_phase", "") if isinstance(state.tree, dict) else ""
            )
            prev_report_mode = (
                state.tree.get("report_mode", "") if isinstance(state.tree, dict) else ""
            )
            prev_report_error = (
                state.tree.get("report_error", "") if isinstance(state.tree, dict) else ""
            )
            prev_report_phase_updated_at = (
                state.tree.get("report_phase_updated_at", "")
                if isinstance(state.tree, dict)
                else ""
            )
            prev_updated_at = state.updated_at
            live_usage = state.token_usage if isinstance(state.token_usage, dict) else None
            usage_base = live_usage or base_usage
            if partial_usage:
                state.token_usage = self._merge_usage_dicts(
                    usage_base, partial_usage
                )
            elif usage_base:
                state.token_usage = usage_base
            created = self._append_report_version_locked(
                state=state,
                report_text=report_text,
                report_file_path=str(report_path),
                trigger_type="manual",
                usage_snapshot=partial_usage,
            )
            self._set_report_phase_locked(
                state,
                report_status="completed",
                report_phase="completed",
                report_mode=report_mode,
                report_error="",
            )
            self._next_state_version_locked(state)
            self._refresh_lifecycle_fields_locked(state)
            try:
                self._persist_run_state_locked(state)
            except Exception as exc:
                state.report_versions = prev_report_versions
                state.current_report_version_index = prev_current_idx
                state.report_text = prev_report_text
                state.report_file_path = prev_report_file_path
                state.token_usage = prev_token_usage
                state.updated_at = prev_updated_at
                state.tree["report_status"] = prev_report_status
                state.tree["report_phase"] = prev_report_phase
                state.tree["report_mode"] = prev_report_mode
                state.tree["report_error"] = prev_report_error
                state.tree["report_phase_updated_at"] = prev_report_phase_updated_at
                self._refresh_lifecycle_fields_locked(state)
                persist_error = (
                    "Failed to persist run checkpoint while generating report: "
                    f"{exc}"
                )
            with self._lock:
                self._sync_session_from_state_locked(state)
                self._save_sessions_index_locked(force=True)
            self._active_report_runs.discard(run_id)
        if persist_error:
            self._publish_event(
                run_id,
                {
                    "run_id": run_id,
                    "timestamp": _now().isoformat(),
                    "event_type": "report_generation_failed",
                    "payload": {"mode": report_mode, "error": persist_error},
                },
            )
            return {"error": persist_error, "error_code": "persist_failed"}

        self._publish_event(
            run_id,
            {
                "run_id": run_id,
                "timestamp": _now().isoformat(),
                "event_type": "partial_report_generated",
                "payload": {
                    "report_file_path": str(report_path),
                    "version_index": int(created.get("version_index", 1)),
                    "mode": report_mode,
                },
            },
        )
        self._publish_event(
            run_id,
            {
                "run_id": run_id,
                "timestamp": _now().isoformat(),
                "event_type": "report_generation_completed",
                "payload": {
                    "report_file_path": str(report_path),
                    "version_index": int(created.get("version_index", 1)),
                    "mode": report_mode,
                },
            },
        )
        return {
            "report_file_path": str(report_path),
            "version_index": int(created.get("version_index", 1)),
            "version": max(1, int(state.version or 1)),
        }

    def select_report_version(
        self, run_id: str, version_index: int, expected_version: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        session_lock = self._session_lock_for(run_id)
        with session_lock:
            state = self._ensure_run_loaded_session_locked(run_id)
            if not state:
                return None
            conflict = self._version_conflict_payload_locked(state, expected_version)
            if conflict:
                return conflict
            versions = state.report_versions if isinstance(state.report_versions, list) else []
            if not versions:
                return {}
            idx = max(1, int(version_index))
            if idx > len(versions):
                idx = len(versions)
            prev_idx = state.current_report_version_index
            prev_report_text = state.report_text
            prev_report_file_path = state.report_file_path
            prev_updated_at = state.updated_at
            selected = versions[idx - 1]
            state.current_report_version_index = idx
            state.report_text = str(selected.get("report_text", ""))
            state.report_file_path = str(selected.get("report_file_path", "") or "") or None
            state.updated_at = _now()
            self._next_state_version_locked(state)
            self._refresh_lifecycle_fields_locked(state)
            try:
                self._persist_run_state_locked(state)
            except Exception as exc:
                state.current_report_version_index = prev_idx
                state.report_text = prev_report_text
                state.report_file_path = prev_report_file_path
                state.updated_at = prev_updated_at
                return {
                    "error": (
                        "Failed to persist run checkpoint while selecting report version: "
                        f"{exc}"
                    )
                }
            with self._lock:
                self._sync_session_from_state_locked(state)
                self._save_sessions_index_locked(force=True)
        self._publish_event(
            run_id,
            {
                "run_id": run_id,
                "timestamp": _now().isoformat(),
                "event_type": "report_version_selected",
                "payload": {"version_index": idx},
            },
        )
        return {"version_index": idx, "version": max(1, int(state.version or 1))}

    def _append_report_version_locked(
        self,
        state: RunState,
        report_text: str,
        report_file_path: str,
        trigger_type: str,
        usage_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        versions = state.report_versions if isinstance(state.report_versions, list) else []
        next_idx = len(versions) + 1
        version = {
            "version_index": next_idx,
            "created_at": _now().isoformat(),
            "trigger_type": trigger_type,
            "report_text": report_text,
            "report_file_path": report_file_path,
            "token_usage": usage_snapshot if isinstance(usage_snapshot, dict) else None,
        }
        versions.append(version)
        state.report_versions = versions
        state.current_report_version_index = next_idx
        state.report_text = report_text
        state.report_file_path = report_file_path
        return version

    def get_snapshot(self, run_id: str) -> Optional[RunState]:
        session_lock = self._session_lock_for(run_id)
        acquired_session = session_lock.acquire(timeout=0.5)
        if acquired_session:
            try:
                state = self._ensure_run_loaded_session_locked(run_id)
                if state:
                    self._refresh_lifecycle_fields_locked(state)
                    payload = self._snapshot_payload_locked(state)
                    return RunState.model_validate(payload)
                session_item = None
            finally:
                session_lock.release()
        else:
            session_item = None

        loaded = self._load_sessions_index()
        for item in loaded.get("sessions", []):
            if isinstance(item, dict) and str(item.get("session_id", "")).strip() == run_id:
                session_item = item
                break
        if not session_item:
            state_path = str(self._runs_dir / f"state_web_{run_id}.json")
            if not Path(state_path).exists():
                return None
            session_item = {
                "session_id": run_id,
                "state_file_path": state_path,
                "title": "Recovered session",
                "owner_id": None,
            }
        else:
            state_path = str(session_item.get("state_file_path", "")).strip()
        state = self._load_run_state_from_files(run_id, state_path, session_item)
        if state:
            self._recover_stale_report_generation_on_load_locked(state, run_id=run_id)
            state.research_state = self._derive_research_state_locked(state)
            state.report_state = self._derive_report_state_locked(state)
            state.terminal_reason = self._derive_terminal_reason_locked(state)
        return state

    def subscribe(self, run_id: str) -> Optional[Queue]:
        session_lock = self._session_lock_for(run_id)
        with session_lock:
            state = self._ensure_run_loaded_session_locked(run_id)
            if not state:
                return None
            with self._lock:
                q: Queue = Queue()
                self._subscribers[run_id].append(q)
                return q

    def unsubscribe(self, run_id: str, queue: Queue) -> None:
        session_lock = self._session_lock_for(run_id)
        with session_lock:
            self._ensure_run_loaded_session_locked(run_id)
            with self._lock:
                subs = self._subscribers.get(run_id, [])
                if queue in subs:
                    subs.remove(queue)

    def _publish_event(self, run_id: str, event: Dict[str, Any]) -> None:
        session_lock = self._session_lock_for(run_id)
        subscribers: List[Queue] = []
        with session_lock:
            state = self._ensure_run_loaded_session_locked(run_id)
            if not state:
                return
            event_type = str(event.get("event_type", ""))
            if event_type in ("run_abort_requested", "abort_requested") and state.status in (
                "completed",
                "failed",
            ):
                return
            state.events.append(event)
            if len(state.events) > self._MAX_EVENTS_PER_RUN:
                del state.events[:-self._MAX_EVENTS_PER_RUN]
            state.updated_at = _now()
            self._project_event(state, event)
            self._next_state_version_locked(state)
            self._refresh_lifecycle_fields_locked(state)

            if event_type in {
                "report_generation_started",
                "report_generation_completed",
                "report_generation_failed",
                "partial_report_generated",
                "report_version_created",
                "report_version_selected",
            }:
                try:
                    self._persist_run_state_locked(state)
                except Exception as exc:
                    print(f"[state] report checkpoint persistence failed: {exc}")

            with self._lock:
                self._sync_session_from_state_locked(state)
                if self._should_flush_sessions_index(event_type):
                    self._save_sessions_index_locked(force=True)
                subscribers = list(self._subscribers.get(run_id, []))
        for q in subscribers:
            try:
                q.put(event)
            except Exception:
                # Best-effort fan-out: don't block run progress on subscriber failures.
                pass

    def _lock_acquire(self, section: str, timeout: Optional[float] = None) -> tuple[bool, float]:
        t0 = time.perf_counter()
        if timeout is None:
            acquired = self._lock.acquire()
        else:
            acquired = self._lock.acquire(timeout=timeout)
        wait_ms = (time.perf_counter() - t0) * 1000.0
        if acquired:
            self._lock_owner_section = str(section or "")
            self._lock_owner_thread_id = int(threading.get_ident())
            self._lock_owner_acquired_at = time.perf_counter()
        if self._lock_debug and wait_ms >= self._lock_wait_warn_ms:
            print(f"[lock] wait {wait_ms:.1f}ms section={section}")
        return acquired, time.perf_counter()

    def _lock_release(self, section: str, acquired_at: Optional[float] = None) -> None:
        hold_ms = 0.0
        if acquired_at is not None:
            hold_ms = (time.perf_counter() - acquired_at) * 1000.0
        if self._lock_debug and acquired_at is not None and hold_ms >= self._lock_hold_warn_ms:
            print(f"[lock] hold {hold_ms:.1f}ms section={section}")
        self._lock_owner_section = ""
        self._lock_owner_thread_id = 0
        self._lock_owner_acquired_at = 0.0
        self._lock.release()

    def _snapshot_payload_locked(self, state: RunState) -> Dict[str, Any]:
        # Build a copy outside pydantic model_dump path to keep lock hold predictable.
        # `events` are trimmed to tail because UI only uses latest thought.
        events_tail = [state.events[-1]] if state.events else []
        return {
            "run_id": state.run_id,
            "session_id": state.session_id,
            "owner_id": state.owner_id,
            "title": state.title,
            "status": state.status,
            "version": max(1, int(getattr(state, "version", 1) or 1)),
            "execution_state": state.execution_state,
            "research_state": state.research_state,
            "report_state": state.report_state,
            "terminal_reason": state.terminal_reason,
            "created_at": state.created_at,
            "updated_at": state.updated_at,
            "last_checkpoint_at": state.last_checkpoint_at,
            "has_manual_edits": state.has_manual_edits,
            "task": state.task,
            "max_depth": state.max_depth,
            "max_rounds": state.max_rounds,
            "results_per_query": state.results_per_query,
            "model": state.model,
            "report_model": state.report_model,
            "tree": copy.deepcopy(state.tree),
            "events": copy.deepcopy(events_tail),
            "report_text": state.report_text,
            "report_file_path": state.report_file_path,
            "report_versions": copy.deepcopy(state.report_versions),
            "current_report_version_index": state.current_report_version_index,
            "state_file_path": state.state_file_path,
            "error": state.error,
            "token_usage": copy.deepcopy(state.token_usage),
            "manual_edit_log": copy.deepcopy(state.manual_edit_log),
            "manual_assertions": copy.deepcopy(state.manual_assertions),
        }

    def _persist_run_state_locked(self, state: RunState) -> None:
        state_file_path = str(state.state_file_path or "").strip()
        if not state_file_path:
            return
        path = Path(state_file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        existing: Dict[str, Any] = {}
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    existing = raw
            except Exception:
                existing = {}

        existing["version"] = max(
            1, int(getattr(state, "version", existing.get("version", 1)) or 1)
        )
        existing["status"] = state.status
        existing["research_state"] = str(state.research_state or "")
        existing["report_state"] = str(state.report_state or "")
        existing["terminal_reason"] = str(state.terminal_reason or "")
        existing["task"] = state.task
        existing["model"] = state.model
        existing["decompose_model"] = state.report_model
        existing["report_model"] = state.report_model
        existing["started_at"] = (
            str(existing.get("started_at", "") or "") or state.created_at.isoformat()
        )
        existing["updated_at"] = state.updated_at.isoformat()
        existing["error"] = state.error or ""
        tree = state.tree if isinstance(state.tree, dict) else {}
        existing["report_status"] = str(tree.get("report_status", "pending") or "pending")
        existing["report_phase"] = str(tree.get("report_phase", "") or "")
        existing["report_mode"] = str(tree.get("report_mode", "") or "")
        existing["report_phase_updated_at"] = str(tree.get("report_phase_updated_at", "") or "")
        existing["report_error"] = str(tree.get("report_error", "") or "")
        existing["token_usage"] = (
            copy.deepcopy(state.token_usage)
            if isinstance(state.token_usage, dict)
            else {}
        )

        trace = existing.get("trace", {})
        if not isinstance(trace, dict):
            trace = {}
        trace["task"] = state.task
        trace["research_state"] = str(state.research_state or "")
        trace["report_state"] = str(state.report_state or "")
        trace["terminal_reason"] = str(state.terminal_reason or "")
        trace["model"] = state.model
        trace["decompose_model"] = state.report_model
        trace["report_model"] = state.report_model
        trace["max_depth"] = state.max_depth
        trace["max_rounds"] = state.max_rounds
        trace["results_per_query"] = state.results_per_query
        trace["rounds"] = copy.deepcopy(tree.get("rounds", []))
        trace["final_sufficiency"] = copy.deepcopy(tree.get("final_sufficiency", None))
        trace["report"] = state.report_text or ""
        trace["report_status"] = str(tree.get("report_status", "pending") or "pending")
        trace["report_phase"] = str(tree.get("report_phase", "") or "")
        trace["report_mode"] = str(tree.get("report_mode", "") or "")
        trace["report_phase_updated_at"] = str(tree.get("report_phase_updated_at", "") or "")
        trace["report_error"] = str(tree.get("report_error", "") or "")
        trace["report_versions"] = copy.deepcopy(
            state.report_versions if isinstance(state.report_versions, list) else []
        )
        trace["current_report_version_index"] = state.current_report_version_index
        trace["token_usage"] = (
            copy.deepcopy(state.token_usage)
            if isinstance(state.token_usage, dict)
            else {}
        )
        existing["trace"] = trace

        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def _project_event(self, state: RunState, event: Dict[str, Any]) -> None:
        et = event.get("event_type", "")
        payload = event.get("payload", {})
        tree = state.tree

        if et == "run_started":
            state.status = "running"
            state.execution_state = "running"
            self._set_report_phase_locked(
                state,
                report_status="pending",
                report_phase="",
                report_mode="",
                report_error="",
            )
            return
        if et == "run_paused":
            if state.status == "running":
                state.execution_state = "paused"
            return
        if et == "run_resumed":
            if state.status == "running":
                state.execution_state = "running"
            return
        if et == "plan_created":
            tree["plan"] = {
                "sub_questions": payload.get("sub_questions", []),
                "success_criteria": payload.get("success_criteria", []),
            }
            return
        if et == "round_started":
            round_i = int(payload.get("round", 0))
            self._ensure_round(tree, round_i)
            return
        if et == "sub_question_started":
            round_i = int(payload.get("round", 0))
            sq = str(payload.get("sub_question", ""))
            depth = int(payload.get("depth", 1))
            parent = str(payload.get("parent", ""))
            node_id = str(payload.get("node_id", ""))
            rnd = self._ensure_round(tree, round_i)
            qnode = self._ensure_question(rnd, sq)
            qnode["depth"] = depth
            qnode["parent"] = parent
            qnode["status"] = "running"
            if node_id:
                qnode["node_id"] = node_id
            return
        if et == "queries_generated":
            round_i = int(payload.get("round", 0))
            sq = str(payload.get("sub_question", ""))
            queries = payload.get("queries", [])
            rnd = self._ensure_round(tree, round_i)
            qnode = self._ensure_question(rnd, sq)
            if "depth" in payload:
                qnode["depth"] = int(payload.get("depth", 1))
            for query in queries:
                self._ensure_query_step(qnode, str(query))
            return
        if et == "query_started":
            round_i = int(payload.get("round", 0))
            sq = str(payload.get("sub_question", ""))
            query = str(payload.get("query", ""))
            rnd = self._ensure_round(tree, round_i)
            qnode = self._ensure_question(rnd, sq)
            step = self._ensure_query_step(qnode, query)
            step["status"] = "running"
            return
        if et == "query_diagnostic":
            round_i = int(payload.get("round", 0))
            sq = str(payload.get("sub_question", ""))
            query = str(payload.get("query", ""))
            rnd = self._ensure_round(tree, round_i)
            qnode = self._ensure_question(rnd, sq)
            step = self._ensure_query_step(qnode, query)
            step["diagnostic"] = {
                "classification": str(payload.get("classification", "")),
                "prior_query": str(payload.get("prior_query", "")),
                "similarity": float(payload.get("similarity", 0.0)),
                "raw_intent_key": str(payload.get("raw_intent_key", "")),
                "execution_intent_key": str(payload.get("execution_intent_key", "")),
                "intent_mapped": bool(payload.get("intent_mapped", False)),
                "intent_map_similarity": float(payload.get("intent_map_similarity", 0.0)),
                "new_tokens": payload.get("new_tokens", []),
                "dropped_tokens": payload.get("dropped_tokens", []),
                "is_broadened": bool(payload.get("is_broadened", False)),
                "base_k": int(payload.get("base_k", 0)),
                "effective_k": int(payload.get("effective_k", 0)),
                "decision": str(payload.get("decision", "")),
                "explicit_recheck": bool(payload.get("explicit_recheck", False)),
            }
            return
        if et == "query_skipped_cached":
            round_i = int(payload.get("round", 0))
            sq = str(payload.get("sub_question", ""))
            query = str(payload.get("query", ""))
            rnd = self._ensure_round(tree, round_i)
            qnode = self._ensure_question(rnd, sq)
            step = self._ensure_query_step(qnode, query)
            step["status"] = "cached"
            step["cache_hit"] = True
            step["rerun_reason"] = "cache_hit"
            return
        if et == "query_blocked_diminishing_returns":
            round_i = int(payload.get("round", 0))
            sq = str(payload.get("sub_question", ""))
            query = str(payload.get("query", ""))
            rnd = self._ensure_round(tree, round_i)
            qnode = self._ensure_question(rnd, sq)
            step = self._ensure_query_step(qnode, query)
            step["status"] = "blocked"
            step["blocked_by_diminishing_returns"] = True
            step["rerun_reason"] = "blocked_diminishing_returns"
            return
        if et == "query_rerun_allowed":
            round_i = int(payload.get("round", 0))
            sq = str(payload.get("sub_question", ""))
            query = str(payload.get("query", ""))
            rnd = self._ensure_round(tree, round_i)
            qnode = self._ensure_question(rnd, sq)
            step = self._ensure_query_step(qnode, query)
            step["rerun_reason"] = payload.get("reason")
            return
        if et == "query_broadened":
            round_i = int(payload.get("round", 0))
            sq = str(payload.get("sub_question", ""))
            query = str(payload.get("query", ""))
            rnd = self._ensure_round(tree, round_i)
            qnode = self._ensure_question(rnd, sq)
            step = self._ensure_query_step(qnode, query)
            step["effective_results_per_query"] = int(
                payload.get("effective_results_per_query", 0)
            )
            step["broadening_step"] = int(payload.get("broadening_step", 0))
            return
        if et == "search_completed":
            round_i = int(payload.get("round", 0))
            sq = str(payload.get("sub_question", ""))
            query = str(payload.get("query", ""))
            rnd = self._ensure_round(tree, round_i)
            qnode = self._ensure_question(rnd, sq)
            step = self._ensure_query_step(qnode, query)
            step["search_error"] = payload.get("search_error")
            step["selected_results_count"] = int(payload.get("selected_results_count", 0))
            step["primary_count"] = int(payload.get("primary_count", 0))
            selected_sources = payload.get("selected_sources", [])
            if isinstance(selected_sources, list):
                cleaned_sources = []
                for item in selected_sources:
                    if not isinstance(item, dict):
                        continue
                    url = str(item.get("url", "")).strip()
                    title = str(item.get("title", "")).strip()
                    if not url:
                        continue
                    cleaned_sources.append({"title": title, "url": url})
                step["selected_sources"] = cleaned_sources
            else:
                step["selected_sources"] = []
            if step["selected_results_count"] == 0:
                step["status"] = "skipped"
            return
        if et == "synthesis_completed":
            round_i = int(payload.get("round", 0))
            sq = str(payload.get("sub_question", ""))
            query = str(payload.get("query", ""))
            rnd = self._ensure_round(tree, round_i)
            qnode = self._ensure_question(rnd, sq)
            step = self._ensure_query_step(qnode, query)
            step["status"] = "success"
            step["synthesis_summary"] = payload.get("summary", "")
            return
        if et == "sufficiency_completed":
            round_i = int(payload.get("round", 0))
            rnd = self._ensure_round(tree, round_i)
            rnd["sufficiency"] = payload
            tree["final_sufficiency"] = payload
            return
        if et == "node_decomposed":
            round_i = int(payload.get("round", 0))
            sq = str(payload.get("sub_question", ""))
            rnd = self._ensure_round(tree, round_i)
            qnode = self._ensure_question(rnd, sq)
            qnode["children"] = payload.get("children", [])
            child_node_ids = payload.get("child_node_ids", [])
            qnode["child_node_ids"] = (
                child_node_ids if isinstance(child_node_ids, list) else []
            )
            qnode["status"] = "decomposed"
            return
        if et == "node_decomposition_started":
            round_i = int(payload.get("round", 0))
            sq = str(payload.get("sub_question", ""))
            rnd = self._ensure_round(tree, round_i)
            qnode = self._ensure_question(rnd, sq)
            qnode["status"] = "decomposing"
            return
        if et == "node_sufficiency_completed":
            round_i = int(payload.get("round", 0))
            sq = str(payload.get("sub_question", ""))
            rnd = self._ensure_round(tree, round_i)
            qnode = self._ensure_question(rnd, sq)
            qnode["node_sufficiency"] = {
                "is_sufficient": bool(payload.get("is_sufficient", False)),
                "reasoning": str(payload.get("reasoning", "")),
                "gaps": payload.get("gaps", []),
            }
            return
        if et == "node_completed":
            round_i = int(payload.get("round", 0))
            sq = str(payload.get("sub_question", ""))
            rnd = self._ensure_round(tree, round_i)
            qnode = self._ensure_question(rnd, sq)
            if bool(payload.get("derived_from_children", False)):
                qnode["status"] = "solved_via_children"
            else:
                qnode["status"] = "solved"
            return
        if et == "node_unresolved":
            round_i = int(payload.get("round", 0))
            sq = str(payload.get("sub_question", ""))
            rnd = self._ensure_round(tree, round_i)
            qnode = self._ensure_question(rnd, sq)
            qnode["status"] = "unresolved"
            qnode["unresolved_reason"] = str(payload.get("reason", ""))
            return
        if et == "run_completed":
            state.status = "completed"
            state.execution_state = "completed"
            self._set_report_phase_locked(
                state,
                report_status="completed",
                report_phase="completed",
                report_mode=str(tree.get("report_mode", "") or ""),
                report_error="",
            )
            return
        if et == "run_failed":
            state.status = "failed"
            state.execution_state = "failed"
            state.error = str(payload.get("error", "unknown error"))
            self._set_report_phase_locked(
                state,
                report_status="failed",
                report_phase="failed",
                report_mode=str(tree.get("report_mode", "") or ""),
                report_error=state.error,
            )
            return
        if et == "run_aborted":
            state.status = "failed"
            state.execution_state = "aborted"
            state.error = str(payload.get("error", "Run aborted by user"))
            self._set_report_phase_locked(
                state,
                report_status="failed",
                report_phase="failed",
                report_mode=str(tree.get("report_mode", "") or ""),
                report_error=state.error,
            )
            return
        if et == "abort_requested":
            return
        if et == "abort_acknowledged":
            return
        if et == "partial_report_generated":
            self._set_report_phase_locked(
                state,
                report_status="partial_ready",
                report_phase="completed",
                report_mode=str(payload.get("mode", "") or ""),
                report_error="",
            )
            return
        if et == "report_version_selected":
            return
        if et == "report_generation_started":
            self._set_report_phase_locked(
                state,
                report_status="running",
                report_phase="started",
                report_mode=str(payload.get("mode", "") or ""),
                report_error="",
            )
            return
        if et == "report_generation_completed":
            self._set_report_phase_locked(
                state,
                report_status="completed",
                report_phase="completed",
                report_mode=str(payload.get("mode", "") or ""),
                report_error="",
            )
            return
        if et == "report_generation_failed":
            self._set_report_phase_locked(
                state,
                report_status="failed",
                report_phase="failed",
                report_mode=str(payload.get("mode", "") or ""),
                report_error=str(payload.get("error", "") or ""),
            )
            return
        if et == "report_version_created":
            self._set_report_phase_locked(
                state,
                report_status="completed",
                report_phase="completed",
                report_mode=str(payload.get("mode", "") or str(tree.get("report_mode", "") or "")),
                report_error="",
            )
            return

    def _set_report_phase_locked(
        self,
        state: RunState,
        report_status: str,
        report_phase: str,
        report_mode: str = "",
        report_error: str = "",
    ) -> None:
        tree = state.tree if isinstance(state.tree, dict) else {}
        state.tree = tree
        mode = str(report_mode or tree.get("report_mode", "") or "").strip()
        tree["report_status"] = str(report_status or "pending")
        tree["report_phase"] = str(report_phase or "")
        tree["report_mode"] = mode
        tree["report_error"] = str(report_error or "")
        tree["report_phase_updated_at"] = _now().isoformat()
        state.updated_at = _now()
        # Keep canonical lifecycle fields in sync with legacy/compat fields.
        state.report_state = self._derive_report_state_locked(state)

    def _ensure_round(self, tree: Dict[str, Any], round_i: int) -> Dict[str, Any]:
        rounds = tree.setdefault("rounds", [])
        for rnd in rounds:
            if rnd.get("round") == round_i:
                return rnd
        rnd = {"round": round_i, "questions": []}
        rounds.append(rnd)
        return rnd

    def _ensure_question(self, round_node: Dict[str, Any], sub_question: str) -> Dict[str, Any]:
        questions = round_node.setdefault("questions", [])
        for q in questions:
            if q.get("sub_question") == sub_question:
                return q
        q = {
            "sub_question": sub_question,
            "node_id": "",
            "depth": 1,
            "parent": "",
            "status": "pending",
            "query_steps": [],
        }
        questions.append(q)
        return q

    def _ensure_query_step(self, question_node: Dict[str, Any], query: str) -> Dict[str, Any]:
        steps = question_node.setdefault("query_steps", [])
        for s in steps:
            if s.get("query") == query:
                return s
        s = {
            "query": query,
            "status": "queued",
            "search_error": None,
            "selected_results_count": 0,
            "primary_count": 0,
        }
        steps.append(s)
        return s

    def _execute_run(self, run_id: str) -> None:
        snapshot = self.get_snapshot(run_id)
        if not snapshot:
            return
        session_lock = self._session_lock_for(run_id)
        task = snapshot.task
        runs_dir = Path("runs")
        runs_dir.mkdir(parents=True, exist_ok=True)
        state_path = runs_dir / f"state_web_{run_id}.json"
        with session_lock:
            with self._lock:
                state_for_path = self._ensure_run_loaded_locked(run_id)
                if state_for_path:
                    state_for_path.state_file_path = str(state_path)
                    state_for_path.updated_at = _now()
                    state_for_path.last_checkpoint_at = _now()
                    self._refresh_lifecycle_fields_locked(state_for_path)
                    self._next_state_version_locked(state_for_path)
                    self._sync_session_from_state_locked(state_for_path)
                    self._save_sessions_index_locked()
        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        slug = slugify_for_filename(task)
        report_path = reports_dir / f"report_{slug}_{ts}.md"

        phase_lock = threading.Lock()
        phase_info: Dict[str, Any] = {
            "phase": "initializing",
            "last_event_type": "run_started",
            "sub_question": "",
            "query": "",
            "updated_at": _now().isoformat(),
        }

        def set_phase(event_type: str, payload: Dict[str, Any]) -> None:
            payload = payload if isinstance(payload, dict) else {}
            phase = "working"
            if event_type == "run_started":
                phase = "initializing"
            elif event_type in {"sub_question_started", "queries_generated"}:
                phase = "planning_node"
            elif event_type in {
                "query_started",
                "query_diagnostic",
                "query_rerun_allowed",
                "query_broadened",
            }:
                phase = "searching"
            elif event_type in {"query_skipped_cached", "query_blocked_diminishing_returns"}:
                phase = "query_decision"
            elif event_type in {"search_completed", "synthesis_completed"}:
                phase = "synthesizing"
            elif event_type in {"node_sufficiency_started", "node_sufficiency_completed"}:
                phase = "node_sufficiency"
            elif event_type in {"node_decomposition_started", "node_decomposed"}:
                phase = "decomposing"
            elif event_type in {"sufficiency_started", "sufficiency_completed"}:
                phase = "run_sufficiency"
            elif event_type in {"report_generation_started", "partial_report_generated"}:
                phase = "writing_report"
            elif event_type == "run_paused":
                phase = "paused"
            elif event_type == "run_resumed":
                phase = "working"
            elif event_type in {"run_completed", "run_failed", "run_aborted"}:
                phase = "terminal"

            with phase_lock:
                phase_info["phase"] = phase
                phase_info["last_event_type"] = event_type
                phase_info["sub_question"] = str(payload.get("sub_question", "")).strip()
                phase_info["query"] = str(payload.get("query", "")).strip()
                phase_info["updated_at"] = _now().isoformat()

        heartbeat_stop = threading.Event()

        def heartbeat_worker() -> None:
            while not heartbeat_stop.wait(self._heartbeat_interval_sec):
                with self._lock:
                    state = self._runs.get(run_id)
                    if not state or state.status in {"completed", "failed"}:
                        return
                with phase_lock:
                    payload = {
                        "phase": str(phase_info.get("phase", "working")),
                        "last_event_type": str(
                            phase_info.get("last_event_type", "")
                        ),
                        "sub_question": str(phase_info.get("sub_question", "")),
                        "query": str(phase_info.get("query", "")),
                        "phase_updated_at": str(phase_info.get("updated_at", "")),
                    }
                self._publish_event(
                    run_id,
                    {
                        "run_id": run_id,
                        "timestamp": _now().isoformat(),
                        "event_type": "run_heartbeat",
                        "payload": payload,
                    },
                )

        heartbeat_thread = threading.Thread(target=heartbeat_worker, daemon=True)
        heartbeat_thread.start()

        def callback(event: Dict[str, Any]) -> None:
            set_phase(str(event.get("event_type", "")), event.get("payload", {}))
            self._publish_event(run_id, event)

        paused_announced = False

        def should_abort() -> bool:
            nonlocal paused_announced
            while True:
                with self._lock:
                    cancelled = bool(self._cancel_flags.get(run_id, False))
                    paused = bool(self._pause_flags.get(run_id, False))
                    state = self._runs.get(run_id)
                if cancelled:
                    return True
                if paused and state and state.status == "running":
                    if not paused_announced:
                        set_phase("run_paused", {})
                        self._publish_event(
                            run_id,
                            {
                                "run_id": run_id,
                                "timestamp": _now().isoformat(),
                                "event_type": "run_paused",
                                "payload": {},
                            },
                        )
                        paused_announced = True
                    time.sleep(0.2)
                    continue
                if paused_announced:
                    set_phase("run_resumed", {})
                    self._publish_event(
                        run_id,
                        {
                            "run_id": run_id,
                            "timestamp": _now().isoformat(),
                            "event_type": "run_resumed",
                            "payload": {},
                        },
                    )
                    paused_announced = False
                return False

        agent = DeepResearchAgent(
            model=snapshot.model,
            report_model=snapshot.report_model,
            max_depth=snapshot.max_depth,
            max_rounds=snapshot.max_rounds,
            results_per_query=snapshot.results_per_query,
            state_file=str(state_path),
            verbose=True,
            event_callback=callback,
            run_id=run_id,
            should_abort=should_abort,
        )
        try:
            report = agent.run(task)
            report_path.write_text(report, encoding="utf-8")
            with session_lock:
                with self._lock:
                    state = self._ensure_run_loaded_locked(run_id)
                    if not state:
                        return
                    state.token_usage = agent.usage_tracker.to_dict()
                    created = self._append_report_version_locked(
                        state=state,
                        report_text=report,
                        report_file_path=str(report_path),
                        trigger_type="auto_natural_stop",
                        usage_snapshot=state.token_usage,
                    )
                    if state.status != "failed":
                        state.status = "completed"
                        state.execution_state = "completed"
                        self._pause_flags[run_id] = False
                        self._set_report_phase_locked(
                            state,
                            report_status="completed",
                            report_phase="completed",
                            report_mode="final",
                            report_error="",
                        )
                    self._next_state_version_locked(state)
                    self._sync_session_from_state_locked(state)
                    self._save_sessions_index_locked()
            self._publish_event(
                run_id,
                {
                    "run_id": run_id,
                    "timestamp": _now().isoformat(),
                    "event_type": "report_version_created",
                    "payload": {
                        "version_index": int(created.get("version_index", 1)),
                        "trigger_type": "auto_natural_stop",
                        "report_file_path": str(report_path),
                    },
                },
            )
        except Exception as exc:
            error_text = str(exc)
            terminal_event_type = (
                "run_aborted" if error_text == "Run aborted by user" else "run_failed"
            )
            emit_terminal_event = False
            with session_lock:
                with self._lock:
                    state = self._ensure_run_loaded_locked(run_id)
                    if not state:
                        return
                    state.status = "failed"
                    state.execution_state = (
                        "aborted" if error_text == "Run aborted by user" else "failed"
                    )
                    self._pause_flags[run_id] = False
                    state.error = error_text
                    state.token_usage = agent.usage_tracker.to_dict()
                    self._set_report_phase_locked(
                        state,
                        report_status="failed",
                        report_phase="failed",
                        report_mode="final",
                        report_error=error_text,
                    )
                    self._next_state_version_locked(state)
                    self._sync_session_from_state_locked(state)
                    self._save_sessions_index_locked()
                    last_event_type = (
                        state.events[-1].get("event_type", "") if state.events else ""
                    )
                    emit_terminal_event = last_event_type not in {
                        "run_completed",
                        "run_failed",
                        "run_aborted",
                    }
            if emit_terminal_event:
                self._publish_event(
                    run_id,
                    {
                        "run_id": run_id,
                        "timestamp": _now().isoformat(),
                        "event_type": terminal_event_type,
                        "payload": {"error": error_text},
                    },
                )
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=1.0)

    def _best_effort_usage_snapshot(self, snapshot: RunState) -> Optional[Dict[str, Any]]:
        if isinstance(snapshot.token_usage, dict):
            return snapshot.token_usage
        state_file_path = snapshot.state_file_path or ""
        if not state_file_path:
            return None
        state_path = Path(state_file_path)
        if not state_path.exists():
            return None
        try:
            parsed = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(parsed, dict):
            return None
        token_usage = parsed.get("token_usage", {})
        return token_usage if isinstance(token_usage, dict) else None

    def _build_partial_report_with_agent(
        self, snapshot: RunState, report_mode: str
    ) -> tuple[str, Optional[Dict[str, Any]]]:
        try:
            agent = DeepResearchAgent(
                model=snapshot.model,
                report_model=snapshot.report_model,
                max_depth=max(1, int(snapshot.max_depth)),
                max_rounds=max(1, int(snapshot.max_rounds)),
                results_per_query=max(1, int(snapshot.results_per_query)),
                token_breakdown=True,
                cost_estimate_enabled=True,
                verbose=False,
            )
        except Exception:
            return "", None

        state_file_path = snapshot.state_file_path or ""
        raw: Dict[str, Any] = {}
        if state_file_path:
            state_path = Path(state_file_path)
            if state_path.exists():
                try:
                    parsed = json.loads(state_path.read_text(encoding="utf-8"))
                    if isinstance(parsed, dict):
                        raw = parsed
                except Exception:
                    raw = {}

        try:
            if raw:
                findings = agent._deserialize_findings(raw.get("findings", {}))
                success_criteria = [
                    str(v).strip()
                    for v in raw.get("success_criteria", [])
                    if isinstance(v, str) and str(v).strip()
                ]
                stats = raw.get("search_stats", {})
                if not isinstance(stats, dict):
                    stats = {}
                total_search_calls = int(stats.get("total_calls", 0))
                failed_search_calls = int(stats.get("failed_calls", 0))
                queries_with_evidence = int(stats.get("queries_with_evidence", 0))
                total_selected_results = int(stats.get("total_selected_results", 0))
            else:
                findings, success_criteria, stats = self._derive_partial_context_from_tree(
                    snapshot
                )
                total_search_calls = int(stats.get("total_calls", 0))
                failed_search_calls = int(stats.get("failed_calls", 0))
                queries_with_evidence = int(stats.get("queries_with_evidence", 0))
                total_selected_results = int(stats.get("total_selected_results", 0))

            evidence_status, evidence_note = agent._assess_evidence_strength(
                total_search_calls=total_search_calls,
                failed_search_calls=failed_search_calls,
                queries_with_evidence=queries_with_evidence,
                total_selected_results=total_selected_results,
            )
            if report_mode == "partial":
                evidence_note = (
                    f"{evidence_note} This is a partial report generated before run completion."
                )
            else:
                evidence_note = (
                    f"{evidence_note} This report is generated from the current finalized findings snapshot."
                )
            prompt = agent._format_report_prompt(
                task=snapshot.task,
                success_criteria=success_criteria,
                findings=findings,
                evidence_status=evidence_status,
                evidence_note=evidence_note,
                search_stats=stats,
            )
            report_text = agent.report_llm.text(
                SYSTEM_REPORT,
                prompt,
                stage=(
                    "report_partial"
                    if report_mode == "partial"
                    else "report_final_from_snapshot"
                ),
                metadata={"task": snapshot.task[:120]},
            )
            return report_text, agent.usage_tracker.to_dict()
        except Exception:
            return "", None

    def _merge_usage_dicts(
        self, base: Optional[Dict[str, Any]], delta: Dict[str, Any]
    ) -> Dict[str, Any]:
        merged = dict(base) if isinstance(base, dict) else {}
        base_events = (
            merged.get("events", [])
            if isinstance(merged.get("events", []), list)
            else []
        )
        delta_events = (
            delta.get("events", [])
            if isinstance(delta.get("events", []), list)
            else []
        )
        combined_events: List[Dict[str, Any]] = []
        for e in base_events + delta_events:
            if isinstance(e, dict):
                combined_events.append(e)
        enabled = bool(
            (merged.get("enabled", False) if isinstance(merged, dict) else False)
            or bool(delta.get("enabled", False))
        )
        cost_enabled = bool(
            (merged.get("cost_enabled", False) if isinstance(merged, dict) else False)
            or bool(delta.get("cost_enabled", False))
        )
        pricing_source = str(
            delta.get("pricing_source")
            or merged.get("pricing_source", "")
            or "unknown"
        )
        tracker = UsageTracker(
            enabled=enabled,
            cost_enabled=cost_enabled,
            pricing_source=pricing_source,
            pricing_models={},
            pricing_default={"input_per_1m": 0.0, "output_per_1m": 0.0},
            search_query_pricing={
                "default_per_1k_queries": 0.0,
                "by_tool_type": {},
                "reasoning_model_prefixes": [],
                "reasoning_model_names": [],
            },
        )
        tracker.load_from_dict({"events": combined_events})
        out = tracker.to_dict()
        out["pricing_source"] = pricing_source
        return out

    def _derive_partial_context_from_tree(
        self, snapshot: RunState
    ) -> tuple[Dict[str, SubQuestionFinding], List[str], Dict[str, int]]:
        tree = snapshot.tree if isinstance(snapshot.tree, dict) else {}
        rounds = tree.get("rounds", [])
        findings: Dict[str, SubQuestionFinding] = {}
        total_calls = 0
        failed_calls = 0
        queries_with_evidence = 0
        total_selected_results = 0
        if isinstance(rounds, list):
            for rnd in rounds:
                if not isinstance(rnd, dict):
                    continue
                questions = rnd.get("questions", [])
                if not isinstance(questions, list):
                    continue
                for q in questions:
                    if not isinstance(q, dict):
                        continue
                    sq = str(q.get("sub_question", "")).strip()
                    if not sq:
                        continue
                    finding = findings.setdefault(
                        sq, SubQuestionFinding(sub_question=sq)
                    )
                    if q.get("unresolved_reason"):
                        finding.uncertainties.append(str(q.get("unresolved_reason", "")))
                    steps = q.get("query_steps", [])
                    if not isinstance(steps, list):
                        continue
                    for st in steps:
                        if not isinstance(st, dict):
                            continue
                        status = str(st.get("status", ""))
                        if status in {"success", "skipped", "cached", "running", "blocked"}:
                            total_calls += 1
                        if st.get("search_error"):
                            failed_calls += 1
                        selected = int(st.get("selected_results_count", 0))
                        total_selected_results += selected
                        if selected > 0:
                            queries_with_evidence += 1
                        summary = str(st.get("synthesis_summary", "")).strip()
                        if summary:
                            finding.summaries.append(summary)
        stats = {
            "total_calls": total_calls,
            "failed_calls": failed_calls,
            "queries_with_evidence": queries_with_evidence,
            "total_selected_results": total_selected_results,
        }
        return findings, [], stats

    def _default_title_from_task(self, task: str) -> str:
        clean = " ".join(str(task or "").split()).strip()
        if not clean:
            return "Untitled session"
        if len(clean) > 80:
            return clean[:77] + "..."
        return clean

    def _sync_session_from_state_locked(self, state: RunState) -> None:
        self._refresh_lifecycle_fields_locked(state)
        sid = state.session_id or state.run_id
        state.session_id = sid
        if not state.title:
            state.title = self._default_title_from_task(state.task)
        state.last_checkpoint_at = state.last_checkpoint_at or _now()
        report_file_path = state.report_file_path or ""
        if not report_file_path and isinstance(state.tree, dict):
            report_file_path = str(state.tree.get("report_file_path", "") or "")
        report_versions = state.report_versions if isinstance(state.report_versions, list) else []
        report_versions_count = len(report_versions)
        latest_report_at: Optional[str] = None
        if report_versions_count > 0:
            latest = report_versions[-1]
            if isinstance(latest, dict):
                latest_report_at = str(latest.get("created_at", "") or "") or None
        self._sessions[sid] = {
            "session_id": sid,
            "owner_id": state.owner_id,
            "title": state.title,
            "status": state.status,
            "version": max(1, int(getattr(state, "version", 1) or 1)),
            "execution_state": state.execution_state,
            "research_state": state.research_state,
            "report_state": state.report_state,
            "terminal_reason": state.terminal_reason,
            "created_at": state.created_at.isoformat(),
            "updated_at": state.updated_at.isoformat(),
            "state_file_path": state.state_file_path or "",
            "report_file_path": report_file_path or None,
            "report_versions_count": report_versions_count,
            "latest_report_at": latest_report_at,
            "current_report_version_index": state.current_report_version_index,
            "has_manual_edits": bool(state.has_manual_edits),
        }

    def _save_sessions_index_locked(self, force: bool = False) -> None:
        now_ts = time.time()
        if not force and (now_ts - self._last_index_flush_ts) < self._index_flush_min_interval_sec:
            return
        self._sessions_index_meta["schema_version"] = self._SESSIONS_INDEX_SCHEMA_VERSION
        payload = {
            "schema_version": int(self._sessions_index_meta.get("schema_version", self._SESSIONS_INDEX_SCHEMA_VERSION)),
            "title_migration_v1_done": bool(self._sessions_index_meta.get("title_migration_v1_done", False)),
            "execution_state_migration_v1_done": bool(
                self._sessions_index_meta.get("execution_state_migration_v1_done", False)
            ),
            "updated_at": _now().isoformat(),
            "sessions": list(self._sessions.values()),
        }
        self._last_index_flush_ts = now_ts

        # Queue latest payload for the single writer thread; this preserves write ordering
        # while keeping filesystem I/O out of hot lock paths.
        self._index_flush_seq += 1
        self._index_flush_pending = (self._index_flush_seq, payload)
        self._index_flush_event.set()

    def _sessions_index_writer_loop(self) -> None:
        while True:
            self._index_flush_event.wait()
            pending = self._index_flush_pending
            if not pending:
                self._index_flush_event.clear()
                continue
            seq, payload = pending
            with self._index_write_lock:
                try:
                    tmp_path = self._sessions_index_path.with_suffix(".json.tmp")
                    tmp_path.write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    tmp_path.replace(self._sessions_index_path)
                except Exception as exc:
                    # Keep session/runs in memory even if persistence is temporarily unavailable.
                    print(f"[sessions] index flush failed: {exc}")
            # If no newer payload arrived while writing, clear pending + sleep gate.
            latest = self._index_flush_pending
            if latest and latest[0] == seq:
                self._index_flush_pending = None
                self._index_flush_event.clear()

    def _should_flush_sessions_index(self, event_type: str) -> bool:
        if event_type in {
            "run_started",
            "run_paused",
            "run_resumed",
            "run_completed",
            "run_failed",
            "run_aborted",
            "report_generation_started",
            "report_generation_completed",
            "report_generation_failed",
            "partial_report_generated",
            "report_version_created",
            "report_version_selected",
            "abort_requested",
            "run_abort_requested",
        }:
            return True
        return False

    def _load_sessions_index(self) -> Dict[str, Any]:
        if not self._sessions_index_path.exists():
            return {"sessions": []}
        try:
            parsed = json.loads(self._sessions_index_path.read_text(encoding="utf-8"))
        except Exception:
            return {"sessions": []}
        if not isinstance(parsed, dict):
            return {"sessions": []}
        try:
            schema_version = int(
                parsed.get("schema_version", self._SESSIONS_INDEX_SCHEMA_VERSION)
            )
        except (TypeError, ValueError):
            schema_version = self._SESSIONS_INDEX_SCHEMA_VERSION
        self._sessions_index_meta["schema_version"] = schema_version
        self._sessions_index_meta["title_migration_v1_done"] = bool(
            parsed.get("title_migration_v1_done", False)
        )
        self._sessions_index_meta["execution_state_migration_v1_done"] = bool(
            parsed.get("execution_state_migration_v1_done", False)
        )
        raw_sessions = parsed.get("sessions", [])
        if not isinstance(raw_sessions, list):
            return {"sessions": []}
        out: List[Dict[str, Any]] = []
        for item in raw_sessions:
            if isinstance(item, dict):
                out.append(item)
        return {"sessions": out}

    def _bootstrap_sessions_from_disk(self) -> None:
        loaded = self._load_sessions_index()
        loaded_sessions = loaded.get("sessions", [])
        title_migration_needed = not bool(
            self._sessions_index_meta.get("title_migration_v1_done", False)
        )
        execution_state_migration_needed = not bool(
            self._sessions_index_meta.get("execution_state_migration_v1_done", False)
        )
        migrated_titles = 0
        migrated_execution_states = 0
        with self._lock:
            for item in loaded_sessions:
                sid = str(item.get("session_id", "")).strip()
                if not sid:
                    continue
                self._sessions[sid] = item
                self._session_locks.setdefault(sid, threading.RLock())
                self._subscribers.setdefault(sid, [])
                self._cancel_flags.setdefault(sid, False)
                self._pause_flags.setdefault(sid, False)

            known_state_paths = {
                str(v.get("state_file_path", "")).strip()
                for v in self._sessions.values()
                if isinstance(v, dict)
            }
            for sp in glob(str(self._runs_dir / "state_web_*.json")):
                if sp not in known_state_paths:
                    run_id = Path(sp).stem.replace("state_web_", "")
                    self._sessions.setdefault(
                        run_id,
                        {
                            "session_id": run_id,
                            "owner_id": None,
                            "title": "Recovered session",
                            "status": "failed",
                            "execution_state": "failed",
                            "created_at": _now().isoformat(),
                            "updated_at": _now().isoformat(),
                            "state_file_path": sp,
                            "report_file_path": None,
                            "has_manual_edits": False,
                        },
                    )

            for sid, item in list(self._sessions.items()):
                state_path = str(item.get("state_file_path", "")).strip()
                state = self._load_run_state_from_files(
                    sid,
                    state_path,
                    item,
                    force_execution_state_recompute=execution_state_migration_needed,
                )
                if state:
                    recovered = self._recover_stale_report_generation_on_load_locked(
                        state, run_id=sid
                    )
                    prior_title = str(item.get("title", "")).strip()
                    prior_exec = str(item.get("execution_state", "")).strip().lower()
                    self._refresh_lifecycle_fields_locked(state)
                    self._runs[sid] = state
                    self._session_locks.setdefault(sid, threading.RLock())
                    self._subscribers.setdefault(sid, [])
                    self._cancel_flags.setdefault(sid, False)
                    self._pause_flags.setdefault(sid, False)
                    self._sync_session_from_state_locked(state)
                    new_title = str(self._sessions.get(sid, {}).get("title", "")).strip()
                    if (
                        title_migration_needed
                        and prior_title.lower() == "recovered session"
                        and new_title
                        and new_title.lower() != "recovered session"
                    ):
                        migrated_titles += 1
                    new_exec = str(
                        self._sessions.get(sid, {}).get("execution_state", "")
                    ).strip().lower()
                    if execution_state_migration_needed and prior_exec and prior_exec != new_exec:
                        migrated_execution_states += 1
                    if recovered:
                        try:
                            self._persist_run_state_locked(state)
                        except Exception:
                            pass

            if title_migration_needed:
                self._sessions_index_meta["title_migration_v1_done"] = True
            if execution_state_migration_needed:
                self._sessions_index_meta["execution_state_migration_v1_done"] = True
            self._save_sessions_index_locked()
        if title_migration_needed:
            print(f"[sessions] title migration v1 applied; updated {migrated_titles} session titles.")
        if execution_state_migration_needed:
            print(
                "[sessions] execution-state migration v1 applied; "
                f"updated {migrated_execution_states} sessions."
            )

    def _load_run_state_from_files(
        self,
        session_id: str,
        state_file_path: str,
        session_item: Dict[str, Any],
        force_execution_state_recompute: bool = False,
    ) -> Optional[RunState]:
        # Authoritative source: state_web_<session_id>.json checkpoint.
        # sessions_index entries are cache/projection only and should not
        # override lifecycle fields when hydrating RunState.
        status = "failed"
        version = 1
        execution_state = "failed"
        raw_title = str(session_item.get("title", "") or "").strip()
        task = raw_title if raw_title and raw_title.lower() != "recovered session" else "Recovered session"
        tree: Dict[str, Any] = {
            "task": task,
            "plan": {"sub_questions": [], "success_criteria": []},
            "max_depth": 3,
            "rounds": [],
            "final_sufficiency": None,
            "report_status": "pending",
            "report_phase": "",
            "report_mode": "",
            "report_phase_updated_at": "",
            "report_error": "",
        }
        report_text = None
        report_path = ""
        report_versions: List[Dict[str, Any]] = []
        current_report_version_index: Optional[int] = None
        created_at = _now()
        updated_at = _now()
        model = "gpt-4.1"
        report_model = "gpt-5.2"
        max_depth = 3
        max_rounds = 1
        results_per_query = 3
        token_usage: Optional[Dict[str, Any]] = None
        error = None
        research_state = ""
        report_state = ""
        terminal_reason = ""

        try:
            ci = str(session_item.get("created_at", "") or "")
            if ci:
                created_at = datetime.fromisoformat(ci)
        except Exception:
            pass
        try:
            ui = str(session_item.get("updated_at", "") or "")
            if ui:
                updated_at = datetime.fromisoformat(ui)
        except Exception:
            pass

        if state_file_path and Path(state_file_path).exists():
            try:
                raw = json.loads(Path(state_file_path).read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    raw_task = str(raw.get("task", "") or "").strip()
                    if raw_task:
                        task = raw_task
                    status = str(raw.get("status", status) or status)
                    try:
                        version = max(1, int(raw.get("version", version) or version))
                    except Exception:
                        version = max(1, version)
                    execution_state = str(raw.get("execution_state", execution_state) or execution_state)
                    research_state = str(raw.get("research_state", "") or "")
                    report_state = str(raw.get("report_state", "") or "")
                    terminal_reason = str(raw.get("terminal_reason", "") or "")
                    report_path = str(raw.get("report_file_path", "") or report_path)
                    raw_top_versions = raw.get("report_versions", [])
                    if isinstance(raw_top_versions, list):
                        report_versions = [v for v in raw_top_versions if isinstance(v, dict)]
                    raw_top_idx = raw.get("current_report_version_index")
                    if isinstance(raw_top_idx, int):
                        current_report_version_index = raw_top_idx
                    trace = raw.get("trace", {})
                    if isinstance(trace, dict):
                        trace_task = str(trace.get("task", "") or "").strip()
                        if trace_task:
                            task = trace_task
                        model = str(trace.get("model", model) or model)
                        report_model = str(trace.get("report_model", report_model) or report_model)
                        max_depth = int(trace.get("max_depth", max_depth))
                        max_rounds = int(trace.get("max_rounds", max_rounds))
                        results_per_query = int(
                            trace.get("results_per_query", results_per_query)
                        )
                        rounds = trace.get("rounds", [])
                        if isinstance(rounds, list):
                            tree["rounds"] = rounds
                        fs = trace.get("final_sufficiency", None)
                        tree["final_sufficiency"] = fs
                        report = trace.get("report", "")
                        if isinstance(report, str) and report.strip():
                            report_text = report
                        trace_versions = trace.get("report_versions", [])
                        if isinstance(trace_versions, list):
                            report_versions = [v for v in trace_versions if isinstance(v, dict)]
                        trace_current_idx = trace.get("current_report_version_index")
                        if isinstance(trace_current_idx, int):
                            current_report_version_index = trace_current_idx
                        tu = trace.get("token_usage", {})
                        if isinstance(tu, dict):
                            token_usage = tu
                        error = str(raw.get("error", "") or "")
                        research_state = str(trace.get("research_state", research_state) or research_state)
                        report_state = str(trace.get("report_state", report_state) or report_state)
                        terminal_reason = str(trace.get("terminal_reason", terminal_reason) or terminal_reason)
                        report_status = str(
                            trace.get("report_status", raw.get("report_status", "")) or ""
                        ).strip()
                        if report_status:
                            tree["report_status"] = report_status
                        else:
                            tree["report_status"] = (
                                "completed"
                                if status == "completed"
                                else ("failed" if status == "failed" else "pending")
                            )
                        tree["report_phase"] = str(
                            trace.get("report_phase", raw.get("report_phase", "")) or ""
                        )
                        tree["report_mode"] = str(
                            trace.get("report_mode", raw.get("report_mode", "")) or ""
                        )
                        tree["report_phase_updated_at"] = str(
                            trace.get(
                                "report_phase_updated_at",
                                raw.get("report_phase_updated_at", ""),
                            )
                            or ""
                        )
                        tree["report_error"] = str(
                            trace.get("report_error", raw.get("report_error", "")) or ""
                        )
                    st = str(raw.get("started_at", "") or "")
                    if st:
                        try:
                            created_at = datetime.fromisoformat(st)
                        except Exception:
                            pass
                    ut = str(raw.get("updated_at", "") or "")
                    if ut:
                        try:
                            updated_at = datetime.fromisoformat(ut)
                        except Exception:
                            pass
            except Exception:
                pass

        if (not task or task.lower() == "recovered session") and raw_title:
            task = raw_title
        if not task:
            task = "Recovered session"
        if not state_file_path:
            inferred = str(self._runs_dir / f"state_web_{session_id}.json")
            if Path(inferred).exists():
                state_file_path = inferred

        if report_path and Path(report_path).exists() and not report_text:
            try:
                report_text = Path(report_path).read_text(encoding="utf-8")
            except Exception:
                report_text = None

        if not report_versions and report_text:
            report_versions = [
                {
                    "version_index": 1,
                    "created_at": updated_at.isoformat(),
                    "trigger_type": "manual_legacy",
                    "report_text": report_text,
                    "report_file_path": report_path or "",
                    "token_usage": token_usage if isinstance(token_usage, dict) else None,
                }
            ]
            current_report_version_index = 1
        elif report_versions and (
            current_report_version_index is None
            or current_report_version_index < 1
            or current_report_version_index > len(report_versions)
        ):
            current_report_version_index = len(report_versions)
        if report_versions and current_report_version_index:
            selected = report_versions[current_report_version_index - 1]
            report_text = str(selected.get("report_text", "") or "")
            rp = str(selected.get("report_file_path", "") or "")
            if rp:
                report_path = rp
            if (not report_text) and report_path and Path(report_path).exists():
                try:
                    report_text = Path(report_path).read_text(encoding="utf-8")
                except Exception:
                    report_text = None

        if status not in {"queued", "running", "completed", "failed"}:
            status = "failed"
        if force_execution_state_recompute:
            execution_state = self._execution_state_from_status(status)
        elif execution_state not in {"idle", "running", "paused", "completed", "failed", "aborted"}:
            execution_state = self._execution_state_from_status(status)

        # Recovered sessions are loaded from disk without reattaching a live worker.
        # Any non-terminal status from prior process is stale and should be normalized.
        if status in {"queued", "running"}:
            status = "failed"
            execution_state = "failed"
            if not str(error or "").strip():
                error = "Recovered interrupted run (non-terminal checkpoint without active worker)."

        raw_title = str(session_item.get("title", "") or "").strip()
        if not raw_title or raw_title.lower() == "recovered session":
            resolved_title = self._default_title_from_task(task)
        else:
            resolved_title = raw_title

        return RunState(
            run_id=session_id,
            session_id=session_id,
            owner_id=session_item.get("owner_id"),
            title=resolved_title,
            status=status,  # type: ignore[arg-type]
            version=max(1, int(version or 1)),
            execution_state=execution_state,  # type: ignore[arg-type]
            research_state=research_state or None,
            report_state=report_state or None,
            terminal_reason=terminal_reason or None,
            created_at=created_at,
            updated_at=updated_at,
            task=task,
            max_depth=max(1, max_depth),
            max_rounds=max(1, max_rounds),
            results_per_query=max(1, results_per_query),
            model=model,
            report_model=report_model,
            tree=tree,
            report_text=report_text,
            report_file_path=report_path or None,
            report_versions=report_versions,
            current_report_version_index=current_report_version_index,
            state_file_path=state_file_path or None,
            error=error,
            token_usage=token_usage,
        )

    def _execution_state_from_status(self, status: str) -> str:
        s = str(status or "").strip().lower()
        if s == "running":
            return "running"
        if s == "queued":
            return "idle"
        if s == "completed":
            return "completed"
        if s == "failed":
            return "failed"
        return "idle"
