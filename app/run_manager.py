import json
import copy
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

    def __init__(self, heartbeat_interval_sec: float = _DEFAULT_HEARTBEAT_INTERVAL_SEC) -> None:
        self._runs: Dict[str, RunState] = {}
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._subscribers: Dict[str, List[Queue]] = {}
        self._cancel_flags: Dict[str, bool] = {}
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
            },
        )
        with self._lock:
            self._runs[run_id] = state
            self._subscribers[run_id] = []
            self._cancel_flags[run_id] = False
            self._sync_session_from_state_locked(state)
            self._save_sessions_index_locked(force=True)

        t = threading.Thread(
            target=self._execute_run,
            args=(run_id,),
            daemon=True,
        )
        t.start()
        return run_id

    def list_sessions(self) -> List[SessionSummary]:
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
                out.append(SessionSummary.model_validate(it))
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
            if state and state.status in {"queued", "running"}:
                return "conflict_running"
            sess = self._sessions.get(session_id)
            if not sess:
                return None
            state_file_path = str(sess.get("state_file_path", "")).strip()
            self._sessions.pop(session_id, None)
            self._runs.pop(session_id, None)
            self._subscribers.pop(session_id, None)
            self._cancel_flags.pop(session_id, None)
            self._save_sessions_index_locked(force=True)
        if state_file_path:
            try:
                path = Path(state_file_path)
                if path.exists():
                    path.unlink()
            except Exception:
                pass
        return "deleted"

    def abort_run(self, run_id: str) -> Optional[str]:
        should_emit = False
        with self._lock:
            state = self._runs.get(run_id)
            if not state:
                return None
            if state.status in ("completed", "failed"):
                return state.status
            self._cancel_flags[run_id] = True
            state.updated_at = _now()
            should_emit = True
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
        return "aborting"

    def generate_partial_report(self, run_id: str) -> Optional[Dict[str, str]]:
        snapshot = self.get_snapshot(run_id)
        if not snapshot:
            return None
        base_usage = self._best_effort_usage_snapshot(snapshot)
        self._publish_event(
            run_id,
            {
                "run_id": run_id,
                "timestamp": _now().isoformat(),
                "event_type": "report_generation_started",
                "payload": {"mode": "partial"},
            },
        )
        report_text, partial_usage = self._build_partial_report_with_agent(snapshot)
        if not report_text.strip():
            self._publish_event(
                run_id,
                {
                    "run_id": run_id,
                    "timestamp": _now().isoformat(),
                    "event_type": "report_generation_failed",
                    "payload": {"mode": "partial"},
                },
            )
            return {"error": "Partial report generation failed (report agent unavailable)."}
        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        slug = slugify_for_filename(snapshot.task)
        report_path = reports_dir / f"report_{slug}_{ts}.md"
        report_path.write_text(report_text, encoding="utf-8")

        with self._lock:
            state = self._runs.get(run_id)
            if not state:
                return None
            state.report_text = report_text
            state.report_file_path = str(report_path)
            live_usage = state.token_usage if isinstance(state.token_usage, dict) else None
            usage_base = live_usage or base_usage
            if partial_usage:
                state.token_usage = self._merge_usage_dicts(
                    usage_base, partial_usage
                )
            elif usage_base:
                state.token_usage = usage_base
            state.updated_at = _now()
            state.tree["report_status"] = "completed"

        self._publish_event(
            run_id,
            {
                "run_id": run_id,
                "timestamp": _now().isoformat(),
                "event_type": "partial_report_generated",
                "payload": {
                    "report_file_path": str(report_path),
                    "mode": "partial",
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
                    "mode": "partial",
                },
            },
        )
        return {"report_file_path": str(report_path)}

    def get_snapshot(self, run_id: str) -> Optional[RunState]:
        acquired, acquired_at = self._lock_acquire("get_snapshot", timeout=0.5)
        if acquired:
            try:
                state = self._runs.get(run_id)
                if state:
                    payload = self._snapshot_payload_locked(state)
                    return RunState.model_validate(payload)
                session_item = self._sessions.get(run_id)
            finally:
                self._lock_release("get_snapshot", acquired_at)
        else:
            loaded = self._load_sessions_index()
            session_item = None
            for item in loaded.get("sessions", []):
                if isinstance(item, dict) and str(item.get("session_id", "")).strip() == run_id:
                    session_item = item
                    break

        if not session_item:
            return None
        state_path = str(session_item.get("state_file_path", "")).strip()
        return self._load_run_state_from_files(run_id, state_path, session_item)

    def subscribe(self, run_id: str) -> Optional[Queue]:
        with self._lock:
            if run_id not in self._runs:
                return None
            q: Queue = Queue()
            self._subscribers[run_id].append(q)
            return q

    def unsubscribe(self, run_id: str, queue: Queue) -> None:
        with self._lock:
            subs = self._subscribers.get(run_id, [])
            if queue in subs:
                subs.remove(queue)

    def _publish_event(self, run_id: str, event: Dict[str, Any]) -> None:
        subscribers: List[Queue] = []
        acquired, _ = self._lock_acquire("_publish_event")
        if not acquired:
            return
        try:
            state = self._runs.get(run_id)
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
            self._sync_session_from_state_locked(state)
            event_type = str(event.get("event_type", ""))
            if self._should_flush_sessions_index(event_type):
                self._save_sessions_index_locked(force=True)
            subscribers = list(self._subscribers.get(run_id, []))
        finally:
            self._lock_release("_publish_event")
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
        if self._lock_debug and wait_ms >= self._lock_wait_warn_ms:
            print(f"[lock] wait {wait_ms:.1f}ms section={section}")
        return acquired, time.perf_counter()

    def _lock_release(self, section: str, acquired_at: Optional[float] = None) -> None:
        hold_ms = 0.0
        if acquired_at is not None:
            hold_ms = (time.perf_counter() - acquired_at) * 1000.0
        if self._lock_debug and acquired_at is not None and hold_ms >= self._lock_hold_warn_ms:
            print(f"[lock] hold {hold_ms:.1f}ms section={section}")
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
            "execution_state": state.execution_state,
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
            "state_file_path": state.state_file_path,
            "error": state.error,
            "token_usage": copy.deepcopy(state.token_usage),
            "manual_edit_log": copy.deepcopy(state.manual_edit_log),
            "manual_assertions": copy.deepcopy(state.manual_assertions),
        }

    def _project_event(self, state: RunState, event: Dict[str, Any]) -> None:
        et = event.get("event_type", "")
        payload = event.get("payload", {})
        tree = state.tree

        if et == "run_started":
            state.status = "running"
            state.execution_state = "running"
            tree["report_status"] = "pending"
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
            tree["report_status"] = "completed"
            return
        if et == "run_failed":
            state.status = "failed"
            state.execution_state = "failed"
            state.error = str(payload.get("error", "unknown error"))
            tree["report_status"] = "failed"
            return
        if et == "run_aborted":
            state.status = "failed"
            state.execution_state = "aborted"
            state.error = str(payload.get("error", "Run aborted by user"))
            tree["report_status"] = "failed"
            return
        if et == "abort_requested":
            return
        if et == "abort_acknowledged":
            return
        if et == "partial_report_generated":
            tree["report_status"] = "partial_ready"
            return
        if et == "report_generation_started":
            tree["report_status"] = "running"
            return
        if et == "report_generation_completed":
            tree["report_status"] = "completed"
            return
        if et == "report_generation_failed":
            tree["report_status"] = "failed"
            return

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
        task = snapshot.task
        runs_dir = Path("runs")
        runs_dir.mkdir(parents=True, exist_ok=True)
        state_path = runs_dir / f"state_web_{run_id}.json"
        with self._lock:
            state_for_path = self._runs.get(run_id)
            if state_for_path:
                state_for_path.state_file_path = str(state_path)
                state_for_path.updated_at = _now()
                state_for_path.last_checkpoint_at = _now()
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

        def should_abort() -> bool:
            with self._lock:
                return bool(self._cancel_flags.get(run_id, False))

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
            with self._lock:
                state = self._runs.get(run_id)
                if not state:
                    return
                state.report_text = report
                state.report_file_path = str(report_path)
                state.token_usage = agent.usage_tracker.to_dict()
                state.updated_at = _now()
                if state.status != "failed":
                    state.status = "completed"
                    state.execution_state = "completed"
                    state.tree["report_status"] = "completed"
                self._sync_session_from_state_locked(state)
                self._save_sessions_index_locked()
        except Exception as exc:
            error_text = str(exc)
            terminal_event_type = (
                "run_aborted" if error_text == "Run aborted by user" else "run_failed"
            )
            emit_terminal_event = False
            with self._lock:
                state = self._runs.get(run_id)
                if not state:
                    return
                state.status = "failed"
                state.execution_state = (
                    "aborted" if error_text == "Run aborted by user" else "failed"
                )
                state.error = error_text
                state.token_usage = agent.usage_tracker.to_dict()
                state.updated_at = _now()
                state.tree["report_status"] = "failed"
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
        self, snapshot: RunState
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
            evidence_note = (
                f"{evidence_note} This is a partial report generated before run completion."
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
                stage="report_partial",
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
        sid = state.session_id or state.run_id
        state.session_id = sid
        if not state.title:
            state.title = self._default_title_from_task(state.task)
        state.last_checkpoint_at = state.last_checkpoint_at or _now()
        report_file_path = state.report_file_path or ""
        if not report_file_path and isinstance(state.tree, dict):
            report_file_path = str(state.tree.get("report_file_path", "") or "")
        self._sessions[sid] = {
            "session_id": sid,
            "owner_id": state.owner_id,
            "title": state.title,
            "status": state.status,
            "execution_state": state.execution_state,
            "created_at": state.created_at.isoformat(),
            "updated_at": state.updated_at.isoformat(),
            "state_file_path": state.state_file_path or "",
            "report_file_path": report_file_path or None,
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
            "run_completed",
            "run_failed",
            "run_aborted",
            "report_generation_started",
            "report_generation_completed",
            "report_generation_failed",
            "partial_report_generated",
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
                self._subscribers.setdefault(sid, [])
                self._cancel_flags.setdefault(sid, False)

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
                    prior_title = str(item.get("title", "")).strip()
                    prior_exec = str(item.get("execution_state", "")).strip().lower()
                    self._runs[sid] = state
                    self._subscribers.setdefault(sid, [])
                    self._cancel_flags.setdefault(sid, False)
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
        status = str(session_item.get("status", "failed"))
        execution_state = str(session_item.get("execution_state", status))
        raw_title = str(session_item.get("title", "") or "").strip()
        if raw_title and raw_title.lower() != "recovered session":
            task = raw_title
        else:
            task = "Recovered session"
        tree: Dict[str, Any] = {
            "task": task,
            "plan": {"sub_questions": [], "success_criteria": []},
            "max_depth": 3,
            "rounds": [],
            "final_sufficiency": None,
            "report_status": "pending",
        }
        report_text = None
        report_path = str(session_item.get("report_file_path", "") or "")
        created_at = _now()
        updated_at = _now()
        model = "gpt-4.1"
        report_model = "gpt-5.2"
        max_depth = 3
        max_rounds = 1
        results_per_query = 3
        token_usage: Optional[Dict[str, Any]] = None
        error = None

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
                        tu = trace.get("token_usage", {})
                        if isinstance(tu, dict):
                            token_usage = tu
                        error = str(raw.get("error", "") or "")
                        tree["report_status"] = (
                            "completed"
                            if status == "completed"
                            else ("failed" if status == "failed" else "pending")
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

        if report_path and Path(report_path).exists() and not report_text:
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
            execution_state=execution_state,  # type: ignore[arg-type]
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
