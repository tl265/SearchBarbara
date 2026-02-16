import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue
from typing import Any, Dict, List, Optional

from app.models import RunState
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
    def __init__(self) -> None:
        self._runs: Dict[str, RunState] = {}
        self._subscribers: Dict[str, List[Queue]] = {}
        self._cancel_flags: Dict[str, bool] = {}
        self._lock = threading.Lock()

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
            status="queued",
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

        t = threading.Thread(
            target=self._execute_run,
            args=(run_id,),
            daemon=True,
        )
        t.start()
        return run_id

    def abort_run(self, run_id: str) -> Optional[str]:
        with self._lock:
            state = self._runs.get(run_id)
            if not state:
                return None
            if state.status in ("completed", "failed"):
                return state.status
            self._cancel_flags[run_id] = True
            state.updated_at = _now()
            for q in self._subscribers.get(run_id, []):
                q.put(
                    {
                        "run_id": run_id,
                        "timestamp": _now().isoformat(),
                        "event_type": "run_abort_requested",
                        "payload": {},
                    }
                )
            return "aborting"

    def generate_partial_report(self, run_id: str) -> Optional[Dict[str, str]]:
        snapshot = self.get_snapshot(run_id)
        if not snapshot:
            return None
        report_text, partial_usage = self._build_partial_report_with_agent(snapshot)
        if not report_text.strip():
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
            if partial_usage:
                state.token_usage = self._merge_usage_dicts(
                    state.token_usage, partial_usage
                )
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
        return {"report_file_path": str(report_path)}

    def get_snapshot(self, run_id: str) -> Optional[RunState]:
        with self._lock:
            state = self._runs.get(run_id)
            if not state:
                return None
            return RunState.model_validate(state.model_dump())

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
        with self._lock:
            state = self._runs.get(run_id)
            if not state:
                return
            state.events.append(event)
            state.updated_at = _now()
            self._project_event(state, event)
            for q in self._subscribers.get(run_id, []):
                q.put(event)

    def _project_event(self, state: RunState, event: Dict[str, Any]) -> None:
        et = event.get("event_type", "")
        payload = event.get("payload", {})
        tree = state.tree

        if et == "run_started":
            state.status = "running"
            tree["report_status"] = "running"
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
            rnd = self._ensure_round(tree, round_i)
            qnode = self._ensure_question(rnd, sq)
            qnode["depth"] = depth
            qnode["parent"] = parent
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
            qnode["status"] = "decomposed"
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
            tree["report_status"] = "completed"
            return
        if et == "run_failed":
            state.status = "failed"
            state.error = str(payload.get("error", "unknown error"))
            tree["report_status"] = "failed"
            return
        if et == "run_aborted":
            state.status = "failed"
            state.error = str(payload.get("error", "Run aborted by user"))
            tree["report_status"] = "failed"
            return
        if et == "partial_report_generated":
            tree["report_status"] = "partial_ready"
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
        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        slug = slugify_for_filename(task)
        report_path = reports_dir / f"report_{slug}_{ts}.md"

        def callback(event: Dict[str, Any]) -> None:
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
                    state.tree["report_status"] = "completed"
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
                state.error = error_text
                state.updated_at = _now()
                state.tree["report_status"] = "failed"
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
