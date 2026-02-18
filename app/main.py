from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, Iterator, List
import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from app.models import (
    CreateRunRequest,
    CreateRunResponse,
    PatchSessionRequest,
    RunSnapshotResponse,
    SessionListResponse,
)
from app.run_manager import RunManager
from app.sse import format_sse


BASE_DIR = Path(__file__).resolve().parent
WEB_CONFIG_PATH = BASE_DIR.parent / "web_config.json"
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(title="SearchBarbara Web")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


def _load_web_config() -> Dict[str, Any]:
    default = {
        "max_rounds": 1,
        "default_max_depth": 3,
        "default_results_per_query": 3,
        "model": "gpt-4.1",
        "report_model": "gpt-5.2",
        "min_canvas_zoom": 0.45,
        "auto_fit_safety_px": 10,
        "heartbeat_interval_sec": 8,
    }
    if not WEB_CONFIG_PATH.exists():
        return default
    try:
        raw = json.loads(WEB_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return default
    if not isinstance(raw, dict):
        return default
    try:
        max_rounds = int(raw.get("max_rounds", default["max_rounds"]))
    except (TypeError, ValueError):
        max_rounds = default["max_rounds"]
    try:
        default_max_depth = int(
            raw.get("default_max_depth", default["default_max_depth"])
        )
    except (TypeError, ValueError):
        default_max_depth = default["default_max_depth"]
    try:
        default_results_per_query = int(
            raw.get(
                "default_results_per_query", default["default_results_per_query"]
            )
        )
    except (TypeError, ValueError):
        default_results_per_query = default["default_results_per_query"]
    model = str(raw.get("model", default["model"]) or default["model"])
    report_model = str(
        raw.get("report_model", default["report_model"]) or default["report_model"]
    )
    try:
        min_canvas_zoom = float(
            raw.get("min_canvas_zoom", default["min_canvas_zoom"])
        )
    except (TypeError, ValueError):
        min_canvas_zoom = default["min_canvas_zoom"]
    try:
        auto_fit_safety_px = int(
            raw.get("auto_fit_safety_px", default["auto_fit_safety_px"])
        )
    except (TypeError, ValueError):
        auto_fit_safety_px = default["auto_fit_safety_px"]
    try:
        heartbeat_interval_sec = float(
            raw.get("heartbeat_interval_sec", default["heartbeat_interval_sec"])
        )
    except (TypeError, ValueError):
        heartbeat_interval_sec = default["heartbeat_interval_sec"]
    return {
        "max_rounds": max(1, max_rounds),
        "default_max_depth": max(1, min(default_max_depth, 8)),
        "default_results_per_query": max(1, min(default_results_per_query, 30)),
        "model": model,
        "report_model": report_model,
        "min_canvas_zoom": min(max(min_canvas_zoom, 0.1), 0.95),
        "auto_fit_safety_px": max(0, min(auto_fit_safety_px, 200)),
        "heartbeat_interval_sec": min(max(heartbeat_interval_sec, 1.0), 120.0),
    }


WEB_CONFIG = _load_web_config()
run_manager = RunManager(
    heartbeat_interval_sec=float(WEB_CONFIG.get("heartbeat_interval_sec", 8))
)


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index_v2.html",
        {
            "min_canvas_zoom": float(WEB_CONFIG.get("min_canvas_zoom", 0.45)),
            "auto_fit_safety_px": int(WEB_CONFIG.get("auto_fit_safety_px", 10)),
            "default_max_depth": int(WEB_CONFIG.get("default_max_depth", 3)),
            "default_results_per_query": int(
                WEB_CONFIG.get("default_results_per_query", 3)
            ),
        },
    )


@app.get("/legacy", response_class=HTMLResponse)
def index_legacy(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {"min_canvas_zoom": float(WEB_CONFIG.get("min_canvas_zoom", 0.45))},
    )


def _compute_research_status(status: str, error: str | None) -> str:
    if status == "queued":
        return "idle"
    if status == "running":
        return "researching"
    if status == "completed":
        return "complete"
    if status == "failed" and (error or "").strip() == "Run aborted by user":
        return "aborted"
    if status == "failed":
        return "failed"
    return status


def _collect_tree_summaries(tree: Dict[str, Any]) -> tuple[List[str], List[str]]:
    insights: List[str] = []
    syntheses: List[str] = []
    rounds = tree.get("rounds", [])
    if not isinstance(rounds, list):
        return insights, syntheses
    for rnd in rounds:
        if not isinstance(rnd, dict):
            continue
        questions = rnd.get("questions", [])
        if not isinstance(questions, list):
            continue
        for q in questions:
            if not isinstance(q, dict):
                continue
            node_suff = q.get("node_sufficiency", {})
            if isinstance(node_suff, dict):
                reasoning = str(node_suff.get("reasoning", "")).strip()
                if reasoning:
                    insights.append(reasoning)
            steps = q.get("query_steps", [])
            if not isinstance(steps, list):
                continue
            for st in steps:
                if not isinstance(st, dict):
                    continue
                summary = str(st.get("synthesis_summary", "")).strip()
                if summary:
                    syntheses.append(summary)
    return insights, syntheses


def _coverage_note(tree: Dict[str, Any]) -> str:
    rounds = tree.get("rounds", [])
    total_queries = 0
    evidence_queries = 0
    if isinstance(rounds, list):
        for rnd in rounds:
            if not isinstance(rnd, dict):
                continue
            for q in rnd.get("questions", []) if isinstance(rnd.get("questions", []), list) else []:
                if not isinstance(q, dict):
                    continue
                for st in q.get("query_steps", []) if isinstance(q.get("query_steps", []), list) else []:
                    if not isinstance(st, dict):
                        continue
                    total_queries += 1
                    if int(st.get("selected_results_count", 0)) > 0:
                        evidence_queries += 1
    if total_queries == 0:
        return "No search evidence collected yet."
    return (
        f"Evidence-backed queries: {evidence_queries}/{total_queries}. "
        "Coverage may be partial if unresolved branches remain."
    )


def _latest_thought(events: List[Dict[str, Any]]) -> str:
    if not events:
        return ""
    ev = events[-1]
    et = str(ev.get("event_type", ""))
    payload = ev.get("payload", {}) if isinstance(ev.get("payload", {}), dict) else {}
    if et == "sub_question_started":
        return f"Working on: {payload.get('sub_question', '')}"
    if et == "query_started":
        return f"Searching: {payload.get('query', '')}"
    if et == "search_completed":
        return (
            f"Search returned {int(payload.get('selected_results_count', 0))} selected results "
            f"for '{payload.get('query', '')}'."
        )
    if et == "node_decomposed":
        return f"Decomposed into {len(payload.get('children', [])) if isinstance(payload.get('children', []), list) else 0} child tasks."
    if et == "node_decomposition_started":
        return "Node sufficiency failed; decomposing into child tasks."
    if et == "sufficiency_completed":
        return "Pass-level sufficiency check completed."
    if et == "run_completed":
        return "Research completed and report generated."
    if et == "run_aborted":
        return "Research stopped by user request."
    if et == "run_failed":
        return f"Run failed: {payload.get('error', 'unknown error')}"
    if et == "partial_report_generated":
        return "Partial report generated from current findings."
    if et == "run_heartbeat":
        phase = str(payload.get("phase", "")).strip()
        sq = str(payload.get("sub_question", "")).strip()
        query = str(payload.get("query", "")).strip()
        if query:
            return f"Still working ({phase or 'processing'}): {query}"
        if sq:
            return f"Still working ({phase or 'processing'}): {sq}"
        return f"Still working ({phase or 'processing'})."
    return et.replace("_", " ").strip().capitalize()


def _stop_reason(state: Any) -> str:
    if state.status == "failed" and (state.error or "").strip() == "Run aborted by user":
        return "Stopped by user request."
    if state.status == "failed":
        return f"Stopped due to error: {state.error or 'unknown error'}"
    final_suff = state.tree.get("final_sufficiency", {}) if isinstance(state.tree, dict) else {}
    if state.status == "completed" and isinstance(final_suff, dict):
        if bool(final_suff.get("is_sufficient", False)):
            return "Stopped because sufficiency passed."
        return "Stopped after reaching configured search limits."
    return ""


@app.post("/api/runs", response_model=CreateRunResponse)
def create_run(req: CreateRunRequest) -> CreateRunResponse:
    run_id = run_manager.create_run(
        task=req.task,
        max_depth=req.max_depth,
        max_rounds=int(WEB_CONFIG.get("max_rounds", 1)),
        results_per_query=req.results_per_query,
        model=str(WEB_CONFIG.get("model", "gpt-4.1")),
        report_model=str(WEB_CONFIG.get("report_model", "gpt-5.2")),
    )
    return CreateRunResponse(run_id=run_id, status="queued")


@app.get("/api/runs/{run_id}", response_model=RunSnapshotResponse)
def get_run(run_id: str) -> RunSnapshotResponse:
    state = run_manager.get_snapshot(run_id)
    if not state:
        raise HTTPException(status_code=404, detail="Run not found")
    insights, syntheses = _collect_tree_summaries(state.tree)
    report_status = str(state.tree.get("report_status", "pending")) if isinstance(state.tree, dict) else "pending"
    return RunSnapshotResponse(
        run_id=state.run_id,
        session_id=state.session_id or state.run_id,
        title=state.title or state.task,
        status=state.status,
        execution_state=state.execution_state,
        research_status=_compute_research_status(state.status, state.error),
        report_status=report_status,
        created_at=state.created_at,
        updated_at=state.updated_at,
        task=state.task,
        max_depth=state.max_depth,
        max_rounds=state.max_rounds,
        results_per_query=state.results_per_query,
        tree=state.tree,
        facts=[],
        insights=insights,
        syntheses=syntheses,
        stop_reason=_stop_reason(state),
        coverage_note=_coverage_note(state.tree if isinstance(state.tree, dict) else {}),
        latest_thought=_latest_thought(state.events),
        report_text=state.report_text,
        report_file_path=state.report_file_path,
        error=state.error,
        token_usage=state.token_usage,
    )


@app.get("/api/sessions", response_model=SessionListResponse)
def list_sessions() -> SessionListResponse:
    sessions = run_manager.list_sessions()
    return SessionListResponse(sessions=sessions)


@app.get("/api/sessions/{session_id}", response_model=RunSnapshotResponse)
def get_session(session_id: str) -> RunSnapshotResponse:
    state = run_manager.get_snapshot(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    insights, syntheses = _collect_tree_summaries(state.tree)
    report_status = str(state.tree.get("report_status", "pending")) if isinstance(state.tree, dict) else "pending"
    return RunSnapshotResponse(
        run_id=state.run_id,
        session_id=state.session_id or state.run_id,
        title=state.title or state.task,
        status=state.status,
        execution_state=state.execution_state,
        research_status=_compute_research_status(state.status, state.error),
        report_status=report_status,
        created_at=state.created_at,
        updated_at=state.updated_at,
        task=state.task,
        max_depth=state.max_depth,
        max_rounds=state.max_rounds,
        results_per_query=state.results_per_query,
        tree=state.tree,
        facts=[],
        insights=insights,
        syntheses=syntheses,
        stop_reason=_stop_reason(state),
        coverage_note=_coverage_note(state.tree if isinstance(state.tree, dict) else {}),
        latest_thought=_latest_thought(state.events),
        report_text=state.report_text,
        report_file_path=state.report_file_path,
        error=state.error,
        token_usage=state.token_usage,
    )


@app.patch("/api/sessions/{session_id}")
def rename_session(session_id: str, req: PatchSessionRequest) -> dict:
    clean = " ".join(str(req.title or "").split()).strip()
    if not clean:
        raise HTTPException(status_code=422, detail="Title must contain non-whitespace characters.")
    renamed = run_manager.rename_session(session_id, clean)
    if renamed is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session": renamed.model_dump()}


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str) -> dict:
    result = run_manager.delete_session(session_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if result == "conflict_running":
        raise HTTPException(status_code=409, detail="Cannot delete a running session. Abort first.")
    return {"session_id": session_id, "status": "deleted"}


def _event_stream(run_id: str, q: Queue) -> Iterator[str]:
    try:
        while True:
            try:
                event = q.get(timeout=15)
                yield format_sse(event)
            except Empty:
                yield ": keep-alive\n\n"
    finally:
        run_manager.unsubscribe(run_id, q)


@app.get("/api/runs/{run_id}/events")
def stream_events(run_id: str) -> StreamingResponse:
    q = run_manager.subscribe(run_id)
    if q is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return StreamingResponse(
        _event_stream(run_id, q),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.get("/api/runs/{run_id}/report/download")
def download_report(run_id: str) -> FileResponse:
    state = run_manager.get_snapshot(run_id)
    if not state:
        raise HTTPException(status_code=404, detail="Run not found")
    if not state.report_file_path:
        raise HTTPException(status_code=404, detail="Report not available")
    path = Path(state.report_file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report file missing")
    return FileResponse(
        str(path),
        media_type="text/markdown",
        filename=path.name,
    )


@app.post("/api/runs/{run_id}/abort")
def abort_run(run_id: str) -> dict:
    status = run_manager.abort_run(run_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"run_id": run_id, "status": status}


@app.post("/api/runs/{run_id}/report/partial")
def generate_partial_report(run_id: str) -> dict:
    result = run_manager.generate_partial_report(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if "error" in result:
        raise HTTPException(status_code=500, detail=str(result.get("error", "Unknown error")))
    return {"run_id": run_id, **result}


@app.post("/api/runs/{run_id}/report")
def generate_report_now(run_id: str) -> dict:
    result = run_manager.generate_partial_report(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if "error" in result:
        raise HTTPException(status_code=500, detail=str(result.get("error", "Unknown error")))
    return {"run_id": run_id, **result}
