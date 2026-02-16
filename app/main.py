import json
from pathlib import Path
from queue import Empty, Queue
from typing import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from app.models import CreateRunRequest, CreateRunResponse, RunSnapshotResponse
from app.run_manager import RunManager
from app.sse import format_sse


BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
run_manager = RunManager()

app = FastAPI(title="SearchBarbara Web")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html", {})


@app.post("/api/runs", response_model=CreateRunResponse)
def create_run(req: CreateRunRequest) -> CreateRunResponse:
    run_id = run_manager.create_run(
        task=req.task,
        max_depth=req.max_depth,
        max_rounds=req.max_rounds,
        results_per_query=req.results_per_query,
        model=req.model,
        report_model=req.report_model,
    )
    return CreateRunResponse(run_id=run_id, status="queued")


@app.get("/api/runs/{run_id}", response_model=RunSnapshotResponse)
def get_run(run_id: str) -> RunSnapshotResponse:
    state = run_manager.get_snapshot(run_id)
    if not state:
        raise HTTPException(status_code=404, detail="Run not found")
    return RunSnapshotResponse(
        run_id=state.run_id,
        status=state.status,
        created_at=state.created_at,
        updated_at=state.updated_at,
        task=state.task,
        tree=state.tree,
        report_text=state.report_text,
        report_file_path=state.report_file_path,
        error=state.error,
        token_usage=state.token_usage,
    )


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
