from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


RunStatus = Literal["queued", "running", "completed", "failed"]
ExecutionState = Literal["idle", "running", "paused", "completed", "failed", "aborted"]
ReportTriggerType = Literal["auto_natural_stop", "manual", "manual_legacy"]


class CreateRunRequest(BaseModel):
    task: str
    max_depth: int = Field(default=3, ge=1)
    max_rounds: int = Field(default=1, ge=1)
    results_per_query: int = Field(default=3, ge=1)
    model: str = "gpt-4.1"
    report_model: str = "gpt-5.2"


class StartFromWorkspaceRequest(BaseModel):
    workspace_id: str = Field(min_length=1, max_length=120)
    task: str
    max_depth: int = Field(default=3, ge=1)
    max_rounds: int = Field(default=1, ge=1)
    results_per_query: int = Field(default=3, ge=1)
    model: str = "gpt-4.1"
    report_model: str = "gpt-5.2"


class CreateRunResponse(BaseModel):
    run_id: str
    status: RunStatus


class RunSnapshotResponse(BaseModel):
    run_id: str
    session_id: Optional[str] = None
    title: Optional[str] = None
    status: RunStatus
    version: int = 1
    execution_state: Optional[ExecutionState] = None
    research_state: Optional[str] = None
    report_state: Optional[str] = None
    terminal_reason: Optional[str] = None
    research_status: Optional[str] = None
    report_status: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    task: str
    max_depth: int = Field(default=3, ge=1)
    max_rounds: int = Field(default=1, ge=1)
    results_per_query: int = Field(default=3, ge=1)
    tree: Dict[str, Any] = Field(default_factory=dict)
    facts: List[Dict[str, Any]] = Field(default_factory=list)
    insights: List[str] = Field(default_factory=list)
    syntheses: List[str] = Field(default_factory=list)
    stop_reason: Optional[str] = None
    coverage_note: Optional[str] = None
    latest_thought: Optional[str] = None
    report_text: Optional[str] = None
    report_file_path: Optional[str] = None
    report_versions: List[Dict[str, Any]] = Field(default_factory=list)
    current_report_version_index: Optional[int] = None
    error: Optional[str] = None
    token_usage: Optional[Dict[str, Any]] = None
    snapshot_source: Optional[str] = None
    snapshot_fallback_reason: Optional[str] = None
    snapshot_lock_wait_ms: Optional[float] = None
    worker_thread_seen: Optional[bool] = None


class RunEvent(BaseModel):
    run_id: str
    timestamp: datetime
    event_type: str
    payload: Dict[str, Any] = Field(default_factory=dict)


class RunState(BaseModel):
    run_id: str
    session_id: Optional[str] = None
    owner_id: Optional[str] = None
    title: Optional[str] = None
    status: RunStatus
    version: int = 1
    execution_state: ExecutionState = "idle"
    research_state: Optional[str] = None
    report_state: Optional[str] = None
    terminal_reason: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    last_checkpoint_at: Optional[datetime] = None
    has_manual_edits: bool = False
    task: str
    max_depth: int
    max_rounds: int
    results_per_query: int
    model: str
    report_model: str
    tree: Dict[str, Any] = Field(default_factory=dict)
    events: List[Dict[str, Any]] = Field(default_factory=list)
    report_text: Optional[str] = None
    report_file_path: Optional[str] = None
    report_versions: List[Dict[str, Any]] = Field(default_factory=list)
    current_report_version_index: Optional[int] = None
    state_file_path: Optional[str] = None
    error: Optional[str] = None
    token_usage: Optional[Dict[str, Any]] = None
    manual_edit_log: List[Dict[str, Any]] = Field(default_factory=list)
    manual_assertions: Dict[str, Any] = Field(default_factory=dict)
    snapshot_source: Optional[str] = None
    snapshot_fallback_reason: Optional[str] = None
    snapshot_lock_wait_ms: Optional[float] = None
    worker_thread_seen: Optional[bool] = None


class SessionSummary(BaseModel):
    session_id: str
    owner_id: Optional[str] = None
    title: str
    status: RunStatus
    version: int = 1
    execution_state: ExecutionState
    research_state: Optional[str] = None
    report_state: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    state_file_path: Optional[str] = None
    report_file_path: Optional[str] = None
    report_versions_count: int = 0
    latest_report_at: Optional[datetime] = None
    has_manual_edits: bool = False
    lock_debug: Optional[Dict[str, Any]] = None


class SessionListResponse(BaseModel):
    sessions: List[SessionSummary] = Field(default_factory=list)


class PatchSessionRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)


ContextDigestStatus = Literal["ready", "stale", "parsing", "error"]


class ContextFile(BaseModel):
    file_id: str
    filename: str
    mime_type: str
    size_bytes: int
    uploaded_at: datetime
    content_hash: str
    extracted_text_hash: str = ""
    chunking_version: str = ""
    spans_hash: str = ""
    prompt_version: str = ""
    digest_status: ContextDigestStatus = "stale"
    digest_hash: str = ""
    digest_ref: str = ""
    error: str = ""


class ContextSet(BaseModel):
    context_set_id: str
    session_id: str
    revision: int
    updated_at: datetime
    files: List[ContextFile] = Field(default_factory=list)
    aggregate_digest_status: ContextDigestStatus = "stale"
    aggregate_digest_ref: str = ""
    aggregate_digest_hash: str = ""
    aggregate_error: str = ""
    last_diff: Dict[str, int] = Field(default_factory=dict)


class ContextSetResponse(BaseModel):
    context_set: ContextSet


class ContextFileDetailResponse(BaseModel):
    file: ContextFile
    download_url: str


class ContextDigestResponse(BaseModel):
    digest: Dict[str, Any] = Field(default_factory=dict)


class ContextMutateResponse(BaseModel):
    context_set: ContextSet
    diff: Dict[str, int] = Field(default_factory=dict)
