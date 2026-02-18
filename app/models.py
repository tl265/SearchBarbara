from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


RunStatus = Literal["queued", "running", "completed", "failed"]
ExecutionState = Literal["idle", "running", "paused", "completed", "failed", "aborted"]


class CreateRunRequest(BaseModel):
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
    execution_state: Optional[ExecutionState] = None
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
    error: Optional[str] = None
    token_usage: Optional[Dict[str, Any]] = None


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
    execution_state: ExecutionState = "idle"
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
    state_file_path: Optional[str] = None
    error: Optional[str] = None
    token_usage: Optional[Dict[str, Any]] = None
    manual_edit_log: List[Dict[str, Any]] = Field(default_factory=list)
    manual_assertions: Dict[str, Any] = Field(default_factory=dict)


class SessionSummary(BaseModel):
    session_id: str
    owner_id: Optional[str] = None
    title: str
    status: RunStatus
    execution_state: ExecutionState
    created_at: datetime
    updated_at: datetime
    state_file_path: Optional[str] = None
    report_file_path: Optional[str] = None
    has_manual_edits: bool = False


class SessionListResponse(BaseModel):
    sessions: List[SessionSummary] = Field(default_factory=list)


class PatchSessionRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
