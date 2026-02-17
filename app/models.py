from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


RunStatus = Literal["queued", "running", "completed", "failed"]


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
    status: RunStatus
    research_status: Optional[str] = None
    report_status: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    task: str
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
    status: RunStatus
    created_at: datetime
    updated_at: datetime
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
