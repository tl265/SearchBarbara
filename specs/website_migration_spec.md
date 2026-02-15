# SearchBarbara Web Migration Specification (MVP + Extensible)

## 1) Objective

Build a local web application for `SearchBarbara` that:

1. accepts a research task in a text input box,
2. runs the existing deep research pipeline,
3. shows **live progress** as a **tree structure** (task → sub-questions → queries → synthesis/sufficiency),
4. displays the final report in the web page when complete,
5. provides a **Download Report** button.

## 2) Scope

### In scope (MVP)
- Single-page web UI.
- Start a new run from browser input.
- Live run updates (verbose-like but structured/tree UI).
- Final report rendering.
- Download `.md` report.
- Keep existing core agent logic as source of truth.

### Out of scope (for now)
- Pause/resume controls in UI.
- Editing sub-questions mid-run.
- Uploading/injecting external files.
- Multi-user auth.
- Cloud deployment hardening.

## 3) Non-functional requirements

- Preserve current behavior and outputs of research engine.
- Avoid large refactors of core logic unless needed for observability.
- Backend should be modular so pause/edit/knowledge-injection can be added later.
- Robust to page refresh during run (status recoverable by run ID).

## 4) Proposed architecture

### Backend
- Python web server (FastAPI preferred).
- Reuse `DeepResearchAgent` from `deep_research_agent.py`.
- Introduce a **RunManager** that executes runs asynchronously and emits structured events.

### Frontend
- Minimal JS/HTML (or React if preferred, but keep simple).
- 3 main regions:
  1. Task input + start button.
  2. Live tree panel.
  3. Final report panel + download button.

### Transport for live updates
- Use **Server-Sent Events (SSE)** for one-way progress stream from backend to browser.
- Polling fallback optional, not required for MVP.

## 5) Backend API spec

### `POST /api/runs`
Create a run.

**Request**
```json
{
  "task": "Should our B2B SaaS expand into Germany in 2026?",
  "max_rounds": 4,
  "results_per_query": 5,
  "model": "gpt-4.1",
  "report_model": "gpt-5.2"
}
```

**Response**
```json
{
  "run_id": "uuid",
  "status": "queued"
}
```

### `GET /api/runs/{run_id}`
Get run snapshot.

**Response**
```json
{
  "run_id": "uuid",
  "status": "queued|running|completed|failed",
  "created_at": "...",
  "updated_at": "...",
  "task": "...",
  "tree": { "...": "see event/tree model below" },
  "report_text": "markdown or null",
  "report_file_path": "path or null",
  "error": "string or null",
  "token_usage": { "...": "same shape as trace token_usage if available" }
}
```

### `GET /api/runs/{run_id}/events` (SSE)
Streams events during execution.

Event types:
- `run_started`
- `plan_created`
- `round_started`
- `sub_question_started`
- `queries_generated`
- `query_started`
- `search_completed`
- `synthesis_completed`
- `sufficiency_completed`
- `run_completed`
- `run_failed`

Each event should include:
- `run_id`
- `timestamp`
- `event_type`
- `payload` (typed fields)

### `GET /api/runs/{run_id}/report/download`
Returns markdown file (`text/markdown`) with filename:
`report_<task-slug>_<timestamp>.md`

## 6) Internal event/tree data model

Represent run progress as a nested tree:

- Root: `task`
  - `plan`:
    - `sub_questions[]`
    - `success_criteria[]`
  - `rounds[]`
    - each round has `questions[]`
      - each question has `query_steps[]`
        - each query step has:
          - `query`
          - `status`
          - `search_error`
          - `selected_results_count`
          - `primary_count`
          - optional synthesis summary
  - `final_sufficiency`
  - `report_status`

This should map closely to existing `trace["rounds"]` so the UI can render without heavy transformations.

## 7) Required refactor in agent code (minimal but important)

Add an **event callback hook** to `DeepResearchAgent`, e.g.:

- constructor arg: `event_callback: Optional[Callable[[dict], None]] = None`
- helper method `_emit(event_type: str, payload: dict)`

Emit events at key lifecycle points (listed above) using existing internal values (`round_i`, `sq`, `query`, selected results, sufficiency output, report completion).
Do not change core reasoning logic; only add observability hooks.

## 8) Frontend behavior spec

### Initial state
- Textarea for task input.
- “Start Research” button.
- Optional advanced fields collapsed (max rounds, results/query, models).

### Running state
- Disable start button for current run.
- Show run status pill.
- Live-updating tree:
  - Task root node.
  - Sub-question nodes.
  - Query child nodes with statuses:
    - queued / running / success / failed / skipped.
  - Show concise metadata (counts/errors).

### Completion state
- Render final markdown report (basic markdown renderer).
- Show download button.
- Show token usage summary table if available.

### Failure state
- Show error banner.
- Keep partial tree visible.

## 9) Future extensibility (design constraints now)

Even though not implemented now, design for:
- `pause_run(run_id)` / `resume_run(run_id)`
- `update_sub_question(run_id, round, old, new)`
- `inject_knowledge(run_id, files|text)`

To enable this later:
- Keep run state in a dedicated `RunState` model, not only transient logs.
- Use explicit node IDs for tree nodes (`round:1|sq:...|query:...`).
- Separate orchestration from pure agent logic (RunManager controls lifecycle).

## 10) Acceptance criteria

1. User can submit a task from web page.
2. Run starts and updates appear live without page reload.
3. UI displays a tree showing decomposition and per-query progress.
4. Final report appears in page on completion.
5. Download button downloads markdown report.
6. Existing CLI still works.
7. No regression to existing trace/report generation behavior.

## 11) Suggested implementation plan for Codex

1. Add `event_callback` + `_emit` to `DeepResearchAgent`.
2. Emit structured events at major lifecycle points.
3. Build `RunManager` with in-memory run registry + background task execution.
4. Implement API endpoints and SSE stream.
5. Build simple frontend (`templates/index.html` + JS) with tree renderer.
6. Add report download endpoint.
7. Validate with manual run and ensure trace/report outputs still generated.
