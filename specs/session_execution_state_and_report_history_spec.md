# Session Execution State + Report History Spec (v1)

## 1. Objective

Define a clear, enforceable session lifecycle model and UI behavior for:

- Research execution states (`idle/running/paused/completed/aborted/failed`)
- Power-user research controls (`pause/resume/abort`)
- Report generation behavior decoupled from research execution
- Multi-version report history within a single session (`<1/2>`, `<2/2>`, ...)

This spec is the source of truth for implementation and QA.

---

## 2. Design Principles

1. A session has two distinct phases:
- `research`
- `report_generation`

2. Research and report are related but not the same control surface:
- Research controls whether new evidence is collected.
- Report generation consumes current findings and can be repeated.

3. Abort is a heavy action:
- Must require confirmation.
- Not shown as a prominent default control.

4. Pause is a power-user control:
- Available, but not in plain sight.

---

## 3. Session Execution States

Allowed `execution_state` values:

- `idle`
- `running`
- `paused`
- `completed`
- `aborted`
- `failed`

### 3.1 State semantics

- `idle`: session exists but research has not started.
- `running`: research actively executing.
- `paused`: research suspended but still alive and resumable.
- `completed`: research ended naturally (terminal for research).
- `aborted`: research manually stopped by user (terminal for research).
- `failed`: research ended due to runtime error (terminal for research).

### 3.2 Alive vs terminal

Alive states:
- `running`
- `paused`

Terminal research states:
- `completed`
- `aborted`
- `failed`

`idle` is pre-research (neither alive nor terminal).

---

## 4. Research State Transitions

Allowed transitions:

- `idle -> running` (start research)
- `running -> paused` (pause)
- `paused -> running` (resume)
- `running -> completed` (natural stop)
- `running -> aborted` (manual abort with confirm)
- `paused -> aborted` (manual abort with confirm)
- `running -> failed` (runtime failure)
- `paused -> failed` (runtime failure)

Disallowed transitions:

- Any terminal (`completed/aborted/failed`) -> `running/paused`
- `completed -> aborted` and similar terminal rewrites

Note:
- Once research becomes terminal, no further search execution is allowed in that session.

---

## 5. Report Generation Model

Report generation is independent from research execution status, with policy constraints.

## 5.1 Auto report policy

- Auto-generate exactly one report when research reaches natural terminal state:
  - Trigger: `running -> completed`
- No repeated auto-generation while staying in `completed`.
- Since `completed` cannot resume research, this yields at most one auto report for that session research lifecycle.

## 5.2 Manual report policy

Manual report generation is allowed when `execution_state` is:

- `paused`
- `completed`
- `aborted`
- `failed`

Manual report generation is not allowed in:

- `idle` (no findings yet)
- `running` (avoid concurrent ambiguity and unstable snapshot semantics)

---

## 6. Report History

Every successful report generation creates a new immutable report version.

Includes:
- auto-generated report at natural stop
- each successful manual re-generation

### 6.1 Versioning UX

- Provide navigator in report panel:
  - e.g., `<1/2>`, `<2/2>`
- User can switch between historical report versions.
- Download button exports currently selected version.

### 6.2 Version metadata

Each version must persist:
- `version_index` (1-based)
- `created_at`
- `trigger_type` (`auto_natural_stop` | `manual`)
- `report_text`
- `report_file_path` (if materialized)
- optional usage/cost snapshot tied to this generation

---

## 7. UI Behavior Requirements

## 7.1 Primary controls

- Keep primary visible control: `Start Research` / state-dependent equivalent.
- `Pause` and `Abort` are power-user controls:
  - placed in overflow menu, advanced controls, or equivalent secondary UI.

## 7.2 Abort confirmation

- Abort must show confirmation modal/warning.
- Copy must clearly state irreversibility for research:
  - abort ends research for this session permanently.

## 7.3 Task input locking

- When a concrete session is active (newly started or loaded), task input is locked/greyed.
- In `New Session` workspace (no active session id), task input is unlocked.

## 7.4 Loading old sessions

When loading a session, UI must restore:
- `execution_state`
- advanced options (`max_depth`, `results_per_query`, and configured `max_rounds` visibility semantics)
- current selected/latest report version and version count

---

## 8. API/Data Contract Changes

Minimum backend contract additions:

1. Session snapshot response must include:
- `execution_state`
- research config fields (`max_depth`, `max_rounds`, `results_per_query`)
- report history collection:
  - `report_versions[]`
  - `current_report_version_index`

2. Report generation endpoints must append versions instead of overwriting single `report_text/report_file_path`.

3. Download endpoint must accept or resolve version selection:
- default to current selected version
- explicit version selector support recommended

4. SSE events should include version metadata on successful generation.

---

## 9. Persistence Requirements

Session persistence must retain:
- execution state transitions
- report history list (all versions)
- currently selected version (if persisted server-side)

Crash/restart recovery:
- terminal states remain terminal
- paused remains paused if persisted as such
- report history remains fully replayable

---

## 10. Acceptance Criteria

### AC-1 Research lifecycle
- User can start -> pause -> resume.
- User can abort from running/paused with confirmation.
- After abort, research cannot restart.

### AC-2 Natural completion
- On natural stop, session enters `completed`.
- Exactly one auto report is generated for that transition.

### AC-3 Manual reports
- Manual report generation works in paused/completed/aborted/failed.
- Each successful generation adds one history version.

### AC-4 Report navigator
- UI displays version position (e.g., `2/5`).
- User can view older/newer versions.
- Download exports currently selected version.

### AC-5 Session restore
- Loading a session restores execution state, advanced options, and report history.
- Task input lock state matches session/new-session rules.

---

## 11. Non-goals (v1)

- Branching/editing report versions.
- Resuming research after terminal state.
- Collaborative multi-user locking semantics.

---

## 12. Known Risks

- Snapshot freshness race in multi-window usage can present stale state transiently.
- This is tracked in `specs/known_issues.md` and should be addressed by fresh-first snapshot semantics.
