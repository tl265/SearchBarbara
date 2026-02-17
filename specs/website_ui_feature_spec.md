# SearchBarbara UI Feature Specification v02 (Thinking-Partner Experience)

## 1. Purpose

Define **version 02** of the UI-focused feature spec for SearchBarbara’s web experience.

This document is intentionally separate from `specs/website_migration_spec.md` and the earlier `specs/website_ui_feature_spec.md`. It should be treated as the v02 feature roadmap for interaction quality, dynamic visualization, and graceful run control.

---

## 2. Product experience goals

The website should feel like a thoughtful research partner:

1. Start with a minimal, elegant chatbox.
2. Show live reasoning progress on a fluid canvas.
3. Expand research visually as a waterfall:
   - first decomposition layer,
   - then deeper traversal from the left-most active child,
   - with clear branch status at all times.
4. Communicate why the agent continues, pivots, or stops.
5. Let the user gracefully abort research at any time.
6. Always allow report generation from accumulated findings.

---

## 3. User journeys

### Journey A: Full completion
1. User enters research question.
2. Research starts, canvas progressively expands.
3. Agent emits dynamic "thinking out loud" messages.
4. Sufficiency test passes (or max-depth reached).
5. UI explains stop rationale in natural language.
6. Report auto-generates and renders in markdown.
7. User can download report.

### Journey B: User abort + partial report
1. User starts research.
2. User presses **Stop research** while run is active.
3. UI transitions to **Stopping…** and gracefully halts.
4. Partial graph remains visible with stopped markers.
5. User can trigger **Generate report from current findings**.
6. Report renders in markdown with coverage note.

---

## 4. Information architecture

## 4.1 Top-level layout
- **Header bar**: run status, current phase, token usage summary.
- **Prompt dock**: input + start/stop controls.
- **Research canvas**: interactive branch visualization.
- **Thought stream rail**: short dynamic narration.
- **Report panel**: rendered markdown + raw toggle + download.

## 4.2 States
- Idle
- Researching
- Aborting
- Aborted
- ReportGenerating
- Complete
- Failed

---

## 5. Feature requirements

## 5.1 Prompt dock
- Single multiline chat-style input.
- Primary CTA: `Start Research`.
- During run: replace CTA with `Stop Research`.
- Optional advanced controls collapsed by default.

## 5.2 Graceful abort
- Abort is cooperative, not hard-kill.
- On click:
  - button transitions to `Stopping…`,
  - current atomic step can complete,
  - no new deeper tasks are scheduled,
  - run becomes `aborted` when safe checkpoint reached.

## 5.3 Waterfall canvas behavior
- Root node appears first.
- Decomposition layer fans out horizontally.
- Active exploration deepens left-most active branch by default.
- Auto-pan keeps active node in view.
- Inactive branches are dimmed, not hidden.
- Node status visible via icon + color + label.

Required statuses:
- `queued`, `running`, `success`, `failed`, `skipped`, `stopped`, `dead_end`, `sufficient`.

## 5.4 Canvas data visibility (research + debug)
- Each query node should support expandable details for:
  - lazy query text / deferred query plan when applicable,
  - returned synthesized result summary,
  - selected source counts and key evidence snippets.
- Query nodes should persist these details after completion/abort so users can audit paths later.
- Add a debug-friendly freshness indicator for each query (reuse existing freshness logic/fields already produced by backend; do not replace with a new scoring system).
- Freshness should be visible at node-level (compact badge) and in expanded details (raw freshness metadata/timestamp window if available).

## 5.5 Dynamic thinking narration
- Narration must not be a fixed sentence set only.
- Every narration message must map to real run context.
- Tone: collaborative, concise, transparent.

Stop rationale must explain one of:
- sufficiency passed,
- max depth/round reached,
- dead-end pivot,
- user-aborted.

## 5.6 Report behavior
- If research completes naturally: report generation is automatic.
- If research is aborted: report generation is manual via CTA.
- Report is always derived from facts/insights/synthesis available so far.
- Report panel supports:
  - rendered markdown,
  - raw markdown toggle,
  - report download (`.md`),
  - token usage display.

---

## 6. Backend contracts required by UI features

## 6.1 API endpoints
- `POST /api/runs` — create run.
- `GET /api/runs/{run_id}` — snapshot.
- `GET /api/runs/{run_id}/events` — SSE stream.
- `POST /api/runs/{run_id}/abort` — graceful abort.
- `POST /api/runs/{run_id}/report` — generate report now.
- `GET /api/runs/{run_id}/report/download` — download markdown.

## 6.2 Snapshot fields
Snapshot must include at minimum:
- `status`, `research_status`, `report_status`
- `tree`
- `facts`, `insights`, `syntheses`
- `report_text`
- `stop_reason`
- `coverage_note`
- `token_usage` (research/report/total + by model)
- `latest_thought`
- query-level detail fields needed by canvas (`lazy_query`, synthesis summary, freshness metadata)

## 6.3 SSE event types
Minimum event set:
- `run_started`
- `plan_created`
- `round_started`
- `sub_question_started`
- `queries_generated`
- `query_started`
- `search_completed`
- `synthesis_completed`
- `sufficiency_completed`
- `abort_requested`
- `abort_acknowledged`
- `run_aborted`
- `report_generation_started`
- `report_generation_completed`
- `report_generation_failed`
- `run_completed`
- `run_failed`
- `narration_event`

---

## 7. UI copy and narrative quality requirements

- Avoid repetitive fixed strings.
- Prefer short first-person collaborative phrasing.
- Ensure stop messages include explicit reason.
- If report is partial, show clear coverage disclaimer.

Example stop messages (illustrative only):
- "Great, I’ve found enough evidence to conclude confidently."
- "I’m stopping here—this branch is a dead end and needs a different angle."
- "Stopping now as requested. I’ll summarize what we’ve learned so far."

---

## 8. Visual quality requirements

- Clean neutral canvas with restrained accents.
- Typography-first readability.
- Motion should feel smooth, not flashy.
- Reduced motion mode must preserve clarity.

Suggested token ranges:
- micro transitions: 150–300ms
- branch transitions: 300–700ms

---

## 9. Accessibility and resilience

- Keyboard-operable controls (start, stop, generate report).
- Sufficient contrast for all statuses.
- `aria-live` for important status updates and stop rationale.
- Page refresh must restore run state from run ID.

---

## 10. Acceptance criteria

Use stable IDs so reviewers can reference each requirement unambiguously.

- **AC-01**: User can start research from chatbox input.
- **AC-02**: UI streams live progress without reload.
- **AC-03**: Canvas shows decomposition then deepening waterfall traversal.
- **AC-04**: Active branch is obvious at all times.
- **AC-05**: User can gracefully abort from running state.
- **AC-06**: Aborted run can still produce report from partial findings.
- **AC-07**: Completed run auto-generates report.
- **AC-08**: Report renders in markdown and can be downloaded.
- **AC-09**: Token usage is shown for available phases.
- **AC-10**: Stop reason is displayed in natural, contextual language.
- **AC-11**: Canvas exposes query-level details for debugging, including lazy queries/deferred plans, synthesized outputs, and freshness indicators based on existing backend freshness metadata.

---

## 11. Implementation notes

- Keep core reasoning logic unchanged except observability/abort hooks.
- Maintain stable node IDs for smooth rendering.
- Separate orchestration (`RunManager`) from reasoning engine.
- Keep this spec as feature-level guidance; migration/architecture details can remain in the original migration spec.

