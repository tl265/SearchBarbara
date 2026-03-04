# Report Template Library v1 Spec

## 1. Objective

Add a user-facing report template system that separates report prompt concerns into:

1. **Background prompt** (audience/scenario specific)
2. **Universal writing system prompt** (shared writing/rubric constraints)

The final report system prompt is composed at runtime as:

`[Background Prompt] + [Universal Writing Prompt]`

Users can choose and define templates in the Report panel.

## 2. Scope Decisions

- Template selection applies to **manual report generation only** (`Generate report from current findings`).
- Auto-final report keeps default behavior: **executive background + universal writing prompt**.
- Template storage scope: **per-user shared library**.
- Editing model: **structured fields + composed-prompt preview**.

## 3. Prompt Assets

Add prompt files:

- `prompts/report.universal.system.txt`
- `prompts/report.background.executive.txt`
- `prompts/report.background.business_head_execution.txt`

Compatibility:

- Existing `prompts/report.system.txt` remains for legacy/fallback paths.

## 4. Template Schema

Per template record:

- `template_id: str`
- `name: str`
- `background_type: "executive" | "business_head_execution" | "custom"`
- `is_builtin: bool`
- `is_default_manual: bool`
- `fields`:
  - `audience: str`
  - `presentation_setup: str`
  - `dos: list[str]`
  - `donts: list[str]`
  - `tone: str`
  - `focus: str`
- `rendered_background_prompt: str`
- `created_at: datetime | null`
- `updated_at: datetime | null`

## 5. Persistence

Store custom templates under:

- `runs/templates/users/<owner_hash>.json`

Store payload:

- `schema_version`
- `owner_id`
- `default_manual_template_id`
- `templates[]` (custom templates only)
- `updated_at`

Built-in templates are synthesized by backend and always available.

## 6. Backend API Changes

### 6.1 New Endpoints

- `GET /api/report/templates`
- `POST /api/report/templates`
- `PUT /api/report/templates/{template_id}`
- `DELETE /api/report/templates/{template_id}`
- `POST /api/report/templates/preview`

### 6.2 Changed Endpoint

- `POST /api/runs/{run_id}/report`
  - Accept optional `template_id`.
  - If omitted/invalid, fallback to default template.
  - Include `template_id` in idempotency payload.

## 7. Backend Runtime Flow

1. UI sends selected `template_id` with manual report request.
2. Run manager resolves template (builtin/custom) for current owner.
3. Run manager composes system prompt:
   - rendered background + universal prompt.
4. Report generation uses composed system prompt.
5. Report events include template metadata (`template_id`, optional `template_name`).

## 8. UI Changes (Report Panel)

Add:

- Template selector dropdown.
- "Manage" entry to open template editor.
- Structured editor fields:
  - name
  - background type
  - audience
  - presentation setup
  - do's (line list)
  - don'ts (line list)
  - tone
  - focus
- Actions:
  - New
  - Save
  - Delete
  - Preview
- Read-only preview showing composed system prompt.

Behavior:

- Selector controls manual report request payload (`template_id`).
- Controls are disabled while report generation is active.

## 9. Validation & Guardrails

- Name required; max length enforced.
- Field lengths capped.
- `dos/donts` count and item length capped.
- Built-in templates cannot be deleted/overwritten.
- Ownership checks enforced for all template CRUD.

## 10. Observability

Report events should include template metadata where applicable:

- `report_generation_started`
- `report_generation_completed`
- `report_generation_failed`
- `partial_report_generated`

## 11. Test Plan

### Backend

1. Built-in templates returned for authenticated users.
2. Create/update/delete custom template works per owner.
3. Invalid template input returns 4xx.
4. Manual report with template_id uses selected prompt.
5. Missing template_id falls back to default.
6. Idempotency differentiates requests by template_id.

### UI

1. Template list loads and selector reflects defaults.
2. Structured editor CRUD works.
3. Preview shows composed prompt.
4. Manual report request includes selected template_id.
5. Controls disable correctly while report is running.

## 12. Backward Compatibility

- Existing sessions and report generation continue working without template selection.
- Auto-final report behavior unchanged in v1.
