# Spec Draft: Context-Grounded Deep Research Integration

## Summary
Upgrade SearchBarbara so uploaded user files become first-class evidence in the research lifecycle, with parsing deferred until run start and context propagated recursively through node exploration/decomposition/sufficiency/reporting.

Chosen defaults:
- Child-node context routing: **LLM relevance filter**
- Prompt payload form: **structured digest only**
- UI behavior when `ui_debug=false`: **hide digest pane entirely**

---

## 1) Goals and Scope

### Goals
1. Parse uploaded files **only after** user clicks `Start Research`.
2. Provide context to every node:
   - root gets full parsed context
   - child nodes get routed, relevant context slices.
3. Include context in:
   - node-level sufficiency
   - decomposition decisions
   - final report writing (full context again).
4. Expose context-processing progress in Thought Stream.
5. Debug UI: digest details only when `ui_debug=true`; during node execution show node-received context in that pane.

### Out of Scope
- Replacing current search pipeline.
- Replacing current context digest schema.
- Building a persistent vector index (future option).

---

## 2) Current-State Constraints (for implementer)
- Context preprocessing currently runs in `RunManager`.
- Research logic is in `deep_research_agent.py`.
- Worker startup now includes binding phase events (`context_binding_*`).
- Context currently parsed during bind; this spec formalizes that parse-on-start behavior and then routes context inside agent execution.

---

## 3) Required Architecture Changes

### 3.1 Parsing lifecycle
- Upload stage:
  - store raw files + metadata only.
  - no per-file/aggregate LLM parsing.
- Start stage:
  - in worker bind phase, parse files into digest(s).
  - emit progress events.
- Persist parsed context artifacts in session state for downstream node/report use.

### 3.2 Context object contracts
Define a normalized internal structure used by the agent:

- `ContextBundle` (session-level):
  - `session_context_id`
  - `revision`
  - `files[]` (metadata + per-file digest refs)
  - `aggregate_digest` (structured JSON)
  - `routing_index` (derived lightweight index from digest: claims/tags/entities/source refs)
- `NodeContextSlice` (node-level):
  - `node_id`
  - `selection_reason`
  - `selected_items[]` (facts/uncertainties/etc with source refs)
  - `coverage_note`
  - `dropped_items_count`

---

## 4) Agent Pipeline Integration

### 4.1 Node entry
When entering any node:
- Root:
  - receives full `ContextBundle.aggregate_digest` + top-level file metadata.
- Non-root:
  - call context-router LLM (`context_router` stage) with:
    - original task
    - parent task
    - current node task
    - node depth
    - parent node context slice
    - unresolved gaps (if any)
  - router outputs `NodeContextSlice` for this node.

Emit event:
- `context_for_node_ready` with `{ node_id, depth, selected_items_count, mode: root|routed }`.

### 4.2 Lazy query stage
No hard requirement to inject context into query generation in this phase unless explicitly configured. Default:
- keep query generation behavior unchanged.
- context participates starting at synthesis/sufficiency/decompose as below.

### 4.3 Node synthesis + sufficiency
Node sufficiency prompt input must include:
- node question
- synthesized web findings
- `NodeContextSlice`
- explicit instruction to judge sufficiency against both web evidence + provided context evidence.

Output unchanged shape (`is_sufficient`, `reasoning`, `gaps`) but reasoning should reference context when used.

### 4.4 Decomposition with context routing
When node is insufficient:
1. Decompose prompt receives:
   - node task
   - insufficiency gaps
   - `NodeContextSlice`
2. Decomposer returns children (MECE as already required).
3. For each child, run router step to derive child slice from parent slice.
4. Recurse DFS with child slice.

Emit events:
- `context_routing_started`
- `context_routing_completed`
- `context_routing_failed` (non-fatal fallback described below)

### 4.5 Final report stage
Report writer receives full context again:
- full aggregate digest
- optional per-file digest summaries
- normal findings/search stats.
Instruction: use user context as supplemental evidence; preserve source attribution style.

---

## 5) Prompt and Model Contracts

### 5.1 New/updated prompt files
- Keep existing:
  - `prompts/context_preprocess_common.system.txt`
  - `prompts/context_preprocess_per_file.system.txt`
  - `prompts/context_preprocess_aggregate.system.txt`
- Add new:
  - `prompts/context_router.system.txt` (node slice selection)
- Update existing:
  - node sufficiency system prompt (or the prompt text builder if inline) to include user context usage rules.
  - `prompts/decompose.system.txt` to explicitly use node context for MECE child planning.
  - `prompts/report.system.txt` add clause: include user context evidence where relevant.

### 5.2 Router output JSON schema (strict)
```json
{
  "selected_items": [
    {
      "kind": "fact|uncertainty|analysis|constraint|case|definition|lead|conflict",
      "text": "...",
      "sources": ["file::S1"],
      "why_relevant": "..."
    }
  ],
  "selection_reason": "...",
  "coverage_note": "..."
}
```

Fallback if malformed:
- deterministic top-N from aggregate facts + keyword overlap.

---

## 6) Eventing and Thought Stream

### 6.1 Required new events
- `context_binding_parsing_started`
- `context_binding_parsing_file_completed`
- `context_binding_parsing_completed`
- `context_for_node_ready`
- `context_routing_started`
- `context_routing_completed`
- `context_routing_failed`
- `report_context_attached`

### 6.2 Thought stream mapping
Each above event must have human-readable narration in `app/static/app_v2.js`.
Examples:
- “Parsing context file 2/5…”
- “Prepared context slice for node 1.2 (12 items).”
- “Routing context to child tasks…”
- “Attached full user context for report writing.”

---

## 7) UI/Debug Behavior

### 7.1 When `ui_debug=false`
- Hide right-side “Aggregate Digest” JSON pane entirely.
- Keep file list/status visible.

### 7.2 When `ui_debug=true`
- Show digest pane.
- Default content:
  - during startup parsing: parsing progress status + file counters.
  - when entering node: show that node’s `NodeContextSlice` JSON in pane.
  - when report starts: show indicator that full context attached.

### 7.3 UX consistency
- Context pane should not require manual refresh.
- All updates via SSE/snapshot as in current pattern.

---

## 8) Error Handling Rules

1. File-level parse failures are isolated:
- failed file marked `error`.
- aggregate built from successfully parsed files when any exist.
2. Context router failures:
- do not fail run.
- fallback to parent slice passthrough (or top-N deterministic).
- emit `context_routing_failed`.
3. No-context case:
- if all files fail parse, continue run with “no usable context” warning in thought stream.

---

## 9) API / State / Persistence Changes

### 9.1 Session state additions
In state tree (or structured state section):
- `context_processing`:
  - `status`
  - `files_total`
  - `files_ready`
  - `files_error`
- per-node:
  - `context_slice_ref` or inline compact slice (debug mode may include full slice)

### 9.2 No breaking REST changes required
Optional additions:
- include context processing summary in `GET /api/sessions/{id}` snapshot payload for easier UI rendering.

---

## 10) Test Plan

### Functional
1. Upload files, no start:
- no LLM parse calls triggered.
2. Start run:
- parsing starts in worker; events emitted.
3. Root node:
- receives full context.
4. Child node:
- receives routed slice, not full context.
5. Sufficiency:
- reflects combined web + context evidence.
6. Decompose:
- uses context and routes to each child recursively.
7. Report:
- receives full context and cites accordingly.

### Failure scenarios
1. One of multiple files fails parse:
- run continues; aggregate from successful files.
2. All files fail parse:
- run continues with no-context warning.
3. Router malformed output:
- fallback logic invoked, run continues.

### UI
1. `ui_debug=false` hides digest pane.
2. `ui_debug=true` shows:
- parsing progress
- per-node context slice on node entry
- report context attach status.

---

## 11) Assumptions and Defaults

- Context routing uses LLM-based selector (default model = same as decompose model unless separately configured).
- Context payload passed to prompts is structured digest only (not raw file bodies).
- Query generation remains unchanged for now; context enters at sufficiency/decompose/report.
- Existing source citation format remains unchanged.

---

## 12) Implementation Sequence (recommended)

1. Introduce context-router prompt + schema validator + fallback.
2. Move upload stage to metadata-only (disable upload-time parse).
3. Ensure start-time parse pipeline emits detailed events.
4. Wire node context propagation in agent (root full, child routed).
5. Update sufficiency/decompose/report prompt builders to include context payloads.
6. Add UI debug behavior switches + node-slice rendering in digest pane.
7. Add tests and regression checks.
