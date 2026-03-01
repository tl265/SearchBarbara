# Pin-First Planning-to-Research Spec (v1)

## 1. Goal

Define a two-phase workflow with strict planning controls:

1. **Planning phase** (manual steering, root-only decomposition control)
2. **Formal research phase** (automated deeper traversal)

In this model:

- `pin / unpin / swap batch` are allowed **only in planning phase**.
- These actions are allowed **only for direct children of the root node**.
- Planning is used to shape the root decomposition until the user is satisfied.
- Formal research inherits planning artifacts and then continues execution normally.

---

## 2. Scope Constraints (Hard Rules)

1. During planning, only the root node is executed for quick lazy search + decomposition.
2. User operations are restricted to root’s direct children:
   - `Pin`
   - `Unpin`
   - `Swap Batch` (replace/re-generate unpinned candidates)
3. No planning-time manual operations on deeper descendants.
4. No orphan-node workflow is needed in this version because controllable nodes are only root children.

---

## 3. Phase Lifecycle

Execution states:

- `idle`
- `planning_running`
- `planning_review`
- `research_running`
- `report_generating`
- `completed`
- `failed`
- `aborted`

Allowed transitions:

- `idle -> planning_running` (`Start Planning`)
- `planning_running -> planning_review` (auto after root decomposition finishes)
- `planning_review -> planning_running` (`Swap Batch`)
- `planning_review -> research_running` (`Start Research`)
- `research_running -> report_generating` (`Generate Report` or formal completion policy)
- `report_generating -> completed`
- `planning_running|planning_review|research_running -> failed|aborted`

Notes:

- `Swap Batch` is explicitly a loop back into `planning_running`.
- `planning_review` may iterate multiple rounds before entering `research_running`.

---

## 4. Planning Semantics

## 4.1 Planning start

When planning starts:

1. Run quick lazy search on the root question.
2. Generate a candidate decomposition set of root children.
3. Stop in `planning_review` for user steering.

## 4.2 Pin

When root child `C` is pinned:

1. `C` is marked as user-approved direction.
2. `C` survives future `Swap Batch` operations.
3. A pin preference record is added to active context.
4. `C` receives default depth bonus `+1` (branch-local), bounded by system max depth.

## 4.3 Unpin

When root child `C` is unpinned:

1. Remove pin flag on `C`.
2. Remove/deactivate corresponding pin preference record from active context.
3. `C` is no longer guaranteed to survive future `Swap Batch`.
4. Remove its pin-derived depth bonus.

## 4.4 Swap Batch

`Swap Batch` is valid only in `planning_review`.

Behavior:

1. Keep root question unchanged.
2. Keep all currently pinned root children unchanged.
3. Re-generate only unpinned candidate children using:
   - uploaded user context,
   - currently active pin preference context.
4. Transition `planning_review -> planning_running -> planning_review`.


Swap diversity requirements (v1):

- Maintain a short-term exclusion memory of recently rejected unpinned candidates in the current planning session.
- Apply explicit novelty pressure in decomposition prompts for `swap_batch` (e.g., “avoid prior angles already shown”).
- Enforce a minimum change threshold for regenerated unpinned candidates (e.g., overlap cap / semantic distance floor).
- If diversity threshold is not met, allow one automatic retry with stronger novelty constraints before returning results.

---

## 5. User Context Composition

`user_context_bundle` is the context passed into decomposition-related reasoning.

It contains:

1. `uploaded_files_context` (user-provided material)
2. `active_pin_records` (from planning pin actions)

Rules:

- Pin adds a record to `active_pin_records`.
- Unpin removes/deactivates that record.
- Decomposition and swap-batch generation must consume this merged context.
- Apply bounded-context policy (recency + cap) to prevent prompt bloat.

---

## 6. Depth Rules

Defaults and limits:

- `default_max_depth = 2`
- `system_max_depth` is a hard upper bound.

Per pinned root child:

- default `pin_depth_bonus = +1`
- optional user increase is allowed during planning, but:
  - only for pinned root children,
  - cannot exceed `system_max_depth` when combined with base depth budget.

Recommended formula:

- `effective_max_depth(branch) = min(system_max_depth, default_max_depth + pin_depth_bonus)`

---

## 7. Inheritance into Formal Research

On `planning_review -> research_running`:

1. Preserve root quick lazy-search artifacts.
2. Preserve accepted root children and their pin metadata.
3. Preserve active preference context.
4. Seed formal frontier from planned root children (pinned prioritized first).
5. Continue normal research traversal from this seeded state.

---

## 8. Data Contract Additions

## 8.1 Node fields (root children)

- `is_pinned: bool`
- `pin_depth_bonus: int`
- `pin_updated_at: timestamp | null`
- `planning_batch_id: int`

## 8.2 Run/session fields

- `phase: "planning" | "research"`
- `default_max_depth: int` (default 2)
- `system_max_depth: int`
- `planning_snapshot`:
  - `root_lazy_queries[]`
  - `root_lazy_evidence[]`
  - `root_children_candidates[]`
  - `pinned_root_child_ids[]`
- `preference_context`:
  - `active_pin_records[]`
  - `pin_event_log[]` (`pin`, `unpin`)
- `user_context_bundle`:
  - `uploaded_files_context[]`
  - `active_pin_records[]`
  - `composition_version`

---

## 9. API Surface (v1 Minimal)

Planning:

- `POST /api/runs/{run_id}/planning/start`
- `POST /api/runs/{run_id}/planning/swap_batch`
- `POST /api/runs/{run_id}/planning/commit` (start formal research)

Root-child controls (planning only):

- `POST /api/runs/{run_id}/nodes/{node_id}/pin`
- `DELETE /api/runs/{run_id}/nodes/{node_id}/pin`
- `POST /api/runs/{run_id}/nodes/{node_id}/depth_bonus`

Report:

- `POST /api/runs/{run_id}/report` (allowed only in/after `research_running`, never in `planning_review`)

Validation requirements:

1. Reject pin/unpin/depth_bonus if state != `planning_review`.
2. Reject pin/unpin/depth_bonus if node is not a direct child of root.
3. Reject swap_batch if state != `planning_review`.
4. Reject report generation when formal research has not started.

---

## 10. UI Requirements

Under `Advanced Controls`:

1. `Start Planning` button.
2. Planning status indicator (`planning_running` / `planning_review`).
3. Root decomposition panel showing only direct children actions:
   - Pin/Unpin
   - Swap Batch
   - Depth increase control (only when pinned, incremental `+1` in v1)
4. `Start Research` button (enabled in `planning_review`).
5. Context hint: pins are merged with uploaded files as decomposition guidance.
6. **Canvas consistency requirement**: planning and research should reuse the same canvas component/layout where feasible; phase differences should be expressed via action availability/state badges rather than separate UIs.

---

## 11. Conflict Policy

1. **Pin idempotency**: repeated pin does not stack bonus unless explicit depth action is invoked.
2. **Pinned survival**: swap batch must not remove pinned root children.
3. **Depth cap**: reject over-limit depth bonus requests with clear validation errors.
4. **Context consistency**: unpin must remove/deactivate active pin context entry.
5. **Diversity over determinism**: swap batch should intentionally produce a meaningfully different decomposition for unpinned candidates (while preserving pinned children).

---

## 12. Locked Decisions (v1)

1. **Report timing**: report generation is allowed only after formal research starts (not directly from `planning_review`).
2. **Pinned traversal priority**: pinned root children are soft-prioritized in formal traversal (not hard-ordered).
3. **Unpin semantics**: unpin hard-deletes the active preference record for that node from runtime context.
4. **Depth bonus control**: v1 uses incremental depth bonus (`+1` per action), bounded by `system_max_depth`.
5. **Future extension**: keep UI/API room for optional set-based depth control in a later version.
6. **Future tuning**: expose swap diversity strictness as a configurable control if needed.

---

## 13. Trade-offs

Pros:

- Much simpler control surface (root children only).
- Removes earlier edge cases (orphan handling, deep manual edits).
- Fast iteration in planning before expensive formal traversal.

Costs:

- Less flexibility for mid-tree manual steering.
- More reliance on root decomposition quality.
- Additional state machine complexity vs one-shot research.

