# Backend Specification: Lazy-First + DFS-First Research Agent (Blank Slate)

## 1) Purpose

Build a backend agent that answers complex research questions by:

1. Trying direct evidence collection first (**lazy-first**).
2. Decomposing only when needed.
3. Exploring one branch deeply before moving to siblings (**DFS-first**).
4. Producing auditable outputs: intermediate reasoning artifacts, citations, confidence, and final report.

---

## 2) Product Requirements

### Functional
- Accept a user research task.
- Run multi-round research with configurable bounds.
- Generate and execute web/search queries.
- Synthesize evidence into structured findings.
- Evaluate sufficiency at node and run levels.
- Decompose unresolved questions into sub-questions.
- Traverse decomposition tree depth-first.
- Produce final report with:
  - answer
  - evidence summary
  - citations
  - limitations / open gaps
  - confidence rating

### Non-Functional
- Resumable from checkpoints.
- Observable via events/streaming progress.
- Deterministic-enough for reproducible audits.
- Failure-tolerant (query/search/model errors do not crash entire run).
- Configurable policies (search, decomposition, sufficiency, ranking).

---

## 3) Core Concepts

- **Run**: One end-to-end execution for a user task.
- **Node**: A research question in a tree (root or sub-question).
- **Finding**: Structured output from evidence synthesis.
- **Sufficiency**: Binary/graded judgment whether evidence answers a node.
- **Frontier**: Set of unresolved nodes to process this round.
- **Policy**: Configurable rules controlling search/decomposition/retry behavior.

---

## 4) Execution Strategy

### 4.1 High-Level Loop
For each run:
1. Initialize root node from user task.
2. Repeat for `round in 1..max_rounds`:
   - Process unresolved frontier with DFS node execution.
   - Perform pass-level sufficiency check.
   - Stop if sufficient or no meaningful progress.
3. Generate final report.
4. Persist terminal state and trace.

### 4.2 Node Processing (Lazy-First, DFS-First)
For node `N`:
1. Generate candidate search queries.
2. Execute query plan (with cache/reuse policy).
3. Rank/filter sources and synthesize findings.
4. Evaluate **node sufficiency**.
5. If sufficient → mark resolved.
6. If insufficient and `depth < max_depth`:
   - Decompose `N` into children.
   - Recurse into each child in order (DFS).
   - Re-evaluate parent from child outcomes.
7. If insufficient and cannot decompose (depth/policy/error) → unresolved with reason.

---

## 5) Data Model (Canonical)

### Run
- `run_id`
- `task`
- `status` (`queued|running|completed|failed|aborted`)
- `started_at`, `updated_at`, `completed_at`
- `config` (depth/round limits + policy versions)
- `root_node_id`
- `summary_metrics` (queries, sources, tokens, cost, latency)

### Node
- `node_id`
- `parent_id` (null for root)
- `depth`
- `question`
- `state` (`pending|running|resolved|unresolved`)
- `resolution_reason`
- `findings[]`
- `citations[]`
- `sufficiency` (score + rationale + gaps)
- `children[]`
- `attempt_counters` (query attempts, decompositions)

### Finding
- `finding_id`
- `claim`
- `evidence_snippets[]`
- `citation_ids[]`
- `confidence`
- `uncertainties[]`

### Citation
- `citation_id`
- `url`
- `title`
- `source_type`
- `retrieved_at`
- `quality_tier`
- `relevance_score`

---

## 6) Policy Specifications

### Search Policy
- Query generation limits (min/max queries per node).
- Cache TTL by normalized intent.
- Rerun rules (time-sensitive, explicit recheck, stale cache, prior failure).
- Diminishing-returns cutoff.
- Progressive broadening (`k` multipliers by attempt).

### Source Quality Policy
- Tiering rules (official/primary > reputable secondary > other).
- Deduplication and domain diversity constraints.
- Citation requirements per critical claim.

### Decomposition Policy
- Trigger only after node insufficiency.
- Max children per decomposition.
- MECE preference and overlap reduction.
- Duplicate/near-duplicate child suppression.

### Sufficiency Policy
- Node-level rubric:
  - coverage of question
  - evidence quality
  - contradiction handling
  - recency (if relevant)
- Pass-level rubric:
  - unresolved critical gaps
  - whether follow-up could materially change conclusion.

---

## 7) API Contract (Backend)

### Endpoints
- `POST /runs` → create run
- `GET /runs/{id}` → snapshot
- `GET /runs/{id}/events` → streaming events (SSE/WebSocket)
- `POST /runs/{id}/abort` → request cancellation
- `GET /runs/{id}/report` → final report
- `GET /runs/{id}/trace` → full machine-readable trace (authorized)

### Event Types (minimum)
- `run_started`, `run_completed`, `run_failed`, `run_aborted`
- `round_started`, `round_completed`
- `node_started`, `node_resolved`, `node_unresolved`
- `queries_generated`, `query_executed`, `query_skipped_cached`
- `sources_ranked`, `synthesis_completed`
- `node_sufficiency_evaluated`, `pass_sufficiency_evaluated`
- `node_decomposed`
- `report_generated`

---

## 8) Persistence & Resume

Persist at safe checkpoints:
- round boundaries
- after node completion
- before/after report generation

Checkpoint must include:
- run metadata
- full node tree state
- unresolved frontier
- query memory/cache metadata
- findings/citations
- trace cursor
- policy versions

Resume behavior:
- load checkpoint
- validate policy compatibility
- continue from next deterministic step
- avoid re-running completed immutable steps unless forced.

---

## 9) Failure Handling

- **Per-query failure**: mark error, continue node.
- **Per-node synthesis/model failure**: retry with backoff; fallback to partial output.
- **Provider outage**: degrade gracefully; mark confidence penalties.
- **Abort request**: cooperative cancellation with consistent terminal state.
- **Hard timeout**: terminate safely with partial report if configured.

---

## 10) Final Report Contract

Output sections:
1. Executive answer
2. Method summary (lazy-first DFS)
3. Key findings
4. Evidence and citations
5. Conflicts/uncertainties
6. Coverage gaps
7. Confidence and limitations
8. Suggested follow-up questions

Machine-readable companion:
- structured claims + citation mapping
- unresolved questions
- sufficiency diagnostics

---

## 11) Configuration Surface

Required knobs:
- `max_depth`
- `max_rounds`
- `max_queries_per_node`
- `search_results_per_query`
- `cache_ttl`
- `max_no_gain_retries`
- `decompose_max_children`
- `timeouts` (query/model/run)
- `retry_policy` (attempts/backoff)
- `report_style` (concise/technical)

---

## 12) Acceptance Criteria

1. Root starts without pre-decomposition.
2. Node decomposes only after direct evidence + insufficiency.
3. DFS ordering is preserved across all children.
4. Global bounds (`max_depth`, `max_rounds`, timeouts) are never exceeded.
5. Query reuse and rerun rules behave exactly per policy.
6. Run can be resumed from checkpoint without losing progress.
7. Final report always includes limitations and confidence.
8. Full event stream is coherent and replayable for UI/audit.

