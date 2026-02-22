# SearchBarbara

Iterative deep research agent that:
1. starts from the original task as a single root node
2. searches proactively for external evidence
3. tries to answer each node directly, then decomposes deeper only if evidence is insufficient
4. checks sufficiency and runs focused follow-up passes until sufficient or max rounds/depth
5. outputs a final report using Barbara Minto's Pyramid Principle

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set API/auth environment variables:

```bash
export OPENAI_API_KEY=...
export AUTH0_DOMAIN=your-tenant.us.auth0.com
export AUTH0_CLIENT_ID=...
export AUTH0_CLIENT_SECRET=...
export PUBLIC_BASE_URL=http://localhost:8000
export AUTH_COOKIE_SECRET=change-me-to-a-long-random-secret
```

Optional model override:

```bash
export OPENAI_MODEL=gpt-4.1
export OPENAI_REPORT_MODEL=gpt-5.2
```

Optional auth tuning:

```bash
export AUTH0_CONNECTION_NAME=email
export SESSION_TTL_DAYS=3
export AUTH_ALLOW_LEGACY=false
# export AUTH_COOKIE_SECURE=false   # useful for local http dev
```

## Usage

```bash
python deep_research_agent.py "Should our B2B SaaS expand into the German market in 2026?"
```

## Web App (MVP)

Run the local web interface:

```bash
python run_web.py
```

Then open:

```text
http://<your-vm-ip>:8000/
```

The web app is now authenticated:

- unauthenticated users are redirected to `/login`
- login uses Auth0 Universal Login (passwordless email OTP)
- all `/api/*` routes require authentication
- users only see/access their own sessions, runs, context files, SSE streams, and reports

Auth routes:

- `GET /login`
- `GET /auth/callback`
- `POST /logout`
- `GET /auth/me`

Key web endpoints:

- `POST /api/runs`
- `GET /api/runs/{run_id}`
- `GET /api/runs/{run_id}/events` (SSE)
- `GET /api/runs/{run_id}/report/download`
- `POST /api/runs/{run_id}/abort` (temporary debug control)

Options:

- `--max-depth` max recursive decomposition depth per question node (default `3`)
- `--max-rounds` max iterative research rounds (default `1`)
- `--results-per-query` baseline results per web query before adaptive broadening (default `3`)
- `--state-file` optional custom path for incremental checkpoint state JSON
- `--resume-from` resume from a previous checkpoint state JSON
- `--report-file` optional custom path for final report output
- `--usage-file` optional custom path for standalone token usage JSON
- `--pricing-file` pricing config JSON for cost estimation (default `pricing.json`)
- `--no-token-breakdown` disable token tracking and usage summary
- `--no-cost-estimate` disable cost estimation while keeping token tracking
- `--quiet` disable progress logs
- `--model` research model for query generation/search/synthesis/sufficiency (defaults to `OPENAI_MODEL` or `gpt-4.1`)
- `--report-model` report/decomposition model for final report writing and question decomposition (defaults to `OPENAI_REPORT_MODEL` or `gpt-5.2`)

## Notes

- The agent uses OpenAI web search via the Responses API (no external search provider key required).
- It tries web-search tool types in this order by default: `web_search_preview`, then `web_search`.
- Override tool order with `OPENAI_WEB_SEARCH_TOOL_TYPES` (comma-separated).
- Each run also writes crash-safe incremental state checkpoints to `runs/state_<query-slug>_<timestamp>.json` by default.
- Use `--resume-from path/to/state.json` to continue from the latest checkpoint.
- Each run writes the final report to `reports/report_<query-slug>_<timestamp>.md` by default.
- Use `--report-file path/to/report.md` to override the report output path.
- Each run records per-call token usage and stage/model aggregates in trace/state by default.
- Use `--usage-file path/to/usage.json` to export standalone token/cost breakdown.
- Cost estimates are config-based and loaded from `pricing.json` (or `--pricing-file`).
- Search results are quality-ranked to prioritize official and primary domains before synthesis.
- Progress messages are printed during execution so you can monitor direction in real time.
- If a query has no usable evidence, synthesis is skipped and the limitation is logged/traced.
- Final reports explicitly acknowledge limitations when evidence quality is meager or absent.
- Sufficiency follow-up queries are deduplicated and assigned once to the best-fit sub-question (no drop, no duplicate fan-out across all sub-questions).
- Repeated queries are conditionally deduped with cache reuse by intent; reruns require explicit justification (freshness/errors/recheck).
- When rerunning the same intent, retrieval is broadened progressively to increase information gain.
- Frontier execution is depth-aware: unresolved nodes can decompose into child sub-questions until `--max-depth`.
- Task traversal is lazy-first and depth-first: each node tries direct search first, then decomposes and explores children before moving to sibling branches.
- Large evidence payloads are compacted before sufficiency/report calls to reduce token pressure.
- LLM calls use retry with backoff on rate limits.
- Web run/session ownership is enforced by `owner_id` (Auth0 `sub`) on backend access paths.
- Workspace-staged context is user-scoped internally to avoid cross-user collisions.

## Auth0 Dashboard Setup (Web App)

Configure your Auth0 Application (Regular Web Application):

- Allowed Callback URLs:
  - `http://localhost:8000/auth/callback`
  - `https://<your-domain>/auth/callback`
- Allowed Logout URLs:
  - `http://localhost:8000/`
  - `https://<your-domain>/`
- Allowed Web Origins:
  - `http://localhost:8000`
  - `https://<your-domain>`

Enable passwordless email OTP:

- Authentication -> Passwordless -> Email
- enable the `email` connection (or your configured `AUTH0_CONNECTION_NAME`) for this app

## Prompt Customization

System prompts for each sub-agent are stored as separate files in `prompts/`:

- `prompts/decompose.system.txt`
- `prompts/query_gen.system.txt`
- `prompts/synthesize.system.txt`
- `prompts/sufficiency_node.system.txt`
- `prompts/sufficiency_pass.system.txt`
- `prompts/report.system.txt`
- `prompts/context_split.system.txt`

Edit these files to change sub-agent behavior without changing Python code.

## Search Policy Customization

Query reuse/rerun controls are stored in `search_policy.json`:

- `cache_ttl_seconds`
- `max_broaden_steps`
- `broaden_k_multipliers`
- `min_new_fact_gain`
- `max_no_gain_retries_per_intent`
- `time_sensitive_terms`
- `allow_rerun_on_search_error`

## Source Policy Customization

Source-tier rules are stored in `source_policy.json`.

- `primary_tlds`: domain suffixes automatically treated as primary/official.
- `primary_domain_suffixes`: explicit trusted domain rules.
  - `example.com` means exact domain only.
  - `*.example.com` means the domain and all subdomains.
- `secondary_tlds`: suffixes treated as secondary/reputable.

Edit this file to control which sources are considered primary without changing Python code.
