# SearchBarbara

SearchBarbara is an iterative deep research agent with:

- a CLI for one-off research runs
- an authenticated web app for sessions, context, reports, and live intent parsing
- a backend/frontend/agents/config split aimed at maintainability

At a high level, it:

1. starts from the original task as a single root node
2. searches for external evidence proactively
3. answers directly when possible, then decomposes only when evidence is insufficient
4. checks sufficiency and runs focused follow-up passes until sufficient or capped
5. outputs a final report using Barbara Minto's Pyramid Principle

## Quick Start

Create a virtualenv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set the required environment variables:

```bash
export OPENAI_API_KEY=...
export AUTH0_DOMAIN=your-tenant.us.auth0.com
export AUTH0_CLIENT_ID=...
export AUTH0_CLIENT_SECRET=...
export PUBLIC_BASE_URL=http://localhost:8000
export AUTH_COOKIE_SECRET=change-me-to-a-long-random-secret
```

Optional runtime tuning:

```bash
export OPENAI_MODEL=gpt-4.1
export OPENAI_REPORT_MODEL=gpt-5.2
export AUTH0_CONNECTION_NAME=email
export SESSION_TTL_DAYS=3
export AUTH_ALLOW_LEGACY=false
# export AUTH_COOKIE_SECURE=false   # useful for local http dev
```

Run the web app:

```bash
python run_web.py
```

Then open `http://localhost:8000/`.

## Main Workflows

### CLI

Run a research task from the command line:

```bash
python deep_research_agent.py "Should our B2B SaaS expand into the German market in 2026?"
```

Common options:

- `--max-depth` max recursive decomposition depth per node
- `--max-rounds` max research rounds
- `--results-per-query` baseline retrieval breadth
- `--resume-from` resume from a saved checkpoint
- `--report-file` write the final report to a specific path
- `--usage-file` write token/cost usage to a specific path
- `--pricing-file` pricing config path, default `config/agent/pricing.json`
- `--model` research model, default `OPENAI_MODEL` or `gpt-4.1`
- `--report-model` report/decomposition model, default `OPENAI_REPORT_MODEL` or `gpt-5.2`

### Web App

The web app is authenticated:

- unauthenticated users are redirected to `/login`
- login uses Auth0 Universal Login with passwordless email OTP
- all `/api/*` routes require authentication
- users only see their own sessions, runs, context files, SSE streams, and reports

Important routes:

- `GET /login`
- `GET /auth/callback`
- `POST /logout`
- `GET /auth/me`
- `POST /api/runs`
- `GET /api/runs/{run_id}`
- `GET /api/runs/{run_id}/events`
- `GET /api/runs/{run_id}/report/download`

## Repo Layout

The codebase is organized by top-level ownership:

- `backend/` web server, API, orchestration, and backend runtime logic
- `frontend/` templates and static assets for the active web UI
- `agents/` research implementation and prompt assets
- `config/` app config and agent policy/config data
- `infra/` deploy and observability helpers
- `app/` compatibility shims for older imports and entrypoints

## Configuration

### App and Web Config

- `config/app/web.json`
  Web UI and backend feature flags such as live intent, preprocessing, and UI tuning.

### Agent Config

- `config/agent/pricing.json`
  Token and search cost estimation data.
- `config/agent/search_policy.json`
  Query reuse, rerun, and broadening controls.
- `config/agent/source_policy.json`
  Domain quality and trust-tier rules.
- `config/agent/decompose_policy.json`
  Decomposition behavior settings.

### Prompts

Prompt files live under `agents/prompts/`:

- `agents/prompts/deep_research/`
- `agents/prompts/context_preprocess/`
- `agents/prompts/live_intent/`

Edit these files to change LLM behavior without changing Python code.

## Runtime Artifacts

By default the app writes local state here:

- `runs/`
  checkpoint state, session indexes, context artifacts, and templates
- `reports/`
  generated markdown reports
- `logs/`
  runtime and agent debug logs

## Research Behavior Notes

- OpenAI web search is used through the Responses API.
- Search tool order can be overridden with `OPENAI_WEB_SEARCH_TOOL_TYPES`.
- Each run writes crash-safe checkpoint state under `runs/`.
- Final reports are written under `reports/` unless overridden.
- Token usage and cost breakdowns can be exported separately.
- Search results are quality-ranked to prioritize official and primary sources.
- Queries are deduplicated and can be rerun only when justified.
- Traversal is lazy-first and depth-first.
- Large evidence payloads are compacted before sufficiency and report calls.
- LLM calls use retry with backoff on rate limits.

## Auth0 Setup

For the web app, configure your Auth0 application as a Regular Web Application.

Allowed callback URLs:

- `http://localhost:8000/auth/callback`
- `https://<your-domain>/auth/callback`

Allowed logout URLs:

- `http://localhost:8000/`
- `https://<your-domain>/`

Allowed web origins:

- `http://localhost:8000`
- `https://<your-domain>`

If you access the app through an SSH tunnel for local testing, use the tunneled
origin consistently for auth as well:

- `http://localhost:8001/auth/callback`
- `http://localhost:8001/`
- `http://localhost:8001`

Enable passwordless email OTP:

- Auth0 Dashboard -> Authentication -> Passwordless -> Email
- enable the `email` connection, or your configured `AUTH0_CONNECTION_NAME`

## Deployment Note

`python run_web.py` is the local entrypoint.

ASR dictation requires two runtime conditions:

- a trusted browser origin for microphone APIs: `https://...` or a local
  tunneled/dev origin such as `http://localhost:8001`
- a websocket backend in the Python environment serving the app:
  `websockets` or `wsproto`

If `asr_enabled` is true and neither websocket backend is installed, the app
will now fail fast at startup instead of serving a broken dictation button.

The repo also contains a systemd template under `infra/deploy/systemd/`, but your installed stable service may use a separate unit and env file outside the repo. Treat the repo unit as a template, not automatically as the live source of truth.
