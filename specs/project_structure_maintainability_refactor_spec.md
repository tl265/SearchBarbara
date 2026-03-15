# Project Structure Maintainability Refactor

## Summary

Refactor the repo around explicit top-level ownership boundaries:

- `backend/` for the web server, backend runtime logic, and orchestration
- `frontend/` for templates and static assets
- `agents/` for LLM workflows and prompt assets
- `config/` for app config and agent policy/config data
- `infra/` for deploy and observability tooling
- `tests/` for integration and unit coverage

Compatibility shims remain in `app/` and the root `deep_research_agent.py` entrypoint so existing imports and commands continue to work during the transition.

## Implemented Changes

- Moved the active web app implementation from `app/` into `backend/` and `frontend/`.
- Moved the deep research implementation into `agents/deep_research/`.
- Moved prompt assets into `agents/prompts/`, grouped by feature:
  - `deep_research/`
  - `live_intent/`
  - `context_preprocess/`
- Moved config and policy files into `config/`:
  - `config/app/web.json`
  - `config/agent/pricing.json`
  - `config/agent/search_policy.json`
  - `config/agent/source_policy.json`
  - `config/agent/decompose_policy.json`
- Added `infra/observability/` as the shared home for logging and future metrics/tracing/debug tooling.
- Removed the dead legacy UI route and its unused assets.

## Compatibility / Interfaces

- `python run_web.py` still works through the existing entrypoint.
- `python deep_research_agent.py ...` still works through a root shim.
- Old imports under `app.*` still resolve through compatibility wrappers.
- API routes and runtime data directories (`runs/`, `reports/`, `logs/`) are unchanged.

## Verification

- `python3 -m py_compile` on moved backend, agent, shim, and test modules
- `node --check frontend/static/app_v2.js`
- Manual UI/runtime smoke testing should continue from the existing entrypoints

## Follow-up Work

- Split `backend/server.py` into smaller API modules under `backend/api/`.
- Break `backend/orchestration/run_manager.py` into smaller orchestration helpers.
- Modularize `frontend/static/app_v2.js` and `frontend/static/styles_v2.css`.
- Add more shared observability utilities under `infra/observability/`.
