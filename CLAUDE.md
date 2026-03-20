# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SearchBarbara is an iterative deep research agent that performs web-based research using OpenAI's web search capabilities. The system decomposes research questions into sub-questions, searches for evidence, and generates reports using Barbara Minto's Pyramid Principle.

## Development Setup

### Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Required Environment Variables
```bash
export OPENAI_API_KEY=your_openai_api_key
export AUTH0_DOMAIN=your-tenant.us.auth0.com
export AUTH0_CLIENT_ID=your_client_id
export AUTH0_CLIENT_SECRET=your_client_secret
export PUBLIC_BASE_URL=http://localhost:8000
export AUTH_COOKIE_SECRET=long-random-secret
```

### Optional Configuration
```bash
export OPENAI_MODEL=gpt-4.1
export OPENAI_REPORT_MODEL=gpt-5.2
export AUTH0_CONNECTION_NAME=email
export SESSION_TTL_DAYS=3
export AUTH_ALLOW_LEGACY=false
```

## Running the Application

### CLI Mode
```bash
python deep_research_agent.py "Research question here"
```

### Web Application
```bash
python run_web.py
```
The web app runs on http://localhost:8000 with authentication via Auth0.

## Architecture

### Core Components

- **deep_research_agent.py**: Main research engine with question decomposition, web search, and report generation
- **app/main.py**: FastAPI web application with authentication and API endpoints
- **app/run_manager.py**: Manages research sessions, state persistence, and idempotency
- **app/auth.py**: Auth0 integration for user authentication
- **app/sse.py**: Server-Sent Events for real-time updates

### Key Directories
- **prompts/**: System prompts for different research sub-agents (decomposition, query generation, synthesis, etc.)
- **specs/**: Technical specifications and design documents
- **app/static/**: Frontend JavaScript and CSS files
- **app/templates/**: HTML templates for web interface

### Configuration Files
- **web_config.json**: Web interface configuration (depth limits, model settings, UI parameters)
- **search_policy.json**: Search behavior configuration (cache TTL, broadening steps, result limits)
- **source_policy.json**: Source prioritization rules (primary/secondary domains, TLD preferences)
- **pricing.json**: Token cost estimation configuration

## API Endpoints

### Authentication
- `GET /login` - Initiate Auth0 login
- `GET /auth/callback` - Auth0 callback handler
- `POST /logout` - Logout user
- `GET /auth/me` - Get current user info

### Research Sessions
- `POST /api/runs` - Create new research session
- `GET /api/runs/{run_id}` - Get session status and results
- `GET /api/runs/{run_id}/events` - SSE stream for real-time updates
- `POST /api/runs/{run_id}/abort` - Abort running research
- `GET /api/runs/{run_id}/report/download` - Download final report

### Context Management
- `POST /api/sessions/{session_id}/context/files` - Upload context files
- `GET /api/sessions/{session_id}/context/files/{file_id}/download` - Download context file
- `GET /api/sessions/{session_id}/context/digest` - Get context summary

## Research Process

The agent follows a structured research workflow:

1. **Question Decomposition**: Breaks main question into sub-questions using `prompts/decompose.system.txt`
2. **Web Search**: Uses OpenAI web search API with source prioritization
3. **Evidence Synthesis**: Combines search results using `prompts/synthesize.system.txt`
4. **Sufficiency Check**: Determines if more research is needed using `prompts/sufficiency.system.txt`
5. **Report Generation**: Creates final report using Pyramid Principle via `prompts/report.system.txt`

## Testing

Tests are located in `tests/` directory:
- `test_idempotency_api.py` - Tests API idempotency and session management
- Uses FastAPI TestClient for endpoint testing
- Tests cover idempotency key reuse, session conflicts, and error handling

## Key Configuration Points

### Search Behavior
- Configure search parameters in `search_policy.json`:
  - `cache_ttl_seconds`: How long to cache search results
  - `max_broaden_steps`: Number of search broadening attempts
  - `results_per_query`: Default number of results per search

### Source Prioritization
- Edit `source_policy.json` to prioritize specific domains:
  - Primary sources: .gov, .edu domains and specific organizations
  - Secondary sources: .org, .com domains

### Model Configuration
- Default research model: `gpt-4.1`
- Default report model: `gpt-5.2`
- Configure via environment variables or web interface

## Development Notes

- The application uses idempotency keys to prevent duplicate operations
- Research state is persisted incrementally for crash recovery
- Authentication is required for all API endpoints except public paths
- Context files are user-scoped to prevent cross-user access
- Real-time updates are provided via Server-Sent Events (SSE)

## Troubleshooting

- Check Auth0 configuration if authentication fails
- Verify OpenAI API key if search operations fail
- Monitor token usage via built-in tracking system
- Use `--resume-from` flag to continue from interrupted research sessions