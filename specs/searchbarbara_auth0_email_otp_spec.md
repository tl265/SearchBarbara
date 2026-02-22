# Spec: Auth0 Email OTP Login (Universal Login) + Multi-User Isolation (SearchBarbara)

**Target repo:** `tl265/SearchBarbara` (FastAPI + Jinja2 web app, `app.main:app`)  
**Auth choice:** **Auth0 Universal Login** using **Passwordless Email OTP** (no passwords stored by our app).  
**Primary requirement:** **each user is fully isolated**; no user can see another user’s runs/sessions/context/report.

---

## 1) Goals

1. **Simple for end users:** “Log in” → enter email → OTP → back to app.
2. **Simple for implementation:** use Auth0 for identity; app keeps a lightweight signed session cookie.
3. **Long-lived login:** keep user logged in for **SESSION_TTL_DAYS (default: 3)** with **sliding renewal**.
4. **Hard tenant isolation:** all `/api/*` endpoints and downloads/SSE are scoped by authenticated user.

Non-goals:
- Storing passwords.
- Supporting multiple login methods.
- Collaboration/sharing between users.

---

## 2) Auth0 Dashboard Setup (must be completed before code works)

### 2.1 Create an Auth0 Application
- Type: **Regular Web Application**
- Note down:
  - `AUTH0_DOMAIN` (e.g., `your-tenant.us.auth0.com`)
  - `AUTH0_CLIENT_ID`
  - `AUTH0_CLIENT_SECRET`

### 2.2 Configure URLs (dev + prod)
In the Auth0 Application settings:

- **Allowed Callback URLs**
  - `http://localhost:8000/auth/callback`
  - `https://<your-domain>/auth/callback`

- **Allowed Logout URLs**
  - `http://localhost:8000/`
  - `https://<your-domain>/`

- **Allowed Web Origins** (if needed by any frontend XHR/CORS later)
  - `http://localhost:8000`
  - `https://<your-domain>`

### 2.3 Enable Passwordless Email OTP
- Dashboard → Authentication → Passwordless → enable **Email**
- Connection name is typically `email` (confirm exact connection name in dashboard)
- Ensure OTP is enabled (OTP length/expiry can remain defaults for MVP)

### 2.4 Universal Login supports passwordless
- Dashboard → Authentication → Authentication Profile → enable **Identifier First** (recommended for passwordless UX)
- In the Application → Connections:
  - Enable the passwordless email connection for this application
  - Optionally disable database username/password connection to reduce confusion

---

## 3) App Auth Architecture

### 3.1 What we trust from Auth0
- We trust **Auth0-issued ID tokens** (OIDC) after signature verification.
- We use the ID token to identify the user:
  - `user_id = sub` (Auth0 subject)
  - `email = email` claim (if present)

### 3.2 What we store in our app
- We do **not** store Auth0 access tokens/refresh tokens in browser storage.
- We set **one signed HttpOnly cookie** for our app session:
  - Cookie name: `sb_session`
  - Contents (signed JSON):
    - `user_id` (Auth0 `sub`)
    - `email` (optional but nice for UI)
    - `iat` (issued_at unix)
    - `exp` (expires_at unix)
- Cookie TTL: `SESSION_TTL_DAYS` (default 3 days)
- Sliding lease: on every authenticated request, if session is valid, extend `exp` to `now + SESSION_TTL_DAYS`.

### 3.3 Logout behavior
- App logout clears `sb_session`.
- App then redirects user to Auth0 logout endpoint so the Auth0 SSO cookie is cleared too:
  - `https://{AUTH0_DOMAIN}/v2/logout?client_id={CLIENT_ID}&returnTo={URL_ENCODED_RETURN_TO}`

---

## 4) Backend Routes to Implement

### 4.1 Login start
`GET /login`

Behavior:
- Generate `state` (random) + `nonce` (random)
- Store `state` (and nonce) in a short-lived server-side store OR signed cookie:
  - simplest: store in a separate short-lived signed cookie (e.g., `sb_auth_state`) with 10-minute TTL
- Redirect to Auth0 `/authorize` with:
  - `response_type=code`
  - `client_id`
  - `redirect_uri={PUBLIC_BASE_URL}/auth/callback`
  - `scope=openid profile email`
  - `state`
  - `nonce`
  - If required to force passwordless: `connection=email` (or whatever passwordless email connection name is)

### 4.2 OAuth callback
`GET /auth/callback?code=...&state=...`

Behavior:
1. Validate `state` matches the stored value; reject on mismatch (show user-friendly error with link back to `/login`).
2. Exchange `code` for tokens by POSTing to:
   - `https://{AUTH0_DOMAIN}/oauth/token`
   - grant type: `authorization_code`
   - includes `client_id`, `client_secret`, `code`, `redirect_uri`
3. Validate the returned **ID token**:
   - Verify signature against Auth0 JWKS
   - Verify `iss`, `aud`, `exp`, and `nonce`
4. Create/renew our app session:
   - Set `sb_session` cookie with `exp = now + SESSION_TTL_DAYS`
5. Redirect to `/`

### 4.3 Logout
`POST /logout` (or `GET /logout` if you want simplest)
Behavior:
- Clear `sb_session` cookie
- Redirect to Auth0 logout (see 3.3) which returns to `/`

### 4.4 Auth “me” endpoint (optional but useful)
`GET /auth/me`
- Requires app auth
- Returns `{ user_id, email, session_expires_at }`

---

## 5) App-Wide Auth Guard

### 5.1 Require auth for:
- `GET /` (index): if not authenticated → redirect to `/login`
- All `/api/*` routes: if not authenticated → 401 JSON `{error: "unauthorized"}`
- All download routes, including:
  - `GET /api/runs/{run_id}/report/download`
- SSE route:
  - `GET /api/runs/{run_id}/events` must check ownership **before** streaming

### 5.2 Implementation pattern
- Add `get_current_user(request) -> UserPrincipal | None`
- Add `require_user(...)` FastAPI dependency that:
  - reads and verifies `sb_session`
  - checks expiry
  - on success sets `request.state.user`
- Add middleware or helper to renew cookie expiry (sliding lease).

---

## 6) Tenant Isolation Requirements (Critical)

### 6.1 Ownership model
- Every run/session has `owner_id` = authenticated `user_id` (Auth0 `sub`).
- `RunState.owner_id` already exists; ensure it is set at creation time and persisted.

### 6.2 Enforce on ALL run APIs
For any endpoint that accepts a `run_id`:
- Load run
- If `run.owner_id != current_user.user_id`: return 404 (preferred) or 403

Endpoints include (not exhaustive):
- create run / draft / start from workspace
- list sessions
- get run snapshot
- SSE events stream
- abort/pause/resume
- report generation endpoints
- report download
- context endpoints for upload/download/digest operations

### 6.3 Storage
- Minimal change approach:
  - keep current filesystem layout
  - enforce owner checks in code for every access path
- Recommended improvement (optional):
  - store per-user under `runs/<owner_id>/...` and `reports/<owner_id>/...` to reduce foot-guns

### 6.4 Legacy data
Runs created before auth will have `owner_id=None`.
Choose behavior:
- Default: **hide** legacy runs from all users (treat as not found)
- Optional dev override: `AUTH_ALLOW_LEGACY=true` to allow access without owner checks in dev only

---

## 7) Changes Required in This Repo (Implementation Checklist)

### 7.1 New files
- `app/auth0_login.py` (or `app/auth.py`)
  - build `/authorize` URL
  - exchange code at `/oauth/token`
  - validate ID token via JWKS
  - cookie sign/verify helpers
  - `require_user` dependency

### 7.2 Modify existing files
- `app/main.py`
  - add `/login`, `/auth/callback`, `/logout`, optional `/auth/me`
  - apply `Depends(require_user)` to all `/api/*` routes
  - in `index()` route, redirect to `/login` if not authed
  - add “Logout” button (template context) if you want UI

- `app/run_manager.py`
  - update `create_run`, `create_draft_session`, `create_run_from_workspace`, etc. to accept `owner_id: str`
  - set `RunState.owner_id = owner_id` at creation
  - ensure sessions index persistence includes owner_id
  - provide helper: `assert_owner(run_id, owner_id)` or `get_run_if_owned(run_id, owner_id)`

- Templates:
  - `app/templates/index_v2.html`: show signed-in email + logout button (optional)
  - (Optional) `app/templates/login.html` if you want a branded “Login” page; otherwise `/login` can immediately redirect

### 7.3 Requirements / dependencies
Add minimal dependencies for implementation:
- `httpx` (token exchange + JWKS fetch) OR use stdlib `urllib` if you prefer no deps
- `python-jose[cryptography]` OR `PyJWT[crypto]` for JWT verification

(Choose one stack and implement consistently.)

---

## 8) Configuration (Env Vars)

Required:
- `AUTH0_DOMAIN`
- `AUTH0_CLIENT_ID`
- `AUTH0_CLIENT_SECRET`
- `PUBLIC_BASE_URL` (e.g., `http://localhost:8000` or `https://your-domain`)
- `AUTH_COOKIE_SECRET` (HMAC secret for signing `sb_session`)

Optional:
- `AUTH0_CONNECTION_NAME` (default: `email`)
- `SESSION_TTL_DAYS` (default: `3`)
- `AUTH_ALLOW_LEGACY` (default: `false`)

---

## 9) Error Handling Requirements

- `/login`: always succeeds (redirect)
- `/auth/callback`:
  - missing/invalid `code` or state mismatch: show a friendly error page + link back to `/login`
  - token exchange failure: show friendly error
  - id_token validation failure: show friendly error
- API routes:
  - unauthenticated: 401 JSON
  - unauthorized access to other user’s run: 404 JSON

---

## 10) Acceptance Tests (must pass)

### Auth flow
1. Visiting `/` unauthenticated redirects to `/login`.
2. `/login` redirects to Auth0 and returns to `/auth/callback`.
3. After callback, `sb_session` cookie is set and `/` renders.
4. Session persists across refreshes.
5. Sliding lease renews expiry on activity.
6. Logout clears `sb_session` and redirects to Auth0 logout then back to `/`.

### Isolation
7. User A creates a run. User B cannot:
   - list it
   - fetch its snapshot
   - stream its SSE events
   - download its report
   - upload/download its context files
8. Legacy run with `owner_id=None` is not visible by default.

---

## 11) Notes for Codex Implementation

- Do NOT add additional login methods.
- Do NOT store user passwords.
- Treat `owner_id` checks as non-negotiable security boundaries.
- Apply auth guards to SSE and download routes (common leak vectors).
