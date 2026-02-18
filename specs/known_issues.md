# Known Issues

This file tracks active product/engineering issues that are accepted for now but not yet fixed.

## ISSUE-0001
- Title: Session open may show stale snapshot after report generation in another window
- Severity: P2
- Status: open
- Owner: unassigned
- Created: 2026-02-18
- Updated: 2026-02-18

### Symptoms
- In multi-window use, a report generated in Window B may not appear when the same session is opened in Window A.
- The report appears only after refreshing the session list and reopening the session.

### Repro Steps
1. Open two browser windows (A and B).
2. In B, open a session and generate a report.
3. In A, open the same session without refreshing.
4. Observe that report content/path may be missing until refresh + reopen.

### Root Cause
- Snapshot reads can fall back to disk-backed state under lock contention.
- Session index/state persistence is asynchronous, so disk may temporarily lag behind in-memory state.
- During that lag, a session open can return a stale snapshot.

### Workaround
- Click `Refresh` in session panel, then reopen the target session.

### Scope
- Web UI session loading across multiple windows/tabs.
- Most visible when report generation completes in a different window.

### Planned Fix Direction
- Prefer in-memory consistency for `GET /api/sessions/{session_id}` (fresh-first snapshot path).
- Keep anti-hang fallback behavior for list endpoints with explicit staleness signaling.
