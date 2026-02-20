from fastapi.testclient import TestClient
from types import SimpleNamespace

from app import main


client = TestClient(main.app)


def _clear_idempotency_records() -> None:
    with main.run_manager._idempotency_lock:
        main.run_manager._idempotency_records.clear()


def test_create_run_replays_same_response_for_same_key(monkeypatch):
    _clear_idempotency_records()
    created = {"count": 0}

    def fake_create_run(**_kwargs):
        created["count"] += 1
        return f"run-{created['count']}"

    monkeypatch.setattr(main.run_manager, "create_run", fake_create_run)
    payload = {"task": "topic-a", "max_depth": 2, "results_per_query": 1}
    headers = {"Idempotency-Key": "same-create-key"}

    first = client.post("/api/runs", json=payload, headers=headers)
    second = client.post("/api/runs", json=payload, headers=headers)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["run_id"] == "run-1"
    assert second.json()["run_id"] == "run-1"
    assert created["count"] == 1


def test_create_run_rejects_same_key_with_different_payload(monkeypatch):
    _clear_idempotency_records()
    monkeypatch.setattr(main.run_manager, "create_run", lambda **_kwargs: "run-1")
    headers = {"Idempotency-Key": "shared-key"}

    first = client.post(
        "/api/runs",
        json={"task": "topic-a", "max_depth": 2, "results_per_query": 1},
        headers=headers,
    )
    second = client.post(
        "/api/runs",
        json={"task": "topic-b", "max_depth": 2, "results_per_query": 1},
        headers=headers,
    )

    assert first.status_code == 200
    assert second.status_code == 409
    assert second.json()["detail"]["error_code"] == "idempotency_key_reused"


def test_create_run_returns_in_progress_for_pending_same_key():
    _clear_idempotency_records()
    model = str(main.WEB_CONFIG.get("model", "gpt-4.1"))
    report_model = str(main.WEB_CONFIG.get("report_model", "gpt-5.2"))
    key = "pending-key"
    payload = {
        "task": "topic-a",
        "max_depth": 2,
        "results_per_query": 1,
        "model": model,
        "report_model": report_model,
    }
    main.run_manager.idempotency_begin(
        "create_run",
        key,
        payload,
        ttl_sec=main.RunManager._IDEMPOTENCY_TTL_CREATE_RUN_SEC,
    )

    rsp = client.post(
        "/api/runs",
        json={"task": "topic-a", "max_depth": 2, "results_per_query": 1},
        headers={"Idempotency-Key": key},
    )

    assert rsp.status_code == 409
    assert rsp.json()["detail"]["error_code"] == "idempotency_in_progress"


def test_report_endpoint_replays_and_new_key_creates_new_result(monkeypatch):
    _clear_idempotency_records()
    calls = {"count": 0}

    def fake_generate_partial_report(_run_id, expected_version=None):
        calls["count"] += 1
        return {
            "report_file_path": f"/tmp/r{calls['count']}.md",
            "version_index": calls["count"],
            "version": 1,
        }

    monkeypatch.setattr(main.run_manager, "generate_partial_report", fake_generate_partial_report)

    r1 = client.post("/api/runs/run-123/report", headers={"Idempotency-Key": "rep-a"})
    r2 = client.post("/api/runs/run-123/report", headers={"Idempotency-Key": "rep-a"})
    r3 = client.post("/api/runs/run-123/report", headers={"Idempotency-Key": "rep-b"})

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r3.status_code == 200
    assert r1.json()["version_index"] == 1
    assert r2.json()["version_index"] == 1
    assert r3.json()["version_index"] == 2
    assert calls["count"] == 2


def test_abort_ignores_expected_version_and_replays_with_same_key(monkeypatch):
    _clear_idempotency_records()
    calls = {"count": 0, "expected_version": "unset"}

    def fake_abort_run(_run_id, expected_version=None):
        calls["count"] += 1
        calls["expected_version"] = expected_version
        return {"status": "aborting", "version": 7}

    monkeypatch.setattr(main.run_manager, "abort_run", fake_abort_run)
    headers = {"Idempotency-Key": "abort-key-1"}

    r1 = client.post("/api/runs/run-42/abort?expected_version=999", headers=headers)
    r2 = client.post("/api/runs/run-42/abort?expected_version=1", headers=headers)

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert calls["count"] == 1
    assert calls["expected_version"] is None
    assert r1.json() == r2.json()


def test_delete_session_blocks_when_report_generating():
    sid = "delete-guard-generating"
    rm = main.run_manager
    with rm._lock:
        rm._runs[sid] = SimpleNamespace(
            status="completed",
            execution_state="completed",
            report_state="generating",
        )
        rm._sessions[sid] = {"session_id": sid, "state_file_path": ""}
    try:
        result = rm.delete_session(sid)
        assert result == "conflict_running"
    finally:
        with rm._lock:
            rm._runs.pop(sid, None)
            rm._sessions.pop(sid, None)
            rm._active_report_runs.discard(sid)


def test_delete_session_blocks_when_report_worker_active():
    sid = "delete-guard-active-worker"
    rm = main.run_manager
    with rm._lock:
        rm._runs[sid] = SimpleNamespace(
            status="completed",
            execution_state="completed",
            report_state="completed",
        )
        rm._sessions[sid] = {"session_id": sid, "state_file_path": ""}
        rm._active_report_runs.add(sid)
    try:
        result = rm.delete_session(sid)
        assert result == "conflict_running"
    finally:
        with rm._lock:
            rm._runs.pop(sid, None)
            rm._sessions.pop(sid, None)
            rm._active_report_runs.discard(sid)
