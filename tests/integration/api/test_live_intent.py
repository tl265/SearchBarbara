from types import SimpleNamespace

from fastapi.testclient import TestClient

from backend import server
from backend.core.live_intent import LiveIntentClassifier


client = TestClient(server.app)


def _enable_test_auth(monkeypatch):
    monkeypatch.setattr(
        type(server.AUTH_CONFIG),
        "configured",
        property(lambda _self: True),
    )
    monkeypatch.setattr(
        server,
        "get_current_user_from_request",
        lambda _request, _auth: SimpleNamespace(
            user_id="test-user",
            email="test@example.com",
            session_expires_at=0,
        ),
    )


def test_live_intent_classifier_returns_empty_for_short_input():
    classifier = LiveIntentClassifier(enabled=True, min_chars=12, max_input_chars=600)

    out = classifier.classify("too short")

    assert out.task_type is None
    assert out.sophistication is None
    assert out.audience is None
    assert out.stake_level is None
    assert out.time_horizon is None


def test_live_intent_classifier_uses_lower_threshold_for_cjk_input():
    classifier = LiveIntentClassifier(
        enabled=True,
        min_chars_default=12,
        min_chars_cjk=6,
        max_input_chars=600,
    )

    out = classifier.classify("帮我对比")

    assert out.task_type == "compare"
    assert out.sophistication == "intermediate"


def test_live_intent_classifier_infers_compare_prompt():
    classifier = LiveIntentClassifier(enabled=True, min_chars=12, max_input_chars=600)

    out = classifier.classify("Compare Postgres and MySQL for a CTO decision next quarter")

    assert out.task_type == "compare"
    assert out.sophistication == "intermediate"
    assert out.audience == "senior_management"
    assert out.stake_level == "medium"
    assert out.time_horizon == "strategic"


def test_live_intent_classifier_infers_troubleshooting_prompt():
    classifier = LiveIntentClassifier(enabled=True, min_chars=12, max_input_chars=600)

    out = classifier.classify("Why is my nginx reverse proxy returning 502 in production right now?")

    assert out.task_type == "troubleshoot"
    assert out.sophistication == "intermediate"
    assert out.audience == "practitioner"
    assert out.stake_level == "high"
    assert out.time_horizon == "immediate"


def test_live_intent_classifier_infers_chinese_management_prompt():
    classifier = LiveIntentClassifier(
        enabled=True,
        min_chars_default=12,
        min_chars_cjk=6,
        max_input_chars=600,
    )

    out = classifier.classify("写一份说服管理层批准迁移预算的备忘录")

    assert out.task_type == "persuade"
    assert out.audience == "senior_management"
    assert out.stake_level in {"medium", "high"}


def test_live_intent_endpoint_returns_prediction(monkeypatch):
    _enable_test_auth(monkeypatch)

    rsp = client.post(
        "/api/intent/live",
        json={"text": "Compare Postgres and MySQL for a CTO decision next quarter"},
    )

    assert rsp.status_code == 200
    body = rsp.json()
    assert body["task_type"] == "compare"
    assert body["audience"] == "senior_management"
    assert body["time_horizon"] == "strategic"


def test_live_intent_endpoint_returns_prediction_for_chinese(monkeypatch):
    _enable_test_auth(monkeypatch)

    rsp = client.post(
        "/api/intent/live",
        json={"text": "为什么我现在的 nginx 反向代理返回 502"},
    )

    assert rsp.status_code == 200
    body = rsp.json()
    assert body["task_type"] == "troubleshoot"
    assert body["audience"] == "practitioner"
    assert body["time_horizon"] == "immediate"
