"""
Question 4 – automated tests for the web application (run:  pytest tests/test_app.py -v).

They send real HTTP requests to the FastAPI app, so the trained model is loaded and runs at request time.
The cases are the same as scripts/q4_test_inputs.py (normal, invalid, out-of-distribution).
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from scripts.q4_test_inputs import OK_STATES, cases, post

CASES = cases()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:          # runs start-up: loads the models once
        yield c


def test_health(client):
    r = client.get("/api/health").json()
    assert r["status"] == "ok" and r["model_loaded"] is True


def test_model_card(client):
    r = client.get("/api/model-card").json()
    assert set(r["thresholds"]) == {"pump", "fan"}
    assert r["input_rules"]["max_mb"] == 25


def test_frontend_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "AcoustiGuard" in r.text


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_case(client, case):
    r = post(client, case)
    assert r.status_code == case.expect_status, r.text[:300]
    body = r.json()
    if r.status_code == 200:
        assert 0.0 <= body["score"] <= 1.0
        assert body["ood_status"] in OK_STATES
        assert body["verdict"] in {"alarm", "no_alarm", "unreliable"}
        assert body["spectrogram_png"].startswith("data:image/png;base64,")
        if case.expect_ood:
            assert body["ood_status"] in case.expect_ood
        if case.expect_check:
            assert any(c["name"] == case.expect_check and c["status"] != "pass" for c in body["checks"])
    else:
        assert "message" in body or "detail" in body


def test_prediction_is_computed_live_and_deterministic(client):
    c = next(x for x in CASES if x.id == "N-pump-normal")
    a, b = post(client, c).json(), post(client, c).json()
    assert a["score"] == b["score"]                 # same input -> same output
    assert a["request_id"] != b["request_id"]       # but each request is a new computation


def test_example_endpoints(client):
    ex = client.get("/api/examples").json()
    if not ex:
        pytest.skip("no local example audio")
    r = client.post(f"/api/examples/{ex[0]['id']}/predict").json()
    assert "score" in r and r["true_label"] in {"normal", "abnormal"}
