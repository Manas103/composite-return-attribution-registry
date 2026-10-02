"""Sanity tests for the FastAPI surface (returns, composites, attribution,
versioning endpoints)."""
from fastapi.testclient import TestClient

from cra.api import build_app


def test_health():
    app = build_app(seed=5)
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["accounts"] == 240
    assert body["days"] == 756


def test_twr_endpoint_rejects_bad_begin_mv():
    app = build_app(seed=5)
    client = TestClient(app)
    r = client.post("/metrics/twr", json={"begin_mv": 0, "end_mv": 100, "flow": 0})
    assert r.status_code == 400


def test_attribution_endpoint_reconciles():
    app = build_app(seed=5)
    client = TestClient(app)
    r = client.post(
        "/metrics/attribution",
        json={
            "entity_type": "account",
            "entity_id": 0,
            "sectors": ["Technology", "Financials"],
            "w_p": [0.6, 0.4],
            "r_p": [0.02, 0.01],
            "w_b": [0.5, 0.5],
            "r_b": [0.015, 0.012],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert abs(body["reconciliation_discrepancy"]) < 1e-10


def test_restate_then_read_versions():
    app = build_app(seed=5)
    client = TestClient(app)
    r = client.post("/admin/restate", json={"security_id": 0, "day_idx": 5, "new_price": 150.0, "affected_account_id": 0})
    assert r.status_code == 200
    as_of = str(app.state.book.dates[5].date())
    r2 = client.get(f"/accounts/0/returns/{as_of}/versions")
    assert r2.status_code == 200
    versions = r2.json()
    assert len(versions) == 1
    assert versions[0]["version"] == 1

    r3 = client.get(f"/accounts/0/returns/{as_of}")
    assert r3.status_code == 200
    assert r3.json()["version"] == 1


def test_missing_return_is_404():
    app = build_app(seed=5)
    client = TestClient(app)
    r = client.get("/accounts/0/returns/2022-01-04")
    assert r.status_code == 404
