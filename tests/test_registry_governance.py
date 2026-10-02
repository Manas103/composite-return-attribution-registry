"""The governance test: proves that the FastAPI consumer and the batch
job consumer compute every metric through the exact same registry
function object, not two independent copies of the formula.

The mechanism: cra.registry.register_metric wraps every metric function
so that each invocation appends (metric_name, id(function)) to the
module-level CALL_LOG. This test drives the API through httpx's
TestClient (one consumer) and the batch module directly (a second,
independent consumer), and asserts the function identity logged by both
is identical.

If someone added a second, independent implementation of a metric
anywhere outside cra/registry.py (for example, hard-coding
`(end_mv - flow) / begin_mv - 1` directly inside an API handler instead of
calling registry.daily_twr), CALL_LOG would simply never see a
"daily_twr" entry for that code path, or it would see one bound to a
different function object if that someone used a second registry-style
decorator. Either way this test would fail. A static source-text grep was
considered and rejected in favor of this because it tests the real call
path at runtime, including through cra.batch and cra.persistence, rather
than pattern-matching source code.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from cra import batch, registry
from cra.api import build_app


def _ids_for_metric(metric_name: str) -> set[int]:
    return {fid for name, fid in registry.CALL_LOG if name == metric_name}


def test_api_and_batch_consumers_call_the_identical_daily_twr_function(small_book):
    canonical_id = id(registry.get_metric("daily_twr"))

    app = build_app(seed=small_book.seed)
    client = TestClient(app)

    registry.reset_call_log()
    r = client.post("/metrics/twr", json={"begin_mv": 100_000, "end_mv": 101_000, "flow": 0})
    assert r.status_code == 200
    api_ids = _ids_for_metric("daily_twr")
    assert api_ids == {canonical_id}, "API consumer did not route through the canonical registry function"

    registry.reset_call_log()
    batch.compute_account_daily_twr_series(small_book, 0)
    batch_ids = _ids_for_metric("daily_twr")
    assert batch_ids == {canonical_id}, "batch consumer did not route through the canonical registry function"

    assert api_ids == batch_ids, "API and batch consumers computed daily_twr through different function objects"


def test_api_and_batch_consumers_call_the_identical_brinson_attribution_function(small_book):
    canonical_id = id(registry.get_metric("brinson_attribution"))

    app = build_app(seed=small_book.seed)
    client = TestClient(app)

    registry.reset_call_log()
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
    api_ids = _ids_for_metric("brinson_attribution")
    assert api_ids == {canonical_id}

    registry.reset_call_log()
    months = small_book.month_boundaries()
    month_label, day_idx = months[0]
    batch.monthly_attribution(small_book, "account", 0, str(month_label), day_idx)
    batch_ids = _ids_for_metric("brinson_attribution")
    assert batch_ids == {canonical_id}

    assert api_ids == batch_ids


def test_restatement_path_also_routes_through_the_canonical_daily_twr_function(small_book):
    """The /admin/restate endpoint calls cra.batch.restate_price_and_republish,
    which recomputes the affected return. This checks that third call site
    is still the same function object, not a bespoke recompute."""
    from cra import db

    canonical_id = id(registry.get_metric("daily_twr"))
    engine = db.make_sqlite_engine(":memory:")
    Session = db.session_factory(engine)
    session = Session()

    registry.reset_call_log()
    security_id = int(small_book.positions.iloc[0]["security_id"])
    old_price = float(small_book.prices[5, security_id])
    batch.restate_price_and_republish(session, small_book, security_id, 5, old_price * 1.02, 0)
    ids = _ids_for_metric("daily_twr")
    assert ids == {canonical_id}


def test_registered_metrics_cover_every_claimed_metric():
    names = set(registry.registered_metrics())
    assert {"daily_twr", "period_return", "composite_return", "excess_return", "brinson_attribution"} <= names
