"""Exercises the real PostgreSQL path when a database is reachable and
skips cleanly when it is not. PostgreSQL is not an always-on service on
the build machine (see README "Building and running" for the exact
stand-up command); this mirrors payment-exception-triage and
eod-cash-balance-forecasting elsewhere in this portfolio: infrastructure
the test can't reach is a skip, not a failure.
"""
from __future__ import annotations

import os

import pytest
from sqlalchemy.exc import OperationalError

from cra import batch, db, persistence, registry


@pytest.fixture()
def pg_session():
    try:
        engine = db.make_postgres_engine(os.environ.get("CRA_PG_DSN"))
    except OperationalError:
        pytest.skip("PostgreSQL not reachable at CRA_PG_DSN; skipping Postgres-backed tests")
    Session = db.session_factory(engine)
    session = Session()
    yield session
    session.close()


def test_publish_and_restate_against_real_postgres(pg_session, small_book):
    total_mv = small_book.total_mv()
    twr = registry.get_metric("daily_twr")
    day_idx = 5
    account_id = 0
    value = twr(
        float(total_mv[account_id, day_idx - 1]),
        float(total_mv[account_id, day_idx]),
        float(small_book.flows[account_id, day_idx]),
    )
    as_of = str(small_book.dates[day_idx].date())
    row1 = persistence.publish(pg_session, "account", account_id, "daily_twr", as_of, value=value)
    assert row1.version == 1

    security_id = int(small_book.positions.iloc[0]["security_id"])
    old_price = float(small_book.prices[day_idx, security_id])
    batch.restate_price_and_republish(pg_session, small_book, security_id, day_idx, old_price * 1.03, account_id)

    original = persistence.get_version(pg_session, "account", account_id, "daily_twr", as_of, 1)
    current = persistence.get_current(pg_session, "account", account_id, "daily_twr", as_of)
    assert original.value == row1.value
    assert current.version == 2
