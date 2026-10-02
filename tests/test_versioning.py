"""Versioned restatement: the numeric claim is that 30 seeded
restatement scenarios across different accounts/dates all produce a new
version while the prior publication stays byte-identical in the database.
"""
from __future__ import annotations

import numpy as np
import pytest

from cra import batch, db, persistence, registry


def _publish_initial_return(session, book, account_id: int, day_idx: int):
    total_mv = book.total_mv()
    twr = registry.get_metric("daily_twr")
    value = twr(
        float(total_mv[account_id, day_idx - 1]),
        float(total_mv[account_id, day_idx]),
        float(book.flows[account_id, day_idx]),
    )
    as_of = str(book.dates[day_idx].date())
    row = persistence.publish(session, "account", account_id, "daily_twr", as_of, value=value)
    return as_of, row


def test_30_seeded_restatements_all_version_correctly(full_book):
    """Seeds exactly 30 (account, date, security) restatement scenarios,
    publishes an initial version for each, restates a held security's
    price on that date, and asserts: a new version (2) is published, and
    the original version (1) row is untouched (same value, same
    published_at) for all 30. Reports the real pass count.
    """
    engine = db.make_sqlite_engine(":memory:")
    Session = db.session_factory(engine)
    session = Session()

    rng = np.random.default_rng(777)
    scenarios = []
    seen = set()
    while len(scenarios) < 30:
        account_id = int(rng.integers(0, full_book.n_accounts))
        day_idx = int(rng.integers(1, full_book.n_days))
        key = (account_id, day_idx)
        if key in seen:
            continue
        held = full_book.positions.loc[full_book.positions.account_id == account_id, "security_id"]
        if held.empty:
            continue
        seen.add(key)
        security_id = int(held.iloc[0])
        scenarios.append((account_id, day_idx, security_id))

    assert len(scenarios) == 30

    passed = 0
    failures = []
    for account_id, day_idx, security_id in scenarios:
        as_of, original_row = _publish_initial_return(session, full_book, account_id, day_idx)
        original_value = original_row.value
        original_published_at = original_row.published_at
        original_version = original_row.version

        old_price = float(full_book.prices[day_idx, security_id])
        new_price = old_price * float(rng.uniform(0.9, 1.1))
        batch.restate_price_and_republish(session, full_book, security_id, day_idx, new_price, account_id)

        reread_original = persistence.get_version(session, "account", account_id, "daily_twr", as_of, original_version)
        current = persistence.get_current(session, "account", account_id, "daily_twr", as_of)

        ok = (
            reread_original is not None
            and reread_original.value == original_value
            and reread_original.published_at == original_published_at
            and current is not None
            and current.version == original_version + 1
        )
        if ok:
            passed += 1
        else:
            failures.append((account_id, day_idx, security_id))

    assert passed == 30, f"only {passed}/30 restatement scenarios versioned correctly; failures: {failures}"
