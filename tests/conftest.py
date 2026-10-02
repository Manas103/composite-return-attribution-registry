"""Shared fixtures.

`small_book` is a small, fast simulated book (12 accounts, 90 trading
days) used by unit tests that do not need the full 240-account, 3-year
scale. `full_book` is the real 240-account, 756-day book used by the
tests that measure the two numeric claims (attribution reconciliation,
30 restatements); it is session-scoped so it is generated only once.
"""
from __future__ import annotations

import pytest

from cra import registry
from cra.simulate import generate_book


@pytest.fixture(autouse=True)
def _clear_call_log():
    registry.reset_call_log()
    yield
    registry.reset_call_log()


@pytest.fixture()
def small_book():
    return generate_book(seed=123, n_accounts=12, n_securities=18, n_days=90)


@pytest.fixture(scope="session")
def full_book():
    return generate_book(seed=42)
