"""Reference-oracle cross-check: an independently coded (not sharing
helper functions, not vectorized) recomputation of both the sector
exposure aggregation and the Brinson attribution, diffed against the
registry fast path across a random sample of accounts, composites and
months from the full simulated book.
"""
import numpy as np
import pytest

from cra import batch, oracle


@pytest.mark.parametrize("trial", range(40))
def test_oracle_matches_fast_path(full_book, trial):
    rng = np.random.default_rng(1000 + trial)
    months = full_book.month_boundaries()
    entity_type = "account" if rng.random() < 0.9 else "composite"
    if entity_type == "account":
        entity_id = int(rng.integers(0, full_book.n_accounts))
    else:
        entity_id = int(rng.integers(0, len(full_book.composites)))
    month_label, day_idx = months[int(rng.integers(0, len(months)))]

    fast = batch.monthly_attribution(full_book, entity_type, entity_id, str(month_label), day_idx)
    slow = oracle.oracle_monthly_attribution(full_book, entity_type, entity_id, day_idx)

    assert fast["total_portfolio_return"] == pytest.approx(slow["total_portfolio_return"], abs=1e-12)
    assert fast["total_benchmark_return"] == pytest.approx(slow["total_benchmark_return"], abs=1e-12)
    assert fast["sum_of_effects"] == pytest.approx(slow["sum_of_effects"], abs=1e-12)
    for fe, se in zip(fast["effects"], slow["effects"]):
        assert fe["sector"] == se["sector"]
        assert fe["allocation"] == pytest.approx(se["allocation"], abs=1e-12)
        assert fe["selection"] == pytest.approx(se["selection"], abs=1e-12)
        assert fe["interaction"] == pytest.approx(se["interaction"], abs=1e-12)
