"""Hand-worked checks of the single daily_twr and period_return
implementations in cra.registry. These are the numbers a human would
compute with a calculator, used to pin down that the formula is really
doing flow-adjusted TWR and not a plain price return.
"""
import math

import pytest

from cra import registry


def test_daily_twr_no_flow_is_plain_price_return():
    twr = registry.get_metric("daily_twr")
    # 100,000 -> 102,000 with no external flow is a plain 2% return.
    assert twr(100_000, 102_000, 0.0) == pytest.approx(0.02)


def test_daily_twr_removes_contribution_before_computing_return():
    twr = registry.get_metric("daily_twr")
    # A 5,000 contribution lands in end_mv; the true investment return is
    # (107,500 - 5,000) / 100,000 - 1 = 0.025, not 0.075.
    value = twr(100_000, 107_500, 5_000)
    assert value == pytest.approx(0.025)
    naive_price_return = 107_500 / 100_000 - 1
    assert value != pytest.approx(naive_price_return)


def test_daily_twr_removes_withdrawal_before_computing_return():
    twr = registry.get_metric("daily_twr")
    # A 3,000 withdrawal: (96,000 - (-3,000)) / 100,000 - 1 = -0.01.
    value = twr(100_000, 96_000, -3_000)
    assert value == pytest.approx(-0.01)


def test_daily_twr_rejects_nonpositive_begin_mv():
    twr = registry.get_metric("daily_twr")
    with pytest.raises(ValueError):
        twr(0.0, 1_000.0, 0.0)
    with pytest.raises(ValueError):
        twr(-500.0, 1_000.0, 0.0)


def test_period_return_chain_links_geometrically():
    period = registry.get_metric("period_return")
    daily = [0.01, -0.02, 0.03]
    expected = (1.01 * 0.98 * 1.03) - 1.0
    assert period(daily) == pytest.approx(expected)


def test_period_return_empty_series_is_zero():
    period = registry.get_metric("period_return")
    assert period([]) == pytest.approx(0.0)


def test_duplicate_metric_registration_is_rejected():
    with pytest.raises(ValueError, match="already registered"):

        @registry.register_metric("daily_twr")
        def _second_daily_twr(begin_mv, end_mv, flow):  # pragma: no cover
            return (end_mv - flow) / begin_mv - 1.0
