"""Brinson attribution: a hand-worked two-sector example, then the
numeric claim measured across the whole simulated book: every
account/composite/period's sum of attribution effects (allocation +
selection + interaction) must equal total portfolio return minus total
benchmark return to within 1e-10 absolute error.

This is not a tuned result: allocation + selection + interaction summed
across sectors is an exact algebraic identity (see cra/registry.py
docstring for the expansion), so the measured discrepancy is float64
machine epsilon, not something approaching a chosen tolerance from below.
"""
import pytest

from cra import batch, registry


def test_brinson_attribution_two_sector_hand_example():
    brinson = registry.get_metric("brinson_attribution")
    # Portfolio overweight Tech (60% vs 50% benchmark), Tech beat its
    # benchmark counterpart (3% vs 2%); Financials underweight (40% vs
    # 50%), in line with its benchmark (1% vs 1%).
    result = brinson(
        sectors=["Technology", "Financials"],
        w_p=[0.60, 0.40],
        r_p=[0.03, 0.01],
        w_b=[0.50, 0.50],
        r_b=[0.02, 0.01],
    )
    total_p = 0.60 * 0.03 + 0.40 * 0.01
    total_b = 0.50 * 0.02 + 0.50 * 0.01
    assert result["total_portfolio_return"] == pytest.approx(total_p)
    assert result["total_benchmark_return"] == pytest.approx(total_b)
    assert result["total_excess_return"] == pytest.approx(total_p - total_b)
    # Allocation for Technology: (0.60-0.50) * 0.02 = 0.0020
    assert result["effects"][0]["allocation"] == pytest.approx(0.0020)
    # Selection for Technology: 0.50 * (0.03-0.02) = 0.0050
    assert result["effects"][0]["selection"] == pytest.approx(0.0050)
    assert abs(result["reconciliation_discrepancy"]) < 1e-10


def test_attribution_reconciliation_across_full_simulated_book(full_book):
    """The numeric claim: measured across every account and composite,
    for every month in the 3-year book, max |sum_of_effects -
    total_excess_return| must be under 1e-10 (absolute).
    """
    results = batch.run_full_attribution(full_book)
    assert len(results) > 0
    max_discrepancy = max(abs(r["reconciliation_discrepancy"]) for r in results)
    n_entities = full_book.n_accounts + len(full_book.composites)
    n_months = len(full_book.month_boundaries())
    assert len(results) == n_entities * n_months
    assert max_discrepancy < 1e-10, (
        f"attribution reconciliation discrepancy {max_discrepancy} exceeded 1e-10 "
        f"across {len(results)} account/composite/period combinations"
    )
