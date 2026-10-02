"""Independent reference-oracle recomputation of Brinson attribution.

This module shares no code with `cra.registry` or `cra.batch`: no shared
helper functions, no zip/sum-comprehension style copied over, explicit
index loops with running accumulators instead. It exists to catch
implementation bugs in the fast path (wrong index alignment, a transposed
w_p/w_b, a summation-order mistake) that an identical-formula copy would
not catch, since the underlying Brinson-Hood-Beebower identity itself is
a single well-known decomposition, not something with two "different but
both-right" formulas to choose between. Both
`oracle_sector_exposure` (independently re-aggregates sector weights and
returns straight from the book, with plain Python loops instead of the
numpy vectorized approach `cra.batch._sector_weights_returns` uses) and
`oracle_attribution` (independently recomputes the effects and totals) are
diffed against the registry/batch fast path in
tests/test_oracle_cross_check.py.
"""
from __future__ import annotations

import numpy as np

from cra.simulate import SECTORS, Book


def oracle_sector_exposure(
    shares_row: np.ndarray, month_start_price: np.ndarray, sec_return: np.ndarray, sec_sector: np.ndarray
) -> tuple[dict[str, float], dict[str, float]]:
    """Plain-loop reimplementation of sector weight/return aggregation.
    Deliberately avoids numpy boolean masking in favor of explicit Python
    iteration over securities, to be structurally different code from
    `cra.batch._sector_weights_returns`.
    """
    n = len(sec_sector)
    sector_value_sum = {s: 0.0 for s in SECTORS}
    sector_weighted_return_sum = {s: 0.0 for s in SECTORS}
    total_value = 0.0
    for i in range(n):
        value = float(shares_row[i]) * float(month_start_price[i])
        sector = sec_sector[i]
        sector_value_sum[sector] += value
        sector_weighted_return_sum[sector] += value * float(sec_return[i])
        total_value += value

    w: dict[str, float] = {}
    r: dict[str, float] = {}
    for sector in SECTORS:
        if total_value > 0:
            w[sector] = sector_value_sum[sector] / total_value
        else:
            w[sector] = 0.0
        if sector_value_sum[sector] > 0:
            r[sector] = sector_weighted_return_sum[sector] / sector_value_sum[sector]
        else:
            r[sector] = 0.0
    return w, r


def oracle_attribution(
    sectors: list[str], w_p: list[float], r_p: list[float], w_b: list[float], r_b: list[float]
) -> dict:
    """Brute-force recomputation of the Brinson decomposition and its
    reconciliation, with explicit index loops and running accumulators
    rather than zip() and generator-expression sums.
    """
    n = len(sectors)

    total_p = 0.0
    for i in range(n):
        total_p = total_p + (w_p[i] * r_p[i])

    total_b = 0.0
    for i in range(n):
        total_b = total_b + (w_b[i] * r_b[i])

    excess = total_p - total_b

    effects = []
    running_sum = 0.0
    for i in range(n):
        alloc = (w_p[i] - w_b[i]) * r_b[i]
        selection = w_b[i] * (r_p[i] - r_b[i])
        interaction = (w_p[i] - w_b[i]) * (r_p[i] - r_b[i])
        effects.append({"sector": sectors[i], "allocation": alloc, "selection": selection, "interaction": interaction})
        running_sum = running_sum + alloc
        running_sum = running_sum + selection
        running_sum = running_sum + interaction

    return {
        "effects": effects,
        "total_portfolio_return": total_p,
        "total_benchmark_return": total_b,
        "total_excess_return": excess,
        "sum_of_effects": running_sum,
        "reconciliation_discrepancy": running_sum - excess,
    }


def oracle_monthly_attribution(book: Book, entity_type: str, entity_id: int, day_idx: np.ndarray) -> dict:
    """End-to-end oracle path for one entity/month: independent sector
    exposure aggregation feeding the independent attribution recomputation.
    """
    day_start = day_idx[0]
    day_end = day_idx[-1]
    prior_day = day_start - 1 if day_start > 0 else day_start
    month_start_price = book.prices[prior_day]
    month_end_price = book.prices[day_end]
    sec_return = month_end_price / month_start_price - 1.0
    sec_sector = book.securities["sector"].to_numpy()

    if entity_type == "account":
        shares_row = book.shares_matrix[entity_id]
    elif entity_type == "composite":
        members = book.accounts.loc[book.accounts.composite_id == entity_id, "account_id"].to_numpy()
        shares_row = np.zeros(book.n_securities)
        for a in members:
            shares_row = shares_row + book.shares_matrix[a]
    else:
        raise ValueError(f"unknown entity_type {entity_type!r}")

    w_p, r_p = oracle_sector_exposure(shares_row, month_start_price, sec_return, sec_sector)

    w_b = dict(book.benchmark_sector_weights)
    r_b = {}
    for sector in SECTORS:
        total = 0.0
        count = 0
        for i in range(book.n_securities):
            if sec_sector[i] == sector:
                total += float(sec_return[i])
                count += 1
        r_b[sector] = total / count if count > 0 else 0.0

    return oracle_attribution(
        SECTORS,
        [w_p[s] for s in SECTORS],
        [r_p[s] for s in SECTORS],
        [w_b[s] for s in SECTORS],
        [r_b[s] for s in SECTORS],
    )
