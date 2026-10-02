"""Batch / reporting job: the first of the two registry consumers.

Everything in here that computes a metric calls through `cra.registry`.
This module does not define daily_twr, composite_return or
brinson_attribution itself; it only orchestrates which inputs to feed the
registry and what to do with the result (publish it, or hand it back as a
report). `cra.api` is the second consumer, calling the exact same
registry functions from request handlers instead of a batch loop; see
tests/test_registry_governance.py for the test that would fail if either
consumer stopped doing that.
"""
from __future__ import annotations

import numpy as np

from cra import persistence, registry
from cra.simulate import SECTORS, Book


def compute_account_daily_twr_series(book: Book, account_id: int) -> list[float]:
    """Daily TWR for one account across the whole book. Day 0 has no prior
    valuation point, so returns are defined for days 1..T-1 (length T-1).
    """
    total_mv = book.total_mv()
    twr = registry.get_metric("daily_twr")
    out = []
    for t in range(1, book.n_days):
        begin_mv = float(total_mv[account_id, t - 1])
        end_mv = float(total_mv[account_id, t])
        flow = float(book.flows[account_id, t])
        out.append(twr(begin_mv, end_mv, flow))
    return out


def compute_all_account_daily_twr(book: Book) -> dict[int, list[float]]:
    """Daily TWR for every account. This is the "batch job touches the
    whole 240-account, 3-year book" code path; it calls registry.daily_twr
    once per account per day (240 * 755 calls for the default book size).
    """
    total_mv = book.total_mv()
    twr = registry.get_metric("daily_twr")
    series_by_account: dict[int, list[float]] = {}
    for acc in range(book.n_accounts):
        out = []
        for t in range(1, book.n_days):
            out.append(twr(float(total_mv[acc, t - 1]), float(total_mv[acc, t]), float(book.flows[acc, t])))
        series_by_account[acc] = out
    return series_by_account


def compute_composite_daily_returns(
    book: Book, composite_id: int, account_twr: dict[int, list[float]]
) -> list[float]:
    """Asset-weighted composite daily return, day 1..T-1, via
    registry.composite_return. `account_twr` is the per-account daily TWR
    series produced by compute_all_account_daily_twr (or a subset).
    """
    total_mv = book.total_mv()
    members = book.accounts.loc[book.accounts.composite_id == composite_id, "account_id"].to_numpy()
    composite_ret = registry.get_metric("composite_return")
    out = []
    for i in range(book.n_days - 1):  # i corresponds to day t = i+1, begin_mv at column i
        begin_mvs = [float(total_mv[a, i]) for a in members]
        member_returns = [account_twr[a][i] for a in members]
        out.append(composite_ret(begin_mvs, member_returns))
    return out


def compute_period_return(daily_returns: list[float]) -> float:
    return registry.get_metric("period_return")(daily_returns)


def _entity_shares_row(book: Book, entity_type: str, entity_id: int) -> np.ndarray:
    if entity_type == "account":
        return book.shares_matrix[entity_id]
    if entity_type == "composite":
        members = book.accounts.loc[book.accounts.composite_id == entity_id, "account_id"].to_numpy()
        return book.shares_matrix[members].sum(axis=0)
    raise ValueError(f"unknown entity_type {entity_type!r}")


def _sector_weights_returns(
    shares_row: np.ndarray, month_start_price: np.ndarray, sec_return: np.ndarray, sec_sector: np.ndarray
) -> tuple[dict[str, float], dict[str, float]]:
    begin_value = shares_row * month_start_price
    total = begin_value.sum()
    w: dict[str, float] = {}
    r: dict[str, float] = {}
    for sector in SECTORS:
        mask = sec_sector == sector
        sector_value = begin_value[mask].sum()
        w[sector] = (sector_value / total) if total > 0 else 0.0
        if sector_value > 0:
            r[sector] = float((begin_value[mask] * sec_return[mask]).sum() / sector_value)
        else:
            r[sector] = 0.0
    return w, r


def monthly_attribution(
    book: Book, entity_type: str, entity_id: int, month_label: str, day_idx: np.ndarray
) -> dict:
    """Brinson attribution for one entity (account or composite) for one
    calendar month, via registry.brinson_attribution. Benchmark sector
    weights are the fixed synthetic BENCHMARK_SECTOR_WEIGHTS; benchmark
    sector returns are the equal-weighted average price return of every
    security in that sector for the month (the synthetic benchmark),
    independent of what this entity actually holds.
    """
    day_start = day_idx[0]
    day_end = day_idx[-1]
    prior_day = day_start - 1 if day_start > 0 else day_start
    month_start_price = book.prices[prior_day]
    month_end_price = book.prices[day_end]
    sec_return = month_end_price / month_start_price - 1.0
    sec_sector = book.securities["sector"].to_numpy()

    w_b = dict(book.benchmark_sector_weights)
    r_b = {}
    for sector in SECTORS:
        mask = sec_sector == sector
        r_b[sector] = float(sec_return[mask].mean()) if mask.any() else 0.0

    shares_row = _entity_shares_row(book, entity_type, entity_id)
    w_p, r_p = _sector_weights_returns(shares_row, month_start_price, sec_return, sec_sector)

    attribution = registry.get_metric("brinson_attribution")(
        SECTORS,
        [w_p[s] for s in SECTORS],
        [r_p[s] for s in SECTORS],
        [w_b[s] for s in SECTORS],
        [r_b[s] for s in SECTORS],
    )
    attribution["entity_type"] = entity_type
    attribution["entity_id"] = entity_id
    attribution["month"] = month_label
    return attribution


def run_full_attribution(book: Book, entity_types: tuple[str, ...] = ("account", "composite")) -> list[dict]:
    """Attribution for every requested entity type, every entity, every
    month in the book. With the default book (240 accounts, 3 composites,
    36 months) and both entity types this is 243 * 36 = 8748 calls to
    registry.brinson_attribution.
    """
    results = []
    months = book.month_boundaries()
    for month_label, day_idx in months:
        if "account" in entity_types:
            for acc in range(book.n_accounts):
                results.append(monthly_attribution(book, "account", acc, str(month_label), day_idx))
        if "composite" in entity_types:
            for comp in range(len(book.composites)):
                results.append(monthly_attribution(book, "composite", comp, str(month_label), day_idx))
    return results


def restate_price_and_republish(
    session,
    book: Book,
    security_id: int,
    day_idx: int,
    new_price: float,
    affected_account_id: int,
    reason: str = "price correction",
) -> dict:
    """The restatement path: correct a price for a prior date, log the
    correction, recompute the one daily_twr calculation it affects for
    `affected_account_id` (the day whose end-of-day value used that
    price), and publish it as a NEW version via cra.persistence.publish.
    The original published version is never touched.

    Scope, disclosed: this republishes the single directly affected day's
    return. It does not cascade the restatement forward through
    subsequent chain-linked period returns (a day t+1 return's beginning
    value also changed, in principle); a production system would need to
    walk that forward. That is out of scope here and called out in the
    README limitations.
    """
    old_price = float(book.prices[day_idx, security_id])
    persistence.log_price_correction(session, security_id, str(book.dates[day_idx].date()), old_price, new_price, reason)
    book.prices[day_idx, security_id] = new_price

    total_mv = book.total_mv()
    as_of = str(book.dates[day_idx].date())
    twr = registry.get_metric("daily_twr")
    if day_idx == 0:
        raise ValueError("cannot restate day 0, no prior valuation point exists")
    begin_mv = float(total_mv[affected_account_id, day_idx - 1])
    end_mv = float(total_mv[affected_account_id, day_idx])
    flow = float(book.flows[affected_account_id, day_idx])
    new_value = twr(begin_mv, end_mv, flow)

    row = persistence.publish(
        session,
        entity_type="account",
        entity_id=affected_account_id,
        metric_name="daily_twr",
        as_of=as_of,
        value=new_value,
        source_note=f"restatement: security {security_id} price corrected {old_price} -> {new_price}",
    )
    return {"old_price": old_price, "new_price": new_price, "new_value": new_value, "published_row_version": row.version}
