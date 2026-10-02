"""Reproducible measurement script for every numeric claim in the README.

Run: .venv/Scripts/python.exe scripts/measure_benchmarks.py
Raw output committed at docs/benchmark_output.txt.
"""
from __future__ import annotations

import time

import numpy as np

from cra import batch, db, oracle, persistence, registry
from cra.simulate import generate_book


def main() -> None:
    t_start = time.time()
    print("=== 1. Simulated book scale ===")
    t0 = time.time()
    book = generate_book(seed=42)
    print(f"generate_book time: {time.time() - t0:.3f}s")
    print(f"accounts: {book.n_accounts}")
    print(f"securities: {book.n_securities}")
    print(f"trading days: {book.n_days}")
    print(f"composites: {len(book.composites)}")
    print(f"position rows: {len(book.positions)}")
    n_months = len(book.month_boundaries())
    print(f"calendar months spanned: {n_months}")

    print()
    print("=== 2. Daily TWR + composite aggregate (existence) ===")
    t0 = time.time()
    all_twr = batch.compute_all_account_daily_twr(book)
    n_twr_calls = book.n_accounts * (book.n_days - 1)
    print(f"all-account daily TWR time: {time.time() - t0:.3f}s ({n_twr_calls} registry.daily_twr calls)")
    comp0_daily = batch.compute_composite_daily_returns(book, 0, all_twr)
    comp0_period = batch.compute_period_return(comp0_daily)
    print(f"composite 0 ({book.composites.iloc[0]['name']}) full-period return: {comp0_period:.6f}")

    print()
    print("=== 3. Attribution reconciliation (numeric claim, tolerance 1e-10 absolute) ===")
    t0 = time.time()
    results = batch.run_full_attribution(book)
    elapsed = time.time() - t0
    discrepancies = np.array([abs(r["reconciliation_discrepancy"]) for r in results])
    print(f"attribution run time: {elapsed:.3f}s")
    print(f"entities x periods measured: {len(results)} "
          f"({book.n_accounts} accounts + {len(book.composites)} composites, x {n_months} months)")
    print(f"max |sum_of_effects - total_excess_return|: {discrepancies.max():.3e}")
    print(f"mean |discrepancy|: {discrepancies.mean():.3e}")
    print(f"meets 1e-10 claim: {bool(discrepancies.max() < 1e-10)}")

    print()
    print("=== 4. Oracle cross-check (independent recomputation) ===")
    rng = np.random.default_rng(2024)
    months = book.month_boundaries()
    max_diff = 0.0
    n_checked = 100
    t0 = time.time()
    for _ in range(n_checked):
        etype = "account" if rng.random() < 0.9 else "composite"
        eid = int(rng.integers(0, book.n_accounts if etype == "account" else len(book.composites)))
        month_label, day_idx = months[int(rng.integers(0, len(months)))]
        fast = batch.monthly_attribution(book, etype, eid, str(month_label), day_idx)
        slow = oracle.oracle_monthly_attribution(book, etype, eid, day_idx)
        diff = abs(fast["sum_of_effects"] - slow["sum_of_effects"])
        diff = max(diff, abs(fast["total_excess_return"] - slow["total_excess_return"]))
        max_diff = max(max_diff, diff)
    print(f"oracle cross-check time: {time.time() - t0:.3f}s ({n_checked} samples)")
    print(f"max |fast_path - oracle|: {max_diff:.3e}")

    print()
    print("=== 5. Versioned restatement: 30 seeded scenarios ===")
    engine = db.make_sqlite_engine(":memory:")
    Session = db.session_factory(engine)
    session = Session()
    rng = np.random.default_rng(777)
    scenarios = []
    seen = set()
    while len(scenarios) < 30:
        account_id = int(rng.integers(0, book.n_accounts))
        day_idx = int(rng.integers(1, book.n_days))
        key = (account_id, day_idx)
        if key in seen:
            continue
        held = book.positions.loc[book.positions.account_id == account_id, "security_id"]
        if held.empty:
            continue
        seen.add(key)
        scenarios.append((account_id, day_idx, int(held.iloc[0])))

    total_mv = book.total_mv()
    twr = registry.get_metric("daily_twr")
    passed = 0
    for account_id, day_idx, security_id in scenarios:
        value = twr(
            float(total_mv[account_id, day_idx - 1]),
            float(total_mv[account_id, day_idx]),
            float(book.flows[account_id, day_idx]),
        )
        as_of = str(book.dates[day_idx].date())
        row1 = persistence.publish(session, "account", account_id, "daily_twr", as_of, value=value)
        old_price = float(book.prices[day_idx, security_id])
        new_price = old_price * float(rng.uniform(0.9, 1.1))
        batch.restate_price_and_republish(session, book, security_id, day_idx, new_price, account_id)
        reread = persistence.get_version(session, "account", account_id, "daily_twr", as_of, row1.version)
        current = persistence.get_current(session, "account", account_id, "daily_twr", as_of)
        if (
            reread is not None
            and reread.value == row1.value
            and reread.published_at == row1.published_at
            and current is not None
            and current.version == row1.version + 1
        ):
            passed += 1
    print(f"restatement scenarios passed: {passed}/30")

    print()
    print(f"total measurement script time: {time.time() - t_start:.3f}s")


if __name__ == "__main__":
    main()
