"""Synthetic book generator.

Everything in this module is simulated. There is no real market data, no
real account data, and no real client anywhere in this repository. Prices
are a seeded geometric random walk with a per-sector common factor and
per-security idiosyncratic noise; accounts, positions, cash flows and the
benchmark are all synthetic constructions chosen to be realistic enough to
exercise the return, composite-aggregate and attribution math honestly,
not to resemble any particular real fund.

Scale (the "simulated 240-account, 3-year book" claim): the defaults below
produce 240 accounts, 60 securities across 6 sectors, and 756 trading days
(3 x 252), measured exactly in scripts/measure_benchmarks.py and
tests/test_simulate_scale.py rather than asserted here.

Simplifications disclosed up front (see README "Limitations"):
  - Positions are static share counts (buy and hold). There is no trading
    activity; cash flows land in the account's cash sleeve, not into new
    purchases. This models a buy-and-hold composite book, not an actively
    traded one.
  - Cash earns no yield.
  - Sector attribution uses each position's own price return for the
    month (no intra-position cash flows are possible since positions
    never trade), which is a standard simplification: attribution is
    computed over the invested sleeve, cash drag is a separate, simpler
    effect not modelled here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SECTORS = ["Technology", "Financials", "Healthcare", "Energy", "Industrials", "Consumer"]

# Fixed synthetic benchmark sector weights (sum to 1.0). Deliberately not
# identical to the realized portfolio sector mix, so allocation effects in
# the attribution are genuine rather than structurally zero.
BENCHMARK_SECTOR_WEIGHTS = {
    "Technology": 0.25,
    "Financials": 0.20,
    "Healthcare": 0.15,
    "Energy": 0.10,
    "Industrials": 0.15,
    "Consumer": 0.15,
}

N_ACCOUNTS = 240
N_COMPOSITES = 3
N_SECURITIES = 60
N_SECTORS = len(SECTORS)
TRADING_DAYS = 756  # 3 years x 252 trading days


@dataclass
class Book:
    seed: int
    dates: pd.DatetimeIndex
    securities: pd.DataFrame  # security_id, ticker, sector
    accounts: pd.DataFrame  # account_id, composite_id
    composites: pd.DataFrame  # composite_id, name
    positions: pd.DataFrame  # account_id, security_id, shares
    prices: np.ndarray  # (T, S)
    shares_matrix: np.ndarray  # (N, S)
    flows: np.ndarray  # (N, T) external cash flow landing at start of day
    initial_cash: np.ndarray  # (N,)
    benchmark_sector_weights: dict = field(default_factory=lambda: dict(BENCHMARK_SECTOR_WEIGHTS))

    @property
    def n_accounts(self) -> int:
        return self.shares_matrix.shape[0]

    @property
    def n_securities(self) -> int:
        return self.shares_matrix.shape[1]

    @property
    def n_days(self) -> int:
        return self.prices.shape[0]

    def invested_mv(self) -> np.ndarray:
        """(N, T) invested market value: shares_matrix (N,S) @ prices.T (S,T)."""
        return self.shares_matrix @ self.prices.T

    def cash_mv(self) -> np.ndarray:
        """(N, T) cash sleeve balance: initial cash plus cumulative flows
        received up to and including each day (flows land at day start)."""
        cum_flows = np.cumsum(self.flows, axis=1)
        return self.initial_cash[:, None] + cum_flows

    def total_mv(self) -> np.ndarray:
        return self.invested_mv() + self.cash_mv()

    def month_boundaries(self) -> list[tuple[pd.Period, np.ndarray]]:
        """List of (year-month period, day-index array) in chronological
        order, used to define attribution periods."""
        periods = pd.PeriodIndex(self.dates, freq="M")
        out = []
        for p in periods.unique():
            idx = np.where(periods == p)[0]
            out.append((p, idx))
        return out


def generate_book(
    seed: int = 42,
    n_accounts: int = N_ACCOUNTS,
    n_securities: int = N_SECURITIES,
    n_days: int = TRADING_DAYS,
) -> Book:
    rng = np.random.default_rng(seed)

    dates = pd.bdate_range("2022-01-03", periods=n_days)

    # --- securities and sectors ---
    sec_sector = np.array([SECTORS[i % N_SECTORS] for i in range(n_securities)])
    rng.shuffle(sec_sector)  # mix the sector assignment order
    securities = pd.DataFrame(
        {
            "security_id": np.arange(n_securities),
            "ticker": [f"SEC{i:03d}" for i in range(n_securities)],
            "sector": sec_sector,
        }
    )

    # --- price paths: per-sector common factor + idiosyncratic noise ---
    sector_mu = {s: rng.uniform(0.00010, 0.00045) for s in SECTORS}
    sector_sigma = {s: rng.uniform(0.006, 0.013) for s in SECTORS}
    idio_sigma = rng.uniform(0.004, 0.010, size=n_securities)

    log_returns = np.zeros((n_days, n_securities))
    for sector in SECTORS:
        mask = sec_sector == sector
        n_sec_in_sector = mask.sum()
        if n_sec_in_sector == 0:
            continue
        factor = rng.normal(sector_mu[sector], sector_sigma[sector], size=n_days)
        idio = rng.normal(0.0, 1.0, size=(n_days, n_sec_in_sector)) * idio_sigma[mask]
        log_returns[:, mask] = factor[:, None] + idio

    start_prices = rng.uniform(20.0, 300.0, size=n_securities)
    prices = np.empty((n_days, n_securities))
    prices[0] = start_prices * (1.0 + log_returns[0])
    for t in range(1, n_days):
        prices[t] = prices[t - 1] * (1.0 + log_returns[t])

    # --- composites and accounts ---
    composites = pd.DataFrame(
        {
            "composite_id": np.arange(N_COMPOSITES),
            "name": [f"Composite-{chr(65 + i)}" for i in range(N_COMPOSITES)],
        }
    )
    account_composite = np.arange(n_accounts) % N_COMPOSITES
    accounts = pd.DataFrame(
        {
            "account_id": np.arange(n_accounts),
            "composite_id": account_composite,
        }
    )

    # --- positions: 5-12 securities per account, static share counts ---
    shares_matrix = np.zeros((n_accounts, n_securities))
    position_rows = []
    for acc in range(n_accounts):
        k = rng.integers(5, 13)
        held = rng.choice(n_securities, size=k, replace=False)
        for sec in held:
            target_value = rng.uniform(5_000, 50_000)
            shares = max(1.0, round(target_value / prices[0, sec]))
            shares_matrix[acc, sec] = shares
            position_rows.append((acc, int(sec), shares))
    positions = pd.DataFrame(position_rows, columns=["account_id", "security_id", "shares"])

    invested_start = shares_matrix @ prices[0]
    initial_cash = invested_start * rng.uniform(0.02, 0.10, size=n_accounts)

    # --- external cash flows: 3-8 events per account over the book ---
    flows = np.zeros((n_accounts, n_days))
    for acc in range(n_accounts):
        n_events = rng.integers(3, 9)
        event_days = rng.choice(np.arange(5, n_days), size=n_events, replace=False)
        base_value = invested_start[acc] + initial_cash[acc]
        for day in event_days:
            sign = rng.choice([1.0, -1.0], p=[0.6, 0.4])
            pct = rng.uniform(0.02, 0.07)
            flows[acc, day] = sign * pct * base_value

    return Book(
        seed=seed,
        dates=dates,
        securities=securities,
        accounts=accounts,
        composites=composites,
        positions=positions,
        prices=prices,
        shares_matrix=shares_matrix,
        flows=flows,
        initial_cash=initial_cash,
    )
