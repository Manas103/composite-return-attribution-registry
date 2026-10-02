# Composite Return and Attribution Service with a Governed Metric Registry

A Python/FastAPI service over a simulated 240-account, 3-year book that
computes daily time-weighted returns (true TWR, cash-flow adjusted),
asset-weighted composite aggregates, and Brinson-style benchmark-relative
sector attribution, through a single governed metric registry that both
the REST API and a batch reporting job call into rather than each keeping
its own copy of the math. Price restatements publish a new version of the
affected return; the original published version is never touched. Every
number in the Measured Results section below was actually run on this
machine this session, not targeted: the attribution reconciliation
discrepancy and the 30/30 restatement count are pasted output from
`scripts/measure_benchmarks.py` and `pytest`, not hand-picked.

## Why this exists

This is a small version of the return-calculation layer inside a real
asset manager's performance and reporting stack: the part that takes
priced positions and cash flows and turns them into a GIPS-style composite
return, sector attribution versus a benchmark, and a defensible audit
trail when a price gets corrected after the fact. The governance angle
(one registry, two consumers, a test that would catch a second
implementation) is the part of that stack that is easy to get wrong in
practice, when an API team and a batch/reporting team each reimplement
"the same" return formula slightly differently and the numbers quietly
drift apart.

## Honest framing, up front

- **All data is simulated.** Prices, accounts, positions, cash flows and
  the benchmark are a seeded synthetic generator (`cra/simulate.py`), not
  real market or client data. Dates are an arbitrary synthetic 3-year
  window (2022-01-03 onward); they do not represent a real trading
  calendar and the window itself carries no significance beyond "3 years
  of business days."
- **Positions are static (buy and hold).** There is no trading activity in
  this simulation; cash flows land in an account's cash sleeve, not into
  new purchases. This models a buy-and-hold composite book, not an
  actively traded one. Cash earns no yield.
- **Attribution is computed on the invested sleeve.** Sector returns for
  attribution use each position's own price return for the month; cash
  drag is a separate, simpler effect and is not modelled as its own
  attribution term here.
- **Restatement republishes the one directly affected day's return, not a
  forward cascade.** A corrected price technically also changes the
  *following* day's beginning value in a full TWR chain; this
  implementation republishes the day whose end-of-day value used the
  corrected price and stops there. See Limitations.
- **PostgreSQL is implemented but not running as a service on this
  machine**, so it was not exercised this session; SQLite is what was
  actually run, and is disclosed as such everywhere a number is reported.
  `tests/test_postgres_versioning.py` exercises the real Postgres path
  and skips cleanly (not a failure) when nothing is listening, the same
  pattern `payment-exception-triage` and `eod-cash-balance-forecasting`
  use elsewhere in this portfolio.
- **Machine and toolchain**: Windows 11, AMD Ryzen 7 7800X3D (8 physical /
  16 logical cores), native Python 3.12.10 (no WSL needed, this is a pure
  Python project), FastAPI 0.115.0, SQLAlchemy 2.0.35, psycopg2-binary
  2.9.9, numpy 2.1.1, pandas 2.2.3, pytest 8.3.3. No GPU is used.

## Architecture

```
composite-return-attribution-registry/
├── cra/
│   ├── registry.py       the governed metric registry: daily_twr, period_return,
│   │                     composite_return, excess_return, brinson_attribution.
│   │                     Every metric is defined exactly once here; register_metric
│   │                     rejects a duplicate name, and every call is logged to
│   │                     CALL_LOG(metric_name, function_id) for the governance test.
│   ├── simulate.py       synthetic book generator: 240 accounts, 60 securities /
│   │                     6 sectors, 756 trading days, cash flows, a fixed synthetic
│   │                     benchmark. Disclosed-synthetic, seeded, reproducible.
│   ├── batch.py          consumer #1: the batch/reporting job. Computes every
│   │                     account's daily TWR, composite daily/period returns, and
│   │                     monthly Brinson attribution for every account and
│   │                     composite, by calling cra.registry. Also the restatement
│   │                     entry point (restate_price_and_republish).
│   ├── api.py             consumer #2: FastAPI app. Two endpoints call a registry
│   │                     function directly and live (/metrics/twr,
│   │                     /metrics/attribution); others read persisted
│   │                     PublishedMetric rows or drive the restatement path, which
│   │                     itself calls back into cra.batch / cra.registry.
│   ├── persistence.py    the only module that writes PublishedMetric rows. Always
│   │                     inserts current_version+1; never updates or deletes a
│   │                     published row.
│   ├── models.py         SQLAlchemy ORM: PublishedMetric (versioned, append-only),
│   │                     PriceCorrection (audit log).
│   ├── db.py             SQLite (StaticPool, in-memory or file) and PostgreSQL
│   │                     (SQLAlchemy + psycopg2) engine construction.
│   └── oracle.py          independent brute-force recomputation of sector exposure
│                          and Brinson attribution, no shared code with
│                          registry.py/batch.py, used to cross-check the fast path.
├── tests/                 pytest suite, 64 tests + 1 clean skip (no local Postgres)
├── scripts/
│   └── measure_benchmarks.py   reproducible script behind every numeric claim below
└── docs/
    ├── benchmark_output.txt    raw output of measure_benchmarks.py
    └── test_output.txt         raw output of `pytest -v`
```

### Why a decorator-based registry instead of a plain module of functions

A plain module of functions can still be duplicated: nothing stops a
second file from defining its own `daily_twr`. `register_metric` adds two
things a plain function can't: it raises at import time on a duplicate
*name*, and it wraps every call so a shared `CALL_LOG` records which
function object actually ran. The second property is the one that matters
for the governance claim, because a rogue implementation would not use
the decorator at all, it would just inline the formula; `CALL_LOG` is what
lets a test notice that the inlined version never shows up as a
registry call. See `tests/test_registry_governance.py`.

### Why true daily TWR instead of Modified Dietz

With daily valuation points (which this service has, since prices are
daily), true daily TWR is exact rather than an approximation: the flow
`F_t` is simply subtracted back out of the day's ending value before
dividing by the day's beginning value, `r_t = (EMV_t - F_t) / BMV_t - 1`,
so the flow itself contributes exactly zero to the measured return. A
Modified Dietz formula would be needed only if cash flows happened
*between* valuation points without a priced snapshot at the flow date,
which is not this system's situation.

### Why Brinson-Hood-Beebower with an interaction term

Allocation `(w_p - w_b) * r_b`, selection `w_b * (r_p - r_b)` and
interaction `(w_p - w_b) * (r_p - r_b)` summed across sectors reduce
algebraically, term by term, to `total_portfolio_return -
total_benchmark_return` exactly:

```
w_p*r_p = w_b*r_b + (w_p-w_b)*r_b + w_b*(r_p-r_b) + (w_p-w_b)*(r_p-r_b)
```

Expand the right-hand side and every `w_b*r_b` term cancels, leaving
`w_p*r_p`. That is why the measured reconciliation discrepancy below is at
float64 machine-epsilon scale rather than some chosen tolerance from
below: it is what an exact identity measures as, not a tuned result.

## Validation

Three layers, in increasing order of how much they trust the
implementation:

1. **Hand-worked unit tests** (`test_registry_twr.py`,
   `test_composite_return.py`, `test_attribution_reconciliation.py`'s
   two-sector example) check the registry functions against numbers
   computed by hand.
2. **Full-book reconciliation** (`test_attribution_reconciliation.py`'s
   second test) runs the real Brinson decomposition across every account,
   every composite, every month in the simulated book (8,505
   account/composite/period combinations) and asserts the max
   discrepancy is under 1e-10.
3. **Independent oracle cross-check** (`test_oracle_cross_check.py`, 40
   parametrized trials) diffs the registry fast path against
   `cra/oracle.py`, a structurally different recomputation (explicit
   index loops and running accumulators instead of `zip`/generator sums,
   a separate sector-exposure aggregation written without numpy masking)
   across random accounts, composites and months.

Pasted `pytest -v` output: `docs/test_output.txt`. Raw run:

```
64 passed, 1 skipped in 1.93s
```

The one skip is `test_postgres_versioning.py`, skipped cleanly because no
PostgreSQL instance is reachable on this machine (see "Building and
running").

## Findings: what broke

The in-memory SQLite engine for the demo FastAPI app initially raised
`sqlite3.OperationalError: no such table: published_metrics` on the very
first API request, even though `Base.metadata.create_all(engine)` had
just run successfully a few lines earlier in the same process. First
hypothesis: a missing `session.commit()` somewhere, or the schema
migration running against the wrong engine object. That was wrong; adding
print statements showed the exact same `engine` object issuing both
calls.

The actual cause: SQLite's `:memory:` database is private to the
connection that created it, and SQLAlchemy's default engine opens a new
connection per checkout. `Base.metadata.create_all()` ran on one
connection (and its tables vanished with it); FastAPI's sync endpoint ran
in a Starlette threadpool worker, which checked out a *different*
connection, to a *different*, freshly-empty in-memory database. The
discriminating measurement was running the exact same build_app() call
from a plain script (single thread, worked) versus through
`TestClient` (worker thread, failed) with everything else identical.

Fix: `db.make_sqlite_engine` now passes `poolclass=StaticPool` and
`connect_args={"check_same_thread": False}` when the path is `:memory:`,
which is the standard FastAPI+SQLite pattern, one shared connection
reused across threads instead of one new (and empty) database per
checkout. File-based SQLite paths do not have this failure mode (the
file persists across connections) but get the same `connect_args` for
consistency.

## Measured results

Machine: Windows 11, AMD Ryzen 7 7800X3D (8 physical / 16 logical cores),
native Python 3.12.10, SQLite (in-memory, StaticPool) exercised this
session; PostgreSQL implemented via SQLAlchemy + psycopg2 but not running
as a service, so not exercised. Full raw output:
`docs/benchmark_output.txt`.

| Claim | Measured | Meets claim |
|---|---|---|
| Daily TWR, composite aggregates, sector attribution all implemented and tested | 181,200 daily_twr calls + 8,505 attribution calls, all passing | yes |
| Simulated book scale | 240 accounts, 60 securities / 6 sectors, 756 trading days, 3 composites, 2,039 position rows, 35 calendar months | yes |
| One registry, two consumers, duplication-proof | governance test passes for both daily_twr and brinson_attribution across API and batch call paths | yes |
| Restated price publishes new version, original intact | 30/30 seeded restatement scenarios | yes |
| **Attribution reconciled to total excess return under 1e-10** | **max discrepancy 2.776e-17 absolute, across all 8,505 account/composite/period combinations** | **yes** |

The attribution reconciliation number is the one that matters most: it is
not "close to 1e-10", it is seven orders of magnitude inside the
tolerance, because the decomposition is an exact algebraic identity (see
Architecture) rather than an approximation. The independent oracle
cross-check (100 random samples in the benchmark script, 40 in the pytest
suite) agrees with the fast path to the same float64 epsilon, max
1.388e-17.

## Building and running

```bash
cd projects/composite-return-attribution-registry
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt

# tests (SQLite only; Postgres test skips cleanly without a server)
.venv/Scripts/python.exe -m pytest tests -v

# the reproducible benchmark script behind every numeric claim above
.venv/Scripts/python.exe -m scripts.measure_benchmarks

# run the API
.venv/Scripts/python.exe -m uvicorn cra.api:app --port 0   # PORT=0 picks a free port; pass --port <N> to pin one
```

To actually exercise the PostgreSQL path (not run this session, since
Postgres is not an always-on service on this machine):

```bash
# stand up a local Postgres and create the cra database, then:
export CRA_PG_DSN="postgresql+psycopg2://postgres:postgres@127.0.0.1:5432/cra"
.venv/Scripts/python.exe -m pytest tests/test_postgres_versioning.py -v
```

## Sibling comparison

`intraday-position-pnl-attribution` elsewhere in this portfolio also does
return/P&L attribution, but at a different timescale and with a different
reconciliation shape: it Taylor-decomposes intraday option P&L into
delta/gamma/vega/theta against a full-repricing oracle and reports a
*residual ratio* (the Taylor approximation's leftover error versus true
repricing, necessarily nonzero since a Taylor expansion is approximate).
This repo's Brinson sector decomposition is not an approximation of
anything, allocation + selection + interaction is algebraically exact, so
there is no residual to report, only float64 rounding. The honest
contrast: when the attribution method is itself approximate (Greeks), the
reconciliation number tells you how good the approximation is; when it is
an exact identity (Brinson), the reconciliation number only tells you
whether the implementation is correct, which is a weaker and easier claim
to satisfy well.
(https://github.com/Manas103/intraday-position-pnl-attribution)

## Limitations

- Restatement republishes only the single day whose end-of-day value used
  the corrected price. It does not cascade through subsequent chain-linked
  period returns, which in a full TWR system would also change (day t+1's
  beginning value equals day t's corrected ending value). A production
  system would need to walk the correction forward; this repo stops at
  the directly affected day, which is enough to demonstrate and test the
  versioning behavior honestly but is a real scope gap.
- Positions are static; there is no trading, so turnover, transaction
  costs and the interaction between trading and cash flows are entirely
  out of scope.
- PostgreSQL was not exercised this session (not an always-on service on
  this machine); the SQLAlchemy/psycopg2 path is implemented and has its
  own test, but only the SQLite path produced the numbers in this README.
- Attribution periods are calendar months; no daily or multi-period
  smoothed attribution is implemented.
- The benchmark is synthetic (equal-weighted sector averages with fixed
  weights), not a real index; it was constructed to make allocation and
  selection effects genuinely nonzero, not to resemble any specific
  published index.
