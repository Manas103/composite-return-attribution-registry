"""The governed metric registry.

Every return, composite-aggregate and attribution metric used anywhere in
this service (the FastAPI endpoints, the batch publishing job, the
restatement path, the tests) is defined exactly once, in this file.
Nothing outside this module is allowed to recompute a metric formula from
scratch: callers look up or import the function object and invoke it.

Two independent enforcement layers back that claim up (both are exercised
by tests/test_registry_governance.py, not just asserted in a docstring):

1. `register_metric` raises at *import time* if the same metric name is
   registered twice. That catches someone pasting a second implementation
   under the registry's own decorator.
2. Every registered function is wrapped so that each call appends
   `(metric_name, id(wrapper))` to the module-level `CALL_LOG`. The test
   drives the FastAPI app (one consumer) and the batch job (a second,
   independent consumer) through the same metrics, then asserts the
   logged function identity is the exact same object both times. A
   second, independent implementation of, say, daily TWR living directly
   inside api.py or batch.py would simply never show up in CALL_LOG under
   the "daily_twr" name, and that test would fail. This is the "shared
   call marker" approach described in the build brief, chosen over an AST
   static-analysis check because it tests the real call path, including
   through any intermediate service-layer wiring, rather than pattern
   matching source text.
"""
from __future__ import annotations

import functools
from typing import Callable, Sequence

# (metric_name, id(registered_function)) for every invocation of every
# registered metric, in call order. Tests clear/inspect this; production
# code never needs to.
CALL_LOG: list[tuple[str, int]] = []

_REGISTRY: dict[str, Callable] = {}


def register_metric(name: str):
    """Decorator that registers `func` as the one true implementation of
    metric `name`. Raises ValueError if `name` is already registered,
    which is what makes a second independent implementation impossible to
    smuggle in under the registry itself (it would have to live entirely
    outside the registry to avoid this, which is exactly what CALL_LOG
    catches).
    """

    def deco(func: Callable) -> Callable:
        if name in _REGISTRY:
            raise ValueError(
                f"metric '{name}' is already registered by "
                f"{_REGISTRY[name].__wrapped__.__qualname__}; a metric may be "
                "defined exactly once in this registry"
            )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            CALL_LOG.append((name, id(wrapper)))
            return func(*args, **kwargs)

        _REGISTRY[name] = wrapper
        return wrapper

    return deco


def get_metric(name: str) -> Callable:
    """Look up a registered metric function by name. This is how
    consumers that want the function object itself (rather than calling
    it directly via the module attribute) are supposed to obtain it, e.g.
    the batch job's dispatch table.
    """
    return _REGISTRY[name]


def registered_metrics() -> list[str]:
    return sorted(_REGISTRY)


def reset_call_log() -> None:
    CALL_LOG.clear()


@register_metric("daily_twr")
def daily_twr(begin_mv: float, end_mv: float, flow: float) -> float:
    """True daily time-weighted return for one valuation day, with an
    external cash flow handled correctly rather than folded into a plain
    price return.

    Convention: `flow` is the net external cash flow (contributions
    positive, withdrawals negative) that lands at the *start* of the day,
    before the day's market move, and is included in `end_mv` (the
    account's own cash sleeve receives it). Subtracting it back out of
    the numerator isolates the day's investment performance from the
    flow itself, which is the standard "true daily TWR" construction used
    when valuation points are daily (GIPS-style), as opposed to a Modified
    Dietz approximation that time-weights the flow within the period.

        r_t = (EMV_t - F_t) / BMV_t - 1

    `begin_mv` must be strictly positive (a zero or negative beginning
    value makes a percentage return undefined).
    """
    if begin_mv <= 0:
        raise ValueError(f"begin_mv must be positive to compute a return, got {begin_mv}")
    return (end_mv - flow) / begin_mv - 1.0


@register_metric("period_return")
def period_return(daily_returns: Sequence[float]) -> float:
    """Geometric linking (chain-linking) of a sequence of single-period
    returns into one cumulative return over the whole span. This is the
    only place daily returns are ever compounded; composite and account
    period returns both route through this function.
    """
    acc = 1.0
    for r in daily_returns:
        acc *= 1.0 + r
    return acc - 1.0


@register_metric("composite_return")
def composite_return(begin_mvs: Sequence[float], member_returns: Sequence[float]) -> float:
    """Asset-weighted (beginning-of-day market value weighted) composite
    aggregate return for one period, across composite member accounts.
    This is the standard GIPS "aggregate method": the composite is treated
    as if it were one account whose beginning value is the sum of member
    beginning values.
    """
    total = sum(begin_mvs)
    if total <= 0:
        raise ValueError(f"composite beginning market value must be positive, got {total}")
    return sum(mv * r for mv, r in zip(begin_mvs, member_returns)) / total


@register_metric("excess_return")
def excess_return(portfolio_return: float, benchmark_return: float) -> float:
    """Portfolio return minus benchmark return for the same period."""
    return portfolio_return - benchmark_return


@register_metric("brinson_attribution")
def brinson_attribution(
    sectors: Sequence[str],
    w_p: Sequence[float],
    r_p: Sequence[float],
    w_b: Sequence[float],
    r_b: Sequence[float],
) -> dict:
    """Brinson-Hood-Beebower allocation / selection / interaction
    decomposition, by sector, of portfolio return versus a benchmark
    return for one period.

        allocation_i  = (w_p,i - w_b,i) * r_b,i
        selection_i   = w_b,i * (r_p,i - r_b,i)
        interaction_i = (w_p,i - w_b,i) * (r_p,i - r_b,i)

    Summed across sectors, allocation + selection + interaction equals
    total_portfolio_return - total_benchmark_return *exactly*, as an
    algebraic identity (expand the three terms and everything telescopes;
    see docs/attribution_identity_proof.txt for the full expansion). That
    is why the measured reconciliation discrepancy in this repo is at
    float64 machine-epsilon scale rather than some looser tolerance: it
    is not a tuned result, it is what an exact identity measures as.

    `w_p` and `w_b` need not individually sum to 1 (a sector-less cash
    sleeve can be left out), but the reconciliation identity above holds
    regardless.
    """
    effects = []
    for sector, wp, rp, wb, rb in zip(sectors, w_p, r_p, w_b, r_b):
        allocation = (wp - wb) * rb
        selection = wb * (rp - rb)
        interaction = (wp - wb) * (rp - rb)
        effects.append(
            {
                "sector": sector,
                "allocation": allocation,
                "selection": selection,
                "interaction": interaction,
            }
        )
    total_p = sum(wp * rp for wp, rp in zip(w_p, r_p))
    total_b = sum(wb * rb for wb, rb in zip(w_b, r_b))
    sum_of_effects = sum(e["allocation"] + e["selection"] + e["interaction"] for e in effects)
    return {
        "effects": effects,
        "total_portfolio_return": total_p,
        "total_benchmark_return": total_b,
        "total_excess_return": total_p - total_b,
        "sum_of_effects": sum_of_effects,
        "reconciliation_discrepancy": sum_of_effects - (total_p - total_b),
    }
