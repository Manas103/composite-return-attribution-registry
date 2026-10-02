"""FastAPI REST API: the second of the two registry consumers.

Two endpoints call a registry metric function live and directly
(`/metrics/twr`, `/metrics/attribution`); three read persisted
`PublishedMetric` rows written by the batch job
(`/accounts/{id}/returns/...`, `/composites/{id}/returns/...`); one drives
the versioned restatement path (`/admin/restate`), which itself calls
`cra.batch.restate_price_and_republish`, which calls the same
`registry.daily_twr`. None of these handlers contain return, composite or
attribution arithmetic of their own; see tests/test_registry_governance.py
for the test that checks this is still true at runtime, not just by
reading the source.
"""
from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from cra import batch, db, persistence, registry
from cra.simulate import generate_book


class TWRRequest(BaseModel):
    begin_mv: float
    end_mv: float
    flow: float = 0.0


class AttributionRequest(BaseModel):
    entity_type: str
    entity_id: int
    sectors: list[str]
    w_p: list[float]
    r_p: list[float]
    w_b: list[float]
    r_b: list[float]


class RestateRequest(BaseModel):
    security_id: int
    day_idx: int
    new_price: float
    affected_account_id: int
    reason: str = "price correction"


def build_app(seed: int = 42, sqlite_path: str = ":memory:") -> FastAPI:
    book = generate_book(seed=seed)
    engine = db.make_sqlite_engine(sqlite_path)
    session_factory = db.session_factory(engine)

    app = FastAPI(title="Composite Return and Attribution Service", version="0.1.0")
    app.state.book = book
    app.state.engine = engine
    app.state.session_factory = session_factory

    def get_session():
        session = app.state.session_factory()
        try:
            yield session
        finally:
            session.close()

    @app.get("/health")
    def health():
        return {"status": "ok", "accounts": app.state.book.n_accounts, "days": app.state.book.n_days}

    @app.post("/metrics/twr")
    def compute_twr(req: TWRRequest):
        """Stateless calculator: calls registry.daily_twr directly. One of
        the two registry consumers the governance test drives."""
        twr = registry.get_metric("daily_twr")
        try:
            value = twr(req.begin_mv, req.end_mv, req.flow)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"daily_twr": value}

    @app.post("/metrics/attribution")
    def compute_attribution(req: AttributionRequest):
        """Stateless attribution: calls registry.brinson_attribution
        directly, live, with caller-supplied sector weights/returns."""
        attribution = registry.get_metric("brinson_attribution")(
            req.sectors, req.w_p, req.r_p, req.w_b, req.r_b
        )
        return attribution

    @app.get("/accounts/{account_id}/returns/{as_of}")
    def get_account_return(account_id: int, as_of: str, session=Depends(get_session)):
        row = persistence.get_current(session, "account", account_id, "daily_twr", as_of)
        if row is None:
            raise HTTPException(status_code=404, detail="no published return for this account/date")
        return {
            "account_id": account_id,
            "as_of": as_of,
            "version": row.version,
            "value": row.value,
            "published_at": row.published_at,
        }

    @app.get("/accounts/{account_id}/returns/{as_of}/versions")
    def get_account_return_versions(account_id: int, as_of: str, session=Depends(get_session)):
        rows = persistence.history(session, "account", account_id, "daily_twr", as_of)
        return [
            {
                "version": r.version,
                "value": r.value,
                "published_at": r.published_at,
                "source_note": r.source_note,
            }
            for r in rows
        ]

    @app.get("/composites/{composite_id}/returns/{as_of}")
    def get_composite_return(composite_id: int, as_of: str, session=Depends(get_session)):
        row = persistence.get_current(session, "composite", composite_id, "daily_twr", as_of)
        if row is None:
            raise HTTPException(status_code=404, detail="no published return for this composite/date")
        return {"composite_id": composite_id, "as_of": as_of, "version": row.version, "value": row.value}

    @app.post("/admin/restate")
    def restate(req: RestateRequest, session=Depends(get_session)):
        """Corrects a prior-date price and publishes a NEW version of the
        one affected account's daily TWR. The previously published
        version is never modified; see
        cra.batch.restate_price_and_republish.
        """
        try:
            result = batch.restate_price_and_republish(
                session,
                app.state.book,
                req.security_id,
                req.day_idx,
                req.new_price,
                req.affected_account_id,
                req.reason,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return result

    return app


app = build_app()
