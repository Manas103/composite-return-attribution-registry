"""Database engine setup.

PostgreSQL is the implemented target (SQLAlchemy + psycopg2), but it is
not installed as an always-on service on the build machine. The pattern
mirrors the sibling repos in this portfolio (payment-exception-triage,
eod-cash-balance-forecasting): a plain DSN, a real connection attempt, and
a clean `pytest.skip` rather than a failure when nothing is listening.
SQLite is the fallback used for the unit tests that are actually exercised
in this session, and is disclosed as such everywhere a number is reported.
"""
from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from cra.models import Base

DEFAULT_PG_DSN = os.environ.get(
    "CRA_PG_DSN", "postgresql+psycopg2://postgres:postgres@127.0.0.1:5432/cra"
)


def make_sqlite_engine(path: str = ":memory:") -> Engine:
    """An in-memory sqlite database is private to the connection that
    created it; FastAPI's sync endpoints run in a threadpool worker that
    is not the thread that built the app, so a plain in-memory engine
    would see "no such table" on the very first request. StaticPool
    (one shared connection, reused across threads) is the standard
    FastAPI+SQLite fix and is what makes the in-memory default usable
    for the demo app and for tests; a file path does not need it but
    accepts the same `check_same_thread=False` safely.
    """
    url = f"sqlite:///{path}" if path != ":memory:" else "sqlite:///:memory:"
    kwargs = {"future": True, "connect_args": {"check_same_thread": False}}
    if path == ":memory:":
        kwargs["poolclass"] = StaticPool
    engine = create_engine(url, **kwargs)
    Base.metadata.create_all(engine)
    return engine


def make_postgres_engine(dsn: str | None = None) -> Engine:
    """Creates a PostgreSQL engine and eagerly tests the connection. Raises
    `sqlalchemy.exc.OperationalError` if nothing is reachable at `dsn`;
    callers (tests, CLI) are expected to catch that and skip, exactly like
    `payment-exception-triage`'s `get_connection`.
    """
    engine = create_engine(dsn or DEFAULT_PG_DSN, future=True)
    with engine.connect():
        pass
    Base.metadata.create_all(engine)
    return engine


def session_factory(engine: Engine) -> sessionmaker:
    return sessionmaker(bind=engine, future=True, expire_on_commit=False)
