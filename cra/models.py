"""SQLAlchemy ORM models.

Only `PublishedMetric` matters for the restatement/versioning claim: it is
an append-only table (rows are never updated or deleted by this codebase).
A restatement inserts a new row with `version = old_version + 1`; the old
row stays exactly as published. "Current" is just "the highest version for
this (entity_type, entity_id, metric_name, as_of)".
"""
from __future__ import annotations

from sqlalchemy import Column, Float, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class PublishedMetric(Base):
    """One published, versioned calculation.

    `value` holds the scalar result for scalar metrics (daily_twr,
    period_return, composite_return, excess_return). `value_json` holds
    the full structured result for brinson_attribution (effects list plus
    totals). Exactly one of the two is populated depending on
    `metric_name`.
    """

    __tablename__ = "published_metrics"
    __table_args__ = (
        UniqueConstraint(
            "entity_type", "entity_id", "metric_name", "as_of", "version",
            name="uq_published_metric_version",
        ),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    entity_type = Column(String(16), nullable=False)  # 'account' | 'composite'
    entity_id = Column(Integer, nullable=False)
    metric_name = Column(String(32), nullable=False)
    as_of = Column(String(16), nullable=False)  # ISO date (YYYY-MM-DD) or year-month (YYYY-MM)
    version = Column(Integer, nullable=False)
    value = Column(Float, nullable=True)
    value_json = Column(Text, nullable=True)
    published_at = Column(String(32), nullable=False)
    source_note = Column(String(128), nullable=True)


class PriceCorrection(Base):
    """Audit log of price restatements. The live `prices` series used for
    computation is simulated in-memory (cra.simulate.Book); in this
    service a restatement is applied to that in-memory series and logged
    here, then the affected PublishedMetric rows are republished as a new
    version. The log itself is never mutated (append-only), matching the
    spirit of the published-metric versioning even though the claim only
    requires the published *return* to be versioned.
    """

    __tablename__ = "price_corrections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    security_id = Column(Integer, nullable=False)
    as_of = Column(String(16), nullable=False)
    old_price = Column(Float, nullable=False)
    new_price = Column(Float, nullable=False)
    corrected_at = Column(String(32), nullable=False)
    reason = Column(String(128), nullable=True)
