"""Publish and restate versioned calculations.

This is the only module that writes `PublishedMetric` rows. It never
computes a metric value itself; every value passed to `publish` must
already have come out of a `cra.registry` function. `publish` always
inserts a brand new row at `current_version + 1`; there is no update or
delete path for a published row anywhere in this codebase, which is what
makes "the original published version remains intact" true by
construction rather than by convention.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from cra.models import PriceCorrection, PublishedMetric


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def current_version(
    session: Session, entity_type: str, entity_id: int, metric_name: str, as_of: str
) -> int:
    stmt = select(func.max(PublishedMetric.version)).where(
        PublishedMetric.entity_type == entity_type,
        PublishedMetric.entity_id == entity_id,
        PublishedMetric.metric_name == metric_name,
        PublishedMetric.as_of == as_of,
    )
    result = session.execute(stmt).scalar_one_or_none()
    return result or 0


def publish(
    session: Session,
    entity_type: str,
    entity_id: int,
    metric_name: str,
    as_of: str,
    value: float | None = None,
    value_obj: dict | None = None,
    source_note: str | None = None,
) -> PublishedMetric:
    """Insert the next version for this (entity, metric, as_of). Version 1
    on first publication, old_version + 1 on every restatement. Never
    touches an existing row.
    """
    next_version = current_version(session, entity_type, entity_id, metric_name, as_of) + 1
    row = PublishedMetric(
        entity_type=entity_type,
        entity_id=entity_id,
        metric_name=metric_name,
        as_of=as_of,
        version=next_version,
        value=value,
        value_json=json.dumps(value_obj) if value_obj is not None else None,
        published_at=_now(),
        source_note=source_note,
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def get_current(session: Session, entity_type: str, entity_id: int, metric_name: str, as_of: str):
    v = current_version(session, entity_type, entity_id, metric_name, as_of)
    if v == 0:
        return None
    return get_version(session, entity_type, entity_id, metric_name, as_of, v)


def get_version(
    session: Session, entity_type: str, entity_id: int, metric_name: str, as_of: str, version: int
):
    stmt = select(PublishedMetric).where(
        PublishedMetric.entity_type == entity_type,
        PublishedMetric.entity_id == entity_id,
        PublishedMetric.metric_name == metric_name,
        PublishedMetric.as_of == as_of,
        PublishedMetric.version == version,
    )
    return session.execute(stmt).scalar_one_or_none()


def history(session: Session, entity_type: str, entity_id: int, metric_name: str, as_of: str):
    stmt = (
        select(PublishedMetric)
        .where(
            PublishedMetric.entity_type == entity_type,
            PublishedMetric.entity_id == entity_id,
            PublishedMetric.metric_name == metric_name,
            PublishedMetric.as_of == as_of,
        )
        .order_by(PublishedMetric.version)
    )
    return list(session.execute(stmt).scalars())


def log_price_correction(
    session: Session, security_id: int, as_of: str, old_price: float, new_price: float, reason: str | None = None
) -> PriceCorrection:
    row = PriceCorrection(
        security_id=security_id,
        as_of=as_of,
        old_price=old_price,
        new_price=new_price,
        corrected_at=_now(),
        reason=reason,
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row
