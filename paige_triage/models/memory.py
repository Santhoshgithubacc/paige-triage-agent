"""Service-scoped Paige memory artifacts.

Memory stays inside the tenant and service. Paige improves by deepening
the catalog of what happened on each service — not by cross-customer learning.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Observation(BaseModel):
    """Durable facts about a service (stack, log location, dashboards, deps)."""

    id: str = Field(default_factory=lambda: f"obs-{uuid4().hex[:8]}")
    service_id: str
    key: str
    value: str
    source: str = "observed"
    created_at: datetime = Field(default_factory=utcnow)


class Recollection(BaseModel):
    """Incident-scoped: what separated signal from noise and what was tried."""

    id: str = Field(default_factory=lambda: f"rec-{uuid4().hex[:8]}")
    service_id: str
    alert_signature: str
    title: str
    discriminative_query: str | None = None
    error_fingerprint: str | None = None
    helpful_steps: list[str] = Field(default_factory=list)
    failed_steps: list[str] = Field(default_factory=list)
    referenced_artifacts: list[str] = Field(default_factory=list)
    outcome: str | None = None
    created_at: datetime = Field(default_factory=utcnow)


class Playbook(BaseModel):
    """Human-promoted recollection: versioned evidence + steps + guardrails."""

    id: str = Field(default_factory=lambda: f"pb-{uuid4().hex[:8]}")
    service_id: str
    alert_signature: str
    title: str
    version: int = 1
    steps: list[str] = Field(default_factory=list)
    evidence_quotes: list[str] = Field(default_factory=list)
    preconditions: list[str] = Field(default_factory=list)
    guardrails: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)


class RunbookDoc(BaseModel):
    """Customer SOP / runbook content (Confluence, GitHub, uploaded)."""

    id: str = Field(default_factory=lambda: f"rb-{uuid4().hex[:8]}")
    service_id: str
    title: str
    source_url: str | None = None
    body: str
    tags: list[str] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=utcnow)


class ServiceProfile(BaseModel):
    """Metadata Paige observed via runbook, payload, and interactions."""

    service_id: str
    cloud_provider: str | None = None
    region: str | None = None
    service_type: str | None = None
    log_queries: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class MemoryBundle(BaseModel):
    service_id: str
    observations: list[Observation] = Field(default_factory=list)
    recollections: list[Recollection] = Field(default_factory=list)
    playbooks: list[Playbook] = Field(default_factory=list)
    runbooks: list[RunbookDoc] = Field(default_factory=list)
    profile: ServiceProfile | None = None

    def match_playbooks(self, signature: str) -> list[Playbook]:
        return [p for p in self.playbooks if p.alert_signature == signature]

    def match_recollections(self, signature: str, limit: int = 5) -> list[Recollection]:
        exact = [r for r in self.recollections if r.alert_signature == signature]
        if exact:
            return exact[:limit]
        # Soft match on shared stems
        stem = signature.split("|")[-1][:40].lower()
        soft = [
            r
            for r in self.recollections
            if stem and stem in (r.error_fingerprint or r.title).lower()
        ]
        return soft[:limit]
