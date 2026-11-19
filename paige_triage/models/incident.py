"""Incident and precomputed Incident Context models.

Paige precomputes a high-fidelity working set when an incident opens —
not a summary, and not lazy one-API-call-at-a-time discovery.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Urgency(str, Enum):
    HIGH = "high"
    LOW = "low"


class IncidentStatus(str, Enum):
    TRIGGERED = "triggered"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class ServiceRef(BaseModel):
    id: str
    name: str
    description: str | None = None
    team: str | None = None


class AlertPayload(BaseModel):
    id: str = Field(default_factory=lambda: f"alert-{uuid4().hex[:8]}")
    summary: str
    severity: str = "critical"
    source: str | None = None
    component: str | None = None
    custom_details: dict[str, Any] = Field(default_factory=dict)
    received_at: datetime = Field(default_factory=utcnow)
    # Normalized signature used for playbook / recollection matching
    signature: str | None = None


class IncidentNote(BaseModel):
    id: str = Field(default_factory=lambda: f"note-{uuid4().hex[:8]}")
    author: str
    body: str
    created_at: datetime = Field(default_factory=utcnow)


class ChangeEvent(BaseModel):
    id: str = Field(default_factory=lambda: f"chg-{uuid4().hex[:8]}")
    summary: str
    service_id: str | None = None
    actor: str | None = None
    occurred_at: datetime = Field(default_factory=utcnow)
    url: str | None = None


class RelatedIncident(BaseModel):
    id: str
    number: int | None = None
    title: str
    service_id: str | None = None
    status: IncidentStatus = IncidentStatus.RESOLVED
    similarity: float = 0.0
    resolution_summary: str | None = None


class Incident(BaseModel):
    id: str = Field(default_factory=lambda: f"inc-{uuid4().hex[:8]}")
    number: int | None = None
    title: str
    status: IncidentStatus = IncidentStatus.TRIGGERED
    urgency: Urgency = Urgency.HIGH
    service: ServiceRef
    created_at: datetime = Field(default_factory=utcnow)
    acknowledged_at: datetime | None = None
    resolved_at: datetime | None = None
    assignees: list[str] = Field(default_factory=list)


class IncidentContext(BaseModel):
    """Precomputed working set Paige reasons over for one incident + service.

    Rebuilt / reconciled on trigger, notes, and resolve (with debounce).
    """

    incident: Incident
    alerts: list[AlertPayload] = Field(default_factory=list)
    notes: list[IncidentNote] = Field(default_factory=list)
    past_incidents: list[RelatedIncident] = Field(default_factory=list)
    related_incidents: list[RelatedIncident] = Field(default_factory=list)
    change_events: list[ChangeEvent] = Field(default_factory=list)
    runbook_sections: list[str] = Field(default_factory=list)
    dependency_hints: list[str] = Field(default_factory=list)
    skipped_sources: list[str] = Field(default_factory=list)
    assembled_at: datetime = Field(default_factory=utcnow)
    version: int = 1

    def primary_alert(self) -> AlertPayload | None:
        return self.alerts[0] if self.alerts else None

    def fingerprint(self) -> str:
        alert = self.primary_alert()
        if alert and alert.signature:
            return alert.signature
        if alert:
            return f"{self.incident.service.id}|{alert.summary}"
        return f"{self.incident.service.id}|{self.incident.title}"
