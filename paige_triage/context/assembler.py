"""Precompute Incident Context — Paige's working set.

Bias: assemble up front on open (and reconcile on notes/resolve). Do not let
the model discover PagerDuty facts one API call at a time.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from paige_triage.models.incident import (
    AlertPayload,
    ChangeEvent,
    Incident,
    IncidentContext,
    IncidentNote,
    RelatedIncident,
)
from paige_triage.models.memory import MemoryBundle


_VOLATILE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
    r"|\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?\b"
    r"|\b\d{10,}\b",
    re.IGNORECASE,
)


def normalize_alert_signature(service_id: str, summary: str, details: dict | None = None) -> str:
    """Drop volatility, keep discriminators — Paige playbook matching."""
    text = summary
    if details:
        # Prefer stable custom_details keys responders reason about
        for key in ("error", "error_type", "component", "check", "metric"):
            if key in details:
                text = f"{text}|{key}={details[key]}"
    cleaned = _VOLATILE.sub("<id>", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return f"{service_id}|{cleaned}"


class ContextAssembler:
    """Builds / refreshes IncidentContext from local stores + connectors."""

    def __init__(
        self,
        *,
        past_incidents: list[RelatedIncident] | None = None,
        related_incidents: list[RelatedIncident] | None = None,
        change_events: list[ChangeEvent] | None = None,
        memory: MemoryBundle | None = None,
        dependency_graph: dict[str, list[str]] | None = None,
    ) -> None:
        self.past_incidents = past_incidents or []
        self.related_incidents = related_incidents or []
        self.change_events = change_events or []
        self.memory = memory
        self.dependency_graph = dependency_graph or {}

    def assemble(
        self,
        incident: Incident,
        alerts: list[AlertPayload],
        notes: list[IncidentNote] | None = None,
        previous: IncidentContext | None = None,
    ) -> IncidentContext:
        skipped: list[str] = []
        for alert in alerts:
            if not alert.signature:
                alert.signature = normalize_alert_signature(
                    incident.service.id, alert.summary, alert.custom_details
                )

        runbook_sections: list[str] = []
        if self.memory:
            for rb in self.memory.runbooks:
                if rb.service_id == incident.service.id:
                    # Cap like Paige custom_details / notes: first useful chunks
                    runbook_sections.append(f"## {rb.title}\n{rb.body[:2000]}")
            # Eager runbook_url from custom_details
            for alert in alerts:
                url = alert.custom_details.get("runbook_url") or alert.custom_details.get(
                    "github"
                )
                if url and isinstance(url, str):
                    runbook_sections.append(f"Linked runbook: {url}")

        deps = self.dependency_graph.get(incident.service.id, [])
        dependency_hints = [f"{incident.service.id} → {d}" for d in deps]

        # Filter history to this service, rank by signature match
        fingerprint = (
            alerts[0].signature
            if alerts and alerts[0].signature
            else f"{incident.service.id}|{incident.title.lower()}"
        )
        past = [
            p
            for p in self.past_incidents
            if p.service_id in (None, incident.service.id)
        ]
        past_ranked = sorted(
            past,
            key=lambda p: (
                1.0
                if fingerprint.split("|")[-1][:30] in p.title.lower()
                else p.similarity
            ),
            reverse=True,
        )[:8]

        related = [
            r
            for r in self.related_incidents
            if r.id != incident.id
        ][:8]

        changes = [
            c
            for c in self.change_events
            if c.service_id in (None, incident.service.id)
        ]
        # Recent-first
        changes = sorted(changes, key=lambda c: c.occurred_at, reverse=True)[:10]

        version = (previous.version + 1) if previous else 1
        return IncidentContext(
            incident=incident,
            alerts=alerts,
            notes=list(notes or []),
            past_incidents=past_ranked,
            related_incidents=related,
            change_events=changes,
            runbook_sections=runbook_sections,
            dependency_hints=dependency_hints,
            skipped_sources=skipped,
            assembled_at=datetime.now(timezone.utc),
            version=version,
        )
