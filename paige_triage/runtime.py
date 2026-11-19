"""Factory wiring a full Paige-style triage stack for demos and API."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable, Awaitable

from paige_triage import data_sources
from paige_triage.agents.supervisor import TriageSupervisor
from paige_triage.connectors.registry import build_default_registry
from paige_triage.context.assembler import ContextAssembler, normalize_alert_signature
from paige_triage.demo_data import sample_alerts, sample_incident, sample_memory
from paige_triage.engine.session import DeepInvestigationSession
from paige_triage.llm import HeuristicLLM, get_llm
from paige_triage.models.incident import (
    AlertPayload,
    ChangeEvent,
    Incident,
    IncidentStatus,
    RelatedIncident,
    ServiceRef,
    Urgency,
)
from paige_triage.skills.loader import DEFAULT_SKILLS, SkillRegistry


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def load_bundle_from_sources(incident_id: str = "inc-demo-001") -> dict:
    """Load incident + alerts from CSV (toy) or raise in production until wired."""
    incidents = data_sources.load_incidents()
    row = next((r for r in incidents if r.get("incident_id") == incident_id), None)
    if row is None and incidents:
        row = incidents[0]
    if row is None:
        # Fallback to in-module samples if CSV missing
        return {
            "incident": sample_incident(),
            "alerts": sample_alerts(),
            "memory": sample_memory(),
        }

    svc = ServiceRef(
        id=row.get("service_id") or "PXXXXXX",
        name=row.get("service_name") or "unknown",
        team=row.get("team"),
    )
    incident = Incident(
        id=row.get("incident_id") or incident_id,
        number=int(row["number"]) if row.get("number") else None,
        title=row.get("title") or "Untitled incident",
        status=IncidentStatus(row.get("status") or "triggered"),
        urgency=Urgency(row.get("urgency") or "high"),
        service=svc,
        created_at=_parse_dt(row.get("created_at")) or sample_incident().created_at,
        assignees=[row["assignee"]] if row.get("assignee") else [],
    )

    alerts: list[AlertPayload] = []
    for a in data_sources.load_alerts(incident_id=incident.id):
        details = {
            "error": a.get("error"),
            "component": a.get("component"),
            "metric": a.get("metric"),
            "runbook_url": a.get("runbook_url"),
        }
        summary = a.get("summary") or incident.title
        alerts.append(
            AlertPayload(
                id=a.get("alert_id") or "alert",
                summary=summary,
                severity=a.get("severity") or "critical",
                source=a.get("source"),
                component=a.get("component"),
                custom_details={k: v for k, v in details.items() if v},
                received_at=_parse_dt(a.get("received_at")) or incident.created_at,
                signature=normalize_alert_signature(svc.id, summary, details),
            )
        )
    if not alerts:
        alerts = sample_alerts()

    return {"incident": incident, "alerts": alerts, "memory": sample_memory()}


def _history_from_sources(service_id: str) -> tuple[
    list[RelatedIncident], list[RelatedIncident], list[ChangeEvent]
]:
    past: list[RelatedIncident] = []
    related: list[RelatedIncident] = []
    for row in data_sources.load_incidents():
        if row.get("incident_id") == "inc-demo-001":
            continue
        item = RelatedIncident(
            id=row.get("incident_id") or "inc",
            number=int(row["number"]) if row.get("number") else None,
            title=row.get("title") or "",
            service_id=row.get("service_id"),
            status=IncidentStatus(row.get("status") or "resolved"),
            similarity=0.9 if row.get("service_id") == service_id else 0.4,
            resolution_summary=(
                "Cleanup playbook; poison offset skipped"
                if "duplicate" in (row.get("title") or "").lower()
                else None
            ),
        )
        if row.get("service_id") == service_id:
            past.append(item)
        else:
            related.append(item)

    changes: list[ChangeEvent] = []
    for row in data_sources.load_changes(service_id=service_id):
        changes.append(
            ChangeEvent(
                id=row.get("change_id") or "chg",
                summary=row.get("summary") or "",
                service_id=row.get("service_id"),
                actor=row.get("actor"),
                occurred_at=_parse_dt(row.get("occurred_at")) or sample_incident().created_at,
                url=row.get("url"),
            )
        )
    return past, related, changes


def build_demo_supervisor(
    checkpoint_dir: Path | str = ".paige_checkpoints",
    on_progress: Callable[[str], Awaitable[None] | None] | None = None,
    use_openai: bool = False,
    incident_id: str = "inc001",
    use_llm: bool | None = None,
) -> tuple[TriageSupervisor, dict]:
    """Build supervisor. Default LLM is local Llama via Ollama (PAIGE_LLM=llama).

    Set PAIGE_LLM=heuristic or use_llm=False for offline/tests.
    """
    bundle = load_bundle_from_sources(incident_id=incident_id)
    memory = bundle["memory"]
    past, related, _ignored = _history_from_sources(bundle["incident"].service.id)
    # Prefer changes tagged for this incident when CSV provides incident_hint
    raw_changes = data_sources.load_changes(service_id=bundle["incident"].service.id)
    filtered_raw = [
        r for r in raw_changes if r.get("incident_hint") in (None, "", incident_id)
    ] or raw_changes
    from paige_triage.models.incident import ChangeEvent

    changes = []
    for row in filtered_raw:
        changes.append(
            ChangeEvent(
                id=row.get("change_id") or "chg",
                summary=row.get("summary") or "",
                service_id=row.get("service_id"),
                actor=row.get("actor"),
                occurred_at=_parse_dt(row.get("occurred_at")) or bundle["incident"].created_at,
                url=row.get("url"),
            )
        )
    deps = data_sources.load_dependencies()

    runbook_docs = {
        r["source_url"]: r["body"]
        for r in data_sources.load_runbooks(service_id=bundle["incident"].service.id)
        if r.get("source_url")
    }

    assembler = ContextAssembler(
        past_incidents=past,
        related_incidents=related,
        change_events=changes,
        memory=memory,
        dependency_graph=deps,
    )
    connectors = build_default_registry(
        log_corpus=data_sources.load_logs(),
        metric_series=data_sources.load_metrics(),
        runbook_docs=runbook_docs,
        commits=data_sources.load_commits(),
    )
    # Default: real Llama via get_llm(). Tests/offline: PAIGE_LLM=heuristic or use_llm=False.
    if use_llm is False:
        llm = HeuristicLLM()
    else:
        llm = get_llm()
    supervisor = TriageSupervisor(
        checkpoint_dir=checkpoint_dir,
        connectors=connectors,
        skills=SkillRegistry(DEFAULT_SKILLS),
        memory=memory,
        assembler=assembler,
        llm=llm,
        on_progress=on_progress,
    )
    return supervisor, bundle


def _stack_for_incident(incident_id: str):
    bundle = load_bundle_from_sources(incident_id=incident_id)
    memory = bundle["memory"]
    past, related, _ignored = _history_from_sources(bundle["incident"].service.id)
    raw_changes = data_sources.load_changes(service_id=bundle["incident"].service.id)
    filtered_raw = [
        r for r in raw_changes if r.get("incident_hint") in (None, "", incident_id)
    ] or raw_changes
    changes = []
    for row in filtered_raw:
        changes.append(
            ChangeEvent(
                id=row.get("change_id") or "chg",
                summary=row.get("summary") or "",
                service_id=row.get("service_id"),
                actor=row.get("actor"),
                occurred_at=_parse_dt(row.get("occurred_at")) or bundle["incident"].created_at,
                url=row.get("url"),
            )
        )
    deps = data_sources.load_dependencies()
    runbook_docs = {
        r["source_url"]: r["body"]
        for r in data_sources.load_runbooks(service_id=bundle["incident"].service.id)
        if r.get("source_url")
    }
    assembler = ContextAssembler(
        past_incidents=past,
        related_incidents=related,
        change_events=changes,
        memory=memory,
        dependency_graph=deps,
    )
    connectors = build_default_registry(
        log_corpus=data_sources.load_logs(),
        metric_series=data_sources.load_metrics(),
        runbook_docs=runbook_docs,
        commits=data_sources.load_commits(),
    )
    return bundle, memory, assembler, connectors


def build_deep_session(
    *,
    incident_id: str = "inc001",
    checkpoint_db: Path | str = ".paige_checkpoints/langgraph.sqlite",
    on_progress: Callable[[str], Awaitable[None] | None] | None = None,
    deep_mode: bool = True,
    max_cycles: int = 6,
    replan_interval_s: float = 0.0,
) -> tuple[DeepInvestigationSession, dict]:
    """Production-style LangGraph session with LLM sub-agents (CSV toys kept)."""
    bundle, memory, assembler, connectors = _stack_for_incident(incident_id)
    session = DeepInvestigationSession(
        connectors=connectors,
        assembler=assembler,
        memory=memory,
        skills=SkillRegistry(DEFAULT_SKILLS),
        checkpoint_db=checkpoint_db,
        deep_mode=deep_mode,
        max_cycles=max_cycles,
        replan_interval_s=replan_interval_s,
        on_progress=on_progress,
    )
    return session, bundle
