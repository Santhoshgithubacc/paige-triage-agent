"""Build a grounded investigation report: sources consulted + remediation steps."""

from __future__ import annotations

from typing import Any

from paige_triage import data_sources
from paige_triage.checkpoint import SupervisorCheckpoint


def build_investigation_report(state: SupervisorCheckpoint) -> dict[str, Any]:
    """Summarize what Paige grounded on and the steps to take."""
    grounded: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(source: str, kind: str, detail: str, quote: str | None = None) -> None:
        key = f"{source}|{detail}|{quote or ''}"
        if key in seen:
            return
        seen.add(key)
        grounded.append(
            {
                "source": source,
                "kind": kind,
                "detail": detail,
                "quote": quote,
                "mode": data_sources.data_mode(),
            }
        )

    ctx = state.context
    add("incident", "pagerduty", f"{ctx.incident.id}: {ctx.incident.title}")
    for alert in ctx.alerts:
        add(
            "alert",
            "pagerduty",
            alert.summary,
            quote=str(alert.custom_details.get("error") or alert.signature),
        )
    for ch in ctx.change_events:
        add("change_event", "deploy", ch.summary, quote=ch.url)
    for section in ctx.runbook_sections:
        first = section.splitlines()[0][:120]
        add("runbook", "knowledge", first, quote=section[:240])
    for dep in ctx.dependency_hints:
        add("topology", "dependency", dep)
    for past in ctx.past_incidents[:3]:
        add(
            "past_incident",
            "memory",
            past.title,
            quote=past.resolution_summary,
        )

    steps: list[str] = []
    evidence_rows: list[dict[str, Any]] = []
    ranked = sorted(state.hypotheses, key=lambda h: h.confidence, reverse=True)
    for hyp in ranked:
        for ev in hyp.evidence:
            evidence_rows.append(
                {
                    "hypothesis": hyp.statement,
                    "hypothesis_status": hyp.status.value,
                    "confidence": hyp.confidence,
                    "evidence_kind": ev.kind.value,
                    "claim": ev.claim,
                    "source": ev.source,
                    "quote": ev.quote,
                }
            )
            add(ev.source, ev.kind.value, ev.claim, quote=ev.quote)

    # Prefer playbook / runbook steps when present in evidence quotes
    for hyp in ranked:
        if hyp.confidence < 0.45:
            continue
        for ev in hyp.evidence:
            if ev.source.startswith("memory.playbook") and ev.quote:
                if ev.quote not in steps and not ev.quote.startswith("Deserialization"):
                    # playbook steps often look like instructions
                    if any(
                        ev.quote.lower().startswith(p)
                        for p in ("confirm", "search", "pause", "skip", "run", "do not", "1.", "2.")
                    ) or "step" in ev.claim.lower():
                        steps.append(ev.quote)
        if steps:
            break

    if not steps:
        # Fall back to recommendations + runbook body lines
        steps.extend(state.recommendations)
        for section in ctx.runbook_sections:
            for line in section.splitlines():
                stripped = line.strip()
                if stripped[:1].isdigit() and ". " in stripped[:4]:
                    steps.append(stripped)

    # Dedupe steps preserving order
    uniq_steps: list[str] = []
    for s in steps:
        if s and s not in uniq_steps:
            uniq_steps.append(s)

    top = ranked[0] if ranked else None
    return {
        "data_mode": data_sources.data_mode(),
        "llm": state.extra.get("llm"),
        "loaded_skill_ids": state.extra.get("loaded_skill_ids") or [],
        "loaded_skills": state.extra.get("loaded_skills") or [],
        "runbook_excerpts": state.extra.get("runbook_excerpts") or [],
        "available_sources": data_sources.grounded_source_catalog(),
        "incident": {
            "id": ctx.incident.id,
            "title": ctx.incident.title,
            "service": ctx.incident.service.name,
            "urgency": ctx.incident.urgency.value,
        },
        "synthesis": state.synthesis,
        "top_hypothesis": top.statement if top else None,
        "top_confidence": top.confidence if top else None,
        "grounded_sources": grounded,
        "evidence": evidence_rows,
        "steps": uniq_steps,
        "recommendations": state.recommendations,
        "progress": state.progress_log,
        "hypotheses": [
            {
                "id": h.id,
                "statement": h.statement,
                "status": h.status.value,
                "confidence": h.confidence,
                "component": h.component,
            }
            for h in ranked
        ],
        "completed": state.completed,
        "canceled": state.canceled,
        "task_id": state.task_id,
    }
