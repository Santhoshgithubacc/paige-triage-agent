"""Pluggable LLM for Paige planning / synthesis.

Default: local Llama via Ollama (llama3.2) — real-world Paige simulation.
Fallback: HeuristicLLM when Ollama is unreachable.

Env:
  PAIGE_LLM=llama|openai|heuristic
  PAIGE_LLM_MODEL=llama3.2:latest
  OLLAMA_BASE_URL=http://127.0.0.1:11434/v1
  OPENAI_API_KEY=...   # only for PAIGE_LLM=openai
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Protocol

import httpx

from paige_triage.models.incident import IncidentContext
from paige_triage.models.memory import MemoryBundle


def _humanize_operator_text(text: str) -> str:
    """Strip internal tooling / scoring language from user-facing copy."""
    t = text or ""
    t = re.sub(r"\btool\.[a-z0-9_]+\b", "the telemetry", t, flags=re.I)
    t = re.sub(r"\b(search_logs|get_related_services|fetch_runbook|query_metrics|get_incident_details)\b", "checks", t, flags=re.I)
    t = re.sub(r"\bconfidence(?:\s+level)?\s*(?:of\s*)?[:=]?\s*0?\.\d+\b", "this looks solid", t, flags=re.I)
    t = re.sub(r"\bconfidence\s*[:=]?\s*\d+(?:\.\d+)?%?\b", "this looks solid", t, flags=re.I)
    t = re.sub(r"\bsource=toy\b", "", t, flags=re.I)
    t = re.sub(r"\bcount=\d+\b", "", t, flags=re.I)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t


def render_structured_reply(data: dict[str, Any]) -> str:
    """Turn model JSON into a short natural Paige reply with clear sections."""
    headline = _humanize_operator_text(
        str(data.get("headline") or data.get("summary") or data.get("answer") or "").strip()
    )
    details = _humanize_operator_text(
        str(data.get("details") or data.get("reasoning") or "").strip()
    )
    next_steps = data.get("next_steps") or data.get("recommendations") or data.get("next") or []
    if isinstance(next_steps, str):
        next_steps = [next_steps] if next_steps.strip() else []
    elif not isinstance(next_steps, list):
        next_steps = []
    cleaned_steps: list[str] = []
    for x in next_steps:
        s = _humanize_operator_text(str(x).strip())
        if not s or s.lower() in {"optional", "optional second", "helpful follow-up"}:
            continue
        if re.search(r"verify quote from|tool\.|source=toy|confidence", s, re.I):
            continue
        cleaned_steps.append(s)
    cleaned_steps = cleaned_steps[:5]

    parts: list[str] = []
    if headline:
        parts.append(headline)
    if details and details != headline:
        parts.append(details)
    if cleaned_steps:
        parts.append("Next:\n" + "\n".join(f"• {s}" for s in cleaned_steps))
    return "\n\n".join(parts).strip()


class LLM(Protocol):
    name: str

    async def propose_hypotheses(
        self,
        context: IncidentContext,
        memory: MemoryBundle | None,
        *,
        skill_catalog: list[dict[str, str]] | None = None,
        loaded_skills: list[str] | None = None,
        runbook_excerpts: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        ...

    async def synthesize(
        self,
        context: IncidentContext,
        hypotheses: list[dict[str, Any]],
        progress: list[str],
        *,
        loaded_skills: list[str] | None = None,
        runbook_excerpts: list[str] | None = None,
    ) -> dict[str, Any]:
        ...


def _extract_json_array(text: str) -> list[Any]:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def _extract_json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


class HeuristicLLM:
    """Deterministic offline fallback — not used when Llama is healthy."""

    name = "heuristic"

    async def propose_hypotheses(
        self,
        context: IncidentContext,
        memory: MemoryBundle | None,
        *,
        skill_catalog: list[dict[str, str]] | None = None,
        loaded_skills: list[str] | None = None,
        runbook_excerpts: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        _ = skill_catalog, loaded_skills, runbook_excerpts
        title = context.incident.title.lower()
        alert = context.primary_alert()
        summary = (alert.summary if alert else context.incident.title).lower()
        blob = f"{title} {summary}"
        hyps: list[dict[str, Any]] = []

        if any(k in blob for k in ("dns", "nxdomain", "resolution")):
            hyps.append(
                {
                    "statement": "DNS resolution failure for an upstream dependency",
                    "component": "dns/network",
                }
            )
        if any(k in blob for k in ("deploy", "release", "version", "rollback")) or context.change_events:
            hyps.append(
                {
                    "statement": "Recent deploy introduced a regression",
                    "component": context.incident.service.name,
                }
            )
        if any(k in blob for k in ("lag", "kafka", "consumer", "offset")):
            hyps.append(
                {
                    "statement": "Kafka consumer lag / poison message stalling the group",
                    "component": "kafka-consumer",
                }
            )
        if any(k in blob for k in ("timeout", "latency", "dependency", "5xx", "error rate")):
            hyps.append(
                {
                    "statement": "Downstream dependency timeout cascading into this service",
                    "component": "downstream",
                }
            )
        if memory:
            sig = context.fingerprint()
            for pb in memory.match_playbooks(sig)[:1]:
                hyps.insert(
                    0,
                    {
                        "statement": f"Known playbook match: {pb.title}",
                        "component": context.incident.service.name,
                        "playbook_id": pb.id,
                    },
                )
            for rec in memory.match_recollections(sig)[:1]:
                hyps.append(
                    {
                        "statement": f"Recurrence of prior pattern: {rec.title}",
                        "component": context.incident.service.name,
                        "recollection_id": rec.id,
                    }
                )
        if not hyps:
            hyps.append(
                {
                    "statement": "Application error in the alerting service",
                    "component": context.incident.service.name,
                }
            )
        if len(hyps) < 3 and context.dependency_hints:
            hyps.append(
                {
                    "statement": "Fault originates in an upstream dependency from the topology graph",
                    "component": "upstream",
                }
            )
        if len(hyps) < 3:
            hyps.append(
                {
                    "statement": "Configuration drift or missing required setting",
                    "component": "config",
                }
            )
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for h in hyps:
            if h["statement"] in seen:
                continue
            seen.add(h["statement"])
            unique.append(h)
        return unique[:4]

    async def synthesize(
        self,
        context: IncidentContext,
        hypotheses: list[dict[str, Any]],
        progress: list[str],
        *,
        loaded_skills: list[str] | None = None,
        runbook_excerpts: list[str] | None = None,
    ) -> dict[str, Any]:
        _ = loaded_skills, runbook_excerpts, progress
        ranked = sorted(hypotheses, key=lambda h: float(h.get("confidence", 0)), reverse=True)
        top = ranked[0] if ranked else None
        rejected = [
            h.get("statement", "")
            for h in ranked[1:]
            if h.get("status") in ("refuted", "skipped")
        ]
        # Operator-facing actions only — tooling/confidence stay internal
        recs: list[str] = []
        conf = float((top or {}).get("confidence") or 0)
        status = str((top or {}).get("status") or "").lower()
        if top and conf >= 0.45 and status != "refuted":
            recs.append("Confirm the primary symptom is still firing in your monitoring.")
            recs.append("Follow the matching service runbook before changing production.")
        else:
            recs.append("Gather a bit more signal before remediating — check logs around the alert time.")
        if context.change_events:
            recs.append(f"Review the recent change: {context.change_events[0].summary}")
        if context.runbook_sections:
            recs.append("Open the linked runbook and work the first mitigation steps.")

        if top and status == "supported" and conf >= 0.45:
            headline = f"Likely cause: {top.get('statement')}"
            details = (
                f"For {context.incident.service.name} — {context.incident.title}. "
                "This looks like the right place to focus handling right now."
            )
        elif top and status == "refuted":
            headline = f"That lead doesn't hold: {top.get('statement')}"
            details = (
                f"On {context.incident.service.name}, park that theory and move to the next checks "
                "from the runbook."
            )
        else:
            headline = f"Still narrowing down {context.incident.title}."
            details = "I don't have a strong enough lead yet to recommend remediation."

        summary = render_structured_reply(
            {
                "headline": headline,
                "details": details,
                "next_steps": recs,
            }
        )
        return {
            "summary": summary,
            "recommendations": recs,
            "top_hypothesis_id": top.get("id") if top else None,
            "rejected": rejected,
        }

    async def answer_question(
        self,
        question: str,
        *,
        context: IncidentContext,
        hypotheses: list[dict[str, Any]] | None = None,
        synthesis: str | None = None,
        recommendations: list[str] | None = None,
        loaded_skills: list[str] | None = None,
        runbook_excerpts: list[str] | None = None,
        chat_history: list[dict[str, str]] | None = None,
    ) -> str:
        _ = chat_history, loaded_skills, runbook_excerpts
        hyps = hypotheses or []
        top = sorted(hyps, key=lambda h: float(h.get("confidence") or 0), reverse=True)
        top_h = top[0] if top else None
        if synthesis:
            headline = synthesis.split("\n")[0][:180]
        elif top_h:
            headline = f"On {context.incident.service.name}, I'd focus on: {top_h.get('statement')}"
        else:
            headline = f"Still working {context.incident.title}."
        return render_structured_reply(
            {
                "headline": headline,
                "details": "Here's the handling view for your question — not the raw investigation internals.",
                "next_steps": [
                    s
                    for s in (recommendations or [])
                    if s and not re.search(r"tool\.|verify quote|confidence", s, re.I)
                ]
                or [
                    "Ask what to do first",
                    "Ask how urgent this is",
                ],
            }
        )

    async def answer_desk(
        self,
        question: str,
        *,
        incidents: list[dict[str, Any]],
        chat_history: list[dict[str, str]] | None = None,
        active_incident: dict[str, Any] | None = None,
    ) -> str:
        _ = chat_history, active_incident
        open_ones = [i for i in incidents if str(i.get("status") or "").lower() != "resolved"]
        q = question.lower()
        if any(k in q for k in ("how many", "count", "number of")):
            sample = ", ".join(
                str(i.get("incident_id")) for i in open_ones[:4] if i.get("incident_id")
            )
            return render_structured_reply(
                {
                    "headline": f"There are {len(open_ones)} open incidents ({len(incidents)} total).",
                    "details": f"Open right now: {sample}." if sample else "No open incidents.",
                    "next_steps": [
                        "Ask “what incidents are available?” for titles",
                        "Say investigate <id> to start triage",
                    ],
                }
            )
        if any(k in q for k in ("list", "available", "which incident", "show incident", "what incident")):
            lines = [
                f"{i.get('incident_id')}: {i.get('title')} ({i.get('status')}, {i.get('service_name')})"
                for i in incidents[:12]
            ]
            return render_structured_reply(
                {
                    "headline": f"{len(incidents)} incidents available — {len(open_ones)} open.",
                    "details": "\n".join(f"• {line}" for line in lines),
                    "next_steps": ["Say investigate <id> to start triage"],
                }
            )
        # Topic hint from catalog only — never invent related incidents
        topic_hits = [
            i
            for i in incidents
            if any(
                tok in f"{i.get('title')} {i.get('service_name')}".lower()
                for tok in re.findall(r"[a-z0-9.]{3,}", q)
            )
        ]
        if topic_hits:
            i = topic_hits[0]
            return render_structured_reply(
                {
                    "headline": (
                        f"Closest match is {i.get('incident_id')}: {i.get('title')} "
                        f"({i.get('status')}, {i.get('service_name')})."
                    ),
                    "details": "I only count incidents that actually match your topic — nothing extra invented.",
                    "next_steps": [f"Say investigate {i.get('incident_id')} to dig in"],
                }
            )
        if open_ones:
            top = open_ones[0]
            return render_structured_reply(
                {
                    "headline": f"I can help with {len(open_ones)} open incidents.",
                    "details": f"Example: {top.get('incident_id')} — {top.get('title')}.",
                    "next_steps": [
                        "Ask how many are open",
                        f"Say investigate {top.get('incident_id')}",
                    ],
                }
            )
        return "No incidents loaded yet."


class LlamaLLM:
    """Real Llama via Ollama OpenAI-compatible chat API."""

    name = "llama"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        self.model = model or os.getenv("PAIGE_LLM_MODEL", "llama3.2:latest")
        self.base_url = (
            base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
        ).rstrip("/")
        self.timeout_s = timeout_s

    async def _chat(self, system: str, user: str, temperature: float = 0.2) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.post(f"{self.base_url}/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"] or ""

    async def propose_hypotheses(
        self,
        context: IncidentContext,
        memory: MemoryBundle | None,
        *,
        skill_catalog: list[dict[str, str]] | None = None,
        loaded_skills: list[str] | None = None,
        runbook_excerpts: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        payload = {
            "incident": context.incident.model_dump(mode="json"),
            "alerts": [a.model_dump(mode="json") for a in context.alerts],
            "changes": [c.model_dump(mode="json") for c in context.change_events[:5]],
            "past_incidents": [p.model_dump(mode="json") for p in context.past_incidents[:5]],
            "dependencies": context.dependency_hints,
            "skill_catalog": skill_catalog or [],
            "loaded_skills": loaded_skills or [],
            "runbook_excerpts": runbook_excerpts or context.runbook_sections[:2],
            "playbooks": (
                [p.model_dump(mode="json") for p in memory.match_playbooks(context.fingerprint())]
                if memory
                else []
            ),
        }
        system = (
            "You are Paige, PagerDuty's SRE Agent. Propose 2-4 concrete root-cause "
            "hypotheses grounded ONLY in the provided incident context, loaded skills, "
            "and runbook excerpts. Return JSON array only: "
            '[{"statement":"...","component":"...","skill_id":"optional"}]. '
            "Prefer hypotheses that match loaded skills/runbooks. No markdown."
        )
        text = await self._chat(system, json.dumps(payload, default=str))
        raw = _extract_json_array(text)
        cleaned: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict) or not item.get("statement"):
                continue
            cleaned.append(
                {
                    "statement": str(item["statement"]),
                    "component": str(item.get("component") or context.incident.service.name),
                    "skill_id": item.get("skill_id"),
                }
            )
        if cleaned:
            return cleaned[:4]
        # Soft fallback so investigation still proceeds
        return await HeuristicLLM().propose_hypotheses(
            context,
            memory,
            skill_catalog=skill_catalog,
            loaded_skills=loaded_skills,
            runbook_excerpts=runbook_excerpts,
        )

    async def synthesize(
        self,
        context: IncidentContext,
        hypotheses: list[dict[str, Any]],
        progress: list[str],
        *,
        loaded_skills: list[str] | None = None,
        runbook_excerpts: list[str] | None = None,
    ) -> dict[str, Any]:
        ranked = sorted(hypotheses, key=lambda h: float(h.get("confidence") or 0), reverse=True)
        top = ranked[0] if ranked else None
        evidence_bits = []
        if top:
            for ev in (top.get("evidence") or [])[:3]:
                if isinstance(ev, dict) and (ev.get("quote") or ev.get("claim")):
                    evidence_bits.append(
                        {
                            "claim": ev.get("claim"),
                            "source": ev.get("source"),
                            "quote": ev.get("quote"),
                        }
                    )
        payload = {
            "incident_title": context.incident.title,
            "service": context.incident.service.name,
            "urgency": getattr(context.incident, "urgency", None),
            "top_hypothesis": {
                "statement": (top or {}).get("statement"),
                "status": (top or {}).get("status"),
                "confidence": (top or {}).get("confidence"),
                "evidence": evidence_bits,
            }
            if top
            else None,
            "other_hypotheses": [
                {
                    "statement": h.get("statement"),
                    "status": h.get("status"),
                    "confidence": h.get("confidence"),
                }
                for h in ranked[1:4]
            ],
            "loaded_skills": loaded_skills or [],
            "runbook_excerpts": (runbook_excerpts or context.runbook_sections[:2])[:3],
            "progress_tail": [
                p
                for p in progress[-20:]
                if not str(p).startswith("tool:") and "LangGraph" not in str(p)
            ],
        }
        system = (
            "You are Paige speaking to an on-call engineer in chat. "
            "They only want incident-handling guidance — what is going on and what to do next. "
            "Your internal scores/tools already ran; do NOT show that machinery. "
            "Return JSON only: "
            '{"headline":"plain-English what we think is wrong",'
            '"details":"2-3 short sentences for handling: impact, likely cause, why it matters",'
            '"recommendations":["operator action 1","operator action 2"]}. '
            "Forbidden in any field: tool names, tool.*, confidence numbers, JSON dumps, "
            "source=toy, count=, Verify quote, or raw dependency maps. "
            "Prefer runbook-style actions (confirm symptom, check recent deploy, mitigate). "
            "No markdown fences."
        )
        try:
            text = await self._chat(system, json.dumps(payload, default=str), temperature=0.2)
            data = _extract_json_object(text)
            if data.get("summary") or data.get("headline") or data.get("details"):
                details = str(data.get("details") or "")
                blob = f"{data.get('headline')} {details} {data.get('recommendations')}"
                if re.search(
                    r"\btool\b|search_logs|confidence\s*(level|of|=)|source=toy|verify quote|returned a signal",
                    blob,
                    re.I,
                ):
                    raise ValueError("tool jargon in synthesis")
                summary = render_structured_reply(
                    {
                        "headline": data.get("headline") or data.get("summary"),
                        "details": details,
                        "next_steps": data.get("recommendations") or data.get("next_steps") or [],
                    }
                )
                return {
                    "summary": summary or str(data.get("summary")),
                    "recommendations": [
                        str(x) for x in (data.get("recommendations") or data.get("next_steps") or [])
                    ][:8],
                    "top_hypothesis_id": data.get("top_hypothesis_id") or (top or {}).get("id"),
                    "rejected": data.get("rejected") or [],
                }
        except Exception:
            pass
        return await HeuristicLLM().synthesize(
            context,
            hypotheses,
            progress,
            loaded_skills=loaded_skills,
            runbook_excerpts=runbook_excerpts,
        )

    async def answer_question(
        self,
        question: str,
        *,
        context: IncidentContext,
        hypotheses: list[dict[str, Any]] | None = None,
        synthesis: str | None = None,
        recommendations: list[str] | None = None,
        loaded_skills: list[str] | None = None,
        runbook_excerpts: list[str] | None = None,
        chat_history: list[dict[str, str]] | None = None,
    ) -> str:
        history = (chat_history or [])[-8:]
        payload = {
            "question": question,
            "incident": {
                "id": context.incident.id,
                "title": context.incident.title,
                "service": context.incident.service.name,
                "urgency": getattr(context.incident, "urgency", None),
            },
            "alerts": [a.summary for a in context.alerts[:3]],
            "hypotheses": hypotheses or [],
            "synthesis": synthesis,
            "recommendations": recommendations or [],
            "loaded_skills": loaded_skills or [],
            "runbook_excerpts": (runbook_excerpts or context.runbook_sections[:2])[:3],
            "recent_chat": history,
        }
        system = (
            "You are Paige in the PagerDuty incident chat. Answer ONLY about this incident "
            "using the provided state. Speak like a calm SRE helping handle the incident. "
            "Return JSON only: "
            '{"headline":"direct answer in 1 natural sentence",'
            '"details":"2-4 sentences of handling guidance",'
            '"next_steps":["operator action1","operator action2"]}. '
            "If asked about root cause, say the likely cause in plain English — "
            "do not quote confidence scores or tool names. "
            "Forbidden: tool.*, confidence numbers, Verify quote, source=toy, raw JSON. "
            "Never invent unrelated incidents. Sound conversational. No markdown fences."
        )
        try:
            text = (await self._chat(system, json.dumps(payload, default=str), temperature=0.25)).strip()
            data = _extract_json_object(text)
            rendered = render_structured_reply(data)
            if rendered:
                return rendered
            if text and not text.strip().startswith("{"):
                return text.strip()
        except Exception:
            pass
        return await HeuristicLLM().answer_question(
            question,
            context=context,
            hypotheses=hypotheses,
            synthesis=synthesis,
            recommendations=recommendations,
            loaded_skills=loaded_skills,
            runbook_excerpts=runbook_excerpts,
            chat_history=chat_history,
        )

    async def answer_desk(
        self,
        question: str,
        *,
        incidents: list[dict[str, Any]],
        chat_history: list[dict[str, str]] | None = None,
        active_incident: dict[str, Any] | None = None,
    ) -> str:
        open_count = sum(
            1 for i in incidents if str(i.get("status") or "").lower() != "resolved"
        )
        q = question.lower()
        # Grounded facts for count/list — don't let the model invent next-step topic drift
        if any(k in q for k in ("how many", "count", "number of", "list", "available", "what incident", "which incident")):
            factual = await HeuristicLLM().answer_desk(
                question,
                incidents=incidents,
                chat_history=chat_history,
                active_incident=None,
            )
            # Light natural rewrite of headline only
            try:
                polish = await self._chat(
                    "Rewrite ONLY as JSON "
                    '{"headline":"natural one-sentence answer","details":"keep the ids/titles",'
                    '"next_steps":["Ask for the list or investigate an id","optional"]}. '
                    "Do not invent incidents. No markdown.",
                    json.dumps({"draft": factual, "question": question}, default=str),
                    temperature=0.2,
                )
                data = _extract_json_object(polish)
                rendered = render_structured_reply(data)
                if rendered and (str(open_count) in rendered or "inc" in rendered.lower()):
                    return rendered
            except Exception:
                pass
            return factual
        # Pre-filter topic matches so the model cannot invent "4 DNS incidents"
        tokens = [t for t in re.findall(r"[a-z0-9.]{3,}", q) if t not in {
            "the", "and", "for", "with", "what", "whats", "going", "about", "open",
            "how", "many", "list", "available", "incident", "incidents", "root",
            "cause", "case", "investigate", "triage",
        }]
        topic_matches = []
        for i in incidents:
            blob = f"{i.get('incident_id')} {i.get('title')} {i.get('service_name')}".lower()
            if any(t in blob for t in tokens):
                topic_matches.append(
                    {
                        "id": i.get("incident_id"),
                        "title": i.get("title"),
                        "status": i.get("status"),
                        "urgency": i.get("urgency"),
                        "service": i.get("service_name"),
                    }
                )
        payload = {
            "question": question,
            "facts": {
                "incident_count": len(incidents),
                "open_count": open_count,
                "resolved_count": len(incidents) - open_count,
            },
            "all_incidents": [
                {
                    "id": i.get("incident_id"),
                    "title": i.get("title"),
                    "status": i.get("status"),
                    "urgency": i.get("urgency"),
                    "service": i.get("service_name"),
                    "team": i.get("team"),
                }
                for i in incidents[:20]
            ],
            "topic_matches": topic_matches,
            "active_investigation": active_incident,
            "recent_chat": (chat_history or [])[-6:],
        }
        system = (
            "You are Paige, PagerDuty's SRE chat assistant. "
            "Scope: incident triage and on-call help only. "
            "If the question is outside that scope, return JSON "
            '{"headline":"Please ask within my scope — incident triage and on-call help.",'
            '"details":"I can help with open incidents, status, root cause, urgency, and next steps.",'
            '"next_steps":["How many open incidents?","Investigate checkout lag"]}. '
            "Otherwise use ONLY facts / all_incidents / topic_matches. "
            "If topic_matches is non-empty, answer about those only — never claim extra related incidents. "
            "If the user asks counts or lists, use facts.open_count and all_incidents exactly, "
            "and next_steps must be catalog follow-ups (list/investigate), never leftover DNS/checkout advice "
            "from active_investigation unless the question is about that incident. "
            "Return JSON only: "
            '{"headline":"1 natural sentence answering the question",'
            '"details":"short natural explanation naming real incident ids/titles",'
            '"next_steps":["helpful follow-up","optional second"]}. '
            "Be conversational, not templated. No markdown fences. Do not invent data."
        )
        try:
            text = (await self._chat(system, json.dumps(payload, default=str), temperature=0.25)).strip()
            data = _extract_json_object(text)
            rendered = render_structured_reply(data)
            if rendered:
                return rendered
        except Exception:
            pass
        return await HeuristicLLM().answer_desk(
            question,
            incidents=incidents,
            chat_history=chat_history,
            active_incident=active_incident,
        )


class OpenAILLM:
    """Optional cloud OpenAI-compatible backend."""

    name = "openai"

    def __init__(self, model: str | None = None) -> None:
        from openai import AsyncOpenAI  # type: ignore

        self.client = AsyncOpenAI()
        self.model = model or os.getenv("PAIGE_LLM_MODEL", "gpt-4o-mini")
        self._llama_style = LlamaLLM.__new__(LlamaLLM)
        # reuse prompt helpers via composition
        self._fallback = HeuristicLLM()

    async def answer_question(self, question: str, **kwargs: Any) -> str:
        return await self._fallback.answer_question(question, **kwargs)

    async def answer_desk(self, question: str, **kwargs: Any) -> str:
        return await self._fallback.answer_desk(question, **kwargs)

    async def propose_hypotheses(
        self,
        context: IncidentContext,
        memory: MemoryBundle | None,
        *,
        skill_catalog: list[dict[str, str]] | None = None,
        loaded_skills: list[str] | None = None,
        runbook_excerpts: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        # Delegate shape to LlamaLLM prompts via a thin chat call
        llama = LlamaLLM(model=self.model, base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
        # Use OpenAI SDK directly
        payload = {
            "incident": context.incident.model_dump(mode="json"),
            "alerts": [a.model_dump(mode="json") for a in context.alerts],
            "skill_catalog": skill_catalog or [],
            "loaded_skills": loaded_skills or [],
            "runbook_excerpts": runbook_excerpts or [],
        }
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are Paige. Return JSON array of hypotheses "
                        '[{"statement","component"}] only.'
                    ),
                },
                {"role": "user", "content": json.dumps(payload, default=str)},
            ],
            temperature=0.2,
        )
        text = resp.choices[0].message.content or "[]"
        raw = _extract_json_array(text)
        cleaned = [
            {
                "statement": str(i["statement"]),
                "component": str(i.get("component") or context.incident.service.name),
            }
            for i in raw
            if isinstance(i, dict) and i.get("statement")
        ]
        return cleaned or await self._fallback.propose_hypotheses(
            context, memory, skill_catalog=skill_catalog, loaded_skills=loaded_skills,
            runbook_excerpts=runbook_excerpts,
        )

    async def synthesize(
        self,
        context: IncidentContext,
        hypotheses: list[dict[str, Any]],
        progress: list[str],
        *,
        loaded_skills: list[str] | None = None,
        runbook_excerpts: list[str] | None = None,
    ) -> dict[str, Any]:
        return await LlamaLLM().synthesize(
            context,
            hypotheses,
            progress,
            loaded_skills=loaded_skills,
            runbook_excerpts=runbook_excerpts,
        )


async def llama_available() -> bool:
    base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1").rstrip("/")
    root = base.replace("/v1", "")
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{root}/api/tags")
            return r.status_code == 200
    except Exception:
        return False


def get_llm() -> LLM:
    """Prefer local Llama (Ollama). Override with PAIGE_LLM=heuristic|openai|llama."""
    choice = os.getenv("PAIGE_LLM", "llama").strip().lower()
    if choice == "heuristic":
        return HeuristicLLM()
    if choice == "openai":
        if os.getenv("OPENAI_API_KEY"):
            try:
                return OpenAILLM()
            except Exception:
                pass
        return HeuristicLLM()
    # default llama
    try:
        return LlamaLLM()
    except Exception:
        return HeuristicLLM()
