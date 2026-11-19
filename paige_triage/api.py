"""HTTP + chatbot UI for production-style Paige (LangGraph + Llama sub-agents)."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, AsyncIterator
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from paige_triage import data_sources
from paige_triage.engine.session import DeepInvestigationSession
from paige_triage.llm import get_llm
from paige_triage.runtime import build_deep_session

app = FastAPI(
    title="Paige Triage Agent",
    version="2.0.0",
    description="Production-style Paige clone: LangGraph, LLM sub-agents, sqlite checkpoints, chat.",
)

_sessions: dict[str, DeepInvestigationSession] = {}
_desks: dict[str, "DeskSession"] = {}
_locks = asyncio.Lock()

DEMO_INCIDENTS = [
    {"id": "inc001", "title": "Consumer lag elevated on checkout-api", "service": "checkout-api"},
    {"id": "inc002", "title": "DNS NXDOMAIN for payments-fraud.internal", "service": "fraud-gateway"},
    {"id": "inc003", "title": "Checkout API 5xx spike after v2.15.0 deploy", "service": "checkout-api"},
]

OPENER = (
    "Hi — I'm Paige, your SRE triage assistant. Ask about incidents: how many are open, "
    "what's available, status on a service, or tell me to investigate something. "
    "Please keep questions within that scope."
)

SCOPE_REDIRECT = (
    "Please ask within my scope — incident triage and on-call help "
    "(open incidents, service status, root cause, urgency, what to do next). "
    "I can't help with topics outside that."
)

# Optimized from PagerDuty Paige / AI docs — short, high-signal on-call prompts
DEFAULT_NUDGES = [
    "What are some likely root causes?",
    "What changed in the last 24 hours?",
    "Has this happened before?",
    "How urgent is this?",
    "What should I do first?",
    "Are there open incidents for checkout-api?",
]

# After triage starts — match Paige incident-tab example questions
TRIAGE_NUDGES = [
    "What is the probable cause?",
    "What changed recently?",
    "Analyze past incidents",
    "How urgent is this based on the data?",
    "What steps should I take next?",
    "What is the customer impact?",
]

_SCOPE_HINTS = (
    "incident",
    "alert",
    "outage",
    "sre",
    "on-call",
    "oncall",
    "pager",
    "pagerduty",
    "triage",
    "service",
    "deploy",
    "rollback",
    "runbook",
    "playbook",
    "log",
    "metric",
    "latency",
    "error",
    "timeout",
    "5xx",
    "4xx",
    "dns",
    "kafka",
    "checkout",
    "fraud",
    "orders",
    "payment",
    "urgency",
    "severity",
    "sev",
    "root cause",
    "root case",
    "rca",
    "investigate",
    "hypothesis",
    "lag",
    "spike",
    "nxdomain",
    "consumer",
    "noisy",
    "page the",
    "mitigat",
    "remediat",
    "open incident",
    "available",
    "status",
    "what should i do",
    "next step",
    "how urgent",
    "how many",
    "probable cause",
    "customer impact",
    "happened before",
    "past incident",
    "related service",
    "what changed",
    "catch me up",
    "status update",
)


class StartRequest(BaseModel):
    incident_id: str = "inc001"
    steer: str | None = None
    deep_mode: bool = True
    max_cycles: int = 3
    replan_interval_s: float = 0.0


class ChatMessage(BaseModel):
    text: str = Field(min_length=1)


def _incident_catalog() -> list[dict[str, Any]]:
    rows = data_sources.load_incidents()
    if rows:
        return rows
    return [
        {
            "incident_id": i["id"],
            "title": i["title"],
            "status": "triggered",
            "service_name": i["service"],
        }
        for i in DEMO_INCIDENTS
    ]


def _match_incident(text: str, catalog: list[dict[str, Any]]) -> str | None:
    lower = text.lower()
    # explicit ids + common typos: in001, inc001, inc-001, incident 001
    for row in catalog:
        iid = str(row.get("incident_id") or "")
        if not iid:
            continue
        if iid.lower() in lower:
            return iid
        digits = re.sub(r"\D", "", iid)
        if digits and re.search(rf"\bin?c?[-_]?0*{digits}\b", lower):
            return iid
        if digits and re.search(rf"\bincident\s*#?\s*0*{digits}\b", lower):
            return iid
    # title / service fuzzy
    for row in catalog:
        title = str(row.get("title") or "").lower()
        svc = str(row.get("service_name") or "").lower()
        tokens = [t for t in re.split(r"[^a-z0-9]+", title) if len(t) > 3]
        hits = sum(1 for t in tokens if t in lower)
        if svc and svc in lower:
            return str(row.get("incident_id"))
        if hits >= 2:
            return str(row.get("incident_id"))
    # keyword shortcuts for demo
    if any(k in lower for k in ("checkout lag", "consumer lag", "kafka")):
        return "inc001"
    if any(k in lower for k in ("dns", "nxdomain", "fraud")):
        return "inc002"
    if any(k in lower for k in ("5xx", "pricing", "v2.15")):
        return "inc003"
    return None


def _wants_investigate(text: str) -> bool:
    lower = text.lower()
    triggers = (
        "investigate",
        "triage",
        "look into",
        "look at",
        "dig into",
        "root cause",
        "root case",  # common typo
        "rootcause",
        "rca",
        "cause of",
        "start triage",
        "check on",
        "what's wrong with",
        "what is wrong with",
        "whats wrong",
        "debug",
        "what's going on",
        "whats going on",
        "what is going on",
        "going on with",
        "happening with",
        "status of",
    )
    return any(t in lower for t in triggers)


def _is_bare_incident_ref(text: str) -> bool:
    return bool(re.fullmatch(r"\s*in?c?[-_]?\d+\s*", text or "", re.I))


def _is_smalltalk(text: str) -> bool:
    """Greetings / identity / thanks — not incident work."""
    lower = (text or "").strip().lower()
    if not lower or len(lower) > 120:
        return False
    # If they clearly named an incident, it's not smalltalk
    if re.search(r"\bin?c?[-_]?\d+\b", lower) or "investigate" in lower or "triage" in lower:
        return False
    patterns = (
        r"^(hi|hello|hey|yo|sup|hola)\b",
        r"\bhow are you\b",
        r"\bhow('?s| is) it going\b",
        r"\bwhat('?s| is) up\b",
        r"\bwho are you\b",
        r"\bwhat are you\b",
        r"\bwhat kind of bot\b",
        r"\bare you a bot\b",
        r"\bwhat can you (do|help)\b",
        r"\bhelp me\b$",
        r"^help\??$",
        r"^(thanks|thank you|thx|ty)\b",
        r"^(good (morning|afternoon|evening)|gm)\b",
        r"^(bye|goodbye|see you)\b",
        r"^(ok|okay|cool|nice|great)\b!?$",
    )
    return any(re.search(p, lower) for p in patterns)


def _is_in_scope(text: str, catalog: list[dict[str, Any]]) -> bool:
    """True for smalltalk, incident ops, or anything matching the live catalog."""
    if _is_smalltalk(text):
        return True
    if _match_incident(text, catalog):
        return True
    lower = (text or "").lower()
    if any(h in lower for h in _SCOPE_HINTS):
        return True
    # bare ops shorthand
    if re.search(r"\b(sre|sev[0-3]|p[0-3]|mttr|sla)\b", lower):
        return True
    return False


async def _smalltalk_reply(text: str, llm: Any) -> str:
    lower = (text or "").lower()
    if re.search(r"who are you|what are you|what kind of bot|are you a bot", lower):
        base = (
            "I'm Paige — an SRE chat assistant for incident triage "
            "(PagerDuty-style). I can list open incidents, answer status questions, "
            "or investigate something when you ask. Please keep questions in that scope."
        )
    elif re.search(r"how are you|how('?s| is) it going|what('?s| is) up", lower):
        base = (
            "I'm good, thanks for asking! Ready when you are — "
            "ask about open incidents or tell me what to investigate."
        )
    elif re.search(r"what can you (do|help)|^help\??$|help me", lower):
        base = (
            "I stay in scope for on-call triage: open incidents, status, urgency, "
            "and investigation (e.g. “investigate checkout lag” or “inc001”). "
            "Please ask within that scope."
        )
    elif re.search(r"^(thanks|thank you|thx|ty)\b", lower):
        base = "Anytime — ping me if another incident pops up."
    elif re.search(r"^(bye|goodbye|see you)\b", lower):
        base = "Bye! I'll be here when the next alert lands."
    elif re.search(r"^(hi|hello|hey|yo|sup|hola|good )", lower):
        base = (
            "Hi! I'm Paige. Ask me about your incidents — lists, status, or triage. "
            "Please keep questions within that scope."
        )
    else:
        base = (
            "Hey — I'm here for incident triage. "
            "Want an incident list, or should we dig into something in scope?"
        )

    # Optional light natural rewrite; never block the chat on it
    try:
        if hasattr(llm, "_chat"):
            raw = await asyncio.wait_for(
                llm._chat(  # type: ignore[attr-defined]
                    "You are Paige, a friendly SRE assistant. "
                    "Rewrite the draft as 1-2 warm natural sentences. "
                    "Keep the meaning. No JSON. No bullet lists.",
                    f"User said: {text}\nDraft: {base}",
                    temperature=0.4,
                ),
                timeout=1.5,
            )
            cleaned = (raw or "").strip().strip('"')
            if cleaned and len(cleaned) < 400 and "{" not in cleaned:
                return cleaned
    except Exception:
        pass
    return base


class DeskSession:
    """Open chat desk — talk first; optionally attach a deep investigation."""

    def __init__(self) -> None:
        self.id = f"desk-{uuid4().hex[:16]}"
        self.chat: list[dict[str, str]] = [{"role": "assistant", "content": OPENER}]
        self.llm = get_llm()
        self.investigation: DeepInvestigationSession | None = None
        self.incident_id: str | None = None

    def snapshot(self) -> dict[str, Any]:
        inv = None
        if self.investigation:
            inv = {
                "task_id": self.investigation.task_id,
                "incident_id": self.incident_id,
                "completed": bool(self.investigation._values.get("completed")),
            }
        return {
            "desk_id": self.id,
            "chat": list(self.chat),
            "incident_id": self.incident_id,
            "investigation": inv,
            "stream_url": (
                f"/investigations/{self.investigation.task_id}/stream"
                if self.investigation
                else None
            ),
        }

    async def start_investigation(self, incident_id: str, *, max_cycles: int = 2) -> str:
        session, bundle = build_deep_session(
            incident_id=incident_id,
            deep_mode=True,
            max_cycles=max_cycles,
        )
        await session.start(bundle["incident"], bundle["alerts"])
        async with _locks:
            _sessions[session.task_id] = session
        self.investigation = session
        self.incident_id = incident_id
        inc = bundle["incident"]
        alert = bundle["alerts"][0] if bundle["alerts"] else None
        from paige_triage.llm import render_structured_reply

        # Dedicated kickoff — only this incident (avoid cross-talk from prior chat)
        msg = None
        try:
            payload = {
                "question": f"Acknowledge starting triage on {incident_id} only.",
                "facts": {"incident_count": 1, "open_count": 1, "resolved_count": 0},
                "all_incidents": [
                    {
                        "id": incident_id,
                        "title": inc.title,
                        "status": "triggered",
                        "service": inc.service.name,
                    }
                ],
                "topic_matches": [
                    {
                        "id": incident_id,
                        "title": inc.title,
                        "status": "triggered",
                        "service": inc.service.name,
                        "alert": alert.summary if alert else None,
                    }
                ],
                "active_investigation": None,
                "recent_chat": [],
            }
            if hasattr(self.llm, "_chat"):
                system = (
                    "You are Paige. The engineer just asked to triage ONE incident. "
                    "Speak naturally like the PagerDuty Paige panel — e.g. "
                    "'I'm looking at inc002 — DNS NXDOMAIN on fraud-gateway.' "
                    "Return JSON only: "
                    '{"headline":"warm natural ack naming incident id + issue",'
                    '"details":"1-2 sentences with service and alert",'
                    '"next_steps":["I will dig in with skills and tools",'
                    '"Ask about root cause, urgency, or next steps anytime"]}. '
                    "Never say 'acknowledge starting' or echo instructions. "
                    "Do not mention any other incident. No markdown fences."
                )
                raw = await self.llm._chat(  # type: ignore[attr-defined]
                    system, json.dumps(payload, default=str), temperature=0.2
                )
                from paige_triage.llm import _extract_json_object

                msg = render_structured_reply(_extract_json_object(raw))
        except Exception:
            msg = None
        if not msg:
            msg = render_structured_reply(
                {
                    "headline": f"On it — starting triage for {incident_id}.",
                    "details": (
                        f"{inc.title} on {inc.service.name}. "
                        f"Alert: {alert.summary if alert else 'n/a'}."
                    ),
                    "next_steps": [
                        "I'll load skills/runbooks and test hypotheses",
                        "Ask about root cause, urgency, or next steps anytime",
                    ],
                }
            )
        self.chat.append({"role": "assistant", "content": msg})
        return msg

    async def handle(self, text: str) -> dict[str, Any]:
        text = (text or "").strip()
        self.chat.append({"role": "user", "content": text})
        catalog = _incident_catalog()
        matched = _match_incident(text, catalog)
        lower = text.lower()
        meta = any(
            k in lower
            for k in (
                "how many",
                "count",
                "list",
                "available",
                "all incident",
                "other incident",
                "which incident",
                "what incident",
            )
        )
        bare_id = _is_bare_incident_ref(text)

        # Casual chat first — never force triage / heavy investigation for "hi"
        if _is_smalltalk(text):
            reply = await _smalltalk_reply(text, self.llm)
            self.chat.append({"role": "assistant", "content": reply})
            return {
                **self.snapshot(),
                "reply": reply,
                "started_investigation": False,
                "nudges": DEFAULT_NUDGES,
            }

        # Politely refuse off-topic asks (weather, recipes, homework, etc.)
        if not _is_in_scope(text, catalog):
            reply = SCOPE_REDIRECT
            self.chat.append({"role": "assistant", "content": reply})
            return {
                **self.snapshot(),
                "reply": reply,
                "started_investigation": False,
                "nudges": DEFAULT_NUDGES,
            }

        triage_intent = (not meta) and matched and (
            _wants_investigate(text) or bare_id
        )

        # Start / switch triage when the user clearly points at an incident
        if triage_intent:
            same = self.investigation is not None and self.incident_id == matched
            if same:
                try:
                    report = await self.investigation.chat(text)
                    reply = str(report.get("reply") or "").strip()
                    if reply:
                        self.chat.append({"role": "assistant", "content": reply})
                        return {
                            **self.snapshot(),
                            "reply": reply,
                            "started_investigation": False,
                            "nudges": TRIAGE_NUDGES,
                        }
                except Exception:
                    pass
            else:
                note = await self.start_investigation(matched)
                return {
                    **self.snapshot(),
                    "reply": note,
                    "started_investigation": True,
                    "nudges": TRIAGE_NUDGES,
                }

        # Typed a different incident id while another triage is active → switch
        if (
            matched
            and self.incident_id
            and matched != self.incident_id
            and not meta
            and (bare_id or re.search(r"\bin?c?[-_]?\d+\b", lower))
        ):
            note = await self.start_investigation(matched)
            return {
                **self.snapshot(),
                "reply": note,
                "started_investigation": True,
                "nudges": TRIAGE_NUDGES,
            }

        # Follow-ups on the active investigation (not catalog meta questions)
        if self.investigation and not meta:
            try:
                report = await self.investigation.chat(text)
                reply = str(report.get("reply") or "").strip()
                if reply:
                    self.chat.append({"role": "assistant", "content": reply})
                    return {
                        **self.snapshot(),
                        "reply": reply,
                        "started_investigation": False,
                        "nudges": TRIAGE_NUDGES,
                    }
            except Exception:
                pass

        # Catalog / meta answers must not be biased by the active triage topic
        active = None
        if self.investigation and self.incident_id and not meta:
            snap = self.investigation.snapshot()
            active = {
                "incident_id": self.incident_id,
                "synthesis": snap.get("synthesis"),
                "top_hypothesis": snap.get("top_hypothesis"),
                "completed": snap.get("completed"),
            }
        reply = await self.llm.answer_desk(
            text,
            incidents=catalog,
            chat_history=[] if meta else self.chat,
            active_incident=active,
        )
        reply = (reply or "").strip() or (
            "Ask me about open incidents, or say investigate <id> to start triage."
        )
        self.chat.append({"role": "assistant", "content": reply})
        return {
            **self.snapshot(),
            "reply": reply,
            "started_investigation": False,
            "nudges": DEFAULT_NUDGES,
        }


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "engine": "langgraph",
        "data_mode": data_sources.data_mode(),
        "demo_incidents": [i["id"] for i in DEMO_INCIDENTS],
    }


@app.get("/incidents")
async def list_incidents() -> dict[str, Any]:
    return {"demo": DEMO_INCIDENTS, "incidents": data_sources.load_incidents()}


@app.get("/sources")
async def sources() -> dict[str, Any]:
    return {
        "data_mode": data_sources.data_mode(),
        "sources": data_sources.grounded_source_catalog(),
    }


@app.post("/desk")
async def open_desk() -> dict[str, Any]:
    desk = DeskSession()
    async with _locks:
        _desks[desk.id] = desk
    catalog = _incident_catalog()
    open_n = sum(1 for i in catalog if str(i.get("status") or "").lower() != "resolved")
    return {
        "desk_id": desk.id,
        "opener": OPENER,
        "chat_url": f"/desk/{desk.id}/chat",
        "incident_count": len(catalog),
        "open_count": open_n,
        "nudges": DEFAULT_NUDGES,
    }


@app.post("/desk/{desk_id}/chat")
async def desk_chat(desk_id: str, body: ChatMessage) -> dict[str, Any]:
    desk = _desks.get(desk_id)
    if not desk:
        raise HTTPException(404, "unknown desk_id")
    return await desk.handle(body.text)


@app.get("/desk/{desk_id}")
async def get_desk(desk_id: str) -> dict[str, Any]:
    desk = _desks.get(desk_id)
    if not desk:
        raise HTTPException(404, "unknown desk_id")
    return desk.snapshot()


@app.post("/investigations")
async def start_investigation(body: StartRequest | None = None) -> dict[str, Any]:
    body = body or StartRequest()
    session, bundle = build_deep_session(
        incident_id=body.incident_id,
        deep_mode=body.deep_mode,
        max_cycles=body.max_cycles,
        replan_interval_s=body.replan_interval_s,
    )
    await session.start(bundle["incident"], bundle["alerts"])
    inc = bundle["incident"]
    alert = bundle["alerts"][0] if bundle["alerts"] else None
    opener = (
        f"I'm looking at **{inc.title}** on `{inc.service.name}`.\n\n"
        f"Alert: {alert.summary if alert else 'n/a'}\n\n"
        "I'll load skills and runbooks, investigate likely causes, and suggest next steps. "
        "Ask me anything, or tap a suggestion below."
    )
    session._chat.append({"role": "assistant", "content": opener})
    if body.steer:
        await asyncio.sleep(0.2)
        await session.chat(body.steer)
    async with _locks:
        _sessions[session.task_id] = session
    snap = session.snapshot()
    return {
        "task_id": session.task_id,
        "engine": "langgraph",
        "llm": snap.get("llm"),
        "incident": bundle["incident"].model_dump(mode="json"),
        "status": "running",
        "opener": opener,
        "stream_url": f"/investigations/{session.task_id}/stream",
        "chat_url": f"/investigations/{session.task_id}/chat",
        "nudges": TRIAGE_NUDGES,
    }


@app.get("/investigations/{task_id}")
async def get_investigation(task_id: str) -> dict[str, Any]:
    session = _sessions.get(task_id)
    if not session:
        raise HTTPException(404, "unknown task_id")
    return session.snapshot()


@app.post("/investigations/{task_id}/chat")
async def chat(task_id: str, body: ChatMessage) -> dict[str, Any]:
    session = _sessions.get(task_id)
    if not session:
        raise HTTPException(404, "unknown task_id")
    return await session.chat(body.text)


@app.post("/investigations/{task_id}/messages")
async def messages(task_id: str, body: ChatMessage) -> dict[str, Any]:
    return await chat(task_id, body)


@app.post("/investigations/{task_id}/cancel")
async def cancel(task_id: str) -> dict[str, Any]:
    session = _sessions.get(task_id)
    if not session:
        raise HTTPException(404, "unknown task_id")
    return await session.cancel()


@app.post("/investigations/{task_id}/wait")
async def wait(task_id: str, timeout: float = 900.0) -> dict[str, Any]:
    session = _sessions.get(task_id)
    if not session:
        raise HTTPException(404, "unknown task_id")
    return await session.wait_until_complete(timeout=timeout)


@app.get("/investigations/{task_id}/stream")
async def stream(task_id: str) -> StreamingResponse:
    session = _sessions.get(task_id)
    if not session:
        raise HTTPException(404, "unknown task_id")

    async def gen() -> AsyncIterator[str]:
        last_prog = 0
        last_chat = 0
        while True:
            snap = session.snapshot()
            prog = snap.get("progress") or []
            while last_prog < len(prog):
                yield f"data: {json.dumps({'type': 'progress', 'message': prog[last_prog]})}\n\n"
                last_prog += 1
            chat_log = snap.get("chat") or []
            while last_chat < len(chat_log):
                yield f"data: {json.dumps({'type': 'chat', 'message': chat_log[last_chat]})}\n\n"
                last_chat += 1
            if snap.get("completed") or snap.get("canceled"):
                yield f"data: {json.dumps({'type': 'final', 'report': snap})}\n\n"
                break
            await asyncio.sleep(0.35)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/", response_class=HTMLResponse)
@app.get("/demo", response_class=HTMLResponse)
@app.get("/chat", response_class=HTMLResponse)
async def chatbot_page() -> str:
    """PagerDuty-style Paige chat: type immediately, no incident picker."""
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Paige</title>
  <style>
    :root {
      --pd-green:#06AC38; --pd-green-dark:#048A2C; --bg:#F6F7F9; --panel:#fff;
      --ink:#1B2B34; --muted:#5B6B75; --line:#E3E7EB; --user:#E8F5E9;
      --bot:#F3F5F7; --shadow:0 8px 28px rgba(20,40,60,.08);
    }
    * { box-sizing:border-box; }
    body {
      margin:0; min-height:100vh; display:flex; align-items:stretch; justify-content:center;
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; color:var(--ink);
      background: linear-gradient(180deg, #EEF1F4 0%, var(--bg) 40%);
    }
    .shell {
      width:min(440px, 100%); margin:0 auto; background:var(--panel);
      display:flex; flex-direction:column; min-height:100vh; box-shadow:var(--shadow);
    }
    @media (min-width:900px) {
      body { padding:24px; align-items:center; }
      .shell { min-height:calc(100vh - 48px); max-height:calc(100vh - 48px); border-radius:16px; border:1px solid var(--line); }
    }
    .top {
      padding:14px 16px; border-bottom:1px solid var(--line); display:flex; gap:12px; align-items:center;
    }
    .avatar {
      width:36px; height:36px; border-radius:50%; background:var(--pd-green); color:#fff;
      display:flex; align-items:center; justify-content:center; font-weight:700; font-size:14px;
    }
    .top h1 { margin:0; font-size:16px; font-weight:700; }
    .top .meta { color:var(--muted); font-size:12px; margin-top:2px; }
    #thread {
      flex:1; overflow:auto; padding:16px; display:flex; flex-direction:column; gap:12px; background:#fff;
    }
    .row { display:flex; gap:10px; align-items:flex-start; max-width:100%; }
    .row.me { flex-direction:row-reverse; }
    .bubble {
      max-width:85%; padding:10px 12px; border-radius:12px; font-size:14px; line-height:1.45;
      white-space:pre-wrap; word-break:break-word;
    }
    .row.bot .bubble { background:var(--bot); border:1px solid var(--line); }
    .row.me .bubble { background:var(--user); border:1px solid #CDEAD3; }
    .who { font-size:11px; color:var(--muted); margin-bottom:4px; font-weight:600; }
    .typing { color:var(--muted); font-size:13px; padding:0 4px; }
    .nudges {
      display:flex; flex-wrap:wrap; gap:8px; padding:0 16px 10px; background:#fff;
    }
    .nudge {
      border:1px solid var(--line); background:#fff; border-radius:999px; padding:7px 12px;
      font-size:12px; color:var(--pd-green-dark); cursor:pointer;
    }
    .nudge:hover { border-color:var(--pd-green); background:#F7FFF9; }
    .composer {
      border-top:1px solid var(--line); padding:12px 16px; display:flex; gap:8px; background:#fff;
    }
    #prompt {
      flex:1; border:1px solid var(--line); border-radius:20px; padding:10px 14px; font:inherit;
      resize:none; min-height:42px; max-height:120px; outline:none;
    }
    #prompt:focus { border-color:var(--pd-green); }
    #send {
      border:none; border-radius:20px; padding:0 16px; background:var(--pd-green); color:#fff;
      font-weight:700; cursor:pointer; min-width:72px;
    }
    #send:disabled { opacity:.45; cursor:not-allowed; }
  </style>
</head>
<body>
  <div class="shell">
    <div class="top">
      <div class="avatar">P</div>
      <div>
        <h1>Paige</h1>
        <div class="meta" id="meta">Ask anything about your incidents</div>
      </div>
    </div>
    <div id="thread"></div>
    <div class="nudges" id="nudges"></div>
    <div class="composer">
      <textarea id="prompt" rows="1" placeholder="Message Paige…"></textarea>
      <button id="send">Send</button>
    </div>
  </div>
<script>
let deskId = null;
let es = null;
let seen = new Set();

function esc(s) {
  return String(s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function mdLite(s) {
  return esc(s).replace(/\\*\\*(.+?)\\*\\*/g,'<b>$1</b>').replace(/`([^`]+)`/g,'<code>$1</code>');
}
function addMsg(role, text, key) {
  if (!text) return;
  const k = key || (role + ':' + text);
  if (seen.has(k)) return;
  seen.add(k);
  const thread = document.getElementById('thread');
  const row = document.createElement('div');
  row.className = 'row ' + (role === 'user' ? 'me' : 'bot');
  const who = role === 'user' ? 'You' : 'Paige';
  row.innerHTML = `<div><div class="who">${who}</div><div class="bubble">${mdLite(text)}</div></div>`;
  thread.appendChild(row);
  thread.scrollTop = thread.scrollHeight;
}
function setTyping(on) {
  let el = document.getElementById('typing');
  if (!on) { if (el) el.remove(); return; }
  if (el) return;
  el = document.createElement('div');
  el.id = 'typing'; el.className = 'typing'; el.textContent = 'Paige is typing…';
  document.getElementById('thread').appendChild(el);
  document.getElementById('thread').scrollTop = document.getElementById('thread').scrollHeight;
}
function friendlyProgress(msg) {
  const m = String(msg || '');
  if (/Loaded skill/i.test(m)) return 'Loading the matching skill and runbook…';
  if (/Planned .*hypothes/i.test(m)) return 'Forming root-cause hypotheses…';
  if (/Spawned LLM sub-agent/i.test(m)) return 'Running an investigator sub-agent with tools…';
  if (/tool:search_logs/i.test(m)) return 'Searching logs…';
  if (/tool:query_metrics/i.test(m)) return 'Checking metrics…';
  if (/tool:fetch_runbook/i.test(m)) return 'Reading runbook…';
  if (/Hypothesis result/i.test(m)) return null; // final card covers findings
  if (/Synthesiz|Synthesis complete/i.test(m)) return 'Writing up what I found…';
  if (/^tool:/i.test(m)) return null;
  if (/LLM sub-agent investigating/i.test(m)) return 'Investigating a hypothesis…';
  if (/LLM sub-agent done/i.test(m)) return null;
  if (/LangGraph|Spawn requested|Active LLM|Resumed on|bootstrap|skill catalog|Runbook excerpts/i.test(m)) return null;
  return null; // keep progress quiet; final reply is structured
}
function showNudges(list) {
  const box = document.getElementById('nudges');
  box.innerHTML = '';
  (list || []).forEach(t => {
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'nudge'; b.textContent = t;
    b.onclick = () => sendText(t);
    box.appendChild(b);
  });
}
function attachStream(url) {
  if (es) { es.close(); es = null; }
  if (!url) return;
  es = new EventSource(url);
  es.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === 'progress') {
      const f = friendlyProgress(msg.message);
      if (f) addMsg('bot', f, 'p:'+f);
    }
    if (msg.type === 'chat' && msg.message && msg.message.role !== 'user' && msg.message.content) {
      // Skip raw intermediate chat noise; wait for structured final
      const c = msg.message.content;
      if (/^tool:|^LLM sub-agent|^Hypothesis result|^Starting triage/i.test(c)) return;
    }
    if (msg.type === 'final') {
      setTyping(false);
      const r = msg.report || {};
      let card = r.synthesis || '';
      const steps = r.steps || r.recommendations || [];
      if (steps.length && !/Next:/i.test(card)) {
        card = (card ? card + '\\n\\n' : '') + 'Next:\\n' + steps.map(s => '• ' + s).join('\\n');
      }
      if (card) addMsg('bot', card, 'syn');
      showNudges([
        'What is the probable cause?',
        'What steps should I take next?',
        'How urgent is this based on the data?',
        'Has this happened before?'
      ]);
      es.close();
    }
  };
}

async function boot() {
  setTyping(true);
  const desk = await fetch('/desk', { method:'POST' }).then(r => r.json());
  deskId = desk.desk_id;
  setTyping(false);
  addMsg('bot', desk.opener, 'opener');
  document.getElementById('meta').textContent = 'Ask anything — counts, status, triage…';
  showNudges(desk.nudges || []);
  document.getElementById('prompt').focus();
}

async function sendText(text) {
  if (!deskId || !text) return;
  addMsg('user', text);
  setTyping(true);
  document.getElementById('send').disabled = true;
  try {
    const res = await fetch(`/desk/${deskId}/chat`, {
      method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify({text})
    });
    const report = await res.json();
    if (!res.ok) {
      addMsg('bot', report.detail || 'Sorry — I could not answer that just now.');
      return;
    }
    if (report.reply) addMsg('bot', report.reply, 'a:' + Date.now() + ':' + String(report.reply).slice(0,40));
    if (report.started_investigation && report.stream_url) {
      document.getElementById('meta').textContent = (report.incident_id || 'incident') + ' · investigating';
      setTyping(true);
      attachStream(report.stream_url);
    }
    if (report.nudges) showNudges(report.nudges);
  } catch (err) {
    addMsg('bot', 'Connection issue talking to Paige.');
  } finally {
    if (!es) setTyping(false);
    document.getElementById('send').disabled = false;
    document.getElementById('prompt').focus();
  }
}

document.getElementById('send').onclick = () => {
  const box = document.getElementById('prompt');
  const t = box.value.trim();
  box.value = '';
  sendText(t);
};
document.getElementById('prompt').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    document.getElementById('send').click();
  }
});
boot();
</script>
</body>
</html>"""
