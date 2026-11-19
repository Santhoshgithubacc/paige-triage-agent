"""LangGraph supervisor with interrupt/resume (Paige reactive loop).

Nodes:
  bootstrap → load_context → plan → spawn → wait_event(interrupt)
           → route → handle_user|handle_result|replan|cancel
           → synthesize | wait_event | plan

Resume with Command(resume=event_dict).
"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from paige_triage.engine.state import InvestigationState
from paige_triage.llm import get_llm
from paige_triage.models.incident import IncidentContext
from paige_triage.models.memory import MemoryBundle
from paige_triage.skills.loader import SkillRegistry


def _progress(msg: str) -> dict[str, Any]:
    return {"progress": [msg]}


def build_supervisor_graph_uncompiled(
    *,
    skills: SkillRegistry,
    memory: MemoryBundle | None = None,
):
    llm = get_llm()

    async def bootstrap(state: InvestigationState) -> dict[str, Any]:
        return {
            **_progress("LangGraph supervisor bootstrap"),
            "llm_name": getattr(llm, "name", "llm"),
            "completed": False,
            "canceled": False,
            "waiting": False,
            "cycle": state.get("cycle") or 0,
            "hypotheses": state.get("hypotheses") or [],
            "pending_spawn": state.get("pending_spawn") or [],
            "active_sub_tasks": state.get("active_sub_tasks") or {},
            "chat": [],
            "extra": {"engine": "langgraph"},
        }

    async def load_skills_runbooks(state: InvestigationState) -> dict[str, Any]:
        ctx = state.get("context") or {}
        title = (ctx.get("incident") or {}).get("title") or state.get("incident", {}).get("title", "")
        alerts = ctx.get("alerts") or []
        summary = alerts[0].get("summary", "") if alerts else ""
        blob = f"{title} {summary}"
        catalog = skills.catalog()
        msgs = [f"skill catalog size={len(catalog)}"]
        loaded_ids: list[str] = []
        loaded_texts: list[str] = []
        for skill in skills.select_candidates(blob, limit=3):
            full = skills.load(skill.id)
            if full:
                loaded_ids.append(full.id)
                loaded_texts.append(f"[{full.name}]\n{full.instructions}")
                msgs.append(f"Loaded skill: {full.name}")
        runbooks = list(ctx.get("runbook_sections") or [])[:3]
        # Also pull alert runbook url into excerpts
        for a in alerts:
            url = (a.get("custom_details") or {}).get("runbook_url")
            if url:
                runbooks.append(f"Linked runbook: {url}")
        msgs.append(f"Runbook excerpts ready: {len(runbooks)}")
        return {
            "loaded_skill_ids": loaded_ids,
            "loaded_skills": loaded_texts,
            "runbook_excerpts": runbooks,
            **_progress("; ".join(msgs)),
        }

    async def plan(state: InvestigationState) -> dict[str, Any]:
        # Replan keeps prior hypotheses; only seed when empty or deep cycle asks
        existing = list(state.get("hypotheses") or [])
        force = bool((state.get("last_event") or {}).get("force_replan"))
        if existing and not force and state.get("cycle", 0) == 0:
            return _progress("Plan skipped — hypotheses already present")

        ctx_dict = state.get("context") or {}
        context = IncidentContext.model_validate(ctx_dict)
        proposals = await llm.propose_hypotheses(
            context,
            memory,
            skill_catalog=skills.catalog(),
            loaded_skills=list(state.get("loaded_skills") or []),
            runbook_excerpts=list(state.get("runbook_excerpts") or []),
        )
        hyps = [] if force else existing
        pending = []
        seen = {h.get("statement") for h in hyps}
        for p in proposals:
            stmt = p.get("statement")
            if not stmt or stmt in seen:
                continue
            seen.add(stmt)
            hid = f"hyp-{len(hyps)+1}-{abs(hash(stmt)) % 10_000_000}"
            hyps.append(
                {
                    "id": hid,
                    "statement": stmt,
                    "component": p.get("component"),
                    "status": "pending",
                    "confidence": 0.0,
                    "evidence": [],
                    "rejected_alternatives": [],
                    "sub_task_id": None,
                }
            )
            pending.append(hid)
        cycle = int(state.get("cycle") or 0) + (1 if force else 0)
        return {
            "hypotheses": hyps,
            "pending_spawn": pending,
            "cycle": cycle if force else state.get("cycle") or 0,
            **_progress(
                f"Planned {len(pending)} hypotheses via {getattr(llm, 'name', 'llm')} "
                f"(cycle={state.get('cycle') or 0})"
            ),
        }

    async def mark_spawn(state: InvestigationState) -> dict[str, Any]:
        """Session layer actually starts LLM sub-agents; graph only marks intent."""
        pending = list(state.get("pending_spawn") or [])
        return {
            **_progress(f"Spawn requested for {len(pending)} hypotheses"),
            "extra": {"spawn_requested": pending},
        }

    async def wait_event(state: InvestigationState) -> dict[str, Any]:
        """Paige interrupt barrier — durable wait for user / sub-agent / replan."""
        event = interrupt(
            {
                "type": "wait_event",
                "active_sub_tasks": state.get("active_sub_tasks") or {},
                "pending_spawn": state.get("pending_spawn") or [],
                "cycle": state.get("cycle") or 0,
            }
        )
        if not isinstance(event, dict):
            event = {"kind": "user_input", "text": str(event)}
        return {"last_event": event, "waiting": False, **_progress(f"Resumed on {event.get('kind')}"), "extra": {"last_interrupt_handled": True}}

    async def route(state: InvestigationState) -> Command[Literal["handle_user", "handle_result", "replan", "synthesize", "cancel", "wait_event", "plan", "spawn"]]:
        event = state.get("last_event") or {}
        kind = event.get("kind")
        if kind in ("cancel",):
            return Command(goto="cancel")
        if kind in ("user_input", "note_added", "chat"):
            return Command(goto="handle_user")
        if kind in ("sub_agent_result",):
            return Command(goto="handle_result")
        if kind in ("replan",):
            return Command(goto="replan")
        if kind in ("spawn_ack",):
            # after session spawns, wait again
            active = state.get("active_sub_tasks") or {}
            if not active and not (state.get("pending_spawn") or []):
                return Command(goto="synthesize")
            return Command(goto="wait_event")
        if kind in ("tick",):
            active = state.get("active_sub_tasks") or {}
            if not active and not (state.get("pending_spawn") or []):
                return Command(goto="synthesize")
            return Command(goto="wait_event")
        return Command(goto="wait_event")

    async def handle_user(state: InvestigationState) -> Command[Literal["spawn", "wait_event", "synthesize", "replan"]]:
        event = state.get("last_event") or {}
        text = str(event.get("text") or event.get("payload", {}).get("text") or "").strip()
        chat_msg = {"role": "user", "content": text}
        updates: dict[str, Any] = {
            "chat": [chat_msg],
            **_progress(f"User/chat: {text[:200]}"),
        }
        lower = text.lower()
        hyps = list(state.get("hypotheses") or [])
        pending = list(state.get("pending_spawn") or [])
        active = dict(state.get("active_sub_tasks") or {})

        if lower.startswith("skip "):
            for h in hyps:
                if h["id"] in text or h.get("statement", "").lower() in lower:
                    h["status"] = "skipped"
                    sid = h.get("sub_task_id")
                    if sid:
                        active.pop(sid, None)
            updates["hypotheses"] = hyps
            updates["active_sub_tasks"] = active
            updates["chat"] = [chat_msg, {"role": "assistant", "content": "Skipped matching hypotheses."}]
            return Command(update=updates, goto="wait_event" if active else "synthesize")

        if lower in ("done", "resolve", "finish", "synthesize"):
            updates["chat"] = [chat_msg, {"role": "assistant", "content": "Synthesizing now."}]
            return Command(update=updates, goto="synthesize")

        if lower.startswith("replan") or "replan" in lower:
            updates["chat"] = [chat_msg, {"role": "assistant", "content": "Replanning investigation…"}]
            return Command(update=updates, goto="replan")

        # Explicit investigate / also-check → spawn hypothesis sub-agent
        if lower.startswith("also check") or lower.startswith("investigate") or lower.startswith("check "):
            lead = text.split(" ", 2)[-1] if " " in text else text
            hid = f"hyp-user-{abs(hash(lead)) % 10_000_000}"
            hyps.append(
                {
                    "id": hid,
                    "statement": f"User lead: {lead}",
                    "component": "user",
                    "status": "pending",
                    "confidence": 0.0,
                    "evidence": [],
                    "rejected_alternatives": [],
                    "sub_task_id": None,
                }
            )
            pending.append(hid)
            updates.update(
                {
                    "hypotheses": hyps,
                    "pending_spawn": pending,
                    "chat": [
                        chat_msg,
                        {
                            "role": "assistant",
                            "content": f"Queued new hypothesis and spawning LLM sub-agent: {lead}",
                        },
                    ],
                }
            )
            return Command(update=updates, goto="spawn")

        # Normal chatbot Q&A against current investigation state
        top = sorted(hyps, key=lambda h: float(h.get("confidence") or 0), reverse=True)
        top_s = top[0].get("statement") if top else "none yet"
        answer = (
            f"Strongest lead so far: {top_s}. "
            f"Ask another question, or say 'investigate <lead>' / 'synthesize'."
        )
        try:
            ctx = IncidentContext.model_validate(state.get("context") or {})
            answer = await llm.answer_question(
                text,
                context=ctx,
                hypotheses=hyps,
                synthesis=state.get("synthesis"),
                recommendations=list(state.get("recommendations") or []),
                loaded_skills=list(state.get("loaded_skills") or []),
                runbook_excerpts=list(state.get("runbook_excerpts") or []),
                chat_history=list(state.get("chat") or [])[-8:],
            )
        except Exception:
            pass
        updates["chat"] = [chat_msg, {"role": "assistant", "content": answer}]
        active = state.get("active_sub_tasks") or {}
        return Command(update=updates, goto="wait_event" if active or not state.get("completed") else "synthesize")

    async def handle_result(state: InvestigationState) -> Command[Literal["wait_event", "synthesize", "replan"]]:
        event = state.get("last_event") or {}
        payload = event.get("payload") or event
        hyp_raw = payload.get("hypothesis") or {}
        hyps = list(state.get("hypotheses") or [])
        active = dict(state.get("active_sub_tasks") or {})
        sub_id = event.get("task_id") or hyp_raw.get("sub_task_id")
        if sub_id:
            active.pop(sub_id, None)
        for i, h in enumerate(hyps):
            if h.get("id") == hyp_raw.get("id"):
                hyps[i] = hyp_raw
                break
        status = str(hyp_raw.get("status") or "").lower()
        stmt = hyp_raw.get("statement") or "this lead"
        if status == "supported":
            msg = f"This looks like a solid lead: {stmt}"
        elif status == "refuted":
            msg = f"That lead doesn't hold up: {stmt}"
        else:
            msg = f"Update on a lead: {stmt}"
        updates = {
            "hypotheses": hyps,
            "active_sub_tasks": active,
            **_progress(msg),
            "chat": [{"role": "assistant", "content": msg}],
        }
        if active or (state.get("pending_spawn") or []):
            return Command(update=updates, goto="wait_event")
        # Deep mode: optionally replan if confidence low and cycles remain
        top_conf = max((float(h.get("confidence") or 0) for h in hyps), default=0)
        cycle = int(state.get("cycle") or 0)
        max_cycles = int(state.get("max_cycles") or 1)
        if state.get("deep_mode") and top_conf < 0.55 and cycle + 1 < max_cycles:
            return Command(update=updates, goto="replan")
        return Command(update=updates, goto="synthesize")

    async def replan(state: InvestigationState) -> Command[Literal["plan"]]:
        cycle = int(state.get("cycle") or 0) + 1
        return Command(
            update={
                "cycle": cycle,
                "last_event": {"kind": "replan", "force_replan": True},
                **_progress(f"Deep replan cycle={cycle}"),
                "chat": [{"role": "assistant", "content": f"Continuous replan cycle {cycle}…"}],
                "pending_spawn": [],
            },
            goto="plan",
        )

    async def synthesize(state: InvestigationState) -> dict[str, Any]:
        ctx = IncidentContext.model_validate(state.get("context") or {})
        hyps = list(state.get("hypotheses") or [])
        result = await llm.synthesize(
            ctx,
            hyps,
            list(state.get("progress") or []),
            loaded_skills=list(state.get("loaded_skills") or []),
            runbook_excerpts=list(state.get("runbook_excerpts") or []),
        )
        summary = str(result.get("summary") or "")
        recs = [str(x) for x in (result.get("recommendations") or [])]
        return {
            "synthesis": summary,
            "recommendations": recs,
            "completed": True,
            **_progress("Synthesis complete"),
            "chat": [{"role": "assistant", "content": summary}],
        }

    async def cancel(state: InvestigationState) -> dict[str, Any]:
        return {
            "canceled": True,
            "completed": True,
            **_progress("Investigation canceled"),
            "chat": [{"role": "assistant", "content": "Investigation canceled."}],
        }

    g = StateGraph(InvestigationState)
    g.add_node("bootstrap", bootstrap)
    g.add_node("load_skills_runbooks", load_skills_runbooks)
    g.add_node("plan", plan)
    g.add_node("spawn", mark_spawn)
    g.add_node("wait_event", wait_event)
    g.add_node("route", route)
    g.add_node("handle_user", handle_user)
    g.add_node("handle_result", handle_result)
    g.add_node("replan", replan)
    g.add_node("synthesize", synthesize)
    g.add_node("cancel", cancel)

    g.add_edge(START, "bootstrap")
    g.add_edge("bootstrap", "load_skills_runbooks")
    g.add_edge("load_skills_runbooks", "plan")
    g.add_edge("plan", "spawn")
    g.add_edge("spawn", "wait_event")
    g.add_edge("wait_event", "route")
    # route/handle_* use Command(goto=...)
    g.add_edge("synthesize", END)
    g.add_edge("cancel", END)
    return g


def build_supervisor_graph(
    *,
    skills: SkillRegistry,
    memory: MemoryBundle | None = None,
    checkpointer: Any = None,
):
    g = build_supervisor_graph_uncompiled(skills=skills, memory=memory)
    if checkpointer is not None:
        return g.compile(checkpointer=checkpointer)
    return g.compile()
