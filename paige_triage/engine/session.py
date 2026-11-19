"""Deep investigation session: LangGraph + durable sqlite + LLM sub-agents + chat."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from paige_triage.checkpoint import SupervisorCheckpoint
from paige_triage.connectors.registry import ConnectorRegistry
from paige_triage.context.assembler import ContextAssembler
from paige_triage.engine.graph import build_supervisor_graph
from paige_triage.engine.subagent_llm import LLMHypothesisSubAgent
from paige_triage.identity import TaskIdentity
from paige_triage.llm import get_llm
from paige_triage.mailbox import PriorityMailbox
from paige_triage.models.events import EventKind
from paige_triage.models.hypothesis import Hypothesis
from paige_triage.models.incident import AlertPayload, Incident, IncidentContext, IncidentNote
from paige_triage.models.memory import MemoryBundle
from paige_triage.report import build_investigation_report
from paige_triage.skills.loader import DEFAULT_SKILLS, SkillRegistry


class DeepInvestigationSession:
    def __init__(
        self,
        *,
        connectors: ConnectorRegistry,
        assembler: ContextAssembler,
        memory: MemoryBundle | None = None,
        skills: SkillRegistry | None = None,
        checkpoint_db: Path | str = ".paige_checkpoints/langgraph.sqlite",
        model: str | None = None,
        ollama_base: str | None = None,
        deep_mode: bool = True,
        max_cycles: int = 6,
        replan_interval_s: float = 0.0,
        on_progress: Any = None,
    ) -> None:
        self.identity = TaskIdentity.supervisor()
        self.connectors = connectors
        self.assembler = assembler
        self.memory = memory
        self.skills = skills or SkillRegistry(DEFAULT_SKILLS)
        self.mailbox = PriorityMailbox()
        self.checkpoint_db = Path(checkpoint_db)
        self.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
        self.model = model or os.getenv("PAIGE_LLM_MODEL", "llama3.2:latest")
        self.ollama_base = ollama_base or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        self.deep_mode = deep_mode
        self.max_cycles = max_cycles
        self.replan_interval_s = replan_interval_s
        self.on_progress = on_progress
        self._sub_agents: dict[str, LLMHypothesisSubAgent] = {}
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._graph = None
        self._cm = None
        self._checkpointer = None
        self._watcher: asyncio.Task[Any] | None = None
        self._replan_task: asyncio.Task[Any] | None = None
        self._config: dict[str, Any] = {
            "configurable": {"thread_id": self.identity.thread_id}
        }
        self._values: dict[str, Any] = {}
        self._chat: list[dict[str, str]] = []
        self._last_reply: str | None = None
        self._llm = get_llm()

    @property
    def task_id(self) -> str:
        return self.identity.task_id

    @staticmethod
    def _is_tool_noise(text: str) -> bool:
        t = (text or "").strip()
        return t.startswith("tool:") or t.startswith("LLM sub-agent") or t.startswith("sub-agent error")

    async def _ensure_graph(self) -> None:
        if self._graph is not None:
            return
        self._cm = AsyncSqliteSaver.from_conn_string(str(self.checkpoint_db))
        self._checkpointer = await self._cm.__aenter__()
        self._graph = build_supervisor_graph(
            skills=self.skills,
            memory=self.memory,
            checkpointer=self._checkpointer,
        )

    async def start(
        self,
        incident: Incident,
        alerts: list[AlertPayload],
        notes: list[IncidentNote] | None = None,
    ) -> dict[str, Any]:
        await self._ensure_graph()
        context = self.assembler.assemble(incident, alerts, notes=notes)
        self.connectors.bind_context(context)
        initial: dict[str, Any] = {
            "task_id": self.identity.task_id,
            "thread_id": self.identity.thread_id,
            "incident": incident.model_dump(mode="json"),
            "context": context.model_dump(mode="json"),
            "hypotheses": [],
            "pending_spawn": [],
            "active_sub_tasks": {},
            "loaded_skills": [],
            "loaded_skill_ids": [],
            "runbook_excerpts": [],
            "progress": [],
            "chat": [],
            "synthesis": None,
            "recommendations": [],
            "cycle": 0,
            "max_cycles": self.max_cycles,
            "deep_mode": self.deep_mode,
            "completed": False,
            "canceled": False,
            "waiting": True,
            "extra": {"engine": "langgraph"},
        }
        await self._invoke(initial)
        await self._spawn_pending_from_state()
        await self.resume_event({"kind": "spawn_ack"})
        self._watcher = asyncio.create_task(self._watch_mailbox(), name="mailbox-watch")
        if self.deep_mode and self.replan_interval_s > 0:
            self._replan_task = asyncio.create_task(self._replan_loop(), name="replan-loop")
        return self.snapshot()

    async def _invoke(self, payload: Any) -> None:
        assert self._graph is not None
        result = await self._graph.ainvoke(payload, config=self._config)
        self._values = dict(result or {})
        for m in self._values.get("chat") or []:
            if m not in self._chat:
                self._chat.append(m)

    async def _refresh(self) -> None:
        assert self._graph is not None
        snap = await self._graph.aget_state(self._config)
        if snap and snap.values:
            self._values = dict(snap.values)

    async def resume_event(self, event: dict[str, Any]) -> dict[str, Any]:
        await self._invoke(Command(resume=event))
        await self._spawn_pending_from_state()
        # If plan→spawn happened via replan, ack again while interrupted
        await self._refresh()
        if not self._values.get("completed"):
            pending = self._values.get("pending_spawn") or []
            spawn_req = (self._values.get("extra") or {}).get("spawn_requested") or []
            if pending or spawn_req:
                await self._spawn_pending_from_state()
                # Only resume spawn_ack if graph is waiting on interrupt
                snap = await self._graph.aget_state(self._config)
                if snap and snap.tasks:
                    for t in snap.tasks:
                        if getattr(t, "interrupts", None):
                            await self._invoke(Command(resume={"kind": "spawn_ack"}))
                            break
        await self._refresh()
        return self.snapshot()

    async def _spawn_pending_from_state(self) -> None:
        assert self._graph is not None
        await self._refresh()
        values = dict(self._values)
        pending = list(values.get("pending_spawn") or [])
        if not pending:
            pending = list((values.get("extra") or {}).get("spawn_requested") or [])
        if not pending:
            return

        hyps = list(values.get("hypotheses") or [])
        active = dict(values.get("active_sub_tasks") or {})
        skill_text = "\n\n".join(values.get("loaded_skills") or [])
        runbook_text = "\n\n".join(values.get("runbook_excerpts") or [])
        context = IncidentContext.model_validate(values.get("context") or {})

        for hid in pending:
            hyp_dict = next((h for h in hyps if h.get("id") == hid), None)
            if not hyp_dict or hyp_dict.get("status") in ("skipped", "supported", "refuted"):
                continue
            if any(v == hid for v in active.values()):
                continue
            child = self.identity.spawn_child()
            hyp = Hypothesis.model_validate(
                {**hyp_dict, "status": "investigating", "sub_task_id": child.task_id}
            )
            agent = LLMHypothesisSubAgent(
                identity=child,
                hypothesis=hyp,
                context=context,
                mailbox=self.mailbox,
                connectors=self.connectors,
                model=self.model,
                base_url=self.ollama_base,
                skill_text=skill_text,
                runbook_text=runbook_text,
                on_progress=self.on_progress,
            )
            self._sub_agents[child.task_id] = agent
            active[child.task_id] = hid
            for i, h in enumerate(hyps):
                if h.get("id") == hid:
                    hyps[i] = hyp.model_dump(mode="json")
            self._tasks[child.task_id] = asyncio.create_task(agent.run(), name=child.task_id)
            self._emit_local(f"Spawned LLM sub-agent {child.task_id} for {hyp.statement}")

        await self._graph.aupdate_state(
            self._config,
            {
                "hypotheses": hyps,
                "pending_spawn": [],
                "active_sub_tasks": active,
                "extra": {"spawn_requested": []},
                "progress": [f"Active LLM sub-agents: {len(active)}"],
            },
        )
        await self._refresh()

    async def _watch_mailbox(self) -> None:
        try:
            while True:
                await self._refresh()
                if self._values.get("completed"):
                    break
                event = await self.mailbox.get_nowait_or_wait(timeout=0.5)
                if event is None:
                    continue
                if event.kind == EventKind.SUB_AGENT_PROGRESS:
                    msg = str(event.payload.get("message") or "")
                    self._emit_local(msg)
                    # Keep tool chatter out of the chat transcript (progress stream only)
                    if msg and not self._is_tool_noise(msg):
                        self._chat.append({"role": "assistant", "content": msg})
                    continue
                if event.kind == EventKind.SUB_AGENT_RESULT:
                    await self.resume_event(
                        {
                            "kind": "sub_agent_result",
                            "task_id": event.task_id,
                            "payload": event.payload,
                        }
                    )
        except asyncio.CancelledError:
            return

    async def _replan_loop(self) -> None:
        try:
            while self.replan_interval_s > 0:
                await asyncio.sleep(self.replan_interval_s)
                await self._refresh()
                if self._values.get("completed") or self._values.get("canceled"):
                    break
                cycle = int(self._values.get("cycle") or 0)
                if cycle + 1 >= int(self._values.get("max_cycles") or self.max_cycles):
                    break
                await self.resume_event({"kind": "replan", "force_replan": True})
        except asyncio.CancelledError:
            return

    async def _llm_chat_reply(self, text: str) -> str:
        await self._refresh()
        values = dict(self._values)
        try:
            ctx = IncidentContext.model_validate(values.get("context") or {})
        except Exception:
            from paige_triage.models.incident import Incident, ServiceRef

            inc_raw = values.get("incident") or {}
            if inc_raw:
                inc = Incident.model_validate(inc_raw)
            else:
                inc = Incident(
                    id="unknown",
                    title="Unknown incident",
                    service=ServiceRef(name="unknown"),
                )
            ctx = IncidentContext(incident=inc, alerts=[])
        reply = await self._llm.answer_question(
            text,
            context=ctx,
            hypotheses=list(values.get("hypotheses") or []),
            synthesis=values.get("synthesis"),
            recommendations=list(values.get("recommendations") or []),
            loaded_skills=list(values.get("loaded_skills") or []),
            runbook_excerpts=list(values.get("runbook_excerpts") or []),
            chat_history=list(self._chat),
        )
        return (reply or "").strip() or (
            "I don't have enough signal yet — ask me to investigate a specific lead."
        )

    async def chat(self, text: str) -> dict[str, Any]:
        text = (text or "").strip()
        self._chat.append({"role": "user", "content": text})
        await self._refresh()
        completed = bool(self._values.get("completed") or self._values.get("canceled"))

        if not completed:
            try:
                await self.resume_event({"kind": "chat", "text": text})
            except Exception:
                pass

        last = self._chat[-1] if self._chat else None
        if (
            last
            and last.get("role") == "assistant"
            and last.get("content")
            and not self._is_tool_noise(str(last.get("content") or ""))
        ):
            reply = str(last["content"])
        else:
            reply = await self._llm_chat_reply(text)
            self._chat.append({"role": "assistant", "content": reply})

        self._last_reply = reply
        report = self.snapshot()
        report["reply"] = reply
        return report

    async def cancel(self) -> dict[str, Any]:
        for a in self._sub_agents.values():
            a.cancel()
        return await self.resume_event({"kind": "cancel"})

    async def wait_until_complete(self, timeout: float = 900.0) -> dict[str, Any]:
        async def _wait() -> None:
            while True:
                await self._refresh()
                if self._values.get("completed"):
                    return
                active = self._values.get("active_sub_tasks") or {}
                snap = await self._graph.aget_state(self._config)
                interrupted = False
                if snap and snap.tasks:
                    for t in snap.tasks:
                        if getattr(t, "interrupts", None):
                            interrupted = True
                            break
                if interrupted and not active:
                    await self.resume_event({"kind": "tick"})
                await asyncio.sleep(0.5)

        await asyncio.wait_for(_wait(), timeout=timeout)
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        values = dict(self._values)
        try:
            ctx = IncidentContext.model_validate(values.get("context") or {})
            hyps = [Hypothesis.model_validate(h) for h in (values.get("hypotheses") or [])]
            ckpt = SupervisorCheckpoint(
                task_id=self.identity.task_id,
                thread_id=self.identity.thread_id,
                node="langgraph",
                context=ctx,
                hypotheses=hyps,
                pending_spawn=list(values.get("pending_spawn") or []),
                active_sub_tasks=dict(values.get("active_sub_tasks") or {}),
                synthesis=values.get("synthesis"),
                recommendations=list(values.get("recommendations") or []),
                progress_log=list(values.get("progress") or []),
                canceled=bool(values.get("canceled")),
                completed=bool(values.get("completed")),
                extra={
                    "llm": values.get("llm_name") or "llama",
                    "loaded_skill_ids": values.get("loaded_skill_ids") or [],
                    "loaded_skills": values.get("loaded_skills") or [],
                    "runbook_excerpts": values.get("runbook_excerpts") or [],
                    "engine": "langgraph",
                    "cycle": values.get("cycle"),
                    "deep_mode": values.get("deep_mode"),
                    "checkpoint_db": str(self.checkpoint_db),
                },
            )
            report = build_investigation_report(ckpt)
        except Exception as exc:
            report = {"error": str(exc), "raw": values}
        merged = list(self._chat) + list(values.get("chat") or [])
        seen: set[tuple[str | None, str | None]] = set()
        uniq: list[dict[str, str]] = []
        for m in merged:
            key = (m.get("role"), m.get("content"))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(m)
        report["chat"] = uniq
        report["task_id"] = self.task_id
        report["engine"] = "langgraph"
        if self._last_reply:
            report["reply"] = self._last_reply
        return report

    def _emit_local(self, msg: str) -> None:
        if self.on_progress:
            r = self.on_progress(msg)
            if asyncio.iscoroutine(r):
                asyncio.create_task(r)

    async def shutdown(self) -> None:
        if self._watcher:
            self._watcher.cancel()
        if self._replan_task:
            self._replan_task.cancel()
        for t in self._tasks.values():
            t.cancel()
        self.mailbox.close()
        if self._cm is not None:
            await self._cm.__aexit__(None, None, None)
