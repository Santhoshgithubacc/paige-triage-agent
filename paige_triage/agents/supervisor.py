"""Durable supervisor with reactive drain loop.

Graph nodes (Paige experiment → production):
  accept_event → route_event → handle_sub_agent_result | handle_user_input
               → plan (spawn) → synthesize → accept_event (interrupt)

Parallel fan-out + concurrent fan-in. User events priority 0.
Resume lock serializes concurrent completions.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable, Awaitable

from paige_triage.agents.subagent import HypothesisSubAgent
from paige_triage.checkpoint import CheckpointStore, SupervisorCheckpoint
from paige_triage.connectors.registry import ConnectorRegistry
from paige_triage.context.assembler import ContextAssembler
from paige_triage.identity import TaskIdentity
from paige_triage.llm import HeuristicLLM, LLM, get_llm
from paige_triage.mailbox import PriorityMailbox
from paige_triage.models.events import AgentEvent, EventKind, EventPriority
from paige_triage.models.hypothesis import Hypothesis, HypothesisStatus
from paige_triage.models.incident import Incident, IncidentContext, IncidentNote, AlertPayload
from paige_triage.models.memory import MemoryBundle
from paige_triage.skills.loader import DEFAULT_SKILLS, SkillRegistry


ProgressCallback = Callable[[str], Awaitable[None] | None]


class TriageSupervisor:
    def __init__(
        self,
        *,
        checkpoint_dir: Path | str = ".paige_checkpoints",
        connectors: ConnectorRegistry | None = None,
        skills: SkillRegistry | None = None,
        memory: MemoryBundle | None = None,
        assembler: ContextAssembler | None = None,
        llm: LLM | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        self.identity = TaskIdentity.supervisor()
        self.mailbox = PriorityMailbox()
        self.store = CheckpointStore(checkpoint_dir)
        self.connectors = connectors or ConnectorRegistry()
        self.skills = skills or SkillRegistry(DEFAULT_SKILLS)
        self.memory = memory
        self.assembler = assembler or ContextAssembler(memory=memory)
        self.llm = llm or get_llm()
        self.on_progress = on_progress
        self._sub_agents: dict[str, HypothesisSubAgent] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._state: SupervisorCheckpoint | None = None
        self._drain_task: asyncio.Task[None] | None = None
        self._stopped = asyncio.Event()

    @property
    def task_id(self) -> str:
        return self.identity.task_id

    @property
    def state(self) -> SupervisorCheckpoint:
        if not self._state:
            raise RuntimeError("supervisor not started")
        return self._state

    async def start(
        self,
        incident: Incident,
        alerts: list[AlertPayload],
        notes: list[IncidentNote] | None = None,
    ) -> SupervisorCheckpoint:
        context = self.assembler.assemble(incident, alerts, notes=notes)
        self.connectors.bind_context(context)
        self._state = SupervisorCheckpoint(
            task_id=self.identity.task_id,
            thread_id=self.identity.thread_id,
            node="plan",
            context=context,
        )
        self._persist("Investigation started — assembling context")
        await self.mailbox.put(
            AgentEvent(
                kind=EventKind.INVESTIGATION_START,
                priority=EventPriority.SYSTEM,
                task_id=self.identity.task_id,
                payload={"incident_id": incident.id},
            )
        )
        self._drain_task = asyncio.create_task(self._drain_loop(), name="paige-drain")
        return self._state

    async def inject_user_input(self, text: str) -> None:
        await self.mailbox.put(
            AgentEvent(
                kind=EventKind.USER_INPUT,
                priority=EventPriority.USER,
                task_id=self.identity.task_id,
                payload={"text": text},
            )
        )

    async def add_note(self, author: str, body: str) -> None:
        assert self._state
        note = IncidentNote(author=author, body=body)
        notes = list(self._state.context.notes) + [note]
        self._state.context = self.assembler.assemble(
            self._state.context.incident,
            self._state.context.alerts,
            notes=notes,
            previous=self._state.context,
        )
        self.connectors.bind_context(self._state.context)
        await self.mailbox.put(
            AgentEvent(
                kind=EventKind.NOTE_ADDED,
                priority=EventPriority.USER,
                task_id=self.identity.task_id,
                payload={"note": note.model_dump(mode="json")},
            )
        )

    async def cancel(self) -> None:
        await self.mailbox.put(
            AgentEvent(
                kind=EventKind.CANCEL,
                priority=EventPriority.USER,
                task_id=self.identity.task_id,
                payload={},
            )
        )

    async def wait_until_complete(self, timeout: float | None = 120.0) -> SupervisorCheckpoint:
        assert self._state
        async def _wait() -> None:
            while not self._state.completed and not self._state.canceled:
                await asyncio.sleep(0.05)

        await asyncio.wait_for(_wait(), timeout=timeout)
        return self._state

    async def shutdown(self) -> None:
        self._stopped.set()
        for agent in self._sub_agents.values():
            agent.cancel()
        for task in self._tasks.values():
            task.cancel()
        self.mailbox.close()
        if self._drain_task:
            self._drain_task.cancel()
            try:
                await self._drain_task
            except asyncio.CancelledError:
                pass

    # --- reactive loop ---

    async def _drain_loop(self) -> None:
        """Spine of the architecture: serialize resumes with the lock."""
        try:
            # Initial plan before waiting on mailbox
            async with self.mailbox.resume_lock:
                await self._node_plan()
                self._state.node = "accept_event"
                self._persist("Waiting for events")

            while not self._stopped.is_set():
                if self._state and (self._state.completed or self._state.canceled):
                    break
                event = await self.mailbox.get_nowait_or_wait(timeout=0.5)
                if event is None:
                    # If all sub-agents done and nothing pending → synthesize
                    if self._state and not self._state.active_sub_tasks and not self._state.pending_spawn:
                        if self._state.hypotheses and not self._state.completed:
                            async with self.mailbox.resume_lock:
                                await self._node_synthesize()
                        continue
                    continue

                async with self.mailbox.resume_lock:
                    await self._route_event(event)
                    self._state.node = "accept_event"
                    self._persist(f"Processed {event.kind.value}")
        except asyncio.CancelledError:
            return

    async def _route_event(self, event: AgentEvent) -> None:
        assert self._state
        self._state.node = "route_event"
        if event.kind == EventKind.CANCEL:
            self._state.canceled = True
            for agent in self._sub_agents.values():
                agent.cancel()
            self._state.active_sub_tasks.clear()
            self._emit("Investigation canceled by user")
            return
        if event.kind == EventKind.USER_INPUT:
            await self._handle_user_input(event)
            return
        if event.kind == EventKind.NOTE_ADDED:
            text = event.payload.get("note", {}).get("body", "")
            await self._handle_user_input(
                AgentEvent(
                    kind=EventKind.USER_INPUT,
                    priority=EventPriority.USER,
                    task_id=self.identity.task_id,
                    payload={"text": f"New incident note: {text}"},
                )
            )
            return
        if event.kind == EventKind.SUB_AGENT_PROGRESS:
            msg = event.payload.get("message", "")
            self._emit(msg)
            return
        if event.kind == EventKind.SUB_AGENT_RESULT:
            await self._handle_sub_agent_result(event)
            return
        if event.kind == EventKind.INVESTIGATION_START:
            return

    async def _handle_user_input(self, event: AgentEvent) -> None:
        assert self._state
        self._state.node = "handle_user_input"
        text = str(event.payload.get("text", "")).strip()
        self._emit(f"User steering: {text}")

        lower = text.lower()
        # Skip hypothesis N
        if lower.startswith("skip hypothesis"):
            for hyp in self._state.hypotheses:
                if hyp.id in text or hyp.statement.lower() in lower:
                    hyp.status = HypothesisStatus.SKIPPED
                    if hyp.sub_task_id and hyp.sub_task_id in self._sub_agents:
                        self._sub_agents[hyp.sub_task_id].cancel()
                        self._state.active_sub_tasks.pop(hyp.sub_task_id, None)
                    self._emit(f"Skipped: {hyp.statement}")
            return

        # Inject new hypothesis mid-run
        if lower.startswith("also check") or lower.startswith("investigate"):
            statement = text.split(" ", 2)[-1] if " " in text else text
            hyp = Hypothesis(statement=statement, status=HypothesisStatus.PENDING)
            self._state.hypotheses.append(hyp)
            self._state.pending_spawn.append(hyp.id)
            await self._node_plan()
            return

        # Generic enrichment — refresh context cue into a new hyp if discriminative
        hyp = Hypothesis(
            statement=f"User-provided lead: {text}",
            status=HypothesisStatus.PENDING,
            component="user",
        )
        self._state.hypotheses.append(hyp)
        self._state.pending_spawn.append(hyp.id)
        await self._node_plan()

    async def _handle_sub_agent_result(self, event: AgentEvent) -> None:
        assert self._state
        self._state.node = "handle_sub_agent_result"
        sub_id = event.task_id
        self._state.active_sub_tasks.pop(sub_id, None)
        self._sub_agents.pop(sub_id, None)
        self._tasks.pop(sub_id, None)

        raw = event.payload.get("hypothesis") or {}
        updated = Hypothesis.model_validate(raw)
        for i, hyp in enumerate(self._state.hypotheses):
            if hyp.id == updated.id:
                self._state.hypotheses[i] = updated
                break
        self._emit(
            f"Hypothesis result: {updated.statement} → {updated.status.value} "
            f"(confidence={updated.confidence:.2f})"
        )

        if not self._state.active_sub_tasks and not self._state.pending_spawn:
            await self._node_synthesize()

    async def _load_skills_and_runbooks(self) -> tuple[list[str], list[str]]:
        """Paige progressive load: catalog → relevant skills → fetch runbooks."""
        assert self._state
        blob = (
            f"{self._state.context.incident.title} "
            f"{self._state.context.primary_alert().summary if self._state.context.primary_alert() else ''}"
        )
        catalog = self.skills.catalog()
        self._emit(f"LLM={getattr(self.llm, 'name', 'unknown')} · skill catalog={len(catalog)}")
        candidates = self.skills.select_candidates(blob, limit=3)
        loaded_texts: list[str] = []
        loaded_ids: list[str] = []
        for skill in candidates:
            full = self.skills.load(skill.id)
            if not full:
                continue
            loaded_ids.append(full.id)
            loaded_texts.append(f"[{full.name}]\n{full.instructions}")
            self._emit(f"Loaded skill: {full.name} ({full.id})")

        runbook_excerpts: list[str] = list(self._state.context.runbook_sections[:2])
        # Explicit connector fetch (CSV toy or PRODUCTION later)
        alert = self._state.context.primary_alert()
        url = None
        if alert:
            url = alert.custom_details.get("runbook_url") or alert.custom_details.get("github")
        result = await self.connectors.invoke(
            "fetch_runbook",
            url=url or self._state.context.incident.service.name,
            title=self._state.context.incident.title,
        )
        if result.ok:
            body = str(result.data.get("body") or result.quote or "")
            title = str(result.data.get("title") or "runbook")
            excerpt = f"## {title}\n{body[:1200]}"
            if excerpt not in runbook_excerpts:
                runbook_excerpts.append(excerpt)
            self._emit(f"Loaded runbook: {title}")
        else:
            self._emit(f"Runbook fetch skipped: {result.error or 'not found'}")

        self._state.extra["loaded_skill_ids"] = loaded_ids
        self._state.extra["loaded_skills"] = loaded_texts
        self._state.extra["runbook_excerpts"] = runbook_excerpts
        self._state.extra["llm"] = getattr(self.llm, "name", "unknown")
        return loaded_texts, runbook_excerpts

    async def _node_plan(self) -> None:
        assert self._state
        self._state.node = "plan"

        # Initial hypothesis fan-out — load skills/runbooks first (real Paige flow)
        if not self._state.hypotheses:
            loaded_skills, runbook_excerpts = await self._load_skills_and_runbooks()
            self._emit("Asking LLM for hypotheses with loaded skills + runbooks…")
            proposals = await self.llm.propose_hypotheses(
                self._state.context,
                self.memory,
                skill_catalog=self.skills.catalog(),
                loaded_skills=loaded_skills,
                runbook_excerpts=runbook_excerpts,
            )
            for p in proposals:
                hyp = Hypothesis(
                    statement=p["statement"],
                    component=p.get("component"),
                    status=HypothesisStatus.PENDING,
                )
                self._state.hypotheses.append(hyp)
                self._state.pending_spawn.append(hyp.id)
            self._emit(f"Planned {len(proposals)} hypotheses via {getattr(self.llm, 'name', 'llm')}")

        # Spawn pending
        to_spawn = list(self._state.pending_spawn)
        self._state.pending_spawn.clear()
        for hyp_id in to_spawn:
            hyp = next((h for h in self._state.hypotheses if h.id == hyp_id), None)
            if not hyp or hyp.status == HypothesisStatus.SKIPPED:
                continue
            child = self.identity.spawn_child()
            agent = HypothesisSubAgent(
                identity=child,
                hypothesis=hyp,
                context=self._state.context,
                mailbox=self.mailbox,
                connectors=self.connectors,
                skills=self.skills,
                memory=self.memory,
            )
            self._sub_agents[child.task_id] = agent
            self._state.active_sub_tasks[child.task_id] = hyp.id
            hyp.sub_task_id = child.task_id
            hyp.status = HypothesisStatus.INVESTIGATING
            task = asyncio.create_task(agent.run(), name=child.task_id)
            self._tasks[child.task_id] = task
            self._emit(f"Spawned sub-agent {child.task_id} for: {hyp.statement}")

        self._persist("Plan complete")

    async def _node_synthesize(self) -> None:
        assert self._state
        self._state.node = "synthesize"
        payload = [h.model_dump(mode="json") for h in self._state.hypotheses]
        self._emit("Synthesizing grounded answer with LLM in real time…")
        result = await self.llm.synthesize(
            self._state.context,
            payload,
            self._state.progress_log,
            loaded_skills=list(self._state.extra.get("loaded_skills") or []),
            runbook_excerpts=list(self._state.extra.get("runbook_excerpts") or []),
        )
        self._state.synthesis = result.get("summary")
        self._state.recommendations = list(result.get("recommendations") or [])
        self._state.completed = True
        self._emit(self._state.synthesis or "Synthesis complete")
        # Persist learning hook — caller may promote to memory on resolve
        await self.connectors.invoke(
            "add_incident_note",
            author="paige",
            body=self._state.synthesis or "",
        )
        self._persist("Investigation complete")

    def _persist(self, message: str | None = None) -> None:
        assert self._state
        if message:
            self._state.progress_log.append(message)
        self._state.step += 1
        # Checkpoint after each atomic step; mailbox not stored (drained)
        self.store.save(self._state)

    def _emit(self, message: str) -> None:
        assert self._state
        self._state.progress_log.append(message)
        if self.on_progress:
            result = self.on_progress(message)
            if asyncio.iscoroutine(result):
                asyncio.create_task(result)

    def snapshot(self) -> dict[str, Any]:
        assert self._state
        return self._state.model_dump(mode="json")
