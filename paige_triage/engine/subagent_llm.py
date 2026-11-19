"""LLM hypothesis sub-agent — separate Llama agent with tools (not scripted)."""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Awaitable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama

from paige_triage.connectors.registry import ConnectorRegistry
from paige_triage.engine.tools import build_investigation_tools
from paige_triage.identity import TaskIdentity
from paige_triage.mailbox import PriorityMailbox
from paige_triage.models.events import AgentEvent, EventKind, EventPriority
from paige_triage.models.hypothesis import (
    EvidenceItem,
    EvidenceKind,
    Hypothesis,
    HypothesisStatus,
)
from paige_triage.models.incident import IncidentContext


ProgressFn = Callable[[str], Awaitable[None] | None]


class LLMHypothesisSubAgent:
    """Stateless LLM investigator for one hypothesis.

    Uses ChatOllama tool-calling. Reports via mailbox. No own checkpoint —
    supervisor re-spawns on failure (Paige asymmetry).
    """

    def __init__(
        self,
        *,
        identity: TaskIdentity,
        hypothesis: Hypothesis,
        context: IncidentContext,
        mailbox: PriorityMailbox,
        connectors: ConnectorRegistry,
        model: str = "llama3.2:latest",
        base_url: str = "http://127.0.0.1:11434",
        max_tool_rounds: int = 4,
        skill_text: str = "",
        runbook_text: str = "",
        on_progress: ProgressFn | None = None,
    ) -> None:
        self.identity = identity
        self.hypothesis = hypothesis
        self.context = context
        self.mailbox = mailbox
        self.connectors = connectors
        self.max_tool_rounds = max_tool_rounds
        self.skill_text = skill_text
        self.runbook_text = runbook_text
        self.on_progress = on_progress
        self._canceled = False
        self.tools = build_investigation_tools(
            connectors, default_service=context.incident.service.name
        )
        self.tool_map = {t.name: t for t in self.tools}
        self.llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=0.15,
        ).bind_tools(self.tools)

    def cancel(self) -> None:
        self._canceled = True

    async def _emit(self, message: str) -> None:
        await self.mailbox.put(
            AgentEvent(
                kind=EventKind.SUB_AGENT_PROGRESS,
                priority=EventPriority.SUB_AGENT,
                task_id=self.identity.task_id,
                parent_task_id=self.identity.parent_task_id,
                payload={"message": message, "hypothesis_id": self.hypothesis.id},
            )
        )
        if self.on_progress:
            r = self.on_progress(message)
            if hasattr(r, "__await__"):
                await r  # type: ignore[misc]

    async def run(self) -> Hypothesis:
        hyp = self.hypothesis
        hyp.status = HypothesisStatus.INVESTIGATING
        hyp.sub_task_id = self.identity.task_id
        await self._emit(f"LLM sub-agent investigating: {hyp.statement}")

        system = (
            "You are a Paige hypothesis investigator (sub-agent). "
            "Investigate ONLY this hypothesis using tools. "
            "Bias to relevance — call tools only when discriminative. "
            "After tools, conclude with JSON only:\n"
            '{"status":"supported|refuted","confidence":0.0-1.0,'
            '"evidence":[{"kind":"direct|circumstantial","claim":"...","source":"...","quote":"..."}],'
            '"reasoning":"..."}\n'
            "Ground claims in tool quotes. No markdown fences."
        )
        user = (
            f"HYPOTHESIS: {hyp.statement}\n"
            f"COMPONENT: {hyp.component}\n"
            f"INCIDENT: {self.context.incident.title}\n"
            f"SERVICE: {self.context.incident.service.name}\n"
            f"ALERTS: {json.dumps([a.model_dump(mode='json') for a in self.context.alerts], default=str)}\n"
            f"CHANGES: {json.dumps([c.model_dump(mode='json') for c in self.context.change_events[:5]], default=str)}\n"
            f"LOADED_SKILL:\n{self.skill_text[:1200]}\n"
            f"RUNBOOK:\n{self.runbook_text[:1200]}\n"
        )
        messages: list[Any] = [SystemMessage(content=system), HumanMessage(content=user)]

        try:
            for round_i in range(self.max_tool_rounds):
                if self._canceled:
                    hyp.status = HypothesisStatus.SKIPPED
                    await self._complete(hyp)
                    return hyp
                ai: AIMessage = await self.llm.ainvoke(messages)
                messages.append(ai)
                tool_calls = getattr(ai, "tool_calls", None) or []
                if not tool_calls:
                    self._apply_conclusion(hyp, str(ai.content or ""))
                    break
                for tc in tool_calls:
                    if self._canceled:
                        break
                    name = tc.get("name")
                    args = tc.get("args") or {}
                    tool = self.tool_map.get(name)
                    await self._emit(f"tool:{name}({json.dumps(args)[:120]})")
                    if not tool:
                        out = f"unknown tool {name}"
                    else:
                        out = await tool.ainvoke(args)
                    messages.append(
                        ToolMessage(content=str(out)[:3000], tool_call_id=tc.get("id") or name)
                    )
                    # Capture tool output as circumstantial/direct evidence seed
                    if out and not str(out).startswith("ERROR") and "no hits" not in str(out):
                        quote = str(out).splitlines()[0][:240]
                        hyp.add_evidence(
                            EvidenceItem(
                                kind=EvidenceKind.CIRCUMSTANTIAL,
                                claim=f"Tool {name} returned signal",
                                source=f"tool.{name}",
                                quote=quote,
                                confidence=0.45,
                            )
                        )
            else:
                # max rounds without bare conclusion — force final invoke
                messages.append(
                    HumanMessage(
                        content="Stop calling tools. Return the final JSON conclusion now."
                    )
                )
                ai = await self.llm.ainvoke(messages)
                self._apply_conclusion(hyp, str(ai.content or ""))
        except Exception as exc:
            await self._emit(f"sub-agent error: {exc}")
            if hyp.evidence and hyp.confidence >= 0.3:
                hyp.status = HypothesisStatus.SUPPORTED
            else:
                hyp.status = HypothesisStatus.REFUTED
                hyp.rejected_alternatives.append(f"sub-agent failed: {exc}")

        if hyp.status == HypothesisStatus.INVESTIGATING:
            hyp.status = (
                HypothesisStatus.SUPPORTED if hyp.confidence >= 0.45 else HypothesisStatus.REFUTED
            )
        await self._complete(hyp)
        return hyp

    def _apply_conclusion(self, hyp: Hypothesis, text: str) -> None:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        data: dict[str, Any] = {}
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                data = {}
        status = str(data.get("status") or "").lower()
        if status == "supported":
            hyp.status = HypothesisStatus.SUPPORTED
        elif status == "refuted":
            hyp.status = HypothesisStatus.REFUTED
        conf = data.get("confidence")
        if isinstance(conf, (int, float)):
            hyp.confidence = max(hyp.confidence, float(conf))
        for ev in data.get("evidence") or []:
            if not isinstance(ev, dict):
                continue
            kind = EvidenceKind.DIRECT if ev.get("kind") == "direct" else EvidenceKind.CIRCUMSTANTIAL
            item = EvidenceItem(
                kind=kind,
                claim=str(ev.get("claim") or "evidence"),
                source=str(ev.get("source") or "llm_subagent"),
                quote=str(ev.get("quote") or "")[:400] or None,
                confidence=0.7 if kind == EvidenceKind.DIRECT else 0.45,
            )
            hyp.add_evidence(item)
        if data.get("reasoning"):
            hyp.add_evidence(
                EvidenceItem(
                    kind=EvidenceKind.CIRCUMSTANTIAL,
                    claim="Sub-agent reasoning",
                    source="llm_subagent",
                    quote=str(data["reasoning"])[:400],
                    confidence=0.4,
                )
            )

    async def _complete(self, hyp: Hypothesis) -> None:
        await self.mailbox.put(
            AgentEvent(
                kind=EventKind.SUB_AGENT_RESULT,
                priority=EventPriority.SUB_AGENT,
                task_id=self.identity.task_id,
                parent_task_id=self.identity.parent_task_id,
                payload={
                    "hypothesis": hyp.model_dump(mode="json"),
                    "status": hyp.status.value,
                },
            )
        )
        await self._emit(
            f"LLM sub-agent done: {hyp.status.value} conf={hyp.confidence:.2f}"
        )
