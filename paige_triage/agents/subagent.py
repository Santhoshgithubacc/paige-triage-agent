"""Stateless hypothesis sub-agent.

Runs, produces a result, reports back via the injected mailbox.
No checkpoints — if it dies, the supervisor re-spawns it.
Only reaches out to connectors when a fetch would change confidence.
"""

from __future__ import annotations

import asyncio
from typing import Any

from paige_triage.connectors.registry import ConnectorRegistry
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
from paige_triage.models.memory import MemoryBundle
from paige_triage.skills.loader import SkillRegistry


class HypothesisSubAgent:
    def __init__(
        self,
        identity: TaskIdentity,
        hypothesis: Hypothesis,
        context: IncidentContext,
        mailbox: PriorityMailbox,
        connectors: ConnectorRegistry,
        skills: SkillRegistry,
        memory: MemoryBundle | None = None,
    ) -> None:
        self.identity = identity
        self.hypothesis = hypothesis
        self.context = context
        self.mailbox = mailbox
        self.connectors = connectors
        self.skills = skills
        self.memory = memory
        self._canceled = asyncio.Event()

    def cancel(self) -> None:
        self._canceled.set()

    async def run(self) -> None:
        hyp = self.hypothesis
        hyp.status = HypothesisStatus.INVESTIGATING
        hyp.sub_task_id = self.identity.task_id
        await self._progress(f"Investigating: {hyp.statement}")

        if self._canceled.is_set():
            await self._complete(status=HypothesisStatus.SKIPPED)
            return

        # Progressive skill load — catalog first, full instructions only if relevant
        candidates = self.skills.select_candidates(
            f"{hyp.statement} {self.context.incident.title}"
        )
        skill_text = ""
        for skill in candidates[:1]:
            loaded = self.skills.load(skill.id)
            if loaded:
                skill_text = loaded.instructions
                await self._progress(f"Loaded skill: {loaded.name}")

        # Memory / playbook evidence first (no external fetch)
        await self._apply_memory_evidence()
        if self._canceled.is_set():
            await self._complete(status=HypothesisStatus.SKIPPED)
            return

        # Targeted connector fetches only when discriminative
        await self._maybe_search_logs(skill_text)
        if self._canceled.is_set():
            await self._complete(status=HypothesisStatus.SKIPPED)
            return
        await self._maybe_check_changes()
        await self._maybe_query_metrics()
        await self._maybe_search_code()

        # Classify
        if hyp.confidence >= 0.55:
            hyp.status = HypothesisStatus.SUPPORTED
        elif hyp.confidence <= 0.15 and hyp.evidence:
            hyp.status = HypothesisStatus.REFUTED
        elif not hyp.evidence:
            hyp.status = HypothesisStatus.REFUTED
            hyp.rejected_alternatives.append("No supporting evidence found in scoped queries")
        else:
            # Circumstantial only — keep investigating status as weak support
            hyp.status = HypothesisStatus.SUPPORTED if hyp.confidence >= 0.3 else HypothesisStatus.REFUTED

        await self._complete(status=hyp.status)

    async def _apply_memory_evidence(self) -> None:
        if not self.memory:
            return
        sig = self.context.fingerprint()
        stmt = self.hypothesis.statement.lower()
        for pb in self.memory.match_playbooks(sig):
            title_l = pb.title.lower()
            relevant = (
                "playbook" in stmt
                or "known playbook" in stmt
                or any(tok in stmt for tok in title_l.split() if len(tok) > 3)
                or any(tok in stmt for tok in ("lag", "kafka", "consumer", "duplicate"))
            )
            if not relevant:
                continue
            for quote in pb.evidence_quotes[:2]:
                self.hypothesis.add_evidence(
                    EvidenceItem(
                        kind=EvidenceKind.DIRECT,
                        claim=f"Playbook '{pb.title}' matches alert signature",
                        source="memory.playbook",
                        quote=quote,
                        confidence=0.9,
                    )
                )
            for step in pb.steps[:3]:
                self.hypothesis.add_evidence(
                    EvidenceItem(
                        kind=EvidenceKind.CIRCUMSTANTIAL,
                        claim=f"Playbook step: {step}",
                        source="memory.playbook",
                        quote=step,
                        confidence=0.6,
                    )
                )
        for rec in self.memory.match_recollections(sig):
            if rec.discriminative_query:
                self.hypothesis.add_evidence(
                    EvidenceItem(
                        kind=EvidenceKind.CIRCUMSTANTIAL,
                        claim="Prior recollection query available",
                        source="memory.recollection",
                        quote=rec.discriminative_query,
                        confidence=0.5,
                    )
                )

    async def _maybe_search_logs(self, skill_text: str) -> None:
        alert = self.context.primary_alert()
        # Only auto-search when provider/target/window clear from context — Paige bias
        query = None
        if self.memory and self.memory.profile and self.memory.profile.log_queries:
            query = self.memory.profile.log_queries[0]
        if not query and alert:
            # Derive from alert discriminators
            err = alert.custom_details.get("error") or alert.custom_details.get("error_type")
            query = str(err) if err else alert.summary.split(":")[0][:80]
        if not query and "log" not in self.hypothesis.statement.lower():
            # Propose rather than speculative search when unclear
            await self._progress("Skipped speculative log search — target unclear")
            return

        # Hypothesis-specific keywords
        stmt = self.hypothesis.statement.lower()
        if "dns" in stmt:
            query = f"{query} dns OR nxdomain OR servfail"
        elif "kafka" in stmt or "lag" in stmt or "consumer" in stmt:
            query = f"{query} consumer lag OR OffsetOutOfRange OR Deserialization"
        elif "deploy" in stmt:
            query = f"{query} NullPointer OR Exception after deploy"
        elif "timeout" in stmt or "downstream" in stmt:
            query = f"{query} timeout OR connection refused OR 503"

        result = await self.connectors.invoke(
            "search_logs",
            query=query,
            service=self.context.incident.service.name,
            window_minutes=30,
        )
        await self._progress(f"search_logs → hits={result.data.get('count', 0)}")
        hits = list(result.data.get("hits") or [])
        if result.ok and hits:
            best_quote = ""
            best_overlap = -1
            for row in hits:
                quote = str(row.get("message", ""))
                quote_l = quote.lower()
                overlap = sum(1 for tok in stmt.split() if len(tok) > 3 and tok in quote_l)
                # Boost known discriminative tokens for this hyp family
                for tok in ("deserialization", "duplicate", "lag", "nxdomain", "timeout"):
                    if tok in stmt or tok in query.lower():
                        if tok in quote_l:
                            overlap += 2
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_quote = quote
            self.hypothesis.add_evidence(
                EvidenceItem(
                    kind=EvidenceKind.DIRECT if best_overlap > 0 else EvidenceKind.CIRCUMSTANTIAL,
                    claim="Log line supports or correlates with hypothesis",
                    source="logs.search_logs",
                    quote=best_quote or result.quote,
                    confidence=0.85 if best_overlap > 0 else 0.4,
                )
            )
        elif result.ok:
            self.hypothesis.add_evidence(
                EvidenceItem(
                    kind=EvidenceKind.CIRCUMSTANTIAL,
                    claim="Scoped log query returned no hits",
                    source="logs.search_logs",
                    quote=f"query={query}",
                    confidence=0.2,
                )
            )
        # Skill text is guidance only — does not invent evidence
        _ = skill_text

    async def _maybe_check_changes(self) -> None:
        if not self.context.change_events:
            return
        stmt = self.hypothesis.statement.lower()
        if "deploy" not in stmt and "release" not in stmt and "config" not in stmt:
            # Still useful as circumstantial for any hyp when changes exist
            ch = self.context.change_events[0]
            self.hypothesis.add_evidence(
                EvidenceItem(
                    kind=EvidenceKind.CIRCUMSTANTIAL,
                    claim="Recent change event in window",
                    source="pagerduty.change_events",
                    quote=ch.summary,
                    confidence=0.35,
                )
            )
            return
        ch = self.context.change_events[0]
        self.hypothesis.add_evidence(
            EvidenceItem(
                kind=EvidenceKind.DIRECT,
                claim="Change event aligns with deploy/config hypothesis",
                source="pagerduty.change_events",
                quote=ch.summary,
                confidence=0.8,
            )
        )

    async def _maybe_query_metrics(self) -> None:
        stmt = self.hypothesis.statement.lower()
        metric = "error_rate"
        if "lag" in stmt or "kafka" in stmt or "consumer" in stmt:
            metric = "consumer_lag"
        elif "timeout" in stmt or "latency" in stmt:
            metric = "p99_latency"
        elif "dns" in stmt:
            metric = "dns_failure_rate"
        result = await self.connectors.invoke(
            "query_metrics",
            metric=metric,
            service=self.context.incident.service.name,
        )
        if not result.ok:
            return
        series = result.data.get("series") or []
        if not series:
            return
        points = series[0].get("points") or []
        if points and float(points[-1].get("v", 0)) > 0.5:
            self.hypothesis.add_evidence(
                EvidenceItem(
                    kind=EvidenceKind.CIRCUMSTANTIAL,
                    claim=f"Metric {metric} spiked",
                    source="metrics.query_metrics",
                    quote=f"{metric}={points[-1].get('v')}",
                    confidence=0.55,
                )
            )

    async def _maybe_search_code(self) -> None:
        stmt = self.hypothesis.statement.lower()
        if "deploy" not in stmt and "code" not in stmt and "regression" not in stmt:
            return
        result = await self.connectors.invoke(
            "search_code",
            query=self.context.incident.service.name,
        )
        if result.ok and result.quote:
            self.hypothesis.add_evidence(
                EvidenceItem(
                    kind=EvidenceKind.CIRCUMSTANTIAL,
                    claim="Recent commit touches service",
                    source="github.search_code",
                    quote=result.quote,
                    confidence=0.4,
                )
            )

    async def _progress(self, message: str) -> None:
        await self.mailbox.put(
            AgentEvent(
                kind=EventKind.SUB_AGENT_PROGRESS,
                priority=EventPriority.SUB_AGENT,
                task_id=self.identity.task_id,
                parent_task_id=self.identity.parent_task_id,
                payload={"message": message, "hypothesis_id": self.hypothesis.id},
            )
        )

    async def _complete(self, status: HypothesisStatus) -> None:
        self.hypothesis.status = status
        await self.mailbox.put(
            AgentEvent(
                kind=EventKind.SUB_AGENT_RESULT,
                priority=EventPriority.SUB_AGENT,
                task_id=self.identity.task_id,
                parent_task_id=self.identity.parent_task_id,
                payload={
                    "hypothesis": self.hypothesis.model_dump(mode="json"),
                    "status": status.value,
                },
            )
        )
