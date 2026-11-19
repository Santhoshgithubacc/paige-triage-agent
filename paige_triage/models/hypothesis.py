"""Hypothesis + evidence model for deep investigation.

Finding a root cause: localize the component, then determine what went wrong.
Evidence is direct or circumstantial; alternatives that were rejected are kept.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class EvidenceKind(str, Enum):
    DIRECT = "direct"
    CIRCUMSTANTIAL = "circumstantial"


class HypothesisStatus(str, Enum):
    PENDING = "pending"
    INVESTIGATING = "investigating"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    SKIPPED = "skipped"


class EvidenceItem(BaseModel):
    id: str = Field(default_factory=lambda: f"evd-{uuid4().hex[:8]}")
    kind: EvidenceKind
    claim: str
    source: str
    quote: str | None = None
    confidence: float = 0.5
    created_at: datetime = Field(default_factory=utcnow)


class Hypothesis(BaseModel):
    id: str = Field(default_factory=lambda: f"hyp-{uuid4().hex[:8]}")
    statement: str
    component: str | None = None
    status: HypothesisStatus = HypothesisStatus.PENDING
    confidence: float = 0.0
    evidence: list[EvidenceItem] = Field(default_factory=list)
    rejected_alternatives: list[str] = Field(default_factory=list)
    sub_task_id: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    def add_evidence(self, item: EvidenceItem) -> None:
        self.evidence.append(item)
        self.updated_at = utcnow()
        if item.kind == EvidenceKind.DIRECT:
            self.confidence = min(1.0, self.confidence + 0.35 * item.confidence)
        else:
            self.confidence = min(1.0, self.confidence + 0.12 * item.confidence)
