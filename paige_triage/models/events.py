"""Agent lifecycle and mailbox events.

Inspired by A2A lifecycle: working → completed | failed | canceled.
User input is a first-class event with higher priority than sub-agent results.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import IntEnum, Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class EventPriority(IntEnum):
    """Lower number = higher priority (processed first)."""

    USER = 0
    SUB_AGENT = 1
    SYSTEM = 2


class EventKind(str, Enum):
    USER_INPUT = "user_input"
    SUB_AGENT_RESULT = "sub_agent_result"
    SUB_AGENT_PROGRESS = "sub_agent_progress"
    CANCEL = "cancel"
    NOTE_ADDED = "note_added"
    CONTEXT_REFRESHED = "context_refreshed"
    INVESTIGATION_START = "investigation_start"
    SYNTHESIS_READY = "synthesis_ready"


class AgentEvent(BaseModel):
    id: str = Field(default_factory=lambda: f"evt-{uuid4().hex[:8]}")
    kind: EventKind
    priority: EventPriority
    task_id: str
    parent_task_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)

    def sort_key(self) -> tuple[int, float, str]:
        return (int(self.priority), self.created_at.timestamp(), self.id)
