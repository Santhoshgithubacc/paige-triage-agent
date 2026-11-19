"""Durable supervisor checkpoint store.

Only the supervisor is durable. Sub-agents are stateless and re-spawned.
Each drain-loop step runs the graph forward to the next interrupt and
persists before touching the next mailbox event.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from paige_triage.models.hypothesis import Hypothesis
from paige_triage.models.incident import IncidentContext


class SupervisorCheckpoint(BaseModel):
    task_id: str
    thread_id: str
    node: str = "accept_event"
    context: IncidentContext
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    pending_spawn: list[str] = Field(default_factory=list)
    active_sub_tasks: dict[str, str] = Field(default_factory=dict)  # sub_task_id -> hyp_id
    synthesis: str | None = None
    recommendations: list[str] = Field(default_factory=list)
    progress_log: list[str] = Field(default_factory=list)
    canceled: bool = False
    completed: bool = False
    step: int = 0
    extra: dict[str, Any] = Field(default_factory=dict)


class CheckpointStore:
    """File-backed durable store (one source of truth for the supervisor)."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, task_id: str) -> Path:
        return self.root / f"{task_id}.json"

    def save(self, state: SupervisorCheckpoint) -> None:
        path = self.path_for(state.task_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(state.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(path)

    def load(self, task_id: str) -> SupervisorCheckpoint | None:
        path = self.path_for(task_id)
        if not path.exists():
            return None
        return SupervisorCheckpoint.model_validate_json(path.read_text(encoding="utf-8"))

    def delete(self, task_id: str) -> None:
        path = self.path_for(task_id)
        if path.exists():
            path.unlink()

    def list_task_ids(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.json"))
