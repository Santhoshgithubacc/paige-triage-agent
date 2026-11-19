"""Task identity primitive.

Convention from Paige: LangGraph thread_id === agent task_id.
Sub-agents carry parent_task_id so completion events resume the right graph
with no lookup table.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4


def new_task_id(prefix: str = "task") -> str:
    return f"{prefix}-{uuid4().hex}"


@dataclass(frozen=True, slots=True)
class TaskIdentity:
    task_id: str
    parent_task_id: str | None = None

    @property
    def thread_id(self) -> str:
        """Identical to task_id — the Paige identity convention."""
        return self.task_id

    @classmethod
    def supervisor(cls) -> TaskIdentity:
        tid = new_task_id("task")
        return cls(task_id=tid, parent_task_id=None)

    def spawn_child(self) -> TaskIdentity:
        return TaskIdentity(task_id=new_task_id("sub"), parent_task_id=self.task_id)
