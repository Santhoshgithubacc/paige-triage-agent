"""LangGraph state for Paige deep investigation."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


def _merge_dicts(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a or {})
    out.update(b or {})
    return out


class InvestigationState(TypedDict, total=False):
    task_id: str
    thread_id: str
    incident: dict[str, Any]
    context: dict[str, Any]
    hypotheses: list[dict[str, Any]]
    pending_spawn: list[str]
    active_sub_tasks: dict[str, str]
    loaded_skills: list[str]
    loaded_skill_ids: list[str]
    runbook_excerpts: list[str]
    last_event: dict[str, Any] | None
    progress: Annotated[list[str], operator.add]
    chat: Annotated[list[dict[str, str]], operator.add]
    synthesis: str | None
    recommendations: list[str]
    cycle: int
    max_cycles: int
    deep_mode: bool
    completed: bool
    canceled: bool
    waiting: bool
    llm_name: str
    extra: Annotated[dict[str, Any], _merge_dicts]
