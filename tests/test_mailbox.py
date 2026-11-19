import asyncio

import pytest

from paige_triage.mailbox import PriorityMailbox
from paige_triage.models.events import AgentEvent, EventKind, EventPriority


@pytest.mark.asyncio
async def test_user_events_outrank_subagent_results() -> None:
    box = PriorityMailbox()
    await box.put(
        AgentEvent(
            kind=EventKind.SUB_AGENT_RESULT,
            priority=EventPriority.SUB_AGENT,
            task_id="sub-1",
            parent_task_id="task-1",
            payload={"n": 1},
        )
    )
    await box.put(
        AgentEvent(
            kind=EventKind.USER_INPUT,
            priority=EventPriority.USER,
            task_id="task-1",
            payload={"text": "skip hypothesis 3"},
        )
    )
    first = await box.get()
    second = await box.get()
    assert first.kind == EventKind.USER_INPUT
    assert second.kind == EventKind.SUB_AGENT_RESULT
