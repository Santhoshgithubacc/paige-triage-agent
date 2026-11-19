"""In-process priority mailbox.

Production Paige simplification: supervisor and sub-agents share one process.
Transport collapses to an asyncio.Queue injected into each background task.
User input priority 0 jumps ahead of buffered sub-agent results (priority 1).

The queue itself is not durable — the supervisor drains it atomically into
checkpointed state. On restart, reload checkpoint with an empty mailbox and
re-spawn any in-flight sub-agents.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from paige_triage.models.events import AgentEvent


class PriorityMailbox:
    """Priority queue + resume lock for the reactive drain loop."""

    def __init__(self) -> None:
        self._queue: asyncio.PriorityQueue[tuple[tuple[int, float, str], AgentEvent]] = (
            asyncio.PriorityQueue()
        )
        self._resume_lock = asyncio.Lock()
        self._closed = False

    @property
    def resume_lock(self) -> asyncio.Lock:
        return self._resume_lock

    def empty(self) -> bool:
        return self._queue.empty()

    async def put(self, event: AgentEvent) -> None:
        if self._closed:
            raise RuntimeError("mailbox is closed")
        await self._queue.put((event.sort_key(), event))

    def put_nowait(self, event: AgentEvent) -> None:
        if self._closed:
            raise RuntimeError("mailbox is closed")
        self._queue.put_nowait((event.sort_key(), event))

    async def get(self) -> AgentEvent:
        _key, event = await self._queue.get()
        return event

    async def get_nowait_or_wait(self, timeout: float | None = None) -> AgentEvent | None:
        try:
            if timeout is None:
                return await self.get()
            return await asyncio.wait_for(self.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def drain_available(self) -> list[AgentEvent]:
        events: list[AgentEvent] = []
        while not self._queue.empty():
            try:
                _key, event = self._queue.get_nowait()
                events.append(event)
            except asyncio.QueueEmpty:
                break
        return events

    async def stream(self) -> AsyncIterator[AgentEvent]:
        while not self._closed:
            event = await self.get()
            yield event

    def close(self) -> None:
        self._closed = True
