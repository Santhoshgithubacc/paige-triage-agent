"""Connectors expose tools — individual actions Paige may use when allowed."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class ToolSpec(BaseModel):
    name: str
    description: str
    connector: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    ok: bool
    tool: str
    data: dict[str, Any] = Field(default_factory=dict)
    quote: str | None = None
    error: str | None = None


class Connector(ABC):
    name: str
    description: str = ""

    def __init__(self, allowed_tools: set[str] | None = None) -> None:
        self.allowed_tools = allowed_tools  # None = all tools allowed

    @abstractmethod
    def list_tools(self) -> list[ToolSpec]:
        ...

    def is_allowed(self, tool_name: str) -> bool:
        if self.allowed_tools is None:
            return True
        return tool_name in self.allowed_tools

    async def invoke(self, tool_name: str, **kwargs: Any) -> ToolResult:
        if not self.is_allowed(tool_name):
            return ToolResult(
                ok=False,
                tool=tool_name,
                error=f"tool '{tool_name}' not allowed on connector '{self.name}'",
            )
        return await self._invoke(tool_name, **kwargs)

    @abstractmethod
    async def _invoke(self, tool_name: str, **kwargs: Any) -> ToolResult:
        ...
