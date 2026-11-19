"""Connector registry — Paige uses only allowed tools per connector."""

from __future__ import annotations

from typing import Any

from paige_triage.connectors.base import Connector, ToolResult, ToolSpec
from paige_triage.connectors.impl import (
    CodeConnector,
    LogsConnector,
    MetricsConnector,
    PagerDutyConnector,
    RunbookConnector,
)
from paige_triage.models.incident import IncidentContext


class ConnectorRegistry:
    def __init__(self) -> None:
        self._connectors: dict[str, Connector] = {}
        self._tool_index: dict[str, str] = {}

    def register(self, connector: Connector) -> None:
        self._connectors[connector.name] = connector
        for tool in connector.list_tools():
            if connector.is_allowed(tool.name):
                self._tool_index[tool.name] = connector.name

    def bind_context(self, context: IncidentContext) -> None:
        pd = self._connectors.get("pagerduty")
        if isinstance(pd, PagerDutyConnector):
            pd.bind_context(context)

    def list_tools(self) -> list[ToolSpec]:
        tools: list[ToolSpec] = []
        for connector in self._connectors.values():
            for tool in connector.list_tools():
                if connector.is_allowed(tool.name):
                    tools.append(tool)
        return tools

    async def invoke(self, tool_name: str, **kwargs: Any) -> ToolResult:
        connector_name = self._tool_index.get(tool_name)
        if not connector_name:
            return ToolResult(ok=False, tool=tool_name, error="unknown or disallowed tool")
        return await self._connectors[connector_name].invoke(tool_name, **kwargs)


def build_default_registry(
    *,
    context: IncidentContext | None = None,
    log_corpus: list[dict[str, Any]] | None = None,
    metric_series: list[dict[str, Any]] | None = None,
    runbook_docs: dict[str, str] | None = None,
    commits: list[dict[str, Any]] | None = None,
    allowed: dict[str, set[str] | None] | None = None,
) -> ConnectorRegistry:
    allowed = allowed or {}
    registry = ConnectorRegistry()
    registry.register(
        PagerDutyConnector(context=context, allowed_tools=allowed.get("pagerduty"))
    )
    registry.register(LogsConnector(corpus=log_corpus, allowed_tools=allowed.get("logs")))
    registry.register(
        MetricsConnector(series=metric_series, allowed_tools=allowed.get("metrics"))
    )
    registry.register(
        RunbookConnector(docs=runbook_docs, allowed_tools=allowed.get("runbooks"))
    )
    registry.register(CodeConnector(commits=commits, allowed_tools=allowed.get("github")))
    return registry
