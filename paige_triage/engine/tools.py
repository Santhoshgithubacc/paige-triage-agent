"""LangChain tools wrapping Paige connectors — used by LLM sub-agents."""

from __future__ import annotations

from typing import Any, Callable, Awaitable

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from paige_triage.connectors.registry import ConnectorRegistry


class SearchLogsArgs(BaseModel):
    query: str = Field(description="Log search query / keywords")
    service: str = Field(default="", description="Service name filter")
    window_minutes: int = Field(default=30, description="Lookback window")


class QueryMetricsArgs(BaseModel):
    metric: str = Field(description="Metric name e.g. consumer_lag, error_rate, dns_failure_rate")
    service: str = Field(default="", description="Service name filter")


class FetchRunbookArgs(BaseModel):
    url: str = Field(default="", description="Runbook URL if known")
    title: str = Field(default="", description="Runbook title / keyword")


class SearchCodeArgs(BaseModel):
    query: str = Field(description="Commit / code search query")


class GetIncidentArgs(BaseModel):
    unused: str = Field(default="", description="Unused placeholder")


def build_investigation_tools(
    connectors: ConnectorRegistry,
    *,
    default_service: str = "",
) -> list[StructuredTool]:
    async def search_logs(query: str, service: str = "", window_minutes: int = 30) -> str:
        result = await connectors.invoke(
            "search_logs",
            query=query,
            service=service or default_service,
            window_minutes=window_minutes,
        )
        if not result.ok:
            return f"ERROR: {result.error}"
        hits = result.data.get("hits") or []
        lines = [f"count={result.data.get('count', 0)} source={result.data.get('grounded_source')}"]
        for h in hits[:8]:
            lines.append(f"- [{h.get('level')}] {h.get('message')}")
        return "\n".join(lines) if lines else "no hits"

    async def query_metrics(metric: str, service: str = "") -> str:
        result = await connectors.invoke(
            "query_metrics", metric=metric, service=service or default_service
        )
        if not result.ok:
            return f"ERROR: {result.error}"
        series = result.data.get("series") or []
        if not series:
            return "no series"
        out = []
        for s in series[:3]:
            pts = s.get("points") or []
            last = pts[-1] if pts else {}
            out.append(f"{s.get('metric')} service={s.get('service')} last={last}")
        return "\n".join(out)

    async def fetch_runbook(url: str = "", title: str = "") -> str:
        result = await connectors.invoke("fetch_runbook", url=url, title=title or default_service)
        if not result.ok:
            return f"ERROR: {result.error}"
        body = str(result.data.get("body") or "")[:1500]
        return f"title={result.data.get('title')}\n{body}"

    async def search_code(query: str) -> str:
        result = await connectors.invoke("search_code", query=query or default_service)
        if not result.ok:
            return f"ERROR: {result.error}"
        commits = result.data.get("commits") or []
        return "\n".join(
            f"- {c.get('sha')}: {c.get('message')} ({c.get('files')})" for c in commits[:8]
        ) or "no commits"

    async def get_incident_details(unused: str = "") -> str:
        result = await connectors.invoke("get_incident_details")
        if not result.ok:
            return f"ERROR: {result.error}"
        return str(result.data)[:2000]

    async def get_related_services(unused: str = "") -> str:
        result = await connectors.invoke("get_related_services")
        if not result.ok:
            return f"ERROR: {result.error}"
        return str(result.data)

    return [
        StructuredTool.from_function(
            coroutine=search_logs,
            name="search_logs",
            description="Search observability logs for discriminative error lines.",
            args_schema=SearchLogsArgs,
        ),
        StructuredTool.from_function(
            coroutine=query_metrics,
            name="query_metrics",
            description="Query metrics timeseries for spikes (lag, error_rate, dns, latency).",
            args_schema=QueryMetricsArgs,
        ),
        StructuredTool.from_function(
            coroutine=fetch_runbook,
            name="fetch_runbook",
            description="Fetch institutional runbook / SOP text.",
            args_schema=FetchRunbookArgs,
        ),
        StructuredTool.from_function(
            coroutine=search_code,
            name="search_code",
            description="Search recent commits / code related to the failing service.",
            args_schema=SearchCodeArgs,
        ),
        StructuredTool.from_function(
            coroutine=get_incident_details,
            name="get_incident_details",
            description="Get current incident details, notes, and change events.",
            args_schema=GetIncidentArgs,
        ),
        StructuredTool.from_function(
            coroutine=get_related_services,
            name="get_related_services",
            description="List upstream/downstream service dependencies.",
            args_schema=GetIncidentArgs,
        ),
    ]
