"""Built-in connectors mirroring Paige capabilities.

Tools include get_incident_details, list_incidents, add_incident_note,
get_service_details, get_related_services, search_logs, query_metrics,
search_code, fetch_runbook.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from paige_triage.connectors.base import Connector, ToolResult, ToolSpec
from paige_triage import data_sources
from paige_triage.models.incident import IncidentContext


class PagerDutyConnector(Connector):
    name = "pagerduty"
    description = "PagerDuty incident / service / notes tools"

    def __init__(
        self,
        context: IncidentContext | None = None,
        allowed_tools: set[str] | None = None,
    ) -> None:
        super().__init__(allowed_tools)
        self.context = context
        self._notes: list[dict[str, Any]] = []

    def bind_context(self, context: IncidentContext) -> None:
        self.context = context

    def list_tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="get_incident_details",
                description="Look up incident by id (notes, changes, status).",
                connector=self.name,
            ),
            ToolSpec(
                name="list_incidents",
                description="Query incidents by service / filters.",
                connector=self.name,
            ),
            ToolSpec(
                name="add_incident_note",
                description="Add a note to the current incident.",
                connector=self.name,
            ),
            ToolSpec(
                name="get_service_details",
                description="Retrieve service info by id or name.",
                connector=self.name,
            ),
            ToolSpec(
                name="get_related_services",
                description="Upstream and downstream service dependencies.",
                connector=self.name,
            ),
        ]

    async def _invoke(self, tool_name: str, **kwargs: Any) -> ToolResult:
        ctx = self.context
        if tool_name == "get_incident_details":
            if not ctx:
                return ToolResult(ok=False, tool=tool_name, error="no context bound")
            data = {
                "id": ctx.incident.id,
                "title": ctx.incident.title,
                "status": ctx.incident.status.value,
                "service": ctx.incident.service.model_dump(),
                "notes": [n.model_dump(mode="json") for n in ctx.notes] + self._notes,
                "changes": [c.model_dump(mode="json") for c in ctx.change_events],
            }
            return ToolResult(ok=True, tool=tool_name, data=data)
        if tool_name == "list_incidents":
            past = [p.model_dump(mode="json") for p in (ctx.past_incidents if ctx else [])]
            return ToolResult(ok=True, tool=tool_name, data={"incidents": past})
        if tool_name == "add_incident_note":
            body = kwargs.get("body", "")
            author = kwargs.get("author", "paige")
            note = {
                "author": author,
                "body": body,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            self._notes.append(note)
            return ToolResult(ok=True, tool=tool_name, data=note, quote=body)
        if tool_name == "get_service_details":
            if not ctx:
                return ToolResult(ok=False, tool=tool_name, error="no context bound")
            return ToolResult(
                ok=True,
                tool=tool_name,
                data=ctx.incident.service.model_dump(),
            )
        if tool_name == "get_related_services":
            deps = ctx.dependency_hints if ctx else []
            return ToolResult(ok=True, tool=tool_name, data={"dependencies": deps})
        return ToolResult(ok=False, tool=tool_name, error="unknown tool")


class LogsConnector(Connector):
    name = "logs"
    description = "Observability log search (Datadog/Grafana/CloudWatch-style)"

    def __init__(
        self,
        corpus: list[dict[str, Any]] | None = None,
        allowed_tools: set[str] | None = None,
    ) -> None:
        super().__init__(allowed_tools)
        self.corpus = corpus or []

    def list_tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="search_logs",
                description="Scoped log query when provider/target/window are clear.",
                connector=self.name,
                parameters={"query": "str", "service": "str", "window_minutes": "int"},
            )
        ]

    async def _invoke(self, tool_name: str, **kwargs: Any) -> ToolResult:
        if tool_name != "search_logs":
            return ToolResult(ok=False, tool=tool_name, error="unknown tool")
        query = str(kwargs.get("query", "")).lower()
        service = str(kwargs.get("service", "")).lower()
        # Prefer shared data_sources (CSV toy or PRODUCTION API) — same file switch.
        try:
            hits = data_sources.load_logs(service=service or None, query=query or None)
        except NotImplementedError as exc:
            return ToolResult(ok=False, tool=tool_name, error=str(exc))
        if not hits and self.corpus:
            # Legacy in-memory fallback (tests)
            for row in self.corpus:
                text = f"{row.get('message', '')} {row.get('service', '')}".lower()
                if service and service not in str(row.get("service", "")).lower():
                    continue
                if query and not any(
                    tok in text for tok in query.replace(":", " ").split() if len(tok) > 3
                ):
                    continue
                hits.append(row)
        quote = hits[0]["message"] if hits else None
        return ToolResult(
            ok=True,
            tool=tool_name,
            data={
                "hits": hits[:20],
                "query": query,
                "count": len(hits),
                "grounded_source": data_sources.data_mode(),
            },
            quote=quote,
        )


class MetricsConnector(Connector):
    name = "metrics"
    description = "Metrics query connector"

    def __init__(
        self,
        series: list[dict[str, Any]] | None = None,
        allowed_tools: set[str] | None = None,
    ) -> None:
        super().__init__(allowed_tools)
        self.series = series or []

    def list_tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="query_metrics",
                description="Query a metric timeseries for a service/component.",
                connector=self.name,
            )
        ]

    async def _invoke(self, tool_name: str, **kwargs: Any) -> ToolResult:
        metric = str(kwargs.get("metric", "error_rate")).lower()
        service = str(kwargs.get("service", "")).lower()
        try:
            matches = data_sources.load_metrics(service=service or None, metric=metric or None)
        except NotImplementedError as exc:
            return ToolResult(ok=False, tool=tool_name, error=str(exc))
        if not matches and self.series:
            matches = [
                s
                for s in self.series
                if metric in str(s.get("metric", "")).lower()
                and (not service or service in str(s.get("service", "")).lower())
            ]
        if not matches:
            now = datetime.now(timezone.utc)
            matches = [
                {
                    "metric": metric,
                    "service": service or "unknown",
                    "points": [
                        {"t": (now - timedelta(minutes=i)).isoformat(), "v": 0.02}
                        for i in range(10, 0, -1)
                    ]
                    + [{"t": now.isoformat(), "v": 0.87}],
                    "source": "synthetic-fallback",
                }
            ]
        return ToolResult(
            ok=True,
            tool=tool_name,
            data={"series": matches[:5], "grounded_source": data_sources.data_mode()},
        )


class RunbookConnector(Connector):
    name = "runbooks"
    description = "Fetch runbooks from Confluence/GitHub-style sources"

    def __init__(
        self,
        docs: dict[str, str] | None = None,
        allowed_tools: set[str] | None = None,
    ) -> None:
        super().__init__(allowed_tools)
        self.docs = docs or {}

    def list_tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="fetch_runbook",
                description="Fetch runbook body by url or title key.",
                connector=self.name,
            )
        ]

    async def _invoke(self, tool_name: str, **kwargs: Any) -> ToolResult:
        key = str(kwargs.get("url") or kwargs.get("title") or "")
        title = key
        body = self.docs.get(key)
        try:
            rows = data_sources.load_runbooks(key=key or None)
        except NotImplementedError as exc:
            return ToolResult(ok=False, tool=tool_name, error=str(exc))
        if rows:
            title = rows[0].get("title") or key
            body = rows[0].get("body")
            url = rows[0].get("source_url")
        else:
            url = None
            if body is None:
                for k, v in self.docs.items():
                    if key.lower() in k.lower():
                        body = v
                        title = k
                        break
        if body is None:
            return ToolResult(ok=False, tool=tool_name, error=f"runbook not found: {key}")
        return ToolResult(
            ok=True,
            tool=tool_name,
            data={
                "title": title,
                "body": str(body)[:4000],
                "source_url": url,
                "grounded_source": data_sources.data_mode(),
            },
            quote=str(body)[:400],
        )


class CodeConnector(Connector):
    name = "github"
    description = "Search code, PRs, commits"

    def __init__(
        self,
        commits: list[dict[str, Any]] | None = None,
        allowed_tools: set[str] | None = None,
    ) -> None:
        super().__init__(allowed_tools)
        self.commits = commits or []

    def list_tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="search_code",
                description="Search code, pull requests, and commits.",
                connector=self.name,
            )
        ]

    async def _invoke(self, tool_name: str, **kwargs: Any) -> ToolResult:
        q = str(kwargs.get("query", "")).lower()
        try:
            hits = data_sources.load_commits(service=q or None, query=q or None)
        except NotImplementedError as exc:
            return ToolResult(ok=False, tool=tool_name, error=str(exc))
        if not hits and self.commits:
            hits = [
                c
                for c in self.commits
                if not q
                or q in str(c.get("message", "")).lower()
                or q in str(c.get("files", "")).lower()
            ]
        return ToolResult(
            ok=True,
            tool=tool_name,
            data={"commits": hits[:10], "grounded_source": data_sources.data_mode()},
            quote=hits[0]["message"] if hits else None,
        )
