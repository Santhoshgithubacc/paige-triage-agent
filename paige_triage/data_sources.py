"""Grounded data sources for Paige connectors.

TOY MODE (default): read small CSVs under data/toy/.
PRODUCTION MODE: same methods call real APIs — replace the bodies marked
PRODUCTION when you go live, then delete the TOY branches.

Switch with env:
  PAIGE_DATA_MODE=toy          # default — CSV simulation
  PAIGE_DATA_MODE=production   # real connector calls
  PAIGE_TOY_DATA_DIR=/path     # optional override of CSV folder
"""

from __future__ import annotations

import csv
import os
from functools import lru_cache
from pathlib import Path
from typing import Any


def data_mode() -> str:
    """Return 'toy' or 'production'."""
    mode = os.getenv("PAIGE_DATA_MODE", "toy").strip().lower()
    if mode not in {"toy", "production"}:
        return "toy"
    return mode


def toy_data_dir() -> Path:
    override = os.getenv("PAIGE_TOY_DATA_DIR")
    if override:
        return Path(override)
    # repo_root/data/toy
    return Path(__file__).resolve().parent.parent / "data" / "toy"


def _read_csv(name: str) -> list[dict[str, str]]:
    path = toy_data_dir() / name
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# Shared facades — keep TOY + PRODUCTION in the same function.
# After cutover: delete the `if data_mode() == "toy"` branches.
# ---------------------------------------------------------------------------


def load_logs(service: str | None = None, query: str | None = None) -> list[dict[str, Any]]:
    if data_mode() == "toy":
        # TOY: CSV stand-in for Datadog / CloudWatch / Grafana Loki
        rows = _read_csv("logs.csv")
        out: list[dict[str, Any]] = []
        q = (query or "").lower()
        svc = (service or "").lower()
        for row in rows:
            text = f"{row.get('message', '')} {row.get('service', '')}".lower()
            if svc and svc not in str(row.get("service", "")).lower():
                continue
            if q:
                tokens = [t for t in q.replace(":", " ").replace(" OR ", " ").split() if len(t) > 2]
                if tokens and not any(t.lower() in text for t in tokens):
                    continue
            out.append(
                {
                    "service": row.get("service"),
                    "level": row.get("level"),
                    "message": row.get("message"),
                    "timestamp": row.get("timestamp"),
                    "source": row.get("source", "csv"),
                }
            )
        return out

    # PRODUCTION: replace with real log provider SDK / HTTP call
    # Example:
    #   return await datadog_client.search_logs(service=service, query=query)
    raise NotImplementedError(
        "PRODUCTION logs connector not wired — set PAIGE_DATA_MODE=toy or implement load_logs()"
    )


def load_metrics(service: str | None = None, metric: str | None = None) -> list[dict[str, Any]]:
    if data_mode() == "toy":
        # TOY: CSV stand-in for Datadog / Prometheus / CloudWatch metrics
        rows = _read_csv("metrics.csv")
        series: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in rows:
            svc = row.get("service", "")
            name = row.get("metric", "")
            if service and service.lower() not in svc.lower():
                continue
            if metric and metric.lower() not in name.lower():
                continue
            key = (svc, name)
            series.setdefault(key, []).append(
                {
                    "t": row.get("timestamp"),
                    "v": float(row.get("value") or 0),
                }
            )
        return [
            {
                "service": svc,
                "metric": name,
                "points": points,
                "source": "csv",
            }
            for (svc, name), points in series.items()
        ]

    # PRODUCTION: replace with metrics API
    raise NotImplementedError(
        "PRODUCTION metrics connector not wired — set PAIGE_DATA_MODE=toy or implement load_metrics()"
    )


def load_changes(service_id: str | None = None) -> list[dict[str, Any]]:
    if data_mode() == "toy":
        # TOY: CSV stand-in for PagerDuty Change Events / deploy feed
        rows = _read_csv("changes.csv")
        out = []
        for row in rows:
            if service_id and row.get("service_id") not in (None, "", service_id):
                continue
            out.append(dict(row))
        return out

    # PRODUCTION: PagerDuty Change Events API / deploy webhooks
    raise NotImplementedError(
        "PRODUCTION changes connector not wired — set PAIGE_DATA_MODE=toy or implement load_changes()"
    )


def load_runbooks(service_id: str | None = None, key: str | None = None) -> list[dict[str, Any]]:
    if data_mode() == "toy":
        # TOY: CSV stand-in for Confluence / GitHub runbooks
        rows = _read_csv("runbooks.csv")
        out = []
        for row in rows:
            if service_id and row.get("service_id") not in (None, "", service_id):
                continue
            if key:
                blob = f"{row.get('title', '')} {row.get('source_url', '')}".lower()
                if key.lower() not in blob:
                    continue
            out.append(
                {
                    "id": row.get("runbook_id"),
                    "service_id": row.get("service_id"),
                    "title": row.get("title"),
                    "source_url": row.get("source_url"),
                    "body": row.get("body", "").replace("\\n", "\n"),
                    "tags": [t for t in str(row.get("tags", "")).split(";") if t],
                    "source": "csv",
                }
            )
        return out

    # PRODUCTION: Confluence / GitHub Contents API
    raise NotImplementedError(
        "PRODUCTION runbooks connector not wired — set PAIGE_DATA_MODE=toy or implement load_runbooks()"
    )


def load_commits(service: str | None = None, query: str | None = None) -> list[dict[str, Any]]:
    if data_mode() == "toy":
        # TOY: CSV stand-in for GitHub commits / PRs
        rows = _read_csv("commits.csv")
        q = (query or "").lower()
        out = []
        for row in rows:
            if service and service.lower() not in str(row.get("service", "")).lower():
                continue
            text = f"{row.get('message', '')} {row.get('files', '')}".lower()
            if q and q not in text and q not in str(row.get("service", "")).lower():
                continue
            out.append(dict(row))
        return out

    # PRODUCTION: GitHub Search / Commits API
    raise NotImplementedError(
        "PRODUCTION code connector not wired — set PAIGE_DATA_MODE=toy or implement load_commits()"
    )


def load_dependencies(service_id: str | None = None) -> dict[str, list[str]]:
    if data_mode() == "toy":
        # TOY: CSV stand-in for topology / service graph
        rows = _read_csv("dependencies.csv")
        graph: dict[str, list[str]] = {}
        for row in rows:
            sid = row.get("service_id", "")
            if service_id and sid != service_id:
                continue
            graph.setdefault(sid, []).append(row.get("depends_on", ""))
        return graph

    # PRODUCTION: topology service / Backstage / static analysis graph API
    raise NotImplementedError(
        "PRODUCTION topology not wired — set PAIGE_DATA_MODE=toy or implement load_dependencies()"
    )


def load_incidents() -> list[dict[str, Any]]:
    if data_mode() == "toy":
        return [dict(r) for r in _read_csv("incidents.csv")]
    # PRODUCTION: PagerDuty Incidents API
    raise NotImplementedError(
        "PRODUCTION incidents not wired — set PAIGE_DATA_MODE=toy or implement load_incidents()"
    )


def load_alerts(incident_id: str | None = None) -> list[dict[str, Any]]:
    if data_mode() == "toy":
        rows = _read_csv("alerts.csv")
        if incident_id:
            rows = [r for r in rows if r.get("incident_id") == incident_id]
        return [dict(r) for r in rows]
    # PRODUCTION: PagerDuty Alerts API
    raise NotImplementedError(
        "PRODUCTION alerts not wired — set PAIGE_DATA_MODE=toy or implement load_alerts()"
    )


@lru_cache(maxsize=1)
def grounded_source_catalog() -> list[dict[str, str]]:
    """What Paige can ground against in the current mode."""
    mode = data_mode()
    if mode == "toy":
        root = toy_data_dir()
        return [
            {"name": "incidents", "kind": "csv", "path": str(root / "incidents.csv")},
            {"name": "alerts", "kind": "csv", "path": str(root / "alerts.csv")},
            {"name": "logs", "kind": "csv", "path": str(root / "logs.csv")},
            {"name": "metrics", "kind": "csv", "path": str(root / "metrics.csv")},
            {"name": "changes", "kind": "csv", "path": str(root / "changes.csv")},
            {"name": "runbooks", "kind": "csv", "path": str(root / "runbooks.csv")},
            {"name": "commits", "kind": "csv", "path": str(root / "commits.csv")},
            {"name": "dependencies", "kind": "csv", "path": str(root / "dependencies.csv")},
        ]
    return [
        {"name": "pagerduty", "kind": "api", "path": "PRODUCTION"},
        {"name": "logs", "kind": "api", "path": "PRODUCTION"},
        {"name": "metrics", "kind": "api", "path": "PRODUCTION"},
        {"name": "runbooks", "kind": "api", "path": "PRODUCTION"},
        {"name": "github", "kind": "api", "path": "PRODUCTION"},
    ]
