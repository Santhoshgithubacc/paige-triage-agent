"""Sample memory / fallbacks for Paige demos across inc001–inc003."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from paige_triage.context.assembler import normalize_alert_signature
from paige_triage.models.incident import (
    AlertPayload,
    ChangeEvent,
    Incident,
    RelatedIncident,
    ServiceRef,
    Urgency,
)
from paige_triage.models.memory import (
    MemoryBundle,
    Observation,
    Playbook,
    Recollection,
    RunbookDoc,
    ServiceProfile,
)

NOW = datetime.now(timezone.utc)


def sample_service() -> ServiceRef:
    return ServiceRef(
        id="PXXXXXX",
        name="checkout-api",
        description="Checkout API / Kafka consumers",
        team="payments",
    )


def sample_incident() -> Incident:
    svc = sample_service()
    return Incident(
        id="inc001",
        number=5001,
        title="Consumer lag elevated on checkout-api",
        urgency=Urgency.HIGH,
        service=svc,
        created_at=NOW - timedelta(minutes=5),
        assignees=["oncall-alice"],
    )


def sample_alerts() -> list[AlertPayload]:
    svc = sample_service()
    summary = "Consumer lag elevated for group checkout-api-workers"
    details = {
        "error": "consumer_lag",
        "component": "checkout-api-workers",
        "metric": "kafka.consumer.lag",
        "runbook_url": "https://wiki.example/runbooks/checkout-consumer-lag",
    }
    return [
        AlertPayload(
            id="alert-inc001",
            summary=summary,
            severity="critical",
            source="datadog",
            component="checkout-api-workers",
            custom_details=details,
            received_at=NOW - timedelta(minutes=5),
            signature=normalize_alert_signature(svc.id, summary, details),
        )
    ]


def sample_memory() -> MemoryBundle:
    """Multi-service memory covering inc001 / inc002 / inc003 signatures."""
    checkout = "PXXXXXX"
    fraud = "PZZZZZZ"

    sig_lag = normalize_alert_signature(
        checkout,
        "Consumer lag elevated for group checkout-api-workers",
        {
            "error": "consumer_lag",
            "component": "checkout-api-workers",
            "metric": "kafka.consumer.lag",
        },
    )
    sig_dns = normalize_alert_signature(
        fraud,
        "DNS NXDOMAIN resolving payments-fraud.internal",
        {
            "error": "dns_nxdomain",
            "component": "fraud-gateway",
            "metric": "dns.failure_rate",
        },
    )
    sig_deploy = normalize_alert_signature(
        checkout,
        "checkout-api error rate > 5% after deploy v2.15.0",
        {
            "error": "http_5xx",
            "component": "checkout-api",
            "metric": "error_rate",
        },
    )

    return MemoryBundle(
        service_id=checkout,
        observations=[
            Observation(service_id=checkout, key="log_provider", value="datadog"),
            Observation(service_id=fraud, key="log_provider", value="datadog"),
            Observation(service_id=checkout, key="dashboard", value="https://dash.example/checkout"),
        ],
        recollections=[
            Recollection(
                service_id=checkout,
                alert_signature=sig_lag,
                title="Duplicate REST records race → Kafka poison pills",
                discriminative_query="DeserializationException OR duplicate order_id",
                error_fingerprint="consumer_lag|deserialization",
                helpful_steps=[
                    "Quarantine poison offsets",
                    "Run safe cleanup for duplicate REST records",
                ],
                failed_steps=["Restarting brokers did not help"],
                outcome="Resolved via cleanup playbook; constraint still pending",
            ),
            Recollection(
                service_id=fraud,
                alert_signature=sig_dns,
                title="Missing internal CNAME for fraud host",
                discriminative_query="NXDOMAIN payments-fraud.internal",
                error_fingerprint="dns_nxdomain",
                helpful_steps=["Restore CNAME", "Flush resolver cache"],
                failed_steps=["Restarting fraud-gateway pods alone"],
                outcome="DNS record restored",
            ),
            Recollection(
                service_id=checkout,
                alert_signature=sig_deploy,
                title="PricingEngine NPE after flag flip",
                discriminative_query="NullPointerException PricingEngine",
                error_fingerprint="http_5xx|pricing",
                helpful_steps=["Disable new_pricing_engine", "Rollback v2.15.0"],
                failed_steps=["Scaling pods without rollback"],
                outcome="Rolled back + flag off",
            ),
        ],
        playbooks=[
            Playbook(
                service_id=checkout,
                alert_signature=sig_lag,
                title="Checkout consumer lag — duplicate REST race",
                steps=[
                    "Confirm lag metric on checkout-api-workers",
                    "Search logs for DeserializationException",
                    "If duplicates via REST race, run cleanup SQL from runbook",
                    "Do not restart Kafka brokers without approval",
                ],
                evidence_quotes=[
                    "DeserializationException: duplicate order_id on topic checkout.orders",
                    "Rare race: duplicate records created through the REST API",
                ],
                preconditions=["Lag confined to checkout-api-workers group"],
                guardrails=["No broker restarts without human approval"],
            ),
            Playbook(
                service_id=fraud,
                alert_signature=sig_dns,
                title="Fraud gateway DNS NXDOMAIN",
                steps=[
                    "Verify NXDOMAIN in fraud-gateway logs",
                    "Inspect latest DNS zone change for payments-fraud.internal",
                    "Restore CNAME with network on-call approval",
                ],
                evidence_quotes=[
                    "DNS NXDOMAIN for payments-fraud.internal",
                    "java.net.UnknownHostException: payments-fraud.internal",
                ],
                guardrails=["Do not restart pods as first remediation"],
            ),
            Playbook(
                service_id=checkout,
                alert_signature=sig_deploy,
                title="Bad deploy — PricingEngine 5xx",
                steps=[
                    "Confirm deploy v2.15.0 aligns with error spike",
                    "Search logs for NullPointerException in PricingEngine",
                    "Disable feature flag new_pricing_engine",
                    "Rollback to previous version if 5xx continues",
                ],
                evidence_quotes=[
                    "NullPointerException in PricingEngine.calculate after v2.15.0",
                    "Feature flag new_pricing_engine=true evaluated",
                ],
                guardrails=["Human approval before DB migrations"],
            ),
        ],
        runbooks=[
            RunbookDoc(
                service_id=checkout,
                title="Checkout consumer lag",
                source_url="https://wiki.example/runbooks/checkout-consumer-lag",
                body="Pause consumer, skip poison offset, cleanup duplicates, resume.",
            ),
            RunbookDoc(
                service_id=fraud,
                title="DNS resolution failures",
                source_url="https://wiki.example/runbooks/dns",
                body="Verify NXDOMAIN, check zone push, restore record.",
            ),
            RunbookDoc(
                service_id=checkout,
                title="Bad deploy rollback",
                source_url="https://wiki.example/runbooks/bad-deploy",
                body="Confirm deploy, disable flag, rollback release.",
            ),
        ],
        profile=ServiceProfile(
            service_id=checkout,
            cloud_provider="aws",
            region="us-west-2",
            service_type="kafka-consumer",
            log_queries=["service:checkout-api status:error"],
        ),
    )


def sample_history() -> tuple[list[RelatedIncident], list[RelatedIncident], list[ChangeEvent]]:
    svc = sample_service()
    past = [
        RelatedIncident(
            id="inc-old-77",
            number=3910,
            title="Consumer lag from duplicate REST race",
            service_id=svc.id,
            similarity=0.91,
            resolution_summary="Cleanup playbook; poison offset skipped",
        )
    ]
    related = [
        RelatedIncident(
            id="inc-rel-9",
            number=4200,
            title="Elevated error rate on orders-api",
            service_id="PYYYYYY",
            similarity=0.4,
        )
    ]
    changes = [
        ChangeEvent(
            id="chg-1",
            summary="Deploy checkout-api v2.14.3 — consumer deserializer change",
            service_id=svc.id,
            actor="ci-bot",
            occurred_at=NOW - timedelta(minutes=25),
            url="https://github.com/example/checkout-api/releases/v2.14.3",
        )
    ]
    return past, related, changes


def sample_connectors_data() -> dict:
    return {
        "log_corpus": [],
        "metric_series": [],
        "runbook_docs": {},
        "commits": [],
        "dependency_graph": {},
    }
