"""Paige-style SRE triage agent.

Faithful implementation of PagerDuty Paige (SRE Agent) architecture:
durable supervisor, stateless hypothesis sub-agents, in-process priority
mailbox, precomputed incident context, service-scoped memory, connectors,
and progressively loaded skills.
"""

__version__ = "1.0.0"
