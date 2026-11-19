# Paige Triage Agent

Production-style clone of **PagerDuty Paige (SRE Agent)** for incident triage and deep investigation.

## v2 engine (default API)

- **LangGraph** supervisor with real `interrupt()` / `Command(resume=…)`
- **LLM sub-agents** (Llama via Ollama) with tool calling — not scripted runners
- **SQLite checkpoints** (`.paige_checkpoints/langgraph.sqlite`) for durable recovery
- **Continuous re-planning** (`deep_mode`, `max_cycles`, optional `replan_interval_s`)
- **Chatbot UI** with prompt box at `/chat` or `/demo`
- Data plane stays **CSV toys** (`PAIGE_DATA_MODE=toy`) until you wire production APIs

It mirrors the production design PagerDuty described for Paige:

- Precomputed **Incident Context** (working set, not a summary)
- **Durable supervisor** + **stateless hypothesis sub-agents**
- **One process**, in-process **priority mailbox** (`asyncio` queue)
- **Reactive drain loop** with resume lock (parallel fan-out, concurrent fan-in)
- Identity convention: `task_id === thread_id`, children carry `parent_task_id`
- Service-scoped **memory** (observations, recollections, playbooks, runbooks)
- **Connectors / tools** with allow-lists
- **Progressive skills** (catalog first; load instructions only when relevant)
- Bias to **context over cleverness** — fetch logs/metrics only when discriminative

## Architecture

```text
Incident opens
    ↓
ContextAssembler → IncidentContext
    ↓
TriageSupervisor (checkpointed)
    ├── PriorityMailbox (USER=0 > SUB_AGENT=1)
    ├── plan → spawn HypothesisSubAgent × N (async)
    ├── accept_event ← drain loop + resume lock
    ├── route_event → handle_user_input | handle_sub_agent_result
    └── synthesize → recommendations + incident note
            ↓
ConnectorRegistry (pagerduty, logs, metrics, runbooks, github)
SkillRegistry (progressive load)
MemoryBundle (tenant/service scoped)
```

### Mapping to Paige

| Paige concept | This repo |
| --- | --- |
| SRE Agent / Paige | `TriageSupervisor` |
| Deep investigation sub-agents | `HypothesisSubAgent` |
| Incident Context | `IncidentContext` + `ContextAssembler` |
| Memory API artifacts | `MemoryBundle` |
| Connectors & tools | `paige_triage/connectors` |
| Skills | `paige_triage/skills` |
| Nudges / steering | `inject_user_input`, API `/messages` |
| Virtual responder on trigger | `POST /investigations` |

## Quick start

```bash
cd /home/sandy/paige-triage-agent
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Toy CSV mode + local Llama (Ollama llama3.2)
export PAIGE_DATA_MODE=toy
export PAIGE_LLM=llama
export PAIGE_LLM_MODEL=llama3.2:latest
export OLLAMA_BASE_URL=http://127.0.0.1:11434/v1

# Local API + live browser demo (inc001 / inc002 / inc003)
paige-triage api --port 8088
# open http://127.0.0.1:8088/demo
```

Demo incidents:

| ID | Scenario | Skill loaded |
| --- | --- | --- |
| `inc001` | Kafka consumer lag / poison pill | `kafka-consumer-lag` |
| `inc002` | DNS NXDOMAIN for fraud host | `dns-resolution` |
| `inc003` | 5xx after PricingEngine deploy | `bad-deploy` |


### Toy CSV → production cutover

All connector reads go through `paige_triage/data_sources.py`. Each function has:

- a **TOY** branch that reads `data/toy/*.csv`
- a **PRODUCTION** branch stub (`NotImplementedError`) for the real API

Switch with `PAIGE_DATA_MODE=production`, implement the PRODUCTION bodies, then delete the TOY branches and the `data/toy/` folder.

### API

```bash
curl -s -X POST localhost:8088/investigations | jq
TASK=$(curl -s -X POST localhost:8088/investigations | jq -r .task_id)
curl -s -X POST localhost:8088/investigations/$TASK/wait | jq .synthesis
curl -s -X POST localhost:8088/investigations/$TASK/messages \
  -H 'content-type: application/json' \
  -d '{"text":"skip hypothesis bad deploy"}'
```

Optional LLM: set `OPENAI_API_KEY` (and optionally `PAIGE_LLM_MODEL`). Default is `HeuristicLLM`, which stays evidence-grounded without network calls.

## Tests

```bash
pytest -q
```

## Design notes (unchanged from Paige)

1. **Single-agent failure modes** (context rot, instruction overload, sequential latency, no mid-run interactivity) pushed the design to multi-agent.
2. **Sub-agents are co-located** with the supervisor — IO-bound work does not justify distributed brokers between agents.
3. **Only the supervisor is durable.** Mailbox contents are not checkpointed; on recovery, re-spawn in-flight sub-agents.
4. **User input is a first-class event** and jumps the priority queue.
5. **Evidence is typed** direct vs circumstantial; rejected alternatives are retained.

## Layout

```text
paige_triage/
  models/          incident, memory, hypothesis, events
  context/         precomputed Incident Context assembler
  connectors/      tools + allow-lists
  skills/          progressive skill loading
  agents/          supervisor + sub-agents
  mailbox.py       priority mailbox + resume lock
  checkpoint.py    durable supervisor store
  identity.py      task_id === thread_id
  llm.py           HeuristicLLM / OpenAI
  api.py           FastAPI control plane
  cli.py           Typer CLI
  demo_data.py     checkout-api consumer-lag scenario
```

## Sources

Architecture follows PagerDuty public material on Paige / SRE Agent (deep investigation, context-over-cleverness, connectors & skills). This repository implements that shape as a runnable local system; it is not affiliated with PagerDuty.
