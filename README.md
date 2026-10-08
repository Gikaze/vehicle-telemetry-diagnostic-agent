# Vehicle Telemetry & Diagnostic Agent

**Fleet telemetry on AWS, and an AI agent that tells a mechanic what is wrong — and cites its sources.**

<!-- TODO(M7): link to the 2-minute demo video (self-hosted, no view counter) -->
[Demo video](#) · [How it works](#architecture) · [Design decisions](#key-technical-decisions) · [Use cases](#what-it-does)

<!-- TODO(M7): 20-second demo GIF — an anomaly appears, a question is asked, a sourced answer comes back -->

<!-- TODO(M0): CI badge once the workflow exists -->
![License](https://img.shields.io/badge/license-MIT-blue)

## At a glance

<!-- TODO(M6–M7): real, measured values only — even if below target. Details in docs/evaluation/ and docs/COST.md -->

| Detects | Answers | Costs |
| --- | --- | --- |
| 3 fault types · false alarms over 24 h of fleet data: *to be measured* | p95 latency on a laptop GPU: *to be measured* · every answer cites its sources | one real AWS demo: *target < €5* · monthly at 10,000 vehicles: *see COST.md* |

## What it does

| Who | Asks | Gets |
| --- | --- | --- |
| Fleet manager | "Why did VAN-042 raise a brake alert, and what should we check?" | The cause, the evidence from the vehicle's data, the procedure to follow |
| Workshop technician | "Overheating engine — do I take it off the road?" | A sensor fault spotted by physical plausibility, and a clear "not before repair" |
| Quality engineer | "Is any fault unusually frequent on one production batch?" | Per-vehicle rates by batch, computed in code, with a caution on correlation vs cause |

## Why it matters

Connected vehicles stream thousands of signals, yet first-level diagnosis still means querying raw data and
searching documentation by hand. This project ingests fleet telemetry in real time on AWS, flags anomalies with
deterministic rules, and lets a local AI agent explain them in plain language — with data that never has to
leave the organisation. The telemetry comes from a chassis-focused simulator (braking, vehicle dynamics,
steering, tyres) whose signals are physically coupled and plausibility-tested — see
[docs/simulator-spec.md](docs/simulator-spec.md).

---

## Architecture

<!-- TODO(M2): Mermaid diagram, kept in sync with docs/architecture.md -->

The telemetry pipeline lives in AWS (LocalStack during development, a real account for demos only).
The agent, its vector store and the LLM run on the edge — here, a developer laptop — and read AWS data through a tool.

## Quick start

<!-- TODO(M2): verified three-command quick start -->

```bash
git clone https://github.com/<user>/vehicle-telemetry-diagnostic-agent.git
cd vehicle-telemetry-diagnostic-agent
make setup && make up
```

Full prerequisites and troubleshooting: [docs/SETUP.md](docs/SETUP.md). How local emulation and real AWS fit
together, and what is only validated on AWS: [docs/local-vs-aws.md](docs/local-vs-aws.md).

## Key technical decisions

| Decision | Choice | Record |
| --- | --- | --- |
| Local AWS emulation | LocalStack (Hobby) driven by `lstk` | [ADR-001](docs/decisions/001-local-aws-emulation.md) |
| Telemetry storage | DynamoDB vs RDS | ADR-002 (M2) |
| Vector store | Qdrant vs FAISS | ADR-003 (M4) |
| Agent orchestration | LangChain vs LangGraph | ADR-004 (M5) |
| Edge/cloud split | Local inference, cloud ingestion | ADR-005 (M5) |
| LLM | Local model vs API, by measurement | ADR-006 (M6) |

## Project details

- **Requirements and their status:** [docs/requirements.md](docs/requirements.md)
- **Architecture:** docs/architecture.md (M2)
- **Cost analysis (TCO at 100 / 10k / 100k vehicles):** docs/COST.md (M2)
- **Evaluation and benchmark:** docs/evaluation/ (M6)
- **Runbook:** docs/runbook.md (M2)

## Repository structure

```text
simulator/       Chassis-focused fleet telemetry simulator
pipeline/        Lambda processor and anomaly rules
agent/           RAG diagnostic agent and Streamlit UI
infra/           Terraform modules and environments (local, aws-demo)
observability/   Prometheus and Grafana configuration
evaluation/      Reference datasets and model benchmark
tests/           Unit and integration tests
docs/            Requirements, ADRs, setup, cost, runbook
```

## Roadmap

- **v1.0** — pipeline, anomaly detection, diagnostic agent, evaluation, real AWS demo
- **Next** — expose the agent's tools through an MCP server; read-only remote diagnostics (UDS ReadDTCInformation, ReadDataByIdentifier); predictive maintenance (a statistical model forecasts, the agent explains)

## How this project was built

Architecture, requirements, decisions and code review are mine. Implementation was assisted by
[Claude Code](https://docs.claude.com), working under the conventions in [CLAUDE.md](CLAUDE.md), with every change
going through tests, linting, security scans and a pull request. Commits produced with AI assistance carry a
`Co-Authored-By` trailer.

## About the author

<!-- TODO(M0): one or two lines + LinkedIn link -->

## License

MIT — see [LICENSE](LICENSE).
