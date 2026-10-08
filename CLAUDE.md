# CLAUDE.md — Project conventions

Read this file fully before any task. It is the contract for how work is done in this repository.

## What this project is

**Intelligent Vehicle Telemetry & Diagnostic Agent.** A real-time vehicle telemetry pipeline on AWS
(Kinesis → Lambda → DynamoDB → CloudWatch, provisioned with Terraform) coupled with a local RAG
diagnostic agent (LangChain `create_agent`, Qdrant, Ollama) that answers natural-language questions
by combining vehicle data with technical documentation.

- Requirements (source of truth): `docs/requirements.md` (FR-xx functional, NFR-xx non-functional)
- Architecture decisions: `docs/decisions/` (ADR-xxx)
- Environment setup: `docs/SETUP.md`
- Local emulation vs real AWS (modes, limits, validation): `docs/local-vs-aws.md`
- Milestones: M0 Foundations → M7 Demo & v1.0 (GitHub Milestones)

## Non-negotiable rules

1. **AWS safety.** Always target LocalStack: use `lstk terraform` for Terraform and `lstk aws` or
   `--profile localstack` for the AWS CLI. Never use the `default` profile or any other profile with real credentials.
   Never run `terraform apply` or `terraform destroy` in `infra/envs/aws-demo/` — the human runs those.
   Never read, print or modify files in `~/.aws/`.
   These rules are also enforced in `.claude/settings.json`: plain `terraform apply`/`destroy`, force pushes,
   pushes to `main`, `gh pr merge` and reads of `~/.aws/` and `.env` are denied; `git push`, tags, releases,
   `gh api` and `aws` always ask. Do not edit `.claude/settings.json` without an explicit request from the human.
2. **No secrets in the repository.** Secrets live in `.env` (git-ignored). Only `.env.example` is committed,
   with placeholder values. Never bypass hooks (`--no-verify` is forbidden).
3. **No proprietary data.** The documentation corpus is synthetic, written for a fictional manufacturer
   ("Nordwind Automotive"). Never introduce names, codes or documents from real employers or suppliers.
4. **English everywhere** in the repository: code, comments, docs, commits, PRs, issues.
5. **Never push to `main`.** All work goes through a branch and a pull request.

## Environment

| Item | Value |
| --- | --- |
| OS | WSL2 Ubuntu 24.04 on Windows 11 |
| Python | 3.12, managed with `uv` (`uv sync`, `uv run`, `uv add`) |
| AWS emulator | LocalStack (Hobby plan) via `lstk`, endpoint `http://localhost:4566` |
| AWS region | `eu-central-1` everywhere, local included |
| S3 | path-style addressing only (`s3_use_path_style = true`); `*.localhost.localstack.cloud` does not resolve on this network |
| LLM | Ollama on the Windows host, reachable at `http://localhost:11434` (WSL mirrored networking) |
| Containers | Docker Desktop with WSL integration; services defined in `compose.yaml` with profiles |

Do not use `localhost.localstack.cloud` hostnames anywhere — always `localhost` or `127.0.0.1`.

## Commands

Defined in the `Makefile` (created in M0; keep this list in sync with it):

```bash
make setup      # uv sync + pre-commit install
make lint       # ruff, terraform fmt -check, tflint, hadolint
make test       # unit tests with coverage
make test-int   # integration tests against LocalStack (requires `lstk` running)
make up         # start LocalStack and local services
make down       # stop everything
```

## Repository layout

```text
simulator/       Fleet telemetry simulator (FR-01..03, FR-16..18); spec in docs/simulator-spec.md
pipeline/        Lambda processor and anomaly rules (FR-04..07, FR-19)
agent/           RAG diagnostic agent, tools, ingestion, Streamlit UI (FR-08..15)
agent/corpus/    Synthetic technical documentation (Markdown)
infra/modules/   Terraform modules (kinesis, lambda, dynamodb, iot, observability)
infra/envs/      local/ (LocalStack) and aws-demo/ (real AWS, human-operated)
observability/   Prometheus config, Grafana dashboards (JSON)
evaluation/      Reference Q&A datasets and model benchmark
tests/           unit/ and integration/
docs/            requirements, decisions (ADRs), architecture, setup, cost, runbook, evaluation
```

## How a task is done

1. **Start from a GitHub issue.** It lists the requirements it covers (FR-xx / NFR-xx).
2. **Propose a plan first** and wait for approval before writing code.
3. **Branch** named `type/mN-short-topic`, e.g. `feat/m1-fleet-simulator`, `infra/m2-kinesis-lambda`.
4. **Implement with tests.** Every requirement covered gets at least one test that verifies its acceptance criterion.
5. **Run `make lint` and `make test`** (and `make test-int` when infrastructure is touched) until green.
6. **Update docs in the same branch:** requirement status in `docs/requirements.md`, README sections, ADR if a decision was made.
7. **Commit** with Conventional Commits (below), small and atomic; each commit builds and passes tests.
8. **Open the PR with `gh pr create`**, using the PR template, linking the issue (`Closes #N`).
   PRs are squash-merged by the human after review.

At the end of each milestone, add one screenshot or short GIF of what is new to `docs/assets/` and reference it in
the release notes. The README's first screen (tagline, links, demo GIF, "At a glance" table) only shows real,
measured values; never invent figures.

## Commit messages

Format: `type(scope): imperative summary` (max 72 chars), blank line, body explaining *why*, then `Refs: FR-xx`.

| Type | Use | Scopes |
| --- | --- | --- |
| `feat` | New behaviour | simulator, pipeline, agent, rag, ui |
| `fix` | Bug fix | same as feat |
| `infra` | Terraform, Docker, Compose | kinesis, lambda, dynamodb, iot, compose |
| `test` | Tests only | unit, integration, eval |
| `docs` | Documentation, ADRs | readme, adr, setup, cost, requirements |
| `ci` | GitHub Actions, hooks | ci, pre-commit |
| `refactor`, `chore` | Restructuring, maintenance | — |

Keep the `Co-Authored-By` trailer that Claude Code adds: AI assistance is disclosed openly in this project.

## Coding conventions

**Python**
- Type hints everywhere; `ruff` for lint and format; `pytest` for tests (`tests/unit`, `tests/integration`).
- Small, pure functions for domain logic (simulation, anomaly rules) so they are unit-testable without AWS.
- AWS clients receive the endpoint from configuration (`AWS_ENDPOINT_URL`), never hard-coded.
- Configuration through environment variables with documented defaults in `.env.example`.
- New dependencies require a one-line justification in the PR description; add them with `uv add`.

**Terraform**
- Reusable modules in `infra/modules/`, thin environments in `infra/envs/`.
- Pin Terraform (`required_version`) and provider versions; commit `.terraform.lock.hcl`.
- One IAM role per Lambda, scoped to its own resources (least privilege, NFR-04). LocalStack does not enforce IAM:
  validate every policy with Checkov and keep it minimal; it is only truly tested on real AWS.
- Kinesis-triggered Lambdas use partial batch failure reporting, capped retries and an SQS dead-letter queue (FR-19).
- IoT Core resources (FR-20) cannot run on LocalStack Hobby: keep them in their own module, validated statically.
- Never commit certificates or private keys.
- `terraform fmt` and `tflint` clean before commit.

**Docker**
- Multi-stage builds; final stage distroless, running as non-root (NFR-05).
- Explicit image tags, never `latest`.

**Simulator**
- Follow `docs/simulator-spec.md`. The signal catalogue (`simulator/catalogue.yaml`) is the single source of truth;
  the JSON schema is generated from it, never edited by hand.
- Keep models low-order (kinematics, single-track model, first-order thermal). No new signals or fault scenarios
  beyond the spec without an issue.
- Physics is deterministic for a given seed; all randomness comes from one seeded generator.
- Every coupling in the spec has a property-based invariant test (Hypothesis) (NFR-15).
- Chassis fault codes are fictional Nordwind codes; only standard OBD-II P-codes may be real.
- Fault codes use a subset of the UDS DTC status bits (FR-18); do not implement UDS services in this project.

**Agent**
- Agent tools live in a registry. Each tool declares metadata: `name`, `description` and `read_only` (bool).
  All tools in this project are read-only. New tools are added by registering them, without changing the agent.
- Numbers are computed in code, never by the LLM: counts, rates and aggregations come from tested tool
  functions (e.g. fleet anomaly statistics, FR-15); the LLM only interprets them.
- The LLM provider and model are configuration (`LLM_PROVIDER`, `LLM_MODEL`), never hard-coded (NFR-14).
- Every agent answer cites its sources (FR-12).

## Architecture Decision Records

- Write an ADR (copy `docs/decisions/000-template.md`) when a choice affects architecture, cost,
  security or a technology selection. Number sequentially in order of decision.
- An accepted ADR is never rewritten. To change a decision, write a new ADR that supersedes it
  and set the old one's status to `Superseded by ADR-xxx`.
- Commit the ADR in the same PR that implements the decision.

## What not to do

- Do not add features outside the current issue; propose them as a new issue labelled `v2`.
- Do not disable, skip or weaken tests or linters to make CI pass.
- Do not commit generated files, local state (`*.tfstate`, `.terraform/`), `.env`, or model weights.
- Do not leave TODOs without a linked issue.
