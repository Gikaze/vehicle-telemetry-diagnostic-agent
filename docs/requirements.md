# Requirements

Every requirement has an ID, a verifiable acceptance criterion, a milestone and a status.
Commits, pull requests and tests reference requirements by ID (`Refs: FR-06`), giving
requirement → code → test traceability.

**Status values:** `Planned` · `In progress` · `Verified` (an automated test or a documented check proves the criterion) · `Deferred` (moved to a later version, with reason)

## Functional requirements

| ID | Requirement | Acceptance criterion | Milestone | Status | Verified by |
| --- | --- | --- | --- | --- | --- |
| FR-01 | The simulator generates chassis-centred telemetry for a configurable fleet (braking, vehicle dynamics, steering, tyres, simplified powertrain, fault codes): physics at 10 Hz, aggregated periodic messages every 10 s (mean/min/max/last) plus immediate event messages. See `docs/simulator-spec.md` | 100 vehicles; every message validated against the generated schema; 10 s cadence respected (tests) | M1 | Planned | |
| FR-02 | The simulator injects scripted, reproducible anomalies | Scenario selectable by configuration; fixed random seed yields identical data on every run | M1 | Planned | |
| FR-03 | The simulator publishes to Kinesis locally and to IoT Core in the demo environment | Target selected by configuration, no code change | M1, M7 | Planned | |
| FR-04 | The pipeline ingests the Kinesis stream through a Lambda function | Integration test on LocalStack: published message is found in storage | M2 | Planned | |
| FR-05 | Measurements are stored in DynamoDB keyed by vehicle and timestamp | Latest N measurements of one vehicle retrieved in a single query | M2 | Planned | |
| FR-06 | The Lambda detects anomalies with rules (thresholds, persistence over a window) | Each FR-02 scenario produces exactly the expected anomaly; a normal stream produces none | M3 | Planned | |
| FR-07 | Anomalies are stored with their context and published as a CloudWatch metric | Anomaly table entry and metric visible after a scenario run | M3 | Planned | |
| FR-08 | A synthetic technical documentation corpus is indexed in Qdrant | Idempotent ingestion script; vehicle model and component metadata present | M4 | Planned | |
| FR-09 | Document search filters by vehicle model and component | A test shows a filter excludes out-of-scope documents | M4 | Planned | |
| FR-10 | The agent answers natural-language questions about a vehicle or an anomaly | Correct answers on the reference question set (threshold set in M6) | M5 | Planned | |
| FR-11 | The agent has three read-only tools: vehicle data lookup, documentation search, fleet anomaly statistics | Traces show the right tools called for each question type | M5 | Planned | |
| FR-12 | Every answer cites its sources (documents and measurements used) | No answer without references in the evaluation set | M5 | Planned | |
| FR-13 | The agent produces a structured diagnostic alert for an anomaly (finding, reference, recommendation) | Output validated against a schema | M5 | Planned | |
| FR-14 | A Streamlit interface lets users ask questions and see recent anomalies | Usable without a command line; shown in the demo video | M5 | Planned | |
| FR-15 | The agent provides a fleet-wide view: anomaly counts and per-vehicle rates by type, vehicle model and production batch over a period, computed by the tool, not by the LLM | The "batch defect" scenario surfaces the affected batch; per-vehicle rates exact (unit test); the base-rate evaluation question is answered correctly | M5 | Planned | |
| FR-16 | A YAML signal catalogue (name, unit, range, resolution, source ECU, invalid value) is the single source of truth for the telemetry schema | Schema generated from the catalogue; a test checks every emitted value against it | M1 | Planned | |
| FR-17 | Telemetry carries a per-vehicle sequence counter; the transport injects gaps, duplicates and out-of-order delivery; ingestion is idempotent | Duplicates never create a second measurement; gaps detectable from the counter (integration test) | M1, M2 | Planned | |
| FR-18 | Each fault code carries a subset of the UDS DTC status bits (testFailed, pendingDTC, confirmedDTC, warningIndicatorRequested) with a cycle-based debounce, managed by a simulator fault manager; codes and status are stored by the pipeline and returned by the vehicle data tool; no UDS services are simulated | A fault present in one cycle is pending, not confirmed; present in N cycles it becomes confirmed (unit tests). Last item of M1; moves to v1.1 if M1 exceeds its time box | M1 | Planned | |
| FR-19 | The pipeline handles processing errors without blocking the stream: partial batch failure reporting, capped retries, and a dead-letter queue (SQS) for records that keep failing | A malformed record ends in the dead-letter queue while the following records are processed (integration test on LocalStack; behaviour checked again in the AWS smoke test) | M2 | Planned | |
| FR-20 | In demo mode, each vehicle authenticates to IoT Core with its own X.509 certificate; the IoT policy restricts each vehicle to its own client ID and its own topic; no shared key | Publishing on another vehicle's topic is rejected (demo test); private keys never committed | M7 | Planned | |

## Non-functional requirements

| ID | Area | Requirement | Verification | Milestone | Status |
| --- | --- | --- | --- | --- | --- |
| NFR-01 | Cost | €0 AWS spend during development; each real demo under €5; monthly total under €35 | AWS budget alerts at €10/€20/€30; actual demo cost reported in COST.md | M7 | Planned |
| NFR-02 | Cost | LLM API spend capped at €10 per provider | Spend limit configured in each provider console | M6 | Planned |
| NFR-03 | Security | No secrets in the repository, history included | gitleaks in pre-commit and CI | M0 | Planned |
| NFR-04 | Security | Least privilege: one IAM role per Lambda, scoped to its own resources | Checkov without critical findings; policies reviewed in the relevant ADR | M2 | Planned |
| NFR-05 | Security | Distroless images, no critical CVE, running as non-root | hadolint + Trivy in CI | M1 | Planned |
| NFR-06 | Reproducibility | From `git clone` to a running local stack in under 15 minutes | SETUP.md replayed on a clean environment before v1.0 | M7 | Planned |
| NFR-07 | Reproducibility | All versions pinned (Python, dependencies, Terraform, providers, images) | `uv.lock`, `.terraform.lock.hcl`, explicit image tags | M0 | Planned |
| NFR-08 | Quality | Test coverage ≥ 80 % on domain logic (simulator, anomaly rules, agent tools) | pytest-cov report in CI | M3 | Planned |
| NFR-09 | Quality | Lint and formatting clean | ruff, `terraform fmt`, tflint in pre-commit and CI | M0 | Planned |
| NFR-10 | Observability | Every agent request traced end to end (LLM calls, retrieval, tools) | Traces visible in Phoenix | M6 | Planned |
| NFR-11 | Observability | Agent latency, throughput and error rate exposed as metrics | Grafana dashboard versioned in the repository | M6 | Planned |
| NFR-12 | Performance | Agent latency measured at p50 and p95 on the 8 GB GPU; p95 target set after the M6 benchmark | Benchmark report in `docs/evaluation/` | M6 | Planned |
| NFR-13 | Performance | Measurement visible in storage under 5 s after publication, locally | Timed integration test | M2 | Planned |
| NFR-14 | Portability | LLM provider and model switched by configuration, without code changes | Benchmark run on at least 3 models with the same code | M6 | Planned |
| NFR-15 | Fidelity | Simulated data is physically plausible and internally consistent (couplings between speed, wheel speeds, yaw rate, lateral acceleration, braking and temperatures) | Property-based invariant tests; 100 vehicles × 24 simulated hours yield zero anomalies; expert review of a simulated vehicle-day plot | M1 | Planned |

## Open questions

- **CI integration tests:** LocalStack's licensing page lists testing in CI as available on the Hobby plan, but CI
  requires a CI auth token, distinct from the personal developer token. To confirm how such a token is obtained on
  the Hobby plan. Fallback: unit tests and `terraform validate` in CI, integration tests run locally and documented.
