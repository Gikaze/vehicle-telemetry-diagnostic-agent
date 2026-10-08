# Local emulation and real AWS

The same Terraform modules and the same Python code run in two modes. LocalStack and real AWS are never used at
the same time: LocalStack **replaces** AWS during development; real AWS is used for short, time-boxed
validations and the final demo, then destroyed.

## 1. The two modes

```text
DEVELOPMENT MODE (most of the time, €0)            DEMO MODE (a few hours, then destroy)

Developer laptop                                   Developer laptop           AWS (eu-central-1)
┌────────────────────────────────────┐             ┌──────────────┐           ┌───────────────────────────┐
│ Simulator ──► LocalStack           │             │ Simulator ───┼── MQTT ──►│ IoT Core ──► Kinesis      │
│               (Kinesis, Lambda,    │             │              │           │   ──► Lambda ──► DynamoDB │
│                DynamoDB, CloudWatch)│             │ Agent ◄──────┼───────────┤   ──► CloudWatch          │
│ Agent ◄── reads LocalStack         │             │ Ollama       │           └───────────────────────────┘
│ Ollama, Qdrant                     │             │ Qdrant       │
└────────────────────────────────────┘             └──────────────┘
```

| | Development mode | Demo mode |
| --- | --- | --- |
| Pipeline | LocalStack on the laptop | Real AWS, `eu-central-1` |
| Telemetry entry | Simulator writes directly to Kinesis | Simulator publishes over MQTT to IoT Core, which forwards to Kinesis |
| Agent, LLM, vector store | Laptop | Laptop (edge/cloud split) |
| Terraform environment | `infra/envs/local/` | `infra/envs/aws-demo/` (human-operated) |
| Credentials | Fake (`localstack` profile) | Dedicated IAM user or SSO profile |
| Cost | €0 | Per use; destroyed after each session (NFR-01) |
| Who runs `terraform apply` | Developer or Claude Code | The human only |

What stays identical: Terraform modules, Lambda code, simulator code, agent code. What changes: configuration
only — endpoint, credentials, profile, and the simulator's publisher (Kinesis or MQTT, FR-03).

## 2. Known limits of local emulation (LocalStack Hobby plan)

| Limit | Concrete impact | Mitigation |
| --- | --- | --- |
| IoT Core not available on the Hobby plan | MQTT connection, certificates, IoT policy and topic rule are never exercised locally | Simulator writes to Kinesis locally; IoT module checked by `terraform validate` and Checkov; exercised on real AWS (smoke test, demo) |
| IAM policies not enforced | A policy that is too strict passes locally and fails with `AccessDenied` on AWS | Checkov; IAM Access Analyzer policy validation; smoke test on AWS |
| No state persistence | Restarting LocalStack wipes all resources, while the local Terraform state still lists them | `make up` starts from a clean state and re-applies; local Terraform state is disposable |
| Lambda/Kinesis error semantics | On AWS a failing record can block a shard until it expires; batching and retry behaviour is only approximated locally | Designed in from the start (FR-19); verified on AWS |
| Quotas and throttling not emulated | Kinesis shard throughput, DynamoDB capacity and Lambda concurrency never saturate locally | Irrelevant at 100 vehicles; addressed in COST.md for 10k and 100k |
| Unrealistic latency | NFR-13 measured locally says nothing about AWS | Measured again on AWS; both values published |
| CloudWatch partially emulated | Metrics and logs work; alarms and dashboards are approximate | Dashboard validated during the demo only |
| No cost signal | Nothing is billed locally | COST.md built from AWS pricing, then checked against the demo bill |
| Different configuration | `localhost:4566` endpoint, path-style S3, fake credentials | Endpoint from configuration; differences confined to `infra/envs/` |

## 3. Validation on real AWS

Real AWS is used twice, always followed by `terraform destroy` and a cost check the next day.

**Smoke test — end of M2 (about 30 minutes).** Deploy `aws-demo`, send a few hundred messages from the simulator
through IoT Core, check that they arrive in DynamoDB and that no `AccessDenied` appears in the Lambda logs, then
destroy. Goal: catch IAM, IoT and Lambda wiring issues while the code is fresh, instead of at the final demo.

**Final demo — M7.** Full scenario run, recorded video, latency measured on AWS, demo bill reported in COST.md.

## 4. Vehicle identity and IoT security (FR-20)

In demo mode, each simulated vehicle connects to IoT Core with **its own X.509 certificate**. The IoT policy
allows a vehicle to connect only with its own client ID and to publish only on its own topic
(`fleet/<vehicle_id>/telemetry`); there is no shared key. A test publishes on another vehicle's topic and expects
a rejection. Certificates are created for the demo and revoked when the stack is destroyed; private keys never
enter the repository.
