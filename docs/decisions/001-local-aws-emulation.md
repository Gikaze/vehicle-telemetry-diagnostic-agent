# ADR-001: Emulate AWS locally with LocalStack (Hobby plan), driven by lstk

- **Status:** Accepted
- **Date:** 2026-10-06
- **Milestone:** M0
- **Related requirements:** NFR-01, NFR-06, NFR-07

## Context

The project needs Kinesis, Lambda, DynamoDB, CloudWatch and IAM throughout development, with dozens of
deploy–test–destroy cycles per day. Development must cost €0 in AWS spend (NFR-01); real AWS is reserved
for recorded demos, destroyed immediately afterwards.

In March 2026 LocalStack discontinued its open-source Community edition: the single unified image now requires
an account and an auth token, and the legacy `localstack` CLI is deprecated in favour of a new CLI, `lstk`.
Several MIT-licensed emulators appeared as drop-in replacements, but they are young projects.

## Decision drivers

- €0 development cost, with no time-limited trial the project would come to depend on
- Coverage and fidelity for Kinesis, Lambda, DynamoDB, CloudWatch and IAM
- Recognition by reviewers and employers; quality of documentation
- Same Terraform code for local and real AWS environments

## Options considered

### Option A — LocalStack, Hobby plan (free, non-commercial), managed with `lstk`
- Pros: industry reference; mature documentation; covers every service the pipeline needs; `lstk` adds
  `lstk aws` / `lstk terraform` proxies, snapshots and a `doctor` pre-flight check.
- Cons: account and auth token required; some services reserved for paid plans; CI use needs a separate CI token.

### Option B — LocalStack 30-day trial (Ultimate features)
- Pros: full service coverage, including IoT Core.
- Cons: expires mid-project; building on features that later disappear creates a hidden dependency.

### Option C — MIT-licensed alternatives (e.g. MiniStack, Floci)
- Pros: no account, no token, lighter footprint.
- Cons: recent projects with limited track record; fidelity and longevity unproven; less recognisable to reviewers.

### Option D — A dedicated real AWS development account
- Pros: perfect fidelity.
- Cons: every iteration costs money and minutes; risk of forgotten resources; incompatible with NFR-01.

## Decision

We choose **Option A**: LocalStack on the Hobby plan, managed with `lstk`. It meets the cost constraint
without an expiry date, covers the services the pipeline relies on, and is the tool reviewers expect to see.
Real AWS (Option D) is used only for the final demo in `infra/envs/aws-demo/`.

## Consequences

- **Positive:** unlimited free deploy cycles; integration tests run against an AWS-compatible API; the same
  Terraform modules serve `envs/local` and `envs/aws-demo`.
- **Negative / accepted trade-offs:**
  - The design does not rely on IoT Core emulation. Locally, the simulator publishes directly to Kinesis;
    IoT Core is introduced only in the real AWS demo (FR-03). Service availability on the plan is checked
    with `lstk status` and recorded in `docs/SETUP.md`.
  - Emulation is not perfect fidelity (IAM enforcement, quotas, error messages). Mitigation: the M7 demo on
    real AWS is the final validation, and IAM policies are additionally checked with Checkov.
  - The local network blocks DNS answers resolving to 127.0.0.1 (rebind protection), so
    `*.localhost.localstack.cloud` names do not resolve. Mitigation: endpoint is always `http://localhost:4566`
    and S3 uses path-style addressing.
- **Follow-up:** verify CI token availability on the Hobby plan (open question in `docs/requirements.md`);
  document `lstk` installation in `docs/SETUP.md`.

## Business view

Local emulation turns infrastructure experimentation from a metered cost into a fixed cost of zero, and removes
the risk of an unexpected cloud bill. The trade-off — a small gap in fidelity — is closed by a single,
time-boxed validation on real AWS, which is how a cost-conscious team would run a proof of concept.

## Revisit when

- the Hobby plan terms change, or a required service becomes unavailable on it;
- CI integration tests cannot run with the available tokens;
- an MIT alternative reaches proven maturity and fidelity for the services in use.
