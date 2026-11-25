# ADR-002: Ship the simulator as a distroless image with a uv-managed CPython 3.12

- **Status:** Accepted
- **Date:** 2026-11-25
- **Milestone:** M1
- **Related requirements:** NFR-05, NFR-07, FR-02

## Context

The fleet simulator has to ship as a container image. Images must be distroless, run as non-root and carry no
critical CVE (NFR-05), and every version must be pinned, images included (NFR-07). The project requires
Python `>=3.12,<3.13`, so the runtime must provide a 3.12 interpreter. For a given seed the simulator's stream
is deterministic (FR-02); the image must give the same stream as the source tree for the same seed on the same
machine, which CI checks on its runner. No claim is made across machines.

## Decision drivers

- Distroless runtime, non-root user, no critical CVE (NFR-05)
- Every version pinned, base images by tag and digest (NFR-07)
- Python 3.12 exactly, as `requires-python` demands
- Same stream as the source for the same seed on the same machine (FR-02)

## Options considered

### Option A — `distroless/cc-debian13:nonroot` plus CPython 3.12 managed by uv
- Pros: no shell and no package manager in the runtime; runs as UID 65532; the exact 3.12 patch release is
  chosen by the project and installed by the same uv that builds the venv; pip and ensurepip can be removed.
- Cons: the interpreter is not a Debian package, so it has its own update path; a multi-stage build is needed to
  install it and copy it into the final stage.

### Option B — `distroless/python3-debian13:nonroot`
- Pros: distroless, non-root, no shell; the interpreter is a Debian package, listed in the image's package
  database with the other Debian packages.
- Cons: it ships Debian 13's Python 3.13, outside `requires-python`; its patch level follows Debian, not the
  project; its standard library still contains ensurepip.

### Option C — the official `python:3.12-slim` image
- Pros: the reference Python image, with CPython 3.12 on Debian 13; no interpreter to install.
- Cons: not distroless: the runtime has `sh` and `bash`, `apt` and `dpkg`, pip and ensurepip; it runs as root
  unless the image sets another user. Meeting NFR-05 would mean stripping it by hand.

## Decision

We choose **Option A**: it is the only option that gives Python 3.12 in a distroless, non-root runtime. Option B
fails the version constraint, and Option C fails NFR-05 without manual stripping. The interpreter's patch
release and every base image are pinned in `simulator.Dockerfile`, and the dependencies in `uv.lock`, so a
rebuild uses the same inputs.

## Consequences

- **Positive:**
  - No shell or package manager in the runtime; the image runs as UID 65532.
  - pip and ensurepip are removed from the interpreter: nothing in the image runs them, and the scanner no longer
    reports the packages pip vendors.
  - Every base image is pinned by tag and digest in `simulator.Dockerfile`, each published at least seven days
    before it was pinned.
- **Negative / accepted trade-offs:**
  - The scan reports nothing for the interpreter under `/opt/python`, because Trivy identifies Python packages
    from their installed metadata, so no scan flags a new CPython security release. Mitigation: check new 3.12
    patch releases by hand when the base image is re-pinned, and move the pin through a uv release that offers
    it.
  - Two update paths instead of one: the distroless digest, and the CPython patch release through uv.
  - No shell for debugging. Mitigation: `docker run --entrypoint /app/.venv/bin/python` gives an interpreter.
  - `/data` is root-owned, so `--out` needs a mounted directory; without one the run fails instead of writing a
    file that is lost with the container.
- **Scan policy (Simulator image job in CI):**
  - The gate fails on CRITICAL only; the report of all severities goes to the job summary.
  - No `--ignore-unfixed`: a CRITICAL without a fix also fails the gate. Accepting one takes a committed
    `.trivyignore` entry with a reason and a review date. The gate does not read a committed `.trivyignore`
    yet.
  - Once on `main`, the job also runs in CI's weekly scheduled run, which acts as the CVE alert: new
    vulnerabilities appear without code changes.
- **Follow-up:**
  - PID 1 signal handling: `docker stop` on a paced run, by M2.
  - Reaching LocalStack from the container, by M2.
  - The change that adds the first `.trivyignore` entry also mounts the checkout into the Trivy container and
    passes `--ignorefile` to the gate.

## Business view

The simulator image contains only what it needs to run, which shrinks the attack surface and the number of
security findings to triage. The price is a short, deliberate update routine for two components: a weekly
automated scan says when the base image is due, and the Python interpreter is checked by hand at the same time.

## Revisit when

- `requires-python` moves to 3.13;
- a CRITICAL vulnerability without a fix persists in the distroless base;
- a CPython security fix is not available as a uv-managed 3.12 build.
