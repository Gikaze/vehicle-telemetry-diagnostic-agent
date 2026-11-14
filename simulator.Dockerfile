# Simulator image (NFR-05, NFR-07). Build it with `make image` (tag, labels, committed inputs).
# Every base image is pinned by tag and digest, each published at least 7 days before it was
# pinned. The build context is the repository root, filtered by the .dockerignore allowlist.

# CPython is managed by uv; .python-version pins only the minor version (3.12).
ARG PYTHON_VERSION=3.12.15

FROM ghcr.io/astral-sh/uv:0.12.23@sha256:61d393e44e249f2e4b526b6c7ddcecce245946826e608e11c93ad4f5bba55b21 AS uv

FROM debian:trixie-slim@sha256:a72b4433de1cf7d6bae0c4cee74d0547f58601a643b3375f070c51600d63b6b2 AS builder
ARG PYTHON_VERSION
COPY --from=uv /uv /usr/local/bin/uv
# The interpreter goes to /opt/python, outside /root, so UID 65532 can read it; the final stage
# copies it to the same path, because the venv's bin/python is a symlink into it.
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_MANAGED_PYTHON=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
# Install the interpreter, then remove pip and ensurepip from it: nothing runs them (uv builds the
# venv, which has no pip), and pip vendors its own packages, which Trivy reports from pip's SBOM.
# The paths come from the interpreter's sysconfig, not a hard-coded patch directory, and a path
# that does not exist fails the build, so a layout change cannot turn this into a silent no-op.
# Tcl/Tk and idlelib stay.
RUN uv python install "${PYTHON_VERSION}" \
 && python="$(uv python find --no-project "${PYTHON_VERSION}")" \
 && stdlib="$("$python" -c 'import sysconfig; print(sysconfig.get_path("stdlib"))')" \
 && purelib="$("$python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')" \
 && set -- "$purelib"/pip "$purelib"/pip-*.dist-info "$stdlib"/ensurepip "$(dirname "$python")"/pip* \
 && for path in "$@"; do [ -e "$path" ] || { echo "not found: $path" >&2; exit 1; }; done \
 && rm -r -- "$@"
WORKDIR /src
# Layer 1: the runtime dependencies only, from the lock. The workspace pyproject.toml files come
# with it: uv needs them to resolve the vtda-simulator member.
COPY pyproject.toml uv.lock ./
COPY simulator/pyproject.toml simulator/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-workspace --no-dev --no-editable \
        --package vtda-simulator --python "${PYTHON_VERSION}"
# Layer 2: the simulator package itself, not editable; --locked fails on a stale uv.lock.
COPY simulator/src simulator/src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable \
        --package vtda-simulator --python "${PYTHON_VERSION}"

FROM gcr.io/distroless/cc-debian13:nonroot@sha256:08476c244888bfb425bae1ae4dd7c17f093e972aab8426883d97f3c6f280f7c1
COPY --from=builder /opt/python /opt/python
COPY --from=builder /app/.venv /app/.venv
# Outside the source tree, paths.py finds the data files only through SIM_HOME.
COPY simulator/catalogue.yaml /app/sim-home/catalogue.yaml
COPY simulator/config /app/sim-home/config
COPY simulator/schema /app/sim-home/schema
ENV SIM_HOME=/app/sim-home
# /data is root-owned on purpose: --out needs a mounted directory, e.g.
#   docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/out:/data" vtda-simulator:<version> \
#     --target file --out run.jsonl
# Without one, the run fails (permission denied, exit code 2) instead of writing a file into the
# container that is lost with it. The nonroot base already runs as 65532, and WORKDIR creates the
# directory as the current user, so /data is created as root, then the user switches back.
USER 0:0
WORKDIR /data
USER 65532:65532
ENTRYPOINT ["/app/.venv/bin/vtda-simulator"]

# Last, so a new revision invalidates no build layer. make image passes both arguments.
ARG VERSION
ARG REVISION
LABEL org.opencontainers.image.title="vtda-simulator" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}" \
      org.opencontainers.image.source="https://github.com/Gikaze/vehicle-telemetry-diagnostic-agent" \
      org.opencontainers.image.licenses="MIT"
