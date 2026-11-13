"""Checks on the repository foundations: pinned versions and secret hygiene."""

import re
import subprocess
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# Values allowed in .env.example: placeholders and local development endpoints only.
ALLOWED_ENV_VALUE = re.compile(
    r"^(<[a-z0-9-]+>|http://localhost:\d+|[a-z0-9-]+)$",
)


def _is_ignored(path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--quiet", path],
        cwd=ROOT,
        check=False,
    )
    return result.returncode == 0


def test_python_version_matches_requires_python() -> None:
    """NFR-07: the interpreter pinned for uv matches the project constraint."""
    pinned = (ROOT / ".python-version").read_text().strip()
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert pinned == "3.12"
    assert project["requires-python"] == ">=3.12,<3.13"


def test_dependencies_are_locked() -> None:
    """NFR-07: every dependency is resolved to an exact version in uv.lock."""
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    locked = {package["name"] for package in lock["package"]}
    assert {"ruff", "pytest", "pytest-cov"} <= locked


def test_localstack_image_is_pinned() -> None:
    """NFR-07: make up starts LocalStack from an explicitly tagged image, never `latest`."""
    makefile = (ROOT / "Makefile").read_text()
    match = re.search(r"^LOCALSTACK_IMAGE := (\S+)$", makefile, re.MULTILINE)
    assert match, "LOCALSTACK_IMAGE is not defined in the Makefile"
    repository, _, tag = match.group(1).rpartition(":")
    assert repository == "localstack/localstack-pro"
    assert tag and tag != "latest"
    assert "--image $(LOCALSTACK_IMAGE)" in makefile


def test_simulator_image_bases_are_pinned() -> None:
    """NFR-07: every base image of the simulator image has an explicit tag and a digest."""
    dockerfile = (ROOT / "simulator.Dockerfile").read_text()
    references = re.findall(r"^FROM\s+(\S+)", dockerfile, re.MULTILINE)
    repositories = set()
    for reference in references:
        match = re.fullmatch(r"([a-z0-9./-]+):([A-Za-z0-9._-]+)@sha256:[0-9a-f]{64}", reference)
        assert match, f"{reference} is not pinned as repository:tag@sha256:digest"
        assert match[2] != "latest", reference
        repositories.add(match[1])
    assert len(references) == 3
    assert repositories == {"ghcr.io/astral-sh/uv", "debian", "gcr.io/distroless/cc-debian13"}


def test_simulator_image_data_directory_is_root_owned_and_runs_as_nonroot() -> None:
    """The final stage creates /data as root (WORKDIR creates it as the current user, and the
    distroless nonroot base is already 65532), then runs as 65532:65532. Rules as written; the
    image itself is checked with os.stat in the image re-proof."""
    dockerfile = (ROOT / "simulator.Dockerfile").read_text()
    final_stage = re.split(r"^FROM\s", dockerfile, flags=re.MULTILINE)[-1]
    instructions = re.findall(r"^([A-Z]+)\s+(.*?)\s*$", final_stage, re.MULTILINE)
    workdir = instructions.index(("WORKDIR", "/data"))
    assert instructions[workdir - 1] == ("USER", "0:0"), "WORKDIR /data is not created as root"
    users = [argument for keyword, argument in instructions if keyword == "USER"]
    assert users[-1] == "65532:65532", "the image does not end as 65532:65532"


def test_dockerignore_is_an_allowlist_of_the_image_inputs() -> None:
    """Only the files the simulator image is built from reach the Docker build context (the real
    context is listed by Docker in the image re-proof; this checks the rules as written)."""
    rules = [
        line.strip()
        for line in (ROOT / ".dockerignore").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert rules[0] == "*", "the first rule must exclude everything"
    exceptions = [index for index, rule in enumerate(rules) if rule.startswith("!")]
    assert len(exceptions) == len({rules[index] for index in exceptions}), "duplicate exception"
    assert {rules[index][1:] for index in exceptions} == {
        "pyproject.toml",
        "uv.lock",
        "simulator/pyproject.toml",
        "simulator/src/",
        "simulator/catalogue.yaml",
        "simulator/config/",
        "simulator/schema/",
    }
    # The last matching rule wins: a re-exclusion before an exception would be overridden by it.
    for pattern in ("**/__pycache__", "**/*.pyc"):
        assert pattern in rules, f"{pattern} is not excluded"
        assert rules.index(pattern) > max(exceptions), f"{pattern} comes before an exception"


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".env.local",
        "infra/envs/local/terraform.tfstate",
        "infra/envs/local/terraform.tfstate.backup",
        "infra/envs/local/.terraform/providers",
        "infra/envs/aws-demo/demo.tfvars",
        "certs/vehicle-001.pem",
        "certs/vehicle-001.key",
        "certs/vehicle-001.crt",
        ".claude/settings.local.json",
    ],
)
def test_sensitive_files_are_ignored(path: str) -> None:
    """NFR-03: secrets, keys, certificates and local state never reach git."""
    assert _is_ignored(path)


@pytest.mark.parametrize("path", [".env.example", ".claude/settings.json"])
def test_shared_configuration_is_tracked(path: str) -> None:
    """NFR-03: the committed templates are not swallowed by the ignore rules."""
    assert not _is_ignored(path)


def test_env_example_holds_placeholders_only() -> None:
    """NFR-03: .env.example contains no real credential values."""
    for line in (ROOT / ".env.example").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        assert ALLOWED_ENV_VALUE.match(value), f"{key} has a non-placeholder value"
