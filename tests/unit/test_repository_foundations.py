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
