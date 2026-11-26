"""The slow tests run only when asked for with -m slow (make test-24h): a bare pytest over tests/
must never start a fleet day."""

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    expression = config.getoption("-m") or ""
    if "slow" in expression and "not slow" not in expression:
        return
    skip = pytest.mark.skip(reason="slow test: run with make test-24h (pytest -m slow)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
