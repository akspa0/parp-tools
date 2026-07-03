"""Pytest gating for Spec 089 V23 tests."""

from collections.abc import Sequence

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: Sequence[pytest.Item]) -> None:
    """Skip V23 tests unless the caller intentionally selects the v23 marker."""
    if "v23" in config.option.markexpr:
        return

    marker = pytest.mark.skip(reason="run V23 tests with pytest -m v23")
    for item in items:
        item.add_marker(marker)
