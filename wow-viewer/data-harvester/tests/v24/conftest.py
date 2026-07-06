"""Shared fixtures for Spec 094 (V24) tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture()
def synthetic_height() -> np.ndarray:
    """Deterministic smooth 257x257 heightmap with hills."""
    y, x = np.mgrid[0:257, 0:257].astype(np.float32)
    return (
        50.0
        + 30.0 * np.sin(x / 40.0)
        + 20.0 * np.cos(y / 30.0)
        + 5.0 * np.sin((x + y) / 15.0)
    ).astype(np.float32)


def shim_available() -> bool:
    from harvester.v24 import shim

    try:
        shim.find_shim_dll()
        return True
    except RuntimeError:
        return False


requires_shim = pytest.mark.skipif(
    not shim_available(), reason="WowViewer.Tool.WdlRead is not built"
)
