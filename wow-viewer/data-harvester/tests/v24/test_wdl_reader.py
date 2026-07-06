"""Real-WDL reader wrapper tests (FR-001) — real staged client where available."""

from pathlib import Path

import numpy as np
import pytest

from harvester.v24 import wdl_reader

from .conftest import requires_shim

STAGED_335 = Path("i:/parp/parp-tools/output/tmp/wowarchive-clients/3_3_5_12340")

requires_staged_client = pytest.mark.skipif(
    not STAGED_335.exists(), reason="3_3_5_12340 staged client is not present"
)


@pytest.mark.v24
@requires_shim
@requires_staged_client
def test_read_wdl_map_tiles_real_client():
    tiles = wdl_reader.read_wdl_map_tiles(STAGED_335, "Azeroth")
    assert tiles is not None
    assert len(tiles) > 500
    outer, inner = next(iter(tiles.values()))
    assert outer.shape == (17, 17)
    assert inner.shape == (16, 16)
    assert outer.dtype == np.float32


@pytest.mark.v24
@requires_shim
@requires_staged_client
def test_read_wdl_map_missing_map_returns_none():
    assert wdl_reader.read_wdl_map_tiles(STAGED_335, "NoSuchMapExists") is None
