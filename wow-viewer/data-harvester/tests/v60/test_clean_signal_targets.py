from __future__ import annotations

import numpy as np
import pytest

from harvester.v60.clean_signal_targets import (
    LOW_PASS_VERSION,
    CleanTargetError,
    decompose_relative_height,
    encode_relative_height,
    recompose_height,
)


def _height(offset: float = 0.0) -> np.ndarray:
    y, x = np.mgrid[0:257, 0:257]
    return offset + (x * 0.25) + np.sin(y / 17.0) * 5.0


def test_decomposition_is_deterministic_and_recomposes_exact_target() -> None:
    target = decompose_relative_height(_height())
    repeat = decompose_relative_height(_height())

    assert target.low_pass_version == LOW_PASS_VERSION
    np.testing.assert_array_equal(target.relative_height_257, repeat.relative_height_257)
    np.testing.assert_array_equal(target.coarse_relief_257, repeat.coarse_relief_257)
    np.testing.assert_array_equal(target.detail_residual_257, repeat.detail_residual_257)
    np.testing.assert_allclose(
        recompose_height(target.coarse_relief_257, target.detail_residual_257),
        target.relative_height_257,
        atol=2e-6,
    )
    assert float(target.coarse_relief_257.min()) >= 0.0
    assert float(target.coarse_relief_257.max()) <= 1.0
    assert float(target.detail_residual_257.min()) < 0.0
    assert float(target.detail_residual_257.max()) > 0.0


def test_relative_height_is_invariant_to_absolute_altitude() -> None:
    base, base_min, base_max = encode_relative_height(_height())
    shifted, shifted_min, shifted_max = encode_relative_height(_height(10000.0))

    np.testing.assert_allclose(base, shifted, atol=1e-6)
    assert shifted_min == pytest.approx(base_min + 10000.0)
    assert shifted_max == pytest.approx(base_max + 10000.0)


def test_flat_and_near_flat_targets_keep_the_range_floor() -> None:
    flat = np.full((257, 257), 42.0, dtype=np.float32)
    normalized, tile_min, tile_max = encode_relative_height(flat)
    assert tile_min == tile_max == 42.0
    assert float(normalized.max()) == 0.0

    near_flat = flat.copy()
    near_flat[128, 128] += 0.25
    normalized, _, _ = encode_relative_height(near_flat)
    assert float(normalized.max()) == pytest.approx(0.25)


def test_target_requires_the_published_257_grid() -> None:
    with pytest.raises(CleanTargetError, match="shape"):
        decompose_relative_height(np.zeros((16, 16), dtype=np.float32))
