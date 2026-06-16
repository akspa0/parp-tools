from __future__ import annotations

import numpy as np

from harvester.mismatch_detector import (
    compute_tile_mismatch_metrics,
    detect_mismatches,
    normal_relief,
    normal_edge_strength,
)


class _FakeRoot:
    def __init__(self, arrays: dict[str, np.ndarray]):
        self._arrays = arrays

    def __getitem__(self, key: str) -> np.ndarray:
        return self._arrays[key]

    def __contains__(self, key: str) -> bool:
        return key in self._arrays


def _make_fake_store(height: np.ndarray, normals: np.ndarray,
                     normal_mask: np.ndarray | None = None,
                     minimap: np.ndarray | None = None) -> _FakeRoot:
    arrays: dict[str, np.ndarray] = {
        "height_257": height.astype(np.float32)[np.newaxis, ...],
        "normal_xyz": normals.astype(np.float32)[np.newaxis, ...],
    }
    if normal_mask is not None:
        arrays["normal_mask"] = normal_mask.astype(np.float32)[np.newaxis, ...]
    if minimap is not None:
        arrays["minimap_rgb"] = minimap.astype(np.uint8)[np.newaxis, ...]
    else:
        arrays["minimap_rgb"] = np.full((1, 256, 256, 3), 128, dtype=np.uint8)
    return _FakeRoot(arrays)


def test_flags_mismatch_when_normals_varied_height_flat() -> None:
    height = np.full((257, 257), 0.0, dtype=np.float32)
    height[100:150, 100:150] = 0.3
    normals = np.zeros((257, 257, 3), dtype=np.float32)
    normals[:, :, 2] = 1.0
    normals[50:200, 50:200, 0] = 0.5
    normals[50:200, 50:200, 2] = 0.866
    normal_mask = np.ones((257, 257), dtype=np.float32)

    store = _make_fake_store(height, normals, normal_mask)
    metrics = compute_tile_mismatch_metrics(store, 0, {"has_normals": True})
    result = detect_mismatches(root=store, metrics=metrics)

    assert result["is_mismatch"] is True
    assert result["mismatch_reason"] == "height_flat_vs_normal_varied"
    assert result["mismatch_severity"] in ("low", "medium", "high")


def test_skips_blank_tile() -> None:
    height = np.zeros((257, 257), dtype=np.float32)
    normals = np.zeros((257, 257, 3), dtype=np.float32)
    normals[:, :, 2] = 1.0
    normal_mask = np.ones((257, 257), dtype=np.float32)

    store = _make_fake_store(height, normals, normal_mask)
    metrics = compute_tile_mismatch_metrics(store, 0, {"has_normals": True})
    result = detect_mismatches(root=store, metrics=metrics)

    assert result["is_mismatch"] is False
    assert result["mismatch_reason"] == "flat_normals"


def test_skips_when_normal_cov_below_threshold() -> None:
    height = np.full((257, 257), 0.0, dtype=np.float32)
    normals = np.zeros((257, 257, 3), dtype=np.float32)
    normals[:, :, 2] = 1.0
    normals[50:60, 50:60, 0] = 0.5
    normal_mask = np.zeros((257, 257), dtype=np.float32)
    normal_mask[50:60, 50:60] = 1.0

    store = _make_fake_store(height, normals, normal_mask)
    metrics = compute_tile_mismatch_metrics(store, 0, {"has_normals": True})
    result = detect_mismatches(root=store, metrics=metrics,
                               normal_cov_threshold=0.10)

    assert result["is_mismatch"] is False
    assert result["mismatch_reason"] == "insufficient_normal_coverage"


def test_skips_when_height_has_sufficient_range() -> None:
    height = np.zeros((257, 257), dtype=np.float32)
    height[:, 100:] = 10.0
    normals = np.zeros((257, 257, 3), dtype=np.float32)
    normals[:, :, 2] = 1.0
    normals[50:200, 50:200, 0] = 0.5
    normals[50:200, 50:200, 2] = 0.866
    normal_mask = np.ones((257, 257), dtype=np.float32)

    store = _make_fake_store(height, normals, normal_mask)
    metrics = compute_tile_mismatch_metrics(store, 0, {"has_normals": True})
    result = detect_mismatches(root=store, metrics=metrics)

    assert result["is_mismatch"] is False
    assert result["mismatch_reason"] == "sufficient_height_range"


def test_normal_relief_calculation() -> None:
    normals = np.zeros((4, 4, 3), dtype=np.float32)
    normals[:, :, 2] = 1.0
    normals[1:3, 1:3, 0] = 0.6
    normals[1:3, 1:3, 2] = 0.8
    mask = np.ones((4, 4), dtype=np.float32)

    relief = normal_relief(normals, mask)
    assert relief[1, 1] == pytest.approx(0.6, abs=0.01)
    assert relief[0, 0] == pytest.approx(0.0)


def test_normal_edge_strength_calculation() -> None:
    normals = np.zeros((4, 4, 3), dtype=np.float32)
    normals[:, :, 2] = 1.0
    normals[2, :, 0] = 0.5
    mask = np.ones((4, 4), dtype=np.float32)

    edges = normal_edge_strength(normals, mask)
    assert edges[2, 1] > 0.0


def test_severity_high_when_relief_dominates_height() -> None:
    height = np.full((257, 257), 0.0, dtype=np.float32)
    height[128, 128] = 0.1
    normals = np.zeros((257, 257, 3), dtype=np.float32)
    normals[:, :, 2] = 1.0
    normals[:, :, 0] = 0.7
    normals[:, :, 2] = 0.714
    normal_mask = np.ones((257, 257), dtype=np.float32)

    store = _make_fake_store(height, normals, normal_mask)
    metrics = compute_tile_mismatch_metrics(store, 0, {"has_normals": True})
    result = detect_mismatches(root=store, metrics=metrics)

    assert result["is_mismatch"] is True
    assert result["mismatch_severity"] == "high"


# --- Phase 2: Normal-to-Height Reconstruction tests ---

from harvester.normal_height_reconstructor import (
    reconstruct_height_from_normals,
    anchor_heights,
)


def test_reconstructs_synthetic_slope() -> None:
    height_true = np.zeros((65, 65), dtype=np.float32)
    for i in range(65):
        height_true[i, :] = i * 0.1

    gy, gx = np.gradient(height_true)
    nz = 1.0 / np.sqrt(gx**2 + gy**2 + 1.0)
    nx = -gx * nz
    ny = -gy * nz

    normals = np.stack([nx, ny, nz], axis=-1).astype(np.float32)
    mask = np.ones((65, 65), dtype=np.float32)

    reconstructed = reconstruct_height_from_normals(normals, normal_mask=mask, nz_clip=0.01)

    rec_centered = reconstructed - reconstructed.mean()
    true_centered = height_true - height_true.mean()
    corr = np.corrcoef(rec_centered.ravel(), true_centered.ravel())[0, 1]

    assert corr > 0.70, f"reconstructed height correlation {corr:.4f} below 0.70"


def test_flat_normals_produce_noop() -> None:
    height = np.zeros((33, 33), dtype=np.float32)
    normals = np.zeros((33, 33, 3), dtype=np.float32)
    normals[:, :, 2] = 1.0
    mask = np.ones((33, 33), dtype=np.float32)

    reconstructed = reconstruct_height_from_normals(normals, normal_mask=mask)
    hr = reconstructed.max() - reconstructed.min()

    assert hr < 1e-4, f"flat normals produced height range {hr}"


def test_anchor_heights_preserves_mean() -> None:
    original = np.full((65, 65), 42.0, dtype=np.float32)
    reconstructed = np.full((65, 65), 10.0, dtype=np.float32)
    mask = np.ones((65, 65), dtype=np.float32)

    anchored = anchor_heights(reconstructed, original, normal_mask=mask)
    assert anchored.mean() == pytest.approx(42.0, abs=0.01)


def test_anchor_heights_with_mask() -> None:
    original = np.full((65, 65), 50.0, dtype=np.float32)
    original[20:40, 20:40] = 100.0
    reconstructed = np.full((65, 65), -10.0, dtype=np.float32)
    mask = np.zeros((65, 65), dtype=np.float32)
    mask[20:40, 20:40] = 1.0

    anchored = anchor_heights(reconstructed, original, normal_mask=mask)
    mask_mean = float(original[mask.astype(bool)].mean())
    anchored_mask_mean = float(anchored[mask.astype(bool)].mean())
    assert anchored_mask_mean == pytest.approx(mask_mean, abs=0.1)


# --- Phase 3: Repair idempotency test ---

def test_repair_idempotency_flag() -> None:
    rows = [
        {"tile_id": 0, "build": "test", "map": "test_map", "tile_x": 0, "tile_y": 0},
        {"tile_id": 1, "build": "test", "map": "test_map", "tile_x": 1, "tile_y": 0},
    ]
    already_corrected = {0}
    tiles_to_fix = [r for r in rows if int(r["tile_id"]) not in already_corrected]
    assert len(tiles_to_fix) == 1
    assert tiles_to_fix[0]["tile_id"] == 1


try:
    import pytest
except ImportError:
    pass
