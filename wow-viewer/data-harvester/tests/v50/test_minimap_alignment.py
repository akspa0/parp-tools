"""Spec 113 T007: the alignment analyzer must detect a known dihedral misorientation exactly,
pass a genuinely aligned pair as identity, and fail closed (never per-tile fixups) when tiles
disagree about the winning transform."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zarr

from harvester.v50.minimap_alignment import (
    DIHEDRAL_TRANSFORMS,
    apply_dihedral,
    apply_translation,
    analyze_stores,
    evaluate_gate,
    high_frequency_energy,
    mean_pool,
    register_tile,
)


def _structured_tile(seed: int, size: int = 64) -> np.ndarray:
    """A structured (non-symmetric) RGB image so orientation is detectable."""
    rng = np.random.default_rng(seed)
    base = rng.random((size // 8, size // 8))
    coarse = np.kron(base, np.ones((8, 8)))  # blocky structure survives downsampling
    gradient = np.linspace(0, 1, size)[:, None] * np.linspace(1, 0, size)[None, :]
    luma = ((coarse * 0.7 + gradient * 0.3) * 255).astype(np.uint8)
    return np.stack([luma, luma, luma], axis=2)


def _upscale4(rgb: np.ndarray) -> np.ndarray:
    return np.kron(rgb, np.ones((4, 4, 1))).astype(np.uint8)


def test_identity_aligned_pair_registers_as_identity():
    authored = _structured_tile(1)
    detail = _upscale4(authored)

    result = register_tile(authored, detail)

    assert result["best_transform"] == "identity"
    assert result["ncc"] > 0.95
    assert result["offset"] == [0, 0]


@pytest.mark.parametrize("transform,inverse", [("flip_v", "flip_v"), ("rot90", "rot270"), ("transpose", "transpose")])
def test_known_misorientation_is_detected_with_its_inverse(transform, inverse):
    authored = _structured_tile(2)
    # the render is misoriented by `transform`; registering it back needs the inverse
    detail = _upscale4(apply_dihedral(authored, transform).copy())

    result = register_tile(authored, detail)

    assert result["best_transform"] == inverse
    assert result["ncc"] > 0.95


def test_gate_passes_only_on_a_single_consistent_transform():
    consistent = [{"best_transform": "flip_v", "ncc": 0.97, "offset": [1, -2]} for _ in range(5)]
    verdict = evaluate_gate(consistent)
    assert verdict["gate"] == "pass_with_transform"
    assert verdict["corrective_transform"] == "flip_v"
    assert verdict["corrective_offset_lr"] == [1, -2]

    identity = [{"best_transform": "identity", "ncc": 0.98, "offset": [0, 0]} for _ in range(5)]
    assert evaluate_gate(identity)["gate"] == "pass_identity"

    mixed = [
        {"best_transform": "identity", "ncc": 0.97, "offset": [0, 0]},
        {"best_transform": "rot90", "ncc": 0.96, "offset": [0, 0]},
    ]
    verdict = evaluate_gate(mixed)
    assert verdict["gate"] == "fail_inconsistent"
    assert verdict["corrective_transform"] is None

    weak = [{"best_transform": "identity", "ncc": 0.30, "offset": [0, 0]} for _ in range(5)]
    assert evaluate_gate(weak)["gate"] == "fail_inconsistent"

    inconsistent_offsets = [
        {"best_transform": "identity", "ncc": 0.98, "offset": [-4, 0]},
        {"best_transform": "identity", "ncc": 0.98, "offset": [4, 0]},
    ]
    assert evaluate_gate(inconsistent_offsets)["gate"] == "fail_inconsistent"


def test_translation_search_has_no_edge_wrap_and_returns_a_fixed_correction():
    authored = _structured_tile(8)
    shifted = apply_translation(authored, (2, -3))
    detail = _upscale4(shifted)

    result = register_tile(authored, detail)

    assert result["best_transform"] == "identity"
    assert result["offset"] == [-2, 3]
    assert result["ncc"] > 0.95


def test_dihedral_transforms_are_a_closed_correct_set():
    image = _structured_tile(3)
    seen = set()
    for t in DIHEDRAL_TRANSFORMS:
        out = apply_dihedral(image, t)
        assert out.shape == image.shape
        seen.add(out.tobytes())
    assert len(seen) == 8  # all eight orientations genuinely distinct on a structured image


def test_high_frequency_energy_ranks_detail_over_flat():
    flat = np.full((64, 64, 3), 128, dtype=np.uint8)
    checker = np.zeros((64, 64, 3), dtype=np.uint8)
    checker[::2, :, :] = 255

    assert high_frequency_energy(flat) == pytest.approx(0.0, abs=1e-9)
    assert high_frequency_energy(checker) > 100.0


def test_mean_pool_downsamples_exactly():
    image = np.arange(16, dtype=np.float64).reshape(4, 4)
    pooled = mean_pool(image, 2)
    assert pooled.shape == (2, 2)
    assert pooled[0, 0] == pytest.approx((0 + 1 + 4 + 5) / 4)


def _write_alignment_store(path: Path, map_name: str, seed: int) -> None:
    authored_rows = np.stack([_structured_tile(seed), _structured_tile(seed + 1)])
    detail_rows = np.stack([_upscale4(row) for row in authored_rows])
    group = zarr.open_group(str(path), mode="w")
    group.attrs["minimap_rgb_1024_render_mode"] = "detail"
    group.create_array("minimap_rgb_authored", data=authored_rows)
    group.create_array("minimap_rgb_1024", data=detail_rows)
    group.create_array("minimap_rgb", data=authored_rows)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"map": map_name, "tile_x": row, "tile_y": 0}
                for row in range(len(authored_rows))
            ]
        ),
        path / "index.parquet",
    )


def test_cross_map_analyzer_requires_detail_provenance_and_emits_one_gate(tmp_path: Path):
    kalimdor = tmp_path / "Kalimdor.zarr"
    azeroth = tmp_path / "Azeroth.zarr"
    _write_alignment_store(kalimdor, "Kalimdor", 20)
    _write_alignment_store(azeroth, "Azeroth", 30)

    report = analyze_stores([kalimdor, azeroth], sample_per_store=2)

    assert report["sample_size"] == 4
    assert {tile["map"] for tile in report["sample_tiles"]} == {"Kalimdor", "Azeroth"}
    assert report["gate"] == "pass_identity"
    assert report["corrective_offset_lr"] == [0, 0]

    group = zarr.open_group(str(azeroth), mode="r+")
    del group.attrs["minimap_rgb_1024_render_mode"]
    with pytest.raises(ValueError, match="render_mode=detail"):
        analyze_stores([kalimdor, azeroth], sample_per_store=1)
