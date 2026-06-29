"""Tests for spec 077 terrain augmentation and map-grouped split.

Covers:
  * D4 augmentation is geometrically exact for height (scalar field).
  * Augmented normals match normals re-derived from the augmented height
    field, for every transform (the critical normal-convention check).
  * Augmentation preserves array shapes and does not mutate the input.
  * Shadow-safe augmentation is identity-only by default for baked minimap RGB.
  * Explicit D4 augmentation still works for geometry-only ablations.
  * Validation is never augmented even when the base dataset has augment on
    (the _AugmentGuardSubset contract).
  * Map-grouped split holds out entire maps and falls back to random when
    no map metadata is present.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import zarr
import zarr.codecs
import zarr.storage

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
for _entry in (_REPO_ROOT, _SRC_DIR, _SCRIPTS_DIR):
    _entry_str = str(_entry)
    if _entry_str not in sys.path:
        sys.path.insert(0, _entry_str)

from harvester.terrain_augment import (  # noqa: E402
    ALL_TRANSFORMS,
    SHADOW_SAFE_TRANSFORMS,
    augment_sample,
    sample_transform,
)
from harvester.height_only_prior_dataset import HeightOnlyPriorDataset  # noqa: E402
from harvester.height_to_normal import analytic_normals_from_height  # noqa: E402
import train_height_only_prior  # noqa: E402
from train_height_only_prior import (  # noqa: E402
    _AugmentGuardSubset,
    _split_train_val,
    _split_train_val_by_map,
)

CODEC = zarr.codecs.BloscCodec(cname="zstd", clevel=1, shuffle="bitshuffle")


# --- augmentation geometry ------------------------------------------------

@pytest.mark.parametrize("transform", ALL_TRANSFORMS)
def test_augment_height_is_scalar_field_symmetry(transform) -> None:
    """Augmenting then re-deriving normals from the augmented height must
    match the augmented normals exactly (the critical convention check)."""
    rng = np.random.default_rng(42)
    h = rng.normal(size=(1, 257, 257)).astype(np.float32)
    base_n = analytic_normals_from_height(h)[0]  # (3, 257, 257)
    sample = {
        "height_257": h.copy(),
        "normal_xyz": base_n.copy(),
        "weight_257": np.ones((1, 257, 257), dtype=np.float32),
        "input_prior": rng.normal(size=(5, 256, 256)).astype(np.float32),
    }
    out = augment_sample(sample, transform)
    recon_n = analytic_normals_from_height(out["height_257"])[0]
    cos = (out["normal_xyz"] * recon_n).sum(0)
    assert float((1.0 - cos).max()) < 1e-4, f"{transform} broke normal convention"


def test_augment_preserves_shapes() -> None:
    rng = np.random.default_rng(0)
    sample = {
        "input_prior": rng.normal(size=(5, 256, 256)).astype(np.float32),
        "raw_minimap_rgb": rng.normal(size=(3, 256, 256)).astype(np.float32),
        "teacher_object_mask": rng.normal(size=(1, 256, 256)).astype(np.float32),
        "teacher_object_confidence": rng.normal(size=(1, 256, 256)).astype(np.float32),
        "height_257": rng.normal(size=(1, 257, 257)).astype(np.float32),
        "normal_xyz": rng.normal(size=(3, 257, 257)).astype(np.float32),
        "normal_mask": rng.normal(size=(1, 257, 257)).astype(np.float32),
        "weight_257": rng.normal(size=(1, 257, 257)).astype(np.float32),
        "meta_build": "b",
        "meta_map": "m",
        "meta_tile_id": 7,
    }
    for transform in ALL_TRANSFORMS:
        out = augment_sample(sample, transform)
        assert out["input_prior"].shape == (5, 256, 256)
        assert out["height_257"].shape == (1, 257, 257)
        assert out["normal_xyz"].shape == (3, 257, 257)
        assert out["weight_257"].shape == (1, 257, 257)
        assert out["meta_build"] == "b"
        assert out["meta_tile_id"] == 7


def test_augment_identity_is_noop() -> None:
    rng = np.random.default_rng(1)
    sample = {
        "height_257": rng.normal(size=(1, 257, 257)).astype(np.float32),
        "normal_xyz": rng.normal(size=(3, 257, 257)).astype(np.float32),
        "weight_257": rng.normal(size=(1, 257, 257)).astype(np.float32),
        "input_prior": rng.normal(size=(5, 256, 256)).astype(np.float32),
    }
    out = augment_sample(sample, "identity")
    np.testing.assert_array_equal(out["height_257"], sample["height_257"])
    np.testing.assert_array_equal(out["normal_xyz"], sample["normal_xyz"])


def test_augment_does_not_mutate_input() -> None:
    rng = np.random.default_rng(2)
    h = rng.normal(size=(1, 257, 257)).astype(np.float32)
    h_copy = h.copy()
    sample = {
        "height_257": h,
        "normal_xyz": rng.normal(size=(3, 257, 257)).astype(np.float32),
        "weight_257": np.ones((1, 257, 257), dtype=np.float32),
        "input_prior": rng.normal(size=(5, 256, 256)).astype(np.float32),
    }
    _ = augment_sample(sample, "rot90")
    np.testing.assert_array_equal(h, h_copy)


def test_augment_hflip_flips_width_axis() -> None:
    rng = np.random.default_rng(3)
    img = rng.normal(size=(1, 4, 6)).astype(np.float32)
    sample = {"input_prior": img, "height_257": img, "weight_257": img}
    out = augment_sample(sample, "hflip")
    np.testing.assert_array_equal(out["input_prior"], np.ascontiguousarray(np.flip(img, axis=-1)))


def test_sample_transform_covers_all_d4() -> None:
    rng = np.random.default_rng(4)
    seen = set()
    for _ in range(200):
        seen.add(sample_transform(rng, ALL_TRANSFORMS))
    assert seen == set(ALL_TRANSFORMS)


def test_shadow_safe_transform_set_is_identity_only() -> None:
    """Baked minimap shadows have fixed direction, so default augmentation cannot rotate/flip."""
    rng = np.random.default_rng(5)
    seen = {sample_transform(rng, SHADOW_SAFE_TRANSFORMS) for _ in range(20)}
    assert seen == {"identity"}


# --- dataset augment flag -------------------------------------------------

def _make_prior_store(path: Path, n_tiles: int = 3, maps: list[str] | None = None) -> None:
    if path.exists():
        import shutil
        shutil.rmtree(path)
    store = zarr.storage.LocalStore(str(path), read_only=False)
    root = zarr.group(store=store)
    prior = np.zeros((n_tiles, 256, 256, 5), dtype=np.uint8)
    mask = np.zeros((n_tiles, 256, 256), dtype=np.uint8)
    for i in range(n_tiles):
        prior[i, :, :, 0] = (i + 1) * 30
    root.create_array("processed_minimap_prior_256", data=prior, chunks=(n_tiles, 256, 256, 5), compressors=CODEC)
    root.create_array("teacher_object_mask_256", data=mask, chunks=(n_tiles, 256, 256), compressors=CODEC)
    root.attrs.update({"schema": "spec-077-teacher-prior", "build": "test_build"})
    map_names = maps if maps is not None else ["Test"] * n_tiles
    table = pa.table({
        "build": ["test_build"] * n_tiles,
        "map_name": map_names,
        "map": map_names,
        "tile_id": list(range(n_tiles)),
        "tile_x": list(range(n_tiles)),
        "tile_y": list(range(n_tiles)),
    })
    pq.write_table(table, str(path / "tiles.parquet"))


def _make_v18_store(path: Path, n_tiles: int = 3) -> None:
    if path.exists():
        import shutil
        shutil.rmtree(path)
    store = zarr.storage.LocalStore(str(path), read_only=False)
    root = zarr.group(store=store)
    height = np.linspace(100.0, 200.0, n_tiles * 257 * 257, dtype=np.float32).reshape(n_tiles, 257, 257)
    filtered = np.zeros((n_tiles, 257, 257), dtype=np.float32)
    normals = np.zeros((n_tiles, 257, 257, 3), dtype=np.float32)
    normals[:, :, :, 2] = 1.0
    normal_mask = np.ones((n_tiles, 257, 257), dtype=np.float32)
    root.create_array("height_257", data=height, chunks=(n_tiles, 257, 257), compressors=CODEC)
    root.create_array("object_filtered_mask", data=filtered, chunks=(n_tiles, 257, 257), compressors=CODEC)
    root.create_array("normal_xyz", data=normals, chunks=(n_tiles, 257, 257, 3), compressors=CODEC)
    root.create_array("normal_mask", data=normal_mask, chunks=(n_tiles, 257, 257), compressors=CODEC)


def test_dataset_augment_off_by_default_is_deterministic() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        prior_path = root / "prior.zarr"
        v18_path = root / "v18.zarr"
        _make_prior_store(prior_path, n_tiles=3)
        _make_v18_store(v18_path, n_tiles=3)
        ds = HeightOnlyPriorDataset(prior_path=prior_path, v18_path=v18_path, height_norm=False)
        s0 = ds[0]
        s1 = ds[0]
        np.testing.assert_array_equal(s0["height_257"].numpy(), s1["height_257"].numpy())


def test_dataset_augment_default_shadow_safe_is_identity(tmp_path: Path) -> None:
    """Dataset-level augmentation defaults to shadow-safe identity for minimap RGB."""
    prior_path = tmp_path / "prior.zarr"
    v18_path = tmp_path / "v18.zarr"
    if prior_path.exists():
        import shutil
        shutil.rmtree(prior_path)
    store = zarr.storage.LocalStore(str(prior_path), read_only=False)
    root = zarr.group(store=store)
    n_tiles = 3
    prior = np.zeros((n_tiles, 256, 256, 5), dtype=np.uint8)
    yy, xx = np.meshgrid(np.arange(256), np.arange(256), indexing="ij")
    for i in range(n_tiles):
        # Spatial gradient: channel 0 varies along x, channel 1 along y.
        prior[i, :, :, 0] = (xx * (i + 1) % 255).astype(np.uint8)
        prior[i, :, :, 1] = (yy * (i + 1) % 255).astype(np.uint8)
    mask = np.zeros((n_tiles, 256, 256), dtype=np.uint8)
    root.create_array("processed_minimap_prior_256", data=prior, chunks=(n_tiles, 256, 256, 5), compressors=CODEC)
    root.create_array("teacher_object_mask_256", data=mask, chunks=(n_tiles, 256, 256), compressors=CODEC)
    root.attrs.update({"schema": "spec-077-teacher-prior", "build": "test_build"})
    table = pa.table({
        "build": ["test_build"] * n_tiles,
        "map_name": ["Test"] * n_tiles,
        "map": ["Test"] * n_tiles,
        "tile_id": list(range(n_tiles)),
        "tile_x": list(range(n_tiles)),
        "tile_y": list(range(n_tiles)),
    })
    pq.write_table(table, str(prior_path / "tiles.parquet"))
    _make_v18_store(v18_path, n_tiles=n_tiles)

    ds = HeightOnlyPriorDataset(
        prior_path=prior_path, v18_path=v18_path, height_norm=False,
        augment=True, augment_seed=123,
    )
    ds_plain = HeightOnlyPriorDataset(prior_path=prior_path, v18_path=v18_path, height_norm=False)
    for _ in range(20):
        np.testing.assert_array_equal(ds[0]["input_prior"].numpy(), ds_plain[0]["input_prior"].numpy())


def test_dataset_augment_d4_policy_changes_sample(tmp_path: Path) -> None:
    # Use a spatially-varying prior so explicit D4 flips/rotations are detectable.
    prior_path = tmp_path / "prior.zarr"
    v18_path = tmp_path / "v18.zarr"
    if prior_path.exists():
        import shutil
        shutil.rmtree(prior_path)
    store = zarr.storage.LocalStore(str(prior_path), read_only=False)
    root = zarr.group(store=store)
    n_tiles = 3
    prior = np.zeros((n_tiles, 256, 256, 5), dtype=np.uint8)
    yy, xx = np.meshgrid(np.arange(256), np.arange(256), indexing="ij")
    for i in range(n_tiles):
        prior[i, :, :, 0] = (xx * (i + 1) % 255).astype(np.uint8)
        prior[i, :, :, 1] = (yy * (i + 1) % 255).astype(np.uint8)
    mask = np.zeros((n_tiles, 256, 256), dtype=np.uint8)
    root.create_array("processed_minimap_prior_256", data=prior, chunks=(n_tiles, 256, 256, 5), compressors=CODEC)
    root.create_array("teacher_object_mask_256", data=mask, chunks=(n_tiles, 256, 256), compressors=CODEC)
    root.attrs.update({"schema": "spec-077-teacher-prior", "build": "test_build"})
    table = pa.table({
        "build": ["test_build"] * n_tiles,
        "map_name": ["Test"] * n_tiles,
        "map": ["Test"] * n_tiles,
        "tile_id": list(range(n_tiles)),
        "tile_x": list(range(n_tiles)),
        "tile_y": list(range(n_tiles)),
    })
    pq.write_table(table, str(prior_path / "tiles.parquet"))
    _make_v18_store(v18_path, n_tiles=n_tiles)

    ds = HeightOnlyPriorDataset(
        prior_path=prior_path, v18_path=v18_path, height_norm=False,
        augment=True, augment_seed=123, augment_transforms=ALL_TRANSFORMS,
    )
    ds_plain = HeightOnlyPriorDataset(prior_path=prior_path, v18_path=v18_path, height_norm=False)
    changed = False
    for _ in range(20):
        aug = ds[0]["input_prior"].numpy()
        plain = ds_plain[0]["input_prior"].numpy()
        if not np.array_equal(aug, plain):
            changed = True
            break
    assert changed, "explicit D4 augmentation never produced a different sample"


# --- _AugmentGuardSubset --------------------------------------------------

def test_augment_guard_subset_disables_augment_during_getitem() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        prior_path = root / "prior.zarr"
        v18_path = root / "v18.zarr"
        _make_prior_store(prior_path, n_tiles=4)
        _make_v18_store(v18_path, n_tiles=4)
        base = HeightOnlyPriorDataset(
            prior_path=prior_path, v18_path=v18_path, height_norm=False,
            augment=True, augment_seed=999, augment_transforms=ALL_TRANSFORMS,
        )
        train_sub, val_sub = _split_train_val(base, val_fraction=0.5, seed=0)
        guarded = _AugmentGuardSubset(val_sub)
        # The guarded val read must equal a plain (non-augmented) read of the
        # same underlying index, even though base.augment is True.
        plain = HeightOnlyPriorDataset(prior_path=prior_path, v18_path=v18_path, height_norm=False)
        val_idx = val_sub.indices[0]
        guarded_sample = guarded[0]
        plain_sample = plain[val_idx]
        np.testing.assert_array_equal(
            guarded_sample["input_prior"].numpy(),
            plain_sample["input_prior"].numpy(),
        )
        # After the guarded read, base.augment must be restored to True.
        assert base.augment is True


# --- map-grouped split ----------------------------------------------------

def test_split_by_map_holds_out_entire_maps() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        prior_path = root / "prior.zarr"
        v18_path = root / "v18.zarr"
        # 6 tiles: 2 maps with 3 tiles each.
        _make_prior_store(prior_path, n_tiles=6, maps=["A", "A", "A", "B", "B", "B"])
        _make_v18_store(v18_path, n_tiles=6)
        ds = HeightOnlyPriorDataset(prior_path=prior_path, v18_path=v18_path, height_norm=False)
        train, val = _split_train_val_by_map(ds, val_fraction=0.5, seed=0)
        train_maps = {ds[i]["meta_map"] for i in train.indices}
        val_maps = {ds[i]["meta_map"] for i in val.indices}
        # No map may appear in both splits.
        assert train_maps.isdisjoint(val_maps)
        assert len(val) > 0
        assert len(train) > 0


def test_split_by_map_falls_back_to_random_without_metadata() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        prior_path = root / "prior.zarr"
        v18_path = root / "v18.zarr"
        _make_prior_store(prior_path, n_tiles=4, maps=["", "", "", ""])
        _make_v18_store(v18_path, n_tiles=4)
        ds = HeightOnlyPriorDataset(prior_path=prior_path, v18_path=v18_path, height_norm=False)
        train, val = _split_train_val_by_map(ds, val_fraction=0.5, seed=0)
        # Fallback random split still partitions the dataset.
        assert len(train) + len(val) == 4
        assert len(val) > 0
