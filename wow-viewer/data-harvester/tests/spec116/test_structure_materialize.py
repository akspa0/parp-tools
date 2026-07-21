"""Spec 116 US5 T029: structure materialization tests.

Verifies:
- The derived store is row-aligned with the source.
- The source store is untouched.
- The checkpoint sha256 is recorded in the store attrs.
- A taxonomy mismatch is refused.
- A slot mismatch is refused.
- Dry run returns a plan and writes nothing.
- Refuses to overwrite a non-empty output.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from harvester.spec116.structure_materialize import (
    StructureMaterializeError,
    materialize_structure,
)
from harvester.spec116.structure_model import CHUNK_GRID, build_structure_model
from harvester.v50.terrain_feature_labels import TAXONOMY_REVISION
from tests.spec116.conftest import build_store, write_texture_name_dump

CHUNKS = 16


def _save_checkpoint(path: Path, *, slot: int, base: int = 4) -> None:
    import torch

    model, _ = build_structure_model(slot=slot, base=base)
    torch.save(
        {
            "model": model.state_dict(),
            "slot": slot, "base": base,
            "taxonomy_revision": TAXONOMY_REVISION,
            "num_classes": 5, "epoch": 0, "macro_iou": 0.0, "metrics": {},
        },
        path,
    )


def _build_source_store(tmp_path: Path) -> tuple[Path, Path]:
    """Build a 2-tile source store with a texture-name dump."""
    store = tmp_path / "source.zarr"
    store.mkdir()
    rows = []
    tiles_meta = []
    for x in range(2):
        ids = np.full((CHUNKS, CHUNKS, 4), -1, dtype=np.int32)
        ids[:, :, 0] = 0
        ids[:, :, 1] = 1
        mask = np.zeros((CHUNKS, CHUNKS, 4), dtype=np.float32)
        mask[:, :, 0] = 1.0
        mask[:, :, 1] = 1.0
        rows.append({
            "map": "Kalimdor", "tile_x": x, "tile_y": 0,
            "split": "train", "source": "authored",
            "texture_names": [r"Tileset\X\XGrass.blp", r"Tileset\X\XRoad.blp"],
            "mcly_texture_ids": ids, "mcly_layer_mask": mask,
        })
        tiles_meta.append({"TileX": x, "TileY": 0,
                           "TextureNames": [r"Tileset\X\XGrass.blp", r"Tileset\X\XRoad.blp"]})
    build_store(store, rows=rows)
    dump = tmp_path / "names.json"
    write_texture_name_dump(dump, "Kalimdor", tiles_meta)
    return store, dump


@pytest.fixture
def source_fixture(tmp_path: Path) -> dict:
    store, dump = _build_source_store(tmp_path)
    ckpt = tmp_path / "ckpt.pt"
    _save_checkpoint(ckpt, slot=1, base=4)
    return {"store": store, "dump": dump, "checkpoint": ckpt}


class TestDryRun:
    def test_dry_run_returns_plan_and_writes_nothing(self, source_fixture: dict, tmp_path: Path) -> None:
        output = tmp_path / "derived.zarr"
        result = materialize_structure(
            store=source_fixture["store"],
            checkpoint=source_fixture["checkpoint"],
            output=output,
            dumps=[source_fixture["dump"]],
            slot=1, base=4, device="cpu", write=False,
        )
        assert result["schema"] == "v116-structure-materialize-plan-v1"
        assert not output.exists()


class TestWriteDerivedStore:
    def test_derived_store_is_row_aligned(self, source_fixture: dict, tmp_path: Path) -> None:
        """The derived store has the same row count as the source."""
        pytest.importorskip("torch")
        import pyarrow.parquet as pq
        import zarr

        output = tmp_path / "derived.zarr"
        materialize_structure(
            store=source_fixture["store"],
            checkpoint=source_fixture["checkpoint"],
            output=output,
            dumps=[source_fixture["dump"]],
            slot=1, base=4, device="cpu", write=True,
        )

        group = zarr.open_group(str(output), mode="r")
        assert group["structure_family"].shape == (2, CHUNK_GRID, CHUNK_GRID)
        assert group["structure_confidence"].shape == (2, CHUNK_GRID, CHUNK_GRID)
        assert group["structure_legal"].shape == (2, CHUNK_GRID, CHUNK_GRID)

        derived_index = pq.read_table(output / "index.parquet").to_pylist()
        source_index = pq.read_table(source_fixture["store"] / "index.parquet").to_pylist()
        assert len(derived_index) == len(source_index)
        assert derived_index[0]["source_row_index"] == 0
        assert derived_index[1]["source_row_index"] == 1

    def test_source_store_untouched(self, source_fixture: dict, tmp_path: Path) -> None:
        """The source store is never mutated."""
        pytest.importorskip("torch")
        import zarr

        # Read source arrays before.
        source_group = zarr.open_group(str(source_fixture["store"]), mode="r")
        source_minimap_before = np.asarray(source_group["minimap_rgb"][:])

        output = tmp_path / "derived.zarr"
        materialize_structure(
            store=source_fixture["store"],
            checkpoint=source_fixture["checkpoint"],
            output=output,
            dumps=[source_fixture["dump"]],
            slot=1, base=4, device="cpu", write=True,
        )

        # Source unchanged.
        source_group_after = zarr.open_group(str(source_fixture["store"]), mode="r")
        source_minimap_after = np.asarray(source_group_after["minimap_rgb"][:])
        assert np.array_equal(source_minimap_before, source_minimap_after)

    def test_checkpoint_sha256_recorded(self, source_fixture: dict, tmp_path: Path) -> None:
        """The checkpoint sha256 is recorded in the store attrs."""
        pytest.importorskip("torch")
        import zarr

        output = tmp_path / "derived.zarr"
        materialize_structure(
            store=source_fixture["store"],
            checkpoint=source_fixture["checkpoint"],
            output=output,
            dumps=[source_fixture["dump"]],
            slot=1, base=4, device="cpu", write=True,
        )

        group = zarr.open_group(str(output), mode="r")
        attrs = dict(group.attrs)
        assert attrs["schema"] == "v116-structure-store-v1"
        assert "checkpoint_sha256" in attrs
        assert len(attrs["checkpoint_sha256"]) == 64
        assert attrs["slot"] == 1
        assert attrs["taxonomy_revision"] == TAXONOMY_REVISION

    def test_refuses_overwrite_non_empty(self, source_fixture: dict, tmp_path: Path) -> None:
        """Refusing to overwrite a non-empty output directory."""
        pytest.importorskip("torch")
        output = tmp_path / "derived.zarr"
        output.mkdir()
        (output / "stale.txt").write_text("stale", encoding="utf-8")
        with pytest.raises(StructureMaterializeError, match="refusing to overwrite"):
            materialize_structure(
                store=source_fixture["store"],
                checkpoint=source_fixture["checkpoint"],
                output=output,
                dumps=[source_fixture["dump"]],
                slot=1, base=4, device="cpu", write=True,
            )


class TestValidation:
    def test_taxonomy_mismatch_refused(self, source_fixture: dict, tmp_path: Path) -> None:
        """A checkpoint with a different taxonomy revision is refused."""
        pytest.importorskip("torch")
        import torch

        ckpt_bad = tmp_path / "bad_ckpt.pt"
        model, _ = build_structure_model(slot=1, base=4)
        torch.save(
            {
                "model": model.state_dict(),
                "slot": 1, "base": 4,
                "taxonomy_revision": "v999.0",
                "num_classes": 5, "epoch": 0, "macro_iou": 0.0, "metrics": {},
            },
            ckpt_bad,
        )
        with pytest.raises(StructureMaterializeError, match="taxonomy"):
            materialize_structure(
                store=source_fixture["store"],
                checkpoint=ckpt_bad,
                output=tmp_path / "derived.zarr",
                dumps=[source_fixture["dump"]],
                slot=1, base=4, device="cpu", write=False,
            )

    def test_slot_mismatch_refused(self, source_fixture: dict, tmp_path: Path) -> None:
        """A checkpoint for slot 2 cannot materialize slot 1."""
        pytest.importorskip("torch")
        ckpt_slot2 = tmp_path / "slot2_ckpt.pt"
        _save_checkpoint(ckpt_slot2, slot=2, base=4)
        with pytest.raises(StructureMaterializeError, match="slot"):
            materialize_structure(
                store=source_fixture["store"],
                checkpoint=ckpt_slot2,
                output=tmp_path / "derived.zarr",
                dumps=[source_fixture["dump"]],
                slot=1, base=4, device="cpu", write=False,
            )