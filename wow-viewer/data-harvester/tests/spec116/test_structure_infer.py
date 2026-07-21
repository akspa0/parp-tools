"""Spec 116 US3 T026: structure inference contract tests.

Verifies:
- With a tile MTEX table, all emitted references are legal (SC-004).
- With no table (OOD), no reference is fabricated and the audit record flags it (D-05).
- A low-confidence chunk (no same-family texture in the table) is reported.
- The audit record validates against ``v50-structure-infer-v1``.
- A taxonomy mismatch between checkpoint and code is refused.
- A slot mismatch between checkpoint and request is refused.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from harvester.spec116.structure_contract import validate_structure_infer
from harvester.spec116.structure_infer import (
    StructureInferError,
    infer_structure,
    infer_structure_images,
    resolve_legal_reference,
)
from harvester.spec116.structure_model import build_structure_model
from harvester.v50.terrain_feature_labels import FAMILY_NAMES, TAXONOMY_REVISION
from tests.spec116.conftest import build_store, write_texture_name_dump

CHUNKS = 16


def _save_checkpoint(path: Path, *, slot: int, base: int = 4) -> None:
    """Save a random-weight StructureSlotNet checkpoint for testing."""
    import torch

    model, _ = build_structure_model(slot=slot, base=base)
    torch.save(
        {
            "model": model.state_dict(),
            "slot": slot,
            "base": base,
            "taxonomy_revision": TAXONOMY_REVISION,
            "num_classes": 5,
            "epoch": 0,
            "macro_iou": 0.0,
            "metrics": {},
        },
        path,
    )


def _build_infer_store(tmp_path: Path, *, with_dump: bool = True) -> tuple[Path, list[Path] | list]:
    """Build a 2-tile store with grass+road textures; optionally with a dump."""
    store = tmp_path / "infer.zarr"
    store.mkdir()
    rows = []
    tiles_meta = []
    for i, (x, y) in enumerate([(0, 0), (1, 0)]):
        ids = np.full((CHUNKS, CHUNKS, 4), -1, dtype=np.int32)
        ids[:, :, 0] = 0  # grass
        ids[:, :, 1] = 1  # road
        mask = np.zeros((CHUNKS, CHUNKS, 4), dtype=np.float32)
        mask[:, :, 0] = 1.0
        mask[:, :, 1] = 1.0
        rows.append({
            "map": "Kalimdor", "tile_x": x, "tile_y": y,
            "split": "train", "source": "authored",
            "texture_names": [r"Tileset\X\XGrass.blp", r"Tileset\X\XRoad.blp"],
            "mcly_texture_ids": ids, "mcly_layer_mask": mask,
        })
        tiles_meta.append({"TileX": x, "TileY": y,
                           "TextureNames": [r"Tileset\X\XGrass.blp", r"Tileset\X\XRoad.blp"]})
    build_store(store, rows=rows)

    if with_dump:
        dump = tmp_path / "names.json"
        write_texture_name_dump(dump, "Kalimdor", tiles_meta)
        return store, [dump]
    return store, []


class TestResolveLegalReference:
    def test_finds_matching_family(self) -> None:
        names = [r"Tileset\X\XGrass.blp", r"Tileset\X\XRoad.blp"]
        # Road is family 2 (ROAD). The resolver should find local_id=1.
        ref = resolve_legal_reference(2, names)
        assert ref == 1

    def test_finds_terrain_family(self) -> None:
        names = [r"Tileset\X\XGrass.blp", r"Tileset\X\XRoad.blp"]
        # Grass is family 1 (TERRAIN). The resolver should find local_id=0.
        ref = resolve_legal_reference(1, names)
        assert ref == 0

    def test_returns_none_when_no_match(self) -> None:
        names = [r"Tileset\X\XGrass.blp", r"Tileset\X\XRoad.blp"]
        # Water (family 3) is not in this table.
        ref = resolve_legal_reference(3, names)
        assert ref is None

    def test_returns_none_for_empty_table(self) -> None:
        ref = resolve_legal_reference(1, [])
        assert ref is None


class TestInferWithTable:
    def test_all_references_legal(self, tmp_path: Path) -> None:
        """SC-004: with a table, all emitted references are legal."""
        pytest.importorskip("torch")
        store, dumps = _build_infer_store(tmp_path, with_dump=True)
        ckpt = tmp_path / "ckpt.pt"
        _save_checkpoint(ckpt, slot=1, base=4)

        record = infer_structure(
            checkpoint=ckpt, store=store, dumps=dumps, slot=1, base=4, device="cpu",
        )
        validate_structure_infer(record)
        assert record["legal_table_available"] is True
        # Every tile has a table.
        for tile in record["per_tile"]:
            assert tile["legal_table_available"] is True
            # Every reference is either a valid local_id or None (low-confidence).
            if tile["references"] is not None:
                for row in tile["references"]:
                    for ref in row:
                        if ref is not None:
                            assert 0 <= ref < 2  # valid local_id in the 2-entry table

    def test_low_confidence_chunk_reported(self, tmp_path: Path) -> None:
        """A chunk whose predicted family has no match in the table is reported as low-confidence."""
        pytest.importorskip("torch")
        # Build a store with only grass in the table (no road).
        store = tmp_path / "grass_only.zarr"
        store.mkdir()
        rows = []
        tiles_meta = []
        for x in range(2):
            ids = np.full((CHUNKS, CHUNKS, 4), -1, dtype=np.int32)
            ids[:, :, 0] = 0  # grass only
            mask = np.zeros((CHUNKS, CHUNKS, 4), dtype=np.float32)
            mask[:, :, 0] = 1.0
            rows.append({
                "map": "Kalimdor", "tile_x": x, "tile_y": 0,
                "split": "train", "source": "authored",
                "texture_names": [r"Tileset\X\XGrass.blp"],
                "mcly_texture_ids": ids, "mcly_layer_mask": mask,
            })
            tiles_meta.append({"TileX": x, "TileY": 0, "TextureNames": [r"Tileset\X\XGrass.blp"]})
        build_store(store, rows=rows)
        dump = tmp_path / "grass_names.json"
        write_texture_name_dump(dump, "Kalimdor", tiles_meta)

        ckpt = tmp_path / "ckpt.pt"
        _save_checkpoint(ckpt, slot=1, base=4)

        record = infer_structure(
            checkpoint=ckpt, store=store, dumps=[dump], slot=1, base=4, device="cpu",
        )
        validate_structure_infer(record)
        # The model may predict any family; if it predicts non-terrain, those chunks
        # have no legal reference (only grass is in the table).
        for tile in record["per_tile"]:
            assert tile["legal_table_available"] is True
            # legal + low_conf should sum to total chunks.
            assert tile["legal_reference_count"] + tile["low_confidence_count"] == tile["total_chunks"]


class TestInferOOD:
    def test_no_table_no_fabricated_reference(self, tmp_path: Path) -> None:
        """D-05: OOD (no table) → no reference fabricated, audit flags it."""
        pytest.importorskip("torch")
        store, _ = _build_infer_store(tmp_path, with_dump=True)
        ckpt = tmp_path / "ckpt.pt"
        _save_checkpoint(ckpt, slot=1, base=4)

        # Pass no dumps → OOD.
        record = infer_structure(
            checkpoint=ckpt, store=store, dumps=[], slot=1, base=4, device="cpu",
        )
        validate_structure_infer(record)
        assert record["legal_table_available"] is False
        assert record["sc004_all_references_legal"] is False
        for tile in record["per_tile"]:
            assert tile["legal_table_available"] is False
            assert tile["references"] is None
            assert tile["legal_reference_count"] == 0
            assert tile["low_confidence_count"] == tile["total_chunks"]


class TestInferImages:
    """T027: --inputs/--tile-table loose-image inference, incl. a hand-painted OOD tile."""

    def _save_png(self, path: Path, *, seed: int = 0) -> None:
        import numpy as np
        from PIL import Image

        rng = np.random.default_rng(seed)
        rgb = rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8)
        Image.fromarray(rgb, mode="RGB").save(path)

    def test_images_with_tile_table_all_legal(self, tmp_path: Path) -> None:
        """SC-004: with a tile table, all emitted references are legal."""
        pytest.importorskip("torch")
        pytest.importorskip("PIL")
        ckpt = tmp_path / "ckpt.pt"
        _save_checkpoint(ckpt, slot=1, base=4)

        tile_png = tmp_path / "tile.png"
        self._save_png(tile_png, seed=1)
        table = tmp_path / "table.json"
        table.write_text(json.dumps([r"Tileset\X\XGrass.blp", r"Tileset\X\XRoad.blp"]), encoding="utf-8")

        record = infer_structure_images(
            checkpoint=ckpt, inputs=[tile_png], tile_table=table, slot=1, base=4, device="cpu",
        )
        validate_structure_infer(record)
        assert record["legal_table_available"] is True
        assert len(record["per_tile"]) == 1
        tile = record["per_tile"][0]
        assert tile["input"] == str(tile_png)
        for row in tile["references"]:
            for ref in row:
                if ref is not None:
                    assert 0 <= ref < 2

    def test_images_no_table_is_ood_no_fabricated_reference(self, tmp_path: Path) -> None:
        """D-05: a hand-painted image with no tile table never fabricates a reference."""
        pytest.importorskip("torch")
        pytest.importorskip("PIL")
        ckpt = tmp_path / "ckpt.pt"
        _save_checkpoint(ckpt, slot=1, base=4)

        hand_painted = tmp_path / "hand_painted.png"
        self._save_png(hand_painted, seed=2)

        record = infer_structure_images(
            checkpoint=ckpt, inputs=[hand_painted], tile_table=None, slot=1, base=4, device="cpu",
        )
        validate_structure_infer(record)
        assert record["legal_table_available"] is False
        assert record["sc004_all_references_legal"] is False
        tile = record["per_tile"][0]
        assert tile["legal_table_available"] is False
        assert tile["references"] is None
        assert tile["legal_reference_count"] == 0
        assert tile["low_confidence_count"] == tile["total_chunks"]

    def test_multiple_images_and_directory_expansion(self, tmp_path: Path) -> None:
        pytest.importorskip("torch")
        pytest.importorskip("PIL")
        ckpt = tmp_path / "ckpt.pt"
        _save_checkpoint(ckpt, slot=2, base=4)

        image_dir = tmp_path / "tiles"
        image_dir.mkdir()
        self._save_png(image_dir / "a.png", seed=3)
        self._save_png(image_dir / "b.png", seed=4)

        record = infer_structure_images(
            checkpoint=ckpt, inputs=[image_dir], tile_table=None, slot=2, base=4, device="cpu",
        )
        assert len(record["per_tile"]) == 2
        assert len(record["inputs"]) == 2

    def test_cli_requires_exactly_one_of_store_or_inputs(self, tmp_path: Path) -> None:
        from harvester.spec116.structure_infer import main as infer_main

        ckpt = tmp_path / "ckpt.pt"
        _save_checkpoint(ckpt, slot=1, base=4)
        with pytest.raises(SystemExit):
            infer_main(["--checkpoint", str(ckpt), "--slot", "1"])


class TestCheckpointValidation:
    def test_taxonomy_mismatch_refused(self, tmp_path: Path) -> None:
        """A checkpoint with a different taxonomy revision is refused."""
        pytest.importorskip("torch")
        import torch

        store, dumps = _build_infer_store(tmp_path, with_dump=True)
        ckpt = tmp_path / "ckpt.pt"
        model, _ = build_structure_model(slot=1, base=4)
        torch.save(
            {
                "model": model.state_dict(),
                "slot": 1, "base": 4,
                "taxonomy_revision": "v999.0",  # mismatch
                "num_classes": 5, "epoch": 0, "macro_iou": 0.0, "metrics": {},
            },
            ckpt,
        )
        with pytest.raises(StructureInferError, match="taxonomy"):
            infer_structure(
                checkpoint=ckpt, store=store, dumps=dumps, slot=1, base=4, device="cpu",
            )

    def test_slot_mismatch_refused(self, tmp_path: Path) -> None:
        """A checkpoint for slot 2 cannot be used for slot 1 inference."""
        pytest.importorskip("torch")
        store, dumps = _build_infer_store(tmp_path, with_dump=True)
        ckpt = tmp_path / "ckpt.pt"
        _save_checkpoint(ckpt, slot=2, base=4)  # slot 2 checkpoint
        with pytest.raises(StructureInferError, match="slot"):
            infer_structure(
                checkpoint=ckpt, store=store, dumps=dumps, slot=1, base=4, device="cpu",
            )