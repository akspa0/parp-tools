"""Spec 116 US5 bridge tests: v116-structure-store-v1 -> v115-feature-map-v1.

Verifies:
- Dry run returns a plan and writes nothing.
- The written store has the exact schema/array/shape the geometry trainer's --feature-store
  validates against.
- Every pixel's class distribution sums to ~1 (a valid probability distribution).
- The predicted class keeps its confidence as probability mass.
- source_row_index is carried through unchanged from the structure store's own index.
- A non-v116-structure-store-v1 source is refused.
- Refuses to overwrite a non-empty output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from harvester.spec116.structure_feature_bridge import (
    StructureFeatureBridgeError,
    structure_to_feature_map,
)
from harvester.spec116.structure_materialize import materialize_structure
from harvester.spec116.structure_model import build_structure_model
from harvester.v50.terrain_feature_labels import CLASS_COUNT, TAXONOMY_REVISION
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


@pytest.fixture
def structure_store_fixture(tmp_path: Path) -> Path:
    """A real v116-structure-store-v1 built via materialize_structure (not hand-rolled)."""
    pytest.importorskip("torch")
    source = tmp_path / "source.zarr"
    source.mkdir()
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
    build_store(source, rows=rows)
    dump = tmp_path / "names.json"
    write_texture_name_dump(dump, "Kalimdor", tiles_meta)

    ckpt = tmp_path / "ckpt.pt"
    _save_checkpoint(ckpt, slot=1, base=4)

    derived = tmp_path / "structure.zarr"
    materialize_structure(
        store=source, checkpoint=ckpt, output=derived, dumps=[dump],
        slot=1, base=4, device="cpu", write=True,
    )
    return derived


class TestDryRun:
    def test_dry_run_returns_plan_and_writes_nothing(
        self, structure_store_fixture: Path, tmp_path: Path,
    ) -> None:
        output = tmp_path / "feature_map.zarr"
        plan = structure_to_feature_map(structure_store=structure_store_fixture, output=output, write=False)
        assert plan["schema"] == "v116-structure-feature-bridge-plan-v1"
        assert not output.exists()


class TestWriteFeatureMapStore:
    def test_output_matches_geometry_trainer_contract(
        self, structure_store_fixture: Path, tmp_path: Path,
    ) -> None:
        """Exactly what direct_geometry_train.py's --feature-store validates: schema, array, shape."""
        import zarr

        output = tmp_path / "feature_map.zarr"
        result = structure_to_feature_map(structure_store=structure_store_fixture, output=output, write=True)
        assert result["schema"] == "v115-feature-map-v1"

        group = zarr.open_group(str(output), mode="r")
        attrs = dict(group.attrs)
        assert attrs["schema"] == "v115-feature-map-v1"
        assert attrs["class_count"] == CLASS_COUNT
        assert "feature_map" in group
        assert group["feature_map"].shape == (2, CLASS_COUNT, 256, 256)

    def test_every_pixel_is_a_valid_distribution(
        self, structure_store_fixture: Path, tmp_path: Path,
    ) -> None:
        import zarr

        output = tmp_path / "feature_map.zarr"
        structure_to_feature_map(structure_store=structure_store_fixture, output=output, write=True)
        group = zarr.open_group(str(output), mode="r")
        feature_map = np.asarray(group["feature_map"][0], dtype=np.float32)  # (K, 256, 256)
        totals = feature_map.sum(axis=0)
        assert np.allclose(totals, 1.0, atol=1e-2)

    def test_predicted_class_keeps_confidence_as_mass(
        self, structure_store_fixture: Path, tmp_path: Path,
    ) -> None:
        import zarr

        output = tmp_path / "feature_map.zarr"
        structure_to_feature_map(structure_store=structure_store_fixture, output=output, write=True)

        source_group = zarr.open_group(str(structure_store_fixture), mode="r")
        family = np.asarray(source_group["structure_family"][0])  # (16, 16)
        confidence = np.asarray(source_group["structure_confidence"][0], dtype=np.float32)

        out_group = zarr.open_group(str(output), mode="r")
        feature_map = np.asarray(out_group["feature_map"][0], dtype=np.float32)  # (K, 256, 256)

        cy, cx = 0, 0
        predicted_family = int(family[cy, cx])
        pixel_probs = feature_map[:, cy * 16, cx * 16]
        assert pixel_probs[predicted_family] == pytest.approx(float(confidence[cy, cx]), abs=1e-2)

    def test_source_row_index_carried_through(
        self, structure_store_fixture: Path, tmp_path: Path,
    ) -> None:
        import pyarrow.parquet as pq

        output = tmp_path / "feature_map.zarr"
        structure_to_feature_map(structure_store=structure_store_fixture, output=output, write=True)
        derived_index = pq.read_table(output / "index.parquet").to_pylist()
        assert [row["source_row_index"] for row in derived_index] == [0, 1]

    def test_refuses_overwrite_non_empty(self, structure_store_fixture: Path, tmp_path: Path) -> None:
        output = tmp_path / "feature_map.zarr"
        output.mkdir()
        (output / "stale.txt").write_text("stale", encoding="utf-8")
        with pytest.raises(StructureFeatureBridgeError, match="refusing to overwrite"):
            structure_to_feature_map(structure_store=structure_store_fixture, output=output, write=True)


class TestValidation:
    def test_wrong_schema_refused(self, tmp_path: Path) -> None:
        import zarr

        not_a_structure_store = tmp_path / "wrong.zarr"
        not_a_structure_store.mkdir()
        group = zarr.open_group(str(not_a_structure_store), mode="w")
        group.attrs["schema"] = "v50-mixed-curriculum-v1"
        with pytest.raises(StructureFeatureBridgeError, match="not a"):
            structure_to_feature_map(
                structure_store=not_a_structure_store, output=tmp_path / "out.zarr", write=False,
            )
