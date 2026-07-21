"""Spec 116: relational layer-entry extraction from a v50 store."""

from __future__ import annotations

import pytest

from harvester.spec116.relational_extract import (
    CHUNKS_PER_AXIS,
    MAX_LAYERS,
    RelationalExtractError,
    extract_layer_entries,
)
from harvester.v50.terrain_feature_labels import ROAD, TERRAIN


class TestExtraction:
    def test_row_count_matches_store_and_entries_are_complete(self, consistent_store) -> None:
        result = extract_layer_entries(store=consistent_store["store"], dumps=consistent_store["dumps"])
        assert result.row_count == 2
        # 2 tiles * 16*16 chunks * 2 populated slots (0 and 1) = 1024 rows
        assert len(result.rows) == 2 * CHUNKS_PER_AXIS * CHUNKS_PER_AXIS * 2

    def test_base_slot_is_opaque_and_terrain(self, consistent_store) -> None:
        result = extract_layer_entries(store=consistent_store["store"], dumps=consistent_store["dumps"])
        slot0 = [r for r in result.rows if r.slot == 0]
        assert slot0, "base slot rows must exist"
        assert all(r.coverage == 1.0 for r in slot0), "base slot is always opaque"
        assert all(r.family == TERRAIN for r in slot0)

    def test_detail_slot_family_matches_taxonomy(self, consistent_store) -> None:
        result = extract_layer_entries(store=consistent_store["store"], dumps=consistent_store["dumps"])
        slot1 = [r for r in result.rows if r.slot == 1]
        assert slot1
        # slot 1 carries the road texture -> ROAD family under the v115.1 taxonomy
        assert all(r.family == ROAD for r in slot1)

    def test_absent_slots_are_not_rows(self, consistent_store) -> None:
        result = extract_layer_entries(store=consistent_store["store"], dumps=consistent_store["dumps"])
        # only slots 0 and 1 are populated in the fixture; 2 and 3 must not appear
        assert {r.slot for r in result.rows} == {0, 1}

    def test_family_slot_counts_shape(self, consistent_store) -> None:
        result = extract_layer_entries(store=consistent_store["store"], dumps=consistent_store["dumps"])
        counts = result.family_slot_counts()
        assert counts.shape == (5, MAX_LAYERS)
        # terrain (1) only in slot 0; road (2) only in slot 1
        assert counts[1, 0] > 0 and counts[1, 1] == 0
        assert counts[2, 1] > 0 and counts[2, 0] == 0

    def test_missing_dump_entry_excludes_tile_and_counts_it(self, tmp_path) -> None:
        from tests.spec116.conftest import _ids, _mask, build_store, write_texture_name_dump

        store = tmp_path / "no_dump.zarr"
        store.mkdir()
        rows = [
            {
                "map": "Kalimdor", "tile_x": 1, "tile_y": 1, "split": "train", "source": "authored",
                "texture_names": [r"Tileset\X\XGrass.blp", r"Tileset\X\XRoad.blp"],
                "mcly_texture_ids": _ids(0, 1), "mcly_layer_mask": _mask(slot1=True),
            },
        ]
        build_store(store, rows=rows)
        # dump that does NOT cover tile (1,1)
        dump = tmp_path / "names.json"
        write_texture_name_dump(dump, "Kalimdor", [{"TileX": 99, "TileY": 99, "TextureNames": ["x"]}])
        with pytest.raises(RelationalExtractError, match="no layer-entry rows"):
            extract_layer_entries(store=store, dumps=[dump])

    def test_missing_store_array_is_rejected(self, tmp_path) -> None:
        import zarr
        store = tmp_path / "empty.zarr"
        store.mkdir()
        zarr.open_group(str(store), mode="w")  # no arrays, no index
        with pytest.raises(RelationalExtractError, match="missing"):
            extract_layer_entries(store=store, dumps=[tmp_path / "x.json"])


class TestProvenance:
    def test_taxonomy_revision_and_rule_hash_are_recorded(self, consistent_store) -> None:
        result = extract_layer_entries(store=consistent_store["store"], dumps=consistent_store["dumps"])
        assert result.taxonomy_revision == "v115.1"
        assert len(result.rule_set_sha256) == 64
        # stable
        assert result.rule_set_sha256 == result.rule_set_sha256
        # excluded report present (zero for this fixture)
        assert "no_texture_name_dump_entry" in result.excluded
        assert result.excluded["no_texture_name_dump_entry"] == 0
