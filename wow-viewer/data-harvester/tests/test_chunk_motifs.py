from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import zarr

from harvester.chunk_motifs import build_chunk_cells, extract_chunk_motifs, write_motif_outputs


@dataclass(frozen=True)
class _Record:
    build: str
    map_name: str
    tile_id: int
    tile_x: int
    tile_y: int


def _activate_cell(alpha: np.ndarray, tile_id: int, chunk_x: int, chunk_y: int) -> None:
    y0, x0 = chunk_y * 16, chunk_x * 16
    checker = (np.indices((16, 16)).sum(axis=0) % 2).astype(np.float32)
    alpha[tile_id, y0 : y0 + 16, x0 : x0 + 16, 1] = checker


def test_finds_repeated_irregular_graphs_across_tile_border(tmp_path) -> None:
    root = zarr.open_group(str(tmp_path / "motifs.zarr"), mode="w")
    alpha = np.zeros((4, 256, 256, 4), dtype=np.float32)
    height = np.zeros((4, 257, 257), dtype=np.float32)
    root.create_array("alpha_256", data=alpha)
    root.create_array("height_257", data=height)
    records = [_Record("0_5_3_3368", "Azeroth", tile_id, tile_id, 0) for tile_id in range(4)]

    # Same four-cell stair, twice.  The first instance crosses tile 0 -> tile 1.
    for global_x, global_y in ((15, 7), (16, 7), (16, 8), (17, 8), (47, 7), (48, 7), (48, 8), (49, 8)):
        tile_id, chunk_x = divmod(global_x, 16)
        _activate_cell(root["alpha_256"], tile_id, chunk_x, global_y)

    cells = build_chunk_cells(root, records)
    motifs = extract_chunk_motifs(cells, max_hops=3, max_cells=8)
    summary = write_motif_outputs(tmp_path / "output", motifs, min_occurrences=2)

    assert any(motif.crosses_tile_border for motif in motifs)
    assert all(motif.cell_count < motif.bbox_chunk_xywh[2] * motif.bbox_chunk_xywh[3] for motif in motifs)
    assert summary["repeated_family_count"] >= 1
    assert (tmp_path / "output" / "motif_families.parquet").exists()
