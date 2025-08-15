# PLAN: Fixing Surface Completeness via MSLK-driven assembly and MSCN meshing

Date: 2025-08-09

## Simplified Baseline: Composite‑Hierarchy Per‑Object

- Default assembler: Composite‑Hierarchy (more accurate; avoids merging unrelated objects).
- Recommended run: enable `--include-adjacent` to ensure cross‑tile MSCN remap is applied.
- Keep diagnostics stable: `surfaces.csv`, `surface_summary.csv`, `mscn_vertices.csv`, `mslk_links.csv`, `assembly_coverage.csv`.

### Runbook

- Per‑object via composite‑hierarchy:
  - `pm4next-export <pm4|dir> --include-adjacent --format obj --assembly composite-hierarchy --csv-diagnostics`
- Alternative (AA-byte grouping only):
  - `pm4next-export <pm4|dir> --include-adjacent --format obj --assembly surface-key-aa --csv-diagnostics`

### Notes
- Cross‑tile MSCN remap is applied in `Pm4GlobalTileLoader` when `--include-adjacent` is set.
- OBJ exporter swaps face winding on X mirror to preserve orientation.
- SurfaceKey grouping may merge unrelated objects into the same OBJ; prefer composite‑hierarchy for accuracy.

## Goals
- Ensure every relevant surface/triangle for each object instance is exported.
- Stop merging multiple instances of the same object into one OBJ unless requested.
- Keep MSCN geometry correctly remapped and included where referenced.

## Current State (relevant)
- `Pm4GlobalTileLoader.ApplyMscnRemap()` appends MSCN vertices and remaps out-of-bounds indices per-tile using `TileVertexOffsets`/`TileIndexOffsets`.
- `Pm4Adapter` sets `scene.Indices = MSVI?.Indices ?? MSPI?.Indices`, so assemblers can use a single source of indices regardless of source chunk.
- Composite-key assemblers exist (`SurfaceKeyAssembler` AABB, `CompositeHierarchyAssembler` AABBCC, `SurfaceKeyAAAssembler` AA), but composite-hierarchy merges repeated instances into a single OBJ.
- `MSLK` (`parpToolbox/Formats/P4/Chunks/Common/MSLK.cs`) provides per-instance linkage: `ParentId`, `MspiFirstIndex` (signed 24-bit), `MspiIndexCount`, tile coords (`0xFFFFYYXX`), `SurfaceRefIndex`. This is the key to per-instance extraction.

## Hypotheses for missing surfaces
- **H1: MSLK indexing semantics**: `MspiIndexCount` can denote triangles (not indices) in some sources; index spans may be mis-derived.
- **H2: Tile-local vs global offsets**: Without reconstructing tile-local index before tests, MSCN references slip through. (Per-tile MSCN remap implemented to address this.)
- **H3: Surface correlation**: `SurfaceRefIndex` is tile-local; mis-joins across tiles or ignoring it may hide surfaces in composite-key grouping.
- **H4: Mixed MSVI/MSPI paths**: Some tiles are MSPI-only; fallback is correct but needs end-to-end validation.
- **H5: Off-by-one/sign**: 24-bit signed `MspiFirstIndex` or count arithmetic may drop first/last triangles.

## Strategy
Shift default assembly to per-instance, driven by MSLK, then optionally sub-slice by composite-key high 24 bits. Validate coverage with diagnostics.

## Implementation Steps

1) Diagnostics (foundation)
- Add `DiagnosticsService.WriteMslkCsv(scene, path)` with columns:
  - `parentId, tileY, tileX, mspiFirstIndex, mspiIndexCount, surfaceRefIndex, decodedTileOk, derivedStartIndex, derivedCount, indexSemanticsUsed (indices|tris), spansValid`.
- Add per-assembly coverage report `assembly_coverage.csv`:
  - Per OBJ/instance: `expectedIndexTotal` (sum of derived MSLK spans), `exportedIndexTotal` (triangles*3), `coveragePct`, `missingRangesSample`.
- Continue emitting `surfaces.csv` and `mscn_vertices.csv` and logging “MSCN remap: appended X, remapped Y”.

2) MSLK-driven assembler
- Create `PM4NextExporter/Assembly/MslkInstanceAssembler.cs`:
  - Group by instance key: `(ParentId, tileY, tileX)`; optionally include `Type_0x01`/`SortKey_0x02` if collisions.
  - For each `MslkEntry` with `HasGeometry` and `TryDecodeTileCoordinates()`:
    - Map tile coord → `TileIndexOffsets` + `TileVertexOffsets`.
    - Derive span:
      - Path A (indices mode): `start = FirstIndex`, `count = MspiIndexCount`.
      - Path B (triangles mode): `start = FirstIndex*3`, `count = MspiIndexCount*3`.
      - Choose by heuristic: prefer indices if `(start+count)` is 3-multiple and in-bounds; otherwise triangles.
    - Global span: `[tileIndexStart + start .. + count)`.
  - Collect triangles from `scene.Indices` per span; map global → local vertices.
  - Name: `inst_{ParentId:X8}_y{YY}_x{XX}.obj`.
  - Variant: `--assembly mslk-instance+ck24` splits inside an instance by CompositeKey top-24 (`AABBCC`).

3) Loader/data exposure (if needed)
- Ensure `Pm4GlobalTileLoader` exposes `TileVertexOffsets`/`TileIndexOffsets` in returned scene metadata consumed by assemblers.

4) Coverage checks
- For each instance build:
  - Compute coverage vs derived MSLK spans.
  - Report missing ranges; if substantial, attempt alternate span semantics (switch indices/triangles) and re-evaluate.
  - Optionally validate `SurfaceRefIndex` membership (diagnostic-only): some referenced surfaces appear in the assembled OBJ.

5) CLI wiring
- Add `--assembly mslk-instance` and `--assembly mslk-instance+ck24`.
- Add `--csv-diagnostics` emission of `mslk_links.csv` and `assembly_coverage.csv`.

6) Validation
- **Instance separation**: Objects previously merged in composite-hierarchy appear as separate OBJs per `(ParentId, YY, XX)`.
- **MSCN inclusion**: Non-zero “remapped indices” when MSCN is referenced; visual inspection confirms MSCN geometry present.
- **Surface completeness**: `coveragePct` ~ 100%; MeshLab shows no large missing bands.

## Acceptance Criteria
- Each repeated object appears as separate OBJ per instance by default.
- `mslk_links.csv` present; spans are in-bounds and consistent on sampled instances.
- `assembly_coverage.csv` shows ≥ 98% coverage on test instances; no glaring holes.
- Logs confirm MSCN remap activity; OBJs include MSCN-derived geometry where applicable.
- Optional `+ck24` provides consistent sub-slicing within instances.

## Risks & Mitigations
- **Ambiguous MSLK count semantics**: Heuristic may misclassify. Mitigation: coverage auto-switch + diagnostic flag.
- **Cross-tile anomalies**: If spans cross tiles, detect via OOB spans and log; future: stitch multi-tile spans if observed.
- **SurfaceRefIndex mapping**: Use diagnostically only; geometry selection stays span-driven to avoid brittle joins.

## Deliverables
- `MslkInstanceAssembler.cs`, CLI options.
- Diagnostics CSVs: `mslk_links.csv`, `assembly_coverage.csv`.
- This plan document and a short validation guide.

## How to Validate (quick)
- Run include-adjacent on a known area:
  - `--assembly mslk-instance --csv-diagnostics`
- Inspect `assembly_coverage.csv` (coverage ~100%).
- Open 2–3 instance OBJs in MeshLab; confirm no large fragments missing and MSCN presence.
- Compare vs composite-hierarchy outputs: new per-instance outputs should not merge placements.
