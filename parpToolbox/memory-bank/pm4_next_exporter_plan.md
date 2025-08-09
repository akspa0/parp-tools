# PM4 Next Exporter – Full Plan

## Executive Summary
- Build a new, standalone PM4 exporter with pluggable object-assembly strategies, robust cross-tile vertex resolution, rich diagnostics (CSV/JSON), and support for multiple output formats (OBJ baseline, glTF 2.0 optional).
- Decouple from `PM4Rebuilder` experiments; maintain a clean architecture and testable modules.
- Provide multiple grouping/assembly modes for research and validation (ParentIndex_0x04, MSUR.IndexCount, ParentId 16/16 split), with CLI flags and correlation output.

## Goals and Scope
- Correct, complete object exports at building scale.
- Prevent data loss from cross-tile references (~64% loss previously).
- Facilitate analysis: grouping diagnostics, correlation matrices, per-entry snapshots.
- Export format choices: OBJ (with legacy parity mode) and glTF 2.0.
- Batch-friendly and integration-ready CLI.

Out of scope initially:
- Full interactive UI.
- Non-PM4 formats beyond minimal references.

## Repository & Location
- New project: `parpToolbox/src/PM4NextExporter/` (library + CLI entry).
- Documentation: this plan in `parpToolbox/memory-bank/pm4_next_exporter_plan.md` and appendices within `docs/formats/PM4.md`.

## Architecture Overview
- `SceneLoader`:
  - Unified PM4 load (single tile) with optional 3x3 adjacency loading (`--include-adjacent`).
  - Extract canonical MSLK fields: `Flags_0x00`, `Type_0x01`, `SortKey_0x02`, `ParentId`, `MspiFirstIndex`, `MspiIndexCount`, `TileCoordsRaw`, `SurfaceRefIndex`, `Unknown_0x12`.
  - Build vertex/index pools and indexes to chunks (MSVT/MSVI/MSUR/MSLK/MPRL/MPRR).
  - Optional: use `LinkIdDecoder` (pattern FFFFYYXX) to seed tile adjacency discovery.
- `CrossTileVertexResolver`:
  - Stage 1: audit-only (max accessed index, OOB counts, suspected cross-tile refs).
  - Stage 2: optional remapping behind flags; support high/low 16/16 pairs; expand vertex pool with adjacent tiles.
  - Strict bounds checking; skip triangles with any invalid vertex index; detailed logs.
- `ObjectAssembler` (pluggable strategies):
  - `ParentHierarchyAssembler`: Group by `MSLK.ParentIndex_0x04` (links to `MPRL.Unknown4`), handle `MspiFirstIndex == -1` containers.
  - `MsurIndexCountAssembler`: Group by `MSUR.IndexCount` (observed to produce complete objects).
  - `Parent16Assembler`: Group by `ParentId` split (16/16) with optional swap (ContainerId=high16, ObjectId=low16; `--parent16-swap` flips).
  - Future: spatial-cluster or hybrid strategies.
- `GroupingKeyService`:
  - Encapsulates diagnostic grouping key extraction for CLI `--group` modes: `parent16`, `parent16-container`, `parent16-object`, `surface`, `flags`, `type`, `sortkey`, `tile`.
- `DiagnosticsService`:
  - Per-entry CSV snapshot; group histograms; correlation CSVs `keyA:keyB`; vertex index audit CSV.
- `Exporters`:
  - `ObjExporter`: robust triangulation, invalid index filtering, coordinate fix (X inversion), legacy parity toggle.
  - `GltfExporter` (optional Phase 2): via SharpGLTF, with `--gltf`/`--glb`.
- `Logging`:
  - Tee logging (console + file) with concise, structured messages for audits and errors.

## Data Flow
PM4 file(s) → SceneLoader → ObjectAssembler (selected strategy) → CrossTileVertexResolver (audit/remap) → Exporter (OBJ/glTF) → DiagnosticsService (CSVs)

## CLI Design
- Base:
  - `pm4next-export <pm4Input|directory> [--out <dir>] [--include-adjacent] [--format obj|gltf|glb]`
- Object assembly strategy (how objects are built):
  - `--assembly parent-index|msur-indexcount|parent16`
- Diagnostic grouping (how to segment outputs/CSVs):
  - `--group parent16|parent16-container|parent16-object|surface|flags|type|sortkey|tile`
  - `--parent16-swap` usable with `--assembly parent16` and any `--group parent16*`
- Diagnostics:
  - `--csv-diagnostics`
  - `--csv-out <dir>`
  - `--correlate <keyA:keyB>` (repeatable)
  - Defaults if not provided: `(parent16-container:parent16-object)`, `(surface:type)`, `(flags:sortkey)`, `(tile:parent16-container)`
- Batch:
  - `--batch` treats input as directory and processes each `.pm4`.
- Legacy parity:
  - `--legacy-obj-parity` to match legacy OBJ naming/winding/ordering and transforms.
- Advanced:
  - `--audit-only` runs diagnostics without writing geometry.
  - `--no-remap` disables cross-tile remapping (audit-only).

## Grouping vs Assembly
- Assembly determines object coherence:
  - `parent-index`: `MSLK.ParentIndex_0x04` hierarchy (confirmed linkage to placements, `MPRL.Unknown4`).
  - `msur-indexcount`: `MSUR.IndexCount` as object identifier (produces full objects in tests).
  - `parent16`: research/validation grouping by packed `ParentId` (16/16) with swap option.
- Grouping modes are orthogonal for diagnostics and per-group OBJ splitting.

## Cross-Tile Vertex Resolution
- Problem: ~64% OOB due to indices referencing adjacent tiles.
- Stage 1 (MVP): audit-only metrics and CSV; prove adjacency need.
- Stage 2: resolution (behind flags):
  - Load 3x3 tiles; decode 32-bit indices from 16/16 pairs where applicable; remap indices.
  - Write remap table CSV; keep a safety `--no-remap` to revert to audit-only.
- Safety: skip triangles containing any invalid indices; log skipped with chunk+entry context.

## Export Formats
- OBJ:
  - Baseline; robust triangulation; `.mtl` optional; X-axis inversion; n-gon handling.
  - Legacy parity mode via `--legacy-obj-parity` for compatibility/regression.
- glTF 2.0 / GLB (Phase 2):
  - SharpGLTF integration; `--format gltf|glb`; optional simultaneous export.

## Diagnostics (CSV)
- `mslk_entries.csv`: `EntryIndex, Flags_0x00, Type_0x01, SortKey_0x02, ParentId, ContainerId, ObjectId, MspiFirstIndex, MspiIndexCount, TileCoordsRaw, SurfaceRefIndex, Unknown_0x12, HasGeometry, VertexPool, ValidIndexCount, OobIndexCount`.
- `group_histogram_<mode>.csv`: `GroupKey, EntryCount, TriangleCount, VertexCount`.
- `correlation_<A>_vs_<B>.csv`: `KeyA, KeyB, Count`.
- `vertex_index_audit.csv`: `Pool, TotalVertices, MaxIndexAccessed, OobCount`.
- Output to `<outDir>/diagnostics/<timestamp>/` or `--csv-out`.

## Quality & Known Issues Handling
- Bounds validation: prevent (0,0,0) placeholder vertices by skipping invalid triangles and logging their origin.
- Coordinate transforms: X inversion applied consistently; parity mode reproduces legacy transform chain.
- N-gon/strip handling: triangulate consistently; ensure stable face order in parity mode.

## Testing Strategy
- Unit:
  - Group key extraction for all modes; Parent16 split and swap correctness; tile decode.
  - Bounds checks & triangle filtering.
- Integration:
  - Each assembly strategy yields coherent object counts on curated fixtures.
  - Cross-tile audit reduces OOB with adjacency enabled; CSVs exist and are well-formed.
- Regression:
  - Legacy OBJ parity snapshot tests (line order, winding, transforms) on fixtures; stable SHA in parity mode where feasible.
- Performance:
  - Batch runs with time/memory ceilings; report summaries.

## Implementation Phases
1) Foundation
- New project skeleton, logging, CLI stub.
- SceneLoader (single tile), models, GroupingKeyService, ParentIdDecoder reuse.

2) Assemblers
- Implement `parent-index`, `msur-indexcount`, `parent16` strategies.
- Basic OBJ exporter with invalid index filtering + X inversion.

3) Cross-Tile
- Audit-only metrics; 3x3 adjacency hooks.
- Optional remapping prototype (flagged).

4) Diagnostics
- CSV snapshot, histograms, correlations, vertex audit.

5) Formats & Parity
- Legacy OBJ parity mode; snapshot tests.
- Optional glTF/GLB exporter (SharpGLTF).

6) Batch & Docs
- Directory batch flow; finalize CLI help; examples; memory-bank & docs updates.

7) Tests & CI
- Unit, integration, regression; lightweight perf checks.

## Risks & Mitigations
- Conflicting assembly theories (ParentIndex vs MSUR.IndexCount): support both, compare via diagnostics.
- Cross-tile remap correctness: start audit-only; add remap behind flags; thorough logging and CSV remap tables.
- Legacy parity brittleness: optional parity mode; snapshot tests; document acceptable deviations.

## Success Criteria
- No OOB indices in final meshes (except when `--no-remap`).
- Building-scale objects produced on canonical samples by at least one strategy.
- Diagnostics are actionable and reproducible.
- Legacy parity mode matches or is explainably close on fixtures.
- Optional glTF export works on sample scenes.

## Deliverables
- `parpToolbox/src/PM4NextExporter/` project.
- CLI: `pm4next-export` with documented flags.
- Documentation: this plan file + updated `docs/formats/PM4.md`.
- Test suite with fixtures.

## Proposed Module Map
- `PM4NextExporter/`
  - `Cli/Program.cs`
  - `Core/SceneLoader.cs`
  - `Core/CrossTileVertexResolver.cs`
  - `Core/GroupingKeyService.cs`
  - `Assembly/ParentHierarchyAssembler.cs`
  - `Assembly/MsurIndexCountAssembler.cs`
  - `Assembly/Parent16Assembler.cs`
  - `Export/ObjExporter.cs`
  - `Export/GltfExporter.cs` (Phase 2)
  - `Diagnostics/DiagnosticsService.cs`
  - `Model/*` (scene, entries, options)
  - `Tests/*`
