# PM4 Support Plan

Created: 2026-03-19

## Goal

Build PM4 support into the active toolchain and viewer using a complete cross-tile PM4 decode, CK24-based object grouping, and one validated PM4-to-world coordinate transform.

The core requirement is not just "parse PM4".

The required end state is:

- decode the full PM4 dataset
- preserve CK24-linked object identity across multiple ADT tiles
- classify PM4 output into useful object layers
- map PM4-derived geometry and anchors into the same world space as the rest of the viewer

## Current Facts

- The fixed source dataset is `test_data/development/World/Maps/development`.
- There are 616 PM4 files in that dataset.
- Initial Phase 3 validation result on Mar 19, 2026:
  - active-core `MPRL` refs are not tile-local; they are already in ADT placement order
  - `wowmapconverter pm4-validate-coords --tile-limit 100` validated 100 tiles with placements
  - sample metrics: 38,133 `MPRL` refs, 100.0% in expected tile bounds, 94.6% within a 32-unit nearest-placement threshold, 10.86 average nearest-placement distance
  - this validates the first `MPRL` placement path only; CK24 aggregation and MSCN semantics remain separate work
- Existing rollback work already established important PM4 facts:
  - `MSUR.PackedParams` exposes the effective CK24 key
  - `MSUR.MdosIndex` links surfaces to `MSCN` scene nodes
  - CK24 groups can represent the same object across multiple PM4 tiles
  - previous PM4 work already exported layers of the same object via CK24
- The current hard blocker is coordinate correctness, not the existence of a grouping key.

## Mar 20 Checkpoint + Reboot Resume

- PM4 viewer overlay load path now uses filename-based tile mapping aligned to terrain row/col conventions:
  - `map_x_y.pm4` maps to viewer tile `(tileX=x, tileY=y)`
- The previous MPRL-based tile reassignment heuristic was removed from viewer PM4 tile assignment.
- Duplicate PM4 files resolving to one tile now merge instead of overwrite, including:
  - overlay object payloads
  - PM4 tile stats
  - PM4 position-ref markers

Post-restart runtime validation order:

1. Load the reported mismatch area and verify `00_00` aligns to ADT `(0,0)`.
2. Verify tile directly below (`01_00`) no longer appears shifted into `01_01`.
3. Verify sparse/missing PM4 tiles remain blank instead of causing neighbor drift.
4. Verify merged duplicate-tile behavior does not regress selection/object-part uniqueness.

Validation note:

- Build pass exists; runtime signoff is still pending and required before claiming this fixed globally.

## Mar 20 Runtime Update (Post-Reboot)

- Current viewer behavior has improved from fragmented part orientation to coherent per-object assembly:
  - CK24 object parts now stay together on one shared coordinate plane
  - residual orientation issue is now a consistent global yaw offset
- Latest runtime observation:
  - PM4 reconstructed objects appear approximately 90 degrees counter-clockwise from expected placement orientation
  - this is materially better than mixed per-part spins because it indicates the remaining problem is likely one consistent basis correction

Documentation checkpoint:

- The active PM4 decode/assembly/transform logic is now captured in:
  - `documentation/pm4-current-decoding-logic-2026-03-20.md`
- That document records:
  - active chunk usage (`MSUR`, `MSLK`, `MPRL`, `MSVT`, `MSVI`)
  - tile mapping and merge guardrails
  - CK24 assembly pipeline and identity keys
  - transform solver inputs and known residual 90-degree offset

## Source Of Truth

The richer rollback PM4 decoder should be treated as the current behavioral reference, not the thinner core PM4 parser alone.

Reference files:

- `WoWRollback/WoWRollback.PM4Module/Decoding/Pm4Decoder.cs`
- `WoWRollback/WoWRollback.PM4Module/Decoding/Pm4ChunkTypes.cs`
- `WoWRollback/WoWRollback.PM4Module/Decoding/Pm4MapReader.cs`
- `WoWRollback/WoWRollback.PM4Module/PipelineCoordinateService.cs`
- `WoWRollback/WoWRollback.PM4Module/Decoding/MscnObjectDiscovery.cs`

The current `WoWMapConverter.Core/Formats/PM4/Pm4File.cs` parser is useful, but it is not yet the complete contract needed for final PM4 support because it does not expose the full CK24-oriented object model used by the rollback pipeline.

## Target Architecture

### 1. Canonical PM4 Decode Layer

One reusable PM4 decode layer should expose:

- file version and header
- `MSLK` link entries
- `MSPV` and `MSPI` path data
- `MSVT` and `MSVI` surface geometry
- `MSUR` surface records with decoded CK24 accessors
- `MSCN` scene nodes
- `MPRL` position references
- `MPRR` reference graph edges
- source tile identity for every decoded file

### 2. Cross-Tile CK24 Registry

PM4 support should build a map-level registry keyed by CK24.

Each CK24 group should be able to aggregate:

- surfaces from multiple PM4 tiles
- linked MSCN scene nodes
- optional placement references
- bounds and centroid data
- matched asset candidates

This registry must preserve tile provenance so debugging can still answer which PM4 files contributed to a given CK24 object.

### 3. Layered Output Model

PM4 support should not collapse everything into one mesh export.

Minimum output layers:

- terrain or nav/background surfaces
- CK24 object groups
- matched WMO candidates
- matched M2 candidates
- unmatched geometry candidates
- placement/reference markers

### 4. World Coordinate Contract

One authoritative transform API must define how PM4-derived coordinates become world coordinates used by the viewer and ADT placement systems.

This contract must explicitly state, per data source:

- whether the source values are tile-local or already world-space
- whether an axis swap is required
- whether a tile-origin offset is required
- whether the result is viewer world space, ADT placement space, or another intermediate space

Do not allow multiple ad hoc PM4 transforms to drift across exporters, matchers, and viewer code.

## Implementation Phases

### Phase 1: Decoder Consolidation

- Compare `WoWMapConverter.Core/Formats/PM4/Pm4File.cs` with the rollback PM4 decoder.
- Port or align the richer rollback data contract into reusable core types.
- Ensure CK24, `MdosIndex`, `MPRL`, and tile identity are all preserved.

Exit condition:

- one canonical PM4 decode layer can read the full PM4 set without dropping CK24 metadata

### Phase 2: Cross-Tile CK24 Aggregation

- Build a `Pm4MapReader`-style registry over the full PM4 dataset.
- Aggregate geometry and linked scene nodes by CK24 across multiple tiles.
- Keep `CK24 == 0` as the background/nav layer and `CK24 != 0` as object candidates.
- Add optional `MSVI` gap splitting for repeated instances inside one CK24 group when needed.

Exit condition:

- one query can retrieve all decoded geometry and scene-node references for a CK24 key across the full dataset

### Phase 3: Coordinate Validation

- Treat coordinate validation as the hard gate before viewer integration.
- Compare transformed PM4 anchors against known ADT placements in the fixed development dataset.
- Reconcile any differences among `MSCN`, `MPRL`, and `MSVT` coordinate semantics.
- Keep one explicit transform API as the only allowed conversion path.

Current status:

- initial `MPRL` validation slice is now in active core code via `Pm4CoordinateValidator`
- real-data sampling shows `MPRL` is already in ADT placement order, not tile-local
- `MSCN` semantics are still unvalidated in active core

Exit condition:

- PM4-derived placements land in the same world locations as their corresponding ADT or viewer references

### Phase 4: Viewer Debug Integration

- Add PM4 debug rendering first, not final polished rendering.
- Render layer toggles for:
  - background/nav surfaces
  - CK24 object groups
  - placement anchors
  - matched vs unmatched objects
- Use simple wireframe or flat color passes initially.
- Confirm visual overlay correctness before higher-level conversion work.

Exit condition:

- PM4 layers visually align with the rest of the map in the viewer

### Phase 5: Asset Matching And Extraction

- Reuse existing geometry matching for WMO and M2 candidates.
- Preserve unmatched geometry exports as first-class outputs.
- Do not make asset matching a prerequisite for basic PM4 world support.

Exit condition:

- CK24 groups can be visualized and optionally matched without losing unmapped geometry

## Validation Requirements

- Use real PM4 files from the fixed development dataset.
- Validate coordinates against real ADT placements or known world-space references.
- Do not claim PM4 support is correct based only on OBJ exports or decoder builds.
- If only the decoder was improved, say so.
- If cross-tile grouping is still partial, say so.
- If coordinates are still unverified, say so.

## Recommended Next Slice

The first implementation slice should be:

1. consolidate the richer rollback PM4 decode contract into reusable core models
2. build a cross-tile CK24 registry over the fixed development dataset
3. validate one coordinate path against known ADT placements before viewer rendering

That sequence attacks the actual blocker instead of jumping straight into rendering partial tile-local meshes.