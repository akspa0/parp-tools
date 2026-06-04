# PM4 Region-Aware Object Grouping and Position Decoder

**Created**: 2026-05-21
**Status**: Implementation in progress
**Owner**: wow-viewer Core.PM4

---

## Problem Statement

PM4 collision/pathfinding data encodes a hierarchical scene graph. The current codebase groups surfaces by CK24 (a packed 24-bit key) and then subdivides by MSLK.GroupObjectId. However, CK24 alone does not capture the full scene structure:

1. **CK24 groups span multiple tiles** — 21.6% of CK24 values appear in 2+ PM4 files.
2. **CK24 groups within a tile can bleed** — WMO and M2 collision data sometimes mix within a single CK24 seed group.
3. **There is no root grouping layer** between "the entire map" and "individual CK24 objects."

Current caution from later real-data inspection: `MSLK.TypeFlags` now appears to carry per-surface family buckets (`0x03` = M2 top surfaces, `0x10` = interior WMO floors, `0x12` = exterior WMO solid surfaces). Keep that classification signal separate from `MSLK.GroupObjectId`; this document only describes the current sub-object partitioning owner, not a final closure of all MSLK field semantics.

This document is now partially stale. Later corpus evidence shows `MSHD.Field04` is still useful as a grouping/coloring bucket, but it is **not** the primary stitch key for most multi-tile WMO/M2 objects. Empirical evidence from 616 development PM4 files shows:

- Field04 has 227 distinct values across 502 PM4 files with MSHD.
- Field04 values cluster in spatially adjacent tiles (a contiguous 3×2 block shares one Field04).
- The same Field04 can appear in non-adjacent regions (same scene type, different locations).
- Field04=1 appears only on empty stub tiles (140/502). Active tiles never have Field04=1.
- `204/266` cross-tile CK24 values span multiple distinct Field04 values, so the same WMO/M2 candidate often bridges more than one Field04 bucket.

**Working correction**: treat Field04 as a reusable scene/group bucket, not as the root ownership layer for cross-tile object stitching.

## Hierarchy Model

```
Level 0: CK24 + connector evidence (Object stitch across tiles)
  └─ Auxiliary bucket: MSHD.Field04 (scene/group hint, not final stitch owner)
       └─ Level 2: MSLK.GroupObjectId (Sub-object) — linked surface sets
            └─ Level 3: Individual MSUR surfaces + MPRL positions
```

### Auxiliary bucket: Region-like Field04 (MSHD.Field04)

- Groups tiles into a reusable scene/group bucket.
- One Field04 value can span 2-13+ tiles in the development corpus.
- Field04 is per-tile (stored in MSHD header), not per-surface.
- Many cross-tile CK24 objects bridge multiple Field04 values, so Field04 cannot be the only stitch owner.

### Level 1: Object (CK24)

- Across the full corpus, CK24 is the strongest current object stitch candidate.
- CK24 type byte (high byte) classifies object type: 0x00=nav mesh, 0x40-0x41=M2, 0x42-0x43=WMO, 0xC0-0xC3=M2 exterior.
- CK24 ObjectId (low 16 bits) identifies the specific object within its type.
- Cross-tile CK24 objects are merged via MSCN connector keys (existing `BuildMergedGroupMap`).

### Level 2: Sub-object (MSLK.GroupObjectId)

- Within a CK24 object, MSLK.GroupObjectId partitions surfaces into linked sub-objects.
- Union-Find: surfaces sharing the same GroupObjectId become one sub-object.
- GroupObjectId == 0 means "unlinked" (no group assignment).
- The dominant GroupObjectId (most references) becomes the object's primary sub-object key.
- This grouping role is separate from `MSLK.TypeFlags`, which now has partial evidence as a surface-family classifier rather than a partition key.

### Level 3: Surfaces and Positions

- Each sub-object contains MSUR surfaces (collision polygons) and MPRL position references.
- MPRL entries provide world-space placement (position + heading).
- RefIndex on MSLK links surfaces to both MSUR (surface partitioning) and MPRL (position collection).

## Implementation Plan

### Phase 1: Region-Aware Object Grouper

**File**: `wow-viewer/src/core/WowViewer.Core.PM4/Research/Pm4RegionObjectGrouper.cs`

Reads multiple PM4 files and produces a region-first grouping:

```
Region (Field04)
  ├─ Object (CK24)
  │   ├─ SubObject (MSLK.GroupObjectId)
  │   │   ├─ SurfaceIndices[]
  │   │   ├─ MprlEntries[]
  │   │   ├─ MslkGroupObjectIds[]
  │   │   └─ PlacementSolution (coordinate mode, planar transform, yaw)
  │   └─ SubObject ...
  └─ Object ...
```

Key behaviors:
1. Read all PM4 files in a directory.
2. Group surfaces by CK24 across the corpus → object candidates.
3. Use MSHD.Field04 as a diagnostic bucket and compare whether a merged CK24 stays inside one Field04 or bridges several.
4. Within each object, partition by MSLK.GroupObjectId → sub-objects.
5. For each sub-object, collect MPRL position references.
6. Resolve placement (coordinate mode, planar transform, yaw correction) per sub-object.
7. Cross-tile merge: objects with the same CK24 are merged using connector evidence even when they bridge multiple Field04 buckets.

### Phase 2: Object Position Decoder

**File**: `wow-viewer/src/core/WowViewer.Core.PM4/Services/Pm4ObjectPositionDecoder.cs`

Takes the region-grouped output and resolves each object's world-space position:

1. For each sub-object, use the existing `Pm4PlacementMath.ResolveCoordinateMode` and `ResolvePlacementSolution`.
2. Compute the object's world-space AABB from MSUR vertices + planar transform.
3. Compute the object's centroid and heading from MPRL entries.
4. Output a structured `Pm4DecodedObjectPlacement` with:
   - RegionId
   - Ck24 key
   - Ck24Type / Ck24ObjectId
   - SubObjectId (dominant MSLK.GroupObjectId)
   - WorldPosition (Vector3)
   - WorldHeading (float, degrees)
   - WorldBounds (AABB)
   - SurfaceCount / IndexCount
   - TileCoordinates[] (which tiles this object spans)

### Phase 3: Integration Test

**File**: `wow-viewer/tests/WowViewer.Core.PM4.Tests/Pm4RegionObjectGrouperTests.cs`

Test against `test_data/original_development/World/Maps/development/`:
1. Read all 616 PM4 files.
2. Group by Field04 → verify bucket count matches expected (227 distinct values).
3. Verify the cross-tile analyzer reports how many CK24 objects stay inside one Field04 versus bridging several.
4. For reference WMO cases, verify surfaces can merge across tiles even when Field04 changes.
5. Verify MPRL position collection produces non-empty results for each sub-object.

## Validation Criteria

- Field04 bucket count matches the current MSHD corpus result (227 for development corpus).
- The cross-tile analyzer reports the current bridge rate for CK24 spanning multiple Field04 buckets.
- Reference WMO/M2 cases can merge across tiles without requiring one shared Field04 value.
- MPRL positions are non-empty for objects with type 0x40-0x43 (WMO/M2).
- Placement resolution produces valid coordinate mode and planar transform for each object.
- Cross-tile objects that bridge multiple Field04 values still share the same merged CK24 key.

## Open Questions

1. Does Field04 being the same across non-adjacent tiles mean "same scene type" or "same designer area"?
2. How much of the remaining stitch should come from CK24 alone versus MSCN/MSLK connector evidence when Field04 changes?
3. How does the nav mesh (CK24=0x000000) interact with region grouping? It spans the entire map.
4. What is the relationship between Field00/Field08 and Field04?

## Files To Create

| File | Purpose |
|------|---------|
| `wow-viewer/src/core/WowViewer.Core.PM4/Research/Pm4RegionObjectGrouper.cs` | Region-aware object grouping |
| `wow-viewer/src/core/WowViewer.Core.PM4/Models/Pm4RegionObjectModels.cs` | Data models for grouped output |
| `wow-viewer/src/core/WowViewer.Core.PM4/Services/Pm4ObjectPositionDecoder.cs` | Position resolution for grouped objects |
| `wow-viewer/tests/WowViewer.Core.PM4.Tests/Pm4RegionObjectGrouperTests.cs` | Integration tests |

## Existing Code Dependencies

| Component | Location | Role |
|-----------|----------|------|
| `Pm4ResearchReader` | `Core.PM4/Services/` | Reads PM4 files into document model |
| `Pm4PlacementMath` | `Core.PM4/Services/` | Coordinate mode resolution, planar transform, cross-tile merge |
| `Pm4MshdGroupingService` | `Core.PM4/Services/` | Extracts Field04 as RegionId |
| `Pm4CoordinateService` | `Core.PM4/Services/` | PM4-to-ADT coordinate transforms |
| `Pm4ResearchHierarchyAnalyzer` | `Core.PM4/Research/` | Per-tile CK24 grouping with split families |
| `Pm4ResearchCrossTileAnalyzer` | `Core.PM4/Research/` | Cross-tile CK24 tracking |
| `Pm4ResearchDocument` | `Core.PM4/Research/` | Document model for loaded PM4 data |
