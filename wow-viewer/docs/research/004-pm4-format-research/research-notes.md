# PM4 Format Research — Raw Findings

**Created**: 2026-05-20
**Source**: Codebase exploration across wow-viewer, gillijimproject_refactor, PM4Tool, WoWRollback

---

## Chunk Inventory

### Recognized Chunks (16 total)

| # | FourCC | Description | Stride | Reader Location |
|---|--------|-------------|--------|-----------------|
| 1 | MVER | Version | 4 (uint32) | `Pm4ResearchReader.cs:61-65` |
| 2 | MSHD | Header | 32 (8x uint32) | `Pm4ResearchReader.cs:68-69` |
| 3 | MSVT | Mesh Vertices | 12 (Vector3) | `Pm4ResearchReader.cs:84-85` |
| 4 | MSVI | Mesh Vertex Indices | 4 (uint32) | `Pm4ResearchReader.cs:88-89` |
| 5 | MSPV | Path Vertices | 12 (Vector3) | `Pm4ResearchReader.cs:76-77` |
| 6 | MSPI | Path Indices | 4 (uint32) | `Pm4ResearchReader.cs:80-81` |
| 7 | MSUR | Mesh Surfaces | 32 | `Pm4ResearchReader.cs:92-93` |
| 8 | MSLK | Mesh Links | 20 | `Pm4ResearchReader.cs:72-73` |
| 9 | MSCN | Scene Nodes | 12 (Vector3) | `Pm4ResearchReader.cs:96-97` |
| 10 | MPRL | Position References | 24 | `Pm4ResearchReader.cs:100-101` |
| 11 | MPRR | Position Ref Graph | 4 (2x uint16) | `Pm4ResearchReader.cs:104-105` |
| 12 | MDBH | Destructible Building Header | 4 (uint32) | `Pm4ResearchReader.cs:108-109` |
| 13 | MDBI | Destructible Building Indices | 4 (uint32) | `Pm4ResearchReader.cs:112-113` |
| 14 | MDBF | Destructible Building Filename | variable | `Pm4ResearchReader.cs:116-117` |
| 15 | MDOS | Destructible Object States | 8 | `Pm4ResearchReader.cs:120-121` |
| 16 | MDSF | Destructible Surface-to-Object | 8 | `Pm4ResearchReader.cs:124-125` |

### Reader Implementations

| Implementation | Location | Chunks Decoded |
|----------------|----------|----------------|
| Legacy `Pm4Decoder` | `gillijimproject_refactor/src/WoWMapConverter/.../Pm4Decoder.cs` | 10 (core only) |
| Research `Pm4ResearchReader` | `wow-viewer/src/core/WowViewer.Core.PM4/Pm4ResearchReader.cs` | 16 (all known) |

---

## MSHD Deep Dive

### Structure

32 bytes total, 8 fields of 4 bytes each:

```
Offset 0x00: Field00 (uint32) — non-zero, varies per tile
Offset 0x04: Field04 (uint32) — always 1 in development corpus
Offset 0x08: Field08 (uint32) — non-zero, often equals Field00
Offset 0x0C: Field0C (uint32) — always 0
Offset 0x10: Field10 (uint32) — always 0
Offset 0x14: Field14 (uint32) — always 0
Offset 0x18: Field18 (uint32) — always 0
Offset 0x1C: Field1C (uint32) — always 0
```

### Analyzer Results

`Pm4ResearchMshdAnalyzer` correlates each field against:
- MSUR surface count
- MSVT vertex count
- MSPI index count
- MSLK link count
- MPRL position count
- MSCN scene node count
- CK24 object count

**Finding**: No exact-match correlation found. Field00 and Field08 show weak correlation with some counts but nothing statistically conclusive.

### Corpus Evidence

- 616 development tiles: Fields 0x0C-0x1C are ALL zero. Confirmed by integration tests at `Pm4ResearchIntegrationTests.cs:118-122`.
- Field04 is always 1.
- Field00 and Field08 vary together but do not map to any single chunk count.

### Open Questions

1. Could Field00 be a byte offset to the first chunk after MSHD?
2. Could Field08 be the total file size minus the MSHD chunk?
3. Are fields 0x0C-0x1C part of a different header layout (e.g., 2x uint32 header + 6x uint32 reserved)?
4. Do retail/non-development PM4 files populate fields 0x0C-0x1C?

---

## Reference Graph (Current Understanding)

```
MSVT <--- MSVI <--- MSUR
  |                   |
  |                   +--- PackedParams -> CK24 (object grouping)
  |                   +--- MsviFirstIndex -> MSVI
  |                   +--- MdosIndex -> MSCN
  |
MSPV <--- MSPI <--- MSLK.MspiFirstIndex
                    MSLK.RefIndex ------> MSUR (primary, ~96.8%)
                                         MSPI (secondary?)
                                         MSVI (secondary?)
                                         MSCN (secondary?)
                                         MPRL (secondary?)
                    MSLK.LinkId --------> (tile coordinate sentinel)
                    MSLK.GroupObjectId -> (local alias, not global ID)

MSCN <--- MSUR.MdosIndex
       <--- (CK24 groups reference MSCN indirectly)

MPRL <--- MPRR.Value1 (when Value1 < MPRL.Count)
MSVT <--- MPRR.Value1 (when Value1 >= MPRL.Count?)
MPRR <--- MPRR chaining (Value1 -> another MPRR entry?)

MDBH <--- MDOS.DestructibleBuildingIndex
MDOS <--- MDSF.MdosIndex
MSUR <--- MDSF.MsurIndex
MDBI <--- (references MDBH entries?)
MDBF <--- (filename associated with MDBH entries?)
```

---

## Key Code Locations

### Core PM4 Reading

| File | Path | Purpose |
|------|------|---------|
| Pm4ResearchReader.cs | `wow-viewer/src/core/WowViewer.Core.PM4/` | Modern span-based reader |
| Pm4ResearchDocument.cs | `wow-viewer/src/core/WowViewer.Core.PM4/` | Document model |
| Pm4ResearchChunkModels.cs | `wow-viewer/src/core/WowViewer.Core.PM4/` | All chunk record types |
| Pm4Decoder.cs | `gillijimproject_refactor/src/WoWMapConverter/.../Formats/PM4/` | Legacy reader |
| Pm4ChunkTypes.cs | `gillijimproject_refactor/src/WoWMapConverter/.../Formats/PM4/` | Legacy chunk records |
| Pm4File.cs | `gillijimproject_refactor/src/WoWMapConverter/.../Formats/PM4/` | Legacy wrapper + OBJ export |

### Analysis

| File | Path | Purpose |
|------|------|---------|
| Pm4ResearchAnalyzer.cs | `wow-viewer/src/core/WowViewer.Core.PM4/` | Basic analysis, CK24 groups |
| Pm4ResearchAuditAnalyzer.cs | `wow-viewer/src/core/WowViewer.Core.PM4/` | Decode audit, stride validation |
| Pm4ResearchUnknownsAnalyzer.cs | `wow-viewer/src/core/WowViewer.Core.PM4/` | Cross-corpus unknowns, field distributions |
| Pm4ResearchMshdAnalyzer.cs | `wow-viewer/src/core/WowViewer.Core.PM4/` | MSHD field correlation |
| Pm4ResearchMscnAnalyzer.cs | `wow-viewer/src/core/WowViewer.Core.PM4/` | MSCN coordinate space analysis |
| Pm4ResearchLinkageAnalyzer.cs | `wow-viewer/src/core/WowViewer.Core.PM4/` | MSLK RefIndex mismatch analysis |
| Pm4ResearchHierarchyAnalyzer.cs | `wow-viewer/src/core/WowViewer.Core.PM4/` | Object hypothesis generation |
| Pm4Ck24ForensicsAnalyzer.cs | `wow-viewer/src/core/WowViewer.Core.PM4/` | Per-CK24 deep forensics |

### Coordinate Math

| File | Path | Purpose |
|------|------|---------|
| Pm4CoordinateService.cs | `wow-viewer/src/core/WowViewer.Core.PM4/Services/` | PM4-to-ADT transforms |
| Pm4PlacementMath.cs | `wow-viewer/src/core/WowViewer.Core.PM4/Services/` | Axis convention, planar transform, yaw |
| Pm4PlacementContract.cs | `wow-viewer/src/core/WowViewer.Core.PM4/Services/` | Default transforms |
| Pm4CorrelationMath.cs | `wow-viewer/src/core/WowViewer.Core.PM4/Services/` | Footprint hull, AABB overlap |

### Viewer Integration

| File | Path | Purpose |
|------|------|---------|
| WorldScene.cs | `gillijimproject_refactor/src/MdxViewer/Terrain/` | PM4 overlay build (line 4179+) |
| ViewerApp_Pm4Utilities.cs | `gillijimproject_refactor/src/MdxViewer/` | PM4 workbench UI |
| Pm4OverlayCacheService.cs | `gillijimproject_refactor/src/MdxViewer/Terrain/` | Binary cache |

### Object Reconstruction

| File | Path | Purpose |
|------|------|---------|
| Pm4ObjectBuilder.cs | `gillijimproject_refactor/WoWRollback/WoWRollback.PM4Module/Decoding/` | WMO candidate reconstruction |
| MscnObjectBuilder.cs | `gillijimproject_refactor/WoWRollback/WoWRollback.PM4Module/Decoding/` | MSCN collision hull builder |
| PM4BuildingExtractor.cs | `PM4Tool/src/WoWToolbox.PM4Parsing/BuildingExtraction/` | Building extraction |

### Documentation

| File | Path | Lines |
|------|------|-------|
| pm4-current-decoding-logic-2026-03-20.md | `gillijimproject_refactor/documentation/` | 731 |
| pm4-raw-unknowns-map-2026-03-21.md | `gillijimproject_refactor/documentation/` | 421 |
| PM4-Format-Specification.md | `gillijimproject_refactor/WoWRollback/docs/-specifications-/` | 711 |
| pm4-specification.md | `gillijimproject_refactor/next/parpDocumentation/` | 202 |

### Test Data

| Location | Description |
|----------|-------------|
| `gillijimproject_refactor/test_data/development/World/Maps/development/` | 616 PM4 tiles, 309 non-empty |
| `gillijimproject_refactor/test_data/development/development_00_00.pm4` | Primary reference tile |
| `wow-viewer/tests/WowViewer.Core.PM4.Tests/Pm4ResearchIntegrationTests.cs` | 1025-line integration test |

---

## Corpus Statistics (Development Build)

From integration tests and analyzers:

| Metric | Value |
|--------|-------|
| Total tiles | 616 |
| Non-empty tiles | 309 |
| PM4 version | 12304 |
| Reference tile chunks | 54 |
| MSVT vertices (ref tile) | 6,318 |
| MSCN points (ref tile) | 9,990 |
| MPRL refs (ref tile) | 2,493 |
| MSUR surfaces (ref tile) | 4,110 |
| Unknown chunks across corpus | 0 |
| Top CK24 group surfaces | 896 |
| MSLK family (type=0x01, subtype=2, sys=0x8000) | ~185k entries |
| MSLK RefIndex mismatches (MSUR) | 4,553 / 1,273,335 total |
| CK24 cross-tile objects | 21.6% |
| CK24=0 (nav floor) surfaces | 35.9% |
| CK24!=0 (object collision) surfaces | 64.1% |

---

## Datastore Hypothesis — Evidence Summary

### For

1. CK24 type classification separates nav-mesh from WMO/M2 collision (64.1% is object data)
2. MSLK is a scene-graph structure, not a nav-graph (tile coordinates, multi-domain RefIndex)
3. MPRL positions validate against ADT object placements (world-building metadata)
4. MSCN stores collision hull geometry unique from MSVT (physics data)
5. MDBH/MDOS/MDSF encode destructible building state (gameplay data)
6. CK24 objects cross tile boundaries (global scene description)

### Against

1. MSPV/MSPI are genuinely pathfinding-specific data
2. The format was historically called "pathfinding" by the community
3. MSLK's MspiFirstIndex/MspiIndexCount directly reference path data
4. Some MSLK families appear to be navigation-only (type=0x01, subtype=2, system=0x8000)

### Neutral

1. MSHD is completely opaque — cannot confirm or deny either hypothesis from the header alone
2. MPRR graph semantics are unclear — could be nav-graph or scene-graph
3. MSUR.GroupKey and MSUR.AttributeMask are unknown — could support either interpretation
