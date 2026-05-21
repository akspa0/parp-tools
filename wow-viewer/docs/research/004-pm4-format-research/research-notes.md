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

### Test Data (CONFIRMED)

| Location | Description |
|----------|-------------|
| `wow-viewer/test_data/original_development/World/Maps/development/` | **616 PM4 files** + ADTs (confirmed by user) |
| `wow-viewer/test_data/original_development/World/Maps/development/development_00_00.pm4` | Primary reference tile |
| `wow-viewer/tests/WowViewer.Core.PM4.Tests/Pm4ResearchIntegrationTests.cs` | 1025-line integration test |

**Note**: MdxViewer can read this loose map folder against the 3.3.5 client with PM4 overlays enabled. Any findings can be visually verified in the viewer.

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

## CK24 Bit Layout — Deep Analysis

### PackedParams Extraction (MSUR offset 0x1C)

```csharp
// Pm4ResearchChunkModels.cs:64-68 (identical in 5 codebases)
public uint Ck24         => (PackedParams >> 8) & 0x00FF_FFFF;  // bits[8:31]
public byte Ck24Type     => (byte)((PackedParams >> 24) & 0xFF); // top byte = bits[31:24]
public ushort Ck24ObjectId => (ushort)(Ck24 & 0xFFFF);           // low 16 bits = bits[8:23]
```

**PackedParams byte layout:**
```
Byte 3 [31:24] = Ck24Type (0x00, 0x40, 0x42, 0x43, 0x80, etc.)
Byte 2 [23:16] = CK24 bits[23:16] (high byte of CK24ObjectId)
Byte 1 [15:8]  = CK24 bits[15:8]  (low byte of CK24ObjectId)
Byte 0 [7:0]   = DISCARDED by >> 8 shift — never extracted
```

**Critical note:** Ck24Type and CK24 overlap. `(PackedParams >> 24) & 0xFF` == `(CK24 >> 16) & 0xFF`. The type byte is the high byte of the 24-bit CK24.

### Observed CK24Type Values

| Type Byte | Interpretation | Source |
|-----------|---------------|--------|
| 0x00 | No object / terrain mesh (CK24=0) | WorldScene.cs:4151 |
| 0x40 | Renderable type with dedicated viewer filter | WorldScene.cs:10263 |
| 0x42 | WMO collision | WoWRollback Pm4Reader |
| 0x43 | WMO collision (most common in ref tile) | WoWRollback Pm4Reader |
| 0x80 | Renderable type with dedicated viewer filter | WorldScene.cs:10263 |

**No code anywhere assigns semantic names to type byte values.** It is treated as an opaque discriminator.

### Hierarchical Structure Hypothesis

The user's hypothesis: LSBs are base groups (0x000000=group 0, 0x000001=group 1).

**Evidence AGAINST clean two-level hierarchy:**
1. **Ck24ObjectId reuse across types**: Same low16 appears under different type bytes. Linkage analyzer tracks `reusedAcrossTypeGroupCount > 0` (Pm4ResearchIntegrationTests.cs:96-100).
2. **The hierarchy analyzer tests 8 split families** (Pm4ResearchHierarchyAnalyzer.cs:16-26), explicitly acknowledging CK24 alone is insufficient for object decomposition.
3. **Only 2 reuse cases out of 1,601 non-zero object-id groups** (Pm4Research README:108-109), both on tile 36_24.

**Evidence that COULD support partial hierarchy:**
1. CK24 values cluster by type — within a type, ObjectId values likely represent distinct objects.
2. The viewer's seed grouping is `GroupBy(CK24)` — a flat key — but secondary splits by MSLK, MDOS, and connectivity suggest the real structure is multi-level.

### Cross-Tile CK24

- **21.6% of CK24 values span multiple tiles** — same 24-bit key in adjacent .pm4 files.
- Cross-tile merge uses **MSCN connector keys** (quantized positions), NOT CK24 matching alone.
- `BuildMergedGroupMap` (Pm4PlacementMath.cs:720-800+) uses Union-Find with spatial overlap + shared connector keys.
- Merge threshold: 2+ shared connector keys with >= 35-50% overlap ratio.

---

## MSLK Linking — Deep Analysis

### MSLK Entry Structure (20 bytes)

| Offset | Field | Type | Status | Notes |
|--------|-------|------|--------|-------|
| 0x00 | TypeFlags | byte | **Unknown** | "1=walkable, 2=walls" per spec; not validated |
| 0x01 | Subtype | byte | **Unknown** | "Floor level?" — looks layer-like |
| 0x02-0x03 | Padding | ushort | Verified | Always 0 in v48 |
| 0x04 | GroupObjectId | uint | **Partial** | Local alias, not global ID. Low16 maps to CK24ObjectId with reuse. |
| 0x08 | MspiFirstIndex | int24 | Verified | Signed 24-bit index into MSPI |
| 0x0B | MspiIndexCount | byte | **Partial** | Ambiguity: "indices mode" vs "triangles mode" (count*3) |
| 0x0C | LinkId | uint | Verified | Tile coordinate sentinel 0xFFFF_XXYY (100% decoded) |
| 0x10 | RefIndex | ushort | **Partial** | Primary: MSUR. 4553/1.27M don't fit. Multi-domain. |
| 0x12 | SystemFlag | ushort | **Partial** | 0x8000 dominates — constant flag |

### RefIndex Target Domains

When RefIndex doesn't map to MSUR, the mismatch classifier (Pm4ResearchMslkRefIndexClassifier.cs) tests 8 candidate domains:

| Domain | Fit Count | Confidence |
|--------|-----------|------------|
| MSUR | ~1,268,782 (96.4%) | Primary target |
| MSPI | Fits by count range | Secondary |
| MSVI | Fits by count range | Secondary |
| MSCN | Fits by count range | Secondary |
| MSLK | Self-referencing | Secondary |
| MPRL | Weak fit count | Secondary |
| MSPV | Fits by count range | Secondary |
| MPRR | Weak fit count | Secondary |

**The viewer uses RefIndex as BOTH MSUR index AND MPRL index** — dual-use depending on context:
- `SplitSurfaceGroupByMslk`: RefIndex → MSUR (surface partitioning)
- `CollectLinkedPositionRefs`: RefIndex → MPRL (position lookup)

### MSLK Families (TypeFlags/Subtype Combinations)

Key: `$"type=0x{TypeFlags:X2} subtype={Subtype} system=0x{SystemFlag:X4}"`

Dominant family: `type=0x01 subtype=2 system=0x8000` — ~185k entries, all sentinel tile links.

### MSLK → Surface → Position Link Chain

```
MSLK entry
  ├─ RefIndex → MSUR (which surface this link references)
  ├─ GroupObjectId → other MSLK entries sharing this group
  │    └─ those entries' RefIndex → MPRL (position/rotation for the group)
  └─ MspiFirstIndex + MspiIndexCount → MSPI → MSPV (path geometry)
```

The viewer's `CollectLinkedPositionRefs` (WorldScene.cs:4659-4707):
1. Find MSLK entries referencing surfaces in the current object
2. Collect their GroupObjectIds
3. Find ALL MSLK entries sharing those GroupObjectIds
4. Use those entries' RefIndex as MPRL indices

---

## MSCN — Deep Analysis

### What MSCN Is

Flat pool of Vector3 positions (12 bytes each). Described as "collision wall vertices" or "exterior vertex list." No per-vertex attributes — just positions.

### MSCN Consumption vs. Raw Data

| Use Case | Where | What's Consumed | Status |
|----------|-------|----------------|--------|
| **Connector Keys** | WorldScene.cs, Pm4PlacementMath.cs | MSCN positions via `MSUR.MdosIndex` → quantized to 2-unit grid | **ACTIVE** — cross-tile object merging |
| **Cross-tile Remap** | parpToolbox MscnRemapper | ALL MSCN positions, (Y,X,Z) swap | **LEGACY** — parpToolbox only |
| **Object Discovery** | WoWRollback MscnObjectBuilder | MSCN positions grouped by CK24 via MdosIndex | **LEGACY** — WoWRollback only |
| **Coordinate Analysis** | Pm4ResearchMscnAnalyzer | ALL positions per file | **RESEARCH** — statistical |
| **Spatial Proximity** | MscnMeshComparisonAnalyzer | Sampled points | **DIAGNOSTIC** |

**Key finding**: The active viewer only consumes MSCN positions that are **directly referenced by MSUR.MdosIndex**. The remaining MSCN vertices (most of them) are read but NOT used in the active viewer pipeline.

### How MSCN Enables Cross-Tile Object Merging

1. Each MSUR surface has `MdosIndex` → index into MSCN pool
2. MSCN positions are quantized to 2-unit grid → `Pm4ConnectorKey(X,Y,Z)`
3. Per CK24 object group, connector keys are collected
4. Adjacent tiles sharing 2+ connector keys (with overlap >= 35-50%) are merged via Union-Find

**This is the mechanism that identifies objects spanning multiple tiles.**

### MSCN Coordinate Space

Pm4ResearchMscnAnalyzer tests 3 hypotheses per point:
1. Raw world-space (X,Y fit tile bounds)
2. Swapped XY world-space (Y,X fit tile bounds)
3. Tile-local (0..533.33)

Result: "No dominant mode" for 615 of 616 development tiles. Swapped-XY overlap is stronger than raw for some CK24 groups, suggesting axis-swapped companion geometry.

### What's Missing in MSCN Consumption

- **Most MSCN vertices are unused** — only MdosIndex-referenced ones are consumed
- **MSCN as collision hull geometry** — MscnObjectBuilder treats MSCN as "collision wall vertices that better represent object shapes" but this code is in legacy WoWRollback, not the active viewer
- **MSCN-to-WMO coordinate transform** — brute-force MscnWmoComparisonCommand tests 14+ transforms; no conclusion
- **MSCN as object containment boundary** — the user's hypothesis that MSCN identifies where objects exist across tiles is supported by the connector key system, but the full containment/boundary role is unexplored

---

## The MSCN Revelation — Surface Centroids and Pathfinding Network

### What MSCN Actually Is

The user's direct knowledge, corrected from earlier analysis:

1. **MSCN points are the center (centroid) of every MSVT-derived mesh surface.** Not random collision vertices, not a separate geometry layer — the centroid of each surface polygon.

2. **"Peg" or "dowel" connecting points** exist at tile boundaries. Some MSCN points are placed in the space where an adjacent PM4 tile would exist. These are cross-tile connection markers — physical dowels that peg objects across tile seams.

3. **This is old-school AI pathfinding from ~2010.** Not neural networks, not modern AI. It's a graph-based navigation system where MSCN centroids are nodes, MSLK entries are edges, and MSPV/MSPI are the path geometry connecting them.

4. **PM4 is a compressed dataset.** The user describes it as "a giant dataset compressed into 1/1000th the size required to store all that data." The format encodes a scene graph (collision, pathfinding, object placement) in a highly compact chunked structure.

### Revised Understanding of the Data Model

```
MSVT + MSVI + MSUR  →  Collision mesh surfaces (polygons)
        ↓
   MSUR centroid  →  MSCN node (navigation graph node)
        ↓
   MSLK entries  →  Graph edges linking nodes
        ├─ RefIndex → MSUR (which surface this edge connects)
        ├─ GroupObjectId → edge group / sub-graph
        ├─ MspiFirstIndex + MspiIndexCount → MSPI → MSPV (path vertices along this edge)
        └─ LinkId → tile coordinate (cross-tile boundary marker)
        ↓
   MPRL  →  Position/heading for placement in the world
        ↓
   MPRR  →  Graph structure chaining MPRL entries
```

**The "pathfinding overlay" label was actually correct** — but "pathfinding" in the old-school AI sense (graph traversal, A*, visibility checks) not in the modern navmesh sense. The MSLK field of view / linking data the user recalls likely relates to the `TypeFlags` and `Subtype` fields which we couldn't decode.

### Why This Explains the CK24 Problem

CK24 groups surfaces into objects. Within each object:
- MSCN centroids form the navigation nodes
- MSLK edges connect those nodes
- MSPV/MSPI provide the path geometry between nodes
- MPRL provides the world-space placement anchor

The "object decomposition" problem is really "how to identify独立的 pathfinding sub-graphs within a CK24 group." The MSLK.GroupObjectId and TypeFlags/Subtype likely encode sub-graph membership — but we named them wrong and lost the semantic thread.

### Why This Explains the Coordinate Chaos

Different chunks use different coordinate systems because they serve different purposes:
- **MSVT**: Collision mesh in tile-local YXZ (the actual geometry)
- **MSCN**: Surface centroids — same space as MSVT but only the center points
- **MPRL**: World-space placement (ADT coordinate frame)
- **MSPV**: Path vertices in raw XYZ (navigation path geometry)
- **MSLK**: Graph edges (indices, not coordinates)

The "infinitely nested data storage" the user describes is the chunk hierarchy: each chunk is a data layer in a compressed scene graph, with its own coordinate convention optimized for its role.

### What We Got Wrong

1. **MSCN is not "collision wall vertices"** — it's surface centroids. The MscnObjectBuilder treating them as collision hull geometry was wrong.
2. **The coordinate transform chaos** exists because we were trying to make MSCN points align with WMO vertices — but they're centroids, not vertices. The comparison was apples-to-oranges.
3. **The 1/4096 scale hypothesis** from Pm4SceneGraphTest.cs was probably trying to compensate for a coordinate space mismatch that doesn't exist if you treat MSCN as centroids.
4. **MSLK is not a scene-graph linkage** — it's a pathfinding graph edge catalog. TypeFlags/classifies edge types, not object parts.

## Practical Observations — From the Developer

### Object Splitting Reality

1. **PM4s are parsed both as single map objects and as individual tiles.** You have to parse per-tile to split objects, but the full map view is needed for cross-tile objects.

2. **Object splitting is imperfect.** Adjacent M2 data sometimes gets included in WMO objects even when they shouldn't be — the boundary detection between WMO and M2 collision data within a CK24 group is not clean.

3. **MPRL contains WMO doodad references.** This is described as "perplexing" — MPRL appears to contain both:
   - Intersection points where objects clip through the terrain mesh (the primary interpretation)
   - WMO doodad placement data (which shouldn't logically be in MPRL)

4. **MPRL is the odd chunk.** It contains terrain-object intersection points AND WMO doodad data, which are conceptually different things. This suggests MPRL may serve dual purposes or our understanding of its role is incomplete.

### Implications for the Data Model

- MPRL may not be purely "placement anchors" — it may also store doodad/prop placement within the pathfinding context
- The WMO doodad presence in MPRL could mean that doodads are treated as pathfinding obstacles (which would make sense for server-side AI)
- The M2/WMO boundary confusion in object splitting suggests that CK24 type bytes (0x40 vs 0x42/0x43) don't perfectly separate object types — there's bleed

---

## Project Context

### The 2021 Hobbyist Effort

A group of hobbyists attempted to reconstruct the WoW development map from server-side data. They manually fixed object placements and got close on some objects but not 100% correct. The development map remains incomplete.

### The User's PM4 Work

The user (a systems person, not a programmer) decoded enough from PM4 files to surpass what the 2021 group achieved by hand. Key achievements:
- PM4 overlay visualization in MdxViewer with per-object coloring and 3D placement
- Cross-tile object merging via MSCN connector keys
- Terrain reconstruction that impressed the 2021 community
- Coordinate system resolution (axis convention detection, planar transform scoring, yaw correction)

### The Downstream Goal

Match PM4 collision/pathfinding objects to real WMO/M2 assets, then place them correctly on the development map. The PM4 format IS the answer — it encodes what objects exist and where they belong. The challenge is reading it correctly.

### The MdxViewer Advantage

MdxViewer is the most complete PM4 implementation. It has:
- Full coordinate transform pipeline (axis convention → planar transform → yaw correction → renderer space)
- Per-object reconstruction with rotation handling
- Cross-tile merge via MSCN connector keys
- WMO/M2 bleed handling via MdosIndex splitting
- JSON/OBJ export of reconstructed objects

wow-viewer may not be fully implemented yet. MdxViewer should be the primary source for deriving PM4 truths.

---

## MdxViewer — The Authoritative PM4 Surface

The user confirms MdxViewer is the most complete PM4 implementation. wow-viewer may not be fully implemented yet. MdxViewer truths:

### Confirmed PM4 Truths (from MdxViewer)

1. **MSVT uses (Y, X, Z) as the default XY-plane convention.** The axis convention detector at `WorldScene.cs:5744-5804` maps `localU=pm4Vertex.Y, localV=pm4Vertex.X, localUp=pm4Vertex.Z`. This is the proven transform.

2. **MPRL.Unk04 is a uint16 packed angle.** Formula: `angle = rawValue * 2pi / 65536`. The circular mean of linked MPRL headings gives the expected object rotation.

3. **Tile-local coordinates use standard WoW tile convention:** `worldX = tileY * tileSpan + u`, `worldY = tileX * tileSpan + v`. File tile X advances along world Y.

4. **MPRL position transform is fixed:** `(X, Z, Y)` — MPRL X is world X, MPRL Z is world Y, MPRL Y is world Z. Independent of MSVT axis convention.

5. **Planar transform is resolved by footprint scoring**, not centroid matching. Bidirectional nearest-neighbor footprint score (85% weight) + centroid distance (15% weight) selects from up to 4 candidate transforms.

6. **MSCN entries are "exterior vertices"** used for cross-tile connector key generation. Each distinct MdosIndex on a surface selects one MSCN point, quantized to 2-unit resolution.

7. **WMO/M2 bleed** is handled by MdosIndex grouping within a CK24 seed group. User toggle: `splitCk24ByMdos`.

8. **Yaw correction threshold is 12 degrees.** Smaller corrections are noise.

9. **Renderer transform:** `rendererX = MapOrigin - worldY`, `rendererY = MapOrigin - worldX`, `rendererZ = worldZ + 0.5`. MapOrigin = 17066.66666.

10. **Pm4OverlayObject carries localized geometry** — lines/triangles are stored relative to a BaseTransform (rotation + translation), allowing efficient per-object rendering.

### What This Means for the Centroid Hypothesis

The user says "there's no centroid data that seems accurate, as WMO's have internal rotation that does not seem to be accounted for within the PM4." This means:

- **MSCN points are NOT simple centroids of MSUR surfaces.** WMO objects have internal rotation that shifts vertices relative to their pivot, so a naive centroid calculation would not match MSCN positions.
- **MSCN may store centroids in the object's local space** (before rotation), or may store something else entirely.
- **The "peg/dowel" points** at tile boundaries are real — they're MSCN points that extend into adjacent tile space for cross-tile pathfinding connectivity.
- **The Pm4ResearchMscnAnalyzer's raw-vs-swapped-XY tests** were testing the wrong hypothesis (coordinate frame) when the real issue is rotation.

## MSLK Field-by-Field Linkage — Complete Map

### TypeFlags (0x00) — Edge Type Classification

- **Used for**: Grouping/clustering only. Never as a filter, gate, or index.
- **Family key**: `"type=0x{TypeFlags:X2} subtype={Subtype} system=0x{SystemFlag:X4}"`
- **Dominant value**: 0x01 (in the top family with ~185k entries)
- **Other values observed**: 0x02, 0x04, 0x08, 0x10 per the wiki ("seen: &1; &2; &4; &8; &16")
- **Code paths**: Unknowns analyzer family grouping, linkage analyzer mismatch families, RefIndex classifier
- **Hypothesis**: May encode walkability classification — the bitmask pattern (&1, &2, &4, &8, &16) suggests bitfield flags for edge properties (walkable, wall, ledge, water, etc.)

### Subtype (0x01) — Edge Property / Sequence Position

- **Used for**: Grouping/clustering only. Never as an index or filter.
- **Wiki says**: "0…11-ish; position in some sequence? index into something?"
- **Dominant value**: 2 (in the top family)
- **Code paths**: Same as TypeFlags — family key component only
- **Hypothesis**: May encode height level, floor index, or sequence position within a navigation path

### GroupObjectId (0x04) — Sub-Object Membership (PRIMARY GROUPING KEY)

- **Used for**: Union-Find surface partitioning. The core mechanism for splitting CK24 groups into sub-objects.
- **Algorithm**:
  1. Scan MSLK entries for the current CK24 surface set
  2. Skip entries where `GroupObjectId == 0`
  3. For each non-zero GroupObjectId, collect surfaces referenced by entries with that GroupObjectId (via RefIndex → MSUR)
  4. Union-Find: surfaces sharing the same GroupObjectId become one sub-object
  5. Unlinked surfaces (no GroupObjectId reference) fall into a separate bucket
- **Zero means unlinked**: GroupObjectId == 0 = "no group assignment"
- **Dominant selection**: The GroupObjectId that references the most surfaces in a group becomes the object's `DominantLinkGroupObjectId`
- **MPRL collection**: After partitioning, ALL MSLK entries sharing the group's GroupObjectIds are scanned for RefIndex → MPRL position references
- **Partial overlap with CK24ObjectId**: `GroupObjectId & 0xFFFF` sometimes matches `CK24ObjectId`, but with reuse — not globally unique
- **Code paths**: `WorldScene.TryPartitionSurfaceGroupByMslk`, `Pm4ResearchHierarchyAnalyzer.SplitByMslkRefIndex`, `Pm4Ck24ForensicsAnalyzer.BuildLinkGroupSurfaceSets`, `Pm4MatchSupport.SplitByMslkGroupObjectId`

### MspiFirstIndex + MspiIndexCount (0x08-0x0B) — Path Geometry Window

- **Used for**: Defining a window into MSPI (path indices) → MSPV (path vertices)
- **Format**: Signed 24-bit index + unsigned 8-bit count
- **When count == 0**: FirstIndex is -1, meaning no path data
- **Interpretation**: Linear index mode (not triangle mode). The window `MSPI[MspiFirstIndex..+MspiIndexCount]` contains path-vertex indices.
- **Corpus stats**: 598,882 active links. 399,183 fit indices-mode only. 199,699 fit both modes. 0 fit neither.
- **Path geometry**: Separate from the surface mesh (MSVT/MSVI). This is navigation path data — the lines/curves that pathfinding AI follows between surfaces.

### LinkId (0x0C) — Tile Coordinate Tag

- **Used for**: Identifying which tile an MSLK entry belongs to
- **Format**: `0xFFFF_XXYY` where XX = tileX, YY = tileY
- **Development corpus**: 100% sentinel tile links (all 1,273,335 entries)
- **Code paths**: Family key first component (`tile={tileKey}|flags=...|subtype=...`), LinkId pattern summary
- **No filtering**: Never used to filter or gate any behavior. Purely diagnostic.

### RefIndex (0x10) — THE DUAL-USE FIELD

- **Path A (MSUR index)**: When `RefIndex < MSUR.Count`, the entry links to a surface. Used for union-find partitioning.
- **Path B (MPRL index)**: When `RefIndex < MPRL.Count`, the entry links to a position reference. Used for position/heading collection.
- **Both fire simultaneously**: The same RefIndex value indexes into BOTH MSUR and MPRL. This means MSUR and MPRL share a compatible index space (or at least overlapping ranges).
- **Mismatch domain**: 4,553 / 1,273,335 entries don't fit MSUR. Mismatches cluster in MSPI/MSVI/MSCN/MSLK domains.
- **The key insight**: RefIndex is not "sometimes MSUR, sometimes MPRL" — it's ALWAYS both. The code checks MSUR first for surface partitioning, then checks MPRL for position collection. Entries where RefIndex >= MSUR.Count are mismatches but may still validly index into MPRL.

### SystemFlag (0x12) — Constant Flag

- **Value**: Always 0x8000 in version_48
- **Used for**: Family key component only. No filtering.
- **Likely purpose**: Format version marker or constant flag

### The Full Linkage Chain

```
CK24 group (MSUR surfaces)
  ↓ GroupBy(CK24)
MSLK entries (scan for RefIndex → surfaces in this group)
  ↓ GroupObjectId → Union-Find partitioning
Sub-objects (linked surface sets)
  ↓ RefIndex → MPRL (position + heading collection)
  ↓ MspiFirstIndex + MspiIndexCount → MSPI → MSPV (path geometry)
  ↓ LinkId → tile coordinate tag
  ↓ TypeFlags/Subtype → edge classification
Placement resolution (coordinate mode, planar transform, yaw correction)
```

---

## WoWEdit Screenshot Analysis — The Ground Truth

### The Image

WoWEdit Data 1.9.0 — the official WoW map editor. Shows a 3D viewport with:

1. **Gray terrain mesh** — ground surfaces with wireframe edges
2. **Red edges along structures** — wall tops, roof edges, building boundaries (collision/pathfinding boundaries)
3. **Blue/cyan vertical markers** — small vertical lines at regular intervals (navigation graph nodes)
4. **Building structures** — stepped/zigzag shapes with red edges (WMO collision geometry)

### What This Maps To in PM4

| Visual Element | PM4 Chunk | Purpose |
|---------------|-----------|---------|
| Gray terrain polygons | MSUR + MSVT (CK24=0) | Walkable floor surfaces |
| Red wall/barrier edges | MSUR + MSVT (CK24!=0, TypeFlags=&2?) | Non-walkable collision boundaries |
| Blue/cyan vertical markers | MSCN (via MSUR._0x18) | Navigation graph nodes / connection points |
| Building collision geometry | MSUR + MSVT (CK24=0x42/0x43) | WMO wall/floor/roof collision mesh |
| Red edges along building tops | MSLK (TypeFlags=walls) | Wall-edge classification in the pathfinding graph |

### TypeFlags Bitmask Hypothesis (Strengthened)

The WoWEdit visually distinguishes edge types. This maps to MSLK TypeFlags as a bitmask:

| Bit | Likely Meaning | WoWEdit Visual |
|-----|---------------|----------------|
| &1 | Walkable floor edge | Gray floor surface edges |
| &2 | Wall/barrier edge | Red wall edges |
| &4 | Ledge/drop-off edge | Red edges at structure tops |
| &8 | Water/liquid edge | Not visible in this screenshot |
| &16 | Special/unknown | Not visible in this screenshot |

### MSCN as Navigation Nodes

The blue/cyan vertical markers in WoWEdit are placed at:
- Regular intervals along flat terrain
- At structural boundaries (wall corners, edges)
- At transition points between surfaces

This is consistent with MSCN being **navigation graph nodes** — the vertices in the pathfinding network. Each marker is a point where the pathfinding AI can make decisions (turn, climb, jump, etc.).

The "peg/dowel" points at tile boundaries are the same type of markers — they extend into adjacent tile space to connect the navigation graph across tiles.

## WoWEdit Interior Screenshot — The Second Piece

### The Image

WoWEdit Data 1.9.0 showing a WMO interior (dungeon/building). From "The WoW Diary" by John Staats — Kickstarter first pressing.

Visible elements:
- Textured stone floor (walkable surface)
- Walls and archway (barrier collision geometry)
- Staircase going up (multi-level navigation)
- Candelabras (M2 doodads placed on floor)
- Banner hanging on wall (M2 doodad)
- Cobwebs (decorative M2 doodads)

### What This Confirms

1. **MPRL stores doodad placements** — The candelabras and banner are M2 objects whose positions are in MPRL. They're collision obstacles for server-side AI. A pathfinding character needs to walk around them.

2. **MSLK Subtype may be floor/level index** — The staircase shows multiple levels. Wiki says Subtype is "0…11-ish; position in some sequence?" This could be a floor level number (0 = ground, 1 = first floor, 2 = second floor, etc.).

3. **CK24 type separation** — Building walls/floor = CK24 type 0x42/0x43 (WMO). Candelabras and banner = CK24 type 0x40 (M2). The "bleed" problem is when these get mixed.

4. **The complete scene** — PM4 encodes: walkable surfaces, barriers, navigation nodes, prop obstacles, and graph edges. Full interior/exterior navigation dataset.

### The Matching Problem

The user's goal: match PM4 collision objects to real WMO/M2 assets. The WoWEdit shows the visual ground truth — the textured models placed in the scene. PM4 encodes the collision/pathfinding version of the same scene. The challenge is:
- PM4 CK24 groups define collision objects
- Each collision object should map to a real WMO or M2 model
- MPRL positions should match MODF (WMO placement) or MDDF (M2 placement) entries in the ADT
- The development map is incomplete — PM4 is the definitive reference for what SHOULD be there

---

### What WoWEdit Confirms

1. **PM4 is pathfinding data** — the editor shows exactly this: walkable surfaces, wall boundaries, and navigation nodes
2. **MSLK TypeFlags classify edge types** — the red vs gray distinction is the wall vs floor classification
3. **MSCN nodes are connection points** — the blue markers are the graph vertices
4. **The data is a compressed representation** of what would otherwise be a much larger navigation dataset

---

## TypeFlags Bitmask Verification — Revised Hypothesis

### My Initial Hypothesis Was Wrong

I proposed: `&1=walkable, &2=wall, &4=ledge, &8=water, &16=special`

The observed TypeFlags values from WoWRollback (ANCIENT code, pre-dates MdxViewer — labels are unverified guesses):

| Value | WoWRollback Label (UNVERIFIED) | Binary | CK24 Type Byte Match? |
|-------|-------------------------------|--------|----------------------|
| 0x00 | "Nav Mesh" (unverified) | 0b00000000 | CK24 type 0x00 |
| 0x01 | (dominant family, ~185k entries) | 0b00000001 | No match |
| 0x40 | "M2 Interior" (unverified) | 0b01000000 | CK24 type 0x40 |
| 0x41 | "M2 Interior" (unverified) | 0b01000001 | CK24 type 0x41 |
| 0x42 | "WMO" (unverified) | 0b01000010 | CK24 type 0x42 |
| 0x43 | "WMO" (unverified) | 0b01000011 | CK24 type 0x43 |
| 0xC0 | "M2 Exterior" (unverified) | 0b11000000 | CK24 type 0xC0 |
| 0xC1 | "M2 Exterior" (unverified) | 0b11000001 | CK24 type 0xC1 |
| 0xC2 | "M2 Exterior" (unverified) | 0b11000010 | CK24 type 0xC2 |
| 0xC3 | "M2 Exterior" (unverified) | 0b11000011 | CK24 type 0xC3 |

### The Critical Observation (UNVERIFIED)

**The TypeFlags values 0x40-0x43 and 0xC0-0xC3 numerically match CK24 type byte values.** However, the WoWRollback labels ("M2 Interior", "WMO", "M2 Exterior") are **assumptions from the earliest phase of the project** — WoWRollback predates MdxViewer by months and was built before the coordinate system work, CK24 grouping analysis, or any of the deep PM4 research. Those labels should be treated as **initial guesses, not verified facts**.

What IS verified:
- The numeric values exist in the TypeFlags distribution
- The values numerically overlap with CK24 type byte values
- The dominant family has TypeFlags=0x01 (which doesn't match any CK24 type)

What is NOT verified:
- Whether the WoWRollback labels ("M2 Interior", "WMO", etc.) are correct — they're from ancient code
- Whether TypeFlags actually stores the CK24 type byte
- Whether the high-nibble/low-nibble two-layer hypothesis is correct

**We need to load the data and verify empirically.**

### The Bitmask Pattern

Analyzing the binary:
- **Bit 7 (0x80)**: Distinguishes "Interior" (0x40-0x43) from "Exterior" (0xC0-0xC3)
- **Bit 6 (0x40)**: Present in all WMO and M2 types (not in nav mesh)
- **Bits 0-1**: Vary within each group (0x40/0x41, 0x42/0x43, 0xC0-0xC3)

This is NOT a walkable/wall/ledge bitmask. It's an **object type classification** — the same CK24 type byte, stored in MSLK for fast reference.

### What About 0x01?

The dominant family has `type=0x01`. This doesn't match any CK24 type byte. Possible explanations:
1. 0x01 = "terrain pathfinding edge" (not an object type, but an edge type)
2. 0x01 = a different classification layer (edge type vs object type)
3. 0x01 = the actual bitmask meaning &1 = walkable floor edge (my original hypothesis was right for this value)

### The Two-Layer Hypothesis (SPECULATIVE)

If TypeFlags does store the CK24 type byte plus edge flags, the two-layer model would be:
- **High nibble (bits 4-7)**: Object type (from CK24 type byte)
- **Low nibble (bits 0-3)**: Edge type within that object — 0x01=walkable, 0x02=wall, etc.

This would explain:
- 0x00 = nav mesh, no edge flags (high=0x00, low=0x00)
- 0x01 = nav mesh, walkable (high=0x00, low=0x01)
- 0x40 = unknown object, no edge flags (high=0x40, low=0x00)
- 0x41 = unknown object, walkable (high=0x40, low=0x01)
- 0x42 = unknown object, wall (high=0x40, low=0x02)
- 0x43 = unknown object, walkable+wall (high=0x40, low=0x03)

**But this is entirely speculative.** The labels "M2 Interior", "WMO", "M2 Exterior" from WoWRollback are not verified. We need to check the actual data.

### What Needs Verification

1. Run the unknowns analyzer to get exact per-value TypeFlags distribution
2. Cross-tabulate TypeFlags x CK24Type for MSLK entries where RefIndex < MSUR.Count
3. Check if TypeFlags low nibble correlates with path triangle normals (flat vs wall)
4. Check if the dominant 0x01 family has different MSPI patterns than 0x42/0x43 families

---

## The MSHD Suspicion and Rare Value Hypothesis

### MSHD: We Never Dug In

The user says MSHD is suspicious because we never really investigated whether it helps decode other chunks. Current state:
- 8 x uint32 fields, 32 bytes total
- Fields 0x0C-0x1C are ALL zero in 616 development tiles
- Field00 and Field08 vary but don't correlate with chunk counts
- Field04 is always 1

We dismissed MSHD as "probably dead padding" without seriously testing whether it encodes layout information, region boundaries, or root-level grouping keys.

### The Rare Value Hypothesis

The user notes that some unknown field values only appear once or twice in the corpus. These rare values are likely NOT noise — they're likely **root group IDs or top-level identifiers** that would help with object isolation/subdivision from CK24.

In a hierarchy:
- Common values = leaf-level attributes (walkable, wall, floor, etc.)
- Rare values = root-level identifiers (object group, region, scene node)

If MSLK.Subtype (0-11-ish) has some values that appear only once or twice, those might be **top-level scene divisions** — the missing layer between CK24 (which groups too broadly) and individual polygons (which is too granular).

### The Object Decomposition Problem Revisited

Current splitting chain:
```
CK24 (flat key) → MSLK.GroupObjectId (union-find) → sub-objects
```

What we might be missing:
```
CK24 (flat key) → ??? root grouping ??? → MSLK.GroupObjectId → sub-objects
```

The missing root grouping could be in:
1. **MSHD** — header fields encoding top-level scene structure
2. **MSLK.Subtype** — rare values as root group identifiers
3. **MSLK.TypeFlags** — rare values as scene division markers
4. **MSUR.GroupKey** — unknown byte, could be a region/group identifier
5. **The discarded low byte of MSUR.PackedParams** — never extracted, could carry a key layer

### What Analysis Would Reveal This

1. **MSHD field value distribution** — min/max/mode/frequency across 616 tiles. If Field00 has a bimodal distribution (common values + rare outliers), the rare values might be root identifiers.

2. **MSLK.Subtype rare value analysis** — which Subtype values appear only once or twice per tile? What CK24 groups do they belong to? Do they span multiple tiles?

3. **MSLK.TypeFlags rare value analysis** — same question. Are there TypeFlags values that appear < 10 times in the entire corpus?

4. **MSUR.GroupKey distribution** — what values exist? Are there rare values that could be root group markers?

5. **Cross-chunk rare value correlation** — do rare values in different chunks co-occur? If Subtype=7 appears once in a tile and GroupKey=0x12 appears once in the same tile, they might be related.

---

### What We Should Do Next

1. **Verify the centroid hypothesis**: For MSUR surfaces in the reference tile, compute centroids and compare against MSCN points accessed via MSUR._0x18. If they match, the hypothesis is confirmed.
2. **Map the peg/dowel points**: Identify MSCN points that fall outside the tile bounds (adjacent tile space) and correlate them with cross-tile CK24 objects.
3. **Reinterpret MSLK as graph edges**: Treat TypeFlags as edge type (walkable, wall, ledge, etc.) and Subtype as edge property (height level, direction, etc.).
4. **Update all analyzers**: The Pm4ResearchMscnAnalyzer should test "centroid of MSUR surface" as a hypothesis, not just "raw vs swapped coordinate fit."
5. **Fix the naming**: Rename `MdosIndex` → `MscnIndex` (or `_0x18` with comment), add wowdev.wiki mapping comments to all invented names.

---

## Naming Drift from wowdev.wiki — Critical Findings

### The Most Dangerous Error: `MSUR._0x18` → `MdosIndex`

**Our code calls this field `MdosIndex`.** The wowdev.wiki documentation calls it `_0x18`.

**The field is NOT an index into MDOS.** It is an index into **MSCN** (scene nodes / exterior vertices).

Evidence:
- `Pm4PlacementMath.cs:685`: reads `surface.MdosIndex` and uses it to index into `exteriorVertices` (the MSCN list)
- `WorldScene.cs:5196-5204`: uses `surface.MdosIndex` to access `pm4.KnownChunks.Mscn`
- `Pm4ResearchMscnAnalyzer` documents: "MSUR.MdosIndex is the main bridge into MSCN scene-node data"

The name `MdosIndex` collides with the MDOS chunk abbreviation (Destructible Object States). Anyone reading the code will assume this field points to MDOS. It does not.

### Other Naming Drift

| Field | wowdev.wiki Name | Our Name | Status |
|-------|-----------------|----------|--------|
| MSUR._0x00 | `_0x00` ("bitmask32 flags") | `GroupKey` | Invented — add comment |
| MSUR._0x02 | `_0x02` | `AttributeMask` | Invented — add comment |
| MSUR._0x18 | `_0x18` | `MdosIndex` | **WRONG** — must rename to `MscnIndex` |
| MSLK._0x00 | `_0x00` ("flags?") | `TypeFlags` | Invented — close to wiki intent |
| MSLK._0x01 | `_0x01` ("index into something?") | `Subtype` | Invented — wiki says "index" not "subtype" |
| MSLK._0x04 | `_0x04` ("An index somewhere") | `GroupObjectId` | Invented — add comment |
| MSLK._0x0c | `_0x0c` ("Always 0xffffffff") | `LinkId` | Invented — we decoded it as tile coords |
| MSLK._0x10 | `msur_index` | `RefIndex` | Renamed — because 3.6% don't fit MSUR |
| MSLK._0x12 | `_0x12` ("Always 0x8000") | `SystemFlag` | Invented — add comment |
| MSUR floats | `float _0x04; _0x08; _0x0c` | `Vector3 Normal` | Invented — bundled as Normal |

### The Hallucination Path

1. Initial code used correct wowdev.wiki names (`_0x18`, `_0x10`, etc.)
2. Empirical analysis revealed patterns (CK24 grouping, cross-tile objects, RefIndex mismatches)
3. Analysts gave invented names to fields based on observed behavior
4. Invented names were used in analyzers, tests, documentation until they felt canonical
5. New code referenced invented names without checking against wowdev.wiki

### Recommended Fix

1. **Immediate**: Rename `MdosIndex` → `MscnIndex` in `Pm4MsurEntry` and all downstream code
2. **Immediate**: Add `// wowdev.wiki: _0xNN` comments to all invented names
3. **Short-term**: Audit whether MSVT YXZ formula from wowdev.wiki matches our axis detection

Full analysis: `wow-viewer/docs/research/004-pm4-format-research/naming-drift-analysis.md`

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
