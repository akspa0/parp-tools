# PM4 Format Research — Verified Ground Truths

**Created**: 2026-05-20
**Last Updated**: 2026-05-21
**Source**: MdxViewer (active viewer) and wow-viewer (library) only. WoWRollback, PM4Tool, and parpToolbox references have been removed — those codebases contained unverified guesses and hallucinations.

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
| 6 | MSPI | Path Indices | 4 (uint32) | `Pm4ResearchReader.cs:80-89` |
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
| Research `Pm4ResearchReader` | `wow-viewer/src/core/WowViewer.Core.PM4/Pm4ResearchReader.cs` | 16 (all known) |

---

## MSHD — Header Fields

### Structure (Verified)

32 bytes total, 8 fields of 4 bytes each:

```
Offset 0x00: Field00 (uint32) — non-zero, varies. Top=534 in 289/502 files.
Offset 0x04: Field04 (uint32) — SCENE CONTAINER / REGION ID. 227 distinct values. =1 only on empty tiles (140 stubs).
Offset 0x08: Field08 (uint32) — non-zero, varies. Top=534 in 292/502 files. Only 233/502 equal Field00.
Offset 0x0C: Field0C (uint32) — always 0 (confirmed 502/502)
Offset 0x10: Field10 (uint32) — always 0 (confirmed 502/502)
Offset 0x14: Field14 (uint32) — always 0 (confirmed 502/502)
Offset 0x18: Field18 (uint32) — always 0 (confirmed 502/502)
Offset 0x1C: Field1C (uint32) — always 0 (confirmed 502/502)
```

### Field04 — CONFIRMED Scene Container

**User confirmation (2026-05-21)**: "the field04 in the MSHD appears to be a container for a bunch of objects in a set scene"

- Field04 has 227 distinct values across 502 non-empty tiles.
- Field04 values cluster in spatially adjacent tiles (a contiguous 3×2 block shares one Field04).
- Field04=1 appears only on empty stub tiles (140/502). Active tiles never have Field04=1.
- Same Field04 can appear in non-adjacent regions (same scene type, different locations).
- Field04 does NOT encode per-tile metrics — within F04=3262 tiles, surface count ranges from 11 to 6,575.
- **Field04 may be tied to data types** — different Field04 values may correspond to different categories of scene data.

### Field00 and Field08

- F00=534: 132/309 active tiles (42.7%) — dominant
- F08=534: 137/309 active tiles (44.3%)
- F00==F08: 86/309 (27.8%) — they differ more often than match
- Likely a version/source tuple or encoding two facets of the same thing

---

## Reference Graph (Verified from MdxViewer)

```
MSVT <--- MSVI <--- MSUR
  |                   |
  |                   +--- PackedParams -> CK24 (surface grouping)
  |                   +--- MsviFirstIndex -> MSVI
  |                   +--- _0x18 (MscnIndex) -> MSCN
  |
MSPV <--- MSPI <--- MSLK.MspiFirstIndex
                     MSLK.RefIndex ------> MSUR (primary, ~96.8%)
                                          MPRL (secondary)
                     MSLK.GroupObjectId -> (sub-object partitioning within CK24)
                     MSLK.LinkId --------> (tile coordinate sentinel)
                     MSLK.TypeFlags ------> (edge classification, NOT CK24 type)
                     MSLK.Subtype --------> (edge property / sequence position)

MSCN <--- MSUR._0x18 (MscnIndex)
       <--- (cross-tile connector keys via quantized positions)

MPRL <--- MPRR.Value1 (when Value1 < MPRL.Count)
MSVT <--- MPRR.Value1 (when Value1 >= MPRL.Count?)
```

---

## CK24 — Surface Grouping Key (NOT Object Identifier)

### PackedParams Extraction (MSUR offset 0x1C)

```csharp
// Pm4ResearchChunkModels.cs:64-68
public uint Ck24         => (PackedParams >> 8) & 0x00FF_FFFF;  // bits[8:31]
public byte Ck24Type     => (byte)((PackedParams >> 24) & 0xFF); // top byte = bits[31:24]
public ushort Ck24ObjectId => (ushort)(Ck24 & 0xFFFF);           // low 16 bits = bits[8:23]
```

### Critical Ground Truth (2026-05-21)

**CK24 is NOT the object matching unit.** A single WMO produces multiple CK24 entries with different type bytes and multiple sub-objects within each CK24 group.

**User confirmation**: "now the whole object is actually multiple pm4 sub-objects, not a single object"

Example: WESTFALL_HUMAN_FARMB.WMO on tile (23,32):
- CK24=0x43855A (type=0x43, WMO) — multiple sub-objects with different MSLK.GroupObjectIds
- Each sub-object has different surfaces, different MPRL positions, different MscnRefIndices

### Observed CK24Type Values (Verified from MdxViewer + Corpus)

| Type Byte | Interpretation | Source |
|-----------|---------------|--------|
| 0x00 | No object / terrain mesh (CK24=0) | WorldScene.cs |
| 0x40 | M2 Interior | MdxViewer filter |
| 0x41 | M2 Interior | MdxViewer filter |
| 0x42 | WMO collision | MdxViewer filter |
| 0x43 | WMO collision | MdxViewer filter |
| 0x80 | M2 Exterior | MdxViewer filter |
| 0xBF | Unknown (exists in corpus) | Corpus analysis |
| 0xC0 | M2 Exterior | Corpus analysis |
| 0xC3 | M2 Exterior variant | Corpus analysis |

**No code anywhere assigns semantic names to type byte values.** The labels above are from MdxViewer's filter logic, not from format documentation.

---

## MSLK — Link Entries (Verified from MdxViewer)

### Entry Structure (20 bytes)

| Offset | Field | Type | Status | Notes |
|--------|-------|------|--------|-------|
| 0x00 | TypeFlags | byte | **Edge classification** | NOT CK24 type byte. Bitmask pattern observed. |
| 0x01 | Subtype | byte | **Edge property** | 0-11 range. May be floor/level index. |
| 0x02-0x03 | Padding | ushort | Verified | Always 0 in v48 |
| 0x04 | GroupObjectId | uint | **Sub-object partitioning** | Links surfaces into sub-objects within a CK24 group |
| 0x08 | MspiFirstIndex | int24 | Verified | Signed 24-bit index into MSPI |
| 0x0B | MspiIndexCount | byte | Verified | Path vertex count |
| 0x0C | LinkId | uint | Verified | Tile coordinate sentinel 0xFFFF_XXYY |
| 0x10 | RefIndex | ushort | **Dual-use** | MSUR index (primary) + MPRL index (secondary) |
| 0x12 | SystemFlag | ushort | Constant | Always 0x8000 in version_48 |

### MSLK → Surface → Position Link Chain (Verified from MdxViewer)

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

### TypeFlags — NOT CK24 Type Byte (Disproven)

**The hypothesis that TypeFlags stores the CK24 type byte is INCORRECT.** From CK24=0x40AA0A (type=0x40, M2) forensics, the actual MSLK entries have TypeFlags values 0x11, 0x12, 0x14, 0x1A, 0x1C — high nibble is 0x1, NOT 0x4.

TypeFlags has only 10 distinct values across 1.27M entries. The dominant values are 0x01 and 0x02 with subtypes 0, 1, 2, 3.

---

## MSCN — Scene Nodes (Verified from MdxViewer)

### What MSCN Is

Flat pool of Vector3 positions (12 bytes each). Used for:
1. **Connector Keys** — cross-tile object merging via quantized positions (ACTIVE in MdxViewer)
2. **Surface centroids** — positions referenced by MSUR._0x18 (MscnIndex)

### MSCN Coordinate Space

The viewer only consumes MSCN positions that are directly referenced by MSUR._0x18 (MscnIndex). The remaining MSCN vertices are read but NOT used in the active viewer pipeline.

### Cross-Tile Object Merging via MSCN (Verified)

1. Each MSUR surface has `_0x18` (MscnIndex) → index into MSCN pool
2. MSCN positions are quantized to 2-unit grid → `Pm4ConnectorKey(X,Y,Z)`
3. Per CK24 object group, connector keys are collected
4. Adjacent tiles sharing 2+ connector keys (with overlap >= 35-50%) are merged via Union-Find

---

## MPRL — Position References (Verified from MdxViewer)

### Structure

| Offset | Field | Type | Notes |
|--------|-------|------|-------|
| 0x00 | Unk00 | ushort | Unknown |
| 0x02 | Unk02 | short | Unknown |
| 0x04 | Unk04 | ushort | **Packed angle**: `angle = rawValue * 2pi / 65536` |
| 0x06 | Unk06 | ushort | Unknown |
| 0x08 | Position | Vector3 | World-space position |
| 0x14 | Unk14 | short | Floor level (-1..15) |
| 0x16 | Unk16 | ushort | Unknown |

### MPRL Position Transform (Verified from MdxViewer)

Fixed transform: `(X, Z, Y)` — MPRL X is world X, MPRL Z is world Y, MPRL Y is world Z.

### MPRL Heading (Verified from MdxViewer)

Formula: `angle = rawValue * 2pi / 65536`. The circular mean of linked MPRL headings gives the expected object rotation.

---

## The Hierarchical Data Model (Verified)

### What We Know For Certain

1. **MSHD.Field04 groups tiles into scene regions** — confirmed by user and corpus analysis
2. **CK24 groups surfaces into surface groups** — NOT single objects
3. **MSLK.GroupObjectId partitions surfaces within a CK24 group into sub-objects** — verified by MdxViewer's `TryPartitionSurfaceGroupByMslk`
4. **One WMO produces multiple PM4 sub-objects** — confirmed by user observation
5. **Doodads (M2) have separate PM4 collision** — confirmed by user observation

### The Hierarchy

```
Level 0: MSHD.Field04 (Region) — scene container, possibly type-keyed
  └─ Level 1: CK24 (Surface Group) — groups surfaces by packed key, NOT a single asset
       └─ Level 2: MSLK.GroupObjectId (Sub-object) — linked surface sets
            └─ Level 3: Individual MSUR surfaces + MPRL positions
```

### What This Means for Object Matching

The matching pipeline must:
1. Collect ALL sub-objects belonging to a CK24 group
2. Combine their MPRL positions into a single placement candidate
3. Match the combined placement against MODF/MDDF entries
4. Account for one WMO spanning multiple CK24 groups (type 0x42 + 0x43)

---

## What We Got Wrong (Lessons Learned)

1. **CK24 = one WMO/M2** — WRONG. CK24 is a surface grouping key, not an asset identifier.
2. **TypeFlags stores CK24 type byte** — WRONG. TypeFlags is edge classification, not object type.
3. **MSCN = collision wall vertices** — WRONG. MSCN positions are surface centroids and connector keys.
4. **MPRL is purely placement anchors** — WRONG. MPRL also contains doodad/prop placement data.
5. **WoWRollback labels were facts** — WRONG. Those labels were unverified guesses from ancient code.

---

## Corpus Statistics (Development Build)

| Metric | Value |
|--------|-------|
| Total tiles | 616 |
| Non-empty tiles | 309 (502 with PM4 data) |
| PM4 version | 12304 |
| MSVT vertices (ref tile) | 6,318 |
| MSCN points (ref tile) | 9,990 |
| MPRL refs (ref tile) | 2,493 |
| MSUR surfaces (ref tile) | 4,110 |
| Distinct CK24 values | 1,229 |
| Cross-tile CK24 objects | 21.6% |
| CK24=0 (nav floor) surfaces | 35.9% |
| CK24!=0 (object collision) surfaces | 64.1% |
| MSLK RefIndex mismatches | 4,553 / 1,273,335 |

---

## Open Questions

1. Does Field04 encode data-type semantics (terrain vs building vs doodad)?
2. How do multiple CK24 groups with different type bytes relate to a single WMO?
3. Should the matching pipeline merge CK24 groups before matching, or match each group independently?
4. What is the relationship between Field00/Field08 and Field04?
5. Do retail PM4 files populate the trailing zero fields (0x0C-0x1C)?
6. What does MPRR.Value2 encode? (566 distinct values)
7. PM4 data loading performance — draw calls should cull far-off data.
