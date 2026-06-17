# Spec 046: PM4 Asset Matching

**Status**: Active — **MAJOR BREAKTHROUGH**: Ck24ObjectId is a global object identifier spanning tiles. Fingerprint-based model identification works. `pm4 fingerprint-scan` command added. Coordinate mode detection fixed (611/616 PM4s use WorldSpace coords). Next: match fingerprint groups against WMO archive to build identity table.

**Created**: 2026-06-03 | **Last updated**: 2026-06-17

## What Exists

All C# code in `wow-viewer/src/core/WowViewer.Core.PM4/Matching/`:
- `Pm4ObjectSegmentBuilder` — deterministic segment builder (CK24+TypeFlags grouping, 4110→18 segments on dev tile)
- `Pm4SegmentSignalExtractor` — v2 signal contract (bounds, footprint hull, height stats, surface-family histogram)
- `Pm4AssetMatchScorer` — type-profile matching (typed overlap 35% + type profile 15% + shape 50%) — **produces sub-threshold scores on all real data**
- `Pm4ReplacementPlacementSynthesizer` — proposal-grade placement synthesis with provenance — **never produces proposals because scorer never matches**
- `Pm4SegmentExportService` — aggregate segment exports from PM4 files
- `Pm4AssetSignalCorpusSupport` — durable asset corpus with seeded placement path

Inspect CLI commands:
- `pm4 export-segments`, `pm4 match-assets`, `pm4 export-asset-signals`, `pm4 synthesize-placements`, `pm4 match-report`
- `pm4 correlate-models` — **NEW**: reads WMO collision geometry from archive, transforms to world space, computes volumetric overlap against PM4 CK24 group bounds

9 C# unit tests pass. Python scorer matches C# scores within 0.005 (but both produce wrong scores).

## Correlation Experiment Results (2026-06-17)

`pm4 correlate-models` runs on development map tiles:

### Tile 0_0 — HIT
- **CK24 0x421809 (type 0x42 WMO, 890 surfaces, 1927 verts) ↔ ND_IRONDWARF_LARGEBUILDING.WMO at 61.8% volumetric overlap**
- WMO world bounds: (16709,16699,-12)-(16916,16906,134)
- CK24 WoW bounds: (16717,16736,-12)-(16880,16899,134)
- This confirms the coordinate conversion works and PM4 CK24 groups CAN be matched to source WMO models
- Only 3 correlations total on this tile (the 2 smaller ones are the same WMO barely overlapping M2-type CK24 groups)
- 15 WMO placements, 10 M2 placements; most don't overlap the large WMO CK24 group because they're small structures

### Tile 22_18 — 0 correlations
- 81 CK24 groups, only 2 ADT placements (both OILPLATFORM_LOW.WMO)
- The oil platforms are at world coords (-3655,5765) and (8195,-12231); PM4 geometry is at (-4538,-4799) to (-4710,-4713)
- **PM4 contains objects not in the ADT MODF table** — the ADT only records 2 placements but PM4 has 81 CK24 groups
- This tile's PM4 data represents the "snowball fort" with Ulduar titan structures placed through a mechanism other than MODF

### Tile 14_36 — 0 correlations
- 98 CK24 groups, 3 ADT placements (2 oil platforms, 1 Stormwind Harbor)
- Placements are far from PM4 geometry (different coordinate regions)
- **CK24 types 0x3E, 0x3F, 0xC0, 0xC1, 0xC2 observed here** — not seen on tile 0_0

### Key findings
1. **CK24 type mapping confirmed**: 0x40/0x41 = M2, 0x42/0x43 = WMO, 0xC0/0xC1/0xC2 = unknown (interior/navmesh?), 0x3E/0x3F = unknown (building sub-types?)
2. **New CK24 types observed**: 0x3E, 0x3F on city-area tiles — possibly building sub-types
3. **PM4 data is MORE comprehensive than ADT MODF** — many objects in PM4 are placed through mechanisms other than ADT placement tables
4. **M2 collision vertex reading returns 0** — `MdxCollisionSummary` bounds are nullable and fall back to model bounds
5. **Coordinate conversion from raw-ADT to WoW world is correct**: `wowX = MapOrigin - (tileY*TileSize + pm4Y)`, `wowY = MapOrigin - (tileX*TileSize + pm4X)` for XYPlaneZUp convention on tile 0,0
6. **CRITICAL: Ck24ObjectId is a global object identifier** — the same ObjectId appears across multiple tiles for the same physical object. OID 52202 spans 8 tiles (a large WMO). Different OIDs with the same fingerprint = different instances of the same model.
7. **CRITICAL: 611/616 PM4s use WorldSpace (absolute) coordinates** — only 5 use TileLocal. The `Pm4PlacementMath.IsLikelyTileLocal()` function detects this correctly.
8. **Fingerprint (surfaces, indices, vertices) identifies model instances** — same model at different positions produces the same fingerprint. Different models produce different fingerprints. The most common fingerprint `(35, 144, 90)` appears 97 times across 0x41/0x42/0x43 types, representing a common WMO wall/segment template.
9. **CK24 type distribution across 616 PM4s**: 0x40 (M2-a)=80, 0x41 (M2-b)=161, 0x42 (WMO-a)=584, 0x43 (WMO-b)=466, 0xC0=77, 0xC1=100, 0xC2=66, 0xC3=38, 0x3D=2, 0x3E=5, 0x3F=10, 0xB6=1, 0xBD=2, 0xBE=2, 0xBF=10

## What's Broken

### B1: Shape scorer miscalibrated (P1)

`Pm4AssetMatchScorer` produces scores 0.08-0.16 on real data against placement-derived references, and 0.42-0.43 against the full 1985-WMO corpus. The 0.45 `MinimumMatchedScore` threshold is unreachable. Root causes (likely combination):
- PM4 segments are individual surfaces (1-3 triangles each), not whole-object groupings. A WMO with 122 collision triangles becomes ~57 tiny segments, each scoring poorly against the whole-WMO bounds.
- The scoring weights (typed overlap 35% + type profile 15% + shape 50%) were set without real-data calibration.
- Asset reference signals from `BuildFromPlacements` use placement-bounds (position ± fallback AABB), not WMO group collision geometry. The corpus from `BuildFromArchive` reads WMO summaries but may not extract per-group collision bounds correctly.

**Mitigation**: The `pm4 correlate-models` approach bypasses the scorer entirely by directly comparing WMO collision mesh bounds against PM4 CK24 group bounds in WoW world space. This has proven to work (0.618 overlap on tile 0_0). The scorer should be recalibrated based on these correlation results.

### B2: Coordinate mismatch in pm4 match-report (P1)

`Pm4MatchSupport.BuildPm4ObjectMatches` compares PM4 object world coordinates against ADT placement world coordinates. But:
- `ConvertPm4VertexToWorld` (TileLocal mode) produces raw-ADT coordinates: `(tileY * TileSize + mappedU, tileX * TileSize + mappedV, localUp)`
- `AdtPlacementReader.Read` produces WoW world coordinates: `(MapOrigin - rawY, MapOrigin - rawX, rawZ)` where MapOrigin = 17066.666

**Status**: Partially resolved — `pm4 correlate-models` applies the correct conversion and gets matching coordinates. The `match-report` command still has the mismatch.

### B3: Segment grouping too granular for WMO matching (P1)

On tile 22_18, 40543 PM4 objects become only 92 segments. Each segment represents a small surface patch, not an entire placed WMO.

**Mitigation**: The `pm4 correlate-models` command groups all surfaces by CK24 before computing bounds, effectively merging all surfaces sharing the same CK24 into one shape. This is the correct grouping level for WMO matching.

### B4: PM4 contains objects not in ADT placement tables (P1 — NEW)

On tiles 22_18 and 14_36, the ADT MODF/MDDF tables contain far fewer placements than PM4 CK24 groups. Many objects in PM4 are placed through mechanisms other than ADT placement tables. This means ADT placements cannot be used as the sole source of truth for identifying PM4 objects on arbitrary tiles.

**Implication**: The identity mapping must be built from tiles where ADT placements DO overlap PM4 geometry (like tile 0_0), then applied to tiles where they don't.

### B5: M2 collision vertex reading returns 0 (P2 — NEW)

`MdxCollisionSummary.BoundsMin/Max` are `Vector3?` and the collision data is not being extracted properly. M2 placements show 0 collision vertices and 0 collision faces, falling back to model bounds that don't represent the actual collision mesh.

## User Scenarios & Testing

### User Story 1 — Shape scorer produces real matches on known tiles (Priority: P1)

As a PM4 researcher, I want the shape scorer to correctly match PM4 segments to their corresponding WMO/M2 models on tiles where I have known placements, so that I can trust the matching pipeline for tiles without placements.

**Why this priority**: Without this, the entire pipeline is decorative — it builds reports and proposals but none are correct.

**Independent Test**: On tile 24_35 (Duskwood, 13 WMO placements with known model paths), at least 1 WMO placement must score ≥0.45 against its correct PM4 CK24 group after grouping fix and coordinate alignment.

**Acceptance Scenarios**:

1. **Given** tile 24_35 with known WMO placements, **When** `pm4 match-assets` runs with the placement-derived corpus, **Then** at least 1 placement scores ≥0.45 against its corresponding PM4 CK24 group.
2. **Given** the same tile and the full WMO corpus, **When** scoring grouped PM4 segments, **Then** the correct WMO model appears in the top-3 candidates for at least 1 CK24 group.

### User Story 2 — Match report uses correct coordinates (Priority: P1)

As a PM4 researcher, I want `pm4 match-report` to compare PM4 objects and ADT placements in the same coordinate system, so that spatial proximity matching actually works.

**Why this priority**: Without coordinate alignment, the position-based matcher in `match-report` is completely non-functional.

**Independent Test**: On any tile with placements, `pm4 match-report` must show non-zero candidate counts for placements that are spatially near PM4 objects.

**Acceptance Scenarios**:

1. **Given** tile 24_35, **When** `pm4 match-report` runs, **Then** at least 5 placements show "Candidate Count" > 0.
2. **Given** tile 22_18, **When** `pm4 match-report` runs, **Then** the 2 WMO placements show non-zero nearby PM4 objects (even if the PM4 objects are different structures).

### User Story 3 — CK24-to-model identity correlation (Priority: P2)

As a PM4 researcher, I want to learn CK24 → model path mappings from tiles with known placements, so I can resolve PM4 objects on tiles without placements.

**Why this priority**: Depends on the scorer actually working (User Story 1). Once matches are reliable, we can build a lookup table.

**Independent Test**: On tile 24_35, the correlation produces a table mapping at least 10 distinct (CK24, type) pairs to their model paths.

**Acceptance Scenarios**:

1. **Given** tile 24_35, **When** the correlation runs, **Then** it outputs (CK24, type, objectId) → (model path, confidence) for ≥10 distinct CK24 values.

### User Story 4 — Synthetic PM4 signals from WMO/M2 collision geometry (Priority: P2)

As a PM4 researcher, I want to generate what PM4 data would look like for any WMO/M2 model, so I can compare synthetic signals against real PM4 data for identification.

**Why this priority**: Depends on understanding the collision→PM4 transformation (spec 063). Once we know how collision becomes PM4 surfaces, we can reverse it.

**Independent Test**: Given OILPLATFORM_LOW.WMO, generate a synthetic signal whose footprint dimensions are within 20% of the WMO's actual group bounds.

**Acceptance Scenarios**:

1. **Given** a WMO path, **When** synthetic signal generation runs, **Then** it produces a `Pm4AssetReferenceSignalRecord` with bounds, footprint hull, and sub-part bounds from WMO group collision data.
2. **Given** OILPLATFORM_LOW.WMO, **When** comparing synthetic signal bounds against actual WMO group bounds, **Then** footprint dimensions match within 20%.

## Success Criteria

- **SC-001**: On tile 24_35, at least 1 WMO placement scores ≥0.45 via `pm4 match-assets`
- **SC-002**: On tile 24_35, `pm4 match-report` shows non-zero candidate counts for ≥5 placements
- **SC-003**: CK24 identity table for tile 24_35 has ≥10 entries mapping CK24 to model path
- **SC-004**: WMO collision-derived synthetic signals have footprint within 20% of actual group bounds

## Assumptions

- CK24 type 0x42/0x43 = WMO, 0x40/0x41 = M2, 0xC0/0xC1/0xC2 = unknown (interior/navmesh?), 0x3E/0x3F = unknown (building sub-types?)
- The 3.3.5 client at `output/tmp/wowarchive-clients/3_3_5_12340/` is the canonical validation source
- Tile 0_0 (development) is the primary proven-correlation tile (CK24 0x421809 ↔ ND_IRONDWARF_LARGEBUILDING.WMO at 61.8% overlap)
- Tile 22_18's PM4 objects are placed through mechanisms other than ADT MODF tables — the 2 oil platform placements are irrelevant to PM4 content
- The coordinate conversion from PM4 raw-ADT to WoW world is verified correct: `wowX = MapOrigin - (tileY*TileSize + pm4Y)`, `wowY = MapOrigin - (tileX*TileSize + pm4X)` for XYPlaneZUp axis convention
- PM4 data is more comprehensive than ADT placement tables — many objects in PM4 are not listed in MODF/MDDF

## Fingerprint-Based Identity Approach (2026-06-17)

The `pm4 fingerprint-scan` command reads all PM4s in a directory and extracts per-CK24-group fingerprints: (surfaces, indices, vertices, sorted bounding box sizes, coordinate mode). Key discoveries:

### Identity matching results (2026-06-17)

`pm4 identify-models` matches PM4 fingerprint groups against WMO archive local bounds (MOHD BoundsMin/BoundsMax). Results from 616 PM4s vs 506 WMO roots:

- **1223 matches** with score >= 0.30
- **545 matches** with score >= 0.95 (near-exact dimension match)
- **972 matches** with score >= 0.90
- **304 unique WMOs** matched
- **Top match**: GoldshireInn.wmo at 0.996 score (30x32x60 PM4 vs 30x32x60 WMO — exact)

Key findings:
1. **Sorted dimension ratio matching works** — WMO local bounding box dimensions are rotation-invariant identifiers. The `min(dima/dimb, dimb/dima)` formula gives high scores for the correct WMO model.
2. **0x40/0x41 pairs map to the same WMO** — confirming type pair hypothesis (M2 collision + visual).
3. **0x42/0x43 pairs map to the same WMO** — confirming type pair hypothesis (WMO exterior + interior).
4. **0xC0/0xC1/C2/C3 types also match WMOs** — these are navmesh/interior collision variants that share the same model bounds.
5. **Multi-tile objects need separate handling** — fingerprint (surfaces, indices, vertices) changes per tile for the same object. Only single-tile or dominant-tile portions match directly.
6. **506 of 1985 WMOs scanned** — archive enumeration via `GetAllKnownFiles()` missed ~75% of WMO root files. Need to use listfile-based enumeration or on-disk scanning for full coverage.

### Ck24ObjectId is a global object identifier
- Same ObjectId appears across multiple tiles for the same physical object
- OID 52202 spans 8 tiles (14592→88 surfaces per tile, total 33587 surfaces across all tiles)
- OID 43196 (type 0x43) spans 8 tiles
- OID 44166 (type 0x43) spans 7 tiles
- **Only 2 ObjectId reuse cases** across all 616 PM4s (from linkage analysis), meaning ObjectIds are globally unique per object instance

### Fingerprint stability
- Same model at different positions produces the **same** (surfaces, indices, vertices) fingerprint
- Different models produce **different** fingerprints
- Common WMO template `(35, 144, 90)` appears 97 times — different OIDs, same model template
- M2 types (0x40/0x41) often appear in **pairs** with identical fingerprints, representing the same M2 at different collision representations

### Multi-tile reconstruction
- For ObjectIds spanning multiple tiles, combining per-tile bounding boxes reconstructs the full model bounds
- OID 52202 combined bounds: 199×899×930 ADT units (a massive WMO spanning ~5 tiles)
- Single-tile bounds are always ≤ full model bounds since tile-clipping truncates edges

### Coordinate mode distribution
- 611/616 PM4s use **WorldSpace** (absolute) coordinates (`Pm4CoordinateMode.WorldSpace`)
- 5/616 use **TileLocal** (tile-relative) coordinates
- `Pm4PlacementMath.IsLikelyTileLocal()` correctly detects which mode each PM4 uses
- The `pm4 fingerprint-scan` command applies `ConvertPm4VertexToWorld` with the detected coordinate mode

### Identity matching strategy
1. Group all PM4 CK24 groups by (surfaces, indices, vertices) fingerprint + CK24 type
2. For multi-tile ObjectIds, merge bounding boxes across tiles to get full model bounds
3. Read all WMOs from the 3.3.5 archive, compute their local bounding box dimensions
4. Match fingerprint groups against WMO local bounds by **sorted dimension ratio** (invariant under rotation)
5. Match single-tile ObjectIds against WMO local bounds by checking if their single-tile bounds are a **subset** of any WMO's local bounds

Was split from 050 (WMO group matching) + 052 (signature matcher) → consolidated into 046 during 2026-06-09 spec consolidation pass.
C# side completed 2026-06-08 but never produced real matches on real data.
ADT-writing code removed 2026-06-16.
2026-06-17: Diagnosed why pipeline produces 0 matches — scorer miscalibration + coordinate mismatch + segment granularity. Rewrote spec from status-report to actual specification with testable user stories.
2026-06-17: Added `pm4 correlate-models` command. Proved CK24→WMO correlation works: CK24 0x421809 ↔ ND_IRONDWARF_LARGEBUILDING.WMO at 61.8% volumetric overlap on tile 0_0. Found that PM4 contains objects not in ADT MODF tables (B4). Found new CK24 types 0x3E/0x3F/0xC0/0xC1/0xC2 on city-area tiles.