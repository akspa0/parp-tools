# Feature Specification: PM4 Correlation to World Assets & Generator

**Feature Branch**: `065-pm4-correlation-to-world-assets`

**Created**: 2026-06-17

**Status**: Active (revised 2026-06-17 — pivoted from ADT-based correlation to fingerprint-database correlation)

## The Problem

We have 616 PM4 files with 1604 CK24 groups (collision surface clusters). The PM4 format encodes surface mesh, pathing, and placement data that the WoW game client uses for AI navigation, collision queries, and scene graph placement. We do not know which WMO or M2 asset each CK24 group corresponds to.

Previous approach (ABANDONED): correlate CK24 groups to ADT MODF/MDDF placement records via geometric overlap. This is wrong because:
- 222 PM4-only tiles have no ADT — no placement anchors exist.
- `pm4 correlate-models` compares PM4 world bounds vs ADT placement world bounds (3D IoU) — dead on PM4-only tiles.
- `pm4 identify-models` uses sorted-dimension AABB matching only — too coarse, dozens of WMOs share ~33×35×53.
- `pm4 match-assets` has a `sameTileBonus` and typed world-space overlap that depend on ADT-derived `ReferencePosition`/`TileCoordinates` — dead on PM4-only tiles.

The right approach: build a **fingerprint database** from WMO collision geometry directly, using the rotation-invariant convex-hull footprint correlation math we already built (`Pm4CorrelationMath`). Match PM4 CK24 groups against this DB. No ADT, no world position, no bounding-box-only shortcut.

The eventual goal: given any WMO or M2 asset, **generate its PM4 collision/pathing data** so we can produce complete terrain tiles (ADT + PM4 + textures) without needing game client PM4 files.

## What This Spec Covers

**Phase 1 — Fingerprint database.** Extract rotation-invariant geometric fingerprints from:
- WMO collision geometry (MOVT/MOVI from WMO group files via `WmoGroupMeshDetailReader`/`WmoRenderDocumentReader`) — one fingerprint per WMO root (merged across groups) and optionally per group.
- PM4 CK24 groups (MSVT/MSVI/MSUR per CK24 group) — one fingerprint per CK24 group.

Fingerprint signals (extracted via `Pm4CorrelationMath.BuildObjectStatesFromGeometry`):
- Convex hull footprint (XY projection), PCA-normalized for rotation invariance.
- Bounds (sorted dimensions as fast prefilter, full AABB for overlap).
- Footprint area, volume, diagonal, height, aspect ratio.
- Surface count, total index count, vertex count (topology fingerprint).
- TypeFlags profile (semantic fingerprint: 0x03=M2 top, 0x10=interior floor, 0x12=exterior wall).
- CK24 type byte (0x40/0x41=M2, 0x42/0x43=WMO, 0xC0-0xC3=WMO nav variants).

**Phase 2 — Matching.** Match PM4 CK24 fingerprints against WMO fingerprints using `Pm4CorrelationMath.EvaluateMetrics` + `CompareCandidateScores`:
- Sorted-dimension prefilter (fast rejection of dimensionally-incompatible WMOs).
- PCA-normalized convex hull footprint overlap (`ComputeConvexFootprintOverlapRatio`).
- Symmetric footprint distance (`ComputeSymmetricFootprintDistance`).
- Planar gap, vertical gap, center distance (all computed in normalized local space).
- TypeFlags profile consistency check.

**Phase 3 — Validation.** Validate matches on tiles where ADT ground truth EXISTS (280+ tiles) — but use ADT only for validation, never as a matching input. Compare fingerprint-DB match results against ADT-derived ground truth pairs.

**Phase 4 — Generator.** Take any WMO/M2 asset and produce PM4 chunks from its collision geometry. (Already partially done — `Pm4Generator.cs` exists. Kept as downstream phase.)

## User Stories

### User Story 1 — WMO Collision Fingerprint Database (P1)

**As a** PM4 researcher,
**I want** a precomputed fingerprint database covering all WMOs in the staged client archive, with each WMO's collision geometry reduced to a rotation-invariant convex-hull footprint + topology fingerprint,
**So that** I can match PM4 CK24 groups against this database without needing any ADT placement data.

**Why P1**: The fingerprint DB is the prerequisite for all matching. Without it, every match attempt falls back to ADT-dependent or bounding-box-only approaches, both of which are proven broken.

**Independent Test**: Build the DB from the staged 3.3.5 client. Verify GoldshireInn.wmo produces a fingerprint with sorted dimensions ~30×32×60 and a convex hull footprint matching its known L-shaped plan. Verify ≥500 WMO roots have valid fingerprints (non-degenerate hulls, non-zero footprint area).

**Acceptance Scenarios**:

1. **Given** the staged client at `output/tmp/wowarchive-clients/3_3_5_12340/World of Warcraft`,
   **When** I run `pm4 build-wmo-fingerprint-db --archive-root <staged> --output <db.json>`,
   **Then** the output DB contains ≥500 WMO root fingerprints, each with: convex hull footprint (PCA-normalized), sorted dimensions, bounds, footprint area, surface count, vertex count, group count.

2. **Given** a WMO with multiple groups (e.g., ND_IRONDWARF_LARGEBUILDING with 4 groups),
   **When** the fingerprint DB is built,
   **Then** the DB contains both a merged root fingerprint AND per-group fingerprints, so multi-group WMOs can be matched at either granularity.

3. **Given** a WMO with empty collision geometry (no MOVT),
   **When** the fingerprint DB is built,
   **Then** the WMO is skipped with a warning, and the DB entry count reflects only WMOs with valid collision geometry.

---

### User Story 2 — PM4 CK24 Fingerprint Extraction (P1)

**As a** PM4 researcher,
**I want** to extract the same rotation-invariant fingerprint from each PM4 CK24 group,
**So that** PM4 fingerprints and WMO fingerprints are directly comparable via the same correlation math.

**Why P1**: The fingerprint extraction must be symmetric — PM4 and WMO must produce fingerprints in the same signal space for `EvaluateMetrics` to produce meaningful scores.

**Independent Test**: Extract fingerprints from all 616 development PM4s. Verify the 1604 CK24 groups produce fingerprints with non-degenerate convex hulls. Verify OID 52202 (spans 8 tiles) produces per-tile fingerprints whose PCA-normalized hulls are approximately identical across tiles (confirming rotation invariance).

**Acceptance Scenarios**:

1. **Given** the 616 development PM4 files at `test_data/development/World/Maps/development`,
   **When** I run `pm4 extract-pm4-fingerprints --input <dir> --output <fingerprints.json>`,
   **Then** the output contains one fingerprint per CK24 group (1604 entries), each with: PCA-normalized convex hull footprint, sorted dimensions, bounds, footprint area, surface count, vertex count, CK24 type, CK24 object ID.

2. **Given** a CK24 group that spans multiple tiles (same OID on different tiles),
   **When** I compare the PCA-normalized hulls from each tile,
   **Then** the hull overlap ratio is ≥0.90 (confirming the normalization is rotation-invariant and the same object produces the same fingerprint regardless of tile placement).

3. **Given** a CK24 group with <3 vertices or degenerate geometry,
   **When** fingerprint extraction runs,
   **Then** the group is skipped with a warning, not crashing the batch.

---

### User Story 3 — Fingerprint Matching Without ADT (P1)

**As a** PM4 researcher,
**I want** to match PM4 CK24 fingerprints against the WMO fingerprint database using `Pm4CorrelationMath.EvaluateMetrics` + `CompareCandidateScores`, with no ADT input,
**So that** I can identify which WMO each CK24 group corresponds to on ALL 616 tiles, including the 222 PM4-only tiles.

**Why P1**: This is the core deliverable. The matching must work without ADT. ADT is only used for validation (User Story 4).

**Independent Test**: Run matching on tile `development_24_35` (known Goldshire area). Verify GoldshireInn.wmo is the top-1 match for the CK24 group with sorted dimensions ~30×32×60, with footprint overlap ≥0.80 and a clear margin over the second-best candidate.

**Acceptance Scenarios**:

1. **Given** a PM4 fingerprint file and a WMO fingerprint DB,
   **When** I run `pm4 match-fingerprints --pm4-fingerprints <fp.json> --wmo-db <db.json> --output <matches.json>`,
   **Then** each CK24 group gets a ranked list of WMO candidates with `Pm4CorrelationMetrics` (footprint overlap, footprint distance, planar gap, vertical gap, center distance) and a match status (Matched/Ambiguous/Unresolved).

2. **Given** a CK24 group of type 0x42/0x43 (WMO),
   **When** matching runs,
   **Then** only WMO fingerprints are considered as candidates (type-filtered), and the top candidate has footprint overlap ≥0.45 OR is flagged Ambiguous if the top-2 are within 0.03.

3. **Given** a CK24 group of type 0x40/0x41 (M2),
   **When** matching runs,
   **Then** M2 fingerprints are considered (if M2 fingerprint extraction is implemented) or the group is flagged as `Ineligible` with a clear rationale (if M2 fingerprints are not yet in the DB).

4. **Given** the 222 PM4-only tiles (no ADT),
   **When** matching runs on all 616 tiles,
   **Then** the PM4-only tiles produce the same match rate as ADT-backed tiles (no degradation from missing ADT).

---

### User Story 4 — Validation Against ADT Ground Truth (P2)

**As a** PM4 researcher,
**I want** to validate fingerprint-DB matches on tiles where ADT ground truth exists,
**So that** I can measure match accuracy and tune the scoring thresholds.

**Why P2**: Validation requires the matching to work first (P1). ADT is used ONLY as ground truth, never as a matching input.

**Independent Test**: On tile `development_00_00`, compare fingerprint-DB top-1 matches against ADT MODF/MDDF placement overlaps. Report precision@1, precision@3, and coverage.

**Acceptance Scenarios**:

1. **Given** fingerprint-DB matches for tile 00_00 and ADT ground truth placements for tile 00_00,
   **When** I run `pm4 validate-matches --matches <matches.json> --adt-ground-truth <tile.adt>`,
   **Then** a report is produced with: precision@1, precision@3, coverage, and per-CK24-group match-vs-ground-truth comparison.

2. **Given** the validation report shows precision@1 < 0.50,
   **When** I examine the failures,
   **Then** the failure cases are categorized (dimension collision, PCA axis flip, wrong type filter, degenerate hull) so the scoring can be tuned.

---

### User Story 5 — PM4 Generator for Any WMO (P3)

**As a** terrain pipeline developer,
**I want** a CLI tool that takes a WMO file path and produces a valid PM4 file,
**So that** I can generate PM4 collision data for any WMO without needing game client PM4 files.

**Why P3**: This is the end goal. It depends on the correlation analysis from P1/P2 confirming the geometric relationship between PM4 and WMO collision. (Already partially implemented — `Pm4Generator.cs` exists.)

**Independent Test**: Generate PM4 for ND_IRONDWARF_LARGEBUILDING.WMO, then verify the output PM4 has the same surface count, vertex count, and CK24 structure as the game client's original PM4 data for that WMO.

**Acceptance Scenarios**:

1. **Given** a WMO file with known collision geometry,
   **When** I run `pm4 generate --wmo path/to/wmo.wmo --output out.pm4`,
   **Then** the output file is a valid PM4 with MSVT/MSVI/MSUR/MSCN/MSLK chunks.

2. **Given** generated PM4 data for a WMO,
   **When** I overlay it with the original game client PM4 for the same WMO,
   **Then** the vertex positions match within ≤0.1 unit tolerance.

## Requirements

### Functional Requirements

- **FR-001**: System MUST read WMO collision geometry (MOVT/MOVI/MOPY) from WMO root and group files via `WmoRenderDocumentReader` / `WmoGroupMeshDetailReader` and compute a per-WMO-root merged fingerprint AND per-group fingerprints.
- **FR-002**: System MUST read PM4 CK24 groups (MSVT/MSVI/MSUR) and compute one fingerprint per CK24 group.
- **FR-003**: Fingerprint extraction MUST use `Pm4CorrelationMath.BuildObjectStatesFromGeometry` (or equivalent) to compute: convex hull footprint, bounds, footprint area, center.
- **FR-004**: Fingerprint extraction MUST PCA-normalize the geometry (center at centroid, align to principal axes) before hull extraction, so footprints are rotation-invariant. Both PCA axis flip candidates must be tried for near-symmetric shapes.
- **FR-005**: Fingerprint MUST include topology signals: surface count, total index count, vertex count, CK24 type byte, TypeFlags profile.
- **FR-006**: Fingerprint MUST include sorted dimensions (rotation-invariant AABB span) as a fast prefilter signal.
- **FR-007**: System MUST serialize the WMO fingerprint database to a JSON file on disk, loadable by the matching command without re-reading WMO files.
- **FR-008**: System MUST serialize PM4 fingerprints to a JSON file on disk, loadable by the matching command.
- **FR-009**: Matching MUST use `Pm4CorrelationMath.EvaluateMetrics` to compute `Pm4CorrelationMetrics` (planar gap, vertical gap, center distance, planar overlap ratio, volume overlap ratio, footprint overlap ratio, footprint area ratio, footprint distance) for each PM4↔WMO fingerprint pair.
- **FR-010**: Matching MUST use `Pm4CorrelationMath.CompareCandidateScores` to rank WMO candidates per PM4 CK24 group.
- **FR-011**: Matching MUST type-filter candidates: 0x42/0x43/0xC0-0xC3 CK24 groups match against WMO fingerprints only; 0x40/0x41 match against M2 fingerprints (when available) or are flagged Ineligible.
- **FR-012**: Matching MUST apply a sorted-dimension prefilter (reject WMOs whose sorted dimensions differ by >25% on any axis) before computing full footprint overlap, for performance.
- **FR-013**: Matching MUST NOT use ADT placement data, world position, `ReferencePosition`, `TileCoordinates`, or any position-dependent signal as a scoring input. ADT is ONLY used in validation (FR-014).
- **FR-014**: Validation MUST compare fingerprint-DB matches against ADT MODF/MDDF ground truth on tiles where ADT exists, and report precision@1, precision@3, coverage, and failure categorization.
- **FR-015**: System MUST generate PM4 from WMO collision data: MSVT from MOVT, MSVI from MOVI, MSUR from triangulated groups, MSCN from placement origin, MSLK from edge connectivity. (Downstream — already partially implemented.)
- **FR-016**: System MUST assign correct CK24 keys (type byte + object ID convention) to generated PM4 groups.

### Key Entities

- **WMO Fingerprint**: Rotation-invariant geometric signature of a WMO's collision geometry. Computed from MOVT/MOVI via PCA normalization + convex hull. Stored in fingerprint DB.
- **PM4 Fingerprint**: Same signature computed from a PM4 CK24 group's MSVT/MSVI/MSUR. Directly comparable to WMO fingerprints.
- **CK24 Group**: A cluster of collision surfaces in the PM4. Keyed by a 24-bit value: high byte = type (0x40/0x41=M2, 0x42/0x43=WMO, 0xC0-C3=WMO nav), low 16 bits = object ID.
- **PCA Normalization**: Center geometry at centroid, align principal axes via PCA on XY-projected points. Produces a canonical rotation-invariant local frame. Both axis-flip candidates are tried for near-symmetric shapes.
- **Convex Hull Footprint**: The convex hull of the XY-projected geometry points after PCA normalization. Used for `ComputeConvexFootprintOverlapRatio` (convex polygon clipping intersection) and `ComputeSymmetricFootprintDistance` (nearest-neighbor hull distance).
- **Sorted Dimensions**: The three AABB spans (dx, dy, dz) sorted ascending. Rotation-invariant. Used as a fast prefilter.
- **TypeFlags Profile**: Distribution of MSLK.TypeFlags values (0x03=M2 top, 0x10=interior floor, 0x12=exterior solid) across surfaces in a CK24 group. Semantic fingerprint for type consistency checking.
- **MOVT/MOVI**: WMO group collision vertices and indices. The raw collision mesh of a WMO group.
- **MSVT/MSVI/MSUR**: PM4 collision vertices, indices, and surface records. The raw collision mesh of a CK24 group.

## Success Criteria

### Measurable Outcomes

- **SC-001**: WMO fingerprint DB contains ≥500 WMO root fingerprints with valid (non-degenerate) convex hull footprints.
- **SC-002**: PM4 fingerprint extraction produces 1604 CK24 group fingerprints with non-degenerate hulls.
- **SC-003**: On tile 24_35 (Goldshire), GoldshireInn.wmo is the top-1 match for the ~30×32×60 CK24 group with footprint overlap ≥0.80.
- **SC-004**: On PM4-only tiles (no ADT), matching produces the same match rate as ADT-backed tiles (no degradation from missing ADT).
- **SC-005**: Validation on ADT-backed tiles shows precision@1 ≥ 0.40 and precision@3 ≥ 0.60 (baseline; to be tuned).
- **SC-006**: Multi-tile OID (e.g., 52202 spanning 8 tiles) produces PCA-normalized hulls with cross-tile overlap ≥0.90, confirming rotation invariance.
- **SC-007**: Matching the full 616-PM4 corpus against the WMO DB completes in <60 seconds (sorted-dimension prefilter + footprint overlap on survivors only).

## Assumptions

- PM4 vertices are in world space for 611/616 files; 5 use tile-local. PCA normalization makes this irrelevant — both world-space and tile-local geometries are normalized to the same canonical frame.
- WMO collision vertices (MOVT) are in the WMO's local coordinate space. PCA normalization makes this comparable to PM4 fingerprints.
- PCA normalization handles arbitrary yaw rotation. For near-symmetric shapes (e.g., square buildings), both principal-axis flip candidates are tried and the best match is kept.
- Sorted dimensions serve as a fast prefilter: WMOs with >25% dimension mismatch on any axis are rejected before the expensive convex-hull overlap computation.
- Staged client root: `output/tmp/wowarchive-clients/3_3_5_12340/World of Warcraft`.
- Reference: `gillijimproject_refactor` is read-only; code is ported/mirrored, never modified in place.
- ADT is used ONLY for validation ground truth, never as a matching input.

## Edge Cases

- What happens when a WMO has no collision geometry (empty MOVT)? Skip the WMO, warn, exclude from DB.
- What happens when a CK24 group has <3 vertices or degenerate geometry (zero-area hull)? Skip the group, warn, exclude from matching.
- What happens when PCA produces a near-degenerate principal axis (e.g., perfectly circular footprint)? Fall back to sorted-dimension-only matching for that pair, flag as low-confidence.
- What happens when the top-2 WMO candidates are within 0.03 score? Flag as Ambiguous, report both.
- What happens when a CK24 type is not 0x40/0x41/0x42/0x43/0xC0-0xC3? Flag as Ineligible, exclude from matching.
- What happens when M2 fingerprints are not yet in the DB? 0x40/0x41 CK24 groups are flagged Ineligible with rationale "M2 fingerprint DB not yet built."

## Cross-References

- `wow-viewer/docs/architecture/pm4-chunk-semantics.md` — authoritative chunk-by-chunk semantics
- `wow-viewer/docs/architecture/pm4-region-aware-object-grouping-2026-05-21.md` — CK24 grouping and MSHD.Field04
- `wow-viewer/src/core/WowViewer.Core.PM4/Services/Pm4CorrelationMath.cs` — THE correlation math (fingerprint extraction + matching)
- `wow-viewer/src/core/WowViewer.Core.PM4/Models/Pm4CorrelationContracts.cs` — correlation contracts (ObjectState, Metrics, CandidateScore)
- `wow-viewer/src/core/WowViewer.Core.PM4/Matching/Pm4AssetMatchScorer.cs` — existing scorer (to be refactored: remove ADT-dependent signals, use Pm4CorrelationMath matching)
- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoGroupMeshDetailReader.cs` — WMO group MOVT/MOVI reader
- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoRenderDocumentReader.cs` — WMO root + embedded group reader
- `wow-viewer/src/core/WowViewer.Core.PM4/Services/Pm4PlacementMath.cs` — coordinate conversion (still needed for generator, NOT for matching)
- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Pm4CorrelateModelsSupport.cs` — legacy ADT-based support (to be superseded, not deleted — kept for validation ground truth)
