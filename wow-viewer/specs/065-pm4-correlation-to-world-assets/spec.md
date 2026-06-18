# Feature Specification: PM4 Surface Correlation to World Assets & Generator

**Feature Branch**: `065-pm4-correlation-to-world-assets`

**Created**: 2026-06-17

**Status**: Active (revised 2026-06-17 — hull/footprint approach abandoned; surface triangle correlation is primary)

## The Problem

We have 616 PM4 files with 1604 CK24 groups (collision surface clusters). The PM4 format encodes surface mesh, pathing, and placement data that the WoW game client uses for AI navigation, collision queries, and scene graph placement. We do not know which WMO or M2 asset each CK24 group corresponds to.

**Previous approaches (ABANDONED):**

1. **ADT-based correlation** — correlate CK24 groups to ADT MODF/MDDF placements via geometric overlap. Wrong because:
   - 222 PM4-only tiles have no ADT — no placement anchors exist.
   - Many ADTs that exist are missing placements that PM4 contains.
   - PM4 data is more comprehensive than ADT placement tables.

2. **Convex-hull footprint matching** — PCA-normalized convex hull + sorted-dimension prefilter. Abandoned because:
   - Produced false positives: Ironforge and Darnassis scored 0.999 footprint overlap despite NOT being on the development map.
   - Convex hull throws away internal surface structure; a 12×12×48 box matches dozens of unrelated WMOs.

**Current approach: surface triangle correlation.** PM4 MSUR convex polygon surfaces are triangulated into fans; WMO MOVI independent triangles are read directly. Each triangle is reduced to a transform-invariant geometric hash: sorted edge lengths binned to integers. A histogram of these hashes is built per CK24 group and per WMO/group. Matching is histogram intersection → coverage ratios → symmetric F1 score.

**Results to date (1604 PM4 groups vs 2790 WMO surface fingerprints):**
- 217 matched, 956 ambiguous, 158 unresolved, 273 ineligible
- P@1=1.3%, P@3=10.3% (2.3× improvement over hull P@3=4.5%)
- Ironforge/Darnassis false positives eliminated
- 12 correct top-1 matches (GoldshireInn, classicalelfruins, arathistonebridge, orchut, etc.)
- Remaining false positives: GoldshireInn matched tile 0_2 at 0.86 PM4 coverage but no GoldshireInn exists there — edge-length histograms collide across different geometry with similarly-sized triangles.

**The eventual goal**: given any WMO or M2 asset, generate its PM4 collision/pathing data and recover placement transforms so we can produce complete terrain tiles (ADT + PM4 + textures) for tiles where ADT is missing or incomplete.

## What This Spec Covers

**Phase 1 — Surface triangle fingerprint database.** Extract transform-invariant surface triangle histograms from:
- WMO collision geometry (MOVT/MOVI from WMO group files via `WmoGroupMeshDetailReader`/`WmoRenderDocumentReader`) — one histogram per WMO root (merged across groups) and optionally per group.
- PM4 CK24 groups (MSVT/MSVI/MSUR per CK24 group) — one histogram per CK24 group.

**Phase 2 — Matching.** Match PM4 CK24 surface histograms against WMO surface histograms using histogram intersection + F1 scoring. No ADT input. No world position. No bounding-box shortcut.

**Phase 3 — Validation.** Validate matches on tiles where ADT ground truth exists, but use ADT only for validation, never as a matching input. Report precision@1, precision@3, coverage, and failure categorization.

**Phase 4 — Stronger disambiguation.** Reduce the 956 ambiguous cases by adding stronger geometric signals to the histogram key (triangle area, surface normal + height, surface-level structure). Triage between genuine ambiguity (e.g., Stormwind vs StormwindHarbor share architecture) and resolvable collisions.

**Phase 5 — Placement recovery / generator (downstream).** Take identified WMO/M2 assets and produce MODF/MDDF placement entries from PM4 surface transforms. (`Pm4Generator.cs` exists for PM4 chunk generation; placement writing is a separate, later phase.)

## User Stories

### User Story 1 — WMO Surface Triangle Database (P1)

**As a** PM4 researcher,
**I want** a precomputed surface triangle database covering all WMOs in the staged client archive,
**So that** I can match PM4 CK24 groups against this database without needing any ADT placement data.

**Why P1**: The surface DB is the prerequisite for all matching. Without it, every match attempt falls back to ADT-dependent or bounding-box-only approaches, both of which are proven broken.

**Independent Test**: Build the DB from the staged 3.3.5 client. Verify GoldshireInn.wmo produces a surface histogram that matches the PM4 CK24 group for GoldshireInn. Verify ≥500 WMO roots have valid surface fingerprints (≥1 triangle).

**Acceptance Scenarios**:

1. **Given** the staged client at `output/tmp/wowarchive-clients/3_3_5_12340/World of Warcraft`,
   **When** I run `pm4 build-wmo-surface-db --archive-root <staged> --output <db.json>`,
   **Then** the output DB contains ≥500 WMO root fingerprints, each with: edge-length + area histogram, triangle count, vertex count, group count, and source WMO path.

2. **Given** a WMO with multiple groups (e.g., ND_IRONDWARF_LARGEBUILDING with 4 groups),
   **When** the surface DB is built,
   **Then** the DB contains both a merged root fingerprint AND per-group fingerprints, so multi-group WMOs can be matched at either granularity.

3. **Given** a WMO with empty collision geometry (no MOVT),
   **When** the surface DB is built,
   **Then** the WMO is skipped with a warning, and the DB entry count reflects only WMOs with valid collision geometry.

---

### User Story 2 — PM4 CK24 Surface Triangle Extraction (P1)

**As a** PM4 researcher,
**I want** to extract the same transform-invariant surface triangle histogram from each PM4 CK24 group,
**So that** PM4 fingerprints and WMO fingerprints are directly comparable via the same correlation math.

**Why P1**: The extraction must be symmetric — PM4 and WMO must produce fingerprints in the same signal space for matching to produce meaningful scores.

**Independent Test**: Extract fingerprints from all 616 development PM4s. Verify the 1604 CK24 groups produce non-empty histograms. Verify OID 52202 (spans 8 tiles) produces per-tile histograms that are approximately identical across tiles (confirming transform invariance).

**Acceptance Scenarios**:

1. **Given** the 616 development PM4 files at `test_data/development/World/Maps/development`,
   **When** I run `pm4 extract-pm4-surfaces --input <dir> --output <fp.json>`,
   **Then** the output contains one fingerprint per CK24 group (1604 entries), each with: edge-length + area histogram, triangle count, vertex count, CK24 type, CK24 object ID.

2. **Given** a CK24 group that spans multiple tiles (same OID on different tiles),
   **When** I compare the histograms from each tile,
   **Then** the histogram intersection is ≥0.90 (confirming the same object produces the same fingerprint regardless of tile placement).

3. **Given** a CK24 group with <3 vertices or no surfaces,
   **When** surface extraction runs,
   **Then** the group is skipped with a warning, not crashing the batch.

---

### User Story 3 — Surface Histogram Matching Without ADT (P1)

**As a** PM4 researcher,
**I want** to match PM4 CK24 surface histograms against the WMO surface database using histogram intersection + F1 scoring, with no ADT input,
**So that** I can identify which WMO each CK24 group corresponds to on ALL 616 tiles, including the 222 PM4-only tiles.

**Why P1**: This is the core deliverable. The matching must work without ADT. ADT is only used for validation (User Story 4).

**Independent Test**: Run matching on the development corpus. Verify GoldshireInn.wmo is a top-3 match for the Goldshire CK24 group on tile 24_35. Verify no Ironforge/Darnassis false positives appear on tiles where those WMOs are not placed.

**Acceptance Scenarios**:

1. **Given** a PM4 surface fingerprint file and a WMO surface DB,
   **When** I run `pm4 match-surfaces --pm4-fingerprints <fp.json> --wmo-db <db.json> --output <matches.json>`,
   **Then** each CK24 group gets a ranked list of WMO candidates with histogram intersection, PM4 coverage, WMO coverage, symmetric F1 score, and a match status (Matched/Ambiguous/Unresolved).

2. **Given** a CK24 group of type 0x42/0x43 (WMO),
   **When** matching runs,
   **Then** only WMO fingerprints are considered as candidates (type-filtered), and the top candidate has F1 ≥ a tunable threshold OR is flagged Ambiguous if the top-2 are within the ambiguity window.

3. **Given** a CK24 group of type 0x40/0x41 (M2),
   **When** matching runs,
   **Then** M2 fingerprints are considered (if M2 surface extraction is implemented) or the group is flagged as `Ineligible` with a clear rationale (if M2 fingerprints are not yet in the DB).

4. **Given** the 222 PM4-only tiles (no ADT),
   **When** matching runs on all 616 tiles,
   **Then** the PM4-only tiles produce the same match rate as ADT-backed tiles (no degradation from missing ADT).

---

### User Story 4 — Validation Against ADT Ground Truth (P2)

**As a** PM4 researcher,
**I want** to validate surface-DB matches on tiles where ADT ground truth exists,
**So that** I can measure match accuracy, detect false positives, and tune the scoring thresholds.

**Why P2**: Validation requires the matching to work first (P1). ADT is used ONLY as ground truth, never as a matching input.

**Independent Test**: On tiles with ADT placements, compare surface-DB top-1/top-3 matches against ADT MODF/MDDF placement overlaps. Report precision@1, precision@3, and coverage.

**Acceptance Scenarios**:

1. **Given** surface-DB matches for the development corpus and ADT ground truth for ADT-backed tiles,
   **When** I run `pm4 validate-matches --matches <matches.json> --pm4-dir <dir> --archive-root <staged> --output <report.json>`,
   **Then** a report is produced with: precision@1, precision@3, coverage, and per-CK24-group match-vs-ground-truth comparison.

2. **Given** the validation report shows false positives (e.g., a WMO matched to a tile where it is not placed),
   **When** I examine the failures,
   **Then** the failure cases are categorized (histogram collision, missing WMO in DB, multi-group WMO, degenerate geometry) so the scoring can be tuned.

---

### User Story 5 — Stronger Disambiguation Signals (P2)

**As a** PM4 researcher,
**I want** to add triangle area, surface normal, and MSUR plane height to the histogram key,
**So that** edge-length histogram collisions (e.g., GoldshireInn on tile 0_2) are reduced and the 956 ambiguous cases are resolved.

**Why P2**: Edge-length-only histograms are too coarse. Different WMOs with similarly-sized triangles collide. Adding area, normal, and surface structure distinguishes them.

**Independent Test**: After adding area to the histogram key, re-run matching. Verify the GoldshireInn false positive on tile 0_2 is eliminated or downranked, while true GoldshireInn matches remain high.

**Acceptance Scenarios**:

1. **Given** a surface match run with edge-length-only histograms,
   **When** I re-run with `(edge lengths, triangle area)` as the histogram key,
   **Then** the number of ambiguous groups decreases and the number of known false positives decreases.

2. **Given** PM4 MSUR surfaces with Normal + Height fields,
   **When** the histogram key includes `(normal, height, edge lengths, area)`,
   **Then** floor vs wall vs ceiling surfaces are distinguishable, reducing mismatches between buildings with similar footprints but different vertical structure.

---

### User Story 6 — Placement Recovery for ADT-Less Tiles (P3)

**As a** terrain pipeline developer,
**I want** to take a matched WMO/M2 asset and the PM4 surfaces that matched it, and recover a MODF/MDDF placement entry,
**So that** I can regenerate ADT placement data for tiles where ADT does not exist.

**Why P3**: This is the end goal. It depends on the correlation analysis from P1/P2 confirming the geometric relationship between PM4 and WMO collision, and on disambiguation (P2) being strong enough to trust matches.

**Independent Test**: For a tile where ADT exists, run the full pipeline: surface match → identify WMO → extract placement transform from PM4 → write MODF entry → compare to the original ADT MODF entry. Position/rotation should match within a small tolerance.

**Acceptance Scenarios**:

1. **Given** a trusted surface match between a PM4 CK24 group and a WMO,
   **When** I run placement recovery,
   **Then** the output MODF entry contains the correct model path, position, rotation, and scale derived from the PM4 surface geometry.

2. **Given** a PM4-only tile (no ADT),
   **When** the full pipeline runs,
   **Then** the output is a synthetic ADT with MODF/MDDF entries for all matched assets, ready for manual review or further processing.

## Requirements

### Functional Requirements

- **FR-001**: System MUST read WMO collision geometry (MOVT/MOVI/MOPY) from WMO root and group files via `WmoRenderDocumentReader` / `WmoGroupMeshDetailReader` and compute a per-WMO-root merged surface histogram AND per-group surface histograms.
- **FR-002**: System MUST read PM4 CK24 groups (MSVT/MSVI/MSUR) and compute one surface histogram per CK24 group.
- **FR-003**: Surface extraction MUST triangulate PM4 MSUR convex polygon surfaces into triangle fans, and read WMO MOVI indices as independent triangles.
- **FR-004**: Surface extraction MUST compute a transform-invariant geometric hash per triangle from sorted edge lengths (binned to integers). Triangles with zero area MUST be skipped.
- **FR-005**: Surface extraction MUST aggregate per-triangle hashes into a histogram per CK24 group / per WMO root / per WMO group.
- **FR-006**: Surface fingerprint MUST include topology signals: triangle count, total index count, vertex count, CK24 type byte, TypeFlags profile.
- **FR-007**: Matching MUST use histogram intersection between PM4 and WMO histograms, producing PM4 coverage, WMO coverage, and symmetric F1 score.
- **FR-008**: Matching MUST type-filter candidates: 0x42/0x43/0xC0-0xC3 CK24 groups match against WMO fingerprints only; 0x40/0x41 match against M2 fingerprints (when available) or are flagged Ineligible.
- **FR-009**: Matching MUST NOT use ADT placement data, world position, `ReferencePosition`, `TileCoordinates`, or any position-dependent signal as a scoring input. ADT is ONLY used in validation (FR-014).
- **FR-010**: System MUST serialize the WMO surface database to a JSON file on disk, loadable by the matching command without re-reading WMO files.
- **FR-011**: System MUST serialize PM4 surface fingerprints to a JSON file on disk, loadable by the matching command.
- **FR-012**: Matching MUST classify each CK24 group as Matched, Ambiguous, Unresolved, or Ineligible based on tunable score and ambiguity-window thresholds.
- **FR-013**: Validation MUST compare surface-DB matches against ADT MODF/MDDF ground truth on tiles where ADT exists, and report precision@1, precision@3, coverage, and failure categorization.
- **FR-014**: Stronger-disambiguation iteration MUST support extending the histogram key with additional per-triangle signals (area, normal, MSUR height) without rewriting the matching core.
- **FR-015**: System MUST generate PM4 from WMO collision data: MSVT from MOVT, MSVI from MOVI, MSUR from triangulated groups, MSCN from placement origin, MSLK from edge connectivity. (Downstream — already partially implemented.)
- **FR-016**: System MUST assign correct CK24 keys (type byte + object ID convention) to generated PM4 groups.

### Key Entities

- **WMO Surface Fingerprint**: Transform-invariant geometric signature of a WMO's collision triangles. Computed from MOVT/MOVI via edge-length histograms. Stored in surface DB.
- **PM4 Surface Fingerprint**: Same signature computed from a PM4 CK24 group's triangulated MSUR surfaces. Directly comparable to WMO fingerprints.
- **CK24 Group**: A cluster of collision surfaces in the PM4. Keyed by a 24-bit value: high byte = type (0x40/0x41=M2, 0x42/0x43=WMO, 0xC0-C3=WMO nav variants), low 16 bits = object ID.
- **Triangle Hash**: Sorted edge lengths of a triangle, binned to integers. Transform-invariant under translation, rotation, and scale (scale invariance comes from binning ratios, or absolute lengths if scale is preserved).
- **Histogram Intersection**: Sum of min(PM4 bin count, WMO bin count) normalized by total bin count. Measures how much of the WMO's triangle repertoire is present in the PM4 group.
- **Symmetric F1**: Harmonic mean of PM4 coverage and WMO coverage. Penalizes both missing PM4 triangles and missing WMO triangles.
- **MOVT/MOVI**: WMO group collision vertices and indices. The raw collision mesh of a WMO group.
- **MSVT/MSVI/MSUR**: PM4 collision vertices, indices, and surface records. MSUR surfaces are convex polygons (3–12 indices) that must be fan-triangulated.

## Success Criteria

### Measurable Outcomes

- **SC-001**: WMO surface DB contains ≥500 WMO root fingerprints with valid (non-empty) triangle histograms. Stretch: ≥1900 after fixing archive enumeration / listfile gap.
- **SC-002**: PM4 surface extraction produces 1604 CK24 group fingerprints with non-empty histograms.
- **SC-003**: No false positives on known-negative cases: Ironforge and Darnassis do not appear as top-3 matches on development tiles where they are not placed.
- **SC-004**: On PM4-only tiles (no ADT), matching produces the same match rate as ADT-backed tiles (no degradation from missing ADT).
- **SC-005**: Validation on ADT-backed tiles shows P@3 ≥ 10% as a baseline for edge-length-only histograms; area-only improves P@3 to ~25% with fine binning (area-bin-size=1.0) at the cost of P@1; full P@3 ≥ 60% requires Phase 7 (area + normal + height).
- **SC-006**: Multi-tile OID (e.g., 52202 spanning 8 tiles) produces cross-tile histogram overlap ≥0.90, confirming transform invariance.
- **SC-007**: Matching the full 616-PM4 corpus against the WMO DB completes in <60 seconds.
- **SC-008**: Ambiguous group count is reduced from 956 to <400 by stronger disambiguation signals. With area-bin-size=1.0 ambiguous drops to 199; with area-bin-size=10.0 ambiguous is 371.

## Assumptions

- PM4 vertices are in world space for 611/616 files; 5 use tile-local. Triangle edge lengths and area are invariant to coordinate mode, so normalization is unnecessary.
- WMO collision vertices (MOVT) are in the WMO's local coordinate space. Edge-length histograms are invariant to the local frame.
- Edge-length histograms are a necessary but insufficient signal. They eliminate the worst hull false positives but still produce collisions; future iterations add area, normal, and height.
- Sorted dimensions / bounding box may be used as a fast prefilter, but not as the primary matching signal.
- Staged client root: `output/tmp/wowarchive-clients/3_3_5_12340/World of Warcraft`.
- Reference: `gillijimproject_refactor` is read-only; code is ported/mirrored, never modified in place.
- ADT is used ONLY for validation ground truth, never as a matching input.

## Edge Cases

- What happens when a WMO has no collision geometry (empty MOVT)? Skip the WMO, warn, exclude from DB.
- What happens when a CK24 group has no surfaces or <3 vertices? Skip the group, warn, exclude from matching.
- What happens when two different WMOs produce identical edge-length histograms? Flag as Ambiguous; stronger signals (area, normal, height) are required to resolve.
- What happens when the top-2 WMO candidates are within the ambiguity window? Flag as Ambiguous, report both.
- What happens when a CK24 type is not 0x40/0x41/0x42/0x43/0xC0-0xC3? Flag as Ineligible, exclude from matching.
- What happens when M2 fingerprints are not yet in the DB? 0x40/0x41 CK24 groups are flagged Ineligible with rationale "M2 surface DB not yet built."
- What happens when a PM4 MSUR surface is a quad or n-gon (IndexCount > 3)? Fan-triangulate from the first vertex; warn if the polygon is non-convex.
- What happens when a matched CK24 group spans multiple tiles? Each tile's partial surface set is matched independently; multi-tile ObjectIds may need merging before placement recovery.

## Cross-References

- `wow-viewer/docs/architecture/pm4-chunk-semantics.md` — authoritative chunk-by-chunk semantics
- `wow-viewer/docs/architecture/pm4-region-aware-object-grouping-2026-05-21.md` — CK24 grouping and MSHD.Field04
- `wow-viewer/src/core/WowViewer.Core.PM4/Services/Pm4SurfaceCorrelationExtractor.cs` — surface triangulation + histogram extraction
- `wow-viewer/src/core/WowViewer.Core.PM4/Services/Pm4SurfaceCorrelationMatcher.cs` — histogram intersection + F1 scoring
- `wow-viewer/src/core/WowViewer.Core.PM4/Models/Pm4SurfaceCorrelationContracts.cs` — surface correlation data models
- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Pm4SurfaceBuildSupport.cs` — WMO surface DB builder + PM4 extraction CLI support
- `wow-viewer/src/core/WowViewer.Core.PM4/Services/Pm4Generator.cs` — downstream PM4 generator from WMO collision
- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoGroupMeshDetailReader.cs` — WMO group MOVT/MOVI reader
- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoRenderDocumentReader.cs` — WMO root + embedded group reader
- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Pm4CorrelateModelsSupport.cs` — legacy ADT-based support (kept for validation ground truth)

## Legacy / Abandoned Approaches

- **Hull/footprint matching (`Pm4FingerprintExtractor`, `Pm4FingerprintMatcher`)**: Abandoned due to false positives. Code kept as reference but superseded by surface correlation.
- **ADT-based correlation (`pm4 correlate-models`, `sweep-correlate`, `match-assets`)**: Kept for validation ground truth only. Not used as primary matchers.
