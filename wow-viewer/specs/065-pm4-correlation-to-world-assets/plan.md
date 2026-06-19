# Implementation Plan: PM4 Surface Correlation to World Assets & Generator

**Branch**: `065-pm4-correlation-to-world-assets` | **Date**: 2026-06-17 (revised) | **Spec**: `specs/065-pm4-correlation-to-world-assets/spec.md`

## Summary

Build a transform-invariant surface triangle database from WMO collision geometry (MOVT/MOVI) by triangulating independent triangles and hashing each triangle by sorted edge lengths. Extract the same surface triangle histograms from PM4 CK24 groups by fan-triangulating MSUR convex polygons. Match PM4 surface histograms against the WMO DB using histogram intersection + symmetric F1. No ADT dependency for matching — ADT is used only for validation ground truth.

**Current state**: Phases 1–5 are implemented and validated against the staged 3.3.5 client and development PM4 corpus. The matcher produces no hull-style false positives but still collides on edge-length-only histograms. Phase 6 is the active next step.

**Approach**: Surface histogram matching. Three tracks converge:
- Track A — WMO surface fingerprints: read MOVT/MOVI from all WMOs in staged archive, triangulate, build edge-length histograms, serialize to DB JSON.
- Track B — PM4 surface fingerprints: read MSVT/MSVI/MSUR per CK24 group, fan-triangulate MSUR surfaces, build same histograms, serialize to JSON.
- Track C — matching: load both JSONs, type-filter, histogram intersection → PM4 coverage, WMO coverage, symmetric F1 → rank candidates.

## Technical Context

**Language**: C# / .NET 10
**Dependencies**: Silk.NET (existing), existing WowViewer.Core.PM4, WowViewer.Core.IO libraries
**Storage**: fingerprint DBs as JSON files on disk (loaded at match time, no re-reading WMO files)
**Testing**: `dotnet test` on WowViewer.Core.PM4.Tests, manual validation against staged client data
**Target Platform**: Windows CLI
**Project Type**: CLI tool (WowViewer.Tool.Inspect commands), with library code in WowViewer.Core.PM4

## Constitution Check

- **Repo independence**: all code in `wow-viewer/`. No cross-repo .csproj refs. ✓
- **Library-first**: surface extraction + histogram matching in Core.PM4, CLI in Tool.Inspect. ✓
- **Real-data validation**: every phase validated against `output/tmp/wowarchive-clients/3_3_5_12340/World of Warcraft` and `test_data/development/World/Maps/development`. ✓
- **No H:\CLIENTS**: clean. ✓
- **gillijimproject_refactor is read-only**: no writes. Reference only. ✓
- **No ADT for matching**: ADT used only in Phase 5 validation. ✓

## Project Structure

```
wow-viewer/
├── specs/065-pm4-correlation-to-world-assets/
│   ├── spec.md
│   ├── plan.md
│   └── tasks.md
├── src/core/WowViewer.Core.PM4/
│   ├── Services/
│   │   ├── Pm4SurfaceCorrelationExtractor.cs   # EXISTING — triangulate + histogram
│   │   ├── Pm4SurfaceCorrelationMatcher.cs     # EXISTING — histogram intersection + F1
│   │   └── Pm4Generator.cs                     # EXISTING — downstream generator
│   └── Models/
│       └── Pm4SurfaceCorrelationContracts.cs   # EXISTING — surface correlation data models
├── src/core/WowViewer.Core.IO/
│   └── Wmo/
│       ├── WmoGroupMeshDetailReader.cs         # EXISTING — MOVT/MOVI reader
│       └── WmoRenderDocumentReader.cs          # EXISTING — WMO root + embedded group reader
├── tools/inspect/WowViewer.Tool.Inspect/
│   ├── Program.cs                              # EXTEND: pm4 build-wmo-surface-db, extract-pm4-surfaces, match-surfaces, validate-matches
│   └── Pm4SurfaceBuildSupport.cs               # EXISTING — WMO DB builder + PM4 extraction
└── tests/
    └── WowViewer.Core.PM4.Tests/               # EXTEND: surface extraction + matching tests
```

## Implementation Phases

### Phase 1: Surface Extraction Library (DONE)

Built `Pm4SurfaceCorrelationExtractor`, `Pm4SurfaceCorrelationMatcher`, and `Pm4SurfaceCorrelationContracts` in Core.PM4.

**What landed**:
- `Pm4SurfaceCorrelationContracts.cs` — `SurfaceCorrelationFingerprint`, `SurfaceMatchCandidate`, `SurfaceMatchResult`.
- `Pm4SurfaceCorrelationExtractor.cs` — fan-triangulate PM4 MSUR surfaces, read WMO MOVI triangles, build edge-length histograms.
- `Pm4SurfaceCorrelationMatcher.cs` — histogram intersection, PM4 coverage, WMO coverage, symmetric F1.
- Unit tests for triangulation and histogram invariance.

### Phase 2: WMO Surface Database (DONE)

Built the WMO surface DB from the staged archive.

**What landed**:
- `Pm4SurfaceBuildSupport.BuildWmoSurfaceDatabaseAsync` reads WMO roots + groups via `WmoRenderDocumentReader`, merges group geometry for root fingerprints, and extracts per-group fingerprints.
- CLI `pm4 build-wmo-surface-db --archive-root <staged> [--bin-size <1.0>] [--area-bin-size <1.0>] --output <db.json>`.
- Validated: 503 WMO roots + 2287 groups = 2790 fingerprints, 13M triangles, 11MB JSON.

**Known gap**: archive enumeration via `GetAllKnownFiles()` finds only 503/1985 WMO roots. This is deferred to Phase 8.

### Phase 3: PM4 Surface Extraction (DONE)

Extracted surface fingerprints from all 616 development PM4s.

**What landed**:
- `Pm4SurfaceBuildSupport.BuildPm4SurfaceDatabaseAsync` groups MSUR by CK24, collects MSVT/MSVI per group, and extracts fingerprints.
- CLI `pm4 extract-pm4-surfaces --input <dir> [--bin-size <1.0>] [--area-bin-size <1.0>] --output <fp.json>`.
- Validated: 1604 CK24 group fingerprints, 604K triangles, 3MB JSON.

### Phase 4: Surface Matching (DONE)

Matched PM4 surface fingerprints against WMO DB.

**What landed**:
- `Pm4SurfaceCorrelationMatcher.Match` type-filters candidates, computes histogram intersection + F1, ranks candidates, and assigns status.
- CLI `pm4 match-surfaces --pm4-fingerprints <fp.json> --wmo-db <db.json> --output <matches.json>`.
- Validated: 217 matched, 956 ambiguous, 158 unresolved, 273 ineligible.

### Phase 5: Validation Against ADT Ground Truth (DONE)

Validated matches on ADT-backed tiles.

**What landed**:
- CLI `pm4 validate-matches --matches <matches.json> --pm4-dir <dir> --archive-root <staged> --output <report.json>`.
- Reads ADT obj0 placements, computes ground-truth CK24↔WMO pairs via legacy ADT-based support, compares surface-DB top-1/top-3.
- Results: P@1=1.3%, P@3=10.3%. Ironforge/Darnassis false positives eliminated.

### Phase 6: Add Triangle Area to Histogram Key (NEXT)

Reduce edge-length-only histogram collisions by adding triangle area to the geometric hash.

**6a. Extend the histogram key**:
- Change triangle hash from `(int edge0, int edge1, int edge2)` to `(int edge0, int edge1, int edge2, int areaBin)`.
- Bin area to a small integer (e.g., `int areaBin = (int)Math.Round(area / areaBinSize)`).
- Update `Pm4SurfaceCorrelationExtractor` to compute triangle area during triangulation.
- Add `--area-bin-size` CLI option to `build-wmo-surface-db` and `extract-pm4-surfaces`.
- Update `Pm4SurfaceCorrelationMatcher` to use the new histogram key.

**6b. Validate**:
- Re-run `build-wmo-surface-db --area-bin-size <value>`, `extract-pm4-surfaces --area-bin-size <value>`, `match-surfaces`, `validate-matches`.
- Tested `area-bin-size=1.0` and `area-bin-size=10.0`.
- `area-bin-size=1.0`: GoldshireInn false positive on tile 0_2 eliminated; ambiguous drops from 956→199; P@3 improves 10.3%→25.3%; P@1 drops 1.3%→0.0%.
- `area-bin-size=10.0`: GoldshireInn false positive returns; ambiguous 371; P@3=11.3%; P@1=0.0%.
- Conclusion: fine area bin improves precision but is too strict for recall. Phase 7 (normal + height) is needed to recover P@1. Phase 6 is complete.

### Phase 7: Placement-Invariant Geometry + Generator Validation

The first attempt to add absolute MSUR.Normal/MSUR.Height failed because WMO-local and PM4-world coordinate spaces are not comparable. Phase 7 was redesigned around placement-invariant descriptors and direct generator validation.

**7a. Placement-invariant descriptors**:
- Compute each group's area-weighted dominant normal and centroid.
- Add two optional bins: triangle normal alignment (`dot(triNormal, groupDominantNormal)`) and planar offset (`dot(triCentroid - groupCentroid, groupDominantNormal)`).
- These descriptors are rigid-invariant, so they can be compared between WMO-local and placed PM4-world geometry.
- Defaults are 0 (disabled) so the validated area-only baseline is unchanged.

**7b. Validate descriptor improvement**:
- Re-run with `--normal-alignment-bin-size 0.1 --planar-offset-bin-size 1.0`.
- Result: 0 matched, 0 ambiguous, P@3=10.1% (down from 25.3% area-only). The descriptors make the key too specific and do not recover recall.
- Conclusion: surface histogram matching is unlikely to reach high P@1 without additional signal (e.g., WMO enumeration, placement-derived spatial filtering, or a different matcher).

**7c. Generator validation and fix pass (pivot to fingerprints)**:
- Added `pm4 validate-generator-geometry` to directly test `Pm4Generator.cs`.
- For a given PM4 tile and its `_obj0.adt`, the command reads each WMO placement, reads the WMO render mesh, applies the ADT placement transform, runs `Pm4Generator.GenerateFromCollisionMesh`, and compares the generated collision fingerprint to the real PM4 CK24 groups.
- Initial run on `development_16_37`: mean symmetric score 0.004, 0/4 matched placements. `development_16_37` has only M2 groups, so it is not a valid WMO validation tile.
- Correct validation tile: `development_29_18` with 48 real CK24 WMO groups and 48 ADT WMO placements.
- Fixed in this pass:
  - Source geometry: use WMO collision faces (`MOPY` flags `0x08`/`0x20`, exclude `0x04`).
  - `MSVI` first index: write raw uint index, not byte offset.
  - `WorldToPm4Raw`: use `MapOrigin - X`, `MapOrigin - Y`, `Z` (no X/Y swap).
  - Validation uses `Pm4SurfaceCorrelationExtractor` group-relative edge+area fingerprints, not exact polygon comparison.
- Latest result on `development_29_18` (area-only bins edge=1, area=1): mean symmetric score **0.462**, **36/48** groups matched, all farm placements correlate to real group `0x43C510`.
- Insight: we do not need to reproduce real PM4 polygons exactly. The generator only needs to produce a comparable collision-shape fingerprint.

**7d. Full-corpus collision WMO DB and matching**:
- Added `ExtractFromWmoCollisionGroup` and switched `build-wmo-surface-db` to collision-only fingerprints.
- Built `wmo_collision_surface_db_335.json` from staged 3.3.5: 502 WMO roots, 2749 fingerprints, 5.5M collision triangles.
- Extracted `pm4_surface_fps_dev.json` from 616 dev PM4s: 1604 fingerprints, 604K triangles.
- `pm4 match-surfaces`: 30 matched, 195 ambiguous, 1106 unresolved, 273 ineligible.
- `pm4 validate-matches` against ADT ground truth: P@1=1.2%, P@3=28.5%.
- Collision fingerprints produce good candidate sets, but top-1 is unreliable because many unrelated WMOs share simple box-like collision shapes. The correct asset often appears in top-3 or in the unresolved candidate list.

### Phase 8: Fix WMO Enumeration via Listfile

Expand WMO DB coverage from 503 to ≥1900 WMO roots.

**8a. Listfile-based enumeration**:
- Add `--listfile <path>` option to `pm4 build-wmo-surface-db`.
- Read `componentfile.txt` or a provided listfile, filter `.wmo` entries that are root files (no `_` in basename).
- Merge archive catalog results with listfile results, deduplicate.

**8b. Validate**:
- Re-run `build-wmo-surface-db` with listfile.
- Target: ≥1900 WMO root fingerprints.
- Re-run matching and validation; ensure P@3 does not regress due to increased candidate pool.

### Phase 9: Placement Recovery / ADT Regeneration (Downstream)

Use trusted surface matches and PM4 geometry to recover MODF/MDDF placements for ADT-less tiles.

**9a. Placement transform recovery**:
- Given a matched CK24 group and its WMO fingerprint, compute the rigid transform (translation, rotation, scale) that aligns WMO collision triangles to PM4 surfaces.
- Output MODF/MDDF entry candidates.

**9b. ADT regeneration**:
- Write synthetic ADT files with recovered placements for PM4-only tiles.
- Validate on tiles where ADT exists by comparing recovered placements to original ADT MODF/MDDF entries.

**9c. Generator revisit** (in progress as part of Phase 7c):
- `Pm4Generator.cs` now uses WMO collision geometry, writes correct MSVI indices, and uses the correct world→PM4 coordinate transform.
- Remaining: resolve WMO local-axis/rotation convention and surface winding so generated surface normals/tessellation match real PM4 CK24 groups.

## Phases Ordered by Execution

| Phase | What | Depends on | Outcome |
|-------|------|-----------|---------|
| 1 | Surface extraction library (triangulate + histogram) | nothing | `Pm4SurfaceCorrelationExtractor` + `Matcher` + contracts + tests |
| 2 | WMO surface DB from staged archive | 1 | `wmo-surface-db.json` with 503+ entries |
| 3 | PM4 surface extraction from 616 files | 1 | `pm4-surfaces.json` with 1604 entries |
| 4 | Surface matching (no ADT) | 2, 3 | `matches.json` — CK24→WMO candidates |
| 5 | Validation against ADT ground truth | 4 | precision@1/@3 report |
| 6 | Add triangle area to histogram key | 5 | reduced false positives / ambiguous count |
| 7 | Placement-invariant descriptors + generator validation/fix + collision DB | 6 | generator fingerprints match 36/48 groups; full-corpus DB improves P@3 to 28.5% |
| 8 | WMO candidate disambiguation (spatial/placement filters) | 7 | P@1 improvement over 1.2% |
| 9 | Fix WMO enumeration via listfile | 8 | ≥1900 WMO root fingerprints |
| 9 | Placement recovery / ADT regeneration | 8 | synthetic ADT for PM4-only tiles |

## Notes on Abandoned Hull Approach

The previous `Pm4FingerprintExtractor` / `Pm4FingerprintMatcher` hull/PCA approach is superseded. The corresponding CLI commands (`build-wmo-fingerprint-db`, `extract-pm4-fingerprints`, `match-fingerprints`) are kept as legacy references but are not the primary matcher. Do not invest new work in hull-based matching unless validation proves it outperforms surface correlation after disambiguation improvements.
