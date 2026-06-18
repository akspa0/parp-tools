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

### Phase 7: Add Surface Normal + MSUR Height to Histogram Key

Further reduce ambiguity by incorporating surface orientation and vertical plane position.

**7a. Extend the histogram key**:
- Add PM4 MSUR.Normal (as a quantized direction) and MSUR.Height to the key.
- For WMO triangles, compute a face normal and a representative height (e.g., centroid Z).
- The histogram key becomes `(edge lengths, area, normal bucket, height bucket)`.

**7b. Validate**:
- Re-run the full pipeline.
- Target: P@3 improves toward 60%.
- Target: ambiguous group count < 400.

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

**9c. Generator revisit**:
- If correlation findings reveal a vertex-level transform between WMO collision and generated PM4, update `Pm4Generator.cs` accordingly.

## Phases Ordered by Execution

| Phase | What | Depends on | Outcome |
|-------|------|-----------|---------|
| 1 | Surface extraction library (triangulate + histogram) | nothing | `Pm4SurfaceCorrelationExtractor` + `Matcher` + contracts + tests |
| 2 | WMO surface DB from staged archive | 1 | `wmo-surface-db.json` with 503+ entries |
| 3 | PM4 surface extraction from 616 files | 1 | `pm4-surfaces.json` with 1604 entries |
| 4 | Surface matching (no ADT) | 2, 3 | `matches.json` — CK24→WMO candidates |
| 5 | Validation against ADT ground truth | 4 | precision@1/@3 report |
| 6 | Add triangle area to histogram key | 5 | reduced false positives / ambiguous count |
| 7 | Add normal + height to histogram key | 6 | further improved P@3 |
| 8 | Fix WMO enumeration via listfile | 7 | ≥1900 WMO root fingerprints |
| 9 | Placement recovery / ADT regeneration | 8 | synthetic ADT for PM4-only tiles |

## Notes on Abandoned Hull Approach

The previous `Pm4FingerprintExtractor` / `Pm4FingerprintMatcher` hull/PCA approach is superseded. The corresponding CLI commands (`build-wmo-fingerprint-db`, `extract-pm4-fingerprints`, `match-fingerprints`) are kept as legacy references but are not the primary matcher. Do not invest new work in hull-based matching unless validation proves it outperforms surface correlation after disambiguation improvements.
