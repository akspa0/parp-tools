# Implementation Plan: PM4 Correlation to World Assets & Generator

**Branch**: `065-pm4-correlation-to-world-assets` | **Date**: 2026-06-17 (revised) | **Spec**: `specs/065-pm4-correlation-to-world-assets/spec.md`

## Summary

Build a rotation-invariant fingerprint database from WMO collision geometry (MOVT/MOVI) using `Pm4CorrelationMath`'s convex-hull footprint extraction with PCA normalization. Extract the same fingerprints from PM4 CK24 groups. Match PM4 fingerprints against the WMO DB using `EvaluateMetrics` + `CompareCandidateScores`. No ADT dependency for matching — ADT is used only for validation ground truth.

**Approach**: Fingerprint DB. Three tracks converge:
- Track A — WMO fingerprints: read MOVT/MOVI from all WMOs in staged archive, PCA-normalize, extract convex hull + topology, serialize to DB JSON.
- Track B — PM4 fingerprints: read MSVT/MSVI/MSUR per CK24 group, PCA-normalize, extract same signals, serialize to JSON.
- Track C — matching: load both JSONs, sorted-dimension prefilter, `EvaluateMetrics` on survivors, `CompareCandidateScores` for ranking.

## Technical Context

**Language**: C# / .NET 10
**Dependencies**: Silk.NET (existing), existing WowViewer.Core.PM4, WowViewer.Core.IO libraries
**Storage**: fingerprint DBs as JSON files on disk (loaded at match time, no re-reading WMO files)
**Testing**: `dotnet test` on WowViewer.Core.PM4.Tests, manual validation against staged client data
**Target Platform**: Windows CLI
**Project Type**: CLI tool (WowViewer.Tool.Inspect commands), with library code in WowViewer.Core.PM4

## Constitution Check

- **Repo independence**: all code in `wow-viewer/`. No cross-repo .csproj refs. ✓
- **Library-first**: fingerprint extraction + PCA normalization + matching in Core.PM4, CLI in Tool.Inspect. ✓
- **Real-data validation**: every phase validated against `output/tmp/wowarchive-clients/3_3_5_12340/World of Warcraft`. ✓
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
│   │   ├── Pm4CorrelationMath.cs          # EXISTING — the right correlation math
│   │   ├── Pm4FingerprintExtractor.cs     # NEW — PCA normalize + hull + topology signals
│   │   ├── Pm4FingerprintMatcher.cs       # NEW — sorted-dim prefilter + EvaluateMetrics + rank
│   │   └── Pm4Generator.cs               # EXISTING — downstream generator (Phase 6)
│   └── Models/
│       ├── Pm4CorrelationContracts.cs     # EXISTING — ObjectState, Metrics, CandidateScore
│       └── Pm4FingerprintContracts.cs     # NEW — FingerprintRecord, FingerprintDB, MatchResult
├── src/core/WowViewer.Core.IO/
│   └── Wmo/
│       └── WmoGroupMeshDetailReader.cs    # EXISTING — MOVT/MOVI reader (used as-is)
├── tools/inspect/WowViewer.Tool.Inspect/
│   ├── Program.cs                          # EXTEND: pm4 build-wmo-fingerprint-db, extract-pm4-fingerprints, match-fingerprints, validate-matches
│   └── Pm4CorrelateModelsSupport.cs        # KEEP for validation ground truth (ADT-based)
└── tests/
    └── WowViewer.Core.PM4.Tests/           # EXTEND: fingerprint extraction + matching tests
```

## Implementation Phases

### Phase 1: Fingerprint Extraction Library (P0 — prerequisite)

Build `Pm4FingerprintExtractor` and `Pm4FingerprintContracts` in Core.PM4.

**1a. Pm4FingerprintContracts.cs** — data models:
- `Pm4FingerprintRecord`: AssetId, AssetPath, AssetKind, Ck24Type, SortedDim0/1/2, Bounds, Center, FootprintHull (PCA-normalized), FootprintArea, SurfaceCount, VertexCount, IndexCount, TypeFlagsProfile, GroupCount, SourceLabel.
- `Pm4FingerprintDatabase`: list of records + metadata (build date, archive root, WMO count).
- `Pm4FingerprintMatchResult`: CK24 group info + ranked candidates with `Pm4CorrelationMetrics` + status.

**1b. Pm4FingerprintExtractor.cs** — extraction logic:
- `ExtractFromTriangles(List<Vector3> vertices, List<ushort> indices, ...)` → `Pm4FingerprintRecord`.
- PCA normalization: center at centroid, compute covariance of XY-projected points, eigenvectors → principal axes, rotate to align. Try both flip candidates for near-symmetric shapes.
- Use `Pm4CorrelationMath.BuildFootprintHull` on PCA-normalized points.
- Compute sorted dimensions from AABB.
- Compute topology signals (surface count, vertex count, index count).

**1c. Unit tests** — synthetic geometry (known box, known L-shape, rotated copy) → verify PCA normalization produces identical hulls regardless of input rotation.

### Phase 2: WMO Fingerprint Database (P1)

Build the WMO fingerprint DB from the staged archive.

**2a. WMO collision geometry loader** — use `WmoRenderDocumentReader` to read each WMO root + embedded groups. Collect MOVT vertices + MOVI indices across all groups. Produce merged root fingerprint + per-group fingerprints.

**2b. CLI command** `pm4 build-wmo-fingerprint-db --archive-root <staged> --output <db.json>`:
- Enumerate WMO roots via archive catalog (fix the 506/1985 enumeration gap — use listfile-based enumeration if archive catalog misses WMOs).
- For each WMO: read collision geometry, extract fingerprint via `Pm4FingerprintExtractor`, add to DB.
- Serialize DB to JSON.

**2c. Validate** — run on staged 3.3.5 client. Verify ≥500 WMO fingerprints. Verify GoldshireInn.wmo fingerprint has sorted dims ~30×32×60. Report build time.

### Phase 3: PM4 Fingerprint Extraction (P1)

Extract fingerprints from all 616 development PM4s.

**3a. CLI command** `pm4 extract-pm4-fingerprints --input <dir> --output <fp.json>`:
- Read each PM4, group MSUR by CK24, collect MSVT/MSVI per group.
- Extract fingerprint via `Pm4FingerprintExtractor` for each CK24 group.
- Serialize to JSON.

**3b. Validate** — run on 616 development PM4s. Verify 1604 fingerprints. Verify multi-tile OID 52202 produces cross-tile hull overlap ≥0.90 (PCA rotation invariance proof).

### Phase 4: Fingerprint Matching (P1)

Match PM4 fingerprints against WMO DB.

**4a. Pm4FingerprintMatcher.cs** — matching logic:
- Load WMO DB + PM4 fingerprints.
- Type-filter: 0x42/0x43/0xC0-0xC3 → WMO candidates; 0x40/0x41 → M2 candidates (or Ineligible).
- Sorted-dimension prefilter: reject candidates with >25% dim mismatch on any axis.
- For survivors: `Pm4CorrelationMath.EvaluateMetrics` on PCA-normalized hulls + bounds.
- `Pm4CorrelationMath.CompareCandidateScores` for ranking.
- Status: Matched (top score ≥0.45, margin >0.03), Ambiguous (margin ≤0.03), Unresolved (<0.45).

**4b. CLI command** `pm4 match-fingerprints --pm4-fingerprints <fp.json> --wmo-db <db.json> --output <matches.json>`:
- Run matcher, serialize results.

**4c. Validate** — run on tile 24_35. Verify GoldshireInn.wmo is top-1 for the ~30×32×60 group with footprint overlap ≥0.80. Run on all 616 tiles. Report match rate.

### Phase 5: Validation Against ADT Ground Truth (P2)

Validate matches on ADT-backed tiles.

**5a. CLI command** `pm4 validate-matches --matches <matches.json> --adt-ground-truth <tile.adt> --archive-root <staged>`:
- Read ADT MODF/MDDF placements.
- Compute ground-truth CK24↔WMO pairs via existing `Pm4CorrelateModelsSupport.Correlate` (ADT-based, used ONLY here).
- Compare fingerprint-DB top-1/top-3 against ground truth.
- Report precision@1, precision@3, coverage, failure categorization.

**5b. Tune** — if precision@1 < 0.40, examine failures, adjust prefilter threshold / PCA flip handling / scoring weights.

### Phase 6: PM4 Generator (P3 — downstream, partially done)

Already implemented: `Pm4Generator.cs`, `pm4 generate-from-wmo`. Kept as-is unless correlation findings require updates. Not blocking Phases 1-5.

## Phases ordered by execution

| Phase | What | Depends on | Outcome |
|-------|------|-----------|---------|
| 1 | Fingerprint extraction library (PCA + hull + contracts) | nothing | `Pm4FingerprintExtractor` + contracts + unit tests |
| 2 | WMO fingerprint DB from staged archive | 1 | `wmo-fingerprint-db.json` with ≥500 entries |
| 3 | PM4 fingerprint extraction from 616 files | 1 | `pm4-fingerprints.json` with 1604 entries |
| 4 | Fingerprint matching (no ADT) | 2, 3 | `matches.json` — CK24→WMO identity table |
| 5 | Validation against ADT ground truth | 4 | precision@1/@3 report + tuning |
| 6 | PM4 generator (downstream) | 5 | already done; revisit if needed |
