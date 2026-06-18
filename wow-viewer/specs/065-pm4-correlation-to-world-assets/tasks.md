# Tasks: PM4 Surface Correlation to World Assets & Generator

**Plan**: [plan.md](plan.md) | **Spec**: [spec.md](spec.md)

## Phase 1: Surface Extraction Library — DONE

- [x] T001: Create `Pm4SurfaceCorrelationContracts.cs` — `SurfaceCorrelationFingerprint`, `SurfaceMatchCandidate`, `SurfaceMatchResult`.
- [x] T002: Create `Pm4SurfaceCorrelationExtractor.cs` — triangulate PM4 MSUR fans + WMO MOVI tris, build edge-length histograms.
- [x] T003: Create `Pm4SurfaceCorrelationMatcher.cs` — histogram intersection, PM4/WMO coverage, symmetric F1, status classification.
- [x] T004: Add unit tests for triangulation, transform invariance, degenerate skip.
- [x] T005: Build + test `dotnet test wow-viewer/WowViewer.slnx --filter SurfaceCorrelation`.

## Phase 2: WMO Surface Database — DONE

- [x] T006: Add `BuildWmoSurfaceDatabaseAsync` in `Pm4SurfaceBuildSupport.cs` — root + per-group fingerprints.
- [x] T007: Add CLI `pm4 build-wmo-surface-db` with `--bin-size` and `--area-bin-size`.
- [x] T008: Validate on staged 3.3.5 client: 503 roots + 2790 total fingerprints.

## Phase 3: PM4 Surface Extraction — DONE

- [x] T009: Add `BuildPm4SurfaceDatabaseAsync` in `Pm4SurfaceBuildSupport.cs` — group MSUR by CK24.
- [x] T010: Add CLI `pm4 extract-pm4-surfaces` with `--bin-size` and `--area-bin-size`.
- [x] T011: Validate on 616 dev PM4s: 1604 CK24 group fingerprints.

## Phase 4: Surface Matching — DONE

- [x] T012: Ensure matcher type-filters, ranks by F1, applies ambiguity window.
- [x] T013: Add CLI `pm4 match-surfaces`.
- [x] T014: Validate full corpus: 217 matched, 956 ambiguous, 158 unresolved, 273 ineligible; no hull false positives.

## Phase 5: Validation Against ADT Ground Truth — DONE

- [x] T015: Add CLI `pm4 validate-matches` using legacy ADT support for ground truth only.
- [x] T016: Run validation: P@1=1.3%, P@3=10.3%; document failure categories.

## Phase 6: Add Triangle Area to Histogram Key — NEXT

- [x] T017: Compute triangle area in `Pm4SurfaceCorrelationExtractor` and add area bin to histogram key; add `--area-bin-size` to `build-wmo-surface-db` and `extract-pm4-surfaces`.
- [x] T018: Update `Pm4SurfaceCorrelationMatcher` to use area-aware key.
- [x] T019: Rebuild WMO + PM4 DBs with `--area-bin-size`, rematch, re-validate.
- [x] T020: Verify GoldshireInn tile 0_2 false positive is eliminated/downranked; ambiguous count decreases; P@3 does not regress.
  - area-bin-size=1.0: GoldshireInn tile 0_2 false positive eliminated (now Unresolved); ambiguous 199 (down from 956); P@3=25.3% (up from 10.3%); P@1=0.0% (down from 1.3%).
  - area-bin-size=10.0: GoldshireInn tile 0_2 false positive returns; ambiguous 371; P@3=11.3%; P@1=0.0%.
  - Conclusion: fine area bin (1.0) eliminates the known false positive and boosts P@3, but is too strict for P@1. Phase 7 (normal + height) needed to recover recall.

## Phase 7: Placement-Invariant Geometry + Generator Validation

- [x] T021: Replaced absolute normal/height with placement-invariant descriptors: triangle normal alignment to the group's area-weighted dominant normal, and planar offset from the group's area-weighted centroid. These are rigid-invariant and comparable between WMO-local and PM4-world geometry.
- [x] T022: Extended `TriangleFeature`/`TriangleKey` to optionally include normal-alignment and planar-offset bins; defaults are 0 so the baseline area-aware key is unchanged.
- [x] T023: Rebuilt, rematched, validated with placement-invariant bins enabled (`--normal-alignment-bin-size 0.1 --planar-offset-bin-size 1.0`). Result: 0 matched, 0 ambiguous, P@3=10.1% (vs. 25.3% for area-only). The extra descriptors make the key too specific and do not recover P@1.
- [x] T024: Added `pm4 validate-generator-geometry --pm4 <file> --adt <obj0.adt> --archive-root <staged>` to directly validate `Pm4Generator.cs`. It generates PM4 from each ADT WMO placement and compares generated surface histograms to real PM4 CK24 groups in the same coordinate space.
- [x] T025: Ran generator validation on tile 16_37 (development): mean symmetric score 0.004, no groups matched. Generated PM4 surfaces do not reproduce real PM4 surfaces. This confirms `Pm4Generator.cs` needs significant rework (collision mesh source, simplification, or coordinate transform).
- [x] T026: Switched `Pm4Generator` source geometry to WMO collision faces (`MOPY` flags `0x08`/`0x20`, exclude `0x04`).
- [x] T027: Fixed `MSVI` first-index bug (byte offset → raw uint index) and `WorldToPm4Raw` X/Y swap.
- [x] T028: Added vertex welding + coplanar boundary-polygon merging to match real PM4 surface granularity.
- [x] T029: Validated on `development_29_18` (correct WMO tile with 48 CK24 groups): mean score 0.051, 0/48 matched; generated farm bounds overlap real group `0x43C689`.
- [x] T030: Pivoted generator validation from exact surface reproduction to **collision fingerprints** using `Pm4SurfaceCorrelationExtractor`.
- [x] T031: Re-validated `development_29_18` with area-only (edge+area) fingerprints: mean score 0.462, 36/48 groups matched.
- [ ] T032: Confirm fingerprint approach generalizes across other WMO types / tiles.
- [ ] T033: Decide whether to keep or remove unused boundary-polygon merge code in `Pm4Generator.cs`.
- [ ] T034: Integrate precomputed WMO collision fingerprints into the main PM4→WMO matcher so it works on ADT-less tiles.

## Phase 8: Fix WMO Enumeration via Listfile

- [ ] T024: Add `--listfile <path>` option to `pm4 build-wmo-surface-db`.
- [ ] T025: Merge archive catalog + listfile enumeration, deduplicate root WMOs.
- [ ] T026: Validate ≥1900 WMO root fingerprints; re-run matching and ensure P@3 does not regress.

## Phase 9: Placement Recovery / ADT Regeneration

- [ ] T027: Compute rigid transform from matched WMO collision to PM4 surfaces.
- [ ] T028: Generate MODF/MDDF placement candidates from trusted matches.
- [ ] T029: Write synthetic ADT for PM4-only tiles; validate on ADT-backed tiles by comparing recovered placements to original ADT.
- [ ] T030: Revisit `Pm4Generator.cs` if correlation reveals a WMO→PM4 vertex transform.

---

## Legacy Commands (Kept, Not Primary)

- `pm4 identify-models` — sorted-dimension-only matching. KEPT for comparison.
- `pm4 correlate-models` / `sweep-correlate` — ADT-based correlation. KEPT for validation ground truth only.
- `pm4 match-assets` — ADT-dependent scorer. KEPT but deprecated.
- `pm4 build-wmo-fingerprint-db`, `extract-pm4-fingerprints`, `match-fingerprints` — hull-based, superseded by surface correlation.
