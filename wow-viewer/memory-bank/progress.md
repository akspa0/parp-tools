# Progress — wow-viewer

## 2026-06-17 — Surface correlation matcher implemented (commit 21aa0064)

### What landed
- Pivoted from hull footprint matching (false positives: Ironforge/Darnassis at 0.999 overlap, NOT on dev map) to per-triangle edge-length histogram correlation.
- `Pm4SurfaceCorrelationExtractor`: triangulates PM4 MSUR fans + WMO MOVI independent tris, builds edge-length histograms (transform-invariant, binned to integers).
- `Pm4SurfaceCorrelationMatcher`: histogram intersection → pm4Coverage, wmoCoverage, symmetric F1 score.
- CLI: `build-wmo-surface-db` (2790 fingerprints, 13M triangles), `extract-pm4-surfaces` (1604 fingerprints, 604K triangles), `match-surfaces`, `validate-matches`.
- WMO surface DB: 503 roots, 11MB JSON. PM4 surfaces: 1604 groups, 3MB JSON.

### Validation — MATCHER STILL PRODUCES FALSE POSITIVES
- P@1=1.3%, P@3=10.3% (vs hull P@1=1.8%, P@3=4.5%). P@3 improved 2.3x.
- Surface correlation eliminated worst hull false positives (Ironforge/Darnassis) BUT produces its own false positives.
- GoldshireInn matched to tile 0_2 at 0.86 PM4 coverage. User confirmed: NO GoldshireInn exists in PM4 data on tile 0_2. It's a histogram collision — edge-length bins match across different geometry with similarly-sized triangles.
- DO NOT claim matches are correct without verification. Both hull and surface approaches produced false positives.

### What needs doing next (fresh chat)
- Fix WMO enumeration (503/1985 — archive catalog probe bug or need listfile)
- Tune edge bin size, add triangle area to histogram key
- Reduce 956 ambiguous
- Full pipeline: surface match → identify WMO → extract placement transform from PM4 → write MODF → regenerate ADT for tiles without one
- ✅ Update spec 065 to reflect surface correlation as primary approach (hull is abandoned)

## 2026-06-17 — Spec 065 rewritten for surface correlation
- Rewrote `specs/065-pm4-correlation-to-world-assets/spec.md`, `plan.md`, `tasks.md`.
- Documented abandoned hull/footprint approach; documented surface triangle edge-length histogram as primary.
- Phases 1–5 marked DONE; Phase 6 (add triangle area to histogram key) is the active next step.

## 2026-06-18 — Phase 6: triangle area added to histogram key
- Added `AreaBin` to `TriangleKey`; added `--area-bin-size` CLI option to `build-wmo-surface-db` and `extract-pm4-surfaces`.
- Rebuilt WMO + PM4 surface DBs; tested `area-bin-size=1.0` and `10.0`.
- Results (`area-bin-size=1.0`): 11 matched, 199 ambiguous, 1121 unresolved; P@1=0.0%, P@3=25.3%; GoldshireInn tile 0_2 false positive eliminated.
- Results (`area-bin-size=10.0`): 43 matched, 371 ambiguous, 917 unresolved; P@1=0.0%, P@3=11.3%; GoldshireInn tile 0_2 false positive returns.
- Fixed `pm4 validate-matches` to read `Pm4SurfaceMatchOutput` (surface match JSON format) instead of old fingerprint-match format.
- Conclusion: area alone improves precision but is too strict for recall; Phase 7 (normal + height) needed.

## 2026-06-18 — Phase 7: placement-invariant descriptors + generator validation
- Redesigned Phase 7 away from absolute MSUR.Normal/MSUR.Height (coordinate mismatch WMO-local vs PM4-world) to placement-invariant descriptors:
  - Normal alignment: `dot(triNormal, groupDominantNormal)`.
  - Planar offset: `dot(triCentroid - groupCentroid, groupDominantNormal)`.
- Added optional `--normal-alignment-bin-size` and `--planar-offset-bin-size` to `build-wmo-surface-db` and `extract-pm4-surfaces`; defaults are 0 to preserve area-only baseline.
- Validation with bins enabled (`0.1` and `1.0`): 0 matched, 0 ambiguous, P@3=10.1% (worse than area-only 25.3%). Descriptors make the key too specific and do not recover recall.
- Added `pm4 validate-generator-geometry --pm4 <file> --adt <obj0.adt> --archive-root <staged>` to directly validate `Pm4Generator.cs`.
- Generator validation on `development_16_37`: mean symmetric score 0.004, 0/4 placements matched. Generated PM4 surfaces do not reproduce real PM4 surfaces.
- Conclusion: surface histogram matching is capped; next leverage is WMO enumeration (Phase 8) and generator rework.

## 2026-06-18 — Generator fix pass: fingerprint correlation works

### What landed
- `Pm4Generator.cs` now reads WMO collision faces via `MOPY` flags (`0x08` collision, `0x20` render collidable, exclude `0x04` no-collide) instead of all faces.
- Fixed `MSVI` first-index bug: generator wrote byte offsets; now writes raw uint index.
- Fixed `WorldToPm4Raw`: removed X/Y swap; now `MapOrigin - X`, `MapOrigin - Y`, `Z`.
- Pivoted generator validation to **collision fingerprints** instead of exact surface reproduction.
- `Pm4GeneratorValidationSupport.cs` now uses `Pm4SurfaceCorrelationExtractor` to build group-relative edge+area histograms from generated WMO collision triangles and real PM4 CK24 groups.
- Determined `development_29_18` is the correct WMO validation tile (48 CK24 WMO groups); `development_16_37` has only M2 groups.

### Validation
- `pm4 validate-generator-geometry` on `development_29_18` with area-only bins (edge=1, area=1):
  - Mean symmetric score: **0.462**
  - Matched groups (score >= 0.50): **36/48**
  - All farm placements correlate to real group `0x43C510`.
- This confirms we do not need perfect fake PM4 polygons; we only need comparable collision-shape fingerprints.

## 2026-06-17 — Hull fingerprint matcher (ABANDONED — false positives)
- Built PCA-normalized convex hull fingerprint matcher. 751 matched but mostly false positives.
- Ironforge/Darnassis at 0.999 overlap despite NOT being on dev map. Hull throws away surface structure.
- User: "how in the fuck are we still using footprints to figure out what objects are?!"
- Hull code kept for reference but surface correlation is primary.

## 2026-06-17 — PM4 simplification algorithm reverse-engineered and implemented
- PM4 surfaces are variable-size convex polygons (IndexCount 3-12). 43% quads, 43% triangles.
- Simplification: plane clustering + 2D convex hull. Generator produces valid PM4 (954 vs 896 real surfaces).
- WMO cache: 277 roots from ADT + 184 Ulduar + 114 Titan.
- PD4 format: separate per-WMO collision (version 48, quads, YXZ, 24-byte MSLK).
