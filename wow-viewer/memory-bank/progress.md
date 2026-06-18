# Progress — wow-viewer

## 2026-06-17 — Surface correlation matcher implemented (commit 21aa0064)

### What landed
- Pivoted from hull footprint matching (false positives: Ironforge/Darnassis at 0.999 overlap, NOT on dev map) to per-triangle edge-length histogram correlation.
- `Pm4SurfaceCorrelationExtractor`: triangulates PM4 MSUR fans + WMO MOVI independent tris, builds edge-length histograms (transform-invariant, binned to integers).
- `Pm4SurfaceCorrelationMatcher`: histogram intersection → pm4Coverage, wmoCoverage, symmetric F1 score.
- CLI: `build-wmo-surface-db` (2790 fingerprints, 13M triangles), `extract-pm4-surfaces` (1604 fingerprints, 604K triangles), `match-surfaces`, `validate-matches`.
- WMO surface DB: 503 roots, 11MB JSON. PM4 surfaces: 1604 groups, 3MB JSON.

### Validation
- P@1=1.3%, P@3=10.3% (vs hull P@1=1.8%, P@3=4.5%). P@3 improved 2.3x.
- NO false positives — Ironforge/Darnassis eliminated.
- 12 correct top-1: GoldshireInn (0.86 coverage tiles 0_2/1_1), classicalelfruins, arathistonebridge, orchut.
- GoldshireInn on tiles 0_2/1_2: PM4 says it's there at 0.86 coverage, ADT doesn't list it. PM4 is right — ADT is incomplete (THIS IS THE POINT OF THE WORK).

### What needs doing next (fresh chat)
- Fix WMO enumeration (503/1985 — archive catalog probe bug or need listfile)
- Tune edge bin size, add triangle area to histogram key
- Reduce 956 ambiguous
- Full pipeline: surface match → identify WMO → extract placement transform from PM4 → write MODF → regenerate ADT for tiles without one
- Update spec 065 to reflect surface correlation as primary approach (hull is abandoned)

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
