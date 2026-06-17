# Progress

## 2026-06-17 — Fingerprint-database PM4→WMO matching implemented (Phases 1-4)

### What landed
- Phase 1: `Pm4FingerprintExtractor` (PCA normalization + convex hull), `Pm4FingerprintContracts` (serializable records). 9 unit tests.
- Phase 2: `Pm4FingerprintBuildSupport` (WMO collision → fingerprint DB). CLI `build-wmo-fingerprint-db`. 503 root + 2287 group = 2790 fingerprints from 506 staged 3.3.5 WMOs.
- Phase 3: CLI `extract-pm4-fingerprints`. 1604 CK24 group fingerprints from 616 dev PM4s. Type distribution matches: 0x42=584, 0x43=466, 0x41=161, 0xC1=100.
- Phase 4: `Pm4FingerprintMatcher` (sorted-dim prefilter + EvaluateMetrics + 4-flip PCA). CLI `match-fingerprints`. 6 unit tests. Real-data: 50 matched, 1203 ambiguous, 78 unresolved, 273 ineligible. Top: Ironforge 0.94/0.999, Stormwind Harbor 0.92/0.98.

### Commits
- fe7a304e: spec 065 pivot
- e8d4f1d5: Phase 1 fingerprint extraction library
- 4db79689: Phase 2-3 WMO DB + PM4 extraction CLI
- c7239549: Phase 4 fingerprint matcher

### Next
- Phase 5: validate against ADT ground truth (precision@1/@3)
- Reduce ambiguous count: add surface/vertex count ratio, TypeFlags profile, per-group matching
- WMO enumeration: 506/1985 — need listfile for full coverage

## 2026-06-17 — Spec 065 revised: fingerprint-database approach (route change)

### What changed
- ADT-based PM4→WMO matching ABANDONED. `correlate-models`/`sweep-correlate` need ADT anchors (222 PM4-only tiles have none). `identify-models` is bounding-box-only (too coarse). `match-assets` has ADT-dependent `sameTileBonus` (dead on PM4-only tiles).
- New approach: fingerprint database from WMO collision geometry (MOVT/MOVI) via `Pm4CorrelationMath` convex-hull footprint + PCA normalization. Match PM4 CK24 fingerprints against WMO DB. No ADT for matching.
- Spec 065 rewritten: spec.md, plan.md, tasks.md all revised. 6 phases: fingerprint extraction library → WMO DB → PM4 fingerprints → matching → ADT validation → generator (downstream).
- Legacy commands kept for validation ground truth, not as primary matchers.

### Key insight
- The right correlation approach (`Pm4CorrelationMath`: convex hull footprint overlap, symmetric footprint distance, planar gap) already exists but was never used to build a fingerprint database. Instead, we relied on ADT placements for position info. The fix: use the correlation math to extract rotation-invariant fingerprints from WMO collision geometry directly, store to DB, match PM4 against it.

## 2026-06-16 — PM4 matcher is broken; spec 065 written

### Root cause diagnosis
- `pm4 match-report` produces "Candidate Count: 0" for ALL PM4 objects on ALL tiles because it compares PM4 raw-ADT coordinates against WoW world coordinates without any conversion. `ConvertPm4VertexToWorld` produces `(tileY*533+mappedU, tileX*533+mappedV, localUp)` but `AdtPlacementReader` produces `(17066-rawY, 17066-rawX, rawZ)`. Gap is ~24000 units — all spatial matching is dead.
- `pm4 match-assets` (the shape scorer) is architecturally correct but produces sub-threshold scores (~0.42 vs 0.45 minimum) because PM4 segments are individual surfaces, not whole-object groupings. On tile 22_18, 40543 PM4 objects become only 92 segments, each scored against 1985 WMO models.
- Tile 22_18 is NOT the oil platform (2 WMO placements in ADT). It's a "snowball fort" — multiple WMOs stacked together including Ulduar titan structures. CK24 0x0042084C type 66 is the dominant object.
- The `pm4 match-report` command is architecturally wrong for data recovery — it's placement-centered (needs existing ADT placements as anchors) instead of PM4-object-centered (match PM4 shapes against model shapes from archive).

### Fix attempts
- Attempted coordinate conversion in `Pm4MatchSupport` (RawAdtToWowWorld) — reverted. The match-report approach itself is wrong, not just the coordinates.

### Spec written
- `wow-viewer/specs/065-pm4-object-identity/spec.md` — 5-phase plan:
  - P1: Coordinate fix + known-tile correlation on 24_35
  - P1: CK24 identity table (CK24 → model path mapping)
  - P1: Synthetic PM4 signal generation from WMO collision
  - P2: CK24-grouped segment scoring (merge surfaces per CK24)
  - P2: Unknown-tile resolution using identity table + shape fallback

### Key insight
- The right approach is: generate what PM4 data WOULD look like for a given WMO/M2 model, then compare that synthetic data against real PM4 data. Not position-matching against broken ADT placements.

## 2026-06-16 — PM4 ADT writing reverted, replaced with match-report
- Deleted Pm4AdtWriter, Pm4BinaryAdtPatcher — ADT patching was corrupting output
- Replaced `pm4 write-adt` with `pm4 match-report` (human-readable markdown)
- LkAdtWriter untouched — not part of PM4 matcher work
- Checkpoint commit: 5133bfe3

## 2026-06-14 — Spec consolidation + tool fixes
- Replaced engine-program plan with viewer-first + UE bridge (509→35 lines)
- Archived 005, 020, 026, 033, 036, 059 (done/dead)
- Fixed stale status: 025→Complete, 060→Complete, 043→stale noted
- Marked research specs 030/031/032/038/040 consumed by 056
- Fixed 044 T006: removed dead MK Dataset GUI
- Added `--client-root --map` to `terrain-weak-signal-patch` for in-memory MPQ
- Ran weak signal on 0.5.3/0.5.5/1.12.1/3.0.1 maps — proven on real data
- Current focus: **046 PM4 asset matching** (C# done, Python lane needed)

## 2026-06-15 — Session polluted by hallucinations and wrong assumptions
- Implemented `pm4 dump-collision` command and WMO validation (works, 40 OIDs)
- Spent too long on tangents, wrong assumptions about M2/MD20, and coordinate systems
- Key deliverables: collision dumper tool, serialization fixes, Python scorer validation
- Memory bank updated. Needs fresh session with clear direction.
