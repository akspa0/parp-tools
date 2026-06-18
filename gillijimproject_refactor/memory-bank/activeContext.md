# Active Context

## Direction
WoW viewer. Libraries bridge to Unreal Engine. No Vulkan/WebGL/Museums/BASE.

## 2026-06-17 — Surface correlation matcher — PIVOT from hull to per-triangle matching

**Hull footprint matching ABANDONED.** Produced false positives (Ironforge/Darnassis at 0.999 overlap despite NOT being in dev map). User confirmed: "how in the fuck are we still using footprints to figure out what objects are?! use the fucking correlation of surfaces in pm4's to real wmo objects!"

**Surface correlation implemented (commit 21aa0064).** PM4 MSUR surfaces triangulated → per-triangle sorted edge lengths binned to integers (transform-invariant hash) → histogram intersection matching against WMO MOVI/MOVT collision triangles.

**Results (1604 PM4 vs 2790 WMO surface fingerprints):**
- 217 matched, 956 ambiguous, 158 unresolved, 273 ineligible
- P@1=1.3%, P@3=10.3% (2.3x improvement over hull P@3=4.5%)
- NO false positives — Ironforge/Darnassis eliminated
- 12 correct top-1: GoldshireInn (0.86 PM4 coverage), classicalelfruins, arathistonebridge, orchut
- GoldshireInn matches tiles 0_2/1_1 at 0.86 coverage but ADT doesn't list it — likely ADT gap, not matcher error

**Code**: `Pm4SurfaceCorrelationExtractor` (triangulate + histogram), `Pm4SurfaceCorrelationMatcher` (histogram intersection + F1), CLI: `build-wmo-surface-db`, `extract-pm4-surfaces`, `match-surfaces`.

**Remaining gaps**: WMO DB coverage (503/1985), dev map ADT unreliability, edge bin size (1.0 may be too coarse), no triangle area in histogram key.

**Code**: `Pm4FingerprintExtractor` (PCA + hull), `Pm4FingerprintMatcher` (prefilter + EvaluateMetrics + flip), `Pm4FingerprintBuildSupport` (WMO DB builder), CLI: `build-wmo-fingerprint-db`, `extract-pm4-fingerprints`, `match-fingerprints`. 15 unit tests pass.

**Next**: Phase 5 (validate against ADT ground truth), reduce ambiguous count (add surface/vertex count signals, per-group matching, tune thresholds). WMO enumeration still 506/1985 (listfile needed).

## What's Done
012, 014, 024, 025, 033, 037, 041, 043, 044 (P1), 048, 054, 058, 059, 060, 061, 062

## What's Not Started
001, 029, 030/031/032 (research), 038/040 (research), 042, 045, 049, 053, 055, 056, 057

## Biggest Unproven Gap (046)
1. WMO DB coverage: 503/1985 WMOs — archive enumeration misses 75%. Need listfile for full coverage. This is the #1 driver of low ADT validation precision (1.8% P@1).
2. Dev map ADT unreliability: dev map mixes WMOs from all zones, ADT placements are sparse. Need validation on a real game map (Elwynn, Darnassus, etc.).
3. Remaining ambiguity: 502/1604 — mostly Stormwind vs StormwindHarbor (genuinely identical architecture). May need CK24 ObjectId mapping or tile context to resolve.

## Staged Clients
Only `output/tmp/wowarchive-clients/` paths are valid.

## Known Issues
- Viewer click-freeze on dense PM4 (timing shipped, numbers pending)
- Some 0.5.3 Alpha `.wdt.MPQ` fail to parse
- WMO archive enumeration misses ~75% of WMOs (506/1985) — need listfile-based enumeration (spec 065 Task 2.2)
- Alpha WDT write fails on placement-heavy tiles (>14999 bytes)
- 14 pre-existing test failures (stale ChunkedFileReader fixtures)