# Active Context

## Direction
WoW viewer. Libraries bridge to Unreal Engine. No Vulkan/WebGL/Museums/BASE.

## 2026-06-17 — Fingerprint-database PM4→WMO matching implemented + validated

**Phases 1-4 of spec 065 DONE (commits fe7a304e, e8d4f1d5, 4db79689, c7239549).**

**Pipeline**: WMO collision geometry (MOVT/MOVI) → PCA-normalized convex hull fingerprint → DB. PM4 CK24 groups (MSVT/MSVI/MSUR) → same fingerprint. Match via `Pm4CorrelationMath.EvaluateMetrics` with 4-flip axis enumeration. No ADT for matching.

**Real-data results** (1604 PM4 vs 2790 WMO fingerprints, minScore=0.30, v3 tuning):
- 751 matched, 502 ambiguous, 78 unresolved, 273 ineligible (M2/unknown)
- Progression: v1 50 matched → v2 392 (surface/vertex+dedup) → v3 751 (index count + tighter window)
- Top matches: Ironforge 0.999 overlap, Darnassis 1.000, Stormwind 0.975, BlackTemple 0.966
- Remaining ambiguity: Stormwind vs StormwindHarbor (136 cases, margin=0.000) — genuinely identical architecture in two WMO files. Correctly flagged.

**Code**: `Pm4FingerprintExtractor` (PCA + hull), `Pm4FingerprintMatcher` (prefilter + EvaluateMetrics + flip), `Pm4FingerprintBuildSupport` (WMO DB builder), CLI: `build-wmo-fingerprint-db`, `extract-pm4-fingerprints`, `match-fingerprints`. 15 unit tests pass.

**Next**: Phase 5 (validate against ADT ground truth), reduce ambiguous count (add surface/vertex count signals, per-group matching, tune thresholds). WMO enumeration still 506/1985 (listfile needed).

## What's Done
012, 014, 024, 025, 033, 037, 041, 043, 044 (P1), 048, 054, 058, 059, 060, 061, 062

## What's Not Started
001, 029, 030/031/032 (research), 038/040 (research), 042, 045, 049, 053, 055, 056, 057

## Biggest Unproven Gap (046)
Full WMO enumeration (1985 WMOs instead of 506) to improve identity coverage. Multi-tile OID tracking to reconstruct full model bounds from per-tile fragments. M2 type (0x40/0x41) collision vertex reading.

## Staged Clients
Only `output/tmp/wowarchive-clients/` paths are valid.

## Known Issues
- Viewer click-freeze on dense PM4 (timing shipped, numbers pending)
- Some 0.5.3 Alpha `.wdt.MPQ` fail to parse
- WMO archive enumeration misses ~75% of WMOs (506/1985) — need listfile-based enumeration (spec 065 Task 2.2)
- Alpha WDT write fails on placement-heavy tiles (>14999 bytes)
- 14 pre-existing test failures (stale ChunkedFileReader fixtures)