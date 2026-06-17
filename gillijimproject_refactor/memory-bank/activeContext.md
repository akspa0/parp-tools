# Active Context

## Direction
WoW viewer. Libraries bridge to Unreal Engine. No Vulkan/WebGL/Museums/BASE.

## 2026-06-17 — Spec 065 revised: ADT-based correlation ABANDONED, fingerprint-database approach adopted

**Route change.** ADT-based PM4→WMO matching is wrong: 222 PM4-only tiles have no ADT, `correlate-models` produces 0 correlations on those tiles, `identify-models` is bounding-box-only (too coarse — dozens of WMOs share ~33×35×53), `match-assets` has ADT-dependent `sameTileBonus` that's dead on PM4-only tiles.

**New approach (spec 065 revised)**: Build a fingerprint database from WMO collision geometry (MOVT/MOVI) using `Pm4CorrelationMath`'s convex-hull footprint extraction with PCA normalization. Extract same fingerprints from PM4 CK24 groups. Match via `EvaluateMetrics` + `CompareCandidateScores`. No ADT for matching — ADT used only for validation ground truth.

**Prior work retained as reference**:
- `pm4 identify-models` (sorted-dimension-only) — KEPT for comparison, not primary matcher. 545 matches at ≥0.95 from 506 WMOs.
- `pm4 fingerprint-scan` — 1604 CK24 groups extracted. 611/616 PM4s use world-space coords. 272 OIDs span 2+ tiles.
- CK24 type pairs: 0x40/0x41=M2, 0x42/0x43=WMO, 0xC0-0xC3=WMO nav. Ck24ObjectId is global across tiles.
- 506/1985 WMOs scanned — archive enumeration missed ~75%. Need listfile-based enumeration (Task 2.2).

**Spec 065 phases**: P1 fingerprint extraction library (PCA + hull + contracts), P2 WMO fingerprint DB, P3 PM4 fingerprint extraction, P4 matching (no ADT), P5 validation against ADT ground truth, P6 generator (downstream, existing).

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