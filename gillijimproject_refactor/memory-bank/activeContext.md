# Active Context

## Direction (2026-06-14)
WoW viewer. Libraries serve the viewer and bridge to Unreal Engine. No Vulkan, no WebGL, no Museums, no BASE repo extraction.

## Spec Status After Consolidation

| State | Specs |
|-------|-------|
| **Top priority** | 056 (GPU/LOD modernization, 0/81) |
| **Active** | 033 (MdxViewer migration), 046 (PM4 matching, C# done/Python unstarted), 058 (PM4 scene graph, ~18/22), 061 (weak signal detect, 15/21), 062 (tile patcher, 13/20) |
| **Complete** | 012, 014, 024, 025, 044 (P1), 060 |
| **Research consumed by 056** | 030 (WMO pass), 031 (terrain cell), 032 (native parity), 038 (M2 3.0.1 perf) |
| **Not started** | 041 (liquid fix), 045 (scene graph), 049 (UI consolidation), 053 (anim farm), 054 (PM4 cache), 055 (UE bridge), 057 (archive selector) |
| **Archived 2026-06-14** | 005, 020, 026, 036 (dead), 059 (Cata M2 done) |

## Biggest Gaps
1. **056 has 0/81 tasks done** — convergence spec for all renderer work. Phase 0 (test foundation + baselines) is the natural start.
2. **046 Python/Zarr lane** doesn't exist at all.
3. **033 MdxViewer migration** 0/49 tasks, critical for repo independence.

## Known Issues
- Viewer click-freeze on dense PM4 tiles (timing shipped in 30461a1d, numbers pending)
- Some 0.5.3 Alpha `.wdt.MPQ` files fail to parse (Kalimdor, Kalidar, etc.)
- Alpha WDT write fails on tiles with >14999 bytes placement data
- 14 pre-existing test failures in Core.Tests (stale ChunkedFileReader fixtures)
