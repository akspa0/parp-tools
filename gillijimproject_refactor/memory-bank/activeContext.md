# Active Context

## Direction (2026-06-14)
WoW viewer. Libraries serve the viewer and bridge to Unreal Engine. No Vulkan, no WebGL, no Museums, no BASE repo extraction.

## Spec Status After Consolidation

| State | Specs |
|-------|-------|
| **Current focus** | 046 (PM4 asset matching — C# done, Python/Zarr lane unstarted) |
| **Active** | 058 (PM4 scene graph, ~18/22), 061 (weak signal detect, 15/21), 062 (tile patcher, 13/20) |
| **Complete** | 012, 014, 024, 025, 033, 044 (P1), 060, 059 |
| **Research consumed by 056** | 030 (WMO pass), 031 (terrain cell), 032 (native parity), 038 (M2 3.0.1 perf) |
| **Not started** | 041 (liquid fix), 045 (scene graph), 049 (UI consolidation), 053 (anim farm), 054 (PM4 cache), 055 (UE bridge), 056 (GPU/LOD), 057 (archive selector) |

## Biggest Gap
046 Python/Zarr signal-store lane does not exist at all. The C# matching library is complete (Pm4ObjectSegmentBuilder, Pm4AssetMatchScorer, Pm4ReplacementPlacementSynthesizer, etc.) but there's no Python tooling to build Zarr signal corpora from the C# export or to train/evaluate matchers.

## Known Issues
- Viewer click-freeze on dense PM4 tiles (timing shipped in 30461a1d, numbers pending)
- Some 0.5.3 Alpha `.wdt.MPQ` files fail to parse (Kalimdor, Kalidar, etc.)
- Alpha WDT write fails on tiles with >14999 bytes placement data
- 14 pre-existing test failures in Core.Tests (stale ChunkedFileReader fixtures)
