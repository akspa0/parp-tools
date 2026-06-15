# Active Context — wow-viewer

**Branch**: `v0.5.0-dev` | **Last updated**: 2026-06-14

## Direction
WoW viewer. Libraries bridge to Unreal Engine. No Vulkan/WebGL/Museums/BASE.

## Active Specs (priority order)

| Spec | State | Next |
|------|-------|------|
| **046 PM4 asset matching** | **Current focus** — C# done, Python unstarted | Build Zarr signal-store lane |
| **058 PM4 scene graph** | ~18/22 — polish | T020/T028/T029 |
| **061 Weak signal detect** | 15/21 | Verification checklist |
| **062 Weak signal tile patcher** | 13/20 | Cross-tile seam blending, perf |
| **041 Liquid type fix** | Not started | Fix "lava for everything" |
| **055 UE bridge** | North star, out of scope for v0.5.0 |

## Complete
012, 014, 024, 025, 044 (P1 boundary), 060, 059 (archived)

## Known Issues
- Viewer click-freeze on dense PM4 (timing shipped, numbers pending)
- Some 0.5.3 Alpha `.wdt.MPQ` fail to parse
- Alpha WDT write fails on placement-heavy tiles (>14999 bytes)
- 14 pre-existing test failures (stale ChunkedFileReader fixtures)
