# Active Context — wow-viewer

**Branch**: `v0.5.0-dev` | **Last updated**: 2026-06-14

## Direction
WoW viewer. Libraries bridge to Unreal Engine. No Vulkan, no WebGL, no Museums, no BASE repo extraction. Old `wow-engine-modernization-plan-2026-05-14.md` replaced with viewer-first + UE bridge framing.

## Active Specs (priority order)

| Spec | State | Next step |
|------|-------|-----------|
| **056 GPU/LOD modernization** | 0/81 — **top priority** | Phase 0: test project + baselines |
| **033 MdxViewer migration** | 0/49 — critical, too large | Scope down to bite-sized slices |
| **046 PM4 asset matching** | ~26/42 — C# done | Python/Zarr corpus lane |
| **058 PM4 scene graph** | ~18/22 — mostly done | Polish + memory bank update |
| **061 Weak signal restoration** | 15/21 | Verification checklist |
| **062 Weak signal tile patcher** | 11/19 | Phases 3-4 |
| **043 Chunked MDX** | Code landed, tasks stale | Re-open when 0.5.3+ MDX rendering |
| **044 Viewer shell** | 10/13 (3 deferred P2) | US4 cursor-as-model deferred |
| **049 UI consolidation** | Not started | Categorized Tools, sidebar removal |
| **057 Client archive selector** | 0/29 | UI version picker |
| **041 Liquid type fix** | Not started | Fix "lava for everything" bug |
| **055 UE bridge** | North star, out of scope for v0.5.0 | Start after 056 |

## Complete
012, 014, 024, 025, 044 (at P1), 060, 059 (archived)

## Research consumed by 056
030 (WMO pass), 031 (terrain cell), 032 (native parity), 038 (M2 3.0.1 perf), 040 (liquid — feeds 041)

## Archived (2026-06-14)
005, 020, 026, 036 (dead), 059 (done)

## Known Issues
- Viewer click-freeze on dense PM4 tiles (timing shipped, user numbers pending)
- Renderer culling coord bug in ComputeTilePlanarMin/Max
- 14 pre-existing test failures in Core.Tests (stale ChunkedFileReader fixtures)
