# Active Context

## Direction Change (2026-06-14)
- Replaced old engine-program framing (Vulkan-first, museum-explorer, BASE repo extraction). This is a WoW viewer. Libraries serve the viewer and bridge to Unreal Engine.
- OpenGL/Silk.NET is the viewer path. UE bridge (spec 055) consumes Core.* libs from C#/C++ interop.
- Staged clients under `output/tmp/wowarchive-clients/` are the only trusted client roots.

## Active Specs

| Spec | State | Next |
|------|-------|------|
| 056 GPU/LOD modernization | 0/81 tasks, **top priority** | Phase 0: test foundation + baselines |
| 033 MdxViewer migration | 0/49 tasks, critical | Needs scoping — 49 tasks is too large |
| 046 PM4 asset matching | ~26/42 tasks, C# side done | Python/Zarr corpus lane unstarted |
| 058 PM4 scene graph | ~18/22 tasks, mostly done | T020/T028/T029 polish |
| 061 Weak signal restoration | 15/21 tasks | Verification checklist unchecked |
| 062 Weak signal tile patcher | 11/19 tasks | Phases 3-4 unstarted |
| 043 Chunked MDX | Code landed, tasks.md stale | Re-open when 0.5.3+ MDX rendering focus |
| 044 Viewer shell | 10/13 tasks, 3 deferred P2 | US4 cursor-as-model deferred |
| 049 UI consolidation | Defined, not yet started | Categorized Tools menu, sidebar removal |
| 057 Client archive selector | 0/29 tasks | Useful for multi-client UX, not blocking |
| 055 UE bridge | North star, out of scope for v0.5.0 | Start after 056 proves shared renderer |
| 041 Liquid type fix | Not started, needed fix | Blocked on higher priorities |

## Complete specs (100%)
012 (validation capture), 014 (MCAL parity), 024 (V18 canvas), 025 (roof mask), 044 (shell done at P1 boundary), 060 (UI cleanup), 059 (Cata M2 — archived)

## Research consumed by 056
030 (WMO render pass), 031 (terrain cell), 032 (native parity), 038 (M2 3.0.1 perf), 040 (liquid type — feeds 041)

## Biggest Gaps
1. **056 is central** — every renderer spec converges here. Phase 0 unblocks everything else.
2. **043 tasks.md is stale** — code landed but task list never updated after 1.12.1 MD20 discovery (spec 048).
3. **046 Python/Zarr lane doesn't exist** — PM4 asset matching has no Python signal-store writer.
4. **Memory bank split** — `gillijimproject_refactor/memory-bank/` has newer data (2026-06-12) than `wow-viewer/memory-bank/` (2026-06-10/11) despite being "decommissioned."

## Known Issues
- Viewer click-freeze on dense PM4 tiles (timing instrumentation shipped in 30461a1d, numbers pending)
- Renderer culling coordinate bug in `ComputeTilePlanarMin/Max`
- PM4 overlay cache version v7→v8 invalidates old blobs (normal, transient)
- 14 pre-existing test failures in `WowViewer.Core.Tests` (stale ChunkedFileReader fixtures)
