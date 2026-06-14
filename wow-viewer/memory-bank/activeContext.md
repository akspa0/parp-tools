# Active Context — wow-viewer

**Branch**: `v0.5.0-dev` | **Date**: 2026-06-14

## Direction Change

The old `wow-engine-modernization-plan-2026-05-14.md` (engine program, Vulkan-first, museum-explorer, repo extraction) has been replaced. **This is a WoW viewer.** The libraries we build for the viewer serve as the tooling that bridges to Unreal Engine. No Vulkan, no WebGL, no Museums, no BASE repo extraction.

OpenGL/Silk.NET is the viewer rendering path. The UE bridge (spec 055) consumes `Core.*` libraries from C#/C++ interop.

## Current Focus

**Spec 056 — ViewerApp Refactor + GPU Acceleration + LOD Modernization.** Target: shrink `ViewerApp.cs` (621k bytes) by promoting viewer-app `Rendering/*` (28+ files) into a real shared `WowViewer.Core.Renderer` library, modernize OpenGL via Silk.NET (retained-mode VBO/IBO/UBO + instanced rendering + frustum culling), and add the full LOD matrix (terrain mesh LOD, object LOD, water LOD, light LOD, WDL far horizon, BLP mipmaps). 9 phases (0-8), max 10 tasks each. Supersedes `specs/036-renderer-improvements`. Unreal Engine bridge is spec 055 — separate lane.

## Active Specs (by priority)

| Spec | Tasks | Status |
|------|-------|--------|
| 046 PM4 asset matching | 22/39 | WMO group matching phase done; signature matcher unstarted |
| 051 MSCN/MSPV visualization | 15/33 | Core rendering done; signature export + filtering pending |
| 053 M2 animation pose farm | 20/105 | Phase 0-1 complete; Phase 2-9 pending |
| 054 PM4 camera window cache | 17/18 | Nearly complete; 1 real-data test deferred |
| 044 Viewer shell usability | 10/13 | 3 remaining tasks |
| 046 PM4 asset matching | 22/39 (+1 file: `Pm4MatchRunOptions`) | WMO group matching phase done; signature matcher unstarted |
| 035 M2 render parity recovery | 9/28 | Continued fix work needed |
| 020-036, 042-045, 049, 055 | 0-2% | Not yet started |
| 056 ViewerApp + GPU + LOD | 0/80 | **NEW 2026-06-10**. Plan + tasks + contracts written. Phase 0 (test project + baselines) is the next step. Supersedes 036. |

## What Exists (Completed)

- 012 Real validation batch extraction ✅
- 014 Terrain MCAL parity ✅
- 024 V18 canvas paste refinement ✅
- 025 Object roof mask library ✅
- 034 WoWViewer rename ✅
- 047 V18 distill corpus open source loop ✅
- 048 M2 1121 era-aware MD20 reader ✅

## Recently Shipped

- **Spec 054**: Per-file PM4 cache (in-memory + on-disk) wired into WorldScene. Critical stamp-folding bug fixed. MSCN/MSPV lazy guard fixed. 18/18 PM4 per-file tests pass. Next time a staged client is available, T015 real-data test confirms end-to-end.

- **Spec 053 Phase 0-1**: `WowViewer.Core.Anim` library with M2/MDX loaders, path normalization, H:\CLIENTS rejection. 21/21 tests pass. Ready for Phase 2 (alias resolution + bone track extraction).

- **Spec consolidation**: 19 obsolete V16/V17 specs archived to `specs/archived/`. 050 and 052 merged into 046. Active spec count reduced from 52 to 25. ARCHIVED.md created with per-spec rationale. `PLANS-OVERVIEW.md` written as a summary.

- **Memory bank merge**: gillijimproject_refactor/memory-bank → wow-viewer/memory-bank/. All 8 standard files now present in wow-viewer. gillijim copy has a README.md redirect.

- **Documentation**: WoWViewer README.md created. CLI-TOOLS.md advanced guide written covering all 5 tools. Coding standards documented.

## What's Next

1. Pick the next active spec to implement (046/051/053 are the highest-priority)
2. Any new work should start with `$speckit-specify` if no spec exists yet
3. 2026-06-10 small-wins follow-through: launch a focused `WowViewer.Tools.Shared` test project so the new `Pm4MatchRunOptions.Validator` and the rest of the shared lib gain regression coverage
4. 2026-06-10 small-wins follow-through: spec the per-tile visibility work in `TerrainManager` (currently the no-op stub) so the deferred seam is tracked properly
5. 2026-06-10 spec 058 task #3 (CK24 byte decomposition) is shipped: `Ck24HighByte` and `Ck24LowByte` are now exposed on `Pm4MsurEntry` and `Pm4OverlayObject`, and the bulk JSON export includes them. The bond hypothesis is now testable. Next spec 058 follow-up: re-run the export on the development corpus and see whether any CK24 has surfaces with **different** `(Ck24HighByte, Ck24LowByte)` pairs - if so, the hypothesis is worth pursuing; if not, it's moot.
6. **2026-06-11 click-freeze investigation**: timing instrumentation shipped in `30461a1d`. User to run viewer with `WOWVIEWER_PM4_PROFILE=1` + `--verbose` and report the `[PM4-PROFILE]` log lines so we can see whether the freeze is in the pick, the per-frame research-info walk, or the post-click panel rebuild. **No fix code lands until the user reports numbers.**
7. **2026-06-11 MSCN+MSPV research finding**: trees have MSCN+MSPV, WMOs get ~50% of mesh as MSCN+MSPV (invisible containment walls), M2 models carry only top-of-model collision. Growth-potion "weak spots" are explained by containment being object-relative. **This rewrites the PM4 writer and matching-tool design.** The three-mode MSUR strategy and the MSCN+MSPV weighting must be added to spec 058 before any writer/matcher spec slice begins.
8. **2026-06-11 Project framing articulated by the user.** The full "why this project exists" section is now in `memory-bank/projectbrief.md` and was written this turn from the user's own words. Four framings: (1) PM4 is a giant map object addressed by a program/database above it, (2) ADT UniqueIDs and alpha-mask bands are sediment records of art in development, (3) there are countless historical artifacts in the bytes that have not yet been characterized, (4) the weak signal amplifier is the proof that the data is recoverable after the 33.334× downscale. **Specs whose body text needs the framing baked in, in priority order**: 009 (umbrella), 058 (central), 005 (workbench cleanup), 046 (matching tool), 051 (MSCN/MSPV), 057 (client archive selector — comparing the same map across builds is a first-class use case), 001 (V18 dataset), 012 (real-validation batch extraction), 053 (anim pose farm — light cross-ref only), 045 (scene-graph workbench predecessor). Update these in the fresh chat, not in this turn.

## Known Issues

- Renderer culling still has coordinate bug in `ComputeTilePlanarMin/Max` (spec 013, needs spec restoration)
- PM4 cache v7 → v8 version bump invalidates old per-window blobs (normal, transient)
- 14 pre-existing test failures in `WowViewer.Core.Tests` — stale fixtures in `ChunkedFileReader.ReadTopLevelChunks`
- Specs 013, 034, 035, 047, 048, 051 were lost to a PowerShell line-ending corruption during the consolidation pass. Their content exists in git history and `activeContext.md` references.

## Relevant Files

- `wow-viewer/README.md` — project overview
- `wow-viewer/docs/CLI-TOOLS.md` — advanced usage guide
- `wow-viewer/docs/PLANS-OVERVIEW.md` — remaining spec summaries
- `wow-viewer/memory-bank/*.md` — full memory bank
- `wow-viewer/specs/` — active feature specs
- `wow-viewer/specs/archived/` — obsolete specs
