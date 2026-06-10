# Active Context — wow-viewer

**Branch**: `v0.5.0-dev` | **Date**: 2026-06-10

## Current Focus

**Spec 056 — ViewerApp Refactor + GPU Acceleration + LOD Modernization.** New spec, plan, research, data-model, contracts, quickstart, and 80-task breakdown written 2026-06-10. Target: shrink `ViewerApp.cs` (621k bytes) by promoting viewer-app `Rendering/*` (28+ files) into a real shared `WowViewer.Core.Renderer` library, modernize OpenGL via Silk.NET (retained-mode VBO/IBO/UBO + instanced rendering + frustum culling), and add the full LOD matrix (terrain mesh LOD, object LOD, water LOD, light LOD, WDL far horizon, BLP mipmaps). 9 phases (0-8), max 10 tasks each. Supersedes `specs/036-renderer-improvements`. Vulkan primary, compute shaders, async streaming, and the Unreal bridge are explicit out-of-scope follow-ons.

## Active Specs (by priority)

| Spec | Tasks | Status |
|------|-------|--------|
| 046 PM4 asset matching | 22/39 | WMO group matching phase done; signature matcher unstarted |
| 051 MSCN/MSPV visualization | 15/33 | Core rendering done; signature export + filtering pending |
| 053 M2 animation pose farm | 20/105 | Phase 0-1 complete; Phase 2-9 pending |
| 054 PM4 camera window cache | 17/18 | Nearly complete; 1 real-data test deferred |
| 044 Viewer shell usability | 10/13 | 3 remaining tasks |
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
