# Plans Overview — Remaining Specs

After the June 2026 consolidation (see commit 94d733d8), 25 active specs remain. This doc summarizes what each one is about and what's needed to finish it.

## Legend

| Icon | Meaning |
|------|---------|
| ✅ | Complete — all tasks done |
| 🔄 | In progress — actively being worked |
| 📋 | Spec written, not yet started |
| 🔧 | Spec-only (spec.md), needs plan + tasks |

---

## High Priority

### 🔄 046 — PM4 Asset Matching (22/39 done)
Consolidated spec that absorbs old 050 (WMO group matching, 10/12 done) and 052 (signature matcher, 0/35 unstarted). Core PM4 → WMO/M2 matching pipeline for generating replacement placement data. Needs the signature matcher phase (35 tasks) then synthesis + viewer preview.

### 🔄 051 — PM4 MSCN/MSPV Visualization (15/33 done)
The visual analysis that turned MSCN from "navmesh graph nodes" into "scene-graph connector anchors" and MSPV from "shared vertices" into "path-vertex chains." Key remaining work: surface drop diagnostics, signature export, glossary, per-object clustering, filtering hooks.

### 🔄 053 — M2 Animation Pose Farm (20/105 done)
BVH + normalized pose clip extraction from M2/MDX models. Phase 0 (research) and Phase 1 (library + loaders) complete. Phase 2+ (alias resolver, track stream extractor, BVH writer, pose clip, FBX, batch mode, docs) pending.

### 🔄 054 — PM4 Camera Window Cache (17/18 done)
Per-file two-layer cache (in-memory + on-disk) that eliminates the "every camera jump is slow" PM4 UX bug. One deferred task (T015 real-data test) pending staged client fixture.

---

## Medium Priority

### 🔄 044 — Viewer Shell Usability (10/13 done)
Sidebar refinements, UI polish, workspace improvements. 3 remaining tasks.

### 🔄 035 — M2 Render Parity Recovery (9/28 done)
Fix world M2 doodad rendering consistency after the MdxViewer → WoWViewer migration. Needs continued work on alpha-cutout, stable render routing, and regression diagnostics.

### 🔄 013 — Object Mask Rendering Fix (2/10 done — spec lost to corruption)
Fix the visibility pipeline so headless validation capture produces real 3D mesh silhouettes, not empty masks. Spec needs to be restored from git history.

---

## Low Priority / Not Started

### 📋 020 — Renderer Culling and Tile Capture
Fix the culling bug (coordinates swapped in `ComputeTilePlanarMin/Max`) and enable multi-tile batch capture.

### 📋 029 — WMO Minimap Signal
Use WMO render passes to extract minimap-style signals (roof masks, wall silhouettes) for terrain reconstruction.

### 📋 030 — WMO Render Pass Architecture
Architecture doc for WMO render passes — currently has a deep spec but no implementation tasks yet.

### 📋 031 — Terrain Cell Awareness
Make the terrain pipeline aware of cell-family and tile-age metadata so the renderer can route per-age shader behavior.

### 📋 032 — Native Renderer Parity
Biggest unstarted spec (74 tasks). Goal: full WoW renderer parity (shaders, LOD, fog, M2 lighting) using native-client Ghidra evidence. Multi-month work.

### 📋 033 — MdxViewer Migration (79 tasks)
Full migration from the outstanding `gillijimproject_refactor/src/MdxViewer` functionality into wow-viewer.

### 📋 036 — Renderer Improvements
Shader unification, LOD system, lighting improvements.

### 📋 042 — Zarr-first MPQ-fallback Data Source
Replace the current ArchiveVirtualFileReader hierarchy with a Zarr-first data source that falls back to MPQ.

### 📋 043 — M2 Chunked MDX Classic Support
Cover the 0.5.3/0.7.0/0.8.0 chunked MDX format (the 1.12.1 era is handled by 048).

### 📋 045 — Scene Graph Workbench
Right-sidebar tree-view hierarchical scene graph for all scene data (PM4, ADT, WDT).

### 📋 049 — Viewer UI Consolidation
Unify UI patterns across the viewer — sidebar/panel management, theme system, keybinding consistency.

### 📋 055 — Unreal Engine Bridge (87 tasks)
Export wow-viewer scene data (terrain + placements + PM4) to Unreal Engine. Long-range strategic.

---

## Spec Only (Needs Plan + Tasks)

| Spec | What |
|------|------|
| 005 | PM4 workbench cleanup — consolidation of PM4 UI panels |
| 026 | Capture batch tuning — improving headless capture throughput |
| 037 | M2 3.0.1 embedded views adapter — Ghidra-driven M2 research |
| 038 | M2 3.0.1 renderer perf research — native-client performance analysis |
| 040 | MH2O/MCLQ liquid type determination — liquid render fix |
| 041 | MH2O/MCLQ liquid type determination fix — same fix, follow-up |

---

## Complete (Referenced for Context)

- 012 — Real validation batch extraction ✅
- 014 — Terrain MCAL rendering parity ✅
- 024 — V18 canvas paste refinement layer ✅
- 025 — Object roof mask library and minimap sieve ✅
- 034 — WoWViewer rename ✅
- 047 — V18 distill corpus open source loop ✅
- 048 — M2 1121 era-aware MD20 reader ✅

---

## Archived (specs/archived/)

19 obsolete specs — V16/V17 model architectures that were too complex to converge, MdxViewer-specific specs replaced by WoWViewer, and the old 050/052 PM4 matching specs consolidated into 046. See `specs/archived/ARCHIVED.md` for per-spec rationale.
