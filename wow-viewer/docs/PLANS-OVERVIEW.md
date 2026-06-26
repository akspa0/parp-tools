# Plans Overview — Remaining Specs

After the June 2026 consolidation, 25 active specs remain. This doc summarizes what each one is about and what's needed to finish it.

## Legend

| Icon | Meaning |
|------|---------|
| ✅ | Complete — all tasks done |
| 🔄 | In progress — actively being worked |
| 📋 | Spec written, not yet started |
| 🔧 | Spec-only (spec.md), needs plan + tasks |

---

## High Priority

### 🔄 044 — Viewer Shell Usability (10/13 done)
Sidebar refinements, UI polish, workspace improvements. 3 remaining tasks.

### 🔄 046 — PM4 Asset Matching (22/39 done)
Consolidated spec that absorbs old 050 (WMO group matching) and 052 (signature matcher). Core PM4 → WMO/M2 matching pipeline for replacement placement data. Needs signature matcher phase plus synthesis + viewer preview.

### 🔄 051 — PM4 MSCN/MSPV Visualization (15/33 done)
The visual analysis that turned MSCN from "navmesh graph nodes" into "scene-graph connector anchors" and MSPV from "shared vertices" into "path-vertex chains." Key remaining work: surface drop diagnostics, signature export, glossary, per-object clustering, filtering hooks.

### 🔄 053 — M2 Animation Pose Farm (20/105 done)
BVH + normalized pose clip extraction. Phase 0 (research) and Phase 1 (library + loaders) complete. Phase 2+ (alias resolver, track stream extractor, BVH writer, pose clip, FBX, batch mode, docs) pending.

### 🔄 054 — PM4 Camera Window Cache (17/18 done)
One deferred task (real-data test) pending staged client fixture.

---

## Medium Priority

### 🔄 035 — M2 Render Parity Recovery (9/28 done)
Fix world M2 doodad rendering consistency after the MdxViewer → WoWViewer migration. Needs continued work on alpha-cutout, stable render routing, and regression diagnostics.

### 🔄 013 — Object Mask Rendering Fix (2/10 done — spec lost to corruption)
Fix the visibility pipeline so headless validation capture produces real 3D mesh silhouettes, not empty masks. Spec needs to be restored.

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
Biggest unstarted spec (74 tasks). Goal: full WoW renderer parity (shaders, LOD, fog, M2 lighting) using native-client Ghidra evidence.

### 📋 033 — MdxViewer Migration (79 tasks)
Full migration from the `gillijimproject_refactor/src/MdxViewer` functionality into wow-viewer.

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

## Most Critical Blocks (P1)

### **PROBLEM 1: UI Shell Regression**

**Location**: `specs/044-viewer-shell-usability`
**Status**: Phase 1+2 complete (dockable shell, menu declutter)
**Impact**: Viewer shell is broken - no dockable panels, broken map discovery, legacy tools in primary menus
**Why critical**: Without fixing 044, the viewer UX is non-functional

**Remaining tasks**:
- T020: Run `pm4 match-report` on dev tile corpus (not UI)
- T021: Run `pm4 match-report` on 3.3.5 real map data (not UI)

**P1 blockers on other specs**:
- `specs/046-pm4-asset-matching/T022-T027`: Coordinate conversion + CK24 grouping tests rely on known-tile proofs (helping UI debugging)
- `specs/051-mscn-mscv`: Visualization improvements may require new UI panel in the dockable shell

### **PROBLEM 2: PM4 Asset Matching Pipeline Blocked**

**Location**: `specs/046-pm4-asset-matching/T022-T027`
**Status**: Core pipeline broken at coordinate conversion and scoring level
**Impact**: No real matches on real data, researcher cannot get actionable insight
**Why critical**: Users want specific placements (e.g., "match OilPlatformLow.WMO to PM4")

**Remaining critical tasks**:
- **T022**: Fix coordinate mismatch in `Pm4MatchSupport` - the most-tested small bug
- **T023**: Validate on tile 0_0 - quick real-data proof
- **T024**: Add CK24-grouped scoring - the correct grouping level for WMO matching
- **T025**: Validate CK24 grouping works - prove fix enables matching

### **PROBLEM 3: ML Training Pipeline Deferred**

**Status**: V21 terrain pivot active but ML work deferred
**Location**: `specs/068-fractal-aware-height-loss` (archived) and V21 training in memory-bank
**Why deferred**: Focus shifted to V21 terrain pivot + UI shell restoration

---

## Workspace Reset Plan

### Phase 1 (Quarter 3 2026-Q1 2027): Return to UI Shell

1. **Complete Spec 044** - the dockable shell restoration (last barrier)
   - T020/T021: Mark done once dev proof shows no regressions

2. **Fix Spec 046 blockers** - the coordinate conversion and scoring bugs
   - T022-T027: Complete 1-2 per week to unlock matching pipeline

3. **Validate repairs** - small real-data proofs:
   - Verify dockable panels work in viewer
   - Verify `pm4 match-report` produces candidates on tile 0_0

### Phase 2 (Q2 2027): Enable PM4 Matching Research

4. **Spec 046 Phase 5-6** - filtering by model type and evidence trails
   - Port 35 signature-matcher tasks

5. **Spec 046 Phase 8** - correlation research
   - T031-T035: Expand correlation dataset, fix M2 collision reading

### Phase 3 (Q3-Q4 2027): Integration

6. **Consolidate** - remove 047/054 legacy helper scripts
7. **Viewer integration** - move PM4 workbench to dockable shell
8. **Visual pipeline** - surface shows SKF signatures in viewer

---

## Key Architecture Directives

- **Rule 1**: New code in `wow-viewer/` only
- **Rule 2**: One phase at a time - no scope creep
- **Rule 7**: Models are small, predict residuals - no monolithic replacements
- **Rule 11**: Spec Kit first - every non-trivial task has a spec
- **Rule 8**: Validate against real data before moving forward

---

### Current Fix Status

**In progress:**
- Spec 044: Dockable shell panels (9/10 T001-T012 completed)
- Spec 053: M2 animation pose farm (20/105 completed)
- Spec 054: PM4 camera window cache (17/18 completed)

**Awaiting validation:**
- Spec 044: phase 2 complete (menu declutter, shell mode toggle)
- Spec 046: coordinate conversion fix (T022 not started)
- Spec 046: CK24-grouped scoring (T024 not started)

---

This document reflects the consensus after the June 2026 crisis reset: return to UI shell work first, fix critical PM4 blockers, and defer ML training to clarify the path forward.

---

## Current Project State

**🟢 ACTIVE UI FOCUS**: Spec 044 (Viewer Shell Usability)
**🟡 PM4 PIPELINES**: Spec 046 (PM4 Asset Matching) blocked by 4 small blockers
**🔄 OBSOLETED**: ML training and V19/V20 terrain models (spec 066/067 archived)
**📋 NOT STARTED**: Arch complexity, unreal bridge, terrain cell awareness

**Core Mission**: Fix the viewer shell so users can actually use WoW maps and browse assets. PM4 matching is blocked at the coordinate conversion step. ML work is deferred until the shell is stable.
