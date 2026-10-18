# Feature Specification: Client Ground-Truth Renderer Performance & Occlusion

**Feature Branch**: `v0.6.1-dev`

**Created**: 2026-10-15

**Status**: Specified; awaiting phase implementation

**Input**: Operator prompt: *"let's refine our renderer with appropriate stuff from there, since it's in dire need of help, performance is horrific for anything beyond 0.5.3, really. use speckit"* referencing the 3.3.5a decompilation ground truth at `H:\wowdecomps\3.3.5a\wowdecomp-main` (`wowdecomp`).

---

## 1. Background & Problem Statement

In `wow-viewer`, performance on content beyond 0.5.3 (such as 1.12, 3.3.5, and modern CASC maps like `wow_classic_beta` Azeroth) degrades severely:
1. **WMO Group Explosion & Broken Portal Occlusion (`R-15`)**:
   `WmoRenderer.cs` contains a fallback loop (`lines 1813–1819`) that marks every group whose bounding box touches the camera frustum as visible (`_runtimeVisibleGroups[groupIndex] = true`). This completely negates portal culling: when looking towards a city or structure (e.g. Stormwind Cathedral, Ironforge, Dalaran), every single interior room, basement, subterranean corridor, and enclosed doodad is drawn, generating over 7,500 groups and 80,000+ draw calls.
2. **Global WMO Instancing Disabled by Scene Lights (`R-10`)**:
   `WorldScene.cs` forces `canBatch = false` if `_sceneLightManager.Count > 0` (even a single brazier or torch anywhere in the world). Furthermore, `WmoRenderer.SupportsGpuInstancedOpaque` requires `_wmo.Portals.Count == 0`, completely preventing GPU instanced draws for nearly all world buildings.
3. **M2 / Doodad Full-Map Traversal**:
   `WorldScene.cs` iterates through all `_mdxInstances` in the entire world every frame during light collection and visibility preparation, causing severe CPU frame stalls.
4. **Terrain Shader & Lighting Discrepancies**:
   Lighting constants, attenuation factors, and texture matrix orientations diverge from the client's unified `SVertexShaderConstants` and `CShaderEffect` pipelines.

The ground-truth C++ client implementation in `H:\wowdecomps\3.3.5a\wowdecomp-main` provides the exact algorithms and data structures used by Blizzard to maintain 60+ FPS across complex WMOs and dense terrain.

---

## 2. User Scenarios & Acceptance Criteria

### User Story 1 - Authentic WMO Portal Occlusion & Interior Culling (Priority: P1)

As a viewer operator, I want WMO interiors and occluded rooms to be culled when viewing buildings from the outside (and vice-versa), so that frame rates in dense cities and complex structures remain smooth.

**Acceptance Scenarios**:
1. **Given** the camera is outside a WMO, **When** rendering the scene, **Then** interior groups (`group.Flags & 0x2000`) are culled unless visible through a front-facing exterior portal that intersects the frustum.
2. **Given** the camera is inside an interior WMO group, **When** rendering the scene, **Then** only the current group and connected groups reachable via open portals up to traversal depth (depth 4–10) are submitted.
3. **Given** an occluded interior room, **When** culled, **Then** none of its assigned doodads (`DoodadRefs`) are prepared, animated, or drawn.
4. **Given** a WMO with invalid or absent portal definitions, **When** evaluated, **Then** the renderer falls back safely to frustum culling without crashing or leaking memory.

---

### User Story 2 - Decouple WMO Shell Instancing from Scene Lights (Priority: P1)

As a viewer operator, I want identical WMO buildings (e.g., towers, huts, walls) to be GPU-instanced even when local point lights exist elsewhere in the scene.

**Acceptance Scenarios**:
1. **Given** multiple placements of an identical WMO model, **When** point lights exist in the scene, **Then** placements outside the influence sphere of any light are batched and drawn via GPU instancing (`DrawElementsInstanced`).
2. **Given** a placement within the radius of a local point light, **When** rendered, **Then** only that specific placement falls back to individual draw with local light uniforms applied.
3. **Given** WMOs with exterior-only groups or portal-culled shells, **When** batch candidates are planned, **Then** instancing is enabled per group/shell rather than globally rejected by `Portals.Count > 0`.

---

### User Story 3 - Spatialized M2 & Doodad Submission Pipeline (Priority: P2)

As a viewer operator, I want doodad lights and instances to be collected using tile-indexed spatial buckets rather than scanning the entire map's instance array every frame.

**Acceptance Scenarios**:
1. **Given** tens of thousands of placed M2 doodads across loaded tiles, **When** preparing visibility and collecting scene lights, **Then** only instances within the active camera tile window and near-field fog radius are evaluated.
2. **Given** visible opaque M2 doodads, **When** submitted, **Then** identical models sharing material and shader states are batched, matching `CM2Scene::Draw` batch sorting.

---

### User Story 4 - Ground-Truth Terrain Lighting & Shader Constant Alignment (Priority: P2)

As a viewer operator, I want terrain lighting, diffuse/ambient balance, and attenuation to match the 3.3.5 `SVertexShaderConstants` specification.

**Acceptance Scenarios**:
1. **Given** directional and point lighting on terrain, **When** shaders evaluate diffuse, specular, and local light attenuation, **Then** formulas match `CMapRenderChunk` ($1 / (k_c + k_l d + k_q d^2)$) and texture coordinate swaps ($a_0 \leftrightarrow b_0$).
2. **Given** day/night transitions and fog changes, **When** updated, **Then** terrain and object fog bounds blend consistently without clipping artifacts.

---

## 3. Constraints & Architectural Boundaries

1. **God-Class Freeze (§10)**: No new members in `WorldScene.cs` or `ViewerApp.cs`. New portal occlusion and instancing logic must live in cohesive service classes (e.g., in `WoWViewer.Rendering` or `WoWViewer.Terrain.Scene`).
2. **Working Format Readers Frozen (§4)**: Do not modify existing MPQ or WMO readers (`MpqArchiveCatalog.cs`, `NativeMpqService.cs`, `WmoV14ToV17Converter.cs`).
3. **Containment & Receipts (§9.2, §9.3)**: All performance metrics, draw call counts, and memory numbers must be verifiable via the in-viewer diagnostics overlay (`profile-render`, `WorldRenderDiagnostics`) and recorded with exact receipts.
