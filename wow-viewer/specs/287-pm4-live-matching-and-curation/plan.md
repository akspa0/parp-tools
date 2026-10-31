# Implementation Plan: Spec 287 — Live PM4 Object Duplication & Matching, Burgundy-Red MCCV Holes, and Texture Layers Workbench

**Branch**: `287-pm4-live-matching-and-curation` | **Date**: 2026-10-31 | **Spec**: [requirements](spec.md)  
**State**: Planning only; no implementation, client, visual or screenshot completion claimed.

---

## 1. Summary of Architecture & Technical Design

This implementation adds four bounded, cohesive subsystems layered cleanly above the existing viewer:

1. **PM4 Doodad Duplication & Live Matching Service** (`Pm4DoodadDuplicationService.cs`):
   - Extends the Inspector and PM4 Workbench with interactive model cloning.
   - Takes a template `WorldObjectInstance` (e.g. `DRAGONBLIGHT_CLIFF_01.M2`) and scans the active tile for unmatched PM4 doodad collision clusters (`DominantGroupKey == 0x03` or `Ck24 == 0`).
   - Computes 3D transform (centroid position, scale factor from PM4 bounds vs model base bounds, planar orientation).
   - Injects cloned `WorldObjectInstance` objects into `_tileMdxInstances`, calls `RebuildInstanceLists()`, and enables instant visual proof.
   - Integrates with Project Output and Zarr v60 export for non-destructive curation persistence.

2. **MCCV Burgundy Hole Detection Service** (`MccvHoleDetectionService.cs`):
   - Implements color variance and chroma analysis across chunk MCCV vertex colors (145 vertices per chunk).
   - Detects solid burgundy-red underside fog signature ($R > 80, G < 30, B < 30, \sigma < 15$).
   - Applies connected-component spatial filtering: chunks receive `HoleMask = 0xFFFF` only if they belong to a contiguous cluster of $\ge 2$ candidate chunks.
   - Calls `TerrainManager.UpdateChunkHoleMask(...)` to update indices and visually cut the mesh.

3. **Terrain Layers & Shader Blending Workbench** (`TerrainLayersPanelService.cs` + `TerrainRenderer.cs` GLSL additions):
   - Adds `uniform float uLayerOpacity[8]` and `uniform float uAlphaMaskBlend` to the terrain fragment shader in `TerrainRenderer.cs`.
   - Modulates layer contribution: `result = mix(result, layerColor, alpha * uLayerOpacity[i])`.
   - Modulates alpha mask debug overlay: `finalColor = mix(finalColor, vec3(alphaDbg), uAlphaMaskBlend)`.
   - Renders a modern Photoshop-style Layers panel with per-layer thumbnails, toggle checkboxes, opacity sliders, and tooltips decoding 4.3.4 decomp flags.

4. **Skybox & Audio Immersion Corrections**:
   - In `SceneAtmosphere.cs`, removes the `NightVisibility > 0.001f` gating on `_activeLightSkyboxModelKey`, allowing client skybox models (`LightSkybox.dbc`) to render at daytime.
   - Prevents `_skyDome.UpdateFromLighting(...)` from wiping out active DBC/LIT zenith and horizon colors.
   - Connects camera area changes to `AudioService` for zone music playback.

---

## 2. Operator Intent Check

| Request | Planned response | Excluded implication |
|---|---|---|
| "duplicate nearby objects and match them to their pm4 data" (cliff doodads in `development.wdt`) | Phase 1: `Pm4DoodadDuplicationService` with Inspector action, scale/rotation fitting, and Zarr v60 export | Rewriting core format readers or altering original ADT MODF/MDDF structures directly |
| "burgandy red color... set chunks' holes flag to 1... group of 2 or more chunks" | Phase 2: `MccvHoleDetectionService` with 2+ contiguous chunk clustering and solid chroma threshold | Blindly holed terrain without adjacency clustering |
| "toolbar options... toggle buttons... photoshop layers-like opacity/blending... decode texture effects from 4.3.4 decomp" | Phase 3: Dedicated Layers window, `uLayerOpacity`/`uAlphaMaskBlend` shader uniforms, 4.3.4 decomp tooltips | Bloated permanent sidebars; adheres to action-bar popup architecture |
| "skyboxes to work... music/audio" | Phase 4: Full-day skybox rendering in `SceneAtmosphere.cs` and zone music trigger | Rewriting audio engine; leverages existing OpenAL/NAudio service |

---

## 3. Constitution Check (AGENTS.md)

| Rule | Status | Evidence |
|---|---|---|
| **§4 Core Library First** | PASS | Duplication math and hole detection policies reside in Core/Services; UI remains presentation-only. |
| **§4 Do Not Break Format Readers** | PASS | `LkAdtReader`, `NativeMpqService`, etc., remain completely untouched. |
| **§9 Governance Receipts** | PASS | Every task requires command line receipts, tests, and real-client data verification. |
| **§10 God-Class Freeze** | PASS | Zero new members added to `WorldScene.cs` or `ViewerApp.cs`. New features live in owned services (`Pm4DoodadDuplicationService`, `MccvHoleDetectionService`, `TerrainLayersPanelService`). |
| **§11 UI Standardization** | PASS | Uses `SharedUiWidgets` and registers all controls in `ui-controls.json`. |

---

## 4. Phased Implementation Roadmap

- **Phase 1: PM4 Doodad Live Duplication & Matching (`development.wdt`)**
  - Implement `Pm4DoodadDuplicationService.cs`.
  - Add "Fill PM4 Doodads with Model" button to Inspector when an M2 instance is selected.
  - Implement bounding-box extent matching, scale calculation, centroid snapping, and orientation fitting.
  - Unit tests in `Pm4DoodadDuplicationTests.cs`.
  - Wire to Zarr v60 placement export.

- **Phase 2: Burgundy MCCV / Minimap Terrain Hole Detection**
  - Implement `MccvHoleDetectionService.cs`.
  - Implement color clustering algorithm (RGB threshold, low variance, $\ge 2$ contiguous chunks).
  - Add "Derive Holes from Burgundy MCCV" button to Editor Terrain Tools page.
  - Unit tests in `MccvHoleDetectionTests.cs`.

- **Phase 3: Texture Layers Workbench & Shader Blending**
  - Add `uLayerOpacity[8]` and `uAlphaMaskBlend` to `TerrainRenderer.cs`.
  - Implement `TerrainLayersPanelService.cs` with layer thumbnails, opacity sliders, and decomp-backed tooltips.
  - Register controls in `ui-controls.json` and verify with `Verify-UiReference.ps1`.

- **Phase 4: Skybox Full-Day Rendering & Audio Immersion**
  - In `SceneAtmosphere.cs`, remove `_skyDome.NightVisibility > 0.001f` gating on skybox backdrop rendering.
  - Preserve DBC/LIT authored sky colors in `_skyDome.UpdateFromLighting`.
  - Connect camera area transitions in `WorldScene` to `AudioService.PlayZoneMusic`.
