# Phase 3 Receipt: Photoshop-Style Texture Layers Workbench & Ghidra Client Binary Analysis Tooltips

**Date**: 2026-10-31  
**Spec**: Spec 287 (Phase 3: T015–T021)  
**Status**: COMPLETE  

---

## 1. Files Modified & Created

- **Created**:
  - `wow-viewer/src/viewer/WoWViewer/Workbench/Services/TerrainControls/TerrainLayersPanelService.cs`
- **Modified**:
  - `wow-viewer/src/viewer/WoWViewer/Terrain/TerrainRenderer.cs` (GLSL shaders, `LayerOpacities`, `AlphaMaskBlend`, uniform uploads)
  - `wow-viewer/src/viewer/WoWViewer/Workbench/Services/IViewerAppHost.cs` (`TerrainLayersPanelService TerrainLayersPanel`)
  - `wow-viewer/src/viewer/WoWViewer/ViewerApp.cs` (instantiate `_terrainLayersPanel`)
  - `wow-viewer/src/viewer/WoWViewer/ViewerApp_Host.cs` (expose `TerrainLayersPanel`)
  - `wow-viewer/src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerActionBarsService.cs` (add `layers` window)
  - `wow-viewer/src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerUiManager.cs` (`OpenLayersWindow`, `ToggleLayersWindow`, `IsLayersOpen`)
  - `wow-viewer/src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.View.cs` (Add "Terrain Layers & Blending..." menu item)
  - `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Chrome/ViewerChromeService.Options.cs` (Add "Open Layers & Blending Workbench..." button)
  - `wow-viewer/eng/ui/ui-controls.json` & `eng/ui/New-UiControlRegistry.ps1` (Register `open-layers-blending-workbench`, `terrain-layers-blending`)
  - `wow-viewer/docs/WoWViewer/UI-REFERENCE.md` (Update generated reference)

---

## 2. Verification Commands & Exit Status

1. `pwsh -NoProfile -File wow-viewer/eng/ui/Verify-UiReference.ps1`
   - **Exit Status**: 0 (`UI_REFERENCE_CHECK inventory_rows=137 distinct_labels=127 registry_entries=127`, `UI_REFERENCE_CHECK PASSED errors=0 warnings=0`)
2. `dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
   - **Exit Status**: 0 (0 errors)

---

## 3. Criterion -> Real Output Table

| Criterion | Implementation & Evidence | Result |
| :--- | :--- | :--- |
| **AC-301 / T015**: GLSL Fragment Shader Opacity & Blend | Added `uniform float uLayerOpacity[8];` and `uniform float uAlphaMaskBlend;` to chunk and tile fragment shaders; layer diffuse contributions scaled by `uLayerOpacity[i]` and blended via `finalColor = mix(finalColor, vec3(alphaDbg), clamp(uAlphaMaskBlend, 0.0, 1.0))`. | PASS |
| **AC-302 / T016**: Uniform Bindings & Host State | `TerrainRenderer.cs` exposes `public float[] LayerOpacities { get; } = [1f, ...];` and `public float AlphaMaskBlend { get; set; } = 0f;`; uniform locations cached (`_uTileLayerOpacityLocation`, `_uTileAlphaMaskBlendLocation`, `_uChunkLayerOpacityLocation`, `_uChunkAlphaMaskBlendLocation`) and uploaded in `RenderTileMeshes` and `RenderChunks`. | PASS |
| **AC-303 / T017**: Layers Workbench Dialog | `TerrainLayersPanelService.DrawLayersPanelContent` renders per-layer visibility checkboxes, per-layer opacity sliders ($0.0 \dots 1.0$), "Alpha Mask Overlay" blend intensity slider ($0.0 \dots 1.0$), "All 100%" and "Reset Opacity" buttons. | PASS |
| **AC-304 / T018**: Ghidra Client Binary Analysis Tooltips | Implemented `DrawMclyFlagsHelp` with Ghidra client binary analysis documentation: `0x07` (UV Anim Direction), `0x38` (UV Anim Speed), `0x40` (Anim Enabled), `0x80` (Overbright/Specular), `0x100` (Alpha Map), `0x200` (Alpha RLE), `0x400` (Cubemap Reflection), `0x8000` (Dynamic Texture), and `EffectId` (Ground Clutter Doodads). | PASS |
| **AC-305 / T019–T020**: UI Integration & Registry Verification | Registered in `ui-controls.json`; wired to `MainMenuBarService.View.cs` ("Terrain Layers & Blending..."), `ViewerChromeService.Options.cs` ("Open Layers & Blending Workbench..."), and `ViewerActionBarsService.cs` tool window; verified clean 0 error/0 warning via `Verify-UiReference.ps1`. | PASS |

