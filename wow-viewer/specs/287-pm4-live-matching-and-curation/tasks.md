# Tasks: Spec 287 — Live PM4 Object Duplication & Matching, Burgundy-Red MCCV Holes, and Texture Layers Workbench

**Input**: [specification](spec.md), [implementation plan](plan.md).  
**State**: Unchecked implementation handoff, 2026-10-31. No source, test, or runtime completion is asserted.  
**Order**: Phase 1 (PM4 Doodad Duplication) → Phase 2 (Burgundy Holes) → Phase 3 (Texture Layers Workbench) → Phase 4 (Skybox & Audio); 32 total tasks.

---

## Phase 1 — Live PM4 Doodad Duplication & Magnetic Matching (`development.wdt`)

**Goal**: Select a model (e.g. `DRAGONBLIGHT_CLIFF_01.M2` on tile 35_38) and clone it to fill all unmatched red PM4 doodad collision meshes, magnetically fitting position, scale, and rotation.

- [x] T001 [P] Implement unit tests in `Pm4DoodadDuplicationTests.cs` validating:
  - Extent ratio calculation between PM4 collision bounds and model base bounds.
  - Centroid alignment and planar rotation extraction.
  - Multi-instance UniqueId allocation and rejection of already-occupied PM4 meshes.
- [x] T002 Implement `Pm4DoodadDuplicationService.cs` in `Workbench/Services/Pm4Workbench/` with:
  - Method `DuplicateModelToUnmatchedPm4Doodads(WorldObjectInstance template, int tileX, int tileY, float searchRadius)`.
  - Detection of unmatched PM4 doodad hulls (`DominantGroupKey == 0x03` or `Ck24 == 0`).
  - Mathematical fitting of position, 3D scale, and rotation.
- [x] T003 Wire `Pm4DoodadDuplicationService` into `IPm4OverlayHost` and `InspectorPayloadsService.cs`:
  - Add button **"Fill PM4 Doodads with Model"** to the Context Inspector when an M2 instance is selected.
  - Display summary of duplicated count and bounding-box fit quality.
- [x] T004 Implement live scene injection:
  - Add cloned instances to `_tileMdxInstances[(tileX, tileY)]`.
  - Update `_terrainManager.TryUpdateCachedPlacementPosition(...)`.
  - Trigger `RebuildInstanceLists()`.
- [x] T005 Implement undo/discard buffer in `Pm4DoodadDuplicationService` so accidental duplications can be cleanly reverted.
- [x] T006 Wire persistence to Zarr v60 / Project Output:
  - Add **"Save Merged Placements to Zarr v60"** button in PM4 Workbench.
  - Write updated placement ledger to project output without modifying original ADTs.
- [x] T007 **Source/test gate**: Run `dotnet test` on PM4 duplication tests and `dotnet build` on `WoWViewer.csproj`; verify 0 errors and record receipt in `evidence/phase1-duplication-source.md`.
- [ ] T008 **Operator runtime gate**: Load `development.wdt` at tile 35_38, select `DRAGONBLIGHT_CLIFF_01.M2`, trigger "Fill PM4 Doodads with Model", and verify in viewport that red collision meshes are populated with cliff models.

---

## Phase 2 — Burgundy-Red MCCV / Minimap Terrain Hole Detection

**Goal**: Automatically detect the burgundy-red underside fog color in MCCV/minimaps and set chunk `HoleMask = 0xFFFF` for clusters of 2 or more contiguous chunks.

- [x] T009 [P] Implement unit tests in `MccvHoleDetectionTests.cs` validating:
  - Detection of burgundy-red chroma threshold ($R > 80, G < 30, B < 30$) with low variance ($\sigma < 15$).
  - Connected-component clustering requiring $\ge 2$ contiguous candidate chunks.
  - Rejection of isolated single chunks and textured/slope-shaded chunks.
- [x] T010 Implement `MccvHoleDetectionService.cs` in `Workbench/Services/TerrainAnalysis/`:
  - Method `AnalyzeChunkMccvHoles(TerrainChunkData[] chunks)` evaluating 145 BGRA vertices per chunk.
  - Connected-component cluster filtering.
- [x] T011 Wire `MccvHoleDetectionService` into `EditorTerrainToolsPage.cs`:
  - Add button **"Derive Holes from Burgundy MCCV"**.
  - Show candidate chunk count and holed cluster summary.
- [x] T012 Apply detected hole masks:
  - Update `TerrainChunkData.HoleMask` on resident chunks.
  - Rebuild chunk mesh indices via `TerrainChunkMath.BuildChunkIndices`.
  - Trigger terrain mesh redraw.
- [x] T013 **Source/test gate**: Run `dotnet test` on MCCV hole tests and `dotnet build`; record receipt in `evidence/phase2-mccv-holes-source.md`.
- [ ] T014 **Operator runtime gate**: Test on a tile with burgundy MCCV (or painted minimap); verify that 2+ contiguous burgundy chunks receive clean mesh holes and single chunks are ignored.

---

## Phase 3 — Photoshop-Style Texture Layers Workbench & Reverse-Engineered Flag Tooltips

**Goal**: Create a dedicated Layers panel with per-layer opacity sliders, alpha mask overlay blend intensity, and Ghidra client analysis tooltips.

- [x] T015 [P] Update GLSL shader in `TerrainRenderer.cs`:
  - Add `uniform float uLayerOpacity[8];` uniform array.
  - Add `uniform float uAlphaMaskBlend;` uniform.
  - In fragment shader, scale layer contribution by `uLayerOpacity[i]`.
  - In fragment shader, blend `finalColor = mix(finalColor, vec3(alphaDbg), uAlphaMaskBlend)`.
- [x] T016 Implement C# shader uniform bindings and properties in `TerrainRenderer.cs`:
  - `public float[] LayerOpacities { get; } = [1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f];`
  - `public float AlphaMaskBlend { get; set; } = 0f;`
- [x] T017 Implement `TerrainLayersPanelService.cs` in `Workbench/Services/TerrainControls/`:
  - Render list of resident texture layers (0 to 3, up to 7).
  - Checkbox toggle for visibility + float slider for opacity per layer.
  - "Alpha Mask Overlay" slider ($0.0 \dots 1.0$) to blend raw alpha mask over painted diffuse.
- [x] T018 Add Ghidra-verified tooltips for all `MCLY.Flags` and `EffectId`:
  - `0x07` (UV Anim Direction), `0x38` (UV Anim Speed), `0x40` (Anim Enabled), `0x80` (Specular Masking), `0x100` (Alpha Map), `0x200` (Alpha RLE), `0x400` (Cubemap Reflection), `0x8000` (Dynamic Texture), `EffectId` (Ground Clutter Doodads).
- [x] T019 Register new controls in `ui-controls.json` and run `Verify-UiReference.ps1`.
- [x] T020 Wire Layers toggle button to the top action bar or bottom bar.
- [x] T021 **Source/test gate**: Run `Verify-UiReference.ps1` and `dotnet build`; verify 0 errors, 0 warnings; record receipt in `evidence/phase3-layers-source.md`.
- [ ] T022 **Operator runtime gate**: Open Layers window, adjust per-layer opacity and alpha mask overlay slider; confirm in viewport that both painted textures and alpha masks blend simultaneously.

---

## Phase 4 — Skybox Full-Day Rendering & Audio Immersion

**Goal**: Enable client skyboxes to render across the full day cycle and hook camera area transitions to zone audio.

- [x] T023 In `SceneAtmosphere.cs` (`RenderSkyboxBackdrop`, line 361):
  - Remove `if (_skyDome.NightVisibility > 0.001f)` restriction on `_activeLightSkyboxModelKey`.
  - Allow `LightSkybox.dbc` models to render throughout the entire 24-hour day cycle with ambient lighting modulation.
- [x] T024 In `WorldScene.cs` (`RenderFrameCore` / `_skyDome.UpdateFromLighting`):
  - Preserve authored DBC/LIT zenith and horizon colors on `SkyDomeRenderer` instead of unconditionally overwriting with hardcoded constants.
- [x] T025 Connect camera area transitions in `WorldScene.cs` / `AreaContextService.cs` to `AudioService`:
  - When the camera enters a new Area ID, query `AreaTable.dbc` for `ZoneMusic` / `IntroMusic` and stream audio files from MPQs.
- [x] T026 Add audio playback volume and mute controls in the UI.
- [x] T027 **Source/test gate**: Run `dotnet build` on `WoWViewer.csproj`; verify 0 errors and record receipt in `evidence/phase4-sky-audio-source.md`.
- [ ] T028 **Operator runtime gate**: Sweep time of day from 0.0 (midnight) to 0.5 (noon) on a map with `LightSkybox.dbc`; confirm the skybox model remains visible at noon. Walk across zone borders and confirm zone audio playback.

