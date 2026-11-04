# Active Context — wow-viewer

**UI Flicker ROOT CAUSE — ImGui NewFrame/EndFrame desync (2026-11-04):**
Operator reported seizure-inducing whole-UI strobing + dead input + 2 CTDs beginning around
`d346cf24`. Windows Event Log (.NET Runtime 1026, 01:03 + 01:31) captured both crashes: AVs in
`igEndMenu` / `igBeginPopupModal` = **ImGui window-stack corruption**. Root cause: `_imGui.Update()`
(NewFrame) ran in `OnUpdate` while `_imGui.Render()` (EndFrame) ran in `OnRender`; `2b3762d3`'s
independent 60/60 schedules make Silk.NET invoke the callbacks on different cycles, so widget code
regularly ran outside a begun frame — survivable corruption = the strobe; deep corruption = the
crashes. Pacing Render inside OnRender (earlier attempts) desynced further → "overdrive". Fixed:
1. **ImGui frame bound to OnRender** (`ViewerApp`): `_imGui.Update` moved from OnUpdate to just
   before the UI block in OnRender; `DrawUI` + `ImGuiPathPicker.Draw` in try/finally around
   `_imGui.Render()` — NewFrame/EndFrame always paired; a throwing widget closes the frame.
2. Even present cadence (~60 FPS Stopwatch target, hybrid sleep+spin, `timeBeginPeriod(1)`).
3. `OpenLeftDock` idempotent; `SaveLayout` skips byte-identical rewrites (no per-frame disk I/O).
4. TEMPORARY `UiOscilloscope` (on by default, `PARP_UI_OSCILLOSCOPE=0` opts out) logs per-frame
   state/rects/draw-data to `output/settings/ui-oscilloscope.log` — **remove after operator
   confirms the fix**.
Validation: build 0 errors; 22/22 focused UI tests pass. **Owed:** operator run — exercise File
menu + panels; confirm stable/interactive; oscilloscope log verifies cadence + catches any
residual toggle. Amendment: [Spec 292](../specs/292-ui-bottom-bar-toggles-audio-sky/spec.md).

**WMO Alignment Z Elevation Fix & Ck24 "Object Z" UI Relabeling (2026-11-03):**
1. **WMO Alignment & Placement-Z Resolution**:
   - Fixed vertical displacement when aligning WMOs to PM4 hulls in `Pm4DoodadDuplicationService.AlignPlacementToPm4Hull`, `MagnetizeAllPlacementsOnTile`, and `Pm4DirectModelMatcher.MatchPrism`.
   - Root cause: code previously subtracted `rotMin.Z` (the negative foundation depth, e.g. -16.14 yd) from `prism.BaseZ` (`placementZ.Value`), which pushed buildings 16+ yards into the sky. WMO local origin (0,0,0) is authored directly at `MODF.Position.Z`, so `targetPlacementZ` now aligns directly to `placementZ.Value` (or `prism.BaseZ`).
   - Added `NudgePlacementZ(ObjectInstance, float deltaZ)` and responsive interactive UI nudge buttons (`[-1 yd]`, `[-0.1]`, `[+0.1]`, `[+1 yd]`) beside the rotation controls in `Pm4WorkbenchSidebarService.cs`.
2. **Relabeled CK24 to "Object Z" with Decoded Elevation**:
   - Renamed user-facing "CK24" labels to "Object Z" across the UI, displaying the decoded IEEE-754 Z float elevation (e.g. `Object Z 47.160` instead of `0x423CA4`).
   - Updated pairing inspection, nearby hull recommendations, scene graph tree nodes, hover tooltips, and inspector panels to present decoded Object Z as coordinate data rather than an object identity enum.
3. **Validation**:
   - Solution `WoWViewer.CrossPlatform` built with 0 errors.
   - 191/191 PM4 tests passed in `WowViewer.Core.PM4.Tests`.
   - `Verify-UiReference.ps1` passed (133 inventory rows, 122 registry entries, 0 errors, 0 warnings).

**Visual Donor Tile Picker, Placed Tile Offsets & Workspace Lighting (2026-11-03):**
1. **Interactive Visual Donor Tile Picker (`DonorTilePickerService.cs`)**:
   - Built interactive visual 64x64 WDL overview dialog (`DonorTilePickerService.cs`) launched from the Archaeology Phase Layers panel.
   - Allows users to select donor maps, inspect occupied tiles (highlighted in cyan), select single or multi-tile regions via click or drag-box selection (amber highlight), and choose target placement modes (exact match vs custom anchor slot).
   - Generates and applies `PhaseLayerSettings` with `UsePlacedTilesOnly = true` and configured `TilePlacements`.
2. **Placed Donor Tile Offsets & Minimap Dragging**:
   - In `PhaseComposition.cs`, resolved placed donor tile targets through `placement.TargetTileX + layer.TileOffsetX` and `placement.TargetTileY + layer.TileOffsetY`.
   - Updated `WorldScene.GetLayerFootprints()` and `MinimapAndStatusService` so placed-tile donor layers respect tile offsets during footprint generation and allow interactive click-dragging across the loaded map's minimap.
3. **Workspace Lighting & LIT Consolidation**:
   - Moved all lighting controls (ambient, direct, fog range, and LIT loader/status) into `LeftPanel.Workspace` under `Lighting` (`NavigatorPanelService.Sidebars.cs`), removing redundant placement in `Options > Display`.
   - Registered authoritative home in `ui-controls.json` and synchronized `docs/WoWViewer/UI-REFERENCE.md`; `Verify-UiReference.ps1` passed with 0 errors and 0 warnings.
4. **Validation**:
   - Solution `WoWViewer.CrossPlatform` built with 0 errors.
   - 20/20 `PhaseTileSourceTests` passed (including 2 new tests verifying placed-tile offset resolution and locked claims).
   - 17/17 `ViewerFooterLayoutTests` passed; 5/5 `UiControlDocumentationTests` passed.

**Diagnostics Left-Dock Integration, Bounding Box Gating & Solid PM4 Color Fix (2026-11-03):**
1. **Docked Performance & Log Inside Left Sidebar Diagnostics**:
   - Replaced pop-out floating windows (`ImGui.Begin("Performance")` and `ImGui.Begin("Log")`) with embedded segmented tabs (`[ Stats ] [ Performance ] [ Log ]`) within `NavigatorPanelService.Sidebars.cs` under `LeftPanel.Diagnostics`.
   - External triggers (`ShowPerfWindow`, `ShowLogViewer`) cleanly switch active dock to `LeftPanel.Diagnostics` and select the appropriate tab.
   - Registered `stats` control in `ui-controls.json` and synchronized `docs/WoWViewer/UI-REFERENCE.md`; `Verify-UiReference.ps1` passed with 0 errors and 0 warnings.
2. **Fixed Persistent Yellow Bounding Boxes**:
   - In `ExternalSpawnLayer.cs`, defaulted `ShowSpawnBoundingBoxesOnly` to `false` and corrected `IsSpawn(in ObjectInstance inst)`: removed `!string.IsNullOrEmpty(inst.AssetKind)` which previously caused every resident world doodad/WMO to be treated as a spawn.
   - In `WorldScene.cs`, strictly gated bounding box rendering on `_showBoundingBoxes`.
3. **Restored Solid-Color PM4 Geometry**:
   - In `Pm4ColorCodingEngine.cs`, removed `(uint)objectPartId` from the hash in `ComputeInstanceColor`. Multi-part WMOs and doodad connectivity splits now evaluate to identical solid colors per instance rather than multicolored patchwork polygons.
4. **Validation**:
   - Solution `WoWViewer.CrossPlatform` built with 0 errors.
   - 191/191 PM4 tests passed (`WowViewer.Core.PM4.Tests`).
   - 22/22 UI and Footer tests passed (`WowViewer.Core.Tests`).
   - `Verify-UiReference.ps1` passed (133 inventory rows, 122 registry entries, 0 errors, 0 warnings).

**Spec 292 Refinement — PM4 Bottom Toggle, Drop-Up Layers Menu & LeftDock Photoshop Layers (2026-11-03):** [Spec 292](../specs/292-ui-bottom-bar-toggles-audio-sky/spec.md)
Refined UI layout following operator direction to eliminate arbitrary scene toggles from the bottom bar and adhere all panels to sidebars:
1. **Streamlined Bottom Bar (`BottomBarService.cs`)**:
   - Preserved `[x] PM4` as the single essential scene toggle on the bottom bar.
   - Replaced all 8 unwanted scene checkboxes with a `Layers & Opacity` drop-up menu right on the bottom bar, providing quick sliders for Alpha Mask Overlay blending (0.00-1.00), channel combo, resident layer opacities (L0-L7), quick reset/preset buttons, and a link to the full workbench.
   - Retained the 4 mode buttons (`Viewer`, `Editor`, `Archaeology`, `Options`).
2. **LeftDock Photoshop Layers Integration**:
   - Added `Layers` icon (`IconKind.Layers`) with stacked-layer pictogram to the Left Action Rail in `ViewerActionBarsService.cs` for 1-click access to `UiRegion.LeftDock`.
   - Merged the full Photoshop-style layer controls (`TerrainLayersPanelService.cs`) directly into `Options > Layers` (`ViewerChromeService.Options.cs`), including per-layer opacity sliders, alpha overlay blending, Ghidra MCLY flags decoding, global Holes masking, Burgundy MCCV hole derivation, and WDL terrain toggle.
   - Added clean empty states with a direct "Browse Maps..." button when no terrain is loaded.
3. **Dual Sidebar Dock Adhesion**:
   - All panels and views adhere strictly to `UiRegion.LeftDock` or `UiRegion.RightDock` — zero floating or unanchored windows over the viewport.
4. **Validation**:
   - `Verify-UiReference.ps1` passed with 0 errors and 0 warnings (132 inventory rows, 121 registry entries).
   - Solution `WowViewer.slnx` built with 0 errors (`dotnet build -c Debug`).
   - 17/17 `ViewerFooterLayoutTests` passed; 5/5 `UiControlDocumentationTests` passed. Receipt in `specs/292-ui-bottom-bar-toggles-audio-sky/evidence/final-receipt.md`.



**Tile GLB Export Texture Axis Correction (2026-11-03):**
Resolved horizontal texture inversion on exported terrain tile GLBs:
1. **glTF 2.0 Texture Coordinate Alignment**: In `MapGlbExporter.BuildTerrainTileMesh`, corrected `vTile` calculation from `1f - (chunk.ChunkY + vChunk) / 16f` to `(chunk.ChunkY + vChunk) / 16f`.
2. **Root Cause**: The glTF specification defines texture origin $(0, 0)$ at the top-left (North-West) of the image raster with $V$ increasing downwards (South). Inverting $V$ inverted the North-South axis, which in the glTF Y-up conversion maps directly to the 3D $X$ axis ($pos.X = wx$), flipping the texture along the $X$ axis across all exported terrain tiles.
3. **Validation**: Verified land/water metric jump on exported test tiles (`50_30.glb`), full project build succeeded with 0 errors (`dotnet build WoWViewer.csproj -c Debug`), and all 191 PM4 tests passed.

**PM4 Rotation Alignment, Composite Hull PCA & Real ADT MODF/MDDF Patching (2026-11-03):**
Resolved object rotation alignment to PM4 navmesh collision hulls and end-to-end ADT chunk generation:
1. **Composite PM4 Hull Extrusion**: When magnetizing or pairing multi-part composite structures (`Ck24 != 0`), `ExtractCompositeSurfaceVertices` aggregates transformed triangles across all constituent parts in the tile, ensuring PCA orientation, centroid, and bounding dimensions represent the entire building/bridge rather than an isolated sub-part.
2. **Yaw Orientation & Aspect Ratio Solving**: Corrected yaw derivation in `AlignPlacementToPm4Hull` and `Pm4DirectModelMatcher`. Aligns the model's authored major axis ($X$ vs $Y$) with the PM4 footprint's 2D PCA principal axis (`prism.YawDegrees`) according to `WorldPlacementTransform` rendering conventions ($\text{RotZ}(\pi)$ reversal), seamlessly integrating taught pairing and curated database yaw offsets.
3. **1-Click In-Place Rotation Controls**: Added `Rotate +90°`, `Rotate -90°`, and `Flip 180°` quick-action buttons in both Model-selected and Paired workbench modes. Rotates placements in-place around their world centroid, recalculates bounds, updates the scene, and automatically re-teaches/persists the corrected `YawOffset` for cross-tile propagation.
4. **Authentic MODF & MDDF Chunk Generation**: Upgraded `SavePatchedAdts` and `AdtPlacementEditor` to emit `AdtPlacementMoveEdit` and `AdtPlacementRotateEdit` for moved/magnetized resident placements (`inst.UniqueId < SyntheticUniqueIdBase`), and fixed `Pm4AdtPatchSupport` writing `Vector3.Zero` rotations. Full authentic `Position` and `Rotation.Z` (Euler Yaw) are now serialized into binary `MODF` (WMO) and `MDDF` (M2) chunks.
5. **Validation**: All 8 `AdtPlacementEditorTests` pass; all 4 `Pm4PriorKnowledgeDatabaseTests` pass; full `WoWViewer` application builds with 0 errors.

**PM4 Prior Knowledge Database & On-the-Fly Neighborhood Matching (2026-11-03):**
Integrated built-in pre-populated correlation database and neighborhood auto-matching for PM4 footprint prisms:
1. **Prior Knowledge Database (`Pm4PriorKnowledgeDatabase.cs`)**: Pre-indexes verified 3D ground-truth relationships (`pm4_curated_relationships.json`), persistent taught templates (`pm4_taught_pairings.json`), and the 895 whole-map probe models (`object-models-v4-2026-10-20.json`) across 18 architectural and environmental families (`Wintertauren`, `Tauren`, `Bridges`, `Keeps & Castles`, `Towers`, `HumanArchitecture`, `Mines`, `Ships`, etc.).
2. **On-the-Fly Matching Candidate Pool**: Upgraded `Pm4WorkbenchSidebarService.MatchPm4Footprint` to evaluate candidates from the Prior Knowledge Database and the inspector-selected object in addition to active tile-resident models, solving the empty/mismatched candidate problem when matching large architectural hulls on tiles containing only small doodads.
3. **Neighborhood & Family UI**: Added source provenance tags (`[Prior DB]` vs `[Resident]`) to match candidate list, and added a collapsible "Same Neighborhood / Family Models" browser to the workbench sidebar for 1-click inspection and pairing.
4. **Validation**: `WowViewer.Core.PM4` builds with 0 errors; `Pm4PriorKnowledgeDatabaseTests` passes; `Verify-UiReference.ps1` passes with 139 inventory rows, 129 registry entries, 0 errors, 0 warnings.

**Spec 291 Implementation Complete & Verified (2026-11-02):** [Spec 291 Dual Sidebar Docking, Destination Integration & UI Harmonization](../specs/291-ui-sidebar-consolidation/spec.md)
Comprehensive UI audit, deduplication, and unification of viewer, editor, options, and archaeology workbenches into anchored dual-dock sidebars:
1. **Dual Dock Architecture (`UiRegion.LeftDock` & `UiRegion.RightDock`)**: Added `UiRegion.LeftDock` anchored to the left action rail for Workspace, Files, Maps, Assets, Taxi, Capture, Diagnostics, Options, and Layers, eliminating floating tool window clutter over the 3D viewport. Includes interactive edge splitter with persisted width.
2. **Destination Integration**: Retired the lower-left floating `destination` window. Editor and Archaeology workbenches dock directly inside `UiRegion.RightDock` (and left sidebar) with seamless bottom-bar destination switching.
3. **Deduplication & Streamlining**: Removed redundant `Pages` button from the bottom action bar; streamlined `ViewerPanel.World` from 4 to 3 pages (replacing duplicate Exports tab with action button linking to `ViewerPanel.Exports`); streamlined Model Info in Inspector; centralized Terrain File Operations (Alpha, Heightmap 257, MCCV) and DAT exports into `ExportWorkbenchService`; cleaned up duplicate capture and vestigial buttons in Archaeology.
4. **Design System & Registry**: Standardized bespoke styling in `AssetCatalogView.cs`, `ArchaeologyPanelService.PhaseLayers.cs`, and `EditorPanelsService.cs` using `UiTokens`; updated `ui-controls.json` and regenerated `UI-REFERENCE.md`.
5. **Validation**: `Verify-UiReference.ps1` passed with 0 errors and 0 warnings (139 inventory rows, 129 registry entries); full project build succeeded with 0 errors; all targeted unit tests passed (17 `ViewerFooterLayoutTests`, 5 `UiControlDocumentationTests`). Receipt in `specs/291-ui-sidebar-consolidation/evidence/final-receipt.md`.

**Spec 290 Implementation Complete & Verified (2026-11-01):** [Spec 290 Unified Export Workbench & Panel Docking](../specs/290-unified-export-workbench-and-panel-docking/spec.md)
Centralized all PM4, World, Model, and ML exports into dedicated right sidebar workbench (`ViewerPanel.Exports`), and repositioned bottom destination windows:
1. **Consolidated Export Hub (`ExportWorkbenchService`)**: Unifies PM4 collision & navigation exports (JSON, OBJ Set, LLM Evidence Bundle, Visible Overlay Report, Zarr v60 archive, single-object JSON), World & Terrain exports, Model & Asset mesh exports, and Archaeology/ML dataset exports into a single authoritative right-dock panel.
2. **Right Action Rail Integration**: Added `ActionExports` icon button with `IconKind.Export` (upward-arrow tray pictogram) in `ViewerActionBarsService.cs` (`ViewerPanel.Exports`).
3. **Bottom Destination Repositioning**: Moved `destination` tool window (`Editor`, `Archaeology`, `Pages`) from top-right `(rightX, rightY)` overlapping `UiRegion.RightDock` to the lower-left workspace adjacent to the chat dock (`rail + 12f * scale + chatWidth`), ensuring $>500$ px horizontal clearance so inspection sidebars are never obscured.
4. **Validation**: `dotnet build WoWViewer.CrossPlatform.csproj` succeeded with 0 errors; `Verify-UiReference.ps1` passed with 138 inventory rows, 128 registry entries, 0 errors, 0 warnings. Receipt in `specs/290-unified-export-workbench-and-panel-docking/evidence/final-receipt.md`.

**Spec 289 Implementation Complete & Verified (2026-11-01):** [Spec 289 PM4 3D Footprint Extrusion & Direct MPQ Matching](../specs/289-pm4-3d-footprint-and-direct-matching/spec.md)
3D envelope reification, zero-donor asset matching, PCA orientation recovery, and live legend tallying:
1. **Top-Surface Magnetization & Placement Fix (Phase 1)**: Corrected vertical alignment so doodad placement position $Z$ satisfies $P_Z = Z_{\text{pm4\_top}} - (H_{\text{model\_top}} \cdot s)$, ensuring PM4 collision geometry sits on top of the object rather than trapped underneath at ground level.
2. **3D Footprint & Synthetic Wall Extrusion (Phase 2)**: Core algorithm in `WowViewer.Core.PM4/Footprint/Pm4FootprintExtruder.cs` computing 2D convex hull via Andrew's monotone chain, 2D PCA principal axes, and base elevation $Z_{\text{base}}$ via MPRL terrain contacts or Placement-Z markers. Synthesizes vertical wall quads closing the 3D footprint prism.
3. **Scale-Invariant Geometric Matching (Phase 3)**: Dimensionless aspect ratio descriptors ($W/D$, $H/\max(W,D)$) in `Pm4GeometricShapeDescriptor.cs` and direct MPQ candidate matching in `Pm4DirectModelMatcher.cs`. Solves authentic Yaw and accounts for the Warcraft III engine asymmetric pivot heritage ($\vec{P} = \vec{C}_{\text{world}} + R(\text{yaw}) \cdot (-\vec{C}_{\text{local}} \cdot s)$).
4. **Viewer Workbench Integration (Phase 4)**: Added "PM4 Extruded 3D Footprint Walls" toggle checkbox and "Match PM4 Footprint to Resident Models" action button in `Pm4WorkbenchSidebarService.cs`, wired through `Pm4OverlayMeshRenderer.cs`.
5. **Generator/Patcher Tooling Upgrade & Legend Tally (Phase 5)**:
   - Upgraded `Pm4PlacementGeneratorSupport.cs` from naive AABB dimensions to 2D convex hull PCA orientation, enabling rotation-invariant matching and outputting authentic `rotYaw` in placement records.
   - Upgraded `Pm4AdtPatchSupport.cs` to write authentic PCA Yaw rotations directly into `MODF` (`Rotation.Z`) rather than writing zeros.
   - Fixed zero-count display in `Pm4ColorCodingService.cs` and `Pm4OverlayScene.Reports.cs` so legend entries dynamically tally visible objects per category.
6. **Validation**: 183/183 PM4 unit tests pass (`WowViewer.Core.PM4.Tests`); full solution `WowViewer.slnx` builds with 0 errors; `Verify-UiReference.ps1` passes (137 inventory rows, 127 registry entries, 0 errors, 0 warnings). Receipts in `specs/289-pm4-3d-footprint-and-direct-matching/evidence/`.

**Spec 288 Implementation Complete & Verified (2026-11-01):** [Spec 288 PM4 Overlay Modernization, Color-Coding & Sidebar Refactor](../specs/288-pm4-overlay-refactor-and-color-coding/spec.md)
Phases 1–4 complete with receipts in `specs/288-pm4-overlay-refactor-and-color-coding/evidence/`:
1. **Color-Coding & Palettes (Phase 1)**: Pure algorithmic `Pm4ColorCodingEngine` and `Pm4ColorCodingService` implementing Golden-Ratio Instance Clustering (fixing the doodad mono-red bug where `MSUR._0x1C == 0`), Semantic Surface Role classification, 4-stop continuous perceptual Elevation Gradient, and Curation Match Status. Default mode set to InstanceCluster.
2. **Spatial Index & Fast Ray Picking (Phase 2)**: `Pm4SpatialMath` and `Pm4SpatialIndex` with hierarchical ray-picking (tile AABB -> object AABB -> Möller-Trumbore triangle) executing in $\le 0.05$ ms, and $O(1)$ 3x3 tile neighborhood query eliminating UI thread selection hangs.
3. **Mesh Renderer & Per-Tile GPU Caching (Phase 3)**: Extracted 470 lines of inline PM4 mesh, bounds, node, and flush rendering out of `WorldScene.Render()` into `Pm4OverlayMeshRenderer`. Per-tile GPU VAO/VBO caching eliminates per-frame CPU vertex matrix transforms for static tiles. `WorldScene.cs` reduced from 5,111 to 4,662 lines; `Pm4OverlayScene.cs` reduced from 2,080 to 1,789 lines (all files $\le 2,000$ lines, AGENTS.md §10).
4. **Modernized Sidebar Workbench (Phase 4)**: Created `Pm4WorkbenchSidebarService` replacing legacy broken sub-tabs with unified vertical `SharedUiWidgets` sections: Interactive Color Palette with clickable color swatches and active count badges, streamlined Surface Class Filter Matrix (`MSUR.0x00`) with quick-filter presets, and live alignment/duplication controls.
5. **Verification**: 177/177 PM4 unit tests pass (`WowViewer.Core.PM4.Tests`); `WoWViewer.csproj` builds with 0 errors; `Verify-UiReference.ps1` passes (127 controls, 0 errors, 0 warnings).


**PM4 Selection UI Hang Fix (2026-10-31, commit `0a18af2a`):**
Resolved AppHangTransient on PM4 surface click (which previously froze the UI message loop for >5-10s causing Windows crash dialog).
1. TryGetSelectedPm4RegionInfo in Pm4OverlayScene.Reports.cs: Cached output by selected object key with explicit invalidation on selection changes; restricted peer search to immediate 3x3 tile neighborhood around the selected object; replaced exhaustive NearestPointDistance position-ref iteration with direct bounding-box metadata lookup from Pm4OverlayObject.
2. Pm4OverlayScene.Selection.cs: Added tile-level AABB ray intersection test to TryPickPm4ObjectByRay to cull entire non-intersecting tiles before looping objects.
3. Pm4OverlayScene.cs: Added camera distance culling to ShouldRenderPm4Tile and restricted EnumerateVisiblePm4OverlayObjects to rendered tiles.
4. Validation: WoWViewer.csproj builds with 0 errors; Pm4DoodadDuplicationTests passes (3/3); Verify-UiReference.ps1 passes (127 controls, 0 errors/warnings).


**Spec 287 Implementation Handoff (2026-10-31):** [Spec 287 PM4 Live Matching, Burgundy MCCV Holes & Texture Layers Workbench](../specs/287-pm4-live-matching-and-curation/spec.md)
Phases 1–4 source implementation and test gates complete:
1. **Live PM4 Doodad Duplication**: Centroid alignment, bounding-box extent fitting, UniqueId allocation, live scene injection, undo stack, and Zarr v60 placement export. (3/3 unit tests pass).
2. **Burgundy MCCV Hole Detection**: Solid underside fog chroma detection ($R \ge 70, G \le 35, B \le 35, \sigma \le 18$), $\ge 2$ connected component cluster filtering, dynamic chunk `HoleMask = 0xFFFF` application, and GPU tile mesh rebuilding. (5/5 unit tests pass).
3. **Photoshop-Style Texture Layers Workbench**: GLSL `uLayerOpacity[8]` and `uAlphaMaskBlend` fragment shader controls, dedicated "Terrain Layers" tool window and View menu entry, and Ghidra client binary analysis tooltips for `MCLY.Flags` (UV anim, specular, cubemap, dynamic) and `EffectId`.
4. **24-Hour Skybox & Audio Immersion**: Removed night-only gate in `SceneAtmosphere.cs` enabling client skybox models across the full 24-hour day cycle; preserved authored DBC/LIT zenith and horizon colors; defaulted `WorldAudioRuntime` triggers and zone music streaming to active with interactive mute checkbox.
5. **UI Hygiene**: `Verify-UiReference.ps1` passes with 0 errors, 0 warnings (127 controls registered). Solution builds cleanly (0 errors).
[Phase 1 receipt](../specs/287-pm4-live-matching-and-curation/evidence/phase1-duplication-source.md),
[Phase 2 receipt](../specs/287-pm4-live-matching-and-curation/evidence/phase2-mccv-holes-source.md),
[Phase 3 receipt](../specs/287-pm4-live-matching-and-curation/evidence/phase3-layers-source.md),
[Phase 4 receipt](../specs/287-pm4-live-matching-and-curation/evidence/phase4-sky-audio-source.md).
Owed: Operator runtime gates T008, T014, T022, T028.

**Next lane — CASC streaming (operator-directed 2026-10-22; Spec 275 Phase 0 / CASC-1, uncommitted):**
TACTSharp's local read maps the whole `data.NNN` archive per file (250–385 µs fixed, ~1.5× at 8 threads).
`LocalCascArchiveReader` + `CascStorage` fast path (cached handles, stock fallback, first-64 shadow check,
`PARP_CASC_FAST_READS=0` to disable) cut that: real `wow_classic_beta` Azeroth set, 9,401 files
byte-identical, sequential 2.9×, parallel ×8 6.1× (`inspect casc bench --read-ab-only`). New Perf panel
lines "CASC reads"/"loader misses". 9 new tests; full suite = the 7 known failures (one unexplained
extra flake in run 1). **Owed:** operator capture T027 (viewer streaming with/without the env switch);
only then CASC-2 (prefetch priority, coalescing, re-prefetch after eviction) — none started.
[Receipt](../specs/275-asset-streaming-mpq-casc/evidence/casc1-local-read-path-2026-10-22.md).

**Operator's first real run (2026-10-22, 3440×1392) → U0c fix (committed with U1; operator re-check T057 owed):**
dock panels showed only their page selector and the status bar was clipped. Causes: the dock's
full-height splitter `InvisibleButton`+`SameLine` made ImGui's line height the whole window (all
later widgets landed below it), and the 22 px status region was shorter than its button row plus
8 px window padding. Fixed: scrolling child body + last-submitted overlay splitter; footer bars
sized from the measured control row (`UiLayout footerRowHeight`) with tight padding; Left Bar /
Right Bar buttons removed (Tab hides chrome, `[`/`]` toggle rails); buttons at standard height.
90/90 focused tests, viewer compiles (scratch output; the running viewer locks `bin\`), headless
ImGui probe in `specs/286-…/evidence/imgui-layout-probe/` (ImGui.NET 1.89.9.3). Pure-policy tests
cannot see ImGui line-height/padding bugs — verify ImGui layout changes with that probe.
[U0c receipt](../specs/286-view-controls-userguide/evidence/u0c-dock-and-footer-2026-10-22.md).

**Implementation handoff 2026-10-23 — branch 286-view-controls-userguide:** [Spec 286 U0 + U0b + U0c + U1 + U2](../specs/286-view-controls-userguide/tasks.md)
source/tests landed. Footer regions fit their actual client rect with no scrollbars; FPS hover
discloses the existing counters. U0b docks Inspector/World/Model/PM4 under the minimap, resizable
and persisted, chat starts collapsed. **U1:** one `ViewPreferencesService` (owned by
`ViewerUiManager`) owns user text size; View, Settings and the loader all use it and only the
manager writes `FontGlobalScale`. View holds Visual Options..., Camera and Fog..., a Text Size
submenu and three named resets (**Reset Text Size Only / Reset Layout Only / Reset Layout and Text Size**).
**U2 (2026-10-23):** session-owned `MapsActivationCoordinator` and pure `MapsActivationPolicy`
govern client activations (MPQ, CASC, WC3); Maps opens once per successful activation (FR-007/008)
including 0-map empty state (FR-009); ordinary refresh and user close respected; auto-dismisses on map load;
11/11 U2 tests pass, 101/101 cumulative focused pass; solution build 0 errors; registry 119/119, 0 warnings.
T001–T007, T049–T056, T009–T015, and T017–T023 checked. Next: operator runtime proof —
T008 (SC-008/010/011), T057 (SC-012), T016 (SC-001–003), and T024 (SC-004/005) — then U3
(Contextual selected identity and hover).
[U0 receipt](../specs/286-view-controls-userguide/evidence/u0-source-tests.md),
[U0b receipt](../specs/286-view-controls-userguide/evidence/right-dock-and-chat-2026-10-22.md),
[U0c receipt](../specs/286-view-controls-userguide/evidence/u0c-dock-and-footer-2026-10-22.md),
[U1 receipt](../specs/286-view-controls-userguide/evidence/u1-source-tests.md),
[U2 receipt](../specs/286-view-controls-userguide/evidence/u2-source-tests.md),
[U2 route audit](../specs/286-view-controls-userguide/evidence/u2-activation-route-audit.md).
[Renderer plan](../specs/249-epic-renderer-performance-and-correctness/plan.md)
owns evidence-gated sky/liquid work; [flora plan](../specs/259-ground-effects-and-wmo-detail-doodads/plan.md)
owns distance/density/fade. Preserve Spec 273/276 work, culling quarantine and reverted AC-009.
Operator owns real proof; do not fabricate parity or reapply tint from the fade investigation.

**Planning-only handoff, 2026-10-21**: [Spec 285 v65 interchange](../specs/285-v65-terrain-interchange/plan.md)
([62 pending tasks](../specs/285-v65-terrain-interchange/tasks.md)) rebases [Spec 284 converters](../specs/284-map-object-converter-revitalization/plan.md)
([74 pending tasks](../specs/284-map-object-converter-revitalization/tasks.md)) on current data-harvester → decoded dataset → modern target export.
v65 retains native terrain and **typed placement metadata**; excludes object geometry/masks; **CASC reads required**.
Unknown terrain bytes/metadata stay indexed; all referenced terrain textures retain source + native RGBA PNGs, cell/layer mappings and appearance inputs for later viewing/models.
Every-layer MCAL/MCLY reports retain raw/decode/effective opacity metadata; converter output reuses actual viewer New Map ADT/WDT/WDL writers, with field-capability audits and independent report readback.
Python owns Zarr I/O; old Converter is unverified algorithm reference, not a functional dependency/parity gate.
Next chat starts Spec 285 coverage/contracts, then Spec 284 export gates. No implementation/runs; preserve Spec 283/UI WIP. Heavy/visual proof operator-owned.

**Spec 283 UI correction (2026-10-22):** anchored pictogram rails now open small closeable tool windows; full-name tooltips, no two-letter control labels, and no Spellbook. Maps, File Browser, Diagnostics, camera/fog and Editor pages retain their existing owners. Chat log uses its full frame width; minimap/chat no longer inherit saved sidebar widths (layout v5). Spawn-preview show-flag guard and footer sizing fixed. Solution build 0 errors; 13 focused tests pass; reference check 125 rows/116 entries, no warnings. [Receipt](../specs/283-v070-ui-presets-and-explore/evidence/action-bars-and-popup-windows-2026-10-22.md). **Not runtime/visual proven**; operator icon/window walk, full presets/Explore and separate converter work remain open.

Last updated: 2026-10-21 · **Spec 280 UI refactor implemented; startup crash fixed in code, operator smoke owed**:
- **Spec 280 UI refactor — implementation landed, ledger reconciled 2026-10-21** (branch
  `280-ui-rework`; spec, plan, tasks, audit and 18 receipts under
  [specs/280-ui-audit-refactor/](../specs/280-ui-audit-refactor/)). **Landed:** one
  `ViewerUiManager` owns every chrome window (one token set, one draw order, one input rule, one
  popup registry); three destinations (Viewer / Editor / Archaeology) behind a slim bottom bar with
  an Options sub-menu; top menu reduced to File / View / Help; the whole Tools menu and all
  export/import trees deleted with their actions moved to their owning destinations; 7
  compatibility page enums, `Canonicalize`, the legacy tab redirects and `ShellLayoutService`
  deleted, with versioned `output/settings/ui-layout-v1.json` persistence; the 12 duplicate
  floating windows deleted (Log and Perf only remain); archaeology's legacy 5-tab strip, floating
  UniqueId window and Quick launchers deleted (the destination is now **5** pages, PM4 is a
  cross-destination button); Editor sections renamed to their real owners; PM4
  matching/alignment/reconciliation removed and moved to [Spec 281](../specs/281-museum-adt-patcher/spec.md).
  **Where the code lives:** `src/viewer/WoWViewer/Workbench/Services/UiManager/`
  (`ViewerUiManager.cs`, `.Input.cs`, `.Persistence.cs`, `UiLayout.cs`, `UiPopupRegistry.cs`,
  `UiTokens.cs`, `UiControlRegistryService.cs`), plus the content-only providers
  (`Chrome/`, `Navigator/`, `WorkbenchPanels/`, `MainMenu/`, `FileOperations/`).
  **Receipts:** [final receipt](../specs/280-ui-audit-refactor/evidence/final-receipt.md),
  [tasks ledger](../specs/280-ui-audit-refactor/tasks.md),
  [audit §0.11 / §2.1](../specs/280-ui-audit-refactor/audit.md).
  **Verified 2026-10-21:** solution build 0 errors; tests Core 7 failed / 1822 passed / 1 skipped
  (the 7 are the pre-existing format/data set from the P0 baseline, none UI), Editor 91/91,
  Curation 39/39, PM4 165/165; `eng/ui/Verify-UiReference.ps1` exit 0
  (`inventory_rows=125 distinct_labels=117 registry_entries=117`, 0 errors, 0 warnings) and a
  117-control `docs/WoWViewer/UI-REFERENCE.md` generated from `eng/ui/ui-controls.json`.
  **Startup crash found and fixed in code:** `ViewerUiManager..ctor` captured
  `host.MinimapAndStatus.DrawStatusContent` before `ViewerApp` assigned `_minimapAndStatus`
  (`ArgumentException: Delegate to an instance method cannot have null 'this'`). The manager now
  resolves the provider at draw time (`ViewerUiManager.cs:24-30`). The viewer has **not** been
  launched, so startup is still operator-owned proof.
  **Operator-owned / open (T092, T093):** no real-client run, no visual/FPS/audio proof, no
  in-viewer export/import proof. The `Ctrl+K` command palette (T106) was not implemented. The
  type scale (T100) is defined but not applied because the viewer uses one ImGui font. Bespoke
  styling remains outside the five manager-owned surfaces (34 files / 179 hits) and the
  documentation `WC3 Beta` rename plus the final dead-control sweep are not done.
  **Operator feedback 2026-10-21 (first real run) and fixes:** the first launch showed a 4K UI
  with 280 px sidebars, a 12 px font, a floating thumbnail minimap, a chat that resized off-screen,
  bottom bars below the client area, and sidebars whose long lists pushed everything else out of
  reach. Fixed in code: `UiLayout.ScaleFor` scales every metric and the font by display height
  (1.0 at 1080p, 2.0 at 4K); the layout uses the window client size instead of ImGui's
  `DisplaySize` (which could be the framebuffer size); the minimap is **docked at the top of the
  right sidebar by default** and sized to the sidebar; an **Options** button sits in the chat
  header; **Fog Start/End** are back in the Workspace section; the left sidebar order is
  Workspace → destination sections → Diagnostics → World Overview → File Browser (long nested
  lists last); layout persistence bumped to `ui-layout-v2.json` so the stale tiny layout is
  ignored.
  **Chat dock correction (operator report 2026-10-21):** the movable/resizable chat was rejected —
  it covered the UI and could not be moved back. The chat is now a **fixed docked panel** spanning
  the scene between the sidebars, bottom-anchored, drawn with `NoMove | NoResize`
  (`UiLayout.cs:71-80`, `ViewerUiManager.cs:159-161`). The persisted floating rect
  (`ChatWidth`/`ChatHeight`/`ChatX`/`ChatY`) is deleted and the layout document is bumped to
  `ui-layout-v3.json`, so a stale floating rect cannot be restored. The Options panel now anchors
  to the right sidebar's left edge instead of the chat rect.
  **Hover-jump fix (operator report 2026-10-21):** hovering the right sidebar made the Inspector
  content collapse and the controls move. Cause: scene hover picking ran every frame even when the
  pointer was over a sidebar, so the hovered asset cleared, the Inspector payload changed, and the
  controls moved under the cursor. Hover picking now runs only while
  `CanSceneConsumeMouse` is true (`ViewerApp.cs:1039-1047`).
  **Spec 282 (operator-originated 2026-10-21):** real client cursor model
  (`Interface\Cursor\cursor.mdx`/`.m2`) as the in-scene pointer, plus precise PM4 picking.
  [Spec](../specs/282-client-cursor-and-pm4-pick/spec.md) ·
  [Receipt](../specs/282-client-cursor-and-pm4-pick/evidence/receipt-2026-10-21.md).
  **PM4 picking fixed:** the pick was AABB-only, so a large object's box could steal a click; it
  now rejects on the AABB and then tests the actual PM4 triangles (Moller-Trumbore), selecting the
  nearest triangle under the ray (`Pm4OverlayScene.Selection.cs`, `Pm4OverlayScene.cs`).
  **Client cursor model implemented:** `SceneCursorRenderer` dispatches candidates by extension —
  MDX through `MdxRenderer`, M2 through `WowViewerM2RuntimeBridge` + `M2Renderer` with skin
  resolution and an Alpha-era inline-geometry fallback — and `ViewerApp.OnUpdate` re-points it at
  the live data source (it was constructed before any client opened and never re-pointed, which is
  why the authentic cursor never loaded). Options > Objects reports the resolved model or the load
  error and offers `Reload cursor model`. Build 0 errors; `Verify-UiReference.ps1` PASSED; PM4
  165/165; UI doc tests 3/3. **Not proven:** no real client cursor asset has been loaded in a run,
  and PM4 selection accuracy needs an operator spot check.
  **Next bounded step:** operator relaunches, opens a client, and confirms the cursor model and PM4
  click accuracy.
- **Spec 280 UI audit (2026-10-20, operator-directed):** the full nesting map is in
  [audit.md](../specs/280-ui-audit-refactor/audit.md). File menu 18 entries/3 levels; View 15/8
  duplicates; Tools 36/5 levels; Editor 4 pages/13 sections (3 host archaeology-named content);
  Archaeology 6 pages with **four parallel navigation systems** (destination strip, legacy 5-tab
  strip, floating UniqueId window, Quick launchers) and a 4-level Cartography tree; 12 Tools >
  Panels floating windows duplicate destination pages. This audit was the implementation contract
  for the 2026-10-21 pass above.
- **Known bugs to fix (operator-reported 2026-10-20, not yet fixed):**
  1. **Sky / skyboxes** — no proper skybox support. Use Ghidra client analysis for 3.3.5 / 4.3.4 sky algorithms. Add
     Quick-tab controls with a list of known-good settings from DBC/DB2 plus the current blank-sky
     default for simple viewer use.
  2. **Population model fallback** — 0.5.3 and 3.3.5 population tries to load `humanmale.mdx/m2`
     for every quest-giver / gossip NPC. Suspect the bark/`bunnies` handling is not selecting the
     real creature display.
  3. **3.3.5 creature textures** — all 3.3.5 creature textures are missing; 0.5.3 works. Likely a
     texture-path or BLP-version resolution difference between the two clients.
- **PM4 open-question follow-up (2026-10-20):** `MSUR._0x10 = -dot(normal, v0)` in raw MSVT space
  (518,080/518,092, median error 0); `MPRR` first two records are ordered spatial bounds in a
  768×768 tile grid (3,171,133/3,171,912, 99.98%); `_0x1c` matches a placement Z anywhere in the
  map for 69.7% of nonzero surfaces; `MSHD` spans vs raw placement spans r = 0.39–0.58;
  `MPRL._0x14` is not the run index, node level, group size, link count, or run length. Receipt:
  [phase-5](../specs/279-decode-pm4-format/evidence/phase-5-open-questions-2026-10-20.md).
- **PM4 UI consolidation + matcher removal (2026-10-20, operator-directed):** PM4 now lives in the
  **Inspector window** as a tab (Inspector, World, Model, Settings, PM4). The tab exposes the eight
  PM4 pages (Overlay, Selection, Correlation, Info, Reconcile, Alignment, Outliner, Object Groups).
  The duplicate PM4 content in the Editor Placement page, Editor PM4 workspace, and Archaeology PM4
  sub-tab is replaced with a navigation shortcut. The hover matcher and the workbench object-match
  report are removed; matching now goes through the patched-ADT placement services (`Find WMO
  Match` / `Shape Search`). **Human curation:** `Pm4ObjectMatchStore` writes
  `output/pm4_object_matches.json`; the PM4 Selection tab has a Match this object section (path
  field, save/clear, saved-match list, Suggest nearest model), and the Rosetta object library has a
  Match to selected PM4 object button. Viewer build 0 errors.
- **Spec 279 (PM4 comprehension + reference):** 16 parsed chunks, 17 open question groups.
  [Reference](../docs/research/pm4-format-reference.md) - [spec](../specs/279-decode-pm4-format/spec.md).
  **2A** raw MSLK +0x01 = source-edge ordinal; **2B** +0x04 is not a unique edge ID; **MSHD**
  five of eight words always zero; **M2 partition** isolates M2-only surfaces but the per-object
  split is **negative** (ownership remains U); **MPRR** is a quadtree index over MPRL points;
  `MPRL.Unk04` = `MSLK.GroupObjectId` (98.97%); object-to-model join 85.1% within 25 units;
  MPRL points are not MSPV path vertices. Reference consolidated 2026-10-20; opt-in `modelPriors`
  added to `Pm4AssetMatchScorer`. Receipts: [2A](../specs/279-decode-pm4-format/evidence/phase-2a-edge-ordinal-2026-10-20.md),
  [2B](../specs/279-decode-pm4-format/evidence/phase-2b-edge-identity-2026-10-20.md),
  [MSHD](../specs/279-decode-pm4-format/evidence/phase-3-mshd-2026-10-20.md),
  [M2 partition](../specs/279-decode-pm4-format/evidence/phase-3-m2-partition-2026-10-20.md),
  [path nodes](../specs/279-decode-pm4-format/evidence/phase-3-path-nodes-2026-10-20.md),
  [model prior](../specs/279-decode-pm4-format/evidence/phase-4-model-prior-2026-10-20.md).
  Full detail: [archive/2026-10-21-pre-spec280-active-context.md](archive/2026-10-21-pre-spec280-active-context.md).
- **PM4 structural findings (2026-10-20):** 616 files probed; exact MSUR-to-MSLK windows; packed
  link target = tile X/Y + surface index; 15,773 cross-tile links; 32 focused tests pass.
  Unsolved: editor-authored path-node identity, remaining raw fields, M2 object partition.
  [Ledger](../specs/270-pm4-object-association-ledger-and-placement/evidence/structural-foundations-2026-10-20.md).
- **Terrain Texture & Blending Fix (WoW 3.3.5 / Modern ADTs)**:
  - Completely resolved chunk-level checkerboarding, pitch black/dark green slopes, and corrupting layer textures across 3.3.5 maps (Dark Portal, Crystalsong, Dragonblight, Valgarde 18_23).
  - Fixed MCNK subchunk scanning desync in `Mcnk.cs`: `Header.SizeMcsh` is the raw payload (512 bytes), removed erroneous `- 8` which caused an 8-byte advance overrun into shadow data on 71 chunks; prioritized `Header.SizeMcal - 8` when declared size is 0, 4, or 512 to prevent overruns; adjusted walk loop to `while (remaining >= 8)` so 0-payload chunks (`MCSE`) don't break exact walk tracking.
  - Fixed format profile resolution: `FormatProfileRegistry.cs` now routes all `major == 3` clients to `AdtProfile33512340`; relaxed `TryParseBuild` for 2-4 parts; added `TryDetectLichKingAdt` to `StandardTerrainAdapter.cs` checking `MH2O`, `ofsMcnr == 0x520`, or `sizeMcal > 0`.
  - Restored sequential `mix()` layer blending with accumulated alpha tracking in `TerrainRenderer.cs` (strict ASCII).
  - Real-client validation on `Northrend_23_18.adt`: 256/256 chunks parsed with 0 errors; 6/6 `McnkSubchunkWalkTests` pass.
- **PM4 M2 Surface Correlation & Unauthored Placement**:
  - Added PM4 collision surface and M2 mesh correlation services (`Pm4SurfaceCorrelationExtractor.cs`, `Pm4SurfaceCorrelationMatcher.cs`, `Pm4UnauthoredPlacementResolver.cs`).
  - Added M2 collision and attachment metadata extraction in `M2ModelReader.cs` and `M2ModelDocument.cs`.
  - Integrated world dialogue and spawn streaming refinements.

**Older release and regression detail** (2026-10-12 to 2026-10-20) is archived in
[archive/2026-10-21-pre-spec280-active-context.md](archive/2026-10-21-pre-spec280-active-context.md):
the 2026-10-20 committed feature sets, the WC3 browser unification (Spec 272), the release
packaging fix, the ground-effect tint rollback, the 1.60.1 taxi mount fix, the open WMO area
lookup issue, the M2 texture-resolver rollback, and the M2 camera origin/facing work. Live
constraints from those notes: do not reintroduce tint propagation without a diagnosed approach;
do not refactor `ReadStringArray` without a narrowly scoped proven change; the WMO area lookup
and the `DD21` magic remain unexplained.

**Next on `v0.6.1-dev` (fresh chat start here):**
1. **PM4 next:** test raw +0x04/subtype and point/reference chunk relationships against the now
   correctly tile-addressed surface graph. Identify authored path-node layer independently; compare
   any proposed M2 ownership partition with Museum copies/scales. Large export remains operator-owned
   ([usage](../specs/270-pm4-object-association-ledger-and-placement/evidence/map-context-usage.md));
   lightweight whole-map structural summaries have already run and must not be described as unrun.
2. Operator runs the viewer on 1.60.1 inside `MD_CryptOneRm.wmo` and provides `[WMOAreaLookup]` diagnostics; do not implement a key/fallback change before reading the matching row IDs.
3. Operator confirms character M2s render again after the `ReplaceableTextureResolver` rollback; FileDataID texture support remains unresolved.
4. Operator verifies the actual 1.60.1 `CinematicCamera` row for FileDataID 116912 and plays/warmups FlybyUndead on base Azeroth with intended phase layers; record whether origin/facing and tile `(28,28)` agree.
5. Operator-requested public format docs: prepare wowdev.wiki-style articles for the 0.5.5 prototype ADT, WMO v14, MDX, Alpha WDT, and a substantial PM4/PD4 refresh. Ground each in repo evidence; distinguish verified facts from open interpretations.
6. Spec 271 Phase B — integrated Help + image guide (operator captures screenshots; agent prepares shot list and `HelpGuideService`).
7. Spec 270 open gap — objects on tiles adjacent to PM4-covered tiles that exist in museum data are still missing from restored tiles (operator report 2026-10-11).
8. Resolved: `Mcnk.cs` subchunk scanner desync fixed (MCSH raw size 512, MCAL header size fallback, and loop boundary). Alpha/shadow offsets now parse 100% cleanly without tool-local offset repairs.
9. Operator smoke owed on regenerated `output/restored_development` (alpha masks, no orange tint, 63_5 huts flush).

The repeatable procedure is now documented in [Development Map Restoration](../docs/WoWViewer/development-map-restoration.md);
that guide does not close the operator visual smoke item above. The README uses its native Markdown support matrix
instead of a separate graphic; worklog/spec and old release-note references were removed from viewer-facing documentation.

## Fresh-chat route

1. [Spec status](../specs/STATUS.md) — 7 active epics (248–254), all **triage pending**.
2. This compact handoff.
3. [TRIAGE.md](../specs/TRIAGE.md) — the operator's Want / Drop / Later decisions.
4. Only then one epic's `spec.md` → `plan.md` → `tasks.md`, and only the archived design documents
   its `plan.md` adopts for the selected item.

[Progress ledger](progress.md), [memory archive](archive/README.md) and `specs/archived/` are
on-demand history, never default reading.

## Current lanes — operator P1s (both marked Want 2026-09-23)

| Lane | Code | Owed (operator) | Next agent step |
|---|---|---|---|
| **R-10** modern-data lighting perf (Epic 249) | T002, T005, T006 landed 2026-09-23; **T004's per-placement WMO gate reverted 2026-09-27** (operator saw <1 FPS; T009) | check FPS after the revert on `wow_classic_beta` 1.60.1 `Azeroth` (0 WMO placements batched while lights exist); T003/T007 captures | redesign WMO instancing only after a before/after capture |
| **R-39** CASC minimap streaming cost (Epic 249) | code landed and **reverted** 2026-09-27 (operator: performance worse); code = `c2e9517` | say whether to add per-tile read/decode/upload timing + GC counts, then capture on a CASC client | no minimap change before measurement |
| **Spec 256** modern-data streaming & M2 submission | P0 `77f6f02`, minimap `b15e553`, P1 `4f1e6b4`; capture showed ≈ 57 ms/M2 in texture work → P2a shared textures `c9cf7b6`, P2b off-thread decode `6a326d3`, P2c compressed DXT + authored mips `bad4884`; P2b fallback fix `328866d`; missing textures → error texture, no search `983c8ac`; D WMO doodad loads + shared cache `b09654b`; P3 native M2 instancing `2c609f3`; Perf panel out of PM4 service `93d7461`; A animation: cached key frames + GPU skinning `e437673`; T060/T092 file splits landed (`WorldAssetManager.cs` 1,576 lines, `WmoRenderer.cs` 1,937 lines) | T102 capture (MdxAnimation, entering animated areas) + T042 visual A/B (Utilities > Perf > Frame history > Submission efficiency) | operator captures (T015, T022, T042, T081, T102) |
| **Spec 257** unified video recording & automation | **Complete 2026-10-02**: fixed taxi node stop-crash (null deref), unified `RecordingCoordinatorService` in `Recording/`, CLI video recording automation (`--record-taxi-route`, `--record-camera-path`, `--record-feature-tour`), camera path & feature tour integration | Operator smoke test of manual and automated recording | None (code & spec complete; receipt in `evidence/receipt-spec257.md`) |
| **Spec 258** taxi playlists & marketing showreel tour | **Complete 2026-10-02**: multi-route taxi playlists, continuous multi-segment recording, dynamic showreel telemetry HUD, large centered cinematic zone transition banners, landmark callouts; strict tour-mode viewport gating | Operator smoke test of playlist playback and showreel overlays | None (code & spec complete; receipt in `evidence/receipt-spec258.md`) |
| **Spec 273** ground-truth renderer performance | **Implemented & Verified 2026-10-16**: ground-truth 3.3.5a Ghidra client analysis refinements: WMO portal occlusion & interior culling (R-15), decoupled shell instancing (R-10), spatial M2 doodad submission, and terrain shader constant alignment; receipt in `evidence/receipt-spec273.md` | Operator in-viewer smoke owed | None (code complete) |
| **Spec 274** minimap hillshade correlation & domain translation | **Active 2026-10-16**: Batch-export clean paired dataset across Zarr v3 map revisions (real minimap, synthetic DXT1 hillshade, WDL lattice, ADT heightmap), train lightweight style-transfer network for object stripping and texture normalization, and correlate clean hillshade directly with 3D terrain implication data | Operator review of Spec 274 plan | Phase 1 implementation (T001–T004) |
| **Spec 259** ground effects & WMO detail doodads | **Complete 2026-10-02**: dynamic grass/foliage on terrain and WMO surfaces (`MDDL`), client slope ($Z \ge 0.4$) & MCCV rules from recent wowdev.wiki disclosures, 1.60 CASC support, and instanced GPU rendering; UI controls in RenderQuality; receipt in `evidence/receipt-spec259.md` | Operator visual smoke on terrain maps and WMO surfaces | None (code & spec complete) |
| **Spec 260** engine diagnostics & hitch telemetry HUD | **Complete 2026-10-02**: real-time engine pipeline metrics (M2 instancing %, flora count, WMO batches, ADT chunks, RAM & draw calls), rolling hitch detection with dominant stage attribution, flashing crimson/amber alert badge, multi-stage expanded timing panel, UI toggles, and CLI flags; receipt in `evidence/receipt-spec260.md` | Operator smoke test during recording and live viewport preview | None (code & spec complete) |
| **Spec 261** WoW Forever terrain hole fidelity | **Complete 2026-10-04**: verified ADT v18 format conformance, 64-bit high-res hole decoding at MCNK offset 0x14 with Little-Endian mapping (`cellY * 8 + cellX`), 100% seam boundary continuity, real CASC test cases for WoW Forever 1.60.1 / 11.2.7 (Deathknell church crypt and graves); receipt in `evidence/receipt-spec261.md` | Operator visual smoke on WoW Forever (1.60.1) Deathknell/caves/crypts | None (code & spec complete) |
| **Spec 262** minimap residual model & 3D fractal brush reconstruction | **Complete 2026-10-07**: Photometric Blinn-Phong specular solver, Rosetta catalog vision indexer, ComfyUI SAM 3.1 sieve & inpainting, closed-loop shadow difference $\Delta S$, discrete 3D fractal brush discovery & matching pursuit, and $\ge 75\%$ terrain reconstruction benchmark; operator smoke verified live on RTX 4070 Ti SUPER (`cuda:0`, 12.6 GB VRAM free, 2,201 segmented pixels, 97.37% geometric fidelity); receipt in `evidence/receipt-spec262.md` | Operator smoke test passed | None (code & spec complete; 28/28 Python + 218/218 C# tests green) |
| **Spec 263** 1.60 MCCV terrain shadow validation & synthesis | **Complete 2026-10-07**: empirical cross-validation between 1.12 bare minimap residual shadows and 1.60 MCCV vertex data (145 vertices BGRA: 580 bytes), geographic sampling stratification (Westfall/Elwynn/STV as Tier 1 baselines vs composite cities/Wetlands/Badlands as Tier 2 excluded), bidirectional ADT synthesis & injection bridge (`MccvTerrainShadowService.cs`, `mccv_shadow_comparator.py`, `v60_compare_mccv_residuals.py`); road-safe sieve preserving 2D alpha texture splats, labeled 4-panel diagnostic sheets with headers/legends, and 3D Wavefront OBJ + binary glTF 2.0 (`.glb`) mesh exports for side-by-side surface comparison in Windows 3D Viewer/Blender; verified NCC 0.9252 (Westfall) / 0.8778 (STV) and 100% / 87.5% ridge coincidence; receipt in `evidence/receipt-spec263.md` | Operator smoke verified | None (code & spec complete; all 6 C# + 11 Python tests green) |
| **Spec 264** authentic development shadow-to-height & Rosetta object placement | **In Progress (Active Dead End on Poisson SFS)**: Shape-from-shading / Poisson integration hits a mathematical dead end: land-only Pearson $r = 0.3345$, land MAE $= 31.74$ yds, max error $= 179.30$ yds (prior $0.94$ metric was skewed by flat $Z=0$ water plane). Pure shading integration cannot resolve macro-elevation or texture-albedo confusion. Next step: train an end-to-end multi-scale elevation network directly mapping authentic minimap RGB $\to$ absolute Z yards on the 198 development tiles without relying on test-time ground truth. | Operator review | Build deep elevation network on 198 authentic development tiles |
| **Spec 265** two-stage residual elevation refinement & scale calibration | **Active 2026-10-08**: Cascaded Stage 2 residual refiner predicting $\Delta Z = Z_{\text{gt}} - Z_{\text{initial}}$ to uncompress squashed mountain spans (39 yds $\to >200$ yds on `development_16_33`) and correct systematic texture/altitude offsets. Spec Kit authored (`specs/265-residual-elevation-refinement-and-scale-calibration/`); Phase 1 residual dataset extraction underway. | Operator review | Implement Phase 1 residual dataset caching |
| **Spec 266** WDL trestle elevation reconstruction & lineage synthesis | **Complete 2026-10-08**: Modern lean successor to Feb 2026 v7 WDL trestle model (4.7M params vs 100M+); synthesizes missing development WDL lattices from Northrend 3.0.1.8303 prototype matches (2,035 tiles transferred); restores authentic $>250$ yd mountain relief (measured 424.8 yds on `development_16_33`); 100% upward CCW mesh normals (+Z OBJ, +Y GLB); validation MAE 2.87 yds, $r=0.9943$. Fixed oversized sieve: replaced AABB bounding boxes with tight oriented OBB footprinting ($31\times$ reduction on doodads, 0 painted text masked, PM4 collision extraction disabled). Batch generated all 2,252 development LK ADTs with full multi-band SfS Poisson meso-relief, 220° solar normals, and ridge crest spines in 134.7s (16.7 tiles/sec) in `output/development_reconstructed_lk/` with updated `development.wdt`. Receipts in `specs/266-wdl-trestle-elevation-reconstruction/tasks.md`. | Operator smoke test | None (code & spec complete; 6/6 tests green) |
| **Spec 267** Trellis.2 / Pixel3D ComfyUI ADT refinement | **Active / Phases 1-3 Complete 2026-10-08**: Discovered running ComfyUI models (Trellis.2 / Pixel3D on RTX 4070 Ti SUPER, 2,933 nodes cataloged); created `comfyui_parp_nodes` package with `WoW_AdtLoader`, `WoW_ObjectSieveConditioner`, `WoW_HardZCalibrator`, `WoW_MeshExporter` (5/5 tests green). **Complete Chunk & Data Transfer**: Monolithic 3.3.5 ADT preservation for all 2,252 tiles — keeps 100% of authentic chunks (`MTEX`, `MCLY`, `MCAL`, `MMDX`, `MWMO`, `MDDF`, `MODF`, `MCRF`, `MCSE`, `MCSH`, `MCLQ`/`MH2O`, `MFBO`, `MCCV`). Cleaned 447 stray `_obj0`/`_tex0` companion files from output root, enforcing `AdtProfile_335` so WoWViewer loads all textures and objects. **Tileset Texture Stripping & De-Bumping**: Decomposed minimap luminance into structural illumination ($\sigma=4.5$) vs tileset texture noise ($T = I - S$); gated ridge detection to genuine macro relief ($|\nabla Z| \ge 0.08$); smoothed meso SfS relief ($\sigma=2.0$, $1.0$ yd amplitude) and wrote smooth surface normals into `MCNR`. Batch generated all 2,252 monolithic ADTs in 791.0s with updated `development.wdt`. | Operator reload & visual smoke test in WoWViewer | Complete Phase 4 end-to-end ComfyUI workflow JSON |
| **Spec 268** quilt canvas terrain reconstruction & MCAL deciphering | **Phases 1-5 Complete 2026-10-09**: Continuous multi-tile quilt canvas stitching, 1px shared tie-edge boundary solver (C0/C1, $\Delta Z = 0.0000$ yds, normal alignment $0.9995$), minimap albedo de-mixing & dynamic MCAL layer stacking ($\le 4$ per chunk), authentic WDL global macro-trestle lattice interpolation, 36x inches-scale 3D fractal refiner (8 brush stamps + scars), monolithic LK ADT materialization, and textured 3D OBJ/GLB mesh exports; verified on alpine massif `(16, 32)`/`(16, 33)`, 2x2 cluster, and unauthored minimap-only tiles `(17, 13)`/`(17, 14)` via pure V7 foundation; all post-V7 models purged; receipt in `evidence/receipt-spec268.md`. | Operator smoke passed | Complete (2026-10-09) |
| [**Spec 269** v60 classic terrain Zarr ingestion & Felwood clean elevation training](../specs/269-v60-classic-terrain-ingestion-and-elevation-dataset/spec.md) | **Complete 2026-10-10**: Ingest Wrath 3.0.1.8303 terrain (37 maps, 4,448 tiles) via C# harvest engine into `output/datasets/v60/`, train `TrestleElevationUNet` on authentic Felwood canyons/indentations (120 epochs, MAE 4.62 yds, $r=0.8842$), and carve $-96.13$ yd canyon into unauthored quilt tile `(17, 14)`. Receipt in `evidence/receipt-spec269.md`. | Operator harvest & training runs | Complete (2026-10-10) |
| [**Spec 270** PM4 object association ledger & placement pipeline](../specs/270-pm4-object-association-ledger-and-placement/spec.md) | **Complete 2026-10-11**: Harvested 1,929 ground-truth objects into Rosetta Ledger (`pm4_object_association_ledger.json`). **Phase 4 & 5 Monolithic LK Pipeline**: Strict barrier preserving 255 hand-authored Blizzard tiles; 2D PCA Yaw; repaired Schlumpf index defect; restored unauthored PM4 tiles. **Phase 5 Multi-Source Placement Convergence & Magnetic Collision Fix**: Fixed model pivot origin vs bounding centroid Z offset in Strategy 0 and vertical attraction; preserved authentic authored rotations over synthetic PCA; deduplicated stacked instances with proximity gating (20 yds WMO, 5 yds M2); sampled 256x256 minimap PNGs into 256 145-vertex MCCV chunks via `AdtMccvImageSampler` (195 untextured tiles colored, 0 black tiles). Full map batch executed: 616 tiles processed, 1,654 WMOs, 64,301 M2s, 616 PASS / 0 FAIL round-trip. | Operator review & visual smoke in WoWViewer | Complete (2026-10-11) |
| **U-01** god-class decomposition (Epic 251) | **E1 + E3 + ViewerApp campaign landed 2026-09-25**: PM4 overlay → `Terrain/Pm4/` (`WorldScene.cs` 17,175 → 8,326); `ViewerApp` class 42,973 lines / 29 files → 2,512 / 6 (51 services + 3 static helpers under `Workbench/Services/`) | U01-T003 E1 smoke; U01-T007 / T013 smoke of every moved ViewerApp surface; U01-T009 E4 hover/click smoke; **Spec 255 T018** smoke of W0–W7 | E4 landed 2026-09-26; **Spec 255 W0–W7 landed 2026-09-27** (`WorldScene.cs` 8,275 → 4,922; 7 scene services + PM4 helpers). W8/W9 and E2 `Render()` split wait for R-10's after-capture |

Receipts: [E1](../specs/251-epic-viewer-ux-and-code-health/evidence/u01-e1-pm4-extraction-2026-09-25.md) ·
[ViewerApp campaign](../specs/251-epic-viewer-ux-and-code-health/evidence/u01-viewerapp-extraction-2026-09-25.md) ·
[Spec 255 W0–W7](../specs/255-worldscene-decomposition/evidence/w0-w7-extraction-2026-09-27.md) ·
[E4 selection](../specs/251-epic-viewer-ux-and-code-health/evidence/u01-e4-selection-2026-09-26.md).
**Where code lives now:** viewer features are owned services under `src/viewer/WoWViewer/Workbench/Services/<Feature>/`
(namespace `WoWViewer`); they reach app state only through `IViewerAppHost` (implemented in
`ViewerApp_Host.cs`). `ViewerApp.cs` is the shell: fields, composition root, window lifecycle. New behaviour
goes into the owning service; if it needs more app state, add one `IViewerAppHost` member + its one-line
implementation. PM4 overlay code lives in `Terrain/Pm4/` (`WorldScene.Pm4Overlay.X`). World-scene features live in
`Terrain/Scene/<Feature>/` services (`WorldScene.HoverPick`, `.TaxiActors`, `.ObjectFilters`, `.Selection`, `.Atmosphere`,
`.ExternalSpawns`, `.TerrainQueries`); they read scene state only through `IWorldSceneHost` (explicit impls in `WorldScene.cs`).

All other epic items remain untriaged in [TRIAGE.md](../specs/TRIAGE.md); nothing else is scheduled.
Receipt for the 2026-09-23 reconciliation:
[reconciliation README](../specs/archived/reconciliation-2026-09-23/README.md).

**Linux build note (agent sessions):** the viewer builds and tests on Linux with
`-p:EnableWindowsTargeting=true` after `git submodule update --init` of the six `wow-viewer/libs/*`
submodules. 26 tests fail on `HEAD` there for environmental reasons (missing `test_data/`,
Windows-path expectations) — compare against that set, not zero. One Curation test rewrites
`data-harvester/tests/fixtures/spec122_curation_manifest/.../curation_run.json` with the local path;
revert it before staging.

## Release state

`eng/Version.props` is restored to `0.6.1` / `0.6.1-dev` for active development. Stable tags: replacement `v0.6.0.1` at `5ba5eb2a` and base `v0.6.0` (2026-10-11).

**Shipped unverified in alpha2** (operator verification, tracked as V-tasks in the epics): the M2
texture-wrap fix (every M2 era); `LkAdtWriter` chunk-completeness fixes (17 call sites); no exported
map loaded in a client or Noggit; no DAT Cartography layer composed on screen; new export menu and
sidebar buttons never clicked.

## Measured gaps worth knowing before any work (details: TRIAGE §1)

- No map save pipeline exists (`MapSaveService` absent; `EditorSession.SaveAll()` writes nothing).
- Editor undo reverses only 2 of 5+ operation kinds.
- Zone music is disabled by policy and still reads `ZoneMusic` as a SoundEntries id.
- Modern data: the scene-light gate disables WMO instancing (~5.5 FPS, 16,431 WMO draws pre-R-10); R-10's per-placement gate made it worse (<1 FPS, operator) and was reverted 2026-09-27.
- v22 DAT blends layer 0 only; `AMAP` codec unidentified.
- `WorldScene.cs` 4,922 lines (after Spec 255 W0–W7, 2026-09-27), `ViewerApp` class 2,512 lines in 6 files — no new members (AGENTS.md §10).

## Non-negotiable constraints

- Preserve MPQ/ADT/WMO/M2/MDX readers and `AlphaWdtWriter`; no feature scope rides on a refactor.
- Runtime visual, input, FPS, audio, video and client-data proof are operator-owned.
- Training, harvests, GPU and cloud runs are operator-run; hand over PowerShell-ready commands.
- Preserve unrelated dirty work (`imgui.ini`); stage named files only.
- New scope needs operator-approved wording (AGENTS.md §9.1); receipts per §9.2.

## Handoff

**Immediate:** operator runs R10-T003/T007 captures and the U-01 smoke passes (U01-T003 PM4 overlay;
U01-T007/T013 every moved ViewerApp surface — checklist in the campaign receipt). U01-T009 E4 hover/click smoke and
[Spec 255](../specs/255-worldscene-decomposition/spec.md) T018 (W0–W7 smoke — checklist in its
[receipt](../specs/255-worldscene-decomposition/evidence/w0-w7-extraction-2026-09-27.md)) are owed too.
Next agent-owned U-01 work: Spec 255 W8/W9 and E2, after R-10's after-capture (R10-T007). Do not start any other epic
item before it is marked Want. Superseded dashboard:
[archive/2026-09-23-pre-reconciliation-active-context.md](archive/2026-09-23-pre-reconciliation-active-context.md).
