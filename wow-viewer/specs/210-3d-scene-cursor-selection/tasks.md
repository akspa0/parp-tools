# Tasks — Spec 210: 3D Scene Cursor & In-World Spatial Selection

## Phase 1 — OpenSCAD MCP Server & Procedural Mesh Loader

- [x] **210-T001** Configure & verify OpenSCAD MCP server:
  - Configured `openscad` in `.mcp.json` and `~/.gemini/config/mcp_config.json` via `uv` using `git+https://github.com/quellant/openscad-mcp.git` and binary at `C:\Program Files\OpenSCAD\openscad.com`.
  - Verified OpenSCAD version 2021.01 and headless `.off` / `.stl` export.
- [x] **210-T002** Implement `ProceduralMeshLoader` (`src/viewer/WoWViewer/Rendering/ProceduralMeshLoader.cs`) & `OffGeometry` (`src/core/WowViewer.Core/Geometry/OffGeometry.cs`):
  - Ingest OpenSCAD `.off` (Object File Format) into headless geometry model and OpenGL VBO/EBO buffers.
  - Generates 3D primitives (procedural pointer arrows, orbital rings, reticles, pins) for in-scene rendering.
  - Unit tests in `OffGeometryTests.cs` passing (3/3).
- [x] **210-T003** Authored OpenSCAD Source-Tree Assets & Batch Compiler:
  - Created source-tree models in `src/viewer/WoWViewer/Assets/OpenScad/`: `cursor_pointer.scad`, `cursor_reticle.scad`, `cluster_pin.scad`, `camera_hud_gimbal.scad`, `hud_frame_bracket.scad`.
  - Compiled all models to `.off` using `C:\Program Files\OpenSCAD\openscad.com` and bundled into build output via `WoWViewer.csproj`.
  - Built batch compiler script `scripts/compile_openscad_assets.ps1`.
  - Implemented `OpenScadAssetResolver.cs` with source-tree and output directory resolution.

---

## Phase 2 — 3D Cursor Asset Loading & Viewport Rendering (US1, US4)

- [x] **210-T101** Implement `SceneCursorRenderer`:
  - Discover and load `Interface\Cursor\Cursor.mdx` (Alpha 0.5.3) or `Interface\Cursor\Point.m2` / `Cursor.m2` (Standard 1.12+) from `IDataSource`.
  - Load associated textures (`Cursor.blp`, `CursorsCastGlow2.blp`, `CursorIcons.blp`).
  - Support procedural OpenSCAD cursor meshes from `ProceduralMeshLoader` and `OpenScadAssetResolver`.
  - Graceful fallback: Fallback cleanly to procedural pointer.
- [x] **210-T102** Enforce Camera Culling Invariant:
  - Cursor mesh is NEVER culled by camera frustum or far clip.
  - Near-clip protection: Clamp minimum ray distance to $Z \ge Z_{\text{near}} + 0.15\text{ yd}$ so the cursor can never clip behind the near plane.
  - Depth bias / always-visible overlay: Render with depth range bias (`glDepthRange(0.0, 0.05)`) so terrain/doodads never obscure the pointer.
- [x] **210-T103** Distance-proportional 3D cursor transformation:
  - Compute world position $\mathbf{P}_{\text{hit}}$ along the mouse ray at nearest scene surface.
  - Apply scale $S = \text{BaseScale} \times \text{Distance} \times \tan(\text{FOV} / 2)$ to maintain screen-relative readability.
  - Align orientation with camera view plane.
- [x] **210-T104** Seamless ImGui Interop in `ViewerApp`:
  - When mouse is over 3D viewport and `CanSceneConsumeMouse`: call `ImGui.SetMouseCursor(ImGuiMouseCursor.None)`.
  - When mouse hovers ImGui chrome, menus, modals, or popups: restore system cursor immediately without lag or lockup.
- [x] **210-T105** Hook `SceneCursorRenderer` into `ViewerApp.OnRender` and add Settings UI:
  - Draw 3D cursor mesh in 3D viewport before 2D ImGui pass.
  - Add Cursor Style dropdown in Viewer Settings (`Authentic WoW Gauntlet`, `Procedural 3D Pointer`, `3D Target Reticle`, `Classic OS Arrow`) and Cursor Scale slider.
- [x] **210-T106** Camera-Rigged 3D In-Scene HUD (`CameraHudRig3D.cs`):
  - Mounts authored 3D OpenSCAD objects directly to the camera entity's local coordinate hierarchy.
  - Implements 3D Attitude & Heading Gimbal: dynamic in-scene flight compass pointing toward world North and tilting with camera pitch/roll.
  - Implements 3D Viewport Corner Framing Brackets with holographic glow.
  - Added Settings UI controls in `Settings > Interface` (Enable 3D Camera HUD, Gimbal toggle, Brackets toggle).

---

## Phase 3 — Direct 3D Spatial Selection & Dynamic States (US2)

- [x] **210-T201** Implement direct 3D contact detection:
  - Direct 3D contact detection against terrain chunks, WMO groups, M2 doodads, and water planes.
  - Output exact 3D contact point, surface normal, and hit object reference.
- [x] **210-T202** Implement dynamic cursor state machine:
  - `Pointer` (gauntlet pointer) over terrain or open ground.
  - `Interact` / `Speak` when hovering doodads, NPCs, or interactive objects.
  - `CastGlow` during active editor placement or terrain brushes.
- [x] **210-T203** Wire 3D selection into `ViewerApp_ClickSelection.cs`:
  - Direct 3D object pick on left-click, replacing legacy 2D screen-ray heuristics.
- [x] **210-T204** Replace intrusive 2D screen-space hover overlays with clean 3D focus highlighting.

---

## Phase 4 — In-Scene 3D Cluster Disambiguator (US3, US5)

- [x] **210-T301** Implement cluster detection in click selection:
  - Query nearby objects within compact 3D radius around hit point.
  - If single target, proceed to direct selection. If multiple, trigger 3D cluster selector.
- [x] **210-T302** Implement `SceneClusterSelector3D`:
  - Render an in-world 3D radial orbital ring at the cluster datum (using procedural mesh).
  - Display billboarded 3D selection tags/pins for each candidate object in the cluster.
- [x] **210-T303** Wire 3D pin hover preview and click selection commit:
  - Clicking candidate tag commits selection.
  - Pressing 1-9 selects candidate directly.
  - Pressing Escape or clicking outside dismisses the 3D pop-up cleanly.
- [x] **210-T304** Suppress 2D hover overlay (`##SceneHoverAssetOverlay`) when 3D scene cursor is active or cluster selector is open.
- [x] **210-T305** Remove legacy 2D `##ClickSelectionOverlay` modal list and consolidate cluster disambiguation into a single non-overlapping in-scene card above the 3D ring.
- [x] **210-T306** Physical OS hardware cursor hiding via Silk.NET `CursorMode.Hidden` on `_input.Mice` when hovering 3D viewport.
- [x] **210-T307** UI Typography and Font Size scaling (`0.85x`–`2.20x` slider + 100%, 120%, 135%, 150%, 175% presets) with persistence in `ViewerSettings.UiFontScale`.

---

## Phase 5 — Verification & Operator Validation

- [ ] **210-T401** Unit tests in `SceneSpatialPickerTests.cs`:
  - Test distance-proportional scaling calculation.
  - Test 3D ray-to-surface intersection and cluster grouping logic.
  - Test `ProceduralMeshLoader` OFF parsing.
- [ ] **210-T402** Interactive verification in `WoWViewer` (Alpha 0.5.3, `Azeroth`):
  - Confirm `Cursor.mdx` renders authentically in 3D scene without camera culling.
  - Confirm OS cursor hides over viewport and restores over ImGui panels.
  - Confirm 3D object selection and cluster pop-up in dense areas.
  - Confirm switching cursor styles in Settings.
- [ ] **210-T403** Interactive verification in `WoWViewer` (Standard 1.12+).
- [ ] **210-T404** Document completion in `STATUS.md` and `activeContext.md`.
