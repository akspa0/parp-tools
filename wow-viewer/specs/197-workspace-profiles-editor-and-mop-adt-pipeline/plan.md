# Spec 197 Plan: UI Workspace Profiles, Editor Mode Integration, PM4 Mouse Inspection, Multi-Client Map Staging & MoP 5.0.1 ADT Pipeline

## 1. Architectural Overview

This technical plan bridges UI workflow clarity with next-generation engine format pipelines:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              WoWViewer Top Bar                              │
│  [File]  [View]  [Tools]  [Help]    ──│──    [Viewer]  [Editor]  [Research] │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                              ▼
┌───────────────────┐        ┌───────────────────┐        ┌───────────────────┐
│  Viewer Profile   │        │  Editor Profile   │        │ Archaeology Mode  │
│ ───────────────── │        │ ───────────────── │        │ ───────────────── │
│ • Quick Controls  │        │ • Terrain Brush   │        │ • Stratigraphy    │
│ • Unified Inspect │        │ • Overhead Canvas │        │ • WDL Magnetizer  │
│ • Scene / Spawns  │        │ • Placements      │        │ • Multi-Era Diff  │
│ • Utilities & Log │        │ • History / Undo  │        │ • UniqueId Scan   │
│ • PM4 Mouse Pick  │        │ • Staging / Save  │        │ • Rosetta Lookup  │
└───────────────────┘        └───────────────────┘        └───────────────────┘
```

---

## 2. Phase Breakdown

### Phase 1: Workspace Profile Switcher & Editor Tab Restoration
- Define `enum ViewerWorkspaceMode { Viewer, Editor, Archaeology }`.
- Render a dedicated mode switcher button group in the top menu bar / toolbar with distinct active styling.
- Expose the missing `Editor` tab button (`DrawTopTabButton(WorkbenchTab.Editor, "Editor")`) and ensure keyboard shortcut / profile selection switches directly into `DrawEditorContent()`.
- Filter right sidebar top tabs and bottom pages according to the active `ViewerWorkspaceMode`.

### Phase 2: Menu Audit & "MK Dataset" Legacy Purge
- Remove `MkDatasetHarvester.cs` and `VlmProjectLoader.cs` legacy bindings from `ViewerApp.cs`, `ViewerApp_CaptureAutomation.cs`, and `imgui.ini`.
- Consolidate offline ML dataset exports under an explicit `Offline Data / Conversion` sub-menu.
- Audit all menu commands to ensure hotkeys and disabled states reflect active data availability.

### Phase 3: Direct PM4 Viewport Mouse Raycast & Selection
- Implement `TryPickPm4ObjectAtRay(Vector3 rayOrigin, Vector3 rayDir, out Pm4OverlayObject? hitObject)` in `WorldScene.cs` / `ViewerApp_ClickSelection.cs`.
- Add PM4 raycast and screen-space pick hits to `_clickSelectionCandidates`.
- Selecting a PM4 candidate highlights the object in the PM4 workbench and populates the unified Inspector panel.

### Phase 4: Multi-Client Map Staging & Restoration Workspace
- Create `MultiClientMapStagingService.cs` in `WowViewer.Core.Editor`.
- Support mounting multiple `IDataSource` archives (e.g. Alpha 0.5.3 MPQ + 3.3.5a MPQ + 4.0.1 MPQ) simultaneously with unique namespace keys.
- Allow copying terrain chunk slices, heightmaps, vertex colors, and object placements from any mounted client map into the active restoration target map.

### Phase 5: 4.3.4 through 5.1 MoP ADT Format & Rendering Engine
- **Ghidra Analysis Integration**: Use Ghidra MCP connection on `WoW.exe` 5.0.1.15464 to extract decompiled structures for `CMapChunk`, `CMapTile`, `MHID`, `MDID`, `MCXH`, and WMO terrain seam blending.
- Expand `StandardTerrainAdapter`:
  - Support multi-split ADT files (`_obj0.adt`, `_obj1.adt`, `_tex0.adt`, `_tex1.adt`, `_lod.adt`).
  - Parse `MHID` (Material Height ID) and `MDID` (Material Diffuse ID) chunks.
  - Parse `MCXH` per-chunk height blend parameters.
- Update `TerrainRenderer` shader pipeline:
  - Add height-blending terrain shader logic combining diffuse alpha and height map values for sharp, non-blurry texture transitions.
  - Support WMO-to-terrain seam blending.

---

## 3. File Changes

| Component | Target File | Action | Description |
|---|---|---|---|
| **Viewer UI** | `src/viewer/WoWViewer/ViewerApp.cs` | Modify | Add `ViewerWorkspaceMode`, top toolbar profile switcher, clean menus, remove `MkDataset`. |
| **Viewer UI** | `src/viewer/WoWViewer/ViewerApp_Sidebars.cs` | Modify | Expose Editor button, filter workbench tabs by active workspace profile. |
| **Viewer UI** | `src/viewer/WoWViewer/ViewerApp_ClickSelection.cs` | Modify | Add PM4 mouse hover and click-raycast candidate selection. |
| **Core Editor** | `src/core/WowViewer.Core.Editor/Staging/MultiClientMapStagingService.cs` | New | Multi-client archive ingestion and cross-era asset/terrain staging. |
| **Core IO** | `src/core/WowViewer.Core.IO/Maps/MopAdtChunkParser.cs` | New | Split ADT, `MHID`, `MDID`, and `MCXH` chunk reader. |
| **Terrain Adapter** | `src/viewer/WoWViewer/Terrain/StandardTerrainAdapter.cs` | Modify | Wire 4.3.4–5.1 multi-split ADT loading and height blend metadata. |
| **Terrain Renderer** | `src/viewer/WoWViewer/Terrain/TerrainRenderer.cs` | Modify | Shader height-blend rendering and WMO terrain seam blending. |
| **Tests** | `tests/WowViewer.Core.Tests/MopAdtChunkParserTests.cs` | New | Unit tests for 4.3.4/5.0.1 ADT chunk parsing and multi-client staging. |
