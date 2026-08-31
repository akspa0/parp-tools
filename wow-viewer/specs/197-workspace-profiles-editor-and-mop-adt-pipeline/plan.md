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
- **Ghidra Analysis Integration**: Use the read-only Ghidra MCP connection on `WoW.exe` 5.0.1.15464 to extract decompiled structures for `CMapChunk`, `CMapTile`, split-file loading, and WMO terrain seam blending. The current evidence checkpoint is [`research-ghidra-5.0.1.md`](research-ghidra-5.0.1.md:173): this build requires `MVER == 0x12`, loads root + one selected `_obj0`/`_obj1` and `_tex0`/`_tex1` pair, expects 256 outer MCNK records in each file-data object, consumes the MCNK header only in the root slot, and gates area admission on map-table `Flag_Exists`. `MHID`, `MDID`, `MCXH`, and `_lod.adt` remain evidence-gated rather than assumed.
- **Native evidence gates**: Complete the WDT/MAIN tile-table and internal LOD-band extraction, then complete native `MCBB`/alpha/height/WMO seam rendering extraction before changing production readers or shaders. Preserve the distinction between file discovery, map-table existence, and per-band residency.
- **Dead/dormant evidence**: Preserve the separate [`5.0.1-dead-dormant-partial-rendering.md`](evidence/5.0.1-dead-dormant-partial-rendering.md) inventory and the consolidated [`wow-5.0.1-adt-wdt-definitive.md`](../../docs/architecture/wow-5.0.1-adt-wdt-definitive.md). The liquid factory default branch is a confirmed partial path; DepthCache/GBuffer are capability-gated infrastructure; atlas/batching are optional toggles; zero direct xrefs are only low-reachability candidates until indirect dispatch is ruled out.
- **Documentation gate**: The address discrepancy between the earlier `0x00bb9490` path-builder note and the later `FUN_00b94990` inventory must be resolved before the native function map is treated as final. No production parser/renderer edit is authorized by this phase alone.
- **Implemented reader/runtime slice:** Expand `StandardTerrainAdapter` and shared IO to discover the native 5.0.1 multi-split ADT family (`_obj0.adt`, `_obj1.adt`, `_tex0.adt`, `_tex1.adt`), load the root plus one selected object/texture band, route companion `MCNK` wrappers without the root-only 128-byte header, and preserve sparse MCIN physical-slot identity in the supported paths. `_lod.adt` remains separately evidence-gated.
- **Still evidence-gated in the reader/runtime:** Parse and consume `MHID` (Material Height ID), `MDID` (Material Diffuse ID), and `MCXH` per-chunk height-blend parameters only when their exact native 5.0.1 consumers are established. Do not infer native shader behavior from chunk names alone.
- Update `TerrainRenderer` shader pipeline:
  - Add height-blending terrain shader logic combining diffuse alpha and height map values for sharp, non-blurry texture transitions.
  - Support WMO-to-terrain seam blending.

### Phase 5A: Explicit cross-era serialization boundary

This phase is complete for target selection and safety, but not for native MoP
split serialization:

1. Keep source family independent from the output target.
2. Register the explicit targets `AlphaWdt053`, `LkAdtV18`, and `MopSplitAdt` in Core,
   with stable display/CLI names and route validation.
3. Route Alpha identity output to the Alpha writer and Alpha/split normalization to
   the appropriate existing converter.
4. Guard `LkAdtWriter` so target-aware callers can request only LK v18 output.
5. Report lossy conversion for Alpha→LK, split→Alpha, and split→LK.
6. Keep MoP split output unavailable until a native-aligned split writer can emit
   root/object/texture bands, 256 physical slots, root-only MCNK headers, and
   modern split-only state. Never label LK monolithic output as native MoP.
7. Keep archive client-root input separate from the selected loose split overlay
   directory; route split→Alpha through the Alpha command rather than the
   split→LK command.

### Phase 5B: Remaining split conversion/model work

1. Replace compact `LkAdtData.Chunks` assumptions with a slot-aware canonical ADT
   document model before implementing a native split writer.
2. Extend merger, texture-transfer, and converter paths to consume `_obj1` and
   `_tex1` instead of only band 0.
3. Define per-field loss policies for `MCRD`, `MCRW`, `MHID`, `MDID`, `MCXH`,
   `MCBB`, and other modern-only state.
4. Implement and test the native MoP split writer; only then enable the MoP target.

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
