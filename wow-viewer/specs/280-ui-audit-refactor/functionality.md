# Functionality Audit — 2026-10-21

Companion to [audit.md](audit.md). The structural audit maps *where* controls are; this document
maps *what they actually do*. Method: code trace of every handler reachable from the menu, bottom
bar, sidebars, Editor pages, Archaeology pages, and PM4 workbench. Runtime behavior is marked
separately because a handler existing is not proof that it works on a real client.

## Status legend

| Status | Meaning |
|---|---|
| `wired` | A handler exists and performs the action in code. |
| `wired-unverified` | A handler exists; the operator reports it fails or is unreliable; runtime proof owed. |
| `typed-path` | Works, but the only way to supply the path is a text field. |
| `duplicate` | The same action exists on another surface. |
| `legacy` | An implementation switch exposed as a user-facing control. |
| `dead` | No handler, or the handler has no effect. |
| `offline-candidate` | Belongs in the offline Museum-patch tool, not the viewer. |

## 1. File menu

| Control | Handler | Effect | Status | Decision |
|---|---|---|---|---|
| Open File... | `_wantOpenFile` → `OpenFilePopup` → `_modelLoader.LoadFileFromDisk` | Loads a model/WDT from a typed path | `typed-path` | Replace with `ImGuiPathPicker` |
| Open Alpha WDT (loose map)... | `_wantOpenWdtFile` → picker → `LoadFileFromDisk` | Loads a loose WDT | `wired` | Keep |
| Open Loose PM4 / PD4 File... | `_wantOpenPm4File` → picker → `Pm4Overlay.LoadLoosePm4File` | Decodes a PM4/PD4 overlay; requires a loaded world | `wired` | Keep |
| Open Game Folder (MPQ)... | `_showFolderInput` → folder input popup | Loads an MPQ client folder | `typed-path` | Replace with picker |
| Open CASC Install (local)... | `_cascAhdrSource._wantOpenCascInstall` | Picks a Battle.net install and build | `wired` | Keep |
| Open CASC Install (local + CDN fill)... | `_wantOpenCascInstallWithCdnFill` | Same, with CDN fill | `wired` | Keep |
| DAT Terrain > Open DAT Terrain Folder... | `_wantOpenAhdrTerrainFolder` | Loads a DAT terrain folder | `wired` | Keep |
| DAT Terrain > Export format checkboxes | `MapExportFormats.DrawCheckboxes` | Selects output formats | `wired` | Keep |
| DAT Terrain > Export Loaded DAT Map... | `ExportLoadedDatMap` | Writes the loaded DAT map | `wired` | Keep; verify write |
| DAT Terrain > Export Nearby Tiles as DAT v26 | `ExportNearbyTilesAsDatV26(radius: 2)` | Writes nearby tiles | `wired` | Keep; verify write |
| DAT Terrain > Height scale | `DrawAhdrHeightScaleMenu` | Sets export height scale | `wired` | Keep |
| Open Saved Game Folder > [client] | `QueueKnownGoodClientAction` | Loads a saved client | `wired` | Keep |
| Attach Loose Map Folder... | `_wantAttachLooseMapFolder` → picker → `AttachLooseMapOverlay` | Overlays loose maps on an MPQ | `wired` | Keep |
| Load Loose Map Folder Against Saved Base > [client] | `QueueKnownGoodClientAction(attachLooseFolder: true)` | Loads base + overlay | `wired` | Keep |
| Save Current Game Folder As Known-Good Base | `SaveCurrentGameFolderAsKnownGoodBase` | Persists the current MPQ path | `wired` | Keep |
| Forget Known-Good Base > [client] | `QueueForgetKnownGoodClientPath` | Removes a saved path | `wired` | Keep |
| Settings... | `_showSettingsWindow = true` | Opens settings | `wired` | Keep |
| Quit | `_window.Close()` | Closes the viewer | `wired` | Keep |

**File menu verdict**: the operator's "only the File menu mostly works" is accurate. Two actions
(`Open File...`, `Open Game Folder...`) are typed-path; the rest are wired. The DAT export actions
are wired but their write result is unverified.

## 2. View menu

| Control | Handler | Effect | Status | Decision |
|---|---|---|---|---|
| Wireframe | `_renderer?.ToggleWireframe()` | Toggles the standalone model renderer only; terrain has a separate bottom-bar toggle | `wired` (misleading) | Rename to "Model Wireframe" or route to the active renderer |
| Reset Camera | `ResetCamera()` | Resets the camera | `wired` | Keep |
| Hide UI Chrome | `_hideUiChrome` toggle | Hides chrome | `wired` | Keep |
| Tab System (069) | `_useTabUi` toggle | Switches the whole shell implementation | `legacy` | Remove |
| Dockable Shell Panels | `_useDockspaceUi` toggle | Switches the shell layout implementation | `legacy` | Remove |
| Left Sidebar | `_showLeftSidebar` toggle | Shows/hides the left sidebar | `wired` | Keep |
| Right Sidebar | `_showRightSidebar` toggle | Shows/hides the right sidebar | `wired` | Keep |
| Log Console... | Opens the Log tab or window | Shows the log | `duplicate` | Keep one home (Utilities/Log) |
| Performance & Profiling... | Opens the Perf tab or window | Shows perf | `duplicate` | Keep one home |
| Lighting Diagnostics... | Opens the Lighting tab or window | Shows lighting diagnostics | `duplicate` | Keep one home |
| PM4 Workbench | `OpenPm4Workbench(Selection)` | Focuses the PM4 tab | `duplicate` | Keep one entry point |
| Reset Shell Layout | `ResetShellLayoutToDefaults()` | Resets the layout | `wired` | Keep |
| File Browser | `_showFileBrowser` toggle | Shows/hides the file browser | `duplicate` | Keep the left sidebar home |
| Model Info | `_showModelInfo` toggle | Shows/hides model info | `duplicate` | Keep the Inspector home |
| Asset Catalog | Opens the Asset Catalog tab or toggles the catalog view | Shows the catalog | `duplicate` | Keep one home |

**View menu verdict**: every entry is wired, but 8 of 15 duplicate another surface and 2 are
implementation switches. The Wireframe entry is misleading because it does not affect terrain.

## 3. Tools menu

| Control | Handler | Effect | Status | Decision |
|---|---|---|---|---|
| Taxi Routes... | Opens the Taxi tab | Shows taxi routes | `duplicate` | Move to Viewer destination |
| Audio Settings... | Opens the Audio tab | Shows audio settings | `duplicate` | Move to Options |
| Converters > Map Converter... | `_showMapConverterDialog` | Opens the converter dialog | `wired` | Move to Editor > Converters |
| Converters > WMO Converter... | `_showWmoConverterDialog` | Opens the WMO converter | `wired` | Move to Editor > Converters |
| Panels > Model Info | Opens the Model tab | Shows model info | `duplicate` | Delete; Inspector owns it |
| Panels > Log Viewer | Opens the Log tab | Shows the log | `duplicate` | Delete; Utilities owns it |
| Panels > Perf | Opens the Perf tab | Shows perf | `duplicate` | Delete; Utilities owns it |
| Panels > Settings... | `_showSettingsWindow` | Opens settings | `duplicate` | Delete; File > Settings owns it |
| Panels > Asset Catalog | Opens the catalog tab | Shows the catalog | `duplicate` | Delete; Utilities owns it |
| Panels > Capture Automation | `OpenCapturePanelTab(Automation)` | Opens capture automation | `duplicate` | Move to Viewer destination |
| Panels > Camera Path | `OpenCapturePanelTab(CameraPath)` | Opens camera path | `duplicate` | Move to Viewer destination |
| Panels > Taxi | Opens the Taxi tab | Shows taxi | `duplicate` | Delete; one Taxi home |
| Panels > Weak Signal & Stratigraphy | Opens Archaeology page 0 | Shows weak signal | `duplicate` | Delete; Archaeology owns it |
| Panels > UniqueId Archaeology | Opens Archaeology page 1 | Shows UniqueId | `duplicate` | Delete; Archaeology owns it |
| Panels > PM4 Analysis | Opens Archaeology page 4 (redirect) | Redirects to PM4 | `duplicate` | Delete; PM4 workbench owns it |
| Panels > Cartography | Opens Archaeology page 5 | Shows cartography | `duplicate` | Delete; Archaeology owns it |
| Panels > Editor Workbench | Opens Editor page 0 | Shows editor | `duplicate` | Delete; Editor destination owns it |
| Export > Synthesized Terrain Minimap... | `_showSynthesizedMinimapExportDialog` | Opens the export dialog | `wired` | Move to Cartography |
| Export > GLB > Export GLB... | `_wantExportGlb` → `GlbExporter` | Writes a `.glb` to `ExportDir`, reports the path | `wired` | Move to the owning destination |
| Export > GLB > Export GLB (Collision Only)... | `_wantExportGlbCollision` → `GlbExporter` | Writes a collision `.glb`, reports the path | `wired` | Move to the owning destination |
| Export > GLB > Map Tiles > Current/Loaded/Whole | `_wantExportMapGlbTiles` → `RunMapGlbTilesExport` | Writes tile GLBs, reports the folder | `wired` | Move to Terrain Tools |
| Export > Terrain > Alpha Masks > 4 scopes | `_wantTerrainExport` → `RunTerrainExport` | Writes alpha atlases/chunks | `wired` | Move to Terrain Tools |
| Export > Terrain > Heightmaps > 3 scopes | `_wantTerrainExport` → `RunTerrainExport` | Writes 257×257 heightmaps | `wired` | Move to Terrain Tools |
| Export > Terrain > MCCV > 3 scopes | `_wantTerrainExport` → `RunTerrainExport` | Writes MCCV PNGs | `wired` | Move to Terrain Tools |
| Import > Terrain > Alpha Masks > From Folder | `_wantTerrainImport` → `RunTerrainImport` → `BeginAlphaFolderImport` | Opens a scope dialog, then applies | `wired-unverified` | Keep only if runtime proof passes |
| Import > Terrain > Heightmaps > From Folder | `_wantTerrainImport` → `BeginHeightmapFolderImport` | Opens a scope dialog, then applies | `wired-unverified` | Keep only if runtime proof passes |
| Import > Terrain > MCCV > From Folder | `_wantTerrainImport` → `BeginMccvFolderImport` | Opens a scope dialog, then applies | `wired-unverified` | Keep only if runtime proof passes |

**Tools menu verdict**: the export handlers are real and report their paths. The import handlers
exist and open a scope dialog, but the operator reports they do not work; they are
`wired-unverified` until a real import is proven. The Panels submenu is 12 duplicate navigation
entries.

## 4. Bottom bar

| Control | Handler | Effect | Status | Decision |
|---|---|---|---|---|
| Tiles / Chunks / Cells | `renderer.ShowTileGrid/ShowChunkGrid/ShowCellGrid` | Grid overlays | `wired` | Move to Options |
| Layer toggles | `DrawTerrainLayerToggles` | Terrain layer visibility | `wired` | Move to Options |
| Holes | `SetIgnoreTerrainHolesGlobally` | Hole masking | `wired` | Move to Options |
| Alpha / Shadows / MCCV / Contours | `renderer.ShowAlphaMask/ShowShadowMap/UseMccv/ShowContours` | Surface overlays | `wired` | Move to Options |
| Liquid | `liquidRenderer.ShowLiquid` | Liquid visibility | `wired` | Move to Options |
| WL* | `_worldScene.ShowWlLiquids` | WL liquid visibility | `wired` | Move to Options |
| BBs | `_worldScene.ShowBoundingBoxes` | Bounding boxes | `wired` | Move to Options |
| PM4 | `Pm4Overlay.ShowPm4Overlay` | PM4 overlay | `wired` | Move to Options |
| Inspect | Enables PM4 overlay + opens PM4 Selection | PM4 inspect | `wired` | Keep as a Viewer shortcut |
| Object transparency | `DrawObjectTransparencyControls` | Opacity sliders | `wired` | Move to Options |
| Terrain WF / M2-WMO WF / Anim | `SetTerrainWireframeEnabled` / `SetObjectWireframeEnabled` / `WorldDoodadAnimationEnabled` | Wireframe and animation | `wired` | Move to Options |
| 3D Cursor | `_sceneCursorRenderer.Style` | Cursor style | `wired` | Move to Options |
| Settings | `_showSettingsWindow` | Opens settings | `duplicate` | Move to Options |

**Bottom bar verdict**: every toggle is wired. The problem is placement and duplication, not
function.

## 5. Editor pages

| Page / section | Handler | Effect | Status | Decision |
|---|---|---|---|---|
| Placement & Objects > Tasks & Workspace | `DrawArchaeologyEditorTasksSubTab` | Archaeology editor task list | `wired` | Rename to "Archaeology Tasks" |
| Placement & Objects > PM4 Placement Tools | `OpenPm4Workbench` | One navigation button | `duplicate` | Delete |
| Placement & Objects > 3D Object Library | `DrawRosettaObjectLibrarySubTab` | Rosetta library | `wired` | Rename to "Rosetta Object Library" |
| Placement & Objects > Population | `DrawPopulationSubTabContent` | Population list | `wired` | Keep |
| Terrain Tools > Terrain Lab | `DrawTerrainLabSubTab` | Terrain lab | `wired` | Keep |
| Data I/O > PM4 Exports | `DrawPm4ExportCommandSet` | PM4 export commands | `duplicate` | Delete; PM4 workbench owns exports |
| Data I/O > New Map Creator | `NewMapCreatorService.Draw` | Creates a new map | `wired` | Keep |
| Data I/O > Imports & Exports | `DrawArchaeologyEditorImportsSubTab` | Archaeology import surface | `wired` | Rename to "Archaeology Imports" |
| Converters > Converters | `DrawConvertersSubTabContent` | Converter tools | `wired` | Keep |

## 6. Archaeology pages

| Page | Handler | Effect | Status | Decision |
|---|---|---|---|---|
| Weak Signal & Stratigraphy | `DrawTerrainControlsAdjustmentWeakSignalContent` | Weak-signal tooling | `wired` | Keep |
| UniqueId Timeline | `DrawArcheologyRangeSubTab` | UniqueId range + time machine | `wired` | Keep |
| Layers & Provenance | `DrawArcheologyLayersSubTab` | Layer table | `wired` | Keep |
| Playback & Capture | `DrawArcheologyPlaybackSubTab` + `DrawArcheologyCaptureSubTab` + `DrawCapturePanelContent` | Playback and capture | `wired` | Keep |
| PM4 Analysis | Redirect button | Opens PM4 workbench | `duplicate` | Delete; PM4 is a Viewer surface |
| Cartography > Layers | `DrawCartographyLayersSubTab` | Layer stack, DBC discovery, add controls, donor picker | `wired` | Keep; split into sections |
| Cartography > Synthesized Minimap | `DrawSynthesizedMinimapExportContent` | Minimap export | `wired` | Keep |
| Cartography > Save Merged Output | `DrawCartographySaveMergedSubTab` | Manifest + composed exports | `wired` | Keep |
| Legacy 5-tab strip | `DrawArcheologySubTabContent` | Range/Layers/Playback/Capture/Stratigraphy | `duplicate` | Delete |
| Floating UniqueId window | `DrawUniqueIdArchaeologyWindow` | Stacks 3 sub-tabs | `duplicate` | Delete |
| Quick launchers | `DrawArchaeologyQuickSection` | Buttons into pages | `duplicate` | Delete |

## 7. PM4 workbench

| Tab | Handler | Effect | Status | Decision |
|---|---|---|---|---|
| Overlay | `DrawPm4OverlayWorkbenchContent` | Overlay controls and stats | `wired` | Keep |
| Selection | `DrawPm4SelectionWorkbenchContent` | Selected object, match curation | `wired` | Keep selection; move matching offline |
| Correlation | `DrawPm4WmoCorrelationContent` | WMO correlation report | `offline-candidate` | Move to the offline tool |
| Info | `DrawPm4SceneFacts` | Scene facts | `wired` | Keep |
| Reconcile | `DrawPm4ReconciliationContent` | PM4/Museum reconciliation preview + apply | `offline-candidate` | Move to the offline tool |
| Alignment | `DrawPm4AlignmentContent` | Manual alignment + saved matches | `offline-candidate` | Move to the offline tool |
| Outliner | `DrawPm4Outliner` | Object outliner | `wired` | Keep |
| Object Groups | `DrawPm4ObjectGroups` | Group list + camera focus | `wired` | Keep |

**PM4 verdict**: the viewer keeps inspection (Overlay, Selection, Info, Outliner, Object Groups).
The matching/alignment/reconciliation tooling moves to the offline Museum-patch tool. This is the
operator's explicit direction: matching is a batch job that patches ADTs, not an interactive
viewer feature.

## 8. Decisions summary

### Keep in the viewer

- File open/load actions (after the typed-path fixes).
- View: camera, chrome, sidebars, layout reset.
- Bottom-bar toggles (moved into Options).
- Editor: Population, Terrain Lab, New Map Creator, Converters, Rosetta library, Archaeology tasks.
- Archaeology: all six pages, with the Cartography Layers page split into sections.
- PM4: Overlay, Selection, Info, Outliner, Object Groups.

### Reorganize

- Bottom-bar toggles → Options sub-menu.
- Terrain export/import → Editor > Terrain Tools.
- Converters → Editor > Converters.
- Capture/Taxi/Audio → Viewer destination or Options.
- PM4 exports → PM4 workbench only.

### Remove

- `Tab System (069)`, `Dockable Shell Panels`.
- Tools > Panels (12 duplicate entries).
- The legacy Archaeology 5-tab strip, floating UniqueId window, and Quick launchers.
- The one-button "PM4 Placement Tools" section.
- The second PM4 export home in Data I/O.
- The 12 duplicate floating windows.

### Move offline (Spec 281)

- PM4 Correlation, Reconcile, and Alignment.
- The in-viewer object-match store and manual match curation.
- The PM4 asset matcher as a viewer feature.

## 9. Runtime proof still owed

The code trace cannot prove these; the operator must confirm on a real client:

1. DAT export writes the expected files.
2. Terrain export writes the expected files for each of the 10 kinds.
3. Terrain import applies each of the 3 kinds.
4. GLB export writes a loadable file.
5. Map GLB tile export writes a loadable file.
6. The PM4 overlay loads and selects on a real map.
7. Population auto-stream and double-click teleport.
