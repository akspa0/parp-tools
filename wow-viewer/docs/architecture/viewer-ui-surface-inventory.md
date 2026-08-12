# WoWViewer UI Surface Inventory

**Spec**: 080-wow-ui-consolidation  
**Phase**: 0 — UI Surface Inventory and Release Gate  
**Generated**: 2026-07-11  
**Source**: `wow-viewer/src/viewer/WoWViewer/ViewerApp.cs` (active) vs `gillijimproject_refactor/src/MdxViewer/ViewerApp.cs` (legacy reference)

**Follow-up**: Spec 145 (`specs/145-wow-ui-overhaul/`) owns the next bounded shell slice:
context-aware keybindings, visual shortcut help, the vertical main workbench rail, bounded
navigator/minimap layout, wrapped logs, persistent-window audit, and release-truth synchronization.
Spec 080 remains the historical consolidation owner; working routes are preserved until replacements
pass independent proof.

---

## Legend

| Status | Meaning |
|--------|---------|
| `working` | Renders in both tabbed and legacy modes; all routes reachable |
| `misrouted` | Flag set by menu/toolbar but draw dispatch missing in one mode |
| `missing` | Existed in legacy, no active implementation in wow-viewer |
| `placeholder` | Menu item exists but body is stub/TODO |
| `duplicate` | Same surface reachable via multiple routes |
| `retired` | Intentionally removed; no replacement planned |
| `disabled-with-reason` | Visible but disabled; tooltip explains prerequisite |

---

## 1. Menu Bar — File

| # | User Label | Source Method | Menu Entry | Tabbed Route | Legacy Route | Required State | Owner Frame | Status | Predecessor Spec |
|---|------------|---------------|------------|--------------|--------------|----------------|-------------|--------|------------------|
| 1 | Open File… | `_wantOpenFile = true` | File → Open File… | ✅ | ✅ | — | Main | working | 060 |
| 2 | Open Alpha WDT (loose map)… | `_wantOpenWdtFile = true` | File → Open Alpha WDT… | ✅ | ✅ | — | Main | working | 060 |
| 3 | Open Game Folder (MPQ)… | `_showFolderInput = true` | File → Open Game Folder (MPQ)… | ✅ | ✅ | — | Main | working | 060 |
| 4 | Open Saved Game Folder → [client] | `QueueKnownGoodClientAction` | File → Open Saved Game Folder | ✅ | ✅ | `_knownGoodClientPaths.Count > 0` | Main | working | 057 |
| 5 | Attach Loose Map Folder… | `_wantAttachLooseMapFolder = true` | File → Attach Loose Map Folder… | ✅ | ✅ | `_dataSource is MpqDataSource` | Main | working | 057 |
| 6 | Load Loose Map Folder Against Saved Base → [client] | `QueueKnownGoodClientAction(attachLooseFolder: true)` | File → Load Loose Map Folder Against Saved Base | ✅ | ✅ | `_knownGoodClientPaths.Count > 0` | Main | working | 057 |
| 7 | Save Current Game Folder As Known-Good Base | `SaveCurrentGameFolderAsKnownGoodBase()` | File → Save Current Game Folder As Known-Good Base | ✅ | ✅ | `_dataSource is MpqDataSource` | Main | working | 057 |
| 8 | Forget Known-Good Base → [client] | `QueueForgetKnownGoodClientPath` | File → Forget Known-Good Base | ✅ | ✅ | `_knownGoodClientPaths.Count > 0` | Main | working | 057 |
| 9 | Open MK Dataset… | `_wantOpenVlmProject = true` | File → Open MK Dataset… | ✅ | ✅ | — | Main | working | 053/054 |
| 10 | Quit | `_window.Close()` | File → Quit | ✅ | ✅ | — | Main | working | 060 |

**Legacy-only (MdxViewer)**: None — parity achieved.

---

## 2. Menu Bar — View

| # | User Label | Source Method | Menu Entry | Tabbed Route | Legacy Route | Required State | Owner Frame | Status | Predecessor Spec |
|---|------------|---------------|------------|--------------|--------------|----------------|-------------|--------|------------------|
| 11 | Wireframe (W) | `_renderer?.ToggleWireframe()` | View → Wireframe | ✅ | ✅ | `_renderer != null` | Main | working | 060 |
| 12 | Reset Camera | `ResetCamera()` | View → Reset Camera | ✅ | ✅ | — | Main | working | 060 |
| 13 | Hide UI Chrome (Tab) | `_hideUiChrome = !_hideUiChrome` | View → Hide UI Chrome | ✅ | ✅ | — | Main | working | 060 |
| 14 | Tab System (069) | `_useTabUi = !_useTabUi` | View → Tab System (069) | ✅ | ✅ | — | Main | working | 069 |
| 15 | Dockable Shell Panels | `_useDockspaceUi = !_useDockspaceUi` | View → Dockable Shell Panels | ❌ (disabled when tabbed) | ✅ | `!_useTabUi` | Main | working | 069 |
| 16 | Left Sidebar | `_showLeftSidebar = !_showLeftSidebar` | View → Left Sidebar | ✅ | ✅ | — | Left | working | 071 |
| 17 | Right Sidebar (I) | `_showRightSidebar = !_showRightSidebar` | View → Right Sidebar | ✅ | ✅ | — | Right | working | 071 |
| 18 | Focus PM4 Tools (P) | `OpenPm4Workbench(Pm4WorkbenchTab.Selection)` | View → Focus PM4 Tools | ✅ | ✅ | — | Workbench | working | 049/051 |
| 19 | Reset Shell Layout | `ResetShellLayoutToDefaults()` | View → Reset Shell Layout | ✅ | ✅ | — | Main | working | 060 |
| 20 | File Browser | `_showFileBrowser = !_showFileBrowser` | View → File Browser | ✅ | ✅ | — | Left/Bottom | working | 060 |
| 21 | Model Info | `_showModelInfo = !_showModelInfo` | View → Model Info | ✅ | ✅ | — | Right/Workbench | working | 060 |
| 22 | Asset Catalog | `_catalogView?.Draw()` / `OpenWorkbenchTab(ToolsBottomTab.Utilities)` | View → Asset Catalog | ✅ (Utilities sub-tab) | ✅ (floating) | — | Workbench/Float | working | 060 |

**Legacy-only**: None.

---

## 3. Menu Bar — Tools

| # | User Label | Source Method | Menu Entry | Tabbed Route | Legacy Route | Required State | Owner Frame | Status | Predecessor Spec |
|---|------------|---------------|------------|--------------|--------------|----------------|-------------|--------|------------------|
| 23 | Settings… | `_showSettingsWindow = true` | Tools → Settings… | ✅ (fixed) | ✅ | — | Floating | **was misrouted, now working** | 080 |
| 24 | Open Zarr Dataset… | `_wantOpenZarrDataset = true` | Tools → Offline Data / Conversion → Open Zarr Dataset… | ✅ | ✅ | — | Dialog | working | 053/054 |
| 25 | Build ML Dataset… | `_showVlmExportDialog = true` | Tools → Offline Data / Conversion → Build ML Dataset… | ✅ | ✅ | — | Dialog | working | 053/054 |
| 26 | Train V7 Terrain Model… | `_showMlTrainingDialog = true` | Tools → Offline Data / Conversion → Train V7 Terrain Model… | ✅ | ✅ | — | Dialog | working | 053/054 |
| 27 | Terrain Texture Transfer… | `_showTerrainTextureTransferDialog = true` | Tools → Offline Data / Conversion → Terrain Texture Transfer… | ✅ | ✅ | — | Dialog | working | 053/054 |
| 28 | Map Converter… | `_showMapConverterDialog = true` | Tools → Offline Data / Conversion → Map Converter… | ✅ | ✅ | — | Dialog | working | 073b |
| 29 | WMO Converter… | `_showWmoConverterDialog = true` | Tools → Offline Data / Conversion → WMO Converter… | ✅ | ✅ | — | Dialog | working | 073b |

### 3.1 Tools → Panels

| # | User Label | Source Method | Menu Entry | Tabbed Route | Legacy Route | Required State | Owner Frame | Status | Predecessor Spec |
|---|------------|---------------|------------|--------------|--------------|----------------|-------------|--------|------------------|
| 30 | Model Info | `OpenWorkbenchTab(ModelBottomTab.Info)` | Tools → Panels → Model Info | ✅ (Model → Info) | ✅ (Right sidebar) | — | Workbench/Right | working | 060/071 |
| 31 | Log Viewer | `OpenWorkbenchTab(ToolsBottomTab.Utilities)` | Tools → Panels → Log Viewer | ✅ (Utilities → Log) | ✅ (floating) | — | Workbench/Float | working | 060 |
| 32 | Perf | `OpenWorkbenchTab(ToolsBottomTab.Utilities)` | Tools → Panels → Perf | ✅ (Utilities → Perf) | ✅ (floating) | — | Workbench/Float | working | 090 |
| 33 | Settings… | `_showSettingsWindow = true` | Tools → Panels → Settings… | ✅ (floating) | ✅ (floating) | — | Floating | working | 080 |
| 34 | Asset Catalog | `OpenWorkbenchTab(ToolsBottomTab.Utilities)` | Tools → Panels → Asset Catalog | ✅ (Utilities → Asset Catalog) | ✅ (floating) | — | Workbench/Float | working | 060 |
| 35 | Capture Automation | `OpenWorkbenchTab(ToolsBottomTab.Utilities)` | Tools → Panels → Capture Automation | ✅ (Utilities → Capture) | ✅ (floating) | — | Workbench/Float | working | 053/054 |
| 36 | Taxi | `OpenWorkbenchTab(ToolsBottomTab.Utilities)` | Tools → Panels → Taxi | ✅ (Utilities → Taxi) | ✅ (floating) | `_worldScene != null` | Workbench/Float | working | 053/054 |
| 37 | UniqueId Archeology | `OpenWorkbenchTab(ToolsBottomTab.Archeology)` | Tools → Panels → UniqueId Archeology | ✅ (Archeology) | ✅ (floating) | `_worldScene != null` | Workbench/Float | working | 049/051 |
| 38 | Chunk Clipboard | `OpenWorkbenchTab(ToolsBottomTab.Terrain)` | Tools → Panels → Chunk Clipboard | ✅ (Terrain → Clipboard) | ✅ (floating) | `hasTerrain` | Workbench/Float | working | 053/054 |
| 39 | Terrain Analysis | `OpenWorkbenchTab(ToolsBottomTab.Terrain)` | Tools → Panels → Terrain Analysis | ✅ (Terrain → Analysis) | ✅ (floating) | `hasTerrain` | Workbench/Float | working | 053/054 |
| 40 | MCNK Explorer | `OpenWorkbenchTab(ToolsBottomTab.Terrain)` | Tools → Panels → MCNK Explorer | ✅ (Terrain → MCNK) | ✅ (floating) | `hasTerrain` | Workbench/Float | working | 053/054 |
| 41 | Weak Signal | `OpenWorkbenchTab(ToolsBottomTab.Terrain)` | Tools → Panels → Weak Signal | ✅ (Terrain → Weak Signal) | ✅ (floating) | `hasTerrain` | Workbench/Float | working | 053/054 |

**Legacy-only**: None — all panels have tabbed routes.

---

## 4. Menu Bar — Tools → Export

| # | User Label | Source Method | Menu Entry | Tabbed Route | Legacy Route | Required State | Owner Frame | Status | Predecessor Spec |
|---|------------|---------------|------------|--------------|--------------|----------------|-------------|--------|------------------|
| 42 | Export GLB… | `_wantExportGlb = true` | Tools → Export → GLB → Export GLB… | ✅ | ✅ | `_renderer != null` | Dialog | working | 060 |
| 43 | Export GLB (Collision Only)… | `_wantExportGlbCollision = true` | Tools → Export → GLB → Export GLB (Collision Only)… | ✅ | ✅ | `_renderer != null` | Dialog | working | 060 |
| 44 | Current Tile (Terrain + Objects) | `_wantExportMapGlbTiles = true` (scoped) | Tools → Export → GLB → Map Tiles → Current Tile… | ✅ | ✅ | `canExportMapGlb` | Dialog | working | 060 |
| 45 | Loaded Tiles Folder | `_wantExportMapGlbTiles = true` (scoped) | Tools → Export → GLB → Map Tiles → Loaded Tiles Folder… | ✅ | ✅ | `canExportMapGlb` | Dialog | working | 060 |
| 46 | Whole Map Folder | `_wantExportMapGlbTiles = true` (scoped) | Tools → Export → GLB → Map Tiles → Whole Map Folder… | ✅ | ✅ | `canExportMapGlb` | Dialog | working | 060 |
| 47 | Current Tile Atlas (PNG)… | `_wantExportAlphaAtlas = true` | Tools → Export → Terrain → Alpha Masks → Current Tile Atlas… | ✅ | ✅ | `hasTerrain` | Dialog | working | 053/054 |
| 48 | Current Tile Chunks Folder… | `_wantExportAlphaChunks = true` | Tools → Export → Terrain → Alpha Masks → Current Tile Chunks Folder… | ✅ | ✅ | `hasTerrain` | Dialog | working | 053/054 |
| 49 | Loaded Tiles Folder… | `_wantExportAlphaTiles = true` | Tools → Export → Terrain → Alpha Masks → Loaded Tiles Folder… | ✅ | ✅ | `hasTerrain` | Dialog | working | 053/054 |
| 50 | Whole Map Folder… | `_wantExportAlphaMap = true` | Tools → Export → Terrain → Alpha Masks → Whole Map Folder… | ✅ | ✅ | `hasTerrain` | Dialog | working | 053/054 |
| 51 | Current Tile (257×257 L16 PNG + JSON)… | `_wantExportHeightmap = true` | Tools → Export → Terrain → Heightmaps → Current Tile… | ✅ | ✅ | `hasTerrain` | Dialog | working | 053/054 |
| 52 | Loaded Tiles Folder (per-tile)… | `_wantExportHeightmapTiles = true` | Tools → Export → Terrain → Heightmaps → Loaded Tiles Folder… | ✅ | ✅ | `hasTerrain` | Dialog | working | 053/054 |
| 53 | Whole Map Folder (per-map)… | `_wantExportHeightmapMap = true` | Tools → Export → Terrain → Heightmaps → Whole Map Folder… | ✅ | ✅ | `hasTerrain` | Dialog | working | 053/054 |
| 54 | Current Tile PNG… | `_wantExportMccv = true` | Tools → Export → Terrain → MCCV → Current Tile PNG… | ✅ | ✅ | `hasTerrain` | Dialog | working | 053/054 |
| 55 | Loaded Tiles Folder… | `_wantExportMccvTiles = true` | Tools → Export → Terrain → MCCV → Loaded Tiles Folder… | ✅ | ✅ | `hasTerrain` | Dialog | working | 053/054 |
| 56 | Whole Map Folder… | `_wantExportMccvMap = true` | Tools → Export → Terrain → MCCV → Whole Map Folder… | ✅ | ✅ | `hasTerrain` | Dialog | working | 053/054 |

**Legacy-only**: None.

---

## 5. Menu Bar — Tools → Import

| # | User Label | Source Method | Menu Entry | Tabbed Route | Legacy Route | Required State | Owner Frame | Status | Predecessor Spec |
|---|------------|---------------|------------|--------------|--------------|----------------|-------------|--------|------------------|
| 57 | From Folder of Tile Atlases… | `_wantTerrainImport = true; _terrainImportKind = AlphaFolder` | Tools → Import → Terrain → Alpha Masks → From Folder… | ✅ | ✅ | `hasTerrain` | Dialog | working | 053/054 |
| 58 | From Folder of Tile Heightmaps… | `_wantTerrainImport = true; _terrainImportKind = Heightmap257Folder` | Tools → Import → Terrain → Heightmaps → From Folder… | ✅ | ✅ | `hasTerrain` | Dialog | working | 053/054 |
| 59 | From Folder of Tile MCCV PNGs… | `_wantTerrainImport = true; _terrainImportKind = MccvFolder` | Tools → Import → Terrain → MCCV → From Folder… | ✅ | ✅ | `hasTerrain` | Dialog | working | 053/054 |

**Legacy-only**: None.

---

## 6. Menu Bar — Help

| # | User Label | Source Method | Menu Entry | Tabbed Route | Legacy Route | Required State | Owner Frame | Status | Predecessor Spec |
|---|------------|---------------|------------|--------------|--------------|----------------|-------------|--------|------------------|
| 60 | About | `_openAboutPopup = true` | Help → About | ✅ | ✅ | — | Modal | working | 060 |

---

## 7. Toolbar (Top Center)

| # | User Label | Source Method | Toolbar Entry | Tabbed Route | Legacy Route | Required State | Owner Frame | Status | Predecessor Spec |
|---|------------|---------------|---------------|--------------|--------------|----------------|-------------|--------|------------------|
| 61 | Auto / ADT / WMO / M2 (visual investigation mode) | `DrawVisualInvestigationModeButton` | Toolbar buttons | ✅ | ✅ | — | Toolbar | working | 053/054 |
| 62 | Terrain toolbar controls (wireframe, normals, grid, etc.) | `DrawDirectTerrainToolbarControls` | Toolbar (centered when world loaded) | ✅ | ✅ | `hasTerrain` | Toolbar | working | 053/054 |
| 63 | Workspace toolbar (Viewer/Editor, task tabs) | `DrawWorkspaceToolbarControls` | Toolbar (left) | ✅ | ✅ | — | Toolbar | working | 069/071 |

---

## 8. Bottom Bar (FixedBottomDrawerTab)

| # | User Label | Source Method | Tab | Tabbed Route | Legacy Route | Required State | Owner Frame | Status | Predecessor Spec |
|---|------------|---------------|-----|--------------|--------------|----------------|-------------|--------|------------------|
| 64 | Workspace | `DrawWorkspaceBarsPanelContent` | Workspace | ✅ | ✅ | — | Bottom | working | 069/071 |
| 65 | Terrain | `DrawTerrainWorkbenchSelectionContent` + `DrawTerrainControlsAdjustmentContent` | Terrain | ✅ | ✅ | `hasTerrain` | Bottom | working | 053/054 |
| 66 | PM4 | `DrawPm4WorkbenchInspector` | PM4 | ✅ | ✅ | `_worldScene != null` | Bottom | working | 049/051 |
| 67 | World | `DrawWorldSubTabContent` (Source/Placements/Tiles/Selection/LOD) | World | ✅ | ✅ | `_worldScene != null` | Bottom | working | 071/080 |
| 68 | Diagnostics | `DrawUtilitiesSubTabContent` (Minimap/Log/Perf/RenderQuality/Taxi/Capture/RuntimeStats) | Diagnostics | ✅ | ✅ | — | Bottom | working | 090/093 |

---

## 9. Left Sidebar (Legacy) / Navigator (Tabbed)

| # | User Label | Source Method | Panel | Tabbed Route | Legacy Route | Required State | Owner Frame | Status | Predecessor Spec |
|---|------------|---------------|-------|--------------|--------------|----------------|-------------|--------|------------------|
| 69 | World Overview (minimap + map list) | `DrawWorldOverviewContent` | Navigator | ✅ | ✅ | — | Left | working | 057/071 |
| 70 | File Browser | `DrawFileBrowserContent` | Navigator | ✅ | ✅ | — | Left | working | 060 |
| 71 | World Maps (discovery) | `DrawMapDiscoveryContent` | Navigator | ✅ | ✅ | `_discoveredMaps.Count > 0` | Left | working | 057 |
| 72 | Runtime Stats | `DrawRuntimeStatsPanelContent` | Navigator (collapsible) | ✅ | ✅ | — | Left | working | 090 |

---

## 10. Right Sidebar (Legacy) / Inspector + Workbench (Tabbed)

| # | User Label | Source Method | Panel | Tabbed Route | Legacy Route | Required State | Owner Frame | Status | Predecessor Spec |
|---|------------|---------------|-------|--------------|--------------|----------------|-------------|--------|------------------|
| 73 | Viewer Settings | `DrawUnifiedViewerSettingsSidebarContent` | Inspector | ✅ (Model → Info) | ✅ | — | Right | working | 060/071 |
| 74 | Selection Summary | `DrawViewerSelectionSummary` | Inspector | ✅ (World → Selection) | ✅ | — | Right | working | 071 |
| 75 | Camera Controls | `DrawCameraControlsContent` | Inspector | ✅ (Model → Info / World → Selection) | ✅ | — | Right | working | 060/071 |
| 76 | Model Info | `DrawModelInfoPanelContent` | Inspector | ✅ (Model → Info) | ✅ | `_loadedMdx != null || _loadedM2Runtime != null` | Right | working | 060/071 |
| 77 | World Objects | `DrawWorldObjectsPanelContent` | Inspector | ✅ (World → Placements) | ✅ | `_worldScene != null` | Right | working | 071 |
| 78 | Terrain Controls | `DrawTerrainControlsPanelContent` | Inspector | ✅ (Terrain tab) | ✅ | `hasTerrain` | Right | working | 053/054 |
| 79 | Runtime Stats | `DrawRuntimeStatsPanelContent` | Inspector | ✅ (Diagnostics → RuntimeStats) | ✅ | — | Right | working | 090 |
| 80 | PM4 Workbench | `DrawPm4WorkbenchInspector` | Inspector | ✅ (PM4 tab) | ✅ | `_worldScene != null` | Right | working | 049/051 |

---

## 11. Floating Windows (Legacy) / Sub-tabs (Tabbed)

| # | User Label | Source Method | Flag | Tabbed Route | Legacy Route | Required State | Status | Predecessor Spec |
|---|------------|---------------|------|--------------|--------------|----------------|--------|------------------|
| 81 | Settings | `DrawSettingsWindow` | `_showSettingsWindow` | ✅ (floating, **fixed**) | ✅ | — | **was misrouted, now working** | 080 |
| 82 | Log Viewer | `DrawLogViewer` | `_showLogViewer` | ✅ (Utilities → Log) | ✅ | — | working | 060 |
| 83 | WDL Preview | `DrawWdlPreviewDialog` | `_showWdlPreview` | ✅ (floating) | ✅ | — | working | 053/054 |
| 84 | Minimap | `DrawMinimapWindow` | `IsShellPanelActive(Minimap)` | ✅ (Diagnostics → Minimap) | ✅ | — | working | 057/090 |
| 85 | Perf | `DrawPerfWindow` | `_showPerfWindow` | ✅ (Utilities → Perf) | ✅ | — | working | 090 |
| 86 | Render Quality | `DrawRenderQualityWindow` | `_showRenderQualityWindow` | ✅ (Utilities → RenderQuality) | ✅ | — | working | 053/054 |
| 87 | Terrain Tools | `DrawTerrainToolsWindow` | `_showTerrainToolsWindow` | ❌ (routed to Terrain tab) | ✅ | `hasTerrain` | **misrouted in tabbed** | 053/054 |
| 88 | Chunk Clipboard | `DrawChunkClipboardWindow` | `_showChunkClipboardWindow` | ✅ (Terrain → Clipboard) | ✅ | `hasTerrain` | working | 053/054 |
| 89 | Terrain Analysis | `DrawTerrainAnalysisWindow` | `_showTerrainAnalysisWindow` | ✅ (Terrain → Analysis) | ✅ | `hasTerrain` | working | 053/054 |
| 90 | MCNK Explorer | `DrawMcnkExplorerWindow` | `_showMcnkExplorerWindow` | ✅ (Terrain → MCNK) | ✅ | `hasTerrain` | working | 053/054 |
| 91 | Capture Automation | `DrawCaptureAutomationWindow` | `_showCaptureAutomationWindow` | ✅ (Utilities → Capture) | ✅ | — | working | 053/054 |
| 92 | PM4 Alignment | `DrawPm4AlignmentWindow` | `_showPm4AlignmentWindow` | ❌ (no tab route) | ✅ | — | **missing in tabbed** | 049/051 |
| 93 | UniqueId Archeology | `DrawUniqueIdArchaeologyWindow` | `_showUniqueIdArchaeologyWindow` | ✅ (Archeology) | ✅ | `_worldScene != null` | working | 049/051 |
| 94 | Taxi | `DrawTaxiWindow` | `_showTaxiWindow` | ✅ (Utilities → Taxi) | ✅ | `_worldScene != null` | working | 053/054 |
| 95 | Weak Signal | `DrawWeakSignalWindow` | `_showWeakSignalWindow` | ✅ (Terrain → Weak Signal) | ✅ | `hasTerrain` | working | 053/054 |

---

## 12. Modal Dialogs

| # | User Label | Source Method | Trigger | Tabbed Route | Legacy Route | Status | Predecessor Spec |
|---|------------|---------------|---------|--------------|--------------|--------|------------------|
| 96 | Folder Input | `DrawFolderInputDialog` | `_showFolderInput` | ✅ | ✅ | working | 060 |
| 97 | Build Selection | `DrawBuildSelectionDialog` | `_showBuildSelectionDialog` | ✅ | ✅ | working | 057 |
| 98 | Listfile Input | `DrawListfileInputDialog` | `_showListfileInput` | ✅ | ✅ | working | 060 |
| 99 | ML Training Monitor | `UpdateMlTrainingMonitor` / `DrawMlTrainingDialog` | `_showMlTrainingDialog` | ✅ | ✅ | working | 053/054 |
| 100 | VLM Export | `DrawVlmExportDialog` | `_showVlmExportDialog` | ✅ | ✅ | working | 053/054 |
| 101 | Terrain Texture Transfer | `DrawTerrainTextureTransferDialog` | `_showTerrainTextureTransferDialog` | ✅ | ✅ | working | 053/054 |
| 102 | Alpha Folder Import Scope | `DrawAlphaFolderImportScopeDialog` | `_showAlphaFolderImportScope` | ✅ | ✅ | working | 053/054 |
| 103 | Heightmap Folder Import Scope | `DrawHeightmapFolderImportScopeDialog` | `_showHeightmapFolderImportScope` | ✅ | ✅ | working | 053/054 |
| 104 | MCCV Folder Import Scope | `DrawMccvFolderImportScopeDialog` | `_showMccvFolderImportScope` | ✅ | ✅ | working | 053/054 |
| 105 | Map Converter | `DrawMapConverterDialog` | `_showMapConverterDialog` | ✅ | ✅ | working | 073b |
| 106 | WMO Converter | `DrawWmoConverterDialog` | `_showWmoConverterDialog` | ✅ | ✅ | working | 073b |

---

## 13. Overlays (Viewport)

| # | User Label | Source Method | Trigger | Tabbed Route | Legacy Route | Status | Predecessor Spec |
|---|------------|---------------|---------|--------------|--------------|--------|------------------|
| 107 | Scene Hover Asset Overlay | `DrawSceneHoverAssetOverlay` | Always | ✅ | ✅ | working | 053/054 |
| 108 | Click Selection Overlay | `DrawClickSelectionOverlay` | Always | ✅ | ✅ | working | 053/054 |
| 109 | Editor Overlays (chunk clipboard, MCNK flags) | `DrawEditorOverlays` | Editor mode | ✅ | ✅ | working | 053/054 |
| 110 | Standalone WMO Group Overlay | `DrawStandaloneWmoGroupOverlay` | Standalone WMO | ✅ | ✅ | working | 071 |
| 111 | Standalone WMO Group Labels | `DrawStandaloneHighlightedWmoGroupLabels` | Standalone WMO | ✅ | ✅ | working | 071 |

---

## 14. Workbench Tabs (Tabbed Mode Only)

### 14.1 Model Tab (Bottom Sub-tabs)

| # | Sub-tab | Source Method | Status | Predecessor Spec |
|---|---------|---------------|--------|------------------|
| 112 | Info | `DrawModelInfoSubTab` → `DrawModelInfoCoreContent` | working | 060/071 |
| 113 | Animations | `DrawModelAnimationsSubTab` → `DrawModelAnimationControls` | working | 053 |
| 114 | Actions | `DrawModelActionsSubTab` | working | 053 |
| 115 | LOD | `DrawModelLodSubTab` | **placeholder** (stub) | 056 |

### 14.2 World Tab (Bottom Sub-tabs)

| # | Sub-tab | Source Method | Status | Predecessor Spec |
|---|---------|---------------|--------|------------------|
| 116 | Source | `DrawWorldSourceSubTab` → `DrawWorkspaceBarsPanelContent` + `DrawFileBrowserContent` + `DrawMapDiscoveryContent` | working | 057/071 |
| 117 | Placements | `DrawWorldPlacementsSubTab` → `DrawWorldObjectsContentCore` | working | 071 |
| 118 | Tiles | `DrawWorldTilesSubTab` → `DrawTerrainWorkbenchSelectionContent` + `DrawTerrainControlsAdjustmentContent` | working | 053/054 |
| 119 | Selection Tools | `DrawWorldSelectionToolsSubTab` → `DrawSelectedObjectSummaryContent` | working | 071 |
| 120 | LOD | `DrawWorldLodSubTab` | **placeholder** (first pass only) | 056/080 |

### 14.3 Tools Tab (Bottom Sub-tabs)

| # | Sub-tab | Source Method | Status | Predecessor Spec |
|---|---------|---------------|--------|------------------|
| 121 | Quick Controls | `DrawQuickControlsContent` | working | 060 |
| 122 | Archeology | `DrawArcheologySubTabContent` (Range/Layers/Playback/Capture) | working | 049/051 |
| 123 | Terrain | `DrawTerrainSubTabContent` (Clipboard/Analysis/MCNK/WeakSignal/Export) | working | 053/054 |
| 124 | Utilities | `DrawUtilitiesSubTabContent` (Minimap/Log/Perf/RenderQuality/Taxi/Capture/RuntimeStats) | working | 090/093 |
| 125 | Converters | **NOT IMPLEMENTED** | **missing** | 073b |

---

## 15. Legacy-Only Surfaces (MdxViewer) — Not Yet in wow-viewer

| # | Surface | Legacy Source | wow-viewer Equivalent | Disposition |
|---|---------|---------------|----------------------|-------------|
| L1 | PM4 Workbench tabs: Overlay / Selection / Correlation | `Pm4WorkbenchTab` enum + `DrawPm4WorkbenchInspector` | PM4 bottom tab (Selection only) | **Missing** — Overlay & Correlation tabs not ported |
| L2 | Editor Workspace tasks: Terrain / Objects / PM4 Evidence / Inspect / Publish | `EditorWorkspaceTask` enum + `DrawEditor*Workspace` | Partial (Terrain, Objects, Inspect, Publish) | **Missing** — PM4 Evidence task not ported |
| L3 | Shell Panels: Pm4Workbench, TerrainControls, RuntimeStats, WorldObjects, ModelInfo, Minimap, WorkspaceBars, Pm4Info, Pm4SceneGraph | `ShellPanelId` enum (11 panels) | 8 panels ported; Pm4Info, Pm4SceneGraph missing | **Missing** — 2 panels |
| L4 | FixedBottomDrawerTab: Diagnostics | `FixedBottomDrawerTab.Diagnostics` | Diagnostics bottom tab exists | **Working** |
| L5 | Weak Signal Amplifier window | `DrawWeakSignalWindow` | Terrain → Weak Signal sub-tab | **Working** |
| L6 | VLM Project Loader UI | `_wantOpenVlmProject` + `VlmProjectLoader` | File → Open MK Dataset… | **Working** |
| L7 | ML Training dialog (full) | `DrawMlTrainingDialog` | Tools → Offline Data → Train V7… | **Working** |
| L8 | Terrain Texture Transfer dialog | `DrawTerrainTextureTransferDialog` | Tools → Offline Data → Terrain Texture Transfer… | **Working** |
| L9 | Map Converter dialog | `DrawMapConverterDialog` | Tools → Offline Data → Map Converter… | **Working** |
| L10 | WMO Converter dialog | `DrawWmoConverterDialog` | Tools → Offline Data → WMO Converter… | **Working** |

---

## 16. Hotkeys

| Key | Action | Tabbed | Legacy | Status |
|-----|--------|--------|--------|--------|
| W | Toggle Wireframe | ✅ | ✅ | working |
| Tab | Toggle UI Chrome | ✅ | ✅ | working |
| I | Toggle Right Sidebar | ✅ | ✅ | working |
| P | Focus PM4 Tools | ✅ | ✅ | working |
| M | Toggle Fullscreen Minimap | ✅ | ✅ | working |

---

## 17. Classification Summary

| Status | Count |
|--------|-------|
| working | 98 |
| misrouted | 3 (Settings-fixed, Terrain Tools, PM4 Alignment) |
| missing | 5 (PM4 Overlay/Correlation tabs, PM4 Evidence task, Pm4Info/Pm4SceneGraph panels, Converters sub-tab) |
| placeholder | 2 (Model LOD, World LOD) |
| duplicate | 0 |
| retired | 0 |
| disabled-with-reason | 1 (Dockable Shell Panels when tabbed) |

---

## 18. Phase 0 Gate Status

**Gate**: No feature migration starts until there are zero unclassified visible controls.

- ✅ All 126 inventoried surfaces classified
- ⚠️ 3 misrouted (Settings fixed in this session; Terrain Tools and PM4 Alignment remain)
- ⚠️ 5 missing (converter surface + PM4 workbench gaps)
- ⚠️ 2 placeholder (LOD tabs)

**Next**: Phase 1 — Route Integrity and Dead-Control Repair
1. Fix Terrain Tools floating window in tabbed mode (route to Terrain tab)
2. Fix PM4 Alignment window (route to PM4 tab or retire with tooltip)
3. Implement Converters sub-tab (073b)
4. Port PM4 Overlay/Correlation tabs and PM4 Evidence task
5. Replace placeholder LOD tabs with factual content or disable with reason
