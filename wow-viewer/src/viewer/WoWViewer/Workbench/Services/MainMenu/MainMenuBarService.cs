using System.Diagnostics;
using System.Numerics;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Text.Json;
using ImGuiNET;
using WowViewer.Core.IO.Mdx;
using WoWViewer.DataSources;
using WoWViewer.Export;
using WoWViewer.Logging;
using WoWViewer.Rendering;
using WoWViewer.Catalog;
using WoWViewer.Capture;
using WoWViewer.Population;
using WoWViewer.Terrain;
using Silk.NET.Input;
using Silk.NET.Maths;
using Silk.NET.OpenGL;
using Silk.NET.OpenGL.Extensions.ImGui;
using Silk.NET.Windowing;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WoWViewer.Terrain.Vlm;
using WowViewer.Core.IO.M2;
using WowViewer.Core.IO.M2Chunked;
using WowViewer.Core.IO.M2Era1121;
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.M2;
using WoWViewer.Terrain.Vlm;
using WowViewer.Core.Runtime.M2;
using WowViewer.Core.Runtime.Marketing;
using WowViewer.Core.Runtime.World.Visibility;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;
using WowViewer.Core.IO.Converters;
using WowViewer.Core.IO.Maps;
using WoWViewer.Terrain.Vlm;
using WoWViewer.Workbench;
using CoreMdxCollisionSummary = WowViewer.Core.Mdx.MdxCollisionSummary;
using CoreMdxGeometryFile = WowViewer.Core.Mdx.MdxGeometryFile;
using CoreMdxSummary = WowViewer.Core.Mdx.MdxSummary;
using CorePm4DocumentReader = WowViewer.Core.PM4.Services.Pm4ResearchReader;
using Pm4CoordinateService = WowViewer.Core.PM4.Services.Pm4CoordinateService;
using static WoWViewer.ViewerApp;
using static WoWViewer.CameraPathsService;

namespace WoWViewer;

/// <summary>
/// Main menu bar: File/View/Tools/Export/Help menus and the dialog-input preparation they trigger.
/// Extracted verbatim from <see cref="ViewerApp"/> (Epic 251 U-01). World and app state it
/// needs comes only through <see cref="IViewerAppHost"/>; the bridge members below keep the
/// names the moved code used inside ViewerApp, so no moved body was edited.
/// </summary>
internal sealed partial class MainMenuBarService
{
    private readonly IViewerAppHost _host;

    internal MainMenuBarService(IViewerAppHost host)
    {
        _host = host;
    }

    // Host bridge: see MainMenuBarService.Host.cs.

    private const string ViewerAboutPopupTitle = "About WoWViewer";
    private bool _openAboutPopup;
    private bool _wantAttachLooseMapFolder = false;
    private bool _wantOpenWdtFile = false;
    private bool _wantOpenPm4File = false;
    private bool _wantOpenVlmProject = false;
    private bool _wantOpenZarrDataset = false;

    internal void DrawMenuBar()
    {
        if (ImGui.BeginMainMenuBar())
        {
            if (ImGui.BeginMenu("File"))
            {
                if (ImGui.MenuItem("Open File..."))
                    _wantOpenFile = true;

                if (ImGui.MenuItem("Open Alpha WDT (loose map)..."))
                    _wantOpenWdtFile = true;

                if (ImGui.MenuItem("Open Loose PM4 / PD4 File..."))
                    _wantOpenPm4File = true;

                ImGui.Separator();

                if (ImGui.MenuItem("Open Game Folder (MPQ)..."))
                {
                    _showFolderInput = true;
                    _folderInputBuf = string.IsNullOrWhiteSpace(_lastGameFolderPath) ? "" : _lastGameFolderPath;
                }

                if (ImGui.MenuItem("Open CASC Install (local)..."))
                    _cascAhdrSource._wantOpenCascInstall = true;

                if (ImGui.IsItemHovered())
                    ImGui.SetTooltip("Pick a Battle.net install folder, then the game version to load. Reads only what is on disk.");

                if (ImGui.MenuItem("Open CASC Install (local + CDN fill)..."))
                    _cascAhdrSource._wantOpenCascInstallWithCdnFill = true;

                if (ImGui.IsItemHovered())
                    ImGui.SetTooltip("Pick a Battle.net install folder, then the game version to load. Files the build lists but the install does not have on disk are downloaded from Blizzard's CDN for the same build.");

                // Spec 247: every DAT action lives under one submenu instead of four peers in File.
                if (ImGui.BeginMenu("DAT Terrain (v22/23/26)"))
                {
                    if (ImGui.MenuItem("Open DAT Terrain Folder..."))
                        _cascAhdrSource._wantOpenAhdrTerrainFolder = true;

                    ImGui.Separator();
                    ImGui.TextDisabled("Export format");
                    Terrain.MapExportFormats.DrawCheckboxes("filemenu");

                    bool datLoaded = _terrainManager?.Adapter is AhdrTerrainAdapter;
                    if (ImGui.MenuItem("Export Loaded DAT Map...", null, false, datLoaded && Terrain.MapExportFormats.Any))
                        _cascAhdrSource.ExportLoadedDatMap();

                    if (ImGui.IsItemHovered())
                    {
                        ImGui.SetTooltip(!datLoaded
                            ? "Open a DAT terrain folder first."
                            : !Terrain.MapExportFormats.Any
                                ? "Tick at least one output format above."
                                : $"Writes the loaded DAT folder as {Terrain.MapExportFormats.Summary}, plus a manifest naming every field carried and dropped.");
                    }

                    ImGui.Separator();
                    if (ImGui.MenuItem("Export Nearby Tiles as DAT v26 (experimental)", null, false, _terrainManager?.Adapter is StandardTerrainAdapter))
                        _cascAhdrSource.ExportNearbyTilesAsDatV26(radius: 2);

                    if (ImGui.IsItemHovered())
                        ImGui.SetTooltip("Writes the ADT tiles within 2 tiles of the camera as DAT v26 files (terrain, texture layers, vertex colours, normals, objects) to output/dat_v26_export. Reopen with Open DAT Terrain Folder to compare.");

                    _cascAhdrSource.DrawAhdrHeightScaleMenu();
                    ImGui.EndMenu();
                }

                if (ImGui.BeginMenu("Open Saved Game Folder", _knownGoodClientPaths.Count > 0))
                {
                    foreach (var knownClient in _knownGoodClientPaths)
                    {
                        if (ImGui.MenuItem($"{knownClient.Name}##open_saved_{knownClient.Path}"))
                            _clientDialogs.QueueKnownGoodClientAction(knownClient.Path, knownClient.BuildVersion, attachLooseFolder: false);

                        if (ImGui.IsItemHovered())
                            ImGui.SetTooltip(ClientDialogsService.BuildKnownGoodClientTooltip(knownClient));
                    }

                    ImGui.EndMenu();
                }

                if (ImGui.MenuItem("Attach Loose Map Folder...", "", false, _dataSource is MpqDataSource))
                    _wantAttachLooseMapFolder = true;

                if (ImGui.BeginMenu("Load Loose Map Folder Against Saved Base", _knownGoodClientPaths.Count > 0))
                {
                    foreach (var knownClient in _knownGoodClientPaths)
                    {
                        if (ImGui.MenuItem($"{knownClient.Name}##attach_saved_{knownClient.Path}"))
                            _clientDialogs.QueueKnownGoodClientAction(knownClient.Path, knownClient.BuildVersion, attachLooseFolder: true);

                        if (ImGui.IsItemHovered())
                            ImGui.SetTooltip(ClientDialogsService.BuildKnownGoodClientTooltip(knownClient));
                    }

                    ImGui.EndMenu();
                }

                ImGui.Separator();

                if (ImGui.MenuItem("Save Current Game Folder As Known-Good Base", "", false, _dataSource is MpqDataSource))
                    _clientDialogs.SaveCurrentGameFolderAsKnownGoodBase();

                if (ImGui.BeginMenu("Forget Known-Good Base", _knownGoodClientPaths.Count > 0))
                {
                    foreach (var knownClient in _knownGoodClientPaths)
                    {
                        if (ImGui.MenuItem($"{knownClient.Name}##forget_saved_{knownClient.Path}"))
                            _settings.QueueForgetKnownGoodClientPath(knownClient);

                        if (ImGui.IsItemHovered())
                            ImGui.SetTooltip(ClientDialogsService.BuildKnownGoodClientTooltip(knownClient));
                    }

                    ImGui.EndMenu();
                }

                ImGui.Separator();

                if (ImGui.MenuItem("Settings..."))
                    _showSettingsWindow = true;

                ImGui.Separator();

                if (ImGui.MenuItem("Quit"))
                    _window.Close();

                ImGui.EndMenu();
            }

            if (ImGui.BeginMenu("View"))
            {
                if (ImGui.MenuItem("Wireframe", "W"))
                    _renderer?.ToggleWireframe();

                if (ImGui.MenuItem("Reset Camera"))
                    ResetCamera();

                if (ImGui.MenuItem("Hide UI Chrome", "Tab", _hideUiChrome))
                    _hideUiChrome = !_hideUiChrome;

                ImGui.Separator();

                if (ImGui.MenuItem("Tab System (069)", "", ref _useTabUi))
                {
                    // Save preference so it sticks across restarts.
                    _settings.SaveViewerSettings();
                }

                bool useDockspaceUi = _useDockspaceUi;
                if (_useTabUi) ImGui.BeginDisabled();
                if (ImGui.MenuItem("Dockable Shell Panels", "", ref useDockspaceUi))
                {
                    _useDockspaceUi = useDockspaceUi;
                    _forceApplyShellPanelLayout = _useDockspaceUi;
                    _settings.SaveViewerSettings();
                }
                if (_useTabUi) ImGui.EndDisabled();

                ImGui.MenuItem("Left Sidebar", "", ref _showLeftSidebar);
                ImGui.MenuItem("Right Sidebar", "I", ref _showRightSidebar);
                if (ImGui.MenuItem("Log Console..."))
                {
                    if (_useTabUi) OpenWorkbenchTab(UtilitiesBottomTab.Log);
                    else _showLogViewer = true;
                }
                if (ImGui.MenuItem("Performance & Profiling..."))
                {
                    if (_useTabUi) OpenWorkbenchTab(UtilitiesBottomTab.Perf);
                    else _showPerfWindow = true;
                }
                if (ImGui.MenuItem("Lighting Diagnostics..."))
                {
                    if (_useTabUi) OpenWorkbenchTab(UtilitiesBottomTab.Lighting);
                    else OpenLegacyWorkbenchUtility(UtilitiesBottomTab.Lighting);
                }
                if (ImGui.MenuItem("Focus PM4 Tools", "P"))
                    OpenPm4Workbench(Pm4WorkbenchTab.Selection);
                if (ImGui.MenuItem("Reset Shell Layout"))
                    _shellLayout.ResetShellLayoutToDefaults();
                ImGui.Separator();
                ImGui.MenuItem("File Browser", "", ref _showFileBrowser);
                ImGui.MenuItem("Model Info", "", ref _showModelInfo);
                ImGui.Separator();
                if (ImGui.MenuItem("Asset Catalog"))
                {
                    if (_useTabUi)
                        OpenWorkbenchTab(UtilitiesBottomTab.AssetCatalog);
                    else
                    {
                        if (_catalogView == null)
                        {
                            _catalogView = new AssetCatalogView(_gl);
                            _catalogView.SetDataSource(_dataSource);
                            _catalogView.OnLoadModelRequested = _modelLoader.OnCatalogLoadModel;
                        }
                        _catalogView.IsVisible = !_catalogView.IsVisible;
                    }
                }

                ImGui.EndMenu();
            }

            if (ImGui.BeginMenu("Tools"))
            {
                if (ImGui.MenuItem("Taxi Routes...", "", false, _worldScene != null))
                {
                    if (_useTabUi) OpenWorkbenchTab(UtilitiesBottomTab.Taxi);
                    else OpenLegacyWorkbenchUtility(UtilitiesBottomTab.Taxi);
                }
                if (ImGui.MenuItem("Audio Settings..."))
                {
                    if (_useTabUi) OpenWorkbenchTab(UtilitiesBottomTab.Audio);
                    else OpenLegacyWorkbenchUtility(UtilitiesBottomTab.Audio);
                }
                ImGui.Separator();
                // 071: floating-window toggles removed. Every tool lives in a
                // workbench tab under Tools > Panels or the relevant top tab.

                if (ImGui.BeginMenu("Converters"))
                {
                    if (ImGui.MenuItem("Map Converter..."))
                    {
                        PrepareMapConverterDialogInputs();
                        _showMapConverterDialog = true;
                    }

                    if (ImGui.MenuItem("WMO Converter..."))
                    {
                        PrepareWmoConverterDialogInputs();
                        _showWmoConverterDialog = true;
                    }

                    ImGui.EndMenu();
                }

                ImGui.Separator();

                if (ImGui.BeginMenu("Panels"))
                {
                    bool hasTerrain = _terrainManager != null || _vlmTerrainManager != null;
                    bool hasWorld = _worldScene != null;

                    if (ImGui.MenuItem("Model Info"))
                        OpenWorkbenchTab(ModelBottomTab.Info);

                    ImGui.Separator();

                    if (ImGui.MenuItem("Log Viewer"))
                        OpenWorkbenchTab(UtilitiesBottomTab.Log);
                    if (ImGui.MenuItem("Perf"))
                        OpenWorkbenchTab(UtilitiesBottomTab.Perf);
                    if (ImGui.MenuItem("Settings..."))
                        _showSettingsWindow = true;

                    ImGui.Separator();

                    if (ImGui.MenuItem("Asset Catalog"))
                        OpenWorkbenchTab(UtilitiesBottomTab.AssetCatalog);
                    if (ImGui.MenuItem("Capture Automation"))
                        OpenCapturePanelTab(CapturePanelTab.Automation);
                    if (ImGui.MenuItem("Camera Path"))
                        OpenCapturePanelTab(CapturePanelTab.CameraPath);
                    if (ImGui.MenuItem("Taxi", hasWorld))
                        OpenWorkbenchTab(UtilitiesBottomTab.Taxi);

                    ImGui.Separator();

                    if (ImGui.MenuItem("Weak Signal & Stratigraphy", hasTerrain))
                        OpenWorkbenchTab(WorkbenchTab.Archaeology, 0);
                    if (ImGui.MenuItem("UniqueId Archaeology", hasWorld))
                        OpenWorkbenchTab(WorkbenchTab.Archaeology, 1);
                    if (ImGui.MenuItem("PM4 Analysis", hasWorld))
                        OpenWorkbenchTab(WorkbenchTab.Archaeology, 4);
                    if (ImGui.MenuItem("Cartography", hasTerrain))
                        OpenWorkbenchTab(WorkbenchTab.Archaeology, 5);

                    ImGui.Separator();

                    if (ImGui.MenuItem("Editor Workbench", hasTerrain || hasWorld))
                        OpenWorkbenchTab(WorkbenchTab.Editor, 0);

                    ImGui.EndMenu();
                }

                ImGui.Separator();

                if (ImGui.BeginMenu("Export"))
                {
                    if (ImGui.MenuItem("Synthesized Terrain Minimap..."))
                    {
                        PrepareSynthesizedMinimapExportDialogInputs();
                        _showSynthesizedMinimapExportDialog = true;
                    }

                    ImGui.Separator();

                    if (ImGui.BeginMenu("GLB"))
                    {
                        // Terrain (including DAT folders opened with no client data source) can export
                        // GLB too, so the menu is enabled for either a standalone model or a terrain.
                        bool canExportGlb = _renderer != null || _terrainManager != null;
                        if (ImGui.MenuItem("Export GLB...", canExportGlb))
                            _wantExportGlb = true;
                        if (ImGui.MenuItem("Export GLB (Collision Only)...", _renderer != null))
                            _wantExportGlbCollision = true;

                        ImGui.Separator();

                        bool canExportMapGlb = _terrainManager != null;
                        if (ImGui.BeginMenu("Map Tiles", canExportMapGlb))
                        {
                            if (ImGui.MenuItem("Current Tile (Terrain + Objects)", "", false, canExportMapGlb))
                            {
                                _mapGlbScope = TerrainTileScope.CurrentTile;
                                _wantExportMapGlbTiles = true;
                            }
                            if (ImGui.MenuItem("Loaded Tiles Folder", "", false, canExportMapGlb))
                            {
                                _mapGlbScope = TerrainTileScope.LoadedTiles;
                                _wantExportMapGlbTiles = true;
                            }
                            if (ImGui.MenuItem("Whole Map Folder", "", false, canExportMapGlb))
                            {
                                _mapGlbScope = TerrainTileScope.WholeMap;
                                _wantExportMapGlbTiles = true;
                            }
                            ImGui.EndMenu();
                        }

                        ImGui.EndMenu();
                    }

                    if (ImGui.BeginMenu("Terrain"))
                    {
                        bool hasTerrain = _terrainManager != null || _vlmTerrainManager != null;

                        if (ImGui.BeginMenu("Alpha Masks"))
                        {
                            if (ImGui.MenuItem("Current Tile Atlas (PNG)...", hasTerrain))
                            {
                                _wantTerrainExport = true;
                                _terrainExportKind = TerrainExportKind.AlphaCurrentTileAtlas;
                            }

                            if (ImGui.MenuItem("Current Tile Chunks Folder...", hasTerrain))
                            {
                                _wantTerrainExport = true;
                                _terrainExportKind = TerrainExportKind.AlphaCurrentTileChunksFolder;
                            }

                            if (ImGui.MenuItem("Loaded Tiles Folder...", hasTerrain))
                            {
                                _wantTerrainExport = true;
                                _terrainExportKind = TerrainExportKind.AlphaLoadedTilesFolder;
                            }

                            if (ImGui.MenuItem("Whole Map Folder...", hasTerrain))
                            {
                                _wantTerrainExport = true;
                                _terrainExportKind = TerrainExportKind.AlphaWholeMapFolder;
                            }

                            ImGui.EndMenu();
                        }

                        if (ImGui.BeginMenu("Heightmaps"))
                        {
                            if (ImGui.MenuItem("Current Tile (257x257 L16 PNG + JSON)...", hasTerrain))
                            {
                                _wantTerrainExport = true;
                                _terrainExportKind = TerrainExportKind.Heightmap257CurrentTilePerTile;
                            }

                            if (ImGui.MenuItem("Loaded Tiles Folder (per-tile)...", hasTerrain))
                            {
                                _wantTerrainExport = true;
                                _terrainExportKind = TerrainExportKind.Heightmap257LoadedTilesFolderPerTile;
                            }

                            if (ImGui.MenuItem("Whole Map Folder (per-map)...", hasTerrain))
                            {
                                _wantTerrainExport = true;
                                _terrainExportKind = TerrainExportKind.Heightmap257WholeMapFolderPerMap;
                            }

                            ImGui.EndMenu();
                        }

                        if (ImGui.BeginMenu("MCCV"))
                        {
                            if (ImGui.MenuItem("Current Tile PNG...", hasTerrain))
                            {
                                _wantTerrainExport = true;
                                _terrainExportKind = TerrainExportKind.MccvCurrentTilePng;
                            }

                            if (ImGui.MenuItem("Loaded Tiles Folder...", hasTerrain))
                            {
                                _wantTerrainExport = true;
                                _terrainExportKind = TerrainExportKind.MccvLoadedTilesFolder;
                            }

                            if (ImGui.MenuItem("Whole Map Folder...", hasTerrain))
                            {
                                _wantTerrainExport = true;
                                _terrainExportKind = TerrainExportKind.MccvWholeMapFolder;
                            }

                            ImGui.EndMenu();
                        }

                        ImGui.EndMenu();
                    }

                    ImGui.EndMenu();
                }

                if (ImGui.BeginMenu("Import"))
                {
                    if (ImGui.BeginMenu("Terrain"))
                    {
                        bool hasTerrain = _terrainManager != null || _vlmTerrainManager != null;

                        if (ImGui.BeginMenu("Alpha Masks"))
                        {
                            if (ImGui.MenuItem("From Folder of Tile Atlases...", hasTerrain))
                            {
                                _wantTerrainImport = true;
                                _terrainImportKind = TerrainImportKind.AlphaFolder;
                            }
                            ImGui.EndMenu();
                        }

                        if (ImGui.BeginMenu("Heightmaps"))
                        {
                            if (ImGui.MenuItem("From Folder of Tile Heightmaps...", hasTerrain))
                            {
                                _wantTerrainImport = true;
                                _terrainImportKind = TerrainImportKind.Heightmap257Folder;
                            }
                            ImGui.EndMenu();
                        }

                        if (ImGui.BeginMenu("MCCV"))
                        {
                            if (ImGui.MenuItem("From Folder of Tile MCCV PNGs...", hasTerrain))
                            {
                                _wantTerrainImport = true;
                                _terrainImportKind = TerrainImportKind.MccvFolder;
                            }
                            ImGui.EndMenu();
                        }

                        ImGui.EndMenu();
                    }

                    ImGui.EndMenu();
                }

                ImGui.EndMenu();
            }

            if (ImGui.BeginMenu("Help"))
            {
                if (ImGui.MenuItem("Keyboard Shortcuts"))
                    _showKeyboardShortcutsWindow = true;

                if (ImGui.MenuItem("About"))
                {
                    _openAboutPopup = true;
                    _statusMessage = ViewerProductName;
                }
                ImGui.EndMenu();
            }

            // Top-Level Mode & Workspace Profile Switcher (centered on the main menu bar)
            float modeBtnWidthViewer = 90f;
            float modeBtnWidthEditor = 90f;
            float modeBtnWidthArch = 115f;
            float itemSpacing = ImGui.GetStyle().ItemSpacing.X;
            float totalWidth = modeBtnWidthViewer + modeBtnWidthEditor + modeBtnWidthArch + (itemSpacing * 2);
            float windowWidth = ImGui.GetWindowWidth();
            float targetCenterX = (windowWidth - totalWidth) * 0.5f;
            if (targetCenterX > ImGui.GetCursorPosX())
            {
                ImGui.SetCursorPosX(targetCenterX);
            }

            bool isEditor = _workspaceMode == WorkspaceMode.Editor;
            bool isArchaeology = _workspaceMode == WorkspaceMode.Archaeology;
            bool isViewer = _workspaceMode == WorkspaceMode.Viewer;

            if (isViewer)
                ImGui.PushStyleColor(ImGuiCol.Button, new Vector4(0.18f, 0.42f, 0.75f, 1f));
            if (ImGui.Button("Viewer##top_mode_viewer", new Vector2(modeBtnWidthViewer, 0)))
                SetWorkspaceMode(WorkspaceMode.Viewer);
            if (isViewer)
                ImGui.PopStyleColor();

            ImGui.SameLine();
            if (isEditor)
                ImGui.PushStyleColor(ImGuiCol.Button, new Vector4(0.75f, 0.42f, 0.15f, 1f));
            if (ImGui.Button("Editor##top_mode_editor", new Vector2(modeBtnWidthEditor, 0)))
                SetWorkspaceMode(WorkspaceMode.Editor);
            if (isEditor)
                ImGui.PopStyleColor();

            ImGui.SameLine();
            if (isArchaeology)
                ImGui.PushStyleColor(ImGuiCol.Button, new Vector4(0.55f, 0.25f, 0.75f, 1f));
            if (ImGui.Button("Archaeology##top_mode_archaeology", new Vector2(modeBtnWidthArch, 0)))
                SetWorkspaceMode(WorkspaceMode.Archaeology);
            if (isArchaeology)
                ImGui.PopStyleColor();

            ImGui.EndMainMenuBar();
        }

        DrawKeyboardShortcutsWindow();

        if (_openForgetKnownGoodClientConfirm)
        {
            _openForgetKnownGoodClientConfirm = false;
            ImGui.OpenPopup("Confirm Forget Known-Good Base");
        }

        if (_openAboutPopup)
        {
            _openAboutPopup = false;
            ImGui.OpenPopup(ViewerAboutPopupTitle);
        }

        bool keepAboutPopupOpen = true;
        if (ImGui.BeginPopupModal(ViewerAboutPopupTitle, ref keepAboutPopupOpen, ImGuiWindowFlags.AlwaysAutoResize))
        {
            ImGui.TextUnformatted(ViewerProductName);
            ImGui.TextDisabled($"Version {GetViewerDisplayVersion()}");
            ImGui.Spacing();
            ImGui.TextWrapped("World/model viewer and debugging surface for WoW Alpha, Wrath, and early Cataclysm data.");
            ImGui.Spacing();
            ImGui.TextWrapped("Author: github.com/akspa0/parp-tools");
            ImGui.TextWrapped("Discord: discord.gg/6YdUksuKuU");
            ImGui.Spacing();
            ImGui.TextUnformatted("In memory of Hayven Games");
            ImGui.TextWrapped("An inspiration for this project and a friend, whose short films explored World of Warcraft's secrets and little-known details through game footage.");
            ImGui.Spacing();
            ImGui.Separator();
            ImGui.TextWrapped("Thanks to...");
            ImGui.TextWrapped("Marlamin, schlumpf, Dovah, Pirate the Explorer, fean, implave, IS4, Mjollna, Adspartan (Noggit), and Skarn (Noggit-Red).");
            ImGui.TextWrapped("Without the WoW Exploration community, this project would not exist. Everyone named here contributed inspiration to this project in some way.");
            ImGui.TextDisabled("This tooling is about restoration, not touching up or polishing what we recover - it is the instrument for restoring what already exists. Noggit and Noggit-Red remain the preferred editors for fine-tuning the results this library and tooling produce.");
            ImGui.Spacing();
            ImGui.TextWrapped("Special thanks to WoWdev.wiki, Exploration Reboot, The Alpha Project, and everyone in the Pre-Alpha Restoration Project discord!");
            ImGui.Spacing();
            if (ImGui.Button("Close", new Vector2(120f, 0f)))
                ImGui.CloseCurrentPopup();

            ImGui.EndPopup();
        }

        bool keepForgetKnownGoodPopupOpen = true;
        if (ImGui.BeginPopupModal("Confirm Forget Known-Good Base", ref keepForgetKnownGoodPopupOpen, ImGuiWindowFlags.AlwaysAutoResize))
        {
            string displayName = string.IsNullOrWhiteSpace(_pendingForgetKnownGoodClientDisplayName)
                ? "this saved base"
                : _pendingForgetKnownGoodClientDisplayName!;

            ImGui.TextWrapped($"Remove saved base '{displayName}'?");
            if (!string.IsNullOrWhiteSpace(_pendingForgetKnownGoodClientPath))
                ImGui.TextDisabled(_pendingForgetKnownGoodClientPath);

            ImGui.Spacing();
            if (ImGui.Button("Remove", new Vector2(120f, 0f)))
            {
                if (!string.IsNullOrWhiteSpace(_pendingForgetKnownGoodClientPath))
                    _clientDialogs.ForgetKnownGoodClientPath(_pendingForgetKnownGoodClientPath);

                _settings.ClearPendingForgetKnownGoodClientPath();
                ImGui.CloseCurrentPopup();
            }

            ImGui.SameLine();
            if (ImGui.Button("Cancel", new Vector2(120f, 0f)))
            {
                _settings.ClearPendingForgetKnownGoodClientPath();
                ImGui.CloseCurrentPopup();
            }

            ImGui.EndPopup();
        }

        if (!keepForgetKnownGoodPopupOpen)
            _settings.ClearPendingForgetKnownGoodClientPath();

        if (!keepAboutPopupOpen)
            _openAboutPopup = false;

        // Handle deferred actions
        if (_wantOpenFile)
        {
            _wantOpenFile = false;
            _showFolderInput = false;
            // Use ImGui text input as a simple file path dialog
            ImGui.OpenPopup("OpenFilePopup");
        }

        if (ImGui.BeginPopup("OpenFilePopup"))
        {
            ImGui.Text("Enter file path:");
            var buf = _folderInputBuf;
            if (ImGui.InputText("##filepath", ref buf, 512, ImGuiInputTextFlags.EnterReturnsTrue))
            {
                if (File.Exists(buf))
                {
                    _modelLoader.LoadFileFromDisk(buf);
                    ImGui.CloseCurrentPopup();
                }
                else
                {
                    _statusMessage = $"File not found: {buf}";
                }
            }
            _folderInputBuf = buf;
            if (ImGui.Button("Cancel"))
                ImGui.CloseCurrentPopup();
            ImGui.EndPopup();
        }

        if (_wantOpenVlmProject)
        {
            _wantOpenVlmProject = false;
            ImGuiPathPicker.Instance.Open(
                "Select ML Dataset folder (containing dataset/ with JSON files)",
                pickFolder: true,
                initialPath: null,
                filterExtension: null,
                vlmPath =>
                {
                    if (!string.IsNullOrEmpty(vlmPath) && Directory.Exists(vlmPath))
                        _worldLoader.LoadVlmProject(vlmPath);
                });
        }

        if (_wantOpenZarrDataset)
        {
            _wantOpenZarrDataset = false;
            ImGuiPathPicker.Instance.Open(
                "Select Zarr tile dataset folder (parent of <build>.zarr/ or the store root itself)",
                pickFolder: true,
                initialPath: null,
                filterExtension: null,
                zarrPath =>
                {
                    if (!string.IsNullOrEmpty(zarrPath) && Directory.Exists(zarrPath))
                        _worldLoader.LoadZarrDataset(zarrPath);
                });
        }

        if (_wantSelectDatasetCatalogRoot)
        {
            _wantSelectDatasetCatalogRoot = false;
            ImGuiPathPicker.Instance.Open(
                "Select dataset catalog root",
                pickFolder: true,
                initialPath: _datasetCatalogRoot,
                filterExtension: null,
                catalogRoot =>
                {
                    if (!string.IsNullOrWhiteSpace(catalogRoot) && Directory.Exists(catalogRoot))
                    {
                        _datasetCatalogRoot = catalogRoot;
                        RefreshDatasetCatalog();
                        _settings.SaveViewerSettings();
                    }
                });
        }

        if (_wantOpenWdtFile)
        {
            _wantOpenWdtFile = false;
            ImGuiPathPicker.Instance.Open(
                "Select Alpha WDT file (loose map)",
                pickFolder: false,
                initialPath: _lastLooseOverlayPath,
                filterExtension: ".wdt;.mpq",
                wdtPath =>
                {
                    if (!string.IsNullOrEmpty(wdtPath) && File.Exists(wdtPath))
                    {
                        _modelLoader.LoadFileFromDisk(wdtPath);
                        _statusMessage = $"Loaded alpha WDT: {wdtPath}";
                    }
                });
        }

        if (_wantOpenPm4File)
        {
            _wantOpenPm4File = false;
            ImGuiPathPicker.Instance.Open(
                "Select Loose PM4 / PD4 File",
                pickFolder: false,
                initialPath: _lastLooseOverlayPath,
                filterExtension: ".pm4;.pd4",
                pm4Path =>
                {
                    if (!string.IsNullOrEmpty(pm4Path) && File.Exists(pm4Path))
                    {
                        _lastLooseOverlayPath = Path.GetDirectoryName(pm4Path);
                        if (_worldScene != null)
                        {
                            if (_worldScene.Pm4Overlay.LoadLoosePm4File(pm4Path))
                                _statusMessage = _worldScene.Pm4Overlay.Pm4Status;
                            else
                                _statusMessage = $"Failed to decode loose PM4/PD4 file: {pm4Path}";
                        }
                        else
                        {
                            _statusMessage = $"Load a world scene or map first before displaying loose PM4/PD4 overlays: {pm4Path}";
                        }
                    }
                });
        }

        _cascAhdrSource.HandleCascAhdrMenuRequests();

        if (_wantAttachLooseMapFolder)
        {
            _wantAttachLooseMapFolder = false;

            if (_dataSource is MpqDataSource)
            {
                ImGuiPathPicker.Instance.Open(
                    "Select loose map overlay folder (contains World\\Maps or a map directory under World\\Maps)",
                    pickFolder: true,
                    initialPath: string.IsNullOrWhiteSpace(_lastLooseOverlayPath) ? null : _lastLooseOverlayPath,
                    filterExtension: null,
                    overlayPath =>
                    {
                        if (!string.IsNullOrEmpty(overlayPath) && Directory.Exists(overlayPath))
                            _dataSourceSession.AttachLooseMapOverlay(overlayPath);
                    });
            }
        }

        if (!string.IsNullOrWhiteSpace(_pendingKnownGoodClientPath))
        {
            string savedBasePath = _pendingKnownGoodClientPath!;
            string? savedBuildVersion = _pendingKnownGoodClientBuildVersion;
            bool attachLooseFolder = _pendingKnownGoodClientAttachLooseFolder;
            _pendingKnownGoodClientPath = null;
            _pendingKnownGoodClientBuildVersion = null;
            _pendingKnownGoodClientAttachLooseFolder = false;

            if (!Directory.Exists(savedBasePath))
            {
                _statusMessage = $"Saved client path no longer exists: {savedBasePath}";
            }
            else if (attachLooseFolder)
            {
                ImGuiPathPicker.Instance.Open(
                    "Select loose map folder to load against the saved base client",
                    pickFolder: true,
                    initialPath: string.IsNullOrWhiteSpace(_lastLooseOverlayPath) ? null : _lastLooseOverlayPath,
                    filterExtension: null,
                    overlayPath =>
                    {
                        if (!string.IsNullOrWhiteSpace(overlayPath) && Directory.Exists(overlayPath))
                        {
                            _dataSourceSession.LoadMpqDataSource(savedBasePath, null, savedBuildVersion, deferWorldReload: true);
                            _dataSourceSession.AttachLooseMapOverlay(overlayPath);
                            _dataSourceSession.RestoreWorldAfterDataSourceReload();
                        }
                    });
            }
            else
            {
                _dataSourceSession.LoadMpqDataSource(savedBasePath, null, savedBuildVersion);
            }
        }

        if (_wantTerrainExport)
        {
            _wantTerrainExport = false;
            _terrainTileIo.RunTerrainExport();
        }

        if (_wantTerrainImport)
        {
            _wantTerrainImport = false;
            _terrainTileIo.RunTerrainImport();
        }

        if (_wantExportGlbCollision)
        {
            _wantExportGlbCollision = false;
            if (_loadedFilePath != null)
            {
                Directory.CreateDirectory(ExportDir);
                string glbPath = Path.Combine(ExportDir, Path.ChangeExtension(_loadedFileName!, ".collision.glb"));
                try
                {
                    string dir = Path.GetDirectoryName(_loadedFilePath) ?? ".";
                    if (_loadedWmo != null)
                    {
                        GlbExporter.ExportWmoCollision(_loadedWmo, dir, glbPath);
                    }
                    else
                    {
                        var ext = Path.GetExtension(_loadedFilePath).ToLowerInvariant();
                        if (ext == ".wmo")
                        {
                            var converter = new WmoV14ToV17Converter();
                            var wmo = converter.ParseWmoV14(_loadedFilePath);
                            GlbExporter.ExportWmoCollision(wmo, dir, glbPath);
                        }
                        else
                        {
                            throw new InvalidOperationException("Collision-only GLB export is currently supported for WMO and Terrain only.");
                        }
                    }
                    _statusMessage = $"Exported: {glbPath}";
                }
                catch (Exception ex)
                {
                    _statusMessage = $"Export failed: {ex.Message}";
                }
            }
            else if (_terrainManager != null && _dataSource != null)
            {
                Directory.CreateDirectory(ExportDir);
                int curTx = _terrainManager.CameraTileX;
                int curTy = _terrainManager.CameraTileY;
                if (curTx >= 0 && curTy >= 0)
                {
                    string glbPath = Path.Combine(ExportDir, $"{_terrainManager.MapName}_{curTx:D2}_{curTy:D2}.collision.glb");
                    try
                    {
                        MapGlbExporter.ExportTile(_terrainManager, _dataSource, _md5Index, curTx, curTy, glbPath, includePlacements: false);
                        _statusMessage = $"Exported GLB Collision Mesh for Tile ({curTx},{curTy}) to: {glbPath}";
                    }
                    catch (Exception ex)
                    {
                        _statusMessage = $"GLB Collision export failed: {ex.Message}";
                    }
                }
                else
                {
                    _statusMessage = "Camera tile out of range for collision export.";
                }
            }
            else
            {
                _statusMessage = "No model or terrain loaded for GLB collision export.";
            }
        }

        if (_wantExportGlb)
        {
            _wantExportGlb = false;
            if (_loadedFilePath != null)
            {
                Directory.CreateDirectory(ExportDir);
                string glbPath = Path.Combine(ExportDir, Path.ChangeExtension(_loadedFileName!, ".glb"));
                try
                {
                    string dir = Path.GetDirectoryName(_loadedFilePath) ?? ".";
                    if (_loadedWmo != null)
                    {
                        GlbExporter.ExportWmoWithDoodads(_loadedWmo, dir, glbPath, _dataSource);
                    }
                    else if (_loadedMdx != null)
                    {
                        GlbExporter.ExportMdx(_loadedMdx, dir, glbPath, _dataSource);
                    }
                    else
                    {
                        // Fallback: re-parse from disk (legacy path)
                        var ext = Path.GetExtension(_loadedFilePath).ToLowerInvariant();
                        if (ext == ".mdx")
                        {
                            var mdx = MdxFile.Load(_loadedFilePath);
                            GlbExporter.ExportMdx(mdx, dir, glbPath, _dataSource);
                        }
                        else if (ext == ".m2")
                        {
                            byte[] m2Bytes = File.ReadAllBytes(_loadedFilePath);
                            byte[]? skinBytes = null;
                            foreach (var skinPath in WarcraftNetM2Adapter.BuildSkinCandidates(_loadedFilePath))
                            {
                                if (File.Exists(skinPath)) { skinBytes = File.ReadAllBytes(skinPath); break; }
                            }
                            var converter = new WoWViewer.Transfer.M2ToMdxConverter();
                            byte[]? mdxBytes = converter.ConvertToBytes(m2Bytes, skinBytes, null);
                            if (mdxBytes != null)
                            {
                                using var ms = new MemoryStream(mdxBytes);
                                var mdx = MdxFile.Load(ms);
                                GlbExporter.ExportMdx(mdx, dir, glbPath, _dataSource);
                            }
                        }
                        else if (ext == ".wmo")
                        {
                            var converter = new WmoV14ToV17Converter();
                            var wmo = converter.ParseWmoV14(_loadedFilePath);
                            GlbExporter.ExportWmoWithDoodads(wmo, dir, glbPath, _dataSource);
                        }
                    }
                    _statusMessage = $"Exported: {glbPath}";
                }
                catch (Exception ex)
                {
                    _statusMessage = $"Export failed: {ex.Message}";
                }
            }
            else if (_terrainManager != null)
            {
                Directory.CreateDirectory(ExportDir);
                int curTx = _terrainManager.CameraTileX;
                int curTy = _terrainManager.CameraTileY;
                if (curTx >= 0 && curTy >= 0)
                {
                    string glbPath = Path.Combine(ExportDir, $"{_terrainManager.MapName}_{curTx:D2}_{curTy:D2}.glb");
                    try
                    {
                        MapGlbExporter.ExportTile(_terrainManager, _dataSource, _md5Index, curTx, curTy, glbPath, includePlacements: true);
                        _statusMessage = $"Exported GLB Scene for Tile ({curTx},{curTy}) to: {glbPath}";
                    }
                    catch (Exception ex)
                    {
                        _statusMessage = $"GLB Scene export failed: {ex.Message}";
                    }
                }
                else
                {
                    _statusMessage = "Camera tile out of range for GLB export.";
                }
            }
            else
            {
                _statusMessage = "No model or terrain loaded for GLB export.";
            }
        }

        if (_wantExportMapGlbTiles)
        {
            _wantExportMapGlbTiles = false;
            try
            {
                _terrainTileIo.RunMapGlbTilesExport();
            }
            catch (Exception ex)
            {
                _statusMessage = $"Map GLB export failed: {ex.Message}";
            }
        }
    }

    private void PrepareVlmExportDialogInputs()
    {
        string? activeGamePath = _dataSourceSession.GetActiveGamePath();
        if (!string.IsNullOrWhiteSpace(activeGamePath))
            _vlmClientPath = activeGamePath;

        string? currentMapName = _dataSourceSession.GetCurrentSessionMapName();
        if (!string.IsNullOrWhiteSpace(currentMapName))
            _vlmMapName = currentMapName;

        if (!string.IsNullOrWhiteSpace(_vlmClientPath) && !string.IsNullOrWhiteSpace(_vlmMapName) && string.IsNullOrWhiteSpace(_vlmOutputDir))
            _vlmOutputDir = DatasetExportDialogsService.GenerateVlmOutputPath(_vlmClientPath, _vlmMapName);
    }

    private void PrepareTerrainTextureTransferDialogInputs()
    {
        string? overlayMapDir = _dataSourceSession.TryResolveCurrentMapDirectory(preferLooseOverlay: true);
        string? baseMapDir = _dataSourceSession.TryResolveCurrentMapDirectory(preferLooseOverlay: false);

        if (!string.IsNullOrWhiteSpace(overlayMapDir))
            _terrainTransferSourceDir = overlayMapDir;

        if (!string.IsNullOrWhiteSpace(baseMapDir))
            _terrainTransferTargetDir = baseMapDir;
        else if (!string.IsNullOrWhiteSpace(overlayMapDir))
            _terrainTransferTargetDir = overlayMapDir;

        string? currentMapName = _dataSourceSession.GetCurrentSessionMapName();
        bool usingDefaultOutput = string.IsNullOrWhiteSpace(_terrainTransferOutputDir)
            || string.Equals(_terrainTransferOutputDir, Path.Combine("output", "terrain-texture-transfer-ui"), StringComparison.OrdinalIgnoreCase);
        if (usingDefaultOutput && !string.IsNullOrWhiteSpace(currentMapName))
            _terrainTransferOutputDir = Path.Combine("output", "terrain-texture-transfer-ui", currentMapName);
    }

    internal void PrepareMapConverterDialogInputs()
    {
        string? preferredWdt = _dataSourceSession.TryGetLoadedLocalWdtPath();
        preferredWdt ??= _dataSourceSession.TryResolveCurrentMapWdtPath(preferLooseOverlay: true);
        preferredWdt ??= _dataSourceSession.TryResolveCurrentMapWdtPath(preferLooseOverlay: false);

        if (!string.IsNullOrWhiteSpace(preferredWdt))
            _mapConvertSourcePath = preferredWdt;

        string? preferredMapDir = _dataSourceSession.TryResolveCurrentMapDirectory(preferLooseOverlay: true);
        preferredMapDir ??= _dataSourceSession.TryResolveCurrentMapDirectory(preferLooseOverlay: false);
        if (!string.IsNullOrWhiteSpace(preferredMapDir))
            _mapConvertLkMapDir = preferredMapDir;

        if (!string.IsNullOrWhiteSpace(_mapConvertSourcePath))
            _converterDialogs.EnsureMapConverterProjectOutputDirectory(forceNew: false);
    }

    internal void PrepareWmoConverterDialogInputs()
    {
        if (!string.IsNullOrEmpty(_loadedFilePath)
            && string.Equals(Path.GetExtension(_loadedFilePath), ".wmo", StringComparison.OrdinalIgnoreCase))
        {
            _wmoConvertSourcePath = _loadedFilePath;
        }
    }
}
