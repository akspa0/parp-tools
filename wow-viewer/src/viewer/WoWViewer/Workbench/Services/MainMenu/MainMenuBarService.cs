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
using WowViewer.Core.M2;
using WowViewer.Core.Runtime.M2;
using WowViewer.Core.Runtime.PromoVideo;
using WowViewer.Core.Runtime.World.Visibility;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;
using WowViewer.Core.IO.Converters;
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
/// File/View/Help content and deferred path-picker requests. The manager owns all
/// chrome and popup windows; state remains behind the existing host contract.
/// </summary>
internal sealed partial class MainMenuBarService
{
    private readonly IViewerAppHost _host;

    internal MainMenuBarService(IViewerAppHost host)
    {
        _host = host;
        _terrainFiles = new TerrainFileOperationsService(host);
        _modelWorldFiles = new ModelWorldFileOperationsService(host);
    }

    // Host bridge: see MainMenuBarService.Host.cs.

    private bool _openAboutPopup;
    private bool _showUiReferenceWindow;
    private bool _wantAttachLooseMapFolder = false;
    private bool _wantOpenWdtFile = false;
    private bool _wantOpenPm4File = false;
    private bool _wantOpenVlmProject = false;
    private bool _wantOpenZarrDataset = false;

    /// <summary>Content only; caller owns the active menu bar.</summary>
    public void DrawContent()
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

                if (ImGui.MenuItem("Open DAT Terrain Folder..."))
                    _cascAhdrSource._wantOpenAhdrTerrainFolder = true;
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
            if (ImGui.MenuItem("Reset Camera")) ResetCamera();
            if (ImGui.MenuItem("Hide UI Chrome", "Tab", _hideUiChrome)) _hideUiChrome = !_hideUiChrome;
            ImGui.MenuItem("Left Sidebar", "", ref _showLeftSidebar);
            ImGui.MenuItem("Right Sidebar", "I", ref _showRightSidebar);
            if (ImGui.MenuItem("Reset Shell Layout")) _shellLayout.ResetLayout();
            ImGui.EndMenu();
        }
            if (ImGui.BeginMenu("Help"))
            {
                if (ImGui.MenuItem("Keyboard Shortcuts"))
                    RequestKeyboardShortcuts();

                if (ImGui.MenuItem("UI Reference"))
                    RequestUiReference();

                if (ImGui.MenuItem("About"))
                {
                    RequestAbout();
                    _statusMessage = ViewerProductName;
                }
                ImGui.EndMenu();
            }
    }

    // Compile compatibility only. Parent must migrate the standalone ViewerApp.cs call.
    [Obsolete("Call DrawContent inside the manager menu bar; call DrawDeferredActions once at its root.")]
    internal void DrawMenuBar() => DrawContent();

    /// <summary>Call once per frame at a stable root, before the shared path picker draws.</summary>
    public void DrawDeferredActions()
    {
        if (_wantOpenFile)
        {
            _wantOpenFile = false;
            _showFolderInput = false;
            ImGuiPathPicker.Instance.Open("Open File", false, _lastLooseOverlayPath,
                ".mdx;.mdl;.m2;.wmo;.wdt", LoadPickedFile);
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
                    RunFolderAction(catalogRoot, () =>
                    {
                        _datasetCatalogRoot = catalogRoot;
                        RefreshDatasetCatalog();
                        _settings.SaveViewerSettings();
                    });
                });
        }

        if (_wantOpenWdtFile)
        {
            _wantOpenWdtFile = false;
            ImGuiPathPicker.Instance.Open(
                "Select Alpha WDT file (loose map)",
                pickFolder: false,
                initialPath: _lastLooseOverlayPath,
                filterExtension: ".wdt",
                wdtPath =>
                {
                    LoadPickedFile(wdtPath);
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
                    try
                    {
                        if (!File.Exists(pm4Path)) throw new FileNotFoundException("Selected file no longer exists.", pm4Path);
                        _lastLooseOverlayPath = Path.GetDirectoryName(pm4Path);
                        if (_worldScene != null)
                        {
                            if (_worldScene.Pm4Overlay.LoadLoosePm4File(pm4Path))
                                _statusMessage = $"{_worldScene.Pm4Overlay.Pm4Status} — {pm4Path}";
                            else
                                _statusMessage = $"Failed to decode loose PM4/PD4 file: {pm4Path}";
                        }
                        else
                        {
                            _statusMessage = $"Load a world scene or map first before displaying loose PM4/PD4 overlays: {pm4Path}";
                        }
                    }
                    catch (Exception ex) { _statusMessage = $"Open PM4/PD4 failed — {pm4Path}: {ex.Message}"; }
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
                        RunFolderAction(overlayPath, () => _dataSourceSession.AttachLooseMapOverlay(overlayPath));
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
                        RunFolderAction(overlayPath, () =>
                        {
                            _dataSourceSession.LoadMpqDataSource(savedBasePath, null, savedBuildVersion, deferWorldReload: true);
                            _dataSourceSession.AttachLooseMapOverlay(overlayPath);
                            _dataSourceSession.RestoreWorldAfterDataSourceReload();
                        });
                    });
            }
            else
            {
                RunFolderAction(savedBasePath, () => _dataSourceSession.LoadMpqDataSource(savedBasePath, null, savedBuildVersion));
            }
        }
        HandleFileOperationRequests();
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
