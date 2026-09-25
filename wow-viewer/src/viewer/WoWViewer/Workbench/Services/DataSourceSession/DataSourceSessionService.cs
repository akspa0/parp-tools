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

namespace WoWViewer;

/// <summary>
/// Data-source session: MPQ/data-source loading, file lists and discovered maps, session roots and map resolution, data-source reload staging/restore, world return, loose map overlays, listfile and minimap initialisation.
/// Extracted verbatim from <see cref="ViewerApp"/> (Epic 251 U-01). World and app state it
/// needs comes only through <see cref="IViewerAppHost"/>; the bridge members below keep the
/// names the moved code used inside ViewerApp, so no moved body was edited.
/// </summary>
internal sealed class DataSourceSessionService
{
    private readonly IViewerAppHost _host;

    internal DataSourceSessionService(IViewerAppHost host)
    {
        _host = host;
    }

    // Host bridge (same names as the former ViewerApp members).
    private ref AreaTableService? _areaTableService => ref _host.AreaTableService;
    private ref bool _autoOpenWorldMapsPanel => ref _host.AutoOpenWorldMapsPanel;
    private ref Camera _camera => ref _host.Camera;
    private ref AssetCatalogView? _catalogView => ref _host.CatalogView;
    private ref IDataSource? _dataSource => ref _host.DataSource;
    private ref string? _dbcBuild => ref _host.DbcBuild;
    private ref DBCD.Providers.IDBCProvider? _dbcProvider => ref _host.DbcProvider;
    private ref string? _dbdDir => ref _host.DbdDir;
    private ref List<MapDefinition> _discoveredMaps => ref _host.DiscoveredMaps;
    private ref string _extensionFilter => ref _host.ExtensionFilter;
    private ref List<string> _filteredFiles => ref _host.FilteredFiles;
    private ref GL _gl => ref _host.Gl;
    private ref string _lastGameFolderPath => ref _host.LastGameFolderPath;
    private ref string _lastLooseOverlayPath => ref _host.LastLooseOverlayPath;
    private ref string? _lastVirtualPath => ref _host.LastVirtualPath;
    private ref float _lastWorldSceneCameraPitch => ref _host.LastWorldSceneCameraPitch;
    private ref Vector3 _lastWorldSceneCameraPosition => ref _host.LastWorldSceneCameraPosition;
    private ref float _lastWorldSceneCameraYaw => ref _host.LastWorldSceneCameraYaw;
    private ref string? _lastWorldSceneWdtPath => ref _host.LastWorldSceneWdtPath;
    private ref string? _loadedFilePath => ref _host.LoadedFilePath;
    private ref M2StaticRenderModel? _loadedM2Runtime => ref _host.LoadedM2Runtime;
    private ref MdxFile? _loadedMdx => ref _host.LoadedMdx;
    private ref WmoV14ToV17Converter.WmoV14Data? _loadedWmo => ref _host.LoadedWmo;
    private HashSet<string> _loggedStandaloneMissingSkinPaths => _host.LoggedStandaloneMissingSkinPaths;
    private ref Md5TranslateIndex? _md5Index => ref _host.Md5Index;
    private ref MinimapRenderer? _minimapRenderer => ref _host.MinimapRenderer;
    private StandaloneModelLoaderService _modelLoader => _host.ModelLoader;
    private ref Vector3? _pendingWorldSpawnOverride => ref _host.PendingWorldSpawnOverride;
    private ref ISceneRenderer? _renderer => ref _host.Renderer;
    private ref string _searchFilter => ref _host.SearchFilter;
    private ref int _selectedFileIndex => ref _host.SelectedFileIndex;
    private SqlSpawnStreamingService _sqlSpawnStreaming => _host.SqlSpawnStreaming;
    private Dictionary<string, string?> _standaloneSkinPathCache => _host.StandaloneSkinPathCache;
    private ref string _statusMessage => ref _host.StatusMessage;
    private ref TerrainManager? _terrainManager => ref _host.TerrainManager;
    private ref ReplaceableTextureResolver? _texResolver => ref _host.TexResolver;
    private ref VlmTerrainManager? _vlmTerrainManager => ref _host.VlmTerrainManager;
    private WdlPreviewService _wdlPreview => _host.WdlPreview;
    private WorldLoaderService _worldLoader => _host.WorldLoader;
    private ref WorldScene? _worldScene => ref _host.WorldScene;
    private void InvalidatePm4DerivedReports() => _host.InvalidatePm4DerivedReports();

    private string? _pendingDataSourceWorldReloadVirtualPath;
    private string? _pendingDataSourceWorldReloadLocalPath;
    private Vector3? _pendingDataSourceWorldReloadCameraPosition;
    private float _pendingDataSourceWorldReloadCameraYaw = 180f;
    private float _pendingDataSourceWorldReloadCameraPitch = -20f;
    private int _activeDataSourceReloadGeneration;
    private int _pendingDataSourceReloadGeneration;
    private static readonly string[] EarlyModelBrowserExtensions = { ".mdx", ".mdl" };


    internal void RefreshFileList()
    {
        if (_dataSource == null) return;

        var allFiles = GetFilesForBrowserFilter();
        IEnumerable<string> candidates = allFiles;
        if (!string.IsNullOrEmpty(_searchFilter))
            candidates = candidates.Where(f => f.Contains(_searchFilter, StringComparison.OrdinalIgnoreCase));

        var filtered = new List<string>(capacity: 5000);
        foreach (string file in candidates)
        {
            if (!_dataSource.FileExists(file))
                continue;

            filtered.Add(file);
            if (filtered.Count >= 5000)
                break;
        }

        _filteredFiles = filtered;

        _selectedFileIndex = -1;
    }

    private IReadOnlyList<string> GetFilesForBrowserFilter()
    {
        if (_dataSource == null)
            return Array.Empty<string>();

        if (!_extensionFilter.Equals(".mdx", StringComparison.OrdinalIgnoreCase))
            return _dataSource.GetFileList(_extensionFilter);

        var combined = new List<string>();
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (string extension in EarlyModelBrowserExtensions)
        {
            foreach (string file in _dataSource.GetFileList(extension))
            {
                if (seen.Add(file))
                    combined.Add(file);
            }
        }

        return combined;
    }

    internal void RefreshDiscoveredMaps()
    {
        if (_dataSource == null)
        {
            _discoveredMaps.Clear();
            _autoOpenWorldMapsPanel = false;
            return;
        }

        int previousDiscoveredMapCount = _discoveredMaps.Count;

        if (_dbcProvider != null && !string.IsNullOrWhiteSpace(_dbdDir) && !string.IsNullOrWhiteSpace(_dbcBuild))
        {
            var mapDiscovery = new MapDiscoveryService(_dbcProvider, _dbdDir!, _dbcBuild!, _dataSource);
            _discoveredMaps = mapDiscovery.DiscoverMaps();
            ViewerLog.Important(ViewerLog.Category.Dbc,
                $"Discovered {_discoveredMaps.Count} maps via Map.dbc/data source ({_discoveredMaps.Count(m => m.HasWdt)} with WDTs, {_discoveredMaps.Count(m => !m.HasDbcEntry)} custom loose maps)");
        }
        else
        {
            _discoveredMaps = MapDiscoveryService.DiscoverLooseMapsOnly(_dataSource);
            ViewerLog.Important(ViewerLog.Category.Dbc,
                $"Discovered {_discoveredMaps.Count} loose maps without Map.dbc metadata.");
        }

        _autoOpenWorldMapsPanel = _discoveredMaps.Count > 0 && previousDiscoveredMapCount == 0;
        _wdlPreview.WarmDiscoveredWdlPreviews();
    }

    internal void LoadMpqDataSource(string gamePath, string? listfilePath, string? explicitBuildVersion = null, bool deferWorldReload = false)
    {
        _pendingDataSourceReloadGeneration = ++_activeDataSourceReloadGeneration;
        try
        {
            string? resolvedListfilePath = ResolveListfilePath(listfilePath);
            _statusMessage = $"Loading MPQ archives from {gamePath}...";
            StageCurrentWorldForDataSourceReload();
            ClearActiveSceneForDataSourceReload();
            _lastGameFolderPath = Path.GetFullPath(gamePath);
            _standaloneSkinPathCache.Clear();
            _loggedStandaloneMissingSkinPaths.Clear();
            _discoveredMaps.Clear();
            _areaTableService = null;
            _wdlPreview.ResetWdlPreviewSupport();
            _dataSource?.Dispose();
            _dataSource = new MpqDataSource(gamePath, resolvedListfilePath);
            _statusMessage = $"Loaded: {_dataSource.Name}";
            _wdlPreview.InitializeWdlPreviewSupport();

            // Load DBC tables directly from MPQ for replaceable texture resolution
            _texResolver = new ReplaceableTextureResolver();
            _texResolver.SetDataSource(_dataSource);
            _catalogView?.SetDataSource(_dataSource, _texResolver);
            var mpqDs = _dataSource as MpqDataSource;
            _dbcProvider = mpqDs != null
                ? new MpqDBCProvider(mpqDs.ArchiveReader, _dataSource)
                : new MpqDBCProvider(_dataSource);
            var dbcProvider = _dbcProvider;

            InitializeMinimapSupport();

            string? dbdDir = ResolveDbdDefinitionsDir();
            if (dbdDir != null)
            {
                _dbdDir = dbdDir;

                string buildAlias = explicitBuildVersion ?? WorldLoaderService.InferBuildFromPath(gamePath, dbdDir);
                ViewerLog.Trace(explicitBuildVersion == null
                    ? $"[WoWViewer] Inferred build: '{buildAlias}' from path: {gamePath}"
                    : $"[WoWViewer] Using explicitly selected build: '{buildAlias}' for path: {gamePath}");
                
                if (!string.IsNullOrEmpty(buildAlias))
                {
                    _dbcBuild = buildAlias;
                    ViewerLog.Trace($"[WoWViewer] Loading DBCs via DBCD (build: {buildAlias}, DBDs: {dbdDir})");
                    _texResolver.LoadFromDBC(dbcProvider, dbdDir, buildAlias);

                    // Load AreaTable for area name display
                    _areaTableService = new AreaTableService();
                    _areaTableService.Load(dbcProvider, dbdDir, buildAlias);
                }
                else
                {
                    _dbcBuild = null;
                    ViewerLog.Trace("[WoWViewer] Could not determine build version. DBC texture resolution unavailable.");
                }
            }
            else
            {
                _dbcBuild = null;
                ViewerLog.Trace("[WoWViewer] WoWDBDefs definitions not found. DBC texture resolution unavailable.");
            }

            RefreshDiscoveredMaps();

            RefreshFileList();

            if (!deferWorldReload)
                RestoreWorldAfterDataSourceReload();
        }
        catch (Exception ex)
        {
            _statusMessage = $"Failed to load MPQs: {ex.Message}";
        }
    }

    internal string? GetActiveGamePath()
    {
        if (_dataSource is MpqDataSource mpqDataSource && !string.IsNullOrWhiteSpace(mpqDataSource.GamePath))
            return Path.GetFullPath(mpqDataSource.GamePath);

        if (!string.IsNullOrWhiteSpace(_lastGameFolderPath))
            return Path.GetFullPath(_lastGameFolderPath);

        return null;
    }

    internal string? GetCurrentSessionMapName()
    {
        if (_terrainManager != null && !string.IsNullOrWhiteSpace(_terrainManager.MapName))
            return _terrainManager.MapName;

        if (_vlmTerrainManager != null && !string.IsNullOrWhiteSpace(_vlmTerrainManager.MapName))
            return _vlmTerrainManager.MapName;

        return null;
    }

    internal string? TryResolveCurrentMapDirectory(bool preferLooseOverlay)
    {
        string? currentMapName = GetCurrentSessionMapName();
        if (string.IsNullOrWhiteSpace(currentMapName))
            return null;

        foreach (string root in EnumerateCurrentSessionRoots(preferLooseOverlay))
        {
            string? mapDirectory = TryResolveMapDirectoryUnderRoot(root, currentMapName);
            if (!string.IsNullOrWhiteSpace(mapDirectory))
                return mapDirectory;
        }

        return null;
    }

    internal string? TryResolveCurrentMapWdtPath(bool preferLooseOverlay)
    {
        string? currentMapName = GetCurrentSessionMapName();
        if (string.IsNullOrWhiteSpace(currentMapName))
            return null;

        foreach (string root in EnumerateCurrentSessionRoots(preferLooseOverlay))
        {
            string? wdtPath = TryResolveMapWdtUnderRoot(root, currentMapName);
            if (!string.IsNullOrWhiteSpace(wdtPath))
                return wdtPath;
        }

        return null;
    }

    private IEnumerable<string> EnumerateCurrentSessionRoots(bool preferLooseOverlay)
    {
        var yielded = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        if (_dataSource is MpqDataSource mpqDataSource)
        {
            IEnumerable<string> overlayRoots = preferLooseOverlay
                ? mpqDataSource.OverlayRoots.Reverse()
                : mpqDataSource.OverlayRoots;

            foreach (string overlayRoot in overlayRoots)
            {
                string normalizedRoot = Path.GetFullPath(overlayRoot);
                if (yielded.Add(normalizedRoot))
                    yield return normalizedRoot;
            }

            string gamePath = Path.GetFullPath(mpqDataSource.GamePath);
            if (yielded.Add(gamePath))
                yield return gamePath;

            yield break;
        }

        if (!string.IsNullOrWhiteSpace(_lastLooseOverlayPath))
        {
            string looseRoot = Path.GetFullPath(_lastLooseOverlayPath);
            if (yielded.Add(looseRoot))
                yield return looseRoot;
        }

        if (!string.IsNullOrWhiteSpace(_lastGameFolderPath))
        {
            string gameRoot = Path.GetFullPath(_lastGameFolderPath);
            if (yielded.Add(gameRoot))
                yield return gameRoot;
        }
    }

    private static string? TryResolveMapDirectoryUnderRoot(string rootPath, string mapName)
    {
        if (string.IsNullOrWhiteSpace(rootPath) || string.IsNullOrWhiteSpace(mapName))
            return null;

        string[] candidates =
        {
            Path.Combine(rootPath, "World", "Maps", mapName),
            Path.Combine(rootPath, "Data", "World", "Maps", mapName),
            Path.Combine(rootPath, mapName),
        };

        foreach (string candidate in candidates)
        {
            if (Directory.Exists(candidate))
                return candidate;
        }

        return null;
    }

    private static string? TryResolveMapWdtUnderRoot(string rootPath, string mapName)
    {
        string? mapDirectory = TryResolveMapDirectoryUnderRoot(rootPath, mapName);
        if (string.IsNullOrWhiteSpace(mapDirectory))
            return null;

        string wdtPath = Path.Combine(mapDirectory, mapName + ".wdt");
        return File.Exists(wdtPath) ? wdtPath : null;
    }

    private void StageCurrentWorldForDataSourceReload()
    {
        _pendingDataSourceWorldReloadVirtualPath = null;
        _pendingDataSourceWorldReloadLocalPath = null;
        _pendingDataSourceWorldReloadCameraPosition = null;

        if (_worldScene == null || _terrainManager == null)
            return;

        string? virtualWdtPath = !string.IsNullOrWhiteSpace(_lastVirtualPath)
            && string.Equals(Path.GetExtension(_lastVirtualPath), ".wdt", StringComparison.OrdinalIgnoreCase)
            ? _lastVirtualPath
            : null;
        string? localWdtPath = TryGetLoadedLocalWdtPath();

        if (string.IsNullOrWhiteSpace(virtualWdtPath) && string.IsNullOrWhiteSpace(localWdtPath))
            return;

        _pendingDataSourceWorldReloadVirtualPath = virtualWdtPath;
        _pendingDataSourceWorldReloadLocalPath = localWdtPath;
        _pendingDataSourceWorldReloadCameraPosition = _camera.Position;
        _pendingDataSourceWorldReloadCameraYaw = _camera.Yaw;
        _pendingDataSourceWorldReloadCameraPitch = _camera.Pitch;
    }

    internal void ClearActiveSceneForDataSourceReload()
    {
        InvalidatePm4DerivedReports();
        _worldScene?.Dispose();
        _worldScene = null;
        _terrainManager?.Dispose();
        _terrainManager = null;
        _vlmTerrainManager?.Dispose();
        _vlmTerrainManager = null;
        _sqlSpawnStreaming.ResetSqlSpawnStreamingState(clearSceneSpawns: false);
        _renderer = null;
        _loadedWmo = null;
        _loadedMdx = null;
        _loadedM2Runtime = null;
    }

    internal void RestoreWorldAfterDataSourceReload()
    {
        if (_pendingDataSourceReloadGeneration != _activeDataSourceReloadGeneration)
            return;

        string? virtualPath = _pendingDataSourceWorldReloadVirtualPath;
        string? localPath = _pendingDataSourceWorldReloadLocalPath;
        Vector3? cameraPosition = _pendingDataSourceWorldReloadCameraPosition;
        float cameraYaw = _pendingDataSourceWorldReloadCameraYaw;
        float cameraPitch = _pendingDataSourceWorldReloadCameraPitch;

        _pendingDataSourceWorldReloadVirtualPath = null;
        _pendingDataSourceWorldReloadLocalPath = null;
        _pendingDataSourceWorldReloadCameraPosition = null;

        if (cameraPosition == null)
            return;

        // Probe the *new* data source for the WDT before any fallback. The previous-client
        // local cache (if any) was written by the prior data source; loading it through the
        // new data source can hang the viewer when the StandardTerrainAdapter then queries
        // ADTs that the new source does not have. Capture the result up front so the
        // status message at the end can tell the user which case they hit.
        bool newSourceHasWdt = !string.IsNullOrWhiteSpace(virtualPath)
            && _dataSource is MpqDataSource probe
            && probe.FileExists(virtualPath);

        if (!string.IsNullOrWhiteSpace(virtualPath) && _dataSource != null)
            _modelLoader.LoadFileFromDataSource(virtualPath);

        if (_worldScene == null
            && !string.IsNullOrWhiteSpace(localPath)
            && File.Exists(localPath)
            && newSourceHasWdt)
        {
            _worldLoader.LoadWdtTerrain(localPath);
        }

        if (_worldScene == null)
        {
            string missingMapName = Path.GetFileNameWithoutExtension(virtualPath ?? localPath ?? string.Empty);
            _statusMessage = !newSourceHasWdt && !string.IsNullOrWhiteSpace(virtualPath)
                ? $"Map \"{missingMapName}\" not present in the new client; previous world cleared."
                : $"Previous world could not be restored after client switch (data source: {_dataSource?.Name ?? "unknown"}).";
            return;
        }

        _camera.Position = cameraPosition.Value;
        _camera.Yaw = cameraYaw;
        _camera.Pitch = cameraPitch;
        _statusMessage = $"Reloaded world for client: {_terrainManager?.MapName ?? Path.GetFileNameWithoutExtension(virtualPath ?? localPath ?? string.Empty)}";
    }

    internal string? TryGetLoadedLocalWdtPath()
    {
        if (string.IsNullOrWhiteSpace(_loadedFilePath))
            return null;

        if (!string.Equals(Path.GetExtension(_loadedFilePath), ".wdt", StringComparison.OrdinalIgnoreCase))
            return null;

        return File.Exists(_loadedFilePath) ? _loadedFilePath : null;
    }

    internal bool HasWorldReturnTarget()
        => !string.IsNullOrWhiteSpace(_lastWorldSceneWdtPath) && File.Exists(_lastWorldSceneWdtPath);

    internal void ReturnToLastWorldScene()
    {
        if (!HasWorldReturnTarget())
        {
            _statusMessage = "No saved world scene is available to restore.";
            return;
        }

        _pendingWorldSpawnOverride = _lastWorldSceneCameraPosition;
        _worldLoader.LoadWdtTerrain(_lastWorldSceneWdtPath!);
        _camera.Yaw = _lastWorldSceneCameraYaw;
        _camera.Pitch = _lastWorldSceneCameraPitch;
        _statusMessage = $"Returned to world: {_terrainManager?.MapName ?? Path.GetFileNameWithoutExtension(_lastWorldSceneWdtPath!)}";
    }

    internal void AttachLooseMapOverlay(string selectedPath)
    {
        if (_dataSource is not MpqDataSource mpqDataSource)
        {
            _statusMessage = "Load a base MPQ game path first, then attach a loose map overlay.";
            return;
        }

        string selectedFullPath = Path.GetFullPath(selectedPath);
        string? overlayRoot = ResolveLooseMapOverlayRoot(selectedFullPath);
        if (string.IsNullOrWhiteSpace(overlayRoot))
        {
            _statusMessage = $"Selected folder must contain World\\Maps or be a map directory under World\\Maps. Selected: {selectedFullPath}";
            return;
        }

        if (!mpqDataSource.AddOverlayRoot(overlayRoot, out string normalizedRoot, out string message))
        {
            _statusMessage = $"{message} (selected: {selectedFullPath}; resolved root: {overlayRoot})";
            ViewerLog.Important(ViewerLog.Category.MpqData,
                $"Loose overlay attach failed. selected='{selectedFullPath}', resolvedRoot='{overlayRoot}', reason='{message}'");
            return;
        }

        _lastLooseOverlayPath = selectedFullPath;
        _standaloneSkinPathCache.Clear();
    _loggedStandaloneMissingSkinPaths.Clear();
        _wdlPreview.ResetWdlPreviewSupport();
        _wdlPreview.InitializeWdlPreviewSupport();
        InitializeMinimapSupport();
        RefreshDiscoveredMaps();
        RefreshFileList();
        if (_worldScene != null && (_worldScene.Pm4Overlay.ShowPm4Overlay || _worldScene.Pm4Overlay.Pm4LoadAttempted))
            _worldScene.Pm4Overlay.ReloadPm4Overlay();

        string? overlayBuildHint = TryDetectLooseOverlayBuildHint(normalizedRoot);
        if (!string.IsNullOrWhiteSpace(overlayBuildHint) && !string.Equals(_dbcBuild, overlayBuildHint, StringComparison.OrdinalIgnoreCase))
        {
            ViewerLog.Important(ViewerLog.Category.MpqData,
                $"Loose overlay at '{normalizedRoot}' carries PM4 of a format version associated with the {overlayBuildHint} era (the PM4 MVER word is a format version, NOT a build read from the file), but the active base client build is {_dbcBuild ?? "unknown"}. If PM4-linked objects do not match, try a {overlayBuildHint} base client.");
            _statusMessage = $"Attached loose map overlay: {normalizedRoot} (PM4 hint {overlayBuildHint}; current base {_dbcBuild ?? "unknown"})";
        }
        else
        {
            _statusMessage = $"Attached loose map overlay: {normalizedRoot}";
        }
    }

    private static string? TryDetectLooseOverlayBuildHint(string overlayRoot)
    {
        try
        {
            string worldMapsRoot = Path.Combine(overlayRoot, "World", "Maps");
            if (!Directory.Exists(worldMapsRoot))
                return null;

            string? pm4Path = Directory.EnumerateFiles(worldMapsRoot, "*.pm4", SearchOption.AllDirectories)
                .FirstOrDefault();
            if (string.IsNullOrWhiteSpace(pm4Path))
                return null;

            var pm4 = CorePm4DocumentReader.ReadFile(pm4Path);

            // PM4 MVER is a FORMAT VERSION WORD, not a client build number. Measured 2026-08-23:
            // the value is a constant 12304 (0x3010) across corpus files spanning a 20x size range
            // (63,628 to 1,267,923 bytes) with identical 32-byte MSHD, so it is neither a size nor
            // any content-derived quantity. Pm4VersionFormatter reads it as version 16 in the low
            // byte with an undecoded 0x30 high byte; PD4 by comparison stores 0x0030 (version 48).
            // That 12304 also happens to read like the real client build 4.0.1.12304 is a
            // coincidence of digits, and treating it as one is what put a false build in the status
            // bar. The mapping below is retained only as an ERA heuristic for picking a base client
            // - it says "files of this format version belong to this era", never "this file came
            // from that build". Do not present it as read from the file.
            return pm4.Version switch
            {
                11927 => "4.0.0.11927",
                12304 => "4.0.1.12304",
                _ => null,
            };
        }
        catch
        {
            return null;
        }
    }

    private static string? ResolveLooseMapOverlayRoot(string selectedPath)
    {
        string fullPath = Path.GetFullPath(selectedPath);
        if (!Directory.Exists(fullPath))
            return null;

        if (Directory.Exists(Path.Combine(fullPath, "World", "Maps")))
            return fullPath;

        var directoryInfo = new DirectoryInfo(fullPath);

        if (directoryInfo.Name.Equals("World", StringComparison.OrdinalIgnoreCase) &&
            Directory.Exists(Path.Combine(directoryInfo.FullName, "Maps")))
        {
            return directoryInfo.Parent?.FullName;
        }

        if (directoryInfo.Name.Equals("Maps", StringComparison.OrdinalIgnoreCase) &&
            directoryInfo.Parent?.Name.Equals("World", StringComparison.OrdinalIgnoreCase) == true)
        {
            return directoryInfo.Parent.Parent?.FullName;
        }

        if (directoryInfo.Parent?.Name.Equals("Maps", StringComparison.OrdinalIgnoreCase) == true &&
            directoryInfo.Parent.Parent?.Name.Equals("World", StringComparison.OrdinalIgnoreCase) == true)
        {
            return directoryInfo.Parent.Parent.Parent?.FullName;
        }

        // Only resolve ancestors that are part of the selected World\Maps tree.
        // Avoid broad drive-root fallback if an unrelated World\Maps exists elsewhere under the same root.
        for (DirectoryInfo? current = directoryInfo; current != null; current = current.Parent)
        {
            if (current.Name.Equals("World", StringComparison.OrdinalIgnoreCase) &&
                Directory.Exists(Path.Combine(current.FullName, "Maps")))
            {
                return current.Parent?.FullName;
            }

            if (current.Name.Equals("Maps", StringComparison.OrdinalIgnoreCase) &&
                current.Parent?.Name.Equals("World", StringComparison.OrdinalIgnoreCase) == true)
            {
                return current.Parent.Parent?.FullName;
            }

            if (current.Parent?.Name.Equals("Maps", StringComparison.OrdinalIgnoreCase) == true &&
                current.Parent.Parent?.Name.Equals("World", StringComparison.OrdinalIgnoreCase) == true)
            {
                return current.Parent.Parent.Parent?.FullName;
            }
        }

        return null;
    }

    internal static string? ResolveListfilePath(string? explicitListfilePath)
    {
        if (!string.IsNullOrWhiteSpace(explicitListfilePath) && File.Exists(explicitListfilePath))
            return explicitListfilePath;

        string[] bundledCandidates =
        {
            Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "..", "test_data", "community-listfile-withcapitals.csv")),
            Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "test_data", "community-listfile-withcapitals.csv")),
            Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "community-listfile-withcapitals.csv")),
            Path.GetFullPath(Path.Combine(Environment.CurrentDirectory, "gillijimproject_refactor", "test_data", "community-listfile-withcapitals.csv")),
            Path.GetFullPath(Path.Combine(Environment.CurrentDirectory, "test_data", "community-listfile-withcapitals.csv")),
        };

        // New builds get listfile coverage within hours, so prefer the most recently written copy
        // (bundled or downloaded) over the first bundled candidate that happens to exist.
        string? downloadedPath = ListfileDownloader.GetListfilePath();
        string? newest = bundledCandidates
            .Append(downloadedPath ?? string.Empty)
            .Where(static candidate => candidate.Length > 0 && File.Exists(candidate))
            .OrderByDescending(static candidate => File.GetLastWriteTimeUtc(candidate))
            .FirstOrDefault();
        if (newest is not null)
        {
            ViewerLog.Info(ViewerLog.Category.MpqData, $"Using listfile: {newest} ({File.GetLastWriteTime(newest):yyyy-MM-dd HH:mm})");
            return newest;
        }

        ViewerLog.Important(ViewerLog.Category.MpqData, "No external listfile available. MPQ file discovery will rely on archive-internal names only.");
        return null;
    }

    internal void InitializeMinimapSupport()
    {
        _md5Index = null;

        if (_dataSource is MpqDataSource mpqDataSource)
        {
            var searchPaths = new List<string> { mpqDataSource.GamePath };
            searchPaths.AddRange(mpqDataSource.OverlayRoots);
            searchPaths.AddRange(mpqDataSource.LooseRoots);

            if (Md5TranslateResolver.TryLoad(
                searchPaths,
                mpqDataSource.ArchiveReader.FileExists,
                mpqDataSource.ArchiveReader.ReadFile,
                out var md5Idx))
            {
                _md5Index = md5Idx;
                ViewerLog.Important(
                    ViewerLog.Category.Dbc,
                    $"Loaded MD5 Translate Index: {md5Idx?.HashToPlain.Count} entries");
            }
            else
            {
                ViewerLog.Trace(
                    $"[WoWViewer] No MD5 translate index found for minimaps under '{mpqDataSource.GamePath}'. Minimap loading will fall back to direct tile path variants.");
            }
        }

        _minimapRenderer?.Dispose();
        _minimapRenderer = null;
        if (_dataSource != null)
        {
            string minimapCacheSegment = WdlPreviewService.BuildCacheSegment(_wdlPreview.BuildWdlPreviewCacheIdentity());
            _minimapRenderer = new MinimapRenderer(_gl, _dataSource, _md5Index, Path.Combine(CacheDir, "minimap", minimapCacheSegment));
        }
    }
}
