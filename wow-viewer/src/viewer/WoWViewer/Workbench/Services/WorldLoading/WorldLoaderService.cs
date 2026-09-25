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
using WowViewer.Core.Runtime.Marketing;
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

namespace WoWViewer;

/// <summary>
/// World loading: WDT/Alpha-WDT, Rosetta datastore, Zarr and VLM terrain loads, map WDT resolution, default-spawn map opening and build inference from paths.
/// Extracted verbatim from <see cref="ViewerApp"/> (Epic 251 U-01). World and app state it
/// needs comes only through <see cref="IViewerAppHost"/>; the bridge members below keep the
/// names the moved code used inside ViewerApp, so no moved body was edited.
/// </summary>
internal sealed partial class WorldLoaderService
{
    private readonly IViewerAppHost _host;

    internal WorldLoaderService(IViewerAppHost host)
    {
        _host = host;
    }

    // Host bridge (same names as the former ViewerApp members).
    private ref AreaTableService? _areaTableService => ref _host.AreaTableService;
    private ref Camera _camera => ref _host.Camera;
    private ref int _currentMapId => ref _host.CurrentMapId;
    private ref IDataSource? _dataSource => ref _host.DataSource;
    private ref string? _dbcBuild => ref _host.DbcBuild;
    private ref DBCD.Providers.IDBCProvider? _dbcProvider => ref _host.DbcProvider;
    private ref string? _dbdDir => ref _host.DbdDir;
    private ref float _defaultFogEnd => ref _host.DefaultFogEnd;
    private ref float _defaultFogStart => ref _host.DefaultFogStart;
    private ref List<MapDefinition> _discoveredMaps => ref _host.DiscoveredMaps;
    private ref GL _gl => ref _host.Gl;
    private ref Rendering.LoadingScreen? _loadingScreen => ref _host.LoadingScreen;
    private ref MinimapRenderer? _minimapRenderer => ref _host.MinimapRenderer;
    private ref string _modelInfo => ref _host.ModelInfo;
    private StandaloneModelLoaderService _modelLoader => _host.ModelLoader;
    private ref Vector3? _pendingWorldSpawnOverride => ref _host.PendingWorldSpawnOverride;
    private ref ISceneRenderer? _renderer => ref _host.Renderer;
    private HashSet<string> _reportedAreaDiagnostics => _host.ReportedAreaDiagnostics;
    private ref int _savedDetailedAdtTileCountOverride => ref _host.SavedDetailedAdtTileCountOverride;
    private Dictionary<string, SavedObjectPathFilterMap> _savedObjectPathFiltersByMap => _host.SavedObjectPathFiltersByMap;
    private ref MapDefinition? _selectedMapForPreview => ref _host.SelectedMapForPreview;
    private ref Vector2? _selectedSpawnTile => ref _host.SelectedSpawnTile;
    private ref bool _showWdlPreview => ref _host.ShowWdlPreview;
    private ref bool _sqlForceStreamRefresh => ref _host.SqlForceStreamRefresh;
    private SqlSpawnStreamingService _sqlSpawnStreaming => _host.SqlSpawnStreaming;
    private ref string _statusMessage => ref _host.StatusMessage;
    private List<TerrainHiddenTileCandidate> _terrainAnalysisHiddenCandidates => _host.TerrainAnalysisHiddenCandidates;
    private ref int _terrainAnalysisHiddenSelectedIndex => ref _host.TerrainAnalysisHiddenSelectedIndex;
    private ref string _terrainAnalysisHiddenStatus => ref _host.TerrainAnalysisHiddenStatus;
    private ref (int tileX, int tileY)? _terrainAnalysisPreviewCompareTile => ref _host.TerrainAnalysisPreviewCompareTile;
    private ref float? _terrainAnalysisPreviewSimilarity => ref _host.TerrainAnalysisPreviewSimilarity;
    private ref TerrainManager? _terrainManager => ref _host.TerrainManager;
    private TerrainWeakSignalRestoreService _terrainWeakSignalRestore => _host.TerrainWeakSignalRestore;
    private ref ReplaceableTextureResolver? _texResolver => ref _host.TexResolver;
    private ref VlmTerrainManager? _vlmTerrainManager => ref _host.VlmTerrainManager;
    private ref IWindow _window => ref _host.Window;
    private ref WorldScene? _worldScene => ref _host.WorldScene;
    private void ApplyLayoutObjectPreviewModeToScene() => _host.ApplyLayoutObjectPreviewModeToScene();
    private void ApplySavedPm4AlignmentToScene() => _host.ApplySavedPm4AlignmentToScene();
    private string? GetCurrentSessionMapName() => _host.GetCurrentSessionMapName();
    private void InvalidatePm4DerivedReports() => _host.InvalidatePm4DerivedReports();
    private bool FullLoadMode { get => _host.FullLoadMode; set => _host.FullLoadMode = value; }


    private void ApplySavedObjectPathFiltersForCurrentMap()
    {
        if (_worldScene == null)
            return;

        _worldScene.ClearObjectPathFilters();
        _worldScene.ObjectPathFiltersEnabled = true;

        string? currentMapName = GetCurrentSessionMapName();
        if (string.IsNullOrWhiteSpace(currentMapName))
            return;

        if (!_savedObjectPathFiltersByMap.TryGetValue(currentMapName, out SavedObjectPathFilterMap? savedMap))
            return;

        _worldScene.ObjectPathFiltersEnabled = savedMap.Enabled;
        foreach (SavedObjectPathFilterEntry filter in savedMap.Filters)
            _worldScene.AddObjectPathFilter(filter.PathPrefix, filter.AppliesToWmo, filter.AppliesToMdx);
    }

    private static IEnumerable<string> EnumerateMapWdtCandidates(string mapDirectory)
    {
        string basePath = $"World\\Maps\\{mapDirectory}\\{mapDirectory}.wdt";
        yield return basePath;
    }

    internal string? ResolveMapWdtPath(string mapDirectory)
    {
        if (_dataSource == null)
            return null;

        foreach (string candidate in EnumerateMapWdtCandidates(mapDirectory))
        {
            byte[]? data = _dataSource.ReadFile(candidate);
            if (data != null && data.Length > 0)
                return candidate;

            if (_dataSource is not MpqDataSource mpqDataSource)
                continue;

            string? found = mpqDataSource.FindInFileSet(candidate);
            if (string.IsNullOrWhiteSpace(found))
                continue;

            data = _dataSource.ReadFile(found);
            if (data != null && data.Length > 0)
                return found;
        }

        return null;
    }

    internal void LoadMapAtDefaultSpawn(MapDefinition map)
    {
        if (!map.HasWdt)
            return;

        string? resolvedWdtPath = ResolveMapWdtPath(map.Directory);
        if (string.IsNullOrWhiteSpace(resolvedWdtPath))
        {
            _statusMessage = $"Failed to resolve WDT for {map.Directory}.";
            ViewerLog.Important(ViewerLog.Category.Terrain,
                $"[WorldLoad] Failed to resolve map WDT for {map.Directory} from discovery actions.");
            return;
        }

        _selectedMapForPreview = null;
        _selectedSpawnTile = null;
        _pendingWorldSpawnOverride = null;
        _showWdlPreview = false;

        _modelLoader.LoadFileFromDataSource(resolvedWdtPath);
    }

    /// <summary>
    /// Infer the full build string (e.g. "0.10.0.3892") from the game path.
    /// Strategy:
    ///   1. Regex-extract all X.Y.Z.NNNN candidates from the path
    ///   2. Validate each against WoWDBDefs BUILD lines
    ///   3. If no 4-part match, try X.Y.Z short versions and resolve to full build via DBD
    ///   4. Fallback: MPQ heuristics for 3.3.5
    /// </summary>
    internal static string InferBuildFromPath(string path, string? dbdDir)
    {
        // Collect all known builds from WoWDBDefs (cached per call)
        HashSet<string> dbdBuilds = new(StringComparer.OrdinalIgnoreCase);
        if (!string.IsNullOrEmpty(dbdDir) && Directory.Exists(dbdDir))
        {
            // Parse Map.dbd — it covers all versions and is always present
            var mapDbd = Path.Combine(dbdDir, "Map.dbd");
            if (File.Exists(mapDbd))
            {
                foreach (var line in File.ReadLines(mapDbd))
                {
                    var trimmed = line.Trim();
                    if (!trimmed.StartsWith("BUILD ")) continue;
                    // Parse "BUILD X.Y.Z.NNNN" or "BUILD X.Y.Z.NNNN-X.Y.Z.NNNN" or comma-separated
                    var parts = trimmed[6..].Split(',', StringSplitOptions.TrimEntries);
                    foreach (var part in parts)
                    {
                        // Handle ranges: "0.9.0.3807-0.12.0.3988"
                        var rangeParts = part.Split('-', StringSplitOptions.TrimEntries);
                        foreach (var rp in rangeParts)
                            if (Regex.IsMatch(rp, @"^\d+\.\d+\.\d+\.\d+$"))
                                dbdBuilds.Add(rp);
                    }
                }
            }
        }
        ViewerLog.Trace($"[BuildDetect] Loaded {dbdBuilds.Count} known builds from WoWDBDefs");

        // 1. Extract all X.Y.Z.NNNN candidates from the path
        var fullMatches = Regex.Matches(path, @"(\d+\.\d+\.\d+\.\d+)");
        foreach (Match m in fullMatches)
        {
            string candidate = m.Groups[1].Value;
            if (dbdBuilds.Contains(candidate))
            {
                ViewerLog.Trace($"[BuildDetect] Exact match from path: {candidate}");
                return candidate;
            }
        }

        // 2. Extract X.Y.Z short versions and find matching full build in DBD
        var shortMatches = Regex.Matches(path, @"(\d+\.\d+\.\d+)");
        foreach (Match m in shortMatches)
        {
            string shortVer = m.Groups[1].Value;
            // Find any DBD build that starts with this short version
            var match = dbdBuilds.FirstOrDefault(b => b.StartsWith(shortVer + "."));
            if (!string.IsNullOrEmpty(match))
            {
                ViewerLog.Trace($"[BuildDetect] Short version '{shortVer}' resolved to: {match}");
                return match;
            }
        }

        // 3. Check for full build in path that might be in a BUILD range (not exact endpoint)
        foreach (Match m in fullMatches)
        {
            string candidate = m.Groups[1].Value;
            // Try to find it in DBD range lines
            string? rangeMatch = FindBuildInDbdRanges(dbdDir, candidate);
            if (!string.IsNullOrEmpty(rangeMatch))
            {
                ViewerLog.Trace($"[BuildDetect] Range match from path: {candidate}");
                return candidate;
            }
        }

        // 4. Fallback: MPQ heuristics
        if (Directory.Exists(path))
        {
            try
            {
                var mpqs = Directory.GetFiles(path, "*.mpq", SearchOption.AllDirectories)
                    .Select(f => Path.GetFileName(f).ToLowerInvariant()).ToArray();

                // LK 3.3.5: has patch MPQs with "3" in name
                if (mpqs.Any(m => m.Contains("patch") && m.Contains("3")))
                {
                    var lkBuild = dbdBuilds.FirstOrDefault(b => b.StartsWith("3.3.5."));
                    return lkBuild ?? "3.3.5.12340";
                }

                // Alpha 0.5.3: dbc.mpq + model.mpq + texture.mpq, no common.mpq or patch-*.mpq
                bool hasAlphaSignature = mpqs.Contains("dbc.mpq")
                    && mpqs.Contains("model.mpq")
                    && mpqs.Contains("texture.mpq")
                    && !mpqs.Any(m => m.StartsWith("common"))
                    && !mpqs.Any(m => m.StartsWith("patch-"));
                if (hasAlphaSignature)
                {
                    // Check for patch.mpq → 0.7.0+, otherwise 0.5.3
                    bool hasPatch = mpqs.Contains("patch.mpq");
                    if (hasPatch)
                    {
                        // 0.6.0–0.8.0 range: try each in order
                        foreach (var prefix in new[] { "0.8.0.", "0.7.0.", "0.6.0." })
                        {
                            var match = dbdBuilds.FirstOrDefault(b => b.StartsWith(prefix));
                            if (!string.IsNullOrEmpty(match))
                            {
                                ViewerLog.Trace($"[BuildDetect] MPQ heuristic (alpha+patch): {match}");
                                return match;
                            }
                        }
                    }
                    else
                    {
                        var alphaBuild = dbdBuilds.FirstOrDefault(b => b.StartsWith("0.5.3."));
                        if (!string.IsNullOrEmpty(alphaBuild))
                        {
                            ViewerLog.Trace($"[BuildDetect] MPQ heuristic (alpha): {alphaBuild}");
                            return alphaBuild;
                        }
                        return "0.5.3.3368";
                    }
                }
            }
            catch { }
        }

        return "";
    }

    /// <summary>
    /// Check if a build number falls within any BUILD range in the DBD files.
    /// Parses ranges like "BUILD 0.9.0.3807-0.12.0.3988" and checks if the candidate
    /// build falls within [start, end] using numeric tuple comparison.
    /// </summary>
    private static string? FindBuildInDbdRanges(string? dbdDir, string build)
    {
        if (string.IsNullOrEmpty(dbdDir)) return null;
        var mapDbd = Path.Combine(dbdDir, "Map.dbd");
        if (!File.Exists(mapDbd)) return null;

        var buildTuple = ParseBuildTuple(build);
        if (buildTuple == null) return null;

        foreach (var line in File.ReadLines(mapDbd))
        {
            var trimmed = line.Trim();
            if (!trimmed.StartsWith("BUILD ")) continue;

            // Check explicit listing first
            if (trimmed.Contains(build)) return build;

            // Check ranges: "BUILD 0.9.0.3807-0.12.0.3988"
            var entries = trimmed[6..].Split(',', StringSplitOptions.TrimEntries);
            foreach (var entry in entries)
            {
                var rangeParts = entry.Split('-', StringSplitOptions.TrimEntries);
                if (rangeParts.Length == 2)
                {
                    var lo = ParseBuildTuple(rangeParts[0]);
                    var hi = ParseBuildTuple(rangeParts[1]);
                    if (lo != null && hi != null &&
                        CompareBuild(buildTuple, lo) >= 0 &&
                        CompareBuild(buildTuple, hi) <= 0)
                    {
                        ViewerLog.Trace($"[BuildDetect] '{build}' falls within range {rangeParts[0]}-{rangeParts[1]}");
                        return build;
                    }
                }
            }
        }
        return null;
    }

    private static int[]? ParseBuildTuple(string build)
    {
        var parts = build.Split('.');
        if (parts.Length != 4) return null;
        var nums = new int[4];
        for (int i = 0; i < 4; i++)
            if (!int.TryParse(parts[i], out nums[i])) return null;
        return nums;
    }

    private static int CompareBuild(int[] a, int[] b)
    {
        for (int i = 0; i < 4; i++)
        {
            if (a[i] < b[i]) return -1;
            if (a[i] > b[i]) return 1;
        }
        return 0;
    }

    internal void LoadWdtTerrain(string wdtPath)
    {
        _statusMessage = $"Loading world from {Path.GetFileName(wdtPath)}...";

        _terrainWeakSignalRestore.ResetTerrainWeakSignalRestoreSessionState(preserveToggle: true);
        InvalidatePm4DerivedReports();
        _worldScene?.Dispose();
        _worldScene = null;
        _terrainManager?.Dispose();
        _terrainManager = null;
        _vlmTerrainManager?.Dispose();
        _vlmTerrainManager = null;
        _sqlSpawnStreaming.ResetSqlSpawnStreamingState(clearSceneSpawns: false);

        // Show loading screen (replicates Alpha client's EnableLoadingScreen)
        _loadingScreen?.Enable(_dataSource);
        PresentLoadingFrame();

        try
        {
            // Detect Alpha WDT vs Standard WDT by checking for MDNM chunk.
            // Alpha WDTs are monolithic: MVER+MPHD+MAIN+MDNM+MONM+embedded ADTs.
            // Standard WDTs have: MVER+MPHD+MAIN only, referencing external .adt files.
            var wdtRawBytes = File.ReadAllBytes(wdtPath);
            bool isAlpha = StandaloneModelLoaderService.DetectAlphaWdt(wdtRawBytes);
            string wdtType;
            int loadStep = 0;

            // onStatus callback: update loading screen progress and force-present a frame.
            // This replicates the Alpha client's UpdateProgressBar → GxScenePresent pattern.
            void OnLoadStatus(string status)
            {
                _statusMessage = status;
                loadStep++;
                _loadingScreen?.UpdateProgress(loadStep, 20); // Estimate ~20 status updates per load
                PresentLoadingFrame();
            }

            if (isAlpha)
            {
                // Alpha WDT: monolithic file with embedded ADTs
                _worldScene = new WorldScene(_gl, wdtPath, _dataSource, _texResolver, _dbcBuild, _minimapRenderer,
                    onStatus: OnLoadStatus);
                wdtType = "Alpha WDT";
            }
            else
            {
                // Standard WDT: small file referencing separate ADT files via IDataSource (MPQ)
                if (_dataSource == null)
                {
                    _loadingScreen?.Disable();
                    _statusMessage = "Standard WDT requires an MPQ data source. Open a game folder first.";
                    _modelInfo = "Standard WDT detected but no data source loaded.\n\nUse File > Open Game Folder to load MPQ archives first,\nthen open the WDT from the file browser.";
                    return;
                }

                string mapName = Path.GetFileNameWithoutExtension(wdtPath);
                var adapter = new Terrain.StandardTerrainAdapter(wdtRawBytes, mapName, _dataSource, _dbcBuild, _dbcProvider, _dbdDir);
                var tm = new Terrain.TerrainManager(_gl, adapter, mapName, _dataSource);
                _worldScene = new WorldScene(_gl, tm, _dataSource, _texResolver, _dbcBuild, _minimapRenderer,
                    onStatus: OnLoadStatus);
                wdtType = "Standard WDT";
            }

            _terrainManager = _worldScene.Terrain;
            _terrainManager.DetailedTileCountOverride = _savedDetailedAdtTileCountOverride;
            ApplyGlobalFogDefaults(_terrainManager.Lighting);
            _terrainWeakSignalRestore.RefreshTerrainWeakSignalRestoreHooks();
            _terrainWeakSignalRestore.RefreshTerrainWeakSignalRestoreForLoadedTiles();
            _terrainWeakSignalRestore.WireBaseWdlEdgeBlendLookup();
            _renderer = _worldScene;
            ApplyLayoutObjectPreviewModeToScene();
            ApplySavedPm4AlignmentToScene();
            ApplySavedObjectPathFiltersForCurrentMap();
            // Full-load mode: load all tiles synchronously during loading screen
            if (FullLoadMode && !_terrainManager.Adapter.IsWmoBased)
            {
                int total = _terrainManager.Adapter.ExistingTiles.Count;
                ViewerLog.Important(ViewerLog.Category.Terrain,
                    $"Full-load mode: loading all {total} tiles...");
                _terrainManager.LoadAllTiles((loaded, tot, tileName) =>
                {
                    _statusMessage = $"Loading tiles... {loaded}/{tot} ({tileName})";
                    _loadingScreen?.UpdateProgress(loaded, tot);
                    PresentLoadingFrame();
                });
                ViewerLog.Important(ViewerLog.Category.Terrain,
                    $"Full-load complete: {_terrainManager.LoadedTileCount} tiles, {_terrainManager.LoadedChunkCount} chunks");
            }

            // Find mapId for this world
            string curMapName = _terrainManager.MapName;
            var curMapDef = _discoveredMaps.FirstOrDefault(m =>
                string.Equals(m.Directory, curMapName, StringComparison.OrdinalIgnoreCase));
            _currentMapId = curMapDef?.HasDbcEntry == true ? curMapDef.Id : -1;
            _reportedAreaDiagnostics.Clear();
            ViewerLog.Important(ViewerLog.Category.General,
                $"[WorldLoad] Map='{curMapName}' resolvedMapId={_currentMapId} build={_dbcBuild ?? "unknown"} areaTable={_areaTableService?.DescribeLoadContext() ?? "not loaded"}");
            _sqlForceStreamRefresh = true;

            // Store DBC credentials for lazy loading (POI + Taxi deferred until user toggles them on)
            // Only Lighting is loaded eagerly since it affects rendering immediately.
            if (_dbcProvider != null && _dbdDir != null && _dbcBuild != null)
            {
                int mapId = curMapDef?.HasDbcEntry == true ? curMapDef.Id : -1;
                _worldScene.SetDbcCredentials(_dbcProvider, _dbdDir, _dbcBuild, mapId);

                _worldScene.LoadLighting(_dbcProvider, _dbdDir, _dbcBuild, mapId);
            }
            else
            {
                _worldScene.EnableLitFallback(
                    "No Light DBC provider is available for this client; LIT is enabled automatically.");
            }

            // Position camera — WMO-only maps use the WMO position, terrain maps use tile center
            var startPos = _pendingWorldSpawnOverride ?? _worldScene.WmoCameraOverride ?? _terrainManager.GetInitialCameraPosition();
            _camera.Position = startPos;
            _pendingWorldSpawnOverride = null;
            _camera.Yaw = 180f;
            _camera.Pitch = -20f;
            if (!_terrainManager.Adapter.IsWmoBased)
                _terrainManager.UpdateAOI(startPos, _camera.Forward);

            int poiCount = _worldScene.PoiLoader?.Entries.Count ?? 0;
            int taxiNodeCount = _worldScene.TaxiLoader?.Nodes.Count ?? 0;
            int taxiRouteCount = _worldScene.TaxiLoader?.Routes.Count ?? 0;
            _modelInfo = $"Type: {wdtType} World\n" +
                         $"Map: {_terrainManager.MapName}\n\n" +
                         $"Tiles: {_terrainManager.LoadedTileCount}\n" +
                         $"Chunks: {_terrainManager.LoadedChunkCount}\n\n" +
                         $"WMO instances: {_worldScene.WmoInstanceCount} ({_worldScene.UniqueWmoModels} unique)\n" +
                         $"MDX instances: {_worldScene.MdxInstanceCount} ({_worldScene.UniqueMdxModels} unique)\n" +
                         (poiCount > 0 ? $"Area POIs: {poiCount}\n" : "") +
                         (taxiNodeCount > 0 ? $"Taxi Nodes: {taxiNodeCount}, Routes: {taxiRouteCount}\n" : "") +
                         $"\nCamera: ({startPos.X:F0}, {startPos.Y:F0}, {startPos.Z:F0})\n";

            _statusMessage = $"Loaded world: {_terrainManager.MapName} ({_terrainManager.LoadedTileCount} tiles, {_worldScene.WmoInstanceCount} WMOs, {_worldScene.MdxInstanceCount} doodads)";

            // Signal world loaded (progress → 75%). Loading screen stays active
            // until the first terrain tiles are actually rendered (checked in OnRender).
            _loadingScreen?.SetWorldLoaded();
            PresentLoadingFrame();
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[ViewerApp] WDT load failed: {ex}");
            _statusMessage = $"Load failed: {ex.Message}";
            _modelInfo = $"WDT load error:\n{ex.Message}\n\nFile: {wdtPath}\nSize: {(File.Exists(wdtPath) ? new FileInfo(wdtPath).Length : 0)} bytes";
            _worldScene?.Dispose();
            _worldScene = null;
            _terrainManager = null;
            InvalidatePm4DerivedReports();
            _loadingScreen?.Disable();
        }
    }

    internal void LoadRosettaDatastoreTerrain(WowViewer.Core.IO.Maps.RosettaObjectLibrary library, string buildId, string mapName)
    {
        _statusMessage = $"Loading Rosetta world '{mapName}' [{buildId}] from Zarr datastore...";

        _terrainWeakSignalRestore.ResetTerrainWeakSignalRestoreSessionState(preserveToggle: true);
        InvalidatePm4DerivedReports();
        _worldScene?.Dispose();
        _worldScene = null;
        _terrainManager?.Dispose();
        _terrainManager = null;
        _vlmTerrainManager?.Dispose();
        _vlmTerrainManager = null;
        _sqlSpawnStreaming.ResetSqlSpawnStreamingState(clearSceneSpawns: false);

        // Show loading screen
        _loadingScreen?.Enable(_dataSource);
        PresentLoadingFrame();

        try
        {
            int loadStep = 0;
            void OnLoadStatus(string status)
            {
                _statusMessage = status;
                loadStep++;
                _loadingScreen?.UpdateProgress(loadStep, 20);
                PresentLoadingFrame();
            }

            var adapter = new Terrain.RosettaDatastoreTerrainAdapter(library, buildId, mapName, _dataSource);
            var tm = new Terrain.TerrainManager(_gl, adapter, mapName, _dataSource);
            _worldScene = new WorldScene(_gl, tm, _dataSource, _texResolver, _dbcBuild, _minimapRenderer,
                onStatus: OnLoadStatus);

            _terrainManager = _worldScene.Terrain;
            _terrainManager.DetailedTileCountOverride = _savedDetailedAdtTileCountOverride;
            _terrainWeakSignalRestore.WireBaseWdlEdgeBlendLookup();
            ApplyGlobalFogDefaults(_terrainManager.Lighting);
            _terrainWeakSignalRestore.RefreshTerrainWeakSignalRestoreHooks();
            _terrainWeakSignalRestore.RefreshTerrainWeakSignalRestoreForLoadedTiles();
            _renderer = _worldScene;
            ApplyLayoutObjectPreviewModeToScene();
            ApplySavedPm4AlignmentToScene();
            ApplySavedObjectPathFiltersForCurrentMap();

            _worldScene.EnableLitFallback(
                "Rosetta Zarr map loaded directly; LIT/analytical lighting enabled.");

            // Full-load mode: load all tiles synchronously during loading screen
            if (FullLoadMode && !_terrainManager.Adapter.IsWmoBased)
            {
                int total = _terrainManager.Adapter.ExistingTiles.Count;
                _terrainManager.LoadAllTiles((loaded, tot, tileName) =>
                {
                    _statusMessage = $"Loading tiles... {loaded}/{tot} ({tileName})";
                    _loadingScreen?.UpdateProgress(loaded, tot);
                    PresentLoadingFrame();
                });
            }

            // Position camera at initial tile center
            var startPos = _pendingWorldSpawnOverride ?? _worldScene.WmoCameraOverride ?? _terrainManager.GetInitialCameraPosition();
            _camera.Position = startPos;
            _pendingWorldSpawnOverride = null;
            _camera.Yaw = 180f;
            _camera.Pitch = -20f;
            if (!_terrainManager.Adapter.IsWmoBased)
                _terrainManager.UpdateAOI(startPos, _camera.Forward);

            _modelInfo = $"Type: Rosetta Zarr Datastore World\n" +
                         $"Build: {buildId}\n" +
                         $"Map: {mapName}\n\n" +
                         $"Tiles: {adapter.ExistingTiles.Count}\n" +
                         $"WMO instances: {_worldScene.WmoInstanceCount} ({_worldScene.UniqueWmoModels} unique)\n" +
                         $"MDX instances: {_worldScene.MdxInstanceCount} ({_worldScene.UniqueMdxModels} unique)\n" +
                         $"\nCamera: ({startPos.X:F0}, {startPos.Y:F0}, {startPos.Z:F0})\n";

            _statusMessage = $"Loaded Rosetta Zarr World: {mapName} [{buildId}] ({adapter.ExistingTiles.Count} tiles, {_worldScene.WmoInstanceCount} WMOs, {_worldScene.MdxInstanceCount} doodads)";

            _loadingScreen?.SetWorldLoaded();
            PresentLoadingFrame();
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[ViewerApp] Rosetta Datastore load failed: {ex}");
            _statusMessage = $"Load failed: {ex.Message}";
            _modelInfo = $"Rosetta Datastore load error:\n{ex.Message}\n\nBuild: {buildId}\nMap: {mapName}";
            _worldScene?.Dispose();
            _worldScene = null;
            _terrainManager = null;
            InvalidatePm4DerivedReports();
            _loadingScreen?.Disable();
        }
    }

    /// <summary>
    /// Force-present a loading screen frame. Replicates the Alpha client's
    /// UpdateProgressBar → GxScenePresent pattern: clear, draw loading screen, swap.
    /// Called from the blocking WorldScene constructor via onStatus callback.
    /// </summary>
    private void PresentLoadingFrame()
    {
        if (_loadingScreen == null || !_loadingScreen.IsActive) return;
        _gl.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);
        var sz = _window.Size;
        _loadingScreen.Render(sz.X, sz.Y);
        _window.GLContext?.SwapBuffers();
    }

    internal void LoadZarrDataset(string datasetRoot)
    {
        try
        {
            var loader = new ZarrTileDatasetLoader(datasetRoot);
            ZarrStoreSummary summary = loader.Open();
            _statusMessage =
                $"Zarr store '{summary.MapName}' discovered: {summary.Arrays.Count} arrays; " +
                $"liquid mask={(summary.HasLiquidMask ? "yes" : "no")}, " +
                $"height={(summary.HasLiquidHeight ? "yes" : "no")}, " +
                $"type={(summary.HasLiquidType256 || summary.HasLiquidBasicType || summary.HasMh2oTypeMask ? "yes" : "no")}; " +
                $"objects={(summary.HasObjectSignals ? "yes" : "no")}, " +
                $"tilesets={(summary.HasTilesetSignals ? "yes" : "no")}, " +
                $"textures={(summary.HasTextureSignals ? "yes" : "no")}, " +
                $"placements={(summary.HasPlacementSignals ? "yes" : "no")}. " +
                "Summary only: per-tile loading still requires the Zarr decoder/rehydration slice.";
        }
        catch (Exception ex)
        {
            _statusMessage = $"Zarr dataset load failed: {ex.Message}";
        }
    }

    internal void LoadVlmProject(string projectRoot)
    {
        _statusMessage = $"Loading MK dataset from {projectRoot}...";

        _terrainAnalysisHiddenCandidates.Clear();
        _terrainAnalysisHiddenSelectedIndex = -1;
        _terrainAnalysisHiddenStatus = string.Empty;
        _terrainAnalysisPreviewCompareTile = null;
        _terrainAnalysisPreviewSimilarity = null;
        _terrainWeakSignalRestore.ResetTerrainWeakSignalRestoreSessionState(preserveToggle: true);

        // Clean up any existing scene
        InvalidatePm4DerivedReports();
        _worldScene?.Dispose();
        _worldScene = null;
        _terrainManager?.Dispose();
        _terrainManager = null;
        _vlmTerrainManager?.Dispose();
        _vlmTerrainManager = null;
        _renderer = null;

        try
        {
            _vlmTerrainManager = new VlmTerrainManager(_gl, projectRoot);
            ApplyGlobalFogDefaults(_vlmTerrainManager.Lighting);
            _terrainWeakSignalRestore.RefreshTerrainWeakSignalRestoreHooks();
            _terrainWeakSignalRestore.RefreshTerrainWeakSignalRestoreForLoadedTiles();
            _renderer = _vlmTerrainManager;

            // Position camera at center of loaded tiles
            var startPos = _vlmTerrainManager.GetInitialCameraPosition();
            _camera.Position = startPos;
            _camera.Yaw = 180f;
            _camera.Pitch = -20f;

            var loader = _vlmTerrainManager.Loader;
            _modelInfo = $"Type: MK Dataset\n" +
                         $"Map: {loader.MapName}\n" +
                         $"Path: {projectRoot}\n\n" +
                         $"Tiles: {loader.TileCoords.Count}\n" +
                         $"MDX names: {loader.MdxModelNames.Count}\n" +
                         $"WMO names: {loader.WmoModelNames.Count}\n" +
                         $"\nCamera: ({startPos.X:F0}, {startPos.Y:F0}, {startPos.Z:F0})\n";

            // Set MapID for AreaTable lookups
            var vlmMapDef = _discoveredMaps.FirstOrDefault(m =>
                string.Equals(m.Directory, loader.MapName, StringComparison.OrdinalIgnoreCase));
            _currentMapId = vlmMapDef?.Id ?? -1;
            _statusMessage = $"Loaded MK dataset: {loader.MapName} ({loader.TileCoords.Count} tiles)";
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[ViewerApp] VLM project load failed: {ex}");
            _statusMessage = $"MK dataset load failed: {ex.Message}";
            _modelInfo = $"MK dataset load error:\n{ex.Message}\n\nPath: {projectRoot}";
            _vlmTerrainManager?.Dispose();
            _vlmTerrainManager = null;
        }
    }
}
