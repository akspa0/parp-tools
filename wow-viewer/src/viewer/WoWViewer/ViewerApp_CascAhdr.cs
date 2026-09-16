using WoWViewer.DataSources;
using WoWViewer.Logging;
using WoWViewer.Rendering;
using WoWViewer.Terrain;
using WowViewer.Core.IO.Casc;

namespace WoWViewer;

/// <summary>
/// Spec 238 (local CASC install as the data source) and Spec 237 (DAT v26 terrain folder) entry points.
/// </summary>
public partial class ViewerApp
{
    private bool _wantOpenCascInstall;
    private bool _wantOpenAhdrTerrainFolder;
    private string? _lastCascInstallPath;
    private string? _lastAhdrTerrainFolder;

    private void HandleCascAhdrMenuRequests()
    {
        if (_wantOpenCascInstall)
        {
            _wantOpenCascInstall = false;
            ImGuiPathPicker.Instance.Open(
                "Select a CASC install folder (contains .build.info)",
                pickFolder: true,
                initialPath: _lastCascInstallPath,
                filterExtension: null,
                path =>
                {
                    if (!string.IsNullOrEmpty(path) && Directory.Exists(path))
                        LoadCascDataSource(path);
                });
        }

        if (_wantOpenAhdrTerrainFolder)
        {
            _wantOpenAhdrTerrainFolder = false;
            ImGuiPathPicker.Instance.Open(
                "Select a folder of DAT v26 terrain files (any file names)",
                pickFolder: true,
                initialPath: _lastAhdrTerrainFolder,
                filterExtension: null,
                path =>
                {
                    if (!string.IsNullOrEmpty(path) && Directory.Exists(path))
                        LoadAhdrTerrain(path);
                });
        }
    }

    /// <summary>
    /// Opens every product listed in the install's .build.info. Reads try products newest version
    /// first and fall through to older products when a newer one lacks the data locally.
    /// </summary>
    private void LoadCascDataSource(string installDir)
    {
        try
        {
            _statusMessage = $"Opening CASC install {installDir}...";
            string? listfilePath = ResolveListfilePath(null);
            if (listfilePath is null)
            {
                _statusMessage = "CASC needs a community listfile (id;path CSV); none could be found or downloaded.";
                return;
            }

            IReadOnlyList<CascProductInfo> products = CascStorage.ListProducts(installDir);
            string cascCacheDir = Path.Combine(CacheDir, "casc");
            var storages = new List<CascStorage>();
            foreach (CascProductInfo product in products.OrderByDescending(static p => Version.TryParse(p.Version, out Version? v) ? v : new Version()))
            {
                try
                {
                    storages.Add(CascStorage.OpenLocal(installDir, product.Product, cascCacheDir));
                }
                catch (Exception ex)
                {
                    ViewerLog.Important(ViewerLog.Category.MpqData, $"CASC product {product.Product} {product.Version} failed to open: {ex.Message}");
                }
            }

            if (storages.Count == 0)
            {
                _statusMessage = $"No CASC product in {installDir} could be opened.";
                return;
            }

            CommunityListfile listfile = CommunityListfile.Load([listfilePath]);

            ClearActiveSceneForDataSourceReload();
            _lastCascInstallPath = Path.GetFullPath(installDir);
            _standaloneSkinPathCache.Clear();
            _loggedStandaloneMissingSkinPaths.Clear();
            _discoveredMaps.Clear();
            _areaTableService = null;
            ResetWdlPreviewSupport();
            _dataSource?.Dispose();
            _dataSource = new CascDataSource(storages, listfile);
            InitializeWdlPreviewSupport();

            _texResolver = new ReplaceableTextureResolver();
            _texResolver.SetDataSource(_dataSource);
            _catalogView?.SetDataSource(_dataSource, _texResolver);
            _dbcProvider = new MpqDBCProvider(_dataSource);
            _dbcBuild = null; // DB2 table support for CASC builds is Spec 239.
            InitializeMinimapSupport();
            RefreshDiscoveredMaps();
            RefreshFileList();

            _statusMessage = $"Loaded: {_dataSource.Name} (listfile: {listfile.Count} entries)";
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[ViewerApp] CASC load failed: {ex}");
            _statusMessage = $"Failed to open CASC install: {ex.Message}";
        }
    }

    private void LoadAhdrTerrain(string folder)
    {
        _lastAhdrTerrainFolder = Path.GetFullPath(folder);
        _statusMessage = $"Scanning DAT v26 terrain files in {folder}...";

        AhdrTerrainAdapter adapter;
        try
        {
            adapter = new AhdrTerrainAdapter(folder);
        }
        catch (Exception ex)
        {
            _statusMessage = $"DAT v26 scan failed: {ex.Message}";
            return;
        }

        if (adapter.ExistingTiles.Count == 0)
        {
            _statusMessage = $"No DAT v26 (AHDR-family) terrain files with ALOC found in {folder}.";
            return;
        }

        foreach (string skipped in adapter.SkippedFiles)
            ViewerLog.Info(ViewerLog.Category.MpqData, $"[DAT v26] skipped {skipped}");

        string mapName = "DAT v26: " + Path.GetFileName(Path.TrimEndingDirectorySeparator(folder));
        LoadTerrainFromAdapter(adapter, mapName,
            $"Type: DAT v26 terrain folder (provisional: no objects, vertex colours or shadows yet)\n" +
            $"Folder: {folder}\n" +
            $"Tiles: {adapter.ExistingTiles.Count} (skipped {adapter.SkippedFiles.Count})\n" +
            $"Assets: {(_dataSource is null ? "no data source open; textures will be missing" : _dataSource.Name)}\n");
    }

    /// <summary>Shared world load for adapters that are not WDT-driven (mirrors the Rosetta datastore path).</summary>
    private void LoadTerrainFromAdapter(ITerrainAdapter adapter, string mapName, string modelInfoHeader)
    {
        ResetTerrainWeakSignalRestoreSessionState(preserveToggle: true);
        InvalidatePm4DerivedReports();
        _worldScene?.Dispose();
        _worldScene = null;
        _terrainManager?.Dispose();
        _terrainManager = null;
        _vlmTerrainManager?.Dispose();
        _vlmTerrainManager = null;
        ResetSqlSpawnStreamingState(clearSceneSpawns: false);

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

            var tm = new TerrainManager(_gl, adapter, mapName, _dataSource);
            _worldScene = new WorldScene(_gl, tm, _dataSource, _texResolver, _dbcBuild, _minimapRenderer, onStatus: OnLoadStatus);
            _terrainManager = _worldScene.Terrain;
            _terrainManager.DetailedTileCountOverride = _savedDetailedAdtTileCountOverride;
            ApplyGlobalFogDefaults(_terrainManager.Lighting);
            _renderer = _worldScene;
            _worldScene.EnableLitFallback($"{mapName} loaded directly; LIT/analytical lighting enabled.");

            var startPos = _terrainManager.GetInitialCameraPosition();
            _camera.Position = startPos;
            _camera.Yaw = 180f;
            _camera.Pitch = -20f;
            _terrainManager.UpdateAOI(startPos, _camera.Forward);

            _modelInfo = modelInfoHeader + $"\nCamera: ({startPos.X:F0}, {startPos.Y:F0}, {startPos.Z:F0})\n";
            _statusMessage = $"Loaded {mapName} ({adapter.ExistingTiles.Count} tiles)";
            _loadingScreen?.SetWorldLoaded();
            PresentLoadingFrame();
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[ViewerApp] {mapName} load failed: {ex}");
            _statusMessage = $"Load failed: {ex.Message}";
            _modelInfo = $"{mapName} load error:\n{ex.Message}";
            _worldScene?.Dispose();
            _worldScene = null;
            _terrainManager = null;
            _loadingScreen?.Disable();
        }
    }
}
