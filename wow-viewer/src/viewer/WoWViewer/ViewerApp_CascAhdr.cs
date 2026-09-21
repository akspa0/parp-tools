using WoWViewer.DataSources;
using WoWViewer.Logging;
using WoWViewer.Rendering;
using WoWViewer.Terrain;
using WowViewer.Core.IO.Casc;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps.AdtAhdr;

namespace WoWViewer;

/// <summary>
/// Spec 238 (local CASC install as the data source) and Spec 237 (DAT v26 terrain folder) entry points.
/// </summary>
public partial class ViewerApp
{
    private bool _wantOpenCascInstall;
    private bool _wantOpenCascInstallWithCdnFill;

    // Product picker state: shown after a CASC install folder is chosen.
    private string? _cascPickerInstallDir;
    private IReadOnlyList<CascProductInfo> _cascPickerProducts = [];
    private int _cascPickerSelected;
    private bool _cascPickerCdnFill = true;
    private bool _cascPickerFallbackToOtherProducts;
    private string? _cascPickerError;
    private string? _lastCascProduct;
    private bool _wantOpenAhdrTerrainFolder;
    private string? _lastCascInstallPath;
    private string? _lastAhdrTerrainFolder;

    /// <summary>DAT terrain height display divisor; 36 treats heights as inches (see AhdrTerrainAdapter.HeightDivisor).</summary>
    private float _ahdrHeightDivisor = 36f;

    private static readonly (string Label, float Divisor)[] AhdrHeightScaleOptions =
    [
        ("Inches → yards (÷36)", 36f),
        ("Raw (1×)", 1f),
        ("÷12", 12f),
        ("÷100", 100f),
    ];

    private void DrawAhdrHeightScaleMenu()
    {
        if (!ImGuiNET.ImGui.BeginMenu("DAT Terrain Height Scale"))
            return;

        foreach ((string label, float divisor) in AhdrHeightScaleOptions)
        {
            if (ImGuiNET.ImGui.MenuItem(label, null, _ahdrHeightDivisor == divisor))
            {
                _ahdrHeightDivisor = divisor;
                if (_lastAhdrTerrainFolder is not null && _terrainManager?.Adapter is AhdrTerrainAdapter)
                    LoadAhdrTerrain(_lastAhdrTerrainFolder);
            }
        }

        ImGuiNET.ImGui.EndMenu();
    }

    private void HandleCascAhdrMenuRequests()
    {
        if (_wantOpenCascInstall || _wantOpenCascInstallWithCdnFill)
        {
            bool allowCdnFill = _wantOpenCascInstallWithCdnFill;
            _wantOpenCascInstall = false;
            _wantOpenCascInstallWithCdnFill = false;
            ImGuiPathPicker.Instance.Open(
                allowCdnFill
                    ? "Select a CASC install folder (missing local data is fetched from Blizzard's CDN for the same build)"
                    : "Select a CASC install folder (contains .build.info)",
                pickFolder: true,
                initialPath: _lastCascInstallPath,
                filterExtension: null,
                path =>
                {
                    if (!string.IsNullOrEmpty(path) && Directory.Exists(path))
                        OpenCascProductPicker(path, allowCdnFill);
                });
        }

        DrawCascProductPicker();

        if (_wantOpenAhdrTerrainFolder)
        {
            _wantOpenAhdrTerrainFolder = false;
            ImGuiPathPicker.Instance.Open(
                "Select a folder of DAT terrain files (v22/23/26; any file name or extension)",
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

    /// <summary>Game folder a launcher install uses for each product, for the "installed" hint.</summary>
    private static readonly Dictionary<string, string> CascProductFolders = new(StringComparer.OrdinalIgnoreCase)
    {
        ["wow"] = "_retail_",
        ["wowt"] = "_ptr_",
        ["wowxptr"] = "_xptr_",
        ["wow_beta"] = "_beta_",
        ["wow_classic"] = "_classic_",
        ["wow_classic_ptr"] = "_classic_ptr_",
        ["wow_classic_beta"] = "_classic_beta_",
        ["wow_classic_era"] = "_classic_era_",
        ["wow_classic_era_ptr"] = "_classic_era_ptr_",
        ["wow_anniversary"] = "_anniversary_",
    };

    private void OpenCascProductPicker(string installDir, bool allowCdnFill)
    {
        // The menu item chosen sets CDN fill; the checkbox in the picker can still change it.
        _cascPickerCdnFill = allowCdnFill;

        // Pickers often land one level too deep (e.g. inside _retail_); .build.info lives in the install root.
        string root = installDir.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        if (!File.Exists(Path.Combine(root, ".build.info")) && Path.GetDirectoryName(root) is { } parent && File.Exists(Path.Combine(parent, ".build.info")))
            root = parent;

        try
        {
            _cascPickerProducts = CascStorage.ListProducts(root)
                .OrderByDescending(static p => Version.TryParse(p.Version, out Version? v) ? v : new Version())
                .ToArray();
            _cascPickerError = _cascPickerProducts.Count == 0 ? ".build.info lists no products." : null;
        }
        catch (Exception ex)
        {
            _cascPickerProducts = [];
            _cascPickerError = ex.Message;
        }

        _cascPickerInstallDir = root;
        _cascPickerSelected = 0;
        for (int i = 0; i < _cascPickerProducts.Count; i++)
        {
            if (string.Equals(_cascPickerProducts[i].Product, _lastCascProduct, StringComparison.OrdinalIgnoreCase))
                _cascPickerSelected = i;
        }
    }

    private void DrawCascProductPicker()
    {
        if (_cascPickerInstallDir is null)
            return;

        bool open = true;
        ImGuiNET.ImGui.SetNextWindowSize(new System.Numerics.Vector2(620, 0), ImGuiNET.ImGuiCond.Appearing);
        ImGuiNET.ImGui.SetNextWindowPos(ImGuiNET.ImGui.GetMainViewport().GetCenter(), ImGuiNET.ImGuiCond.Appearing, new System.Numerics.Vector2(0.5f, 0.5f));
        if (ImGuiNET.ImGui.Begin("Open CASC Install###CascProductPicker", ref open, ImGuiNET.ImGuiWindowFlags.NoCollapse | ImGuiNET.ImGuiWindowFlags.AlwaysAutoResize))
        {
            ImGuiNET.ImGui.TextUnformatted(_cascPickerInstallDir);
            ImGuiNET.ImGui.Separator();

            if (_cascPickerError is not null)
            {
                ImGuiNET.ImGui.TextColored(new System.Numerics.Vector4(1f, 0.45f, 0.45f, 1f), _cascPickerError);
            }
            else
            {
                ImGuiNET.ImGui.TextUnformatted("Game version:");
                for (int i = 0; i < _cascPickerProducts.Count; i++)
                {
                    CascProductInfo product = _cascPickerProducts[i];
                    bool installed = CascProductFolders.TryGetValue(product.Product, out string? folder)
                        && Directory.Exists(Path.Combine(_cascPickerInstallDir, folder));
                    string label = $"{product.Version}   {product.Product}{(installed ? $"   ({folder})" : "   (no game folder)")}###casc_product_{i}";
                    if (ImGuiNET.ImGui.RadioButton(label, _cascPickerSelected == i))
                        _cascPickerSelected = i;
                }

                ImGuiNET.ImGui.Spacing();
                ImGuiNET.ImGui.Checkbox("Fetch missing files from Blizzard's CDN (same build)", ref _cascPickerCdnFill);
                if (ImGuiNET.ImGui.IsItemHovered())
                    ImGuiNET.ImGui.SetTooltip("Reads local data first. Files the build lists but the install does not have on disk are downloaded once and cached.");

                ImGuiNET.ImGui.Checkbox("Fall back to the other products for missing files", ref _cascPickerFallbackToOtherProducts);
                if (ImGuiNET.ImGui.IsItemHovered())
                    ImGuiNET.ImGui.SetTooltip("Mixes data across game versions: a file missing from the selected build is read from another product instead. Off keeps the selected version pure.");
            }

            ImGuiNET.ImGui.Separator();
            bool canOpen = _cascPickerError is null && _cascPickerProducts.Count > 0;
            if (!canOpen)
                ImGuiNET.ImGui.BeginDisabled();
            if (ImGuiNET.ImGui.Button("Open", new System.Numerics.Vector2(120, 0)))
            {
                CascProductInfo selected = _cascPickerProducts[_cascPickerSelected];
                var products = new List<string> { selected.Product };
                if (_cascPickerFallbackToOtherProducts)
                    products.AddRange(_cascPickerProducts.Where(p => p != selected).Select(static p => p.Product));

                string installDir = _cascPickerInstallDir;
                _lastCascProduct = selected.Product;
                _cascPickerInstallDir = null;
                LoadCascDataSource(installDir, products, _cascPickerCdnFill);
            }
            if (!canOpen)
                ImGuiNET.ImGui.EndDisabled();

            ImGuiNET.ImGui.SameLine();
            if (ImGuiNET.ImGui.Button("Cancel", new System.Numerics.Vector2(120, 0)))
                open = false;
        }

        ImGuiNET.ImGui.End();
        if (!open)
            _cascPickerInstallDir = null;
    }

    /// <summary>
    /// Opens the given products of a CASC install, in order: the first is the game version whose
    /// build drives DB2 definitions and format profiles; later ones only fill files it lacks.
    /// </summary>
    private void LoadCascDataSource(string installDir, IReadOnlyList<string> productOrder, bool allowCdnFill)
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

            IReadOnlyList<CascProductInfo> listed = CascStorage.ListProducts(installDir);
            string cascCacheDir = Path.Combine(CacheDir, "casc");
            var storages = new List<CascStorage>();
            foreach (string productName in productOrder)
            {
                CascProductInfo? product = listed.FirstOrDefault(p => string.Equals(p.Product, productName, StringComparison.OrdinalIgnoreCase));
                if (product is null)
                    continue;

                try
                {
                    CascStorage storage = CascStorage.OpenLocal(installDir, product.Product, cascCacheDir, allowCdnFill);
                    if (storage.ManifestsFetchedFromCdn)
                        ViewerLog.Important(ViewerLog.Category.MpqData, $"CASC {product.Product} {product.Version}: local manifests missing for CDN config {product.CdnConfig}; fetched them from the CDN for this build.");
                    storages.Add(storage);
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
            InitializeMinimapSupport();

            // Spec 239: DB2 tables resolve through the listfile; definitions are picked by the newest
            // product's exact build (WoWDBDefs lists e.g. BUILD 1.60.1.69876 for wow_classic_beta).
            _dbdDir = ResolveDbdDefinitionsDir();
            _dbcBuild = _dbdDir is null ? null : storages[0].Product.Version;
            if (_dbcBuild is not null)
            {
                try
                {
                    _texResolver.LoadFromDBC(_dbcProvider, _dbdDir!, _dbcBuild);
                }
                catch (Exception ex)
                {
                    ViewerLog.Important(ViewerLog.Category.Dbc, $"CASC {_dbcBuild}: replaceable texture tables unavailable: {ex.Message}");
                }

                try
                {
                    _areaTableService = new AreaTableService();
                    _areaTableService.Load(_dbcProvider, _dbdDir!, _dbcBuild);
                }
                catch (Exception ex)
                {
                    _areaTableService = null;
                    ViewerLog.Important(ViewerLog.Category.Dbc, $"CASC {_dbcBuild}: AreaTable unavailable: {ex.Message}");
                }
            }

            try
            {
                RefreshDiscoveredMaps();
            }
            catch (Exception ex)
            {
                ViewerLog.Important(ViewerLog.Category.Dbc, $"CASC {_dbcBuild}: Map.db2 discovery failed ({ex.Message}); falling back to WDT file scan");
                _discoveredMaps = MapDiscoveryService.DiscoverLooseMapsOnly(_dataSource);
            }

            RefreshFileList();

            _statusMessage = $"Loaded: {_dataSource.Name} (listfile: {listfile.Count} entries, build {_dbcBuild ?? "unknown"}, {_discoveredMaps.Count} maps)";
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[ViewerApp] CASC load failed: {ex}");
            _statusMessage = $"Failed to open CASC install: {ex.Message}";
        }
    }

    /// <summary>
    /// Spec 237 (experimental): writes the ADT tiles around the camera as DAT v26 through the measured frames
    /// (<see cref="AdtAhdrTileBuilder"/>). A fresh adapter reads the tiles so the live scene's placement lists are
    /// untouched. Reopening the folder with the DAT loader should reproduce the same terrain and objects.
    /// </summary>
    private void ExportNearbyTilesAsDatV26(int radius)
    {
        if (_terrainManager?.Adapter is not StandardTerrainAdapter || _dataSource is null)
            return;

        string mapName = _terrainManager.MapName;
        byte[]? wdtBytes = _dataSource.ReadFile($"World\\Maps\\{mapName}\\{mapName}.wdt");
        if (wdtBytes is null)
        {
            _statusMessage = $"DAT v26 export: could not read the WDT for {mapName}.";
            return;
        }

        try
        {
            var adapter = new StandardTerrainAdapter(wdtBytes, mapName, _dataSource, _dbcBuild, _dbcProvider, _dbdDir);
            string outputDir = Path.Combine("output", "dat_v26_export", $"{mapName}_{DateTime.Now:yyyyMMdd_HHmmss}");
            Directory.CreateDirectory(outputDir);

            const float tileSpan = WoWConstants.ChunkSize;
            float cellSpan = tileSpan / 16f / 8f;
            var exportedUniqueIds = new HashSet<int>();
            int tiles = 0, objects = 0;
            for (int tileX = _terrainManager.CameraTileX - radius; tileX <= _terrainManager.CameraTileX + radius; tileX++)
            {
                for (int tileY = _terrainManager.CameraTileY - radius; tileY <= _terrainManager.CameraTileY + radius; tileY++)
                {
                    if (tileX is < 0 or > 63 || tileY is < 0 or > 63 || !adapter.TileExists(tileX, tileY))
                        continue;

                    TileLoadResult result = adapter.LoadTileWithPlacements(tileX, tileY);
                    if (result.Chunks.Count == 0)
                        continue;

                    var chunks = result.Chunks.Select(static chunk => new DatV26SourceChunk(
                        chunk.ChunkX,
                        chunk.ChunkY,
                        chunk.Heights,
                        // Renderer normal (X along -row, Y along -column, Z up) -> grid (column, vertical, row).
                        chunk.Normals.Length >= AdtAhdrTileSlicer.VerticesPerChunk
                            ? chunk.Normals.Select(static n => new System.Numerics.Vector3(-n.Y, n.Z, -n.X)).ToArray()
                            : null,
                        chunk.MccvColors,
                        chunk.Layers.Select((layer, index) => (layer.TextureIndex,
                            index > 0 && chunk.AlphaMaps.TryGetValue(index, out byte[]? alpha) && alpha.Length >= AdtAhdrAlpha.Pixels ? alpha : null)).ToArray()))
                        .ToList();

                    var placements = new List<DatV26SourcePlacement>();
                    void AddPlacement(int uniqueId, string name, System.Numerics.Vector3 position, System.Numerics.Vector3 rotation, float scale)
                    {
                        float row = (WoWConstants.MapOrigin - position.X - tileX * tileSpan) / cellSpan;
                        float column = (WoWConstants.MapOrigin - position.Y - tileY * tileSpan) / cellSpan;
                        if (row is < 0 or >= 128 || column is < 0 or >= 128 || !exportedUniqueIds.Add(uniqueId))
                            return;

                        // Renderer rotation is (raw0, raw2, raw1); ACDO keeps MDDF's file order (raw0, raw1, raw2).
                        placements.Add(new DatV26SourcePlacement(name, unchecked((uint)uniqueId), column, row, position.Z,
                            new System.Numerics.Vector3(rotation.X, rotation.Z, rotation.Y), scale));
                    }

                    foreach (MddfPlacement p in result.MddfPlacements)
                    {
                        if ((uint)p.NameIndex < (uint)adapter.MdxModelNames.Count)
                            AddPlacement(p.UniqueId, adapter.MdxModelNames[p.NameIndex], p.Position, p.Rotation, p.Scale);
                    }

                    foreach (ModfPlacement p in result.ModfPlacements)
                    {
                        if ((uint)p.NameIndex < (uint)adapter.WmoModelNames.Count)
                            AddPlacement(p.UniqueId, adapter.WmoModelNames[p.NameIndex], p.Position, p.Rotation, 1f);
                    }

                    List<string> textures = adapter.TileTextures.TryGetValue((tileX, tileY), out List<string>? names) ? names : [];
                    // ALOC X is the grid column axis = renderer tile Y; ALOC Y = renderer tile X (see AhdrTerrainAdapter).
                    AdtAhdrTile dat = AdtAhdrTileBuilder.Build(tileY, tileX, textures, chunks, placements, $"{mapName}_{tileY}_{tileX}");
                    File.WriteAllBytes(Path.Combine(outputDir, $"{mapName}_{tileY}_{tileX}.dat"), AdtAhdrWriter.Write(dat));
                    tiles++;
                    objects += placements.Count;
                }
            }

            _statusMessage = $"DAT v26 export: {tiles} tiles, {objects} objects -> {Path.GetFullPath(outputDir)}";
            ViewerLog.Important(ViewerLog.Category.Terrain, _statusMessage);
        }
        catch (Exception ex)
        {
            _statusMessage = $"DAT v26 export failed: {ex.Message}";
            ViewerLog.Important(ViewerLog.Category.Terrain, $"DAT v26 export failed: {ex}");
        }
    }

    /// <summary>
    /// Spec 247 US3: writes the loaded DAT folder as an LK v18 map. The picker is opened inline (it defers its
    /// own draw), so this needs no ViewerApp state field - AGENTS.md section 10. The conversion itself lives in
    /// DatToLkAdtFolderExporter, shared with the adt-ahdr export-lk command.
    /// </summary>
    private void ExportLoadedDatMap()
    {
        if (_lastAhdrTerrainFolder is not { } source)
        {
            _statusMessage = "Open a DAT terrain folder first (File > Open DAT Terrain Folder).";
            return;
        }

        string map = DatToLkAdtFolderExporter.SanitizeName(
            Path.GetFileName(Path.TrimEndingDirectorySeparator(source)));

        ImGuiPathPicker.Instance.Open(
            $"Choose where to write the {Terrain.MapExportFormats.Summary} export for '{map}'",
            pickFolder: true,
            initialPath: Path.GetFullPath(DatToLkAdtFolderExporter.DefaultOutputDirectory(map)) is { } d && Directory.Exists(d)
                ? d
                : GetProjectOutputRootDirectory(),
            filterExtension: null,
            chosen =>
            {
                if (string.IsNullOrWhiteSpace(chosen))
                    return;

                try
                {
                    DatFolderExportResult result = DatToLkAdtFolderExporter.Export(
                        source, Path.Combine(chosen, map), map, options: null,
                        targets: Terrain.MapExportFormats.Selected);

                    if (result.TilesWritten.Count == 0)
                    {
                        _statusMessage = $"DAT -> LK ADT: nothing written; no placeable AHDR-family tiles in {source}.";
                        ViewerLog.Important(ViewerLog.Category.Terrain, _statusMessage);
                        return;
                    }

                    _statusMessage = result.Summary;
                    ViewerLog.Important(ViewerLog.Category.Terrain, _statusMessage);
                    foreach (string reason in result.SkipReasons)
                        ViewerLog.Info(ViewerLog.Category.Terrain, $"[DAT -> LK] skipped {reason}");
                }
                catch (Exception ex)
                {
                    _statusMessage = $"DAT -> LK ADT export failed: {ex.Message}";
                    ViewerLog.Important(ViewerLog.Category.Terrain, $"DAT -> LK ADT export failed: {ex}");
                }
            });
    }

    private void LoadAhdrTerrain(string folder)
    {
        _lastAhdrTerrainFolder = Path.GetFullPath(folder);
        _statusMessage = $"Scanning DAT terrain files in {folder}...";

        AhdrTerrainAdapter adapter;
        try
        {
            adapter = new AhdrTerrainAdapter(folder, _ahdrHeightDivisor);
        }
        catch (Exception ex)
        {
            _statusMessage = $"DAT terrain scan failed: {ex.Message}";
            return;
        }

        foreach (string skipped in adapter.SkippedFiles)
            ViewerLog.Info(ViewerLog.Category.MpqData, $"[DAT] skipped {skipped}");

        if (adapter.ExistingTiles.Count == 0)
        {
            // Name the folder actually scanned and say whether anything AHDR-family was even seen:
            // the usual cause is a folder that is not the one the user meant.
            _statusMessage = adapter.SkippedFiles.Count == 0
                ? $"No AHDR-family DAT terrain files (v22/23/26) in {Path.GetFullPath(folder)}: {adapter.ScannedFileCount} file(s) scanned, none started with MVER+AHDR or AHDR (file extensions are ignored; content is sniffed)."
                : $"No placeable AHDR-family terrain files in {Path.GetFullPath(folder)}: {adapter.SkippedFiles.Count} of {adapter.ScannedFileCount} scanned file(s) skipped (see the log; first: {adapter.SkippedFiles[0]}).";
            ViewerLog.Important(ViewerLog.Category.Terrain, _statusMessage);
            return;
        }

        string mapName = "DAT: " + Path.GetFileName(Path.TrimEndingDirectorySeparator(folder));
        LoadTerrainFromAdapter(adapter, mapName,
            $"Type: DAT terrain folder, AHDR family\n" +
            $"Revisions: {adapter.VersionSummary} (ASHD is all-zero in the v26 corpus and not rendered)\n" +
            $"Folder: {folder}\n" +
            $"Tiles: {adapter.ExistingTiles.Count} (skipped {adapter.SkippedFiles.Count})\n" +
            $"Height divisor: {adapter.HeightDivisor:0.##} (File > DAT Terrain Height Scale)\n" +
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
