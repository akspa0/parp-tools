using System.Numerics;
using MdxViewer.DataSources;
using MdxViewer.Logging;
using MdxViewer.Population;
using MdxViewer.Rendering;
using Silk.NET.OpenGL;
using WoWMapConverter.Core.Services;

namespace MdxViewer.Terrain;

/// <summary>
/// Combines terrain (WDT/ADT), WMO placements (MODF), and MDX placements (MDDF)
/// into a single world scene — the same way the game client renders a map.
/// 
/// Uses <see cref="WorldAssetManager"/> to ensure each model is loaded exactly once.
/// Instances are lightweight structs holding only a model key + transform.
/// </summary>
public class WorldScene : ISceneRenderer
{
    private readonly GL _gl;
    private readonly TerrainManager _terrainManager;
    private readonly WorldAssetManager _assets;

    // Lightweight instance lists — just a key + transform, no renderer reference
    // These are rebuilt from _tileMdxInstances/_tileWmoInstances when tiles change
    private List<ObjectInstance> _mdxInstances = new();
    private List<ObjectInstance> _wmoInstances = new();

    // Per-tile instance storage for lazy load/unload
    private readonly Dictionary<(int, int), List<ObjectInstance>> _tileMdxInstances = new();
    private readonly Dictionary<(int, int), List<ObjectInstance>> _tileWmoInstances = new();
    private readonly Dictionary<(int, int), List<ObjectInstance>> _tileGroundMdxInstances = new();
    private readonly List<ObjectInstance> _externalMdxInstances = new();
    private readonly List<ObjectInstance> _externalWmoInstances = new();
    private bool _instancesDirty = false;

    private bool _objectsVisible = true;
    private bool _wmosVisible = true;
    private bool _doodadsVisible = true;

    // Ground effects (terrain layer EffectId -> ground effect doodads)
    private bool _groundEffectsVisible = false; // default disabled: extra MDX/M2 draw calls
    private readonly GroundEffectService _groundEffectService = new();
    private bool _groundEffectsLoadAttempted = false;
    public int GroundEffectInstanceCount { get; private set; } = 0;

    // Frustum culling
    private readonly FrustumCuller _frustumCuller = new();
    private const float DoodadCullDistance = 5000f; // Max distance for small doodads; raised to prevent long-range tree pop-out
    private const float DoodadCullDistanceSq = DoodadCullDistance * DoodadCullDistance;
    private const float DoodadSmallThreshold = 10f; // AABB diagonal below this = "small" (relaxed — only cull tiny objects)
    private const float FadeStartFraction = 0.80f;  // Fade begins at 80% of cull distance
    private const float WmoCullDistance = 2000f;     // Max distance for WMO instances (slightly past fog)
    private const float WmoFadeStartFraction = 0.85f;
    private const float NoCullRadius = 256f;         // Objects within this radius are never frustum-culled
    private const float NoCullRadiusSq = NoCullRadius * NoCullRadius;

    // Scratch collections reused every frame to avoid hot-path allocations.
    private readonly HashSet<string> _updatedMdxRenderers = new();
    private readonly List<(int idx, float distSq)> _transparentSortScratch = new();

    // Culling stats (updated each frame)
    public int WmoRenderedCount { get; private set; }
    public int WmoCulledCount { get; private set; }
    public int MdxRenderedCount { get; private set; }
    public int MdxCulledCount { get; private set; }

    // Stats
    public int MdxInstanceCount => _mdxInstances.Count;
    public int WmoInstanceCount => _wmoInstances.Count;
    public int UniqueMdxModels => _assets.MdxModelsLoaded;
    public int UniqueWmoModels => _assets.WmoModelsLoaded;
    public int ExternalSpawnMdxCount => _externalMdxInstances.Count;
    public int ExternalSpawnWmoCount => _externalWmoInstances.Count;
    public int ExternalSpawnInstanceCount => ExternalSpawnMdxCount + ExternalSpawnWmoCount;
    public float SqlGameObjectMdxScaleMultiplier { get; set; } = 1.0f;
    public TerrainManager Terrain => _terrainManager;
    public WorldAssetManager Assets => _assets;
    public bool IsWmoBased => _terrainManager.Adapter.IsWmoBased;

    public bool ShowGroundEffects
    {
        get => _groundEffectsVisible;
        set
        {
            if (_groundEffectsVisible == value) return;
            _groundEffectsVisible = value;

            if (!_groundEffectsVisible)
            {
                _tileGroundMdxInstances.Clear();
                GroundEffectInstanceCount = 0;
                _instancesDirty = true;
                return;
            }

            // Enabling: (re)build for currently loaded tiles
            BuildGroundEffectsForLoadedTiles();
        }
    }

    // Expose raw placement data for UI object list
    public IReadOnlyList<MddfPlacement> MddfPlacements => _terrainManager.Adapter.MddfPlacements;
    public IReadOnlyList<ModfPlacement> ModfPlacements => _terrainManager.Adapter.ModfPlacements;
    public IReadOnlyList<string> MdxModelNames => _terrainManager.Adapter.MdxModelNames;
    public IReadOnlyList<string> WmoModelNames => _terrainManager.Adapter.WmoModelNames;

    // Sky dome
    private readonly SkyDomeRenderer _skyDome;
    public SkyDomeRenderer SkyDome => _skyDome;

    // WDL low-res terrain (far terrain background)
    private WdlTerrainRenderer? _wdlTerrain;
    public WdlTerrainRenderer? WdlTerrain => _wdlTerrain;
    public bool ShowWdlTerrain { get; set; } = true;

    // Bounding box debug rendering
    private bool _showBoundingBoxes = false;
    private BoundingBoxRenderer? _bbRenderer;
    public bool ShowBoundingBoxes { get => _showBoundingBoxes; set => _showBoundingBoxes = value; }

    // Object selection
    private ObjectType _selectedObjectType = ObjectType.None;
    private int _selectedObjectIndex = -1;
    public ObjectType SelectedObjectType => _selectedObjectType;
    public int SelectedObjectIndex => _selectedObjectIndex;

    /// <summary>Get the currently selected object instance, or null if nothing selected.</summary>
    public ObjectInstance? SelectedInstance => _selectedObjectType switch
    {
        ObjectType.Wmo when _selectedObjectIndex >= 0 && _selectedObjectIndex < _wmoInstances.Count => _wmoInstances[_selectedObjectIndex],
        ObjectType.Mdx when _selectedObjectIndex >= 0 && _selectedObjectIndex < _mdxInstances.Count => _mdxInstances[_selectedObjectIndex],
        _ => null
    };

    // Area POI (lazy-loaded on first toggle)
    private AreaPoiLoader? _poiLoader;
    private bool _showPoi = false;
    private bool _poiLoadAttempted = false;
    public bool ShowPoi
    {
        get => _showPoi;
        set { _showPoi = value; if (value && !_poiLoadAttempted) LazyLoadPoi(); }
    }
    public AreaPoiLoader? PoiLoader => _poiLoader;
    public bool PoiLoadAttempted => _poiLoadAttempted;

    // Taxi paths (lazy-loaded on first toggle)
    private TaxiPathLoader? _taxiLoader;
    private bool _showTaxi = false;
    private bool _taxiLoadAttempted = false;
    public bool ShowTaxi
    {
        get => _showTaxi;
        set { _showTaxi = value; if (value && !_taxiLoadAttempted) LazyLoadTaxi(); }
    }
    public TaxiPathLoader? TaxiLoader => _taxiLoader;
    public bool TaxiLoadAttempted => _taxiLoadAttempted;

    // AreaTriggers (lazy-loaded on first toggle)
    private AreaTriggerLoader? _areaTriggerLoader;
    private bool _showAreaTriggers = false;
    private bool _areaTriggerLoadAttempted = false;
    public bool ShowAreaTriggers
    {
        get => _showAreaTriggers;
        set { _showAreaTriggers = value; if (value && !_areaTriggerLoadAttempted) LazyLoadAreaTriggers(); }
    }
    public AreaTriggerLoader? AreaTriggerLoader => _areaTriggerLoader;
    public bool AreaTriggerLoadAttempted => _areaTriggerLoadAttempted;

    // WL loose liquid files (auto-loaded on scene init)
    private WlLiquidLoader? _wlLoader;
    private bool _showWlLiquids = true; // Auto-enable by default
    private bool _wlLoadAttempted = false;
    private IDataSource? _dataSource;
    public bool ShowWlLiquids
    {
        get => _showWlLiquids;
        set
        {
            _showWlLiquids = value;
            if (value && !_wlLoadAttempted) LazyLoadWlLiquids();
            _terrainManager.LiquidRenderer.ShowWlLiquids = value;
        }
    }
    public WlLiquidLoader? WlLoader => _wlLoader;
    public bool WlLoadAttempted => _wlLoadAttempted;

    // Stored DBC credentials for lazy loading
    private DBCD.Providers.IDBCProvider? _dbcProvider;
    private string? _dbdDir;
    private string? _dbcBuild;
    private int _mapId = -1;

    // DBC Lighting
    private LightService? _lightService;
    public LightService? LightService => _lightService;

    // Taxi selection: -1 = show all (or none if !_showTaxi)
    private int _selectedTaxiNodeId = -1;
    private int _selectedTaxiRouteId = -1;
    public int SelectedTaxiNodeId { get => _selectedTaxiNodeId; set { _selectedTaxiNodeId = value; _selectedTaxiRouteId = -1; } }
    public int SelectedTaxiRouteId { get => _selectedTaxiRouteId; set { _selectedTaxiRouteId = value; _selectedTaxiNodeId = -1; } }
    public void ClearTaxiSelection() { _selectedTaxiNodeId = -1; _selectedTaxiRouteId = -1; }

    public bool IsTaxiRouteVisible(TaxiPathLoader.TaxiRoute route)
    {
        if (_selectedTaxiRouteId >= 0) return route.PathId == _selectedTaxiRouteId;
        if (_selectedTaxiNodeId >= 0) return route.FromNodeId == _selectedTaxiNodeId || route.ToNodeId == _selectedTaxiNodeId;
        return true; // no selection = show all
    }

    public bool IsTaxiNodeVisible(TaxiPathLoader.TaxiNode node)
    {
        if (_selectedTaxiNodeId >= 0) return node.Id == _selectedTaxiNodeId;
        if (_selectedTaxiRouteId >= 0)
        {
            var route = _taxiLoader?.Routes.FirstOrDefault(r => r.PathId == _selectedTaxiRouteId);
            return route != null && (route.FromNodeId == node.Id || route.ToNodeId == node.Id);
        }
        return true; // no selection = show all
    }

    /// <summary>
    /// Store DBC credentials for lazy loading of POI, Taxi, and Lighting.
    /// </summary>
    public void SetDbcCredentials(DBCD.Providers.IDBCProvider dbcProvider, string dbdDir, string build, int mapId)
    {
        _dbcProvider = dbcProvider;
        _dbdDir = dbdDir;
        _dbcBuild = build;
        _mapId = mapId;
    }

    private void LazyLoadWlLiquids()
    {
        _wlLoadAttempted = true;
        if (_dataSource == null) return;
        _wlLoader = new WlLiquidLoader(_dataSource, _terrainManager.MapName);
        _wlLoader.LoadAll();
        if (_wlLoader.HasData)
            _terrainManager.LiquidRenderer.AddWlBodies(_wlLoader.Bodies);
    }

    /// <summary>
    /// Reload WL loose liquid bodies (WLW/WLQ/WLM) and rebuild GPU meshes.
    /// Useful when tweaking WL transform settings in the UI.
    /// </summary>
    public void ReloadWlLiquids()
    {
        _terrainManager.LiquidRenderer.ClearWlBodies();
        _wlLoader = null;
        _wlLoadAttempted = false;
        LazyLoadWlLiquids();
    }

    private void LazyLoadPoi()
    {
        _poiLoadAttempted = true;
        if (_dbcProvider == null || _dbdDir == null || _dbcBuild == null) return;
        _poiLoader = new AreaPoiLoader();
        _poiLoader.Load(_dbcProvider, _dbdDir, _dbcBuild, _terrainManager.MapName);
    }

    private void LazyLoadTaxi()
    {
        _taxiLoadAttempted = true;
        if (_dbcProvider == null || _dbdDir == null || _dbcBuild == null || _mapId < 0) return;
        _taxiLoader = new TaxiPathLoader();
        var dbcd = new DBCD.DBCD(_dbcProvider, new DBCD.Providers.FilesystemDBDProvider(_dbdDir));
        _taxiLoader.Load(dbcd, _dbcBuild, _mapId);
    }

    private void LazyLoadAreaTriggers()
    {
        _areaTriggerLoadAttempted = true;
        if (_dbcProvider == null || _dbdDir == null || _dbcBuild == null || _mapId < 0) return;
        _areaTriggerLoader = new AreaTriggerLoader();
        _areaTriggerLoader.Load(_dbcProvider, _dbdDir, _dbcBuild, _mapId);
    }

    /// <summary>
    /// Load Light.dbc and LightData.dbc for zone-based lighting.
    /// </summary>
    public void LoadLighting(DBCD.Providers.IDBCProvider dbcProvider, string dbdDir, string build, int mapId)
    {
        _lightService = new LightService();
        _lightService.Load(dbcProvider, dbdDir, build, mapId);
    }

    public WorldScene(GL gl, string wdtPath, IDataSource? dataSource,
        ReplaceableTextureResolver? texResolver = null,
        Action<string>? onStatus = null)
    {
        _gl = gl;
        _dataSource = dataSource;
        _assets = new WorldAssetManager(gl, dataSource, texResolver);
        _bbRenderer = new BoundingBoxRenderer(gl);
        _skyDome = new SkyDomeRenderer(gl);

        // Create terrain manager (uses AOI-based lazy loading — tiles load as camera moves)
        onStatus?.Invoke("Loading WDT...");
        _terrainManager = new TerrainManager(gl, wdtPath, dataSource);

        InitFromAdapter(onStatus);
    }

    /// <summary>
    /// Create a WorldScene with a pre-built TerrainManager (for Standard WDT, etc.).
    /// </summary>
    public WorldScene(GL gl, TerrainManager terrainManager, IDataSource? dataSource,
        ReplaceableTextureResolver? texResolver = null,
        Action<string>? onStatus = null)
    {
        _gl = gl;
        _dataSource = dataSource;
        _assets = new WorldAssetManager(gl, dataSource, texResolver);
        _bbRenderer = new BoundingBoxRenderer(gl);
        _skyDome = new SkyDomeRenderer(gl);
        _terrainManager = terrainManager;

        InitFromAdapter(onStatus);
    }

    private void InitFromAdapter(Action<string>? onStatus)
    {
        var adapter = _terrainManager.Adapter;

        if (adapter.IsWmoBased && adapter.ModfPlacements.Count > 0)
        {
            // WMO-only maps: pre-load placements + models
            var manifest = _assets.BuildManifest(
                adapter.MdxModelNames, adapter.WmoModelNames,
                adapter.MddfPlacements, adapter.ModfPlacements);
            _assets.LoadManifest(manifest);
            BuildInstances(adapter);

            var p = adapter.ModfPlacements[0];
            var bbCenter = (p.BoundsMin + p.BoundsMax) * 0.5f;
            var bbExtent = p.BoundsMax - p.BoundsMin;
            float dist = MathF.Max(bbExtent.Length() * 0.5f, 100f);
            _wmoCameraOverride = bbCenter + new Vector3(dist, 0, bbExtent.Z * 0.3f);
            ViewerLog.Info(ViewerLog.Category.Terrain, $"WMO-only map, camera at BB center: ({bbCenter.X:F1}, {bbCenter.Y:F1}, {bbCenter.Z:F1}), dist={dist:F0}");

            // Still subscribe for any late-loaded tiles
            _terrainManager.OnTileLoaded += OnTileLoaded;
            _terrainManager.OnTileUnloaded += OnTileUnloaded;
            onStatus?.Invoke("World loaded (WMO-only map).");
        }
        else
        {
            // Terrain maps: load WDL low-res mesh first for instant overview,
            // then stream detailed ADT tiles via AOI as the camera moves.
            if (_dataSource != null)
            {
                onStatus?.Invoke("Loading WDL terrain...");
                _wdlTerrain = new WdlTerrainRenderer(_gl);
                if (!_wdlTerrain.Load(_dataSource, _terrainManager.MapName))
                {
                    _wdlTerrain.Dispose();
                    _wdlTerrain = null;
                }
            }

            _terrainManager.OnTileLoaded += OnTileLoaded;
            _terrainManager.OnTileUnloaded += OnTileUnloaded;

            // Hide WDL for all ADT-backed tiles in the map so WDL only fills gaps
            // where no detailed ADT tile exists (developer/empty tiles).
            if (_wdlTerrain != null)
            {
                foreach (int tileIdx in adapter.ExistingTiles)
                {
                    int tx = tileIdx / 64;
                    int ty = tileIdx % 64;
                    _wdlTerrain.HideTile(tx, ty);
                }
            }
            onStatus?.Invoke("World loaded (tiles stream as you move).");
        }
        
        // Auto-load WL liquids if enabled
        if (_showWlLiquids && !_wlLoadAttempted)
        {
            LazyLoadWlLiquids();
        }
    }

    private Vector3? _wmoCameraOverride;
    /// <summary>For WMO-only maps, returns the WMO position as camera start. Otherwise null.</summary>
    public Vector3? WmoCameraOverride => _wmoCameraOverride;

    private void BuildInstances(ITerrainAdapter adapter)
    {
        var mdxNames = adapter.MdxModelNames;
        var wmoNames = adapter.WmoModelNames;

        // Placement transform for terrain maps.
        // Positions are already converted to renderer coords in AlphaTerrainAdapter:
        //   rendererX = MapOrigin - wowY, rendererY = MapOrigin - wowX, rendererZ = wowZ
        // Triangle winding is reversed at upload (CW→CCW for OpenGL), which flips the
        // model's facing direction by 180°. Compensate with a 180° Z rotation.
        var rot180Z = Matrix4x4.CreateRotationZ(MathF.PI);
        bool wmoBased = adapter.IsWmoBased;

        // MDX (doodad) placements — same rotation as WMO (wiki confirms "same as MODF"),
        // with scale added. Rotation stored as degrees in file.
        foreach (var p in adapter.MddfPlacements)
        {
            if (p.NameIndex < 0 || p.NameIndex >= mdxNames.Count) continue;

            string key = WorldAssetManager.NormalizeKey(mdxNames[p.NameIndex]);
            float scale = p.Scale > 0 ? p.Scale : 1.0f;
            // Rotation stored as degrees in WoW coords (X=North, Y=West, Z=Up).
            // Position axes are swapped: wowX→rendererY, wowY→rendererX (both negated).
            // Rotation axes must follow the same swap:
            //   WoW rotX (tilt around North) → renderer RotationY (negated)
            //   WoW rotY (tilt around West)  → renderer RotationX (negated)
            //   WoW rotZ (heading around Up)  → renderer RotationZ (as-is)
            float rx = -p.Rotation.Y * MathF.PI / 180f;
            float ry = -p.Rotation.X * MathF.PI / 180f;
            float rz = p.Rotation.Z * MathF.PI / 180f;

            var transform = rot180Z
                * Matrix4x4.CreateScale(scale)
                * Matrix4x4.CreateRotationX(rx)
                * Matrix4x4.CreateRotationY(ry)
                * Matrix4x4.CreateRotationZ(rz)
                * Matrix4x4.CreateTranslation(p.Position);

            // Use actual model bounds if available, transformed to world space
            Vector3 bbMin, bbMax;
            if (_assets.TryGetMdxBounds(key, out var modelMin, out var modelMax))
            {
                TransformBounds(modelMin, modelMax, transform, out bbMin, out bbMax);
            }
            else
            {
                bbMin = p.Position - new Vector3(2f);
                bbMax = p.Position + new Vector3(2f);
            }
            string modelPath = mdxNames[p.NameIndex];
            _mdxInstances.Add(new ObjectInstance
            {
                ModelKey = key,
                Transform = transform,
                BoundsMin = bbMin,
                BoundsMax = bbMax,
                ModelName = Path.GetFileName(modelPath),
                ModelPath = modelPath,
                PlacementPosition = p.Position,
                PlacementRotation = p.Rotation,
                PlacementScale = scale,
                UniqueId = p.UniqueId
            });
        }

        // WMO placements
        foreach (var p in adapter.ModfPlacements)
        {
            if (p.NameIndex < 0 || p.NameIndex >= wmoNames.Count) continue;

            string key = WorldAssetManager.NormalizeKey(wmoNames[p.NameIndex]);
            float rx = p.Rotation.X * MathF.PI / 180f;
            float ry = p.Rotation.Y * MathF.PI / 180f;
            float rz = p.Rotation.Z * MathF.PI / 180f;

            var transform = rot180Z
                * Matrix4x4.CreateRotationX(rx)
                * Matrix4x4.CreateRotationY(ry)
                * Matrix4x4.CreateRotationZ(rz)
                * Matrix4x4.CreateTranslation(p.Position);

            // Get MOHD local bounds from the loaded WMO model and transform to world space.
            // The WMO is a container — its internal geometry has its own local bounding box
            // (MOHD bounds) around the WMO's local origin. We transform that through the
            // placement matrix to get the correct world-space AABB.
            // Falls back to MODF file bounds if model isn't loaded.
            Vector3 localMin, localMax, worldMin, worldMax;
            if (_assets.TryGetWmoBounds(key, out localMin, out localMax))
            {
                TransformBounds(localMin, localMax, transform, out worldMin, out worldMax);
            }
            else
            {
                localMin = localMax = Vector3.Zero;
                worldMin = p.BoundsMin;
                worldMax = p.BoundsMax;
            }

            string wmoPath = wmoNames[p.NameIndex];
            _wmoInstances.Add(new ObjectInstance
            {
                ModelKey = key,
                Transform = transform,
                BoundsMin = worldMin,
                BoundsMax = worldMax,
                LocalBoundsMin = localMin,
                LocalBoundsMax = localMax,
                ModelName = Path.GetFileName(wmoPath),
                ModelPath = wmoPath,
                PlacementPosition = p.Position,
                PlacementRotation = p.Rotation,
                PlacementScale = 1.0f,
                UniqueId = p.UniqueId
            });
        }

        ViewerLog.Important(ViewerLog.Category.Terrain, $"Instances: {_mdxInstances.Count} MDX, {_wmoInstances.Count} WMO");
        ViewerLog.Important(ViewerLog.Category.Terrain, $"Unique models: {UniqueMdxModels} MDX, {UniqueWmoModels} WMO");

        // Diagnostic: terrain chunk WorldPosition range
        var camPos = _terrainManager.GetInitialCameraPosition();
        ViewerLog.Info(ViewerLog.Category.Terrain, $"Camera: ({camPos.X:F1}, {camPos.Y:F1}, {camPos.Z:F1})");

        // Compute terrain bounding box from chunk WorldPositions
        float tMinX = float.MaxValue, tMinY = float.MaxValue, tMinZ = float.MaxValue;
        float tMaxX = float.MinValue, tMaxY = float.MinValue, tMaxZ = float.MinValue;
        foreach (var chunk in _terrainManager.Adapter.LastLoadedChunkPositions)
        {
            tMinX = Math.Min(tMinX, chunk.X); tMaxX = Math.Max(tMaxX, chunk.X);
            tMinY = Math.Min(tMinY, chunk.Y); tMaxY = Math.Max(tMaxY, chunk.Y);
            tMinZ = Math.Min(tMinZ, chunk.Z); tMaxZ = Math.Max(tMaxZ, chunk.Z);
        }
        ViewerLog.Info(ViewerLog.Category.Terrain, $"TERRAIN  X:[{tMinX:F1} .. {tMaxX:F1}]  Y:[{tMinY:F1} .. {tMaxY:F1}]  Z:[{tMinZ:F1} .. {tMaxZ:F1}]");

        // Compute object bounding box (from stored positions, which are already transformed)
        float oMinX = float.MaxValue, oMinY = float.MaxValue, oMinZ = float.MaxValue;
        float oMaxX = float.MinValue, oMaxY = float.MinValue, oMaxZ = float.MinValue;
        foreach (var p in adapter.MddfPlacements)
        {
            oMinX = Math.Min(oMinX, p.Position.X); oMaxX = Math.Max(oMaxX, p.Position.X);
            oMinY = Math.Min(oMinY, p.Position.Y); oMaxY = Math.Max(oMaxY, p.Position.Y);
            oMinZ = Math.Min(oMinZ, p.Position.Z); oMaxZ = Math.Max(oMaxZ, p.Position.Z);
        }
        foreach (var p in adapter.ModfPlacements)
        {
            oMinX = Math.Min(oMinX, p.Position.X); oMaxX = Math.Max(oMaxX, p.Position.X);
            oMinY = Math.Min(oMinY, p.Position.Y); oMaxY = Math.Max(oMaxY, p.Position.Y);
            oMinZ = Math.Min(oMinZ, p.Position.Z); oMaxZ = Math.Max(oMaxZ, p.Position.Z);
        }
        ViewerLog.Info(ViewerLog.Category.Terrain, $"OBJECTS  X:[{oMinX:F1} .. {oMaxX:F1}]  Y:[{oMinY:F1} .. {oMaxY:F1}]  Z:[{oMinZ:F1} .. {oMaxZ:F1}]");
        ViewerLog.Info(ViewerLog.Category.Terrain, $"DELTA    X:{(tMinX+tMaxX)/2 - (oMinX+oMaxX)/2:F1}  Y:{(tMinY+tMaxY)/2 - (oMinY+oMaxY)/2:F1}  Z:{(tMinZ+tMaxZ)/2 - (oMinZ+oMaxZ)/2:F1}");

        // Print first 3 MDDF raw values for manual inspection
        for (int i = 0; i < Math.Min(3, adapter.MddfPlacements.Count); i++)
        {
            var p = adapter.MddfPlacements[i];
            string name = p.NameIndex < mdxNames.Count ? Path.GetFileName(mdxNames[p.NameIndex]) : "?";
            ViewerLog.Debug(ViewerLog.Category.Terrain, $"  MDDF[{i}] pos=({p.Position.X:F1}, {p.Position.Y:F1}, {p.Position.Z:F1}) model={name}");
        }
        for (int i = 0; i < Math.Min(3, adapter.ModfPlacements.Count); i++)
        {
            var p = adapter.ModfPlacements[i];
            string name = p.NameIndex < wmoNames.Count ? Path.GetFileName(wmoNames[p.NameIndex]) : "?";
            ViewerLog.Debug(ViewerLog.Category.Terrain, $"  MODF[{i}] pos=({p.Position.X:F1}, {p.Position.Y:F1}, {p.Position.Z:F1}) model={name}");
        }
    }

    /// <summary>
    /// Called by TerrainManager when a new tile enters the AOI.
    /// Builds object instances for the tile and lazy-loads any new models.
    /// </summary>
    private void OnTileLoaded(int tileX, int tileY, TileLoadResult result)
    {
        var adapter = _terrainManager.Adapter;
        var mdxNames = adapter.MdxModelNames;
        var wmoNames = adapter.WmoModelNames;

        // Build MDX instances for this tile
        var tileMdx = new List<ObjectInstance>();
        foreach (var p in result.MddfPlacements)
        {
            if (p.NameIndex < 0 || p.NameIndex >= mdxNames.Count) continue;
            string key = WorldAssetManager.NormalizeKey(mdxNames[p.NameIndex]);
            _assets.EnsureMdxLoaded(key);
            float scale = p.Scale > 0 ? p.Scale : 1.0f;

            // Rotation stored as degrees in WoW coords — axes swapped to match position swap.
            float rx = -p.Rotation.Y * MathF.PI / 180f;
            float ry = -p.Rotation.X * MathF.PI / 180f;
            float rz = p.Rotation.Z * MathF.PI / 180f;

            // 180° Z rotation compensates for winding reversal (CW→CCW)
            var rot180Z = Matrix4x4.CreateRotationZ(MathF.PI);
            var transform = rot180Z
                * Matrix4x4.CreateScale(scale)
                * Matrix4x4.CreateRotationX(rx)
                * Matrix4x4.CreateRotationY(ry)
                * Matrix4x4.CreateRotationZ(rz)
                * Matrix4x4.CreateTranslation(p.Position);
            Vector3 bbMin, bbMax;
            if (_assets.TryGetMdxBounds(key, out var modelMin, out var modelMax))
                TransformBounds(modelMin, modelMax, transform, out bbMin, out bbMax);
            else
            { bbMin = p.Position - new Vector3(2f); bbMax = p.Position + new Vector3(2f); }
            string modelPath = mdxNames[p.NameIndex];
            tileMdx.Add(new ObjectInstance
            {
                ModelKey = key, Transform = transform, BoundsMin = bbMin, BoundsMax = bbMax,
                ModelName = Path.GetFileName(modelPath), ModelPath = modelPath,
                PlacementPosition = p.Position, PlacementRotation = p.Rotation, PlacementScale = scale,
                UniqueId = p.UniqueId
            });
        }

        // Build WMO instances for this tile
        var tileWmo = new List<ObjectInstance>();
        foreach (var p in result.ModfPlacements)
        {
            if (p.NameIndex < 0 || p.NameIndex >= wmoNames.Count) continue;
            string key = WorldAssetManager.NormalizeKey(wmoNames[p.NameIndex]);
            _assets.EnsureWmoLoaded(key);
            float rx = p.Rotation.X * MathF.PI / 180f;
            float ry = p.Rotation.Y * MathF.PI / 180f;
            float rz = p.Rotation.Z * MathF.PI / 180f;

            // 180° Z rotation compensates for winding reversal (CW→CCW)
            var rot180Z = Matrix4x4.CreateRotationZ(MathF.PI);
            var transform = rot180Z
                * Matrix4x4.CreateRotationX(rx)
                * Matrix4x4.CreateRotationY(ry)
                * Matrix4x4.CreateRotationZ(rz)
                * Matrix4x4.CreateTranslation(p.Position);

            // Get MOHD local bounds and transform to world space
            Vector3 localMin, localMax, worldMin, worldMax;
            if (_assets.TryGetWmoBounds(key, out localMin, out localMax))
            {
                TransformBounds(localMin, localMax, transform, out worldMin, out worldMax);
            }
            else
            {
                localMin = localMax = Vector3.Zero;
                worldMin = p.BoundsMin;
                worldMax = p.BoundsMax;
            }

            string wmoPath = wmoNames[p.NameIndex];
            tileWmo.Add(new ObjectInstance
            {
                ModelKey = key,
                Transform = transform,
                BoundsMin = worldMin,
                BoundsMax = worldMax,
                LocalBoundsMin = localMin,
                LocalBoundsMax = localMax,
                ModelName = Path.GetFileName(wmoPath), ModelPath = wmoPath,
                PlacementPosition = p.Position, PlacementRotation = p.Rotation, PlacementScale = 1.0f,
                UniqueId = p.UniqueId
            });
        }

        _tileMdxInstances[(tileX, tileY)] = tileMdx;
        _tileWmoInstances[(tileX, tileY)] = tileWmo;

        if (_groundEffectsVisible)
        {
            var tileGround = BuildGroundEffectInstancesForTile(tileX, tileY, result.Chunks);
            _tileGroundMdxInstances[(tileX, tileY)] = tileGround;
            GroundEffectInstanceCount += tileGround.Count;
        }

        _instancesDirty = true;

        // Hide WDL low-res tile now that detailed ADT is loaded
        _wdlTerrain?.HideTile(tileX, tileY);

        if (tileMdx.Count > 0 || tileWmo.Count > 0)
            ViewerLog.Info(ViewerLog.Category.Terrain, $"Tile ({tileX},{tileY}) loaded: {tileMdx.Count} MDX, {tileWmo.Count} WMO instances");
    }

    /// <summary>
    /// Called by TerrainManager when a tile leaves the AOI.
    /// </summary>
    private void OnTileUnloaded(int tileX, int tileY)
    {
        _tileMdxInstances.Remove((tileX, tileY));
        _tileWmoInstances.Remove((tileX, tileY));

        if (_tileGroundMdxInstances.Remove((tileX, tileY), out var removedGround))
        {
            GroundEffectInstanceCount = Math.Max(0, GroundEffectInstanceCount - removedGround.Count);
        }

        _instancesDirty = true;
    }

    /// <summary>
    /// Rebuild flat instance lists from per-tile dictionaries.
    /// Called lazily before rendering when _instancesDirty is true.
    /// </summary>
    private void RebuildInstanceLists()
    {
        _mdxInstances.Clear();
        foreach (var list in _tileMdxInstances.Values)
            _mdxInstances.AddRange(list);

        if (_groundEffectsVisible)
        {
            foreach (var list in _tileGroundMdxInstances.Values)
                _mdxInstances.AddRange(list);
        }

        _mdxInstances.AddRange(_externalMdxInstances);

        _wmoInstances.Clear();
        foreach (var list in _tileWmoInstances.Values)
            _wmoInstances.AddRange(list);
        _wmoInstances.AddRange(_externalWmoInstances);

        _instancesDirty = false;
    }

    private void BuildGroundEffectsForLoadedTiles()
    {
        _tileGroundMdxInstances.Clear();
        GroundEffectInstanceCount = 0;

        foreach (var (tileX, tileY) in _terrainManager.LoadedTiles)
        {
            if (!_terrainManager.TryGetTileLoadResult(tileX, tileY, out var tile))
                continue;

            var list = BuildGroundEffectInstancesForTile(tileX, tileY, tile.Chunks);
            if (list.Count == 0) continue;
            _tileGroundMdxInstances[(tileX, tileY)] = list;
            GroundEffectInstanceCount += list.Count;
        }

        _instancesDirty = true;
    }

    private bool EnsureGroundEffectDbcLoaded()
    {
        if (_groundEffectsLoadAttempted) return true;
        _groundEffectsLoadAttempted = true;

        try
        {
            if (_dataSource is MpqDataSource mpqDs)
            {
                // MPQ-backed load: prefers DBFilesClient\*.dbc, falls back internally
                _groundEffectService.Load(Array.Empty<string>(), mpqDs.MpqService);
            }
            else
            {
                // Disk-only: only works if caller has placed DBCs in a known folder.
                // Keep minimal: try current directory.
                _groundEffectService.Load(new[] { Environment.CurrentDirectory });
            }

            return true;
        }
        catch (Exception ex)
        {
            ViewerLog.Important(ViewerLog.Category.Terrain, $"GroundEffects load failed: {ex.Message}");
            return false;
        }
    }

    private List<ObjectInstance> BuildGroundEffectInstancesForTile(int tileX, int tileY, IReadOnlyList<TerrainChunkData> chunks)
    {
        var instances = new List<ObjectInstance>();
        if (chunks.Count == 0) return instances;
        if (!EnsureGroundEffectDbcLoaded()) return instances;

        // Conservative density: keep draw calls sane even when enabled
        const int baseSpawnsPerChunkLayer = 2;
        const int maxSpawnsPerChunkLayer = 6;
        const float alphaThreshold = 0.50f;

        foreach (var chunk in chunks)
        {
            if (chunk.Layers == null || chunk.Layers.Length == 0) continue;

            for (int layerIndex = 0; layerIndex < chunk.Layers.Length; layerIndex++)
            {
                uint effectId = chunk.Layers[layerIndex].EffectId;
                if (effectId == 0) continue;

                var models = _groundEffectService.GetDoodadsEffect(effectId);
                if (models == null || models.Length == 0) continue;

                byte[]? alpha = null;
                if (layerIndex > 0)
                    chunk.AlphaMaps.TryGetValue(layerIndex, out alpha);

                float coverage = EstimateLayerCoverage(alpha);
                if (layerIndex > 0 && coverage <= 0.01f) continue;

                int spawnCount = (int)MathF.Round(baseSpawnsPerChunkLayer * MathF.Max(coverage, 0.25f));
                spawnCount = Math.Clamp(spawnCount, 1, maxSpawnsPerChunkLayer);

                int seed = HashCode.Combine(tileX, tileY, chunk.ChunkX, chunk.ChunkY, (int)effectId, layerIndex);
                var rng = new Random(seed);

                for (int s = 0; s < spawnCount; s++)
                {
                    if (!TryPickSpawnPoint(rng, chunk, alpha, alphaThreshold, out float localX, out float localY, out float z))
                        continue;

                    string modelPath = models[rng.Next(models.Length)];
                    string key = WorldAssetManager.NormalizeKey(modelPath);
                    _assets.EnsureMdxLoaded(key);

                    float yaw = (float)(rng.NextDouble() * MathF.Tau);
                    float scale = 0.8f + (float)rng.NextDouble() * 0.4f;
                    var worldPos = new Vector3(chunk.WorldPosition.X - localY, chunk.WorldPosition.Y - localX, z);

                    var transform = Matrix4x4.CreateScale(scale)
                        * Matrix4x4.CreateRotationZ(yaw)
                        * Matrix4x4.CreateTranslation(worldPos);

                    Vector3 bbMin, bbMax;
                    if (_assets.TryGetMdxBounds(key, out var modelMin, out var modelMax))
                        TransformBounds(modelMin, modelMax, transform, out bbMin, out bbMax);
                    else
                    {
                        bbMin = worldPos - new Vector3(0.5f);
                        bbMax = worldPos + new Vector3(0.5f);
                    }

                    instances.Add(new ObjectInstance
                    {
                        ModelKey = key,
                        Transform = transform,
                        BoundsMin = bbMin,
                        BoundsMax = bbMax,
                        ModelName = Path.GetFileName(modelPath),
                        ModelPath = modelPath,
                        PlacementPosition = worldPos,
                        PlacementRotation = new Vector3(0f, 0f, yaw * (180f / MathF.PI)),
                        PlacementScale = scale,
                        UniqueId = unchecked((int)0x8000_0000u) + (seed & 0x7FFF_FFFF)
                    });
                }
            }
        }

        return instances;
    }

    private static float EstimateLayerCoverage(byte[]? alpha)
    {
        if (alpha == null || alpha.Length < 64 * 64) return 1.0f;
        // Sample a small grid to estimate coverage cheaply
        int step = 8;
        float sum = 0;
        int count = 0;
        for (int y = 0; y < 64; y += step)
        {
            int row = y * 64;
            for (int x = 0; x < 64; x += step)
            {
                sum += alpha[row + x] / 255f;
                count++;
            }
        }
        return count > 0 ? (sum / count) : 1.0f;
    }

    private static bool TryPickSpawnPoint(Random rng, TerrainChunkData chunk, byte[]? alpha, float alphaThreshold,
        out float localX, out float localY, out float z)
    {
        const int attempts = 12;
        float chunkSize = WoWConstants.ChunkSize;
        for (int a = 0; a < attempts; a++)
        {
            localX = (float)rng.NextDouble() * (chunkSize - 1e-3f);
            localY = (float)rng.NextDouble() * (chunkSize - 1e-3f);

            if (alpha != null && alpha.Length >= 64 * 64)
            {
                int ax = Math.Clamp((int)(localX / chunkSize * 63f), 0, 63);
                int ay = Math.Clamp((int)(localY / chunkSize * 63f), 0, 63);
                float av = alpha[ay * 64 + ax] / 255f;
                if (av < alphaThreshold) continue;
            }

            z = SampleHeightOuterGrid(chunk, localX, localY) + 0.02f;
            return true;
        }

        localX = localY = z = 0;
        return false;
    }

    private static float SampleHeightOuterGrid(TerrainChunkData chunk, float localX, float localY)
    {
        // Approximate terrain height from the 9x9 outer vertex grid.
        // This is intentionally cheap; it avoids depending on inner vertices.
        if (chunk.Heights == null || chunk.Heights.Length < 145) return chunk.WorldPosition.Z;

        float cellSize = WoWConstants.ChunkSize / 16f;
        float subCellSize = cellSize / 8f;

        Span<float> grid = stackalloc float[9 * 9];
        grid.Clear();

        for (int i = 0; i < 145; i++)
        {
            GetChunkVertexPosition(i, out int row, out int col, out bool isInner);
            if (isInner) continue;
            int gy = row / 2;
            if ((uint)gy >= 9u || (uint)col >= 9u) continue;
            grid[gy * 9 + col] = chunk.Heights[i];
        }

        float gx = localX / subCellSize;
        float gyf = localY / subCellSize;
        int ix = Math.Clamp((int)MathF.Floor(gx), 0, 7);
        int iy = Math.Clamp((int)MathF.Floor(gyf), 0, 7);
        float fx = Math.Clamp(gx - ix, 0f, 1f);
        float fy = Math.Clamp(gyf - iy, 0f, 1f);

        float h00 = grid[iy * 9 + ix];
        float h10 = grid[iy * 9 + (ix + 1)];
        float h01 = grid[(iy + 1) * 9 + ix];
        float h11 = grid[(iy + 1) * 9 + (ix + 1)];

        float h0 = h00 + (h10 - h00) * fx;
        float h1 = h01 + (h11 - h01) * fx;
        return h0 + (h1 - h0) * fy;
    }

    private static void GetChunkVertexPosition(int index, out int row, out int col, out bool isInner)
    {
        int remaining = index;
        row = 0;
        col = 0;
        isInner = false;

        for (int r = 0; r < 17; r++)
        {
            int rowSize = (r % 2 == 0) ? 9 : 8;
            if (remaining < rowSize)
            {
                row = r;
                col = remaining;
                isInner = (r % 2 != 0);
                return;
            }
            remaining -= rowSize;
        }
    }

    public void ClearExternalSpawns()
    {
        _externalMdxInstances.Clear();
        _externalWmoInstances.Clear();
        _instancesDirty = true;
    }

    public void SetExternalSpawns(IEnumerable<WorldSpawnRecord> spawns)
    {
        _externalMdxInstances.Clear();
        _externalWmoInstances.Clear();

        foreach (var spawn in spawns)
        {
            if (string.IsNullOrWhiteSpace(spawn.ModelPath))
                continue;

            string modelPath = spawn.ModelPath.Replace('/', '\\');
            bool isWmo = modelPath.EndsWith(".wmo", StringComparison.OrdinalIgnoreCase);

            string key = WorldAssetManager.NormalizeKey(modelPath);
            float orientationRadians = spawn.OrientationWowRadians;
            float yawOffsetRadians = spawn.SpawnType == WorldSpawnType.Creature ? MathF.PI : 0f;
            float finalYawRadians = orientationRadians + yawOffsetRadians;
            float finalYawDegrees = finalYawRadians * (180f / MathF.PI);
            float baseScale = spawn.EffectiveScale > 0 ? spawn.EffectiveScale : 1.0f;
            float mdxScale = baseScale;
            if (spawn.SpawnType == WorldSpawnType.GameObject)
                mdxScale *= SqlGameObjectMdxScaleMultiplier > 0 ? SqlGameObjectMdxScaleMultiplier : 1.0f;

            var pos = SqlSpawnCoordinateConverter.ToRendererPosition(spawn.PositionWow);

            if (isWmo)
            {
                _assets.EnsureWmoLoaded(key);

                var transform = Matrix4x4.CreateRotationZ(finalYawRadians)
                    * Matrix4x4.CreateTranslation(pos);

                Vector3 localMin, localMax, worldMin, worldMax;
                if (_assets.TryGetWmoBounds(key, out localMin, out localMax))
                {
                    TransformBounds(localMin, localMax, transform, out worldMin, out worldMax);
                }
                else
                {
                    localMin = localMax = Vector3.Zero;
                    worldMin = pos - new Vector3(2f);
                    worldMax = pos + new Vector3(2f);
                }

                _externalWmoInstances.Add(new ObjectInstance
                {
                    ModelKey = key,
                    Transform = transform,
                    BoundsMin = worldMin,
                    BoundsMax = worldMax,
                    LocalBoundsMin = localMin,
                    LocalBoundsMax = localMax,
                    ModelName = Path.GetFileName(modelPath),
                    ModelPath = modelPath,
                    PlacementPosition = pos,
                    PlacementRotation = new Vector3(0f, 0f, finalYawDegrees),
                    PlacementScale = 1.0f,
                    UniqueId = spawn.SpawnId
                });
            }
            else
            {
                _assets.EnsureMdxLoaded(key);

                var transform = Matrix4x4.CreateScale(mdxScale)
                    * Matrix4x4.CreateRotationZ(finalYawRadians)
                    * Matrix4x4.CreateTranslation(pos);

                Vector3 bbMin, bbMax;
                if (_assets.TryGetMdxBounds(key, out var modelMin, out var modelMax))
                    TransformBounds(modelMin, modelMax, transform, out bbMin, out bbMax);
                else
                {
                    bbMin = pos - new Vector3(2f);
                    bbMax = pos + new Vector3(2f);
                }

                _externalMdxInstances.Add(new ObjectInstance
                {
                    ModelKey = key,
                    Transform = transform,
                    BoundsMin = bbMin,
                    BoundsMax = bbMax,
                    ModelName = Path.GetFileName(modelPath),
                    ModelPath = modelPath,
                    PlacementPosition = pos,
                    PlacementRotation = new Vector3(0f, 0f, finalYawDegrees),
                    PlacementScale = mdxScale,
                    UniqueId = spawn.SpawnId
                });
            }
        }

        ViewerLog.Info(ViewerLog.Category.Terrain,
            $"SQL spawns injected: {_externalMdxInstances.Count} MDX, {_externalWmoInstances.Count} WMO");

        _instancesDirty = true;
    }

    /// <summary>
    /// Transform an axis-aligned bounding box through a matrix by transforming all 8 corners
    /// and computing the new AABB that encloses them.
    /// </summary>
    private static void TransformBounds(Vector3 min, Vector3 max, Matrix4x4 m, out Vector3 outMin, out Vector3 outMax)
    {
        outMin = new Vector3(float.MaxValue);
        outMax = new Vector3(float.MinValue);
        Span<float> xs = stackalloc float[] { min.X, max.X };
        Span<float> ys = stackalloc float[] { min.Y, max.Y };
        Span<float> zs = stackalloc float[] { min.Z, max.Z };
        foreach (var x in xs)
        foreach (var y in ys)
        foreach (var z in zs)
        {
            var p = Vector3.Transform(new Vector3(x, y, z), m);
            outMin = Vector3.Min(outMin, p);
            outMax = Vector3.Max(outMax, p);
        }
    }

    // ── ISceneRenderer ──────────────────────────────────────────────────

    private bool _renderDiagPrinted = false;
    public void Render(Matrix4x4 view, Matrix4x4 proj)
    {
        // Rebuild flat instance lists if tiles changed
        if (_instancesDirty)
            RebuildInstanceLists();

        // Extract camera position for sky dome
        Matrix4x4.Invert(view, out var viewInvSky);
        var camPos = new Vector3(viewInvSky.M41, viewInvSky.M42, viewInvSky.M43);

        // 0. Render sky dome (before terrain, no depth write)
        // Update DBC lighting early so sky colors are available
        _lightService?.Update(camPos);
        if (_lightService != null && _lightService.ActiveLightId >= 0)
        {
            // Override sky dome colors from DBC Light data
            _skyDome.ZenithColor = _lightService.SkyTopColor;
            _skyDome.HorizonColor = _lightService.FogColor;
            _skyDome.SkyFogColor = _lightService.FogColor;
        }
        else
        {
            _skyDome.UpdateFromLighting(_terrainManager.Lighting.GameTime);
        }
        _skyDome.Render(view, proj, camPos);

        // Also set clear color to horizon color so any gaps match the sky
        _gl.ClearColor(_skyDome.HorizonColor.X, _skyDome.HorizonColor.Y, _skyDome.HorizonColor.Z, 1f);

        // 0. Render WDL low-res terrain (far background — hidden tiles replaced by detailed ADTs)
        if (ShowWdlTerrain && _wdlTerrain != null)
            _wdlTerrain.Render(view, proj, camPos, _terrainManager.Lighting, _frustumCuller);

        // 1. Render terrain (with frustum culling)
        _terrainManager.Render(view, proj, camPos, _frustumCuller);

        // Reset GL state after terrain
        _gl.DepthFunc(DepthFunction.Lequal);
        _gl.DepthMask(true);
        _gl.Disable(EnableCap.Blend);
        _gl.Enable(EnableCap.DepthTest);
        _gl.UseProgram(0); // unbind terrain shader

        if (!_objectsVisible) return;

        // One-time render diagnostic
        if (!_renderDiagPrinted)
        {
            int wmoFound = 0, wmoMissing = 0;
            foreach (var inst in _wmoInstances)
            {
                if (_assets.GetWmo(inst.ModelKey) != null) wmoFound++;
                else { wmoMissing++; if (wmoMissing <= 3) ViewerLog.Debug(ViewerLog.Category.Wmo, $"NOT FOUND: \"{inst.ModelKey}\""); }
            }
            int mdxFound = 0, mdxMissing = 0;
            foreach (var inst in _mdxInstances)
            {
                if (_assets.GetMdx(inst.ModelKey) != null) mdxFound++;
                else { mdxMissing++; if (mdxMissing <= 3) ViewerLog.Debug(ViewerLog.Category.Mdx, $"NOT FOUND: \"{inst.ModelKey}\""); }
            }
            ViewerLog.Info(ViewerLog.Category.Terrain, $"Render check: WMO {wmoFound} found / {wmoMissing} missing, MDX {mdxFound} found / {mdxMissing} missing");
        }

        // Extract camera position from view matrix (inverse of view translation)
        Matrix4x4.Invert(view, out var viewInv);
        var cameraPos = new Vector3(viewInv.M41, viewInv.M42, viewInv.M43);

        // Fog parameters: prefer DBC light data if available, else terrain defaults
        // (LightService already updated earlier for sky dome)
        var lighting = _terrainManager.Lighting;
        Vector3 fogColor;
        float fogStart, fogEnd;
        if (_lightService != null && _lightService.ActiveLightId >= 0)
        {
            fogColor = _lightService.FogColor;
            fogEnd = _lightService.FogEnd > 10f ? _lightService.FogEnd : lighting.FogEnd;
            fogStart = fogEnd * 0.25f; // Fog starts at 25% of end distance
        }
        else
        {
            fogColor = lighting.FogColor;
            fogStart = lighting.FogStart;
            fogEnd = lighting.FogEnd;
        }

        // Update frustum planes for culling
        var vp = view * proj;
        _frustumCuller.Update(vp);

        // ── PASS 1: OPAQUE ──────────────────────────────────────────────
        // Render all opaque geometry first with depth write ON.
        // This ensures correct depth buffer before any transparent rendering.
        _gl.Enable(EnableCap.DepthTest);
        _gl.DepthFunc(DepthFunction.Less);
        _gl.DepthMask(true);
        _gl.Disable(EnableCap.Blend);

        // 2a. WMO opaque pass (with frustum + distance culling + fade)
        WmoRenderedCount = 0;
        WmoCulledCount = 0;
        if (_wmosVisible)
        {
            float wmoCullDistanceSq = WmoCullDistance * WmoCullDistance;
            float wmoFadeStart = WmoCullDistance * WmoFadeStartFraction;
            float wmoFadeStartSq = wmoFadeStart * wmoFadeStart;
            float wmoFadeRange = WmoCullDistance - wmoFadeStart;

            // State is constant for this pass; set once to avoid per-instance churn.
            _gl.Disable(EnableCap.Blend);
            _gl.DepthMask(true);

            foreach (var inst in _wmoInstances)
            {
                var wmoCenter = (inst.BoundsMin + inst.BoundsMax) * 0.5f;
                float wmoDistSq = Vector3.DistanceSquared(cameraPos, wmoCenter);
                // Skip frustum cull for nearby objects to prevent pop-in when turning
                if (wmoDistSq > NoCullRadiusSq && !_frustumCuller.TestAABB(inst.BoundsMin, inst.BoundsMax))
                { WmoCulledCount++; continue; }
                // Distance cull + fade for WMOs
                if (wmoDistSq > wmoCullDistanceSq)
                { WmoCulledCount++; continue; }

                float wmoFade = 1.0f;
                if (wmoDistSq > wmoFadeStartSq)
                {
                    float wmoDist = MathF.Sqrt(wmoDistSq);
                    wmoFade = 1.0f - (wmoDist - wmoFadeStart) / wmoFadeRange;
                }

                var renderer = _assets.GetWmo(inst.ModelKey);
                if (renderer == null) continue;
                renderer.RenderWithTransform(inst.Transform, view, proj,
                    fogColor, fogStart, fogEnd, cameraPos,
                    lighting.LightDirection, lighting.LightColor, lighting.AmbientColor);
                WmoRenderedCount++;
            }
            if (!_renderDiagPrinted) ViewerLog.Info(ViewerLog.Category.Wmo, $"WMO render: {WmoRenderedCount} drawn, {WmoCulledCount} culled");
        }

        // 3a. MDX opaque pass (with frustum + distance culling + fade)
        MdxRenderedCount = 0;
        MdxCulledCount = 0;
        float mdxFadeStart = DoodadCullDistance * FadeStartFraction;
        float mdxFadeStartSq = mdxFadeStart * mdxFadeStart;
        float mdxFadeRange = DoodadCullDistance - mdxFadeStart;

        // Advance animation once per unique MDX renderer before any render passes
        if (_doodadsVisible)
        {
            _updatedMdxRenderers.Clear();
            foreach (var inst in _mdxInstances)
            {
                if (_updatedMdxRenderers.Add(inst.ModelKey))
                {
                    var r = _assets.GetMdx(inst.ModelKey);
                    r?.UpdateAnimation();
                }
            }
        }

        if (_doodadsVisible)
        {
            // Set up shared per-frame state once (shader, view/proj, fog, lighting).
            // Safe because all MdxRenderers share a single static shader program.
            MdxRenderer? batchRenderer = null;
            foreach (var inst in _mdxInstances)
            {
                batchRenderer = _assets.GetMdx(inst.ModelKey);
                if (batchRenderer != null) break;
            }
            batchRenderer?.BeginBatch(view, proj, fogColor, fogStart, fogEnd, cameraPos,
                lighting.LightDirection, lighting.LightColor, lighting.AmbientColor);

            foreach (var inst in _mdxInstances)
            {
                // Use placement position (transform translation) for distance — more reliable
                // than AABB center when rotation transforms are imprecise
                var placementPos = inst.Transform.Translation;
                float distSq = Vector3.DistanceSquared(cameraPos, placementPos);
                // Skip frustum cull for nearby objects to prevent pop-in when turning
                if (distSq > NoCullRadiusSq && !_frustumCuller.TestAABB(inst.BoundsMin, inst.BoundsMax))
                { MdxCulledCount++; continue; }
                // Distance cull small doodads (with fade)
                var diag = (inst.BoundsMax - inst.BoundsMin).Length();
                if (diag < DoodadSmallThreshold && distSq > DoodadCullDistanceSq)
                { MdxCulledCount++; continue; }
                // Compute fade factor for objects near cull boundary
                float fade = 1.0f;
                if (diag < DoodadSmallThreshold && distSq > mdxFadeStartSq)
                {
                    float dist = MathF.Sqrt(distSq);
                    fade = MathF.Max(0f, 1.0f - (dist - mdxFadeStart) / mdxFadeRange);
                }
                var renderer = _assets.GetMdx(inst.ModelKey);
                if (renderer == null) continue;
                renderer.RenderInstance(inst.Transform, RenderPass.Opaque, fade);
                MdxRenderedCount++;
            }
            if (!_renderDiagPrinted) ViewerLog.Info(ViewerLog.Category.Mdx, $"MDX opaque: {MdxRenderedCount} drawn, {MdxCulledCount} culled");
        }

        // ── PASS 2: TRANSPARENT (back-to-front, frustum-culled) ─────────
        // Render transparent/blended layers sorted by distance to camera.
        // Depth test ON but depth write OFF so transparent objects don't
        // occlude each other incorrectly.
        if (_doodadsVisible)
        {
            _gl.Enable(EnableCap.DepthTest);
            _gl.DepthFunc(DepthFunction.Lequal);

            // Sort visible instances back-to-front by distance to camera
            _transparentSortScratch.Clear();
            for (int i = 0; i < _mdxInstances.Count; i++)
            {
                var inst = _mdxInstances[i];
                if (_assets.GetMdx(inst.ModelKey) == null) continue;
                // Same frustum + distance cull as opaque pass (with NoCullRadius)
                var placementPos = inst.Transform.Translation;
                float dist = Vector3.DistanceSquared(cameraPos, placementPos);
                if (dist > NoCullRadiusSq && !_frustumCuller.TestAABB(inst.BoundsMin, inst.BoundsMax)) continue;
                var diag = (inst.BoundsMax - inst.BoundsMin).Length();
                if (diag < DoodadSmallThreshold && dist > DoodadCullDistanceSq) continue;
                _transparentSortScratch.Add((i, dist));
            }
            _transparentSortScratch.Sort((a, b) => b.distSq.CompareTo(a.distSq)); // back-to-front

            foreach (var (idx, distSq) in _transparentSortScratch)
            {
                var inst = _mdxInstances[idx];
                // Compute fade for transparent pass (same as opaque)
                float tDist = MathF.Sqrt(distSq);
                var tDiag = (inst.BoundsMax - inst.BoundsMin).Length();
                float tFade = 1.0f;
                if (tDiag < DoodadSmallThreshold && tDist > mdxFadeStart)
                    tFade = MathF.Max(0f, 1.0f - (tDist - mdxFadeStart) / mdxFadeRange);
                var renderer = _assets.GetMdx(inst.ModelKey);
                if (renderer == null) continue;
                renderer.RenderInstance(inst.Transform, RenderPass.Transparent, tFade);
            }
            if (!_renderDiagPrinted) _renderDiagPrinted = true;
        }
        else
        {
            if (!_renderDiagPrinted) _renderDiagPrinted = true;
        }

        // ── PASS 3: LIQUID ──────────────────────────────────────────────
        // Render liquid surfaces LAST so all opaque geometry (terrain, WMOs, MDX)
        // is already in the framebuffer. Liquid uses alpha blending + depth mask off,
        // so objects below the water surface are visible through the transparent water.
        _gl.Disable(EnableCap.Blend);
        _gl.DepthMask(true);
        _gl.Enable(EnableCap.DepthTest);
        _gl.DepthFunc(DepthFunction.Lequal);
        _terrainManager.RenderLiquid(view, proj, cameraPos);

        // Reset GL state before bounding boxes
        _gl.Disable(EnableCap.Blend);
        _gl.DepthMask(true);
        _gl.Enable(EnableCap.DepthTest);
        _gl.DepthFunc(DepthFunction.Lequal);
        _gl.UseProgram(0);
        _gl.BindVertexArray(0);

        // 4. Debug bounding boxes for all placements
        if (_showBoundingBoxes && _bbRenderer != null)
        {
            // Depth test ON so boxes behind terrain/objects are hidden,
            // depth write OFF so box lines don't occlude models
            _gl.Enable(EnableCap.DepthTest);
            _gl.DepthFunc(DepthFunction.Lequal);
            _gl.DepthMask(false);

            var adapter = _terrainManager.Adapter;
            if (!_renderDiagPrinted)
            ViewerLog.Debug(ViewerLog.Category.Terrain, $"BB render: {adapter.MddfPlacements.Count} MDDF + {adapter.ModfPlacements.Count} MODF markers");

            // Draw selected object highlight first (thicker visual via slightly larger box)
            if (SelectedInstance is ObjectInstance sel)
            {
                _bbRenderer.DrawBoxMinMax(sel.BoundsMin, sel.BoundsMax, view, proj, new Vector3(1f, 1f, 1f)); // white highlight
            }

            // MDDF bounding boxes (magenta)
            foreach (var inst in _mdxInstances)
                _bbRenderer.DrawBoxMinMax(inst.BoundsMin, inst.BoundsMax, view, proj, new Vector3(1f, 0f, 1f));
            // MODF bounding boxes (cyan)
            foreach (var inst in _wmoInstances)
                _bbRenderer.DrawBoxMinMax(inst.BoundsMin, inst.BoundsMax, view, proj, new Vector3(0f, 1f, 1f));

            _gl.DepthMask(true);
        }

        // 5+6. Batched overlay rendering (POI pins + taxi paths) — single draw call
        if (_bbRenderer != null)
        {
            _bbRenderer.BeginBatch();

            // POI pin markers (magenta)
            if (_showPoi && _poiLoader != null && _poiLoader.Entries.Count > 0)
            {
                var poiColor = new Vector3(1f, 0f, 1f);
                foreach (var poi in _poiLoader.Entries)
                    _bbRenderer.BatchPin(poi.Position, 40f, 6f, poiColor);
            }

            // Taxi paths — filtered by selection
            if (_showTaxi && _taxiLoader != null)
            {
                var nodeColor = new Vector3(1f, 1f, 0f);
                var lineColor = new Vector3(0f, 1f, 1f);

                foreach (var node in _taxiLoader.Nodes)
                {
                    if (!IsTaxiNodeVisible(node)) continue;
                    _bbRenderer.BatchPin(node.Position, 50f, 8f, nodeColor);
                }

                foreach (var route in _taxiLoader.Routes)
                {
                    if (!IsTaxiRouteVisible(route)) continue;
                    for (int i = 0; i < route.Waypoints.Count - 1; i++)
                        _bbRenderer.BatchLine(route.Waypoints[i], route.Waypoints[i + 1], lineColor);
                }
            }

            // AreaTriggers (green wireframe shapes for portals and event markers)
            if (_showAreaTriggers && _areaTriggerLoader != null && _areaTriggerLoader.Count > 0)
            {
                var triggerColor = new Vector3(0f, 1f, 0f); // Green
                foreach (var trigger in _areaTriggerLoader.Triggers)
                {
                    if (trigger.IsSphere && trigger.Radius > 0f)
                    {
                        // Render sphere triggers as simple wireframe circles (3 orthogonal rings)
                        int segments = 16;
                        float r = trigger.Radius;
                        var c = trigger.Position;
                        
                        // XY plane circle
                        for (int i = 0; i < segments; i++)
                        {
                            float a1 = (i / (float)segments) * MathF.PI * 2f;
                            float a2 = ((i + 1) / (float)segments) * MathF.PI * 2f;
                            var p1 = c + new Vector3(MathF.Cos(a1) * r, MathF.Sin(a1) * r, 0f);
                            var p2 = c + new Vector3(MathF.Cos(a2) * r, MathF.Sin(a2) * r, 0f);
                            _bbRenderer.BatchLine(p1, p2, triggerColor);
                        }
                        
                        // XZ plane circle
                        for (int i = 0; i < segments; i++)
                        {
                            float a1 = (i / (float)segments) * MathF.PI * 2f;
                            float a2 = ((i + 1) / (float)segments) * MathF.PI * 2f;
                            var p1 = c + new Vector3(MathF.Cos(a1) * r, 0f, MathF.Sin(a1) * r);
                            var p2 = c + new Vector3(MathF.Cos(a2) * r, 0f, MathF.Sin(a2) * r);
                            _bbRenderer.BatchLine(p1, p2, triggerColor);
                        }
                        
                        // YZ plane circle
                        for (int i = 0; i < segments; i++)
                        {
                            float a1 = (i / (float)segments) * MathF.PI * 2f;
                            float a2 = ((i + 1) / (float)segments) * MathF.PI * 2f;
                            var p1 = c + new Vector3(0f, MathF.Cos(a1) * r, MathF.Sin(a1) * r);
                            var p2 = c + new Vector3(0f, MathF.Cos(a2) * r, MathF.Sin(a2) * r);
                            _bbRenderer.BatchLine(p1, p2, triggerColor);
                        }
                    }
                    else if (trigger.BoxLength > 0f && trigger.BoxWidth > 0f && trigger.BoxHeight > 0f)
                    {
                        // Render box triggers as wireframe boxes (12 edges)
                        float halfL = trigger.BoxLength / 2f;
                        float halfW = trigger.BoxWidth / 2f;
                        float h = trigger.BoxHeight;
                        var c = trigger.Position;
                        
                        // 8 corners of the box
                        var v0 = c + new Vector3(-halfL, -halfW, 0f);
                        var v1 = c + new Vector3( halfL, -halfW, 0f);
                        var v2 = c + new Vector3( halfL,  halfW, 0f);
                        var v3 = c + new Vector3(-halfL,  halfW, 0f);
                        var v4 = c + new Vector3(-halfL, -halfW, h);
                        var v5 = c + new Vector3( halfL, -halfW, h);
                        var v6 = c + new Vector3( halfL,  halfW, h);
                        var v7 = c + new Vector3(-halfL,  halfW, h);
                        
                        // Bottom face
                        _bbRenderer.BatchLine(v0, v1, triggerColor);
                        _bbRenderer.BatchLine(v1, v2, triggerColor);
                        _bbRenderer.BatchLine(v2, v3, triggerColor);
                        _bbRenderer.BatchLine(v3, v0, triggerColor);
                        
                        // Top face
                        _bbRenderer.BatchLine(v4, v5, triggerColor);
                        _bbRenderer.BatchLine(v5, v6, triggerColor);
                        _bbRenderer.BatchLine(v6, v7, triggerColor);
                        _bbRenderer.BatchLine(v7, v4, triggerColor);
                        
                        // Vertical edges
                        _bbRenderer.BatchLine(v0, v4, triggerColor);
                        _bbRenderer.BatchLine(v1, v5, triggerColor);
                        _bbRenderer.BatchLine(v2, v6, triggerColor);
                        _bbRenderer.BatchLine(v3, v7, triggerColor);
                    }
                }
            }

            _bbRenderer.FlushBatch(view, proj);
        }
    }

    public void ToggleWireframe()
    {
        _terrainManager.ToggleWireframe();
    }

    public void ToggleObjects() => _objectsVisible = !_objectsVisible;
    public void ToggleWmos() => _wmosVisible = !_wmosVisible;
    public void ToggleDoodads() => _doodadsVisible = !_doodadsVisible;

    public int SubObjectCount => 3;

    public string GetSubObjectName(int index) => index switch
    {
        0 => $"Terrain ({_terrainManager.LoadedChunkCount} chunks)",
        1 => $"WMOs ({_wmoInstances.Count} instances, {UniqueWmoModels} unique)",
        2 => $"Doodads ({_mdxInstances.Count} instances, {UniqueMdxModels} unique)",
        _ => ""
    };

    public bool GetSubObjectVisible(int index) => index switch
    {
        0 => true,
        1 => _wmosVisible,
        2 => _doodadsVisible,
        _ => false
    };

    public void SetSubObjectVisible(int index, bool visible)
    {
        switch (index)
        {
            case 1: _wmosVisible = visible; break;
            case 2: _doodadsVisible = visible; break;
        }
    }

    /// <summary>
    /// Select the nearest object whose AABB is hit by a ray from camera.
    /// Call with screen-space mouse coords to pick objects.
    /// </summary>
    public void SelectObjectByRay(Vector3 rayOrigin, Vector3 rayDir)
    {
        if (_instancesDirty)
            RebuildInstanceLists();

        float bestT = float.MaxValue;
        ObjectType bestType = ObjectType.None;
        int bestIndex = -1;

        var hits = new List<(string type, int index, float dist, string name)>();

        // Test WMO bounding boxes
        for (int i = 0; i < _wmoInstances.Count; i++)
        {
            // Slightly inflate AABBs to make selection more forgiving for thin geometry.
            Vector3 pad = new(2f, 2f, 2f);
            float t = RayAABBIntersect(rayOrigin, rayDir, _wmoInstances[i].BoundsMin - pad, _wmoInstances[i].BoundsMax + pad);
            if (t >= 0)
            {
                hits.Add(("WMO", i, t, _wmoInstances[i].ModelName));
                if (t < bestT) { bestT = t; bestType = ObjectType.Wmo; bestIndex = i; }
            }
        }

        // Test MDX bounding boxes
        for (int i = 0; i < _mdxInstances.Count; i++)
        {
            Vector3 pad = new(1f, 1f, 1f);
            float t = RayAABBIntersect(rayOrigin, rayDir, _mdxInstances[i].BoundsMin - pad, _mdxInstances[i].BoundsMax + pad);
            if (t >= 0)
            {
                hits.Add(("MDX", i, t, _mdxInstances[i].ModelName));
                if (t < bestT) { bestT = t; bestType = ObjectType.Mdx; bestIndex = i; }
            }
        }

        // Debug: log all hits sorted by distance
        if (hits.Count > 0)
        {
            var sorted = hits.OrderBy(h => h.dist).ToList();
            ViewerLog.Debug(ViewerLog.Category.Terrain, $"[ObjectPick] Ray hit {hits.Count} objects:");
            foreach (var h in sorted.Take(5))
                ViewerLog.Debug(ViewerLog.Category.Terrain, $"  {h.type}[{h.index}] {h.name} @ dist={h.dist:F1}");
            if (sorted.Count > 5)
                ViewerLog.Debug(ViewerLog.Category.Terrain, $"  ... and {sorted.Count - 5} more");
        }

        _selectedObjectType = bestType;
        _selectedObjectIndex = bestIndex;
    }

    public void ClearSelection()
    {
        _selectedObjectType = ObjectType.None;
        _selectedObjectIndex = -1;
    }

    /// <summary>
    /// Ray-AABB slab intersection test. Returns distance along ray, or -1 if no hit.
    /// </summary>
    private static float RayAABBIntersect(Vector3 origin, Vector3 dir, Vector3 bmin, Vector3 bmax)
    {
        float tmin = float.NegativeInfinity;
        float tmax = float.PositiveInfinity;

        for (int i = 0; i < 3; i++)
        {
            float o = i == 0 ? origin.X : i == 1 ? origin.Y : origin.Z;
            float d = i == 0 ? dir.X : i == 1 ? dir.Y : dir.Z;
            float lo = i == 0 ? bmin.X : i == 1 ? bmin.Y : bmin.Z;
            float hi = i == 0 ? bmax.X : i == 1 ? bmax.Y : bmax.Z;

            if (MathF.Abs(d) < 1e-8f)
            {
                if (o < lo || o > hi) return -1;
            }
            else
            {
                float t1 = (lo - o) / d;
                float t2 = (hi - o) / d;
                if (t1 > t2) (t1, t2) = (t2, t1);
                tmin = MathF.Max(tmin, t1);
                tmax = MathF.Min(tmax, t2);
                if (tmin > tmax) return -1;
            }
        }

        return tmin >= 0 ? tmin : tmax >= 0 ? tmax : -1;
    }

    /// <summary>
    /// Build a world-space ray from normalized device coordinates using view/proj matrices.
    /// </summary>
    public static (Vector3 origin, Vector3 dir) ScreenToRay(float ndcX, float ndcY, Matrix4x4 view, Matrix4x4 proj)
    {
        Matrix4x4.Invert(proj, out var invProj);
        Matrix4x4.Invert(view, out var invView);

        // Near point in clip space → world
        var nearClip = new Vector4(ndcX, ndcY, -1f, 1f);
        var nearView = Vector4.Transform(nearClip, invProj);
        nearView /= nearView.W;
        var nearWorld = Vector4.Transform(nearView, invView);

        // Far point in clip space → world
        var farClip = new Vector4(ndcX, ndcY, 1f, 1f);
        var farView = Vector4.Transform(farClip, invProj);
        farView /= farView.W;
        var farWorld = Vector4.Transform(farView, invView);

        var origin = new Vector3(nearWorld.X, nearWorld.Y, nearWorld.Z);
        var farPt = new Vector3(farWorld.X, farWorld.Y, farWorld.Z);
        var dir = Vector3.Normalize(farPt - origin);
        return (origin, dir);
    }

    public void Dispose()
    {
        _terrainManager.OnTileLoaded -= OnTileLoaded;
        _terrainManager.OnTileUnloaded -= OnTileUnloaded;
        _terrainManager.Dispose();
        _wdlTerrain?.Dispose();
        _assets.Dispose();
        _bbRenderer?.Dispose();
        _skyDome.Dispose();
        _mdxInstances.Clear();
        _wmoInstances.Clear();
        _tileMdxInstances.Clear();
        _tileWmoInstances.Clear();
        _externalMdxInstances.Clear();
        _externalWmoInstances.Clear();
    }
}

/// <summary>
/// Lightweight placement instance — just a model key and world transform.
/// The actual renderer is looked up from WorldAssetManager at render time.
/// </summary>
public struct ObjectInstance
{
    public string ModelKey;
    public Matrix4x4 Transform;
    /// <summary>World-space AABB (local bounds transformed through placement matrix).</summary>
    public Vector3 BoundsMin;
    /// <summary>World-space AABB (local bounds transformed through placement matrix).</summary>
    public Vector3 BoundsMax;
    /// <summary>Model-local bounding box min (MOHD for WMO, model extents for MDX). Zero if unavailable.</summary>
    public Vector3 LocalBoundsMin;
    /// <summary>Model-local bounding box max (MOHD for WMO, model extents for MDX). Zero if unavailable.</summary>
    public Vector3 LocalBoundsMax;
    /// <summary>Display name (filename) for UI.</summary>
    public string ModelName;
    /// <summary>Renderer-space position from placement.</summary>
    public Vector3 PlacementPosition;
    /// <summary>Rotation in degrees from placement.</summary>
    public Vector3 PlacementRotation;
    /// <summary>Scale from placement (1.0 = default).</summary>
    public float PlacementScale;
    /// <summary>Full model path for diagnostics.</summary>
    public string ModelPath;
    /// <summary>UniqueId from MODF/MDDF placement (for dedup and display).</summary>
    public int UniqueId;
}

public enum ObjectType { None, Wmo, Mdx }
