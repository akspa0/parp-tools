using System.Numerics;
using MdxViewer.DataSources;
using MdxViewer.Logging;
using MdxViewer.Population;
using MdxViewer.Rendering;
using Silk.NET.OpenGL;
using WoWMapConverter.Core.Formats.PM4;

namespace MdxViewer.Terrain;

public enum Pm4OverlayColorMode
{
    Ck24Type,
    Ck24ObjectId,
    Ck24Key,
    Tile,
    GroupKey,
    AttributeMask,
    Height
}

internal readonly struct Pm4PlanarTransform
{
    public Pm4PlanarTransform(bool swapPlanarAxes, bool invertU, bool invertV)
    {
        SwapPlanarAxes = swapPlanarAxes;
        InvertU = invertU;
        InvertV = invertV;
    }

    public bool SwapPlanarAxes { get; }
    public bool InvertU { get; }
    public bool InvertV { get; }

    public bool InvertsWinding
    {
        get
        {
            int parity = 0;
            if (SwapPlanarAxes) parity++;
            if (InvertU) parity++;
            if (InvertV) parity++;
            return (parity & 1) != 0;
        }
    }
}

public readonly struct Pm4ObjectDebugInfo
{
    public Pm4ObjectDebugInfo(
        uint ck24,
        byte ck24Type,
        ushort ck24ObjectId,
        int objectPartId,
        uint linkGroupObjectId,
        int linkedPositionRefCount,
        int tileX,
        int tileY,
        int surfaceCount,
        byte dominantGroupKey,
        byte dominantAttributeMask,
        uint dominantMdosIndex,
        float averageSurfaceHeight,
        Vector3 boundsMin,
        Vector3 boundsMax,
        Vector3 center,
        float nearestPositionRefDistance,
        bool swapPlanarAxes,
        bool invertU,
        bool invertV,
        bool invertsWinding)
    {
        Ck24 = ck24;
        Ck24Type = ck24Type;
        Ck24ObjectId = ck24ObjectId;
        ObjectPartId = objectPartId;
        LinkGroupObjectId = linkGroupObjectId;
        LinkedPositionRefCount = linkedPositionRefCount;
        TileX = tileX;
        TileY = tileY;
        SurfaceCount = surfaceCount;
        DominantGroupKey = dominantGroupKey;
        DominantAttributeMask = dominantAttributeMask;
        DominantMdosIndex = dominantMdosIndex;
        AverageSurfaceHeight = averageSurfaceHeight;
        BoundsMin = boundsMin;
        BoundsMax = boundsMax;
        Center = center;
        NearestPositionRefDistance = nearestPositionRefDistance;
        SwapPlanarAxes = swapPlanarAxes;
        InvertU = invertU;
        InvertV = invertV;
        InvertsWinding = invertsWinding;
    }

    public uint Ck24 { get; }
    public byte Ck24Type { get; }
    public ushort Ck24ObjectId { get; }
    public int ObjectPartId { get; }
    public uint LinkGroupObjectId { get; }
    public int LinkedPositionRefCount { get; }
    public int TileX { get; }
    public int TileY { get; }
    public int SurfaceCount { get; }
    public byte DominantGroupKey { get; }
    public byte DominantAttributeMask { get; }
    public uint DominantMdosIndex { get; }
    public float AverageSurfaceHeight { get; }
    public Vector3 BoundsMin { get; }
    public Vector3 BoundsMax { get; }
    public Vector3 Center { get; }
    public float NearestPositionRefDistance { get; }
    public bool SwapPlanarAxes { get; }
    public bool InvertU { get; }
    public bool InvertV { get; }
    public bool InvertsWinding { get; }
}

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
    private List<ObjectInstance> _skyboxInstances = new();
    private List<ObjectInstance> _wmoInstances = new();

    // Per-tile instance storage for lazy load/unload
    private readonly Dictionary<(int, int), List<ObjectInstance>> _tileMdxInstances = new();
    private readonly Dictionary<(int, int), List<ObjectInstance>> _tileSkyboxInstances = new();
    private readonly Dictionary<(int, int), List<ObjectInstance>> _tileWmoInstances = new();
    private readonly List<ObjectInstance> _externalMdxInstances = new();
    private readonly List<ObjectInstance> _externalSkyboxInstances = new();
    private readonly List<ObjectInstance> _externalWmoInstances = new();
    private bool _instancesDirty = false;

    private bool _objectsVisible = true;
    private bool _wmosVisible = true;
    private bool _doodadsVisible = true;

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
    private const float WireframeRevealBrushPixels = 96f;
    private const float WireframeRevealMaxScreenRadius = 220f;

    // Scratch collections reused every frame to avoid hot-path allocations.
    private readonly HashSet<string> _updatedMdxRenderers = new();
    private readonly List<(int idx, float distSq)> _transparentSortScratch = new();
    private readonly List<int> _wireframeRevealWmoIndices = new();
    private readonly List<int> _wireframeRevealMdxIndices = new();
    private bool _wireframeRevealEnabled;

    // PM4 debug overlay
    private const int Pm4MaxLinesTotal = 400000;
    private const int Pm4MaxLinesPerTile = 12000;
    private const int Pm4MaxTrianglesTotal = 240000;
    private const int Pm4MaxTrianglesPerTile = 12000;
    private const int Pm4MaxPositionRefsTotal = 120000;
    private const int Pm4MaxPositionRefsPerTile = 3000;
    private const float Pm4MaxEdgeLength = 512f;
    private bool _showPm4Overlay;
    private bool _showPm4SolidOverlay = true;
    private bool _showPm4PositionRefs;
    private bool _showPm4ObjectCentroids;
    private bool _pm4SplitCk24ByConnectivity;
    private bool _showPm4Type40 = true;
    private bool _showPm4Type80 = true;
    private bool _showPm4TypeOther = true;
    private bool _pm4SplitCk24ByMdos;
    private Pm4OverlayColorMode _pm4ColorMode = Pm4OverlayColorMode.Ck24Type;
    private Vector3 _pm4OverlayTranslation = Vector3.Zero;
    private Vector3 _pm4OverlayRotationDegrees = Vector3.Zero;
    private Vector3 _pm4OverlayScale = Vector3.One;
    private bool _pm4LoadAttempted;
    private string _pm4Status = "PM4 overlay not loaded.";
    private int _pm4TotalFiles;
    private int _pm4LoadedFiles;
    private int _pm4ObjectCount;
    private int _pm4LineCount;
    private int _pm4TriangleCount;
    private int _pm4RejectedLongEdges;
    private int _pm4VisibleObjectCount;
    private int _pm4VisibleLineCount;
    private int _pm4VisibleTriangleCount;
    private int _pm4PositionRefCount;
    private int _pm4VisiblePositionRefCount;
    private float _pm4MinObjectZ;
    private float _pm4MaxObjectZ;
    private readonly Dictionary<(int tileX, int tileY), List<Pm4OverlayObject>> _pm4TileObjects = new();
    private readonly Dictionary<(int tileX, int tileY), Pm4OverlayTileStats> _pm4TileStats = new();
    private readonly Dictionary<(int tileX, int tileY), List<Vector3>> _pm4TilePositionRefs = new();
    private readonly Dictionary<(int tileX, int tileY, uint ck24, int objectPart), Pm4OverlayObject> _pm4ObjectLookup = new();
    private readonly Dictionary<(int tileX, int tileY, uint ck24, int objectPart), Vector3> _pm4ObjectTranslations = new();
    private (int tileX, int tileY, uint ck24, int objectPart)? _selectedPm4ObjectKey;

    // Culling stats (updated each frame)
    public int WmoRenderedCount { get; private set; }
    public int WmoCulledCount { get; private set; }
    public int MdxRenderedCount { get; private set; }
    public int MdxCulledCount { get; private set; }

    // Stats
    public int MdxInstanceCount => _mdxInstances.Count;
    public int SkyboxInstanceCount => _skyboxInstances.Count;
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
    public bool WireframeRevealEnabled => _wireframeRevealEnabled;
    public bool ShowPm4Overlay
    {
        get => _showPm4Overlay;
        set
        {
            _showPm4Overlay = value;
            // Allow reattempt when previously loaded with no data, e.g. after attaching a new loose overlay.
            if (value && (!_pm4LoadAttempted || _pm4TileObjects.Count == 0))
                LazyLoadPm4Overlay();
        }
    }
    public bool Pm4LoadAttempted => _pm4LoadAttempted;
    public string Pm4Status => _pm4Status;
    public int Pm4TotalFiles => _pm4TotalFiles;
    public int Pm4LoadedFiles => _pm4LoadedFiles;
    public int Pm4ObjectCount => _pm4ObjectCount;
    public int Pm4LineCount => _pm4LineCount;
    public int Pm4TriangleCount => _pm4TriangleCount;
    public int Pm4RejectedLongEdges => _pm4RejectedLongEdges;
    public int Pm4VisibleObjectCount => _pm4VisibleObjectCount;
    public int Pm4VisibleLineCount => _pm4VisibleLineCount;
    public int Pm4VisibleTriangleCount => _pm4VisibleTriangleCount;
    public int Pm4PositionRefCount => _pm4PositionRefCount;
    public int Pm4VisiblePositionRefCount => _pm4VisiblePositionRefCount;
    public bool ShowPm4SolidOverlay { get => _showPm4SolidOverlay; set => _showPm4SolidOverlay = value; }
    public bool ShowPm4PositionRefs { get => _showPm4PositionRefs; set => _showPm4PositionRefs = value; }
    public bool ShowPm4ObjectCentroids { get => _showPm4ObjectCentroids; set => _showPm4ObjectCentroids = value; }
    public bool Pm4SplitCk24ByConnectivity { get => _pm4SplitCk24ByConnectivity; set => _pm4SplitCk24ByConnectivity = value; }
    public bool ShowPm4Type40 { get => _showPm4Type40; set => _showPm4Type40 = value; }
    public bool ShowPm4Type80 { get => _showPm4Type80; set => _showPm4Type80 = value; }
    public bool ShowPm4TypeOther { get => _showPm4TypeOther; set => _showPm4TypeOther = value; }
    public bool Pm4SplitCk24ByMdos { get => _pm4SplitCk24ByMdos; set => _pm4SplitCk24ByMdos = value; }
    public Pm4OverlayColorMode Pm4ColorMode { get => _pm4ColorMode; set => _pm4ColorMode = value; }
    public Vector3 Pm4OverlayTranslation { get => _pm4OverlayTranslation; set => _pm4OverlayTranslation = value; }
    public Vector3 Pm4OverlayRotationDegrees { get => _pm4OverlayRotationDegrees; set => _pm4OverlayRotationDegrees = value; }
    public Vector3 Pm4OverlayScale { get => _pm4OverlayScale; set => _pm4OverlayScale = value; }
    public bool HasSelectedPm4Object => _selectedPm4ObjectKey.HasValue;
    public (int tileX, int tileY, uint ck24, int objectPart)? SelectedPm4ObjectKey => _selectedPm4ObjectKey;
    public Vector3 SelectedPm4ObjectTranslation
    {
        get
        {
            if (!_selectedPm4ObjectKey.HasValue)
                return Vector3.Zero;

            return _pm4ObjectTranslations.TryGetValue(_selectedPm4ObjectKey.Value, out Vector3 translation)
                ? translation
                : Vector3.Zero;
        }
        set
        {
            if (!_selectedPm4ObjectKey.HasValue)
                return;

            if (value.LengthSquared() < 0.0001f)
                _pm4ObjectTranslations.Remove(_selectedPm4ObjectKey.Value);
            else
                _pm4ObjectTranslations[_selectedPm4ObjectKey.Value] = value;
        }
    }
    public float Pm4OverlayYawDegrees
    {
        get => _pm4OverlayRotationDegrees.Z;
        set => _pm4OverlayRotationDegrees = new Vector3(_pm4OverlayRotationDegrees.X, _pm4OverlayRotationDegrees.Y, value);
    }
    public IReadOnlyCollection<Pm4OverlayTileStats> Pm4TileStats => _pm4TileStats.Values;

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
        _assets.SetBuildVersion(build);
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

    private void LazyLoadPm4Overlay()
    {
        _pm4LoadAttempted = true;
        _pm4TileObjects.Clear();
        _pm4TileStats.Clear();
        _pm4TilePositionRefs.Clear();
        _pm4ObjectLookup.Clear();
        _selectedPm4ObjectKey = null;
        _pm4TotalFiles = 0;
        _pm4LoadedFiles = 0;
        _pm4ObjectCount = 0;
        _pm4LineCount = 0;
        _pm4TriangleCount = 0;
        _pm4RejectedLongEdges = 0;
        _pm4VisibleObjectCount = 0;
        _pm4VisibleLineCount = 0;
        _pm4VisibleTriangleCount = 0;
        _pm4PositionRefCount = 0;
        _pm4VisiblePositionRefCount = 0;
        _pm4MinObjectZ = float.MaxValue;
        _pm4MaxObjectZ = float.MinValue;

        if (_dataSource == null)
        {
            _pm4Status = "PM4 unavailable: no data source.";
            return;
        }

        string mapName = _terrainManager.MapName;
        var pm4Candidates = _dataSource
            .GetFileList(".pm4")
            .Where(path => IsMapPm4Path(path, mapName))
            .ToList();

        _pm4TotalFiles = pm4Candidates.Count;
        if (_pm4TotalFiles == 0)
        {
            _pm4Status = $"PM4: no files found for map '{mapName}'.";
            return;
        }

        int remainingLineBudget = Pm4MaxLinesTotal;
        int remainingTriangleBudget = Pm4MaxTrianglesTotal;
        int remainingPositionRefBudget = Pm4MaxPositionRefsTotal;
        foreach (string pm4Path in pm4Candidates)
        {
            if (remainingLineBudget <= 0)
                break;

            if (!Pm4CoordinateService.TryParseTileCoordinates(pm4Path, out int fileTileX, out int fileTileY))
                continue;

            if (!TryMapPm4FileTileToTerrainTile(fileTileX, fileTileY, out int effectiveTileX, out int effectiveTileY))
                continue;

            byte[]? bytes = _dataSource.ReadFile(pm4Path);
            if (bytes == null || bytes.Length == 0)
                continue;

            try
            {
                var pm4 = new Pm4File(bytes);
                int rejectedLongEdges = 0;
                List<Pm4OverlayObject> objects = BuildPm4TileObjects(
                    pm4,
                    effectiveTileX,
                    effectiveTileY,
                    _pm4SplitCk24ByMdos,
                    _pm4SplitCk24ByConnectivity,
                    ref remainingLineBudget,
                    ref remainingTriangleBudget,
                    ref rejectedLongEdges);
                if (objects.Count == 0)
                    continue;

                if (_pm4TileObjects.TryGetValue((effectiveTileX, effectiveTileY), out List<Pm4OverlayObject>? existingObjects))
                {
                    ViewerLog.Debug(
                        ViewerLog.Category.Terrain,
                        $"[PM4] Multiple files mapped to tile ({effectiveTileX},{effectiveTileY}); merging '{Path.GetFileName(pm4Path)}' into existing overlay tile.");

                    int objectPartOffset = existingObjects.Count;
                    objects = RebasePm4ObjectParts(objects, objectPartOffset);
                    existingObjects.AddRange(objects);
                }
                else
                {
                    _pm4TileObjects[(effectiveTileX, effectiveTileY)] = objects;
                }

                foreach (Pm4OverlayObject obj in objects)
                {
                    _pm4ObjectLookup[(effectiveTileX, effectiveTileY, obj.Ck24, obj.ObjectPartId)] = obj;
                    _pm4MinObjectZ = MathF.Min(_pm4MinObjectZ, obj.Center.Z);
                    _pm4MaxObjectZ = MathF.Max(_pm4MaxObjectZ, obj.Center.Z);
                }

                if (remainingPositionRefBudget > 0)
                {
                    List<Vector3> positionRefs = BuildPm4PositionRefMarkers(pm4, Math.Min(Pm4MaxPositionRefsPerTile, remainingPositionRefBudget));
                    if (positionRefs.Count > 0)
                    {
                        if (_pm4TilePositionRefs.TryGetValue((effectiveTileX, effectiveTileY), out List<Vector3>? existingPositionRefs))
                            existingPositionRefs.AddRange(positionRefs);
                        else
                            _pm4TilePositionRefs[(effectiveTileX, effectiveTileY)] = positionRefs;

                        _pm4PositionRefCount += positionRefs.Count;
                        remainingPositionRefBudget -= positionRefs.Count;
                    }
                }

                _pm4LoadedFiles++;
                _pm4ObjectCount += objects.Count;
                int tileLineCount = objects.Sum(obj => obj.Lines.Count);
                int tileTriangleCount = objects.Sum(obj => obj.Triangles.Count);
                _pm4LineCount += tileLineCount;
                _pm4TriangleCount += tileTriangleCount;
                _pm4RejectedLongEdges += rejectedLongEdges;
                if (_pm4TileStats.TryGetValue((effectiveTileX, effectiveTileY), out Pm4OverlayTileStats existingStats))
                {
                    _pm4TileStats[(effectiveTileX, effectiveTileY)] = new Pm4OverlayTileStats(
                        effectiveTileX,
                        effectiveTileY,
                        existingStats.ObjectCount + objects.Count,
                        existingStats.LineCount + tileLineCount,
                        existingStats.TriangleCount + tileTriangleCount);
                }
                else
                {
                    _pm4TileStats[(effectiveTileX, effectiveTileY)] = new Pm4OverlayTileStats(
                        effectiveTileX,
                        effectiveTileY,
                        objects.Count,
                        tileLineCount,
                        tileTriangleCount);
                }
            }
            catch (Exception ex)
            {
                ViewerLog.Debug(ViewerLog.Category.Terrain, $"[PM4] Failed to decode '{pm4Path}': {ex.Message}");
            }
        }

        if (_pm4LoadedFiles == 0)
        {
            _pm4Status = $"PM4: {_pm4TotalFiles} files found, none decoded into overlay data.";
            return;
        }

        if (_pm4MinObjectZ > _pm4MaxObjectZ)
        {
            _pm4MinObjectZ = 0f;
            _pm4MaxObjectZ = 1f;
        }

        _pm4Status = $"PM4 ready: {_pm4LoadedFiles}/{_pm4TotalFiles} files, {_pm4ObjectCount} CK24 objects, {_pm4LineCount} lines, {_pm4TriangleCount} triangles, {_pm4PositionRefCount} refs, {_pm4RejectedLongEdges} long edges rejected.";
        ViewerLog.Important(ViewerLog.Category.Terrain, "[PM4] " + _pm4Status);
    }

    private static bool IsMapPm4Path(string path, string mapName)
    {
        string normalized = path.Replace('\\', '/');
        string fileName = Path.GetFileName(normalized);
        if (fileName.StartsWith(mapName + "_", StringComparison.OrdinalIgnoreCase))
            return true;

        string mapSegment = "/" + mapName + "/";
        return normalized.Contains(mapSegment, StringComparison.OrdinalIgnoreCase);
    }

    private static bool TryMapPm4FileTileToTerrainTile(int fileTileX, int fileTileY, out int terrainTileX, out int terrainTileY)
    {
        // PM4 filename parsing and PM4 coordinate validation use direct x/y tile pairing.
        // Keep tile reassignment deterministic from filename and avoid heuristic remapping.
        terrainTileX = fileTileX;
        terrainTileY = fileTileY;

        return terrainTileX is >= 0 and <= 63
            && terrainTileY is >= 0 and <= 63;
    }

    private static List<Pm4OverlayObject> RebasePm4ObjectParts(IReadOnlyList<Pm4OverlayObject> objects, int objectPartOffset)
    {
        if (objects.Count == 0 || objectPartOffset == 0)
            return objects.ToList();

        var rebased = new List<Pm4OverlayObject>(objects.Count);
        for (int i = 0; i < objects.Count; i++)
        {
            Pm4OverlayObject obj = objects[i];
            rebased.Add(new Pm4OverlayObject(
                obj.Ck24,
                obj.Ck24Type,
                obj.ObjectPartId + objectPartOffset,
                obj.LinkGroupObjectId,
                obj.LinkedPositionRefCount,
                obj.Lines,
                obj.Triangles,
                obj.SurfaceCount,
                obj.DominantGroupKey,
                obj.DominantAttributeMask,
                obj.DominantMdosIndex,
                obj.AverageSurfaceHeight,
                obj.PlanarTransform));
        }

        return rebased;
    }

    private readonly struct Pm4IndexedSurface
    {
        public Pm4IndexedSurface(int surfaceIndex, MsurEntry surface)
        {
            SurfaceIndex = surfaceIndex;
            Surface = surface;
        }

        public int SurfaceIndex { get; }
        public MsurEntry Surface { get; }
    }

    private static List<Pm4OverlayObject> BuildPm4TileObjects(
        Pm4File pm4,
        int tileX,
        int tileY,
        bool splitCk24ByMdos,
        bool splitCk24ByConnectivity,
        ref int remainingLineBudget,
        ref int remainingTriangleBudget,
        ref int rejectedLongEdges)
    {
        var objects = new List<Pm4OverlayObject>();
        if (remainingLineBudget <= 0 || pm4.MeshVertices.Count == 0)
            return objects;

        if (!pm4.Surfaces.Any(surface => surface.Ck24 != 0))
            return objects;

        bool fallbackTileLocalCoordinates = IsLikelyTileLocal(pm4.MeshVertices);
        int tileLineBudget = Math.Min(Pm4MaxLinesPerTile, remainingLineBudget);
        int tileTriangleBudget = Math.Min(Pm4MaxTrianglesPerTile, remainingTriangleBudget);

        foreach (var ck24Group in pm4.Surfaces
            .Select((surface, surfaceIndex) => new Pm4IndexedSurface(surfaceIndex, surface))
            .Where(indexedSurface => indexedSurface.Surface.Ck24 != 0)
            .GroupBy(indexedSurface => indexedSurface.Surface.Ck24))
        {
            if (tileLineBudget <= 0)
                break;

            uint ck24 = ck24Group.Key;
            byte ck24Type = (byte)(ck24 >> 16);
            int objectPartCounter = 0;
            List<Pm4IndexedSurface> surfaceGroup = ck24Group.ToList();
            Pm4AxisConvention ck24AxisConvention = DetectPm4AxisConvention(pm4, surfaceGroup.Select(static entry => entry.Surface));
            List<MsurEntry> ck24Surfaces = surfaceGroup.Select(static entry => entry.Surface).ToList();
            List<MprlEntry> ck24PositionRefs = CollectLinkedPositionRefs(pm4, surfaceGroup);
            bool useTileLocalCoordinates = ResolveCk24CoordinateMode(
                pm4,
                ck24Surfaces,
                ck24PositionRefs,
                tileX,
                tileY,
                ck24AxisConvention,
                fallbackTileLocalCoordinates);
            // Keep one shared planar transform per CK24 so split linked/components stay on one coordinate plane.
            Pm4PlanarTransform ck24PlanarTransform = ResolvePlanarTransform(pm4, ck24Surfaces, ck24PositionRefs, tileX, tileY, useTileLocalCoordinates, ck24AxisConvention);
            List<List<Pm4IndexedSurface>> linkedGroups = SplitSurfaceGroupByMslk(pm4, surfaceGroup);

            foreach (List<Pm4IndexedSurface> linkedGroup in linkedGroups)
            {
                if (linkedGroup.Count == 0 || tileLineBudget <= 0)
                    continue;

                uint dominantLinkGroupObjectId = SelectDominantMslkGroupObjectId(pm4, linkedGroup);
                List<MsurEntry> linkedSurfaces = linkedGroup.Select(static entry => entry.Surface).ToList();
                List<MprlEntry> linkedPositionRefs = CollectLinkedPositionRefs(pm4, linkedGroup);
                List<List<MsurEntry>> anchorGroups = splitCk24ByMdos
                    ? SplitSurfaceGroupByMdos(linkedSurfaces)
                    : new List<List<MsurEntry>> { linkedSurfaces };

                foreach (List<MsurEntry> anchorGroup in anchorGroups)
                {
                    List<List<MsurEntry>> components = splitCk24ByConnectivity
                        ? SplitSurfaceGroupByConnectivity(pm4, anchorGroup)
                        : new List<List<MsurEntry>> { anchorGroup };

                    foreach (List<MsurEntry> component in components)
                    {
                        if (tileLineBudget <= 0)
                            break;

                        // Keep split components under one linked group on a shared transform basis.
                        // Per-component transform resolution can explode one real object into offset fragments.
                        List<Pm4LineSegment> lines = BuildCk24ObjectLines(pm4, component, tileX, tileY, useTileLocalCoordinates, ck24AxisConvention, ck24PlanarTransform, tileLineBudget, ref rejectedLongEdges);
                        List<Pm4Triangle> triangles = tileTriangleBudget > 0
                            ? BuildCk24ObjectTriangles(pm4, component, tileX, tileY, useTileLocalCoordinates, ck24AxisConvention, ck24PlanarTransform, tileTriangleBudget)
                            : new List<Pm4Triangle>();
                        if (lines.Count == 0 && triangles.Count == 0)
                            continue;

                        byte dominantGroupKey = SelectDominantSurfaceValue(component, static surface => surface.GroupKey);
                        byte dominantAttributeMask = SelectDominantSurfaceValue(component, static surface => surface.AttributeMask);
                        uint dominantMdosIndex = SelectDominantSurfaceValue(component, static surface => surface.MdosIndex);
                        float averageSurfaceHeight = component.Count > 0 ? component.Average(static surface => surface.Height) : 0f;

                        objects.Add(new Pm4OverlayObject(
                            ck24,
                            ck24Type,
                            objectPartCounter++,
                            dominantLinkGroupObjectId,
                            linkedPositionRefs.Count,
                            lines,
                            triangles,
                            component.Count,
                            dominantGroupKey,
                            dominantAttributeMask,
                            dominantMdosIndex,
                            averageSurfaceHeight,
                            ck24PlanarTransform));

                        tileLineBudget -= lines.Count;
                        tileTriangleBudget -= triangles.Count;
                    }
                }
            }
        }

        int linesUsed = objects.Sum(obj => obj.Lines.Count);
        int trianglesUsed = objects.Sum(obj => obj.Triangles.Count);
        remainingLineBudget -= linesUsed;
        remainingTriangleBudget -= trianglesUsed;
        return objects;
    }

    private static bool ResolveCk24CoordinateMode(
        Pm4File pm4,
        IReadOnlyList<MsurEntry> surfaces,
        IReadOnlyList<MprlEntry> anchorPositionRefs,
        int tileX,
        int tileY,
        Pm4AxisConvention axisConvention,
        bool fallbackTileLocalCoordinates)
    {
        if (surfaces.Count == 0)
            return fallbackTileLocalCoordinates;

        IReadOnlyList<MprlEntry> scoringRefs = anchorPositionRefs.Count > 0
            ? anchorPositionRefs
            : pm4.PositionRefs;
        if (scoringRefs.Count == 0)
            return fallbackTileLocalCoordinates;

        List<Vector3> objectVertices = CollectSurfaceVertices(pm4, surfaces);
        if (objectVertices.Count == 0)
            return fallbackTileLocalCoordinates;

        List<Vector3> sampledObjectVertices = SampleObjectVertices(objectVertices, 192);
        List<Vector2> referencePlanarPoints = BuildMprlPlanarPoints(scoringRefs);
        if (sampledObjectVertices.Count == 0 || referencePlanarPoints.Count == 0)
            return fallbackTileLocalCoordinates;

        float tileLocalScore = EvaluateCoordinateModeScore(
            pm4,
            surfaces,
            scoringRefs,
            sampledObjectVertices,
            referencePlanarPoints,
            tileX,
            tileY,
            axisConvention,
            useTileLocalCoordinates: true);
        float worldSpaceScore = EvaluateCoordinateModeScore(
            pm4,
            surfaces,
            scoringRefs,
            sampledObjectVertices,
            referencePlanarPoints,
            tileX,
            tileY,
            axisConvention,
            useTileLocalCoordinates: false);

        if (!float.IsFinite(tileLocalScore) && !float.IsFinite(worldSpaceScore))
            return fallbackTileLocalCoordinates;
        if (!float.IsFinite(tileLocalScore))
            return false;
        if (!float.IsFinite(worldSpaceScore))
            return true;

        const float decisiveMargin = 512f;
        if (tileLocalScore + decisiveMargin < worldSpaceScore)
            return true;
        if (worldSpaceScore + decisiveMargin < tileLocalScore)
            return false;

        return fallbackTileLocalCoordinates;
    }

    private static float EvaluateCoordinateModeScore(
        Pm4File pm4,
        IReadOnlyList<MsurEntry> surfaces,
        IReadOnlyList<MprlEntry> scoringRefs,
        IReadOnlyList<Vector3> sampledObjectVertices,
        IReadOnlyList<Vector2> referencePlanarPoints,
        int tileX,
        int tileY,
        Pm4AxisConvention axisConvention,
        bool useTileLocalCoordinates)
    {
        Pm4PlanarTransform transform = ResolvePlanarTransform(
            pm4,
            surfaces,
            scoringRefs,
            tileX,
            tileY,
            useTileLocalCoordinates,
            axisConvention);

        float footprintScore = ComputeMprlFootprintScore(
            referencePlanarPoints,
            sampledObjectVertices,
            tileX,
            tileY,
            useTileLocalCoordinates,
            axisConvention,
            transform);

        if (!float.IsFinite(footprintScore))
            return float.MaxValue;

        Vector3 centroid = Vector3.Zero;
        for (int i = 0; i < sampledObjectVertices.Count; i++)
            centroid += sampledObjectVertices[i];
        centroid /= sampledObjectVertices.Count;

        Vector3 centroidWorld = ConvertPm4VertexToWorld(
            centroid,
            tileX,
            tileY,
            useTileLocalCoordinates,
            axisConvention,
            transform);
        float centroidScore = NearestPositionRefDistanceSquared(scoringRefs, centroidWorld);

        return footprintScore * 0.85f + centroidScore * 0.15f;
    }

    private static List<List<Pm4IndexedSurface>> SplitSurfaceGroupByMslk(Pm4File pm4, IReadOnlyList<Pm4IndexedSurface> surfaces)
    {
        var groups = new List<List<Pm4IndexedSurface>>();
        if (surfaces.Count == 0)
            return groups;

        if (surfaces.Count == 1 || pm4.LinkEntries.Count == 0)
        {
            groups.Add(surfaces.ToList());
            return groups;
        }

        var surfaceIndexToLocal = new Dictionary<int, int>(surfaces.Count);
        for (int i = 0; i < surfaces.Count; i++)
            surfaceIndexToLocal[surfaces[i].SurfaceIndex] = i;

        var groupToMembers = new Dictionary<uint, HashSet<int>>();
        int surfaceCount = pm4.Surfaces.Count;
        for (int i = 0; i < pm4.LinkEntries.Count; i++)
        {
            MslkEntry link = pm4.LinkEntries[i];
            if (link.GroupObjectId == 0)
                continue;

            int localMsurIndex = -1;
            if (link.MsurIndex < (uint)surfaceCount && surfaceIndexToLocal.TryGetValue((int)link.MsurIndex, out int msurLocal))
                localMsurIndex = msurLocal;

            int localRefIndex = -1;
            if (link.RefIndex < surfaceCount && surfaceIndexToLocal.TryGetValue(link.RefIndex, out int refLocal))
                localRefIndex = refLocal;

            if (localMsurIndex >= 0)
                localRefIndex = -1;

            if (localRefIndex < 0 && localMsurIndex < 0)
                continue;

            if (!groupToMembers.TryGetValue(link.GroupObjectId, out HashSet<int>? members))
            {
                members = new HashSet<int>();
                groupToMembers[link.GroupObjectId] = members;
            }

            if (localRefIndex >= 0)
                members.Add(localRefIndex);
            if (localMsurIndex >= 0)
                members.Add(localMsurIndex);
        }

        if (groupToMembers.Count == 0)
        {
            groups.Add(surfaces.ToList());
            return groups;
        }

        int[] parent = new int[surfaces.Count];
        for (int i = 0; i < parent.Length; i++)
            parent[i] = i;

        static int Find(int[] parentArray, int index)
        {
            while (parentArray[index] != index)
            {
                parentArray[index] = parentArray[parentArray[index]];
                index = parentArray[index];
            }

            return index;
        }

        static void Union(int[] parentArray, int a, int b)
        {
            int rootA = Find(parentArray, a);
            int rootB = Find(parentArray, b);
            if (rootA != rootB)
                parentArray[rootB] = rootA;
        }

        var linkedLocalIndices = new HashSet<int>();
        foreach (HashSet<int> members in groupToMembers.Values)
        {
            if (members.Count < 2)
                continue;

            int first = members.First();
            linkedLocalIndices.Add(first);
            foreach (int member in members)
            {
                linkedLocalIndices.Add(member);
                Union(parent, first, member);
            }
        }

        if (linkedLocalIndices.Count < 2)
        {
            groups.Add(surfaces.ToList());
            return groups;
        }

        var linkedComponents = new Dictionary<int, List<Pm4IndexedSurface>>();
        for (int i = 0; i < surfaces.Count; i++)
        {
            if (!linkedLocalIndices.Contains(i))
                continue;

            int root = Find(parent, i);
            if (!linkedComponents.TryGetValue(root, out List<Pm4IndexedSurface>? component))
            {
                component = new List<Pm4IndexedSurface>();
                linkedComponents[root] = component;
            }

            component.Add(surfaces[i]);
        }

        if (linkedComponents.Count <= 1)
        {
            groups.Add(surfaces.ToList());
            return groups;
        }

        foreach (List<Pm4IndexedSurface> component in linkedComponents.Values.OrderBy(component => component.Min(entry => entry.SurfaceIndex)))
            groups.Add(component);

        var unlinked = new List<Pm4IndexedSurface>();
        for (int i = 0; i < surfaces.Count; i++)
        {
            if (!linkedLocalIndices.Contains(i))
                unlinked.Add(surfaces[i]);
        }

        if (unlinked.Count > 0)
            groups.Add(unlinked);

        return groups;
    }

    private static uint SelectDominantMslkGroupObjectId(Pm4File pm4, IReadOnlyList<Pm4IndexedSurface> surfaces)
    {
        if (surfaces.Count == 0 || pm4.LinkEntries.Count == 0)
            return 0;

        int surfaceCount = pm4.Surfaces.Count;
        var surfaceIndices = new HashSet<int>(surfaces.Select(static surface => surface.SurfaceIndex));
        var counts = new Dictionary<uint, int>();

        uint bestGroupObjectId = 0;
        int bestCount = 0;
        for (int i = 0; i < pm4.LinkEntries.Count; i++)
        {
            MslkEntry link = pm4.LinkEntries[i];
            if (link.GroupObjectId == 0)
                continue;

            if (!LinkReferencesSurface(link, surfaceIndices, surfaceCount))
                continue;

            int nextCount = 1;
            if (counts.TryGetValue(link.GroupObjectId, out int existingCount))
                nextCount = existingCount + 1;
            counts[link.GroupObjectId] = nextCount;

            if (nextCount > bestCount)
            {
                bestCount = nextCount;
                bestGroupObjectId = link.GroupObjectId;
            }
        }

        return bestGroupObjectId;
    }

    private static List<MprlEntry> CollectLinkedPositionRefs(Pm4File pm4, IReadOnlyList<Pm4IndexedSurface> surfaces)
    {
        var refs = new List<MprlEntry>();
        if (surfaces.Count == 0 || pm4.LinkEntries.Count == 0 || pm4.PositionRefs.Count == 0)
            return refs;

        int surfaceCount = pm4.Surfaces.Count;
        var surfaceIndices = new HashSet<int>(surfaces.Select(static surface => surface.SurfaceIndex));
        var seenRefIndices = new HashSet<int>();

        for (int i = 0; i < pm4.LinkEntries.Count; i++)
        {
            MslkEntry link = pm4.LinkEntries[i];
            if ((uint)link.RefIndex >= (uint)pm4.PositionRefs.Count)
                continue;

            if (!LinkReferencesSurface(link, surfaceIndices, surfaceCount))
                continue;

            if (!seenRefIndices.Add(link.RefIndex))
                continue;

            refs.Add(pm4.PositionRefs[link.RefIndex]);
        }

        return refs;
    }

    private static bool LinkReferencesSurface(MslkEntry link, HashSet<int> surfaceIndices, int surfaceCount)
    {
        if (link.MsurIndex < (uint)surfaceCount && surfaceIndices.Contains((int)link.MsurIndex))
            return true;

        // Fallback for variants where RefIndex stores the local surface id.
        if (link.RefIndex < surfaceCount && surfaceIndices.Contains(link.RefIndex))
            return true;

        return false;
    }

    private static bool TryComputeExpectedMprlYawRadians(IReadOnlyList<MprlEntry> positionRefs, out float yawRadians)
    {
        yawRadians = 0f;
        if (positionRefs.Count == 0)
            return false;

        double sumSin = 0d;
        double sumCos = 0d;
        int count = 0;
        for (int i = 0; i < positionRefs.Count; i++)
        {
            // Runtime evidence shows PM4 objects are consistently rotated 90 degrees clockwise
            // when using the raw packed MPRL low-16 angle directly. Treat the packed value as
            // clockwise and rebase by +90 degrees to align with world yaw comparison.
            float rawAngle = (positionRefs[i].RotationOrFlags & 0xFFFF) * (2f * MathF.PI / 65536f);
            float angleRadians = (MathF.PI * 0.5f) - rawAngle;
            sumSin += Math.Sin(angleRadians);
            sumCos += Math.Cos(angleRadians);
            count++;
        }

        if (count == 0)
            return false;

        double length = Math.Sqrt(sumSin * sumSin + sumCos * sumCos);
        if (length < 1e-4)
            return false;

        yawRadians = (float)Math.Atan2(sumSin, sumCos);
        return true;
    }

    private static bool TryComputePlanarPrincipalYaw(
        IReadOnlyList<Vector3> objectVertices,
        int tileX,
        int tileY,
        bool useTileLocalCoordinates,
        Pm4AxisConvention axisConvention,
        Pm4PlanarTransform planarTransform,
        out float yawRadians)
    {
        yawRadians = 0f;
        if (objectVertices.Count < 3)
            return false;

        int sampleCount = Math.Min(512, objectVertices.Count);
        int stride = Math.Max(1, objectVertices.Count / sampleCount);
        double meanX = 0d;
        double meanY = 0d;
        int used = 0;

        for (int i = 0; i < objectVertices.Count; i += stride)
        {
            Vector3 world = ConvertPm4VertexToWorld(objectVertices[i], tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform);
            meanX += world.X;
            meanY += world.Y;
            used++;
        }

        if (used < 3)
            return false;

        meanX /= used;
        meanY /= used;

        double covXX = 0d;
        double covYY = 0d;
        double covXY = 0d;
        for (int i = 0; i < objectVertices.Count; i += stride)
        {
            Vector3 world = ConvertPm4VertexToWorld(objectVertices[i], tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform);
            double dx = world.X - meanX;
            double dy = world.Y - meanY;
            covXX += dx * dx;
            covYY += dy * dy;
            covXY += dx * dy;
        }

        if (covXX + covYY < 1e-4)
            return false;

        yawRadians = 0.5f * (float)Math.Atan2(2.0 * covXY, covXX - covYY);
        return true;
    }

    private static float ComputeUndirectedAngleDelta(float a, float b)
    {
        float delta = MathF.Abs(a - b);
        while (delta > MathF.PI)
            delta -= 2f * MathF.PI;
        delta = MathF.Abs(delta);
        if (delta > MathF.PI * 0.5f)
            delta = MathF.PI - delta;

        return MathF.Abs(delta);
    }

    private static float ComputeMprlYawDeltaWithQuarterTurnFallback(float candidateYaw, float expectedYaw)
    {
        float bestDelta = ComputeUndirectedAngleDelta(candidateYaw, expectedYaw);
        bestDelta = MathF.Min(bestDelta, ComputeUndirectedAngleDelta(candidateYaw, -expectedYaw));

        const float quarterTurn = MathF.PI * 0.5f;
        bestDelta = MathF.Min(bestDelta, ComputeUndirectedAngleDelta(candidateYaw, expectedYaw + quarterTurn));
        bestDelta = MathF.Min(bestDelta, ComputeUndirectedAngleDelta(candidateYaw, expectedYaw - quarterTurn));
        bestDelta = MathF.Min(bestDelta, ComputeUndirectedAngleDelta(candidateYaw, -expectedYaw + quarterTurn));
        bestDelta = MathF.Min(bestDelta, ComputeUndirectedAngleDelta(candidateYaw, -expectedYaw - quarterTurn));

        return bestDelta;
    }

    private static List<Vector3> SampleObjectVertices(IReadOnlyList<Vector3> objectVertices, int maxSamples)
    {
        var sampled = new List<Vector3>();
        if (objectVertices.Count == 0)
            return sampled;

        int sampleCount = Math.Min(maxSamples, objectVertices.Count);
        int stride = Math.Max(1, objectVertices.Count / sampleCount);
        for (int i = 0; i < objectVertices.Count; i += stride)
            sampled.Add(objectVertices[i]);

        if (sampled.Count == 0)
            sampled.Add(objectVertices[0]);

        return sampled;
    }

    private static List<Vector2> BuildMprlPlanarPoints(IReadOnlyList<MprlEntry> positionRefs)
    {
        var points = new List<Vector2>(positionRefs.Count);
        for (int i = 0; i < positionRefs.Count; i++)
        {
            Vector3 refPos = positionRefs[i].Position;
            points.Add(new Vector2(refPos.X, refPos.Z));
        }

        return points;
    }

    private static float NearestDistanceSquared(IReadOnlyList<Vector2> points, in Vector2 target)
    {
        float best = float.MaxValue;
        for (int i = 0; i < points.Count; i++)
        {
            Vector2 delta = points[i] - target;
            float distSq = delta.LengthSquared();
            if (distSq < best)
                best = distSq;
        }

        return best;
    }

    private static float ComputeMprlFootprintScore(
        IReadOnlyList<Vector2> referencePoints,
        IReadOnlyList<Vector3> sampledVertices,
        int tileX,
        int tileY,
        bool useTileLocalCoordinates,
        Pm4AxisConvention axisConvention,
        Pm4PlanarTransform candidate)
    {
        if (referencePoints.Count == 0 || sampledVertices.Count == 0)
            return float.MaxValue;

        var candidatePoints = new List<Vector2>(sampledVertices.Count);
        for (int i = 0; i < sampledVertices.Count; i++)
        {
            Vector3 world = ConvertPm4VertexToWorld(sampledVertices[i], tileX, tileY, useTileLocalCoordinates, axisConvention, candidate);
            candidatePoints.Add(new Vector2(world.X, world.Y));
        }

        float sumObjectToRef = 0f;
        for (int i = 0; i < candidatePoints.Count; i++)
            sumObjectToRef += NearestDistanceSquared(referencePoints, candidatePoints[i]);

        float sumRefToObject = 0f;
        for (int i = 0; i < referencePoints.Count; i++)
            sumRefToObject += NearestDistanceSquared(candidatePoints, referencePoints[i]);

        float avgObjectToRef = sumObjectToRef / Math.Max(1, candidatePoints.Count);
        float avgRefToObject = sumRefToObject / Math.Max(1, referencePoints.Count);
        return avgObjectToRef + avgRefToObject;
    }

    private static List<Pm4LineSegment> BuildCk24ObjectLines(
        Pm4File pm4,
        IReadOnlyList<MsurEntry> surfaces,
        int tileX,
        int tileY,
        bool useTileLocalCoordinates,
        Pm4AxisConvention axisConvention,
        Pm4PlanarTransform planarTransform,
        int lineBudget,
        ref int rejectedLongEdges)
    {
        var lines = new List<Pm4LineSegment>();
        var uniqueEdges = new HashSet<ulong>();

        for (int s = 0; s < surfaces.Count; s++)
        {
            MsurEntry surface = surfaces[s];
            if (lines.Count >= lineBudget)
                break;

            int firstIndex = (int)surface.MsviFirstIndex;
            int surfaceIndexCount = surface.IndexCount;
            if (surfaceIndexCount < 2 || firstIndex < 0 || firstIndex >= pm4.MeshIndices.Count)
                continue;

            int endExclusive = Math.Min(firstIndex + surfaceIndexCount, pm4.MeshIndices.Count);
            if (endExclusive - firstIndex < 2)
                continue;

            int prevVertex = (int)pm4.MeshIndices[firstIndex];
            if ((uint)prevVertex >= (uint)pm4.MeshVertices.Count)
                continue;

            for (int idx = firstIndex + 1; idx < endExclusive && lines.Count < lineBudget; idx++)
            {
                int nextVertex = (int)pm4.MeshIndices[idx];
                if ((uint)nextVertex >= (uint)pm4.MeshVertices.Count)
                    continue;

                AddUniqueEdge(pm4, prevVertex, nextVertex, tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform, uniqueEdges, lines, lineBudget, ref rejectedLongEdges);
                prevVertex = nextVertex;
            }

            // Close each surface loop so CK24 objects stay visually self-contained.
            if (lines.Count < lineBudget)
            {
                int firstVertex = (int)pm4.MeshIndices[firstIndex];
                int lastVertex = (int)pm4.MeshIndices[endExclusive - 1];
                if ((uint)firstVertex < (uint)pm4.MeshVertices.Count && (uint)lastVertex < (uint)pm4.MeshVertices.Count)
                    AddUniqueEdge(pm4, lastVertex, firstVertex, tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform, uniqueEdges, lines, lineBudget, ref rejectedLongEdges);
            }
        }

        return lines;
    }

    private static List<Pm4Triangle> BuildCk24ObjectTriangles(
        Pm4File pm4,
        IReadOnlyList<MsurEntry> surfaces,
        int tileX,
        int tileY,
        bool useTileLocalCoordinates,
        Pm4AxisConvention axisConvention,
        Pm4PlanarTransform planarTransform,
        int triangleBudget)
    {
        var triangles = new List<Pm4Triangle>();

        for (int s = 0; s < surfaces.Count; s++)
        {
            MsurEntry surface = surfaces[s];
            if (triangles.Count >= triangleBudget)
                break;

            int firstIndex = (int)surface.MsviFirstIndex;
            int surfaceIndexCount = surface.IndexCount;
            if (surfaceIndexCount < 3 || firstIndex < 0 || firstIndex >= pm4.MeshIndices.Count)
                continue;

            int endExclusive = Math.Min(firstIndex + surfaceIndexCount, pm4.MeshIndices.Count);
            int indexCount = endExclusive - firstIndex;
            if (indexCount < 3)
                continue;

            // Most PM4 surfaces are listed as loops; use a fan from the first vertex.
            int i0 = (int)pm4.MeshIndices[firstIndex];
            if ((uint)i0 >= (uint)pm4.MeshVertices.Count)
                continue;

            Vector3 v0 = ConvertPm4VertexToRenderer(pm4.MeshVertices[i0], tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform);
            for (int idx = firstIndex + 1; idx + 1 < endExclusive && triangles.Count < triangleBudget; idx++)
            {
                int i1 = (int)pm4.MeshIndices[idx];
                int i2 = (int)pm4.MeshIndices[idx + 1];
                if ((uint)i1 >= (uint)pm4.MeshVertices.Count || (uint)i2 >= (uint)pm4.MeshVertices.Count)
                    continue;

                Vector3 v1 = ConvertPm4VertexToRenderer(pm4.MeshVertices[i1], tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform);
                Vector3 v2 = ConvertPm4VertexToRenderer(pm4.MeshVertices[i2], tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform);
                triangles.Add(planarTransform.InvertsWinding
                    ? new Pm4Triangle(v0, v2, v1)
                    : new Pm4Triangle(v0, v1, v2));
            }
        }

        return triangles;
    }

    private static List<Pm4LineSegment> BuildFallbackMeshLines(
        Pm4File pm4,
        int tileX,
        int tileY,
        bool useTileLocalCoordinates,
        Pm4AxisConvention axisConvention,
        Pm4PlanarTransform planarTransform,
        int lineBudget,
        ref int rejectedLongEdges)
    {
        var lines = new List<Pm4LineSegment>();
        var uniqueEdges = new HashSet<ulong>();

        for (int i = 0; i + 2 < pm4.MeshIndices.Count && lines.Count < lineBudget; i += 3)
        {
            int i0 = (int)pm4.MeshIndices[i];
            int i1 = (int)pm4.MeshIndices[i + 1];
            int i2 = (int)pm4.MeshIndices[i + 2];

            if ((uint)i0 >= (uint)pm4.MeshVertices.Count ||
                (uint)i1 >= (uint)pm4.MeshVertices.Count ||
                (uint)i2 >= (uint)pm4.MeshVertices.Count)
                continue;

            AddUniqueEdge(pm4, i0, i1, tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform, uniqueEdges, lines, lineBudget, ref rejectedLongEdges);
            AddUniqueEdge(pm4, i1, i2, tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform, uniqueEdges, lines, lineBudget, ref rejectedLongEdges);
            AddUniqueEdge(pm4, i2, i0, tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform, uniqueEdges, lines, lineBudget, ref rejectedLongEdges);
        }

        return lines;
    }

    private static List<List<MsurEntry>> SplitSurfaceGroupByConnectivity(Pm4File pm4, IReadOnlyList<MsurEntry> surfaces)
    {
        var components = new List<List<MsurEntry>>();
        if (surfaces.Count == 0)
            return components;
        if (surfaces.Count == 1)
        {
            components.Add(new List<MsurEntry> { surfaces[0] });
            return components;
        }

        var surfaceVertices = new List<List<int>>(surfaces.Count);
        var vertexToSurfaceIndices = new Dictionary<int, List<int>>();

        for (int s = 0; s < surfaces.Count; s++)
        {
            MsurEntry surface = surfaces[s];
            int firstIndex = (int)surface.MsviFirstIndex;
            int endExclusive = Math.Min(firstIndex + surface.IndexCount, pm4.MeshIndices.Count);
            var vertices = new List<int>();
            var unique = new HashSet<int>();

            if (surface.IndexCount > 0 && firstIndex >= 0 && endExclusive > firstIndex)
            {
                for (int idx = firstIndex; idx < endExclusive; idx++)
                {
                    int vertexIndex = (int)pm4.MeshIndices[idx];
                    if ((uint)vertexIndex >= (uint)pm4.MeshVertices.Count)
                        continue;
                    if (!unique.Add(vertexIndex))
                        continue;

                    vertices.Add(vertexIndex);
                    if (!vertexToSurfaceIndices.TryGetValue(vertexIndex, out List<int>? owners))
                    {
                        owners = new List<int>();
                        vertexToSurfaceIndices[vertexIndex] = owners;
                    }
                    owners.Add(s);
                }
            }

            surfaceVertices.Add(vertices);
        }

        var visited = new bool[surfaces.Count];
        var queue = new Queue<int>();
        for (int start = 0; start < surfaces.Count; start++)
        {
            if (visited[start])
                continue;

            visited[start] = true;
            queue.Enqueue(start);
            var component = new List<MsurEntry>();

            while (queue.Count > 0)
            {
                int current = queue.Dequeue();
                component.Add(surfaces[current]);

                List<int> vertices = surfaceVertices[current];
                for (int v = 0; v < vertices.Count; v++)
                {
                    int vertexIndex = vertices[v];
                    if (!vertexToSurfaceIndices.TryGetValue(vertexIndex, out List<int>? neighbors))
                        continue;

                    for (int n = 0; n < neighbors.Count; n++)
                    {
                        int neighborSurface = neighbors[n];
                        if (visited[neighborSurface])
                            continue;

                        visited[neighborSurface] = true;
                        queue.Enqueue(neighborSurface);
                    }
                }
            }

            components.Add(component);
        }

        return components;
    }

    private static List<List<MsurEntry>> SplitSurfaceGroupByMdos(IReadOnlyList<MsurEntry> surfaces)
    {
        if (surfaces.Count <= 1)
            return new List<List<MsurEntry>> { surfaces.ToList() };

        var groups = surfaces
            .GroupBy(static surface => surface.MdosIndex)
            .Select(static group => group.ToList())
            .Where(static group => group.Count > 0)
            .ToList();

        return groups.Count > 0 ? groups : new List<List<MsurEntry>> { surfaces.ToList() };
    }

    private static byte SelectDominantSurfaceValue(IReadOnlyList<MsurEntry> surfaces, Func<MsurEntry, byte> selector)
    {
        if (surfaces.Count == 0)
            return 0;

        Span<int> counts = stackalloc int[256];
        for (int i = 0; i < surfaces.Count; i++)
            counts[selector(surfaces[i])]++;

        int bestCount = -1;
        byte bestValue = 0;
        for (int i = 0; i < counts.Length; i++)
        {
            int count = counts[i];
            if (count <= bestCount)
                continue;

            bestCount = count;
            bestValue = (byte)i;
        }

        return bestValue;
    }

    private static uint SelectDominantSurfaceValue(IReadOnlyList<MsurEntry> surfaces, Func<MsurEntry, uint> selector)
    {
        if (surfaces.Count == 0)
            return 0;

        var counts = new Dictionary<uint, int>();
        uint bestValue = 0;
        int bestCount = -1;

        for (int i = 0; i < surfaces.Count; i++)
        {
            uint value = selector(surfaces[i]);
            int count = 1;
            if (counts.TryGetValue(value, out int existing))
                count = existing + 1;
            counts[value] = count;

            if (count > bestCount)
            {
                bestCount = count;
                bestValue = value;
            }
        }

        return bestValue;
    }

    private static List<Vector3> BuildPm4PositionRefMarkers(Pm4File pm4, int limit)
    {
        var markers = new List<Vector3>();
        int count = Math.Min(limit, pm4.PositionRefs.Count);
        for (int i = 0; i < count; i++)
        {
            Vector3 refPos = pm4.PositionRefs[i].Position;
            // MPRL uses ADT placement: X/Z planar with Y vertical.
            Vector3 world = new(refPos.X, refPos.Z, refPos.Y);
            markers.Add(new Vector3(
                WoWConstants.MapOrigin - world.Y,
                WoWConstants.MapOrigin - world.X,
                world.Z + 0.5f));
        }

        return markers;
    }

    private static List<Pm4Triangle> BuildFallbackMeshTriangles(
        Pm4File pm4,
        int tileX,
        int tileY,
        bool useTileLocalCoordinates,
        Pm4AxisConvention axisConvention,
        Pm4PlanarTransform planarTransform,
        int triangleBudget)
    {
        var triangles = new List<Pm4Triangle>();

        for (int i = 0; i + 2 < pm4.MeshIndices.Count && triangles.Count < triangleBudget; i += 3)
        {
            int i0 = (int)pm4.MeshIndices[i];
            int i1 = (int)pm4.MeshIndices[i + 1];
            int i2 = (int)pm4.MeshIndices[i + 2];

            if ((uint)i0 >= (uint)pm4.MeshVertices.Count ||
                (uint)i1 >= (uint)pm4.MeshVertices.Count ||
                (uint)i2 >= (uint)pm4.MeshVertices.Count)
                continue;

            Vector3 v0 = ConvertPm4VertexToRenderer(pm4.MeshVertices[i0], tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform);
            Vector3 v1 = ConvertPm4VertexToRenderer(pm4.MeshVertices[i1], tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform);
            Vector3 v2 = ConvertPm4VertexToRenderer(pm4.MeshVertices[i2], tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform);
            triangles.Add(planarTransform.InvertsWinding
                ? new Pm4Triangle(v0, v2, v1)
                : new Pm4Triangle(v0, v1, v2));
        }

        return triangles;
    }

    private static void AddUniqueEdge(Pm4File pm4, int ia, int ib,
        int tileX, int tileY, bool useTileLocalCoordinates, Pm4AxisConvention axisConvention, Pm4PlanarTransform planarTransform,
        HashSet<ulong> uniqueEdges, List<Pm4LineSegment> lines, int tileLineBudget,
        ref int rejectedLongEdges)
    {
        if (ia == ib || lines.Count >= tileLineBudget)
            return;

        ulong key = PackEdgeKey(ia, ib);
        if (!uniqueEdges.Add(key))
            return;

        Vector3 from = ConvertPm4VertexToRenderer(pm4.MeshVertices[ia], tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform);
        Vector3 to = ConvertPm4VertexToRenderer(pm4.MeshVertices[ib], tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform);

        if (Vector3.DistanceSquared(from, to) > Pm4MaxEdgeLength * Pm4MaxEdgeLength)
        {
            rejectedLongEdges++;
            return;
        }

        lines.Add(new Pm4LineSegment(from, to));
    }

    private static ulong PackEdgeKey(int ia, int ib)
    {
        uint lo = ia < ib ? (uint)ia : (uint)ib;
        uint hi = ia < ib ? (uint)ib : (uint)ia;
        return ((ulong)lo << 32) | hi;
    }

    private enum Pm4AxisConvention
    {
        XZPlaneYUp,
        XYPlaneZUp,
        YZPlaneXUp
    }

    private static Pm4AxisConvention DetectPm4AxisConvention(Pm4File pm4)
    {
        // Pick the basis that yields the most horizontal (floor-like) triangles.
        // This avoids forcing users to manually undo a 90-degree wall orientation.
        var candidates = new[]
        {
            Pm4AxisConvention.XZPlaneYUp,
            Pm4AxisConvention.XYPlaneZUp,
            Pm4AxisConvention.YZPlaneXUp
        };

        Pm4AxisConvention bestConvention = Pm4AxisConvention.XYPlaneZUp;
        float bestScore = float.MinValue;
        foreach (Pm4AxisConvention candidate in candidates)
        {
            float score = ScoreAxisConventionByTriangleNormals(pm4, candidate);
            if (score > bestScore)
            {
                bestScore = score;
                bestConvention = candidate;
            }
        }

        if (bestScore > 0f)
            return bestConvention;

        return DetectAxisConventionByRanges(pm4.MeshVertices);
    }

    private static Pm4AxisConvention DetectPm4AxisConvention(Pm4File pm4, IEnumerable<MsurEntry> surfaces)
    {
        var surfaceList = surfaces as List<MsurEntry> ?? surfaces.ToList();
        if (surfaceList.Count == 0)
            return DetectPm4AxisConvention(pm4);

        var candidates = new[]
        {
            Pm4AxisConvention.XZPlaneYUp,
            Pm4AxisConvention.XYPlaneZUp,
            Pm4AxisConvention.YZPlaneXUp
        };

        Pm4AxisConvention bestConvention = Pm4AxisConvention.XYPlaneZUp;
        float bestScore = float.MinValue;
        foreach (Pm4AxisConvention candidate in candidates)
        {
            float score = ScoreAxisConventionBySurfaceNormals(pm4, surfaceList, candidate);
            if (score > bestScore)
            {
                bestScore = score;
                bestConvention = candidate;
            }
        }

        if (bestScore > 0f)
            return bestConvention;

        List<Vector3> groupVertices = CollectSurfaceVertices(pm4, surfaceList);
        return groupVertices.Count > 0
            ? DetectAxisConventionByRanges(groupVertices)
            : DetectPm4AxisConvention(pm4);
    }

    private static Pm4PlanarTransform ResolvePlanarTransform(
        Pm4File pm4,
        IEnumerable<MsurEntry> surfaces,
        IReadOnlyList<MprlEntry>? anchorPositionRefs,
        int tileX,
        int tileY,
        bool useTileLocalCoordinates,
        Pm4AxisConvention axisConvention)
    {
        Pm4PlanarTransform defaultTransform = DefaultPlanarTransform(useTileLocalCoordinates);
        if (pm4.PositionRefs.Count == 0)
            return defaultTransform;

        var surfaceList = surfaces as List<MsurEntry> ?? surfaces.ToList();
        if (surfaceList.Count == 0)
            return defaultTransform;

        List<Vector3> objectVertices = CollectSurfaceVertices(pm4, surfaceList);
        if (objectVertices.Count == 0)
            return defaultTransform;

        Vector3 centroid = Vector3.Zero;
        for (int i = 0; i < objectVertices.Count; i++)
            centroid += objectVertices[i];
        centroid /= objectVertices.Count;

        IReadOnlyList<MprlEntry> scoringRefs = (anchorPositionRefs != null && anchorPositionRefs.Count > 0)
            ? anchorPositionRefs
            : pm4.PositionRefs;
        bool useFootprintScoring = anchorPositionRefs != null && anchorPositionRefs.Count >= 2;
        List<Vector3> sampledObjectVertices = useFootprintScoring
            ? SampleObjectVertices(objectVertices, 256)
            : new List<Vector3>();
        List<Vector2> referencePlanarPoints = useFootprintScoring
            ? BuildMprlPlanarPoints(scoringRefs)
            : new List<Vector2>();
        bool hasExpectedYaw = TryComputeExpectedMprlYawRadians(scoringRefs, out float expectedYaw);

        Pm4PlanarTransform bestTransform = defaultTransform;
        float bestScore = float.MaxValue;
        float bestYawDelta = float.MaxValue;
        foreach (Pm4PlanarTransform candidate in EnumeratePlanarTransforms(useTileLocalCoordinates))
        {
            Vector3 candidateWorld = ConvertPm4VertexToWorld(centroid, tileX, tileY, useTileLocalCoordinates, axisConvention, candidate);
            float centroidScore = NearestPositionRefDistanceSquared(scoringRefs, candidateWorld);
            float score = centroidScore;

            if (useFootprintScoring)
            {
                float footprintScore = ComputeMprlFootprintScore(
                    referencePlanarPoints,
                    sampledObjectVertices,
                    tileX,
                    tileY,
                    useTileLocalCoordinates,
                    axisConvention,
                    candidate);
                if (float.IsFinite(footprintScore))
                    score = footprintScore * 0.85f + centroidScore * 0.15f;
            }
            float yawDelta = float.MaxValue;

            if (hasExpectedYaw
                && TryComputePlanarPrincipalYaw(objectVertices, tileX, tileY, useTileLocalCoordinates, axisConvention, candidate, out float candidateYaw))
            {
                yawDelta = ComputeMprlYawDeltaWithQuarterTurnFallback(candidateYaw, expectedYaw);
            }

            // Prefer transforms that do not invert winding when scores are close.
            if (candidate.InvertsWinding)
                score += 0.5f;

            bool isBetterDistance = score < bestScore - 0.001f;
            float tieDistanceThreshold = useFootprintScoring ? 256f : 4096f;
            bool isNearDistance = MathF.Abs(score - bestScore) <= tieDistanceThreshold;
            bool isBetterYaw = yawDelta + 0.01f < bestYawDelta;
            bool yawCanOverrideDistance = useFootprintScoring
                && hasExpectedYaw
                && yawDelta + 0.02f < bestYawDelta
                && score <= bestScore + 1024f;

            if (isBetterDistance || (isNearDistance && isBetterYaw) || yawCanOverrideDistance)
            {
                bestScore = score;
                bestYawDelta = yawDelta;
                bestTransform = candidate;
            }
        }

        return bestTransform;
    }

    private static Pm4PlanarTransform DefaultPlanarTransform(bool useTileLocalCoordinates)
    {
        return useTileLocalCoordinates
            ? new Pm4PlanarTransform(false, true, true)
            : new Pm4PlanarTransform(false, false, false);
    }

    private static IEnumerable<Pm4PlanarTransform> EnumeratePlanarTransforms(bool useTileLocalCoordinates)
    {
        // For tile-local PM4 data, evaluate all swap/axis-inversion combinations.
        // For world-space PM4 data, keep inversion disabled and only test swap parity.
        if (!useTileLocalCoordinates)
        {
            yield return new Pm4PlanarTransform(false, false, false);
            yield return new Pm4PlanarTransform(true, false, false);
            yield break;
        }

        for (int swap = 0; swap <= 1; swap++)
        {
            for (int invertU = 0; invertU <= 1; invertU++)
            {
                for (int invertV = 0; invertV <= 1; invertV++)
                {
                    yield return new Pm4PlanarTransform(
                        swap == 1,
                        invertU == 1,
                        invertV == 1);
                }
            }
        }
    }

    private static float NearestPositionRefDistanceSquared(IReadOnlyList<MprlEntry> positionRefs, Vector3 world)
    {
        float best = float.MaxValue;
        for (int i = 0; i < positionRefs.Count; i++)
        {
            Vector3 refPos = positionRefs[i].Position;
            // MPRL uses ADT-style placement: X/Z planar with Y vertical.
            Vector3 refWorld = new(refPos.X, refPos.Z, refPos.Y);
            float dx = refWorld.X - world.X;
            float dy = refWorld.Y - world.Y;
            float distSq = dx * dx + dy * dy;
            if (distSq < best)
                best = distSq;
        }

        return best;
    }

    private static float ScoreAxisConventionByTriangleNormals(Pm4File pm4, Pm4AxisConvention convention)
    {
        if (pm4.MeshVertices.Count == 0 || pm4.MeshIndices.Count < 3)
            return 0f;

        float sum = 0f;
        int samples = 0;
        const int maxSamples = 1024;

        for (int i = 0; i + 2 < pm4.MeshIndices.Count && samples < maxSamples; i += 3)
        {
            int i0 = (int)pm4.MeshIndices[i];
            int i1 = (int)pm4.MeshIndices[i + 1];
            int i2 = (int)pm4.MeshIndices[i + 2];
            if ((uint)i0 >= (uint)pm4.MeshVertices.Count ||
                (uint)i1 >= (uint)pm4.MeshVertices.Count ||
                (uint)i2 >= (uint)pm4.MeshVertices.Count)
                continue;

            Vector3 a = ConvertPm4VertexToWorld(pm4.MeshVertices[i0], 0, 0, false, convention, DefaultPlanarTransform(false));
            Vector3 b = ConvertPm4VertexToWorld(pm4.MeshVertices[i1], 0, 0, false, convention, DefaultPlanarTransform(false));
            Vector3 c = ConvertPm4VertexToWorld(pm4.MeshVertices[i2], 0, 0, false, convention, DefaultPlanarTransform(false));

            Vector3 normal = Vector3.Cross(b - a, c - a);
            float length = normal.Length();
            if (length < 1e-5f)
                continue;

            // Higher |normal.Z| means more floor-like orientation in this renderer.
            sum += MathF.Abs(normal.Z / length);
            samples++;
        }

        return samples > 0 ? sum / samples : 0f;
    }

    private static float ScoreAxisConventionBySurfaceNormals(Pm4File pm4, IReadOnlyList<MsurEntry> surfaces, Pm4AxisConvention convention)
    {
        if (pm4.MeshVertices.Count == 0 || pm4.MeshIndices.Count < 3 || surfaces.Count == 0)
            return 0f;

        float sum = 0f;
        int samples = 0;
        const int maxSamples = 1024;

        for (int s = 0; s < surfaces.Count && samples < maxSamples; s++)
        {
            MsurEntry surface = surfaces[s];
            int firstIndex = (int)surface.MsviFirstIndex;
            int endExclusive = Math.Min(firstIndex + surface.IndexCount, pm4.MeshIndices.Count);
            if (surface.IndexCount < 3 || firstIndex < 0 || endExclusive - firstIndex < 3)
                continue;

            int i0 = (int)pm4.MeshIndices[firstIndex];
            if ((uint)i0 >= (uint)pm4.MeshVertices.Count)
                continue;

            Vector3 a = ConvertPm4VertexToWorld(pm4.MeshVertices[i0], 0, 0, false, convention, DefaultPlanarTransform(false));
            for (int idx = firstIndex + 1; idx + 1 < endExclusive && samples < maxSamples; idx++)
            {
                int i1 = (int)pm4.MeshIndices[idx];
                int i2 = (int)pm4.MeshIndices[idx + 1];
                if ((uint)i1 >= (uint)pm4.MeshVertices.Count || (uint)i2 >= (uint)pm4.MeshVertices.Count)
                    continue;

                Vector3 b = ConvertPm4VertexToWorld(pm4.MeshVertices[i1], 0, 0, false, convention, DefaultPlanarTransform(false));
                Vector3 c = ConvertPm4VertexToWorld(pm4.MeshVertices[i2], 0, 0, false, convention, DefaultPlanarTransform(false));

                Vector3 normal = Vector3.Cross(b - a, c - a);
                float length = normal.Length();
                if (length < 1e-5f)
                    continue;

                sum += MathF.Abs(normal.Z / length);
                samples++;
            }
        }

        return samples > 0 ? sum / samples : 0f;
    }

    private static List<Vector3> CollectSurfaceVertices(Pm4File pm4, IReadOnlyList<MsurEntry> surfaces)
    {
        var vertices = new List<Vector3>();
        var seen = new HashSet<int>();

        for (int s = 0; s < surfaces.Count; s++)
        {
            MsurEntry surface = surfaces[s];
            int firstIndex = (int)surface.MsviFirstIndex;
            int endExclusive = Math.Min(firstIndex + surface.IndexCount, pm4.MeshIndices.Count);
            if (surface.IndexCount <= 0 || firstIndex < 0 || endExclusive <= firstIndex)
                continue;

            for (int idx = firstIndex; idx < endExclusive; idx++)
            {
                int vertexIndex = (int)pm4.MeshIndices[idx];
                if ((uint)vertexIndex >= (uint)pm4.MeshVertices.Count)
                    continue;
                if (!seen.Add(vertexIndex))
                    continue;

                vertices.Add(pm4.MeshVertices[vertexIndex]);
            }
        }

        return vertices;
    }

    private static Pm4AxisConvention DetectAxisConventionByRanges(IReadOnlyList<Vector3> vertices)
    {
        if (vertices.Count == 0)
            return Pm4AxisConvention.XYPlaneZUp;

        float minX = float.MaxValue;
        float minY = float.MaxValue;
        float minZ = float.MaxValue;
        float maxX = float.MinValue;
        float maxY = float.MinValue;
        float maxZ = float.MinValue;

        for (int i = 0; i < vertices.Count; i++)
        {
            Vector3 v = vertices[i];
            if (v.X < minX) minX = v.X;
            if (v.Y < minY) minY = v.Y;
            if (v.Z < minZ) minZ = v.Z;
            if (v.X > maxX) maxX = v.X;
            if (v.Y > maxY) maxY = v.Y;
            if (v.Z > maxZ) maxZ = v.Z;
        }

        float rangeX = maxX - minX;
        float rangeY = maxY - minY;
        float rangeZ = maxZ - minZ;
        const float tieTolerance = 8f;

        if (rangeY + tieTolerance < rangeX && rangeY + tieTolerance < rangeZ)
            return Pm4AxisConvention.XZPlaneYUp;
        if (rangeZ + tieTolerance < rangeX && rangeZ + tieTolerance < rangeY)
            return Pm4AxisConvention.XYPlaneZUp;
        if (rangeX + tieTolerance < rangeY && rangeX + tieTolerance < rangeZ)
            return Pm4AxisConvention.YZPlaneXUp;

        // Ambiguous ranges: default to WoW-style XY plane with Z up.
        return Pm4AxisConvention.XYPlaneZUp;
    }

    private static bool IsLikelyTileLocal(IReadOnlyList<Vector3> vertices)
    {
        float minX = float.MaxValue;
        float minY = float.MaxValue;
        float minZ = float.MaxValue;
        float maxX = float.MinValue;
        float maxY = float.MinValue;
        float maxZ = float.MinValue;

        for (int i = 0; i < vertices.Count; i++)
        {
            Vector3 v = vertices[i];
            if (v.X < minX) minX = v.X;
            if (v.Y < minY) minY = v.Y;
            if (v.Z < minZ) minZ = v.Z;
            if (v.X > maxX) maxX = v.X;
            if (v.Y > maxY) maxY = v.Y;
            if (v.Z > maxZ) maxZ = v.Z;
        }

        const float tolerance = 64f;
        float tileSpan = Pm4CoordinateService.TileSize;

        bool xyLocal = minX >= -tolerance && minY >= -tolerance &&
                       maxX <= tileSpan + tolerance && maxY <= tileSpan + tolerance;
        bool xzLocal = minX >= -tolerance && minZ >= -tolerance &&
                       maxX <= tileSpan + tolerance && maxZ <= tileSpan + tolerance;
        bool yzLocal = minY >= -tolerance && minZ >= -tolerance &&
                       maxY <= tileSpan + tolerance && maxZ <= tileSpan + tolerance;

        return xyLocal || xzLocal || yzLocal;
    }

    private static Vector3 ConvertPm4VertexToWorld(Vector3 pm4Vertex, int tileX, int tileY, bool useTileLocalCoordinates, Pm4AxisConvention axisConvention, Pm4PlanarTransform planarTransform)
    {
        float localU;
        float localV;
        float localUp;

        switch (axisConvention)
        {
            case Pm4AxisConvention.XZPlaneYUp:
                localU = pm4Vertex.X;
                localV = pm4Vertex.Z;
                localUp = pm4Vertex.Y;
                break;
            case Pm4AxisConvention.YZPlaneXUp:
                localU = pm4Vertex.Y;
                localV = pm4Vertex.Z;
                localUp = pm4Vertex.X;
                break;
            case Pm4AxisConvention.XYPlaneZUp:
            default:
                localU = pm4Vertex.X;
                localV = pm4Vertex.Y;
                localUp = pm4Vertex.Z;
                break;
        }

        if (planarTransform.SwapPlanarAxes)
            (localU, localV) = (localV, localU);

        float tileSpan = Pm4CoordinateService.TileSize;
        float worldX;
        float worldY;

        if (useTileLocalCoordinates)
        {
            float mappedU = planarTransform.InvertU ? tileSpan - localU : localU;
            float mappedV = planarTransform.InvertV ? tileSpan - localV : localV;
            worldX = tileX * tileSpan + mappedU;
            worldY = tileY * tileSpan + mappedV;
        }
        else
        {
            if (planarTransform.InvertU)
                localU = -localU;
            if (planarTransform.InvertV)
                localV = -localV;

            worldX = localU;
            worldY = localV;
        }

        return new Vector3(worldX, worldY, localUp);
    }

    private static Vector3 ConvertPm4VertexToRenderer(Vector3 pm4Vertex, int tileX, int tileY, bool useTileLocalCoordinates, Pm4AxisConvention axisConvention, Pm4PlanarTransform planarTransform)
    {
        Vector3 world = ConvertPm4VertexToWorld(pm4Vertex, tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform);

        // Canonical world->renderer transform used across terrain/object pipelines.
        // rendererX = MapOrigin - wowY, rendererY = MapOrigin - wowX, rendererZ = wowZ
        return new Vector3(
            WoWConstants.MapOrigin - world.Y,
            WoWConstants.MapOrigin - world.X,
            world.Z + 0.5f);
    }

    public void ReloadPm4Overlay()
    {
        _pm4LoadAttempted = false;
        LazyLoadPm4Overlay();
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
        string? buildVersion = null,
        Action<string>? onStatus = null)
    {
        _gl = gl;
        _dataSource = dataSource;
        _dbcBuild = buildVersion;
        _assets = new WorldAssetManager(gl, dataSource, texResolver, buildVersion);
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
        string? buildVersion = null,
        Action<string>? onStatus = null)
    {
        _gl = gl;
        _dataSource = dataSource;
        _dbcBuild = buildVersion;
        _assets = new WorldAssetManager(gl, dataSource, texResolver, buildVersion);
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
            Vector3 localMin = Vector3.Zero;
            Vector3 localMax = Vector3.Zero;
            bool boundsResolved = false;
            if (_assets.TryGetMdxBounds(key, out var modelMin, out var modelMax))
            {
                localMin = modelMin;
                localMax = modelMax;
                boundsResolved = true;
                TransformBounds(modelMin, modelMax, transform, out bbMin, out bbMax);
            }
            else
            {
                bbMin = p.Position - new Vector3(2f);
                bbMax = p.Position + new Vector3(2f);
            }
            string modelPath = mdxNames[p.NameIndex];
            var instance = new ObjectInstance
            {
                ModelKey = key,
                Transform = transform,
                BoundsMin = bbMin,
                BoundsMax = bbMax,
                LocalBoundsMin = localMin,
                LocalBoundsMax = localMax,
                BoundsResolved = boundsResolved,
                ModelName = Path.GetFileName(modelPath),
                ModelPath = modelPath,
                PlacementPosition = p.Position,
                PlacementRotation = p.Rotation,
                PlacementScale = scale,
                UniqueId = p.UniqueId
            };

            if (IsSkyboxModelPath(modelPath))
                _skyboxInstances.Add(instance);
            else
                _mdxInstances.Add(instance);
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
                BoundsResolved = localMin != Vector3.Zero || localMax != Vector3.Zero,
                ModelName = Path.GetFileName(wmoPath),
                ModelPath = wmoPath,
                PlacementPosition = p.Position,
                PlacementRotation = p.Rotation,
                PlacementScale = 1.0f,
                UniqueId = p.UniqueId
            });
        }

        ViewerLog.Important(ViewerLog.Category.Terrain, $"Instances: {_mdxInstances.Count} MDX, {_skyboxInstances.Count} skybox, {_wmoInstances.Count} WMO");
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
        var tileSkyboxes = new List<ObjectInstance>();
        foreach (var p in result.MddfPlacements)
        {
            if (p.NameIndex < 0 || p.NameIndex >= mdxNames.Count) continue;
            string key = WorldAssetManager.NormalizeKey(mdxNames[p.NameIndex]);
            _assets.QueueMdxLoad(key);
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
            Vector3 localMin = Vector3.Zero;
            Vector3 localMax = Vector3.Zero;
            bool boundsResolved = false;
            if (_assets.TryGetMdxBounds(key, out var modelMin, out var modelMax))
            {
                localMin = modelMin;
                localMax = modelMax;
                boundsResolved = true;
                TransformBounds(modelMin, modelMax, transform, out bbMin, out bbMax);
            }
            else
            { bbMin = p.Position - new Vector3(2f); bbMax = p.Position + new Vector3(2f); }
            string modelPath = mdxNames[p.NameIndex];
            var instance = new ObjectInstance
            {
                ModelKey = key, Transform = transform, BoundsMin = bbMin, BoundsMax = bbMax,
                LocalBoundsMin = localMin, LocalBoundsMax = localMax, BoundsResolved = boundsResolved,
                ModelName = Path.GetFileName(modelPath), ModelPath = modelPath,
                PlacementPosition = p.Position, PlacementRotation = p.Rotation, PlacementScale = scale,
                UniqueId = p.UniqueId
            };

            if (IsSkyboxModelPath(modelPath))
                tileSkyboxes.Add(instance);
            else
                tileMdx.Add(instance);
        }

        // Build WMO instances for this tile
        var tileWmo = new List<ObjectInstance>();
        foreach (var p in result.ModfPlacements)
        {
            if (p.NameIndex < 0 || p.NameIndex >= wmoNames.Count) continue;
            string key = WorldAssetManager.NormalizeKey(wmoNames[p.NameIndex]);
            _assets.QueueWmoLoad(key);
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
                BoundsResolved = localMin != Vector3.Zero || localMax != Vector3.Zero,
                ModelName = Path.GetFileName(wmoPath), ModelPath = wmoPath,
                PlacementPosition = p.Position, PlacementRotation = p.Rotation, PlacementScale = 1.0f,
                UniqueId = p.UniqueId
            });
        }

        _tileMdxInstances[(tileX, tileY)] = tileMdx;
        _tileSkyboxInstances[(tileX, tileY)] = tileSkyboxes;
        _tileWmoInstances[(tileX, tileY)] = tileWmo;
        _instancesDirty = true;

        // Hide WDL low-res tile now that detailed ADT is loaded
        _wdlTerrain?.HideTile(tileX, tileY);

        if (tileMdx.Count > 0 || tileSkyboxes.Count > 0 || tileWmo.Count > 0)
            ViewerLog.Info(ViewerLog.Category.Terrain, $"Tile ({tileX},{tileY}) loaded: {tileMdx.Count} MDX, {tileSkyboxes.Count} skybox, {tileWmo.Count} WMO instances");
    }

    /// <summary>
    /// Called by TerrainManager when a tile leaves the AOI.
    /// </summary>
    private void OnTileUnloaded(int tileX, int tileY)
    {
        _tileMdxInstances.Remove((tileX, tileY));
        _tileSkyboxInstances.Remove((tileX, tileY));
        _tileWmoInstances.Remove((tileX, tileY));
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
        _mdxInstances.AddRange(_externalMdxInstances);

        _skyboxInstances.Clear();
        foreach (var list in _tileSkyboxInstances.Values)
            _skyboxInstances.AddRange(list);
        _skyboxInstances.AddRange(_externalSkyboxInstances);

        _wmoInstances.Clear();
        foreach (var list in _tileWmoInstances.Values)
            _wmoInstances.AddRange(list);
        _wmoInstances.AddRange(_externalWmoInstances);

        _instancesDirty = false;
    }

    private MdxRenderer? TryGetQueuedMdx(string modelKey)
    {
        if (_assets.TryGetLoadedMdx(modelKey, out var renderer))
            return renderer;

        _assets.PrioritizeMdxLoad(modelKey);
        return null;
    }

    private WmoRenderer? TryGetQueuedWmo(string modelKey)
    {
        if (_assets.TryGetLoadedWmo(modelKey, out var renderer))
            return renderer;

        _assets.PrioritizeWmoLoad(modelKey);
        return null;
    }

    private void ProcessDeferredAssetLoads()
    {
        int processed = _assets.ProcessPendingLoads(maxLoads: 24, maxBudgetMs: 20.0);
        if (processed <= 0)
            return;

        bool boundsChanged = false;
        foreach (var list in _tileMdxInstances.Values)
            boundsChanged |= RefreshMdxInstanceBounds(list);
        foreach (var list in _tileSkyboxInstances.Values)
            boundsChanged |= RefreshMdxInstanceBounds(list);
        foreach (var list in _tileWmoInstances.Values)
            boundsChanged |= RefreshWmoInstanceBounds(list);

        boundsChanged |= RefreshMdxInstanceBounds(_externalMdxInstances);
        boundsChanged |= RefreshMdxInstanceBounds(_externalSkyboxInstances);
        boundsChanged |= RefreshWmoInstanceBounds(_externalWmoInstances);

        if (boundsChanged)
        {
            _instancesDirty = true;
            RebuildInstanceLists();
        }
    }

    private bool RefreshMdxInstanceBounds(List<ObjectInstance> instances)
    {
        bool changed = false;

        for (int i = 0; i < instances.Count; i++)
        {
            var inst = instances[i];
            if (inst.BoundsResolved)
                continue;

            if (!_assets.TryGetMdxBounds(inst.ModelKey, out var localMin, out var localMax))
                continue;

            TransformBounds(localMin, localMax, inst.Transform, out var worldMin, out var worldMax);
            inst.LocalBoundsMin = localMin;
            inst.LocalBoundsMax = localMax;
            inst.BoundsMin = worldMin;
            inst.BoundsMax = worldMax;
            inst.BoundsResolved = true;
            instances[i] = inst;
            changed = true;
        }

        return changed;
    }

    private bool RefreshWmoInstanceBounds(List<ObjectInstance> instances)
    {
        bool changed = false;

        for (int i = 0; i < instances.Count; i++)
        {
            var inst = instances[i];
            if (inst.BoundsResolved)
                continue;

            if (!_assets.TryGetWmoBounds(inst.ModelKey, out var localMin, out var localMax))
                continue;

            TransformBounds(localMin, localMax, inst.Transform, out var worldMin, out var worldMax);
            inst.LocalBoundsMin = localMin;
            inst.LocalBoundsMax = localMax;
            inst.BoundsMin = worldMin;
            inst.BoundsMax = worldMax;
            inst.BoundsResolved = true;
            instances[i] = inst;
            changed = true;
        }

        return changed;
    }

    public void ClearExternalSpawns()
    {
        _externalMdxInstances.Clear();
        _externalSkyboxInstances.Clear();
        _externalWmoInstances.Clear();
        _instancesDirty = true;
    }

    public void SetExternalSpawns(IEnumerable<WorldSpawnRecord> spawns)
    {
        _externalMdxInstances.Clear();
        _externalSkyboxInstances.Clear();
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
                _assets.QueueWmoLoad(key);

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
                    BoundsResolved = localMin != Vector3.Zero || localMax != Vector3.Zero,
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
                _assets.QueueMdxLoad(key);

                var transform = Matrix4x4.CreateScale(mdxScale)
                    * Matrix4x4.CreateRotationZ(finalYawRadians)
                    * Matrix4x4.CreateTranslation(pos);

                Vector3 bbMin, bbMax;
                Vector3 localMin = Vector3.Zero;
                Vector3 localMax = Vector3.Zero;
                bool boundsResolved = false;
                if (_assets.TryGetMdxBounds(key, out var modelMin, out var modelMax))
                {
                    localMin = modelMin;
                    localMax = modelMax;
                    boundsResolved = true;
                    TransformBounds(modelMin, modelMax, transform, out bbMin, out bbMax);
                }
                else
                {
                    bbMin = pos - new Vector3(2f);
                    bbMax = pos + new Vector3(2f);
                }

                var instance = new ObjectInstance
                {
                    ModelKey = key,
                    Transform = transform,
                    BoundsMin = bbMin,
                    BoundsMax = bbMax,
                    LocalBoundsMin = localMin,
                    LocalBoundsMax = localMax,
                    BoundsResolved = boundsResolved,
                    ModelName = Path.GetFileName(modelPath),
                    ModelPath = modelPath,
                    PlacementPosition = pos,
                    PlacementRotation = new Vector3(0f, 0f, finalYawDegrees),
                    PlacementScale = mdxScale,
                    UniqueId = spawn.SpawnId
                };

                if (IsSkyboxModelPath(modelPath))
                    _externalSkyboxInstances.Add(instance);
                else
                    _externalMdxInstances.Add(instance);
            }
        }

        ViewerLog.Info(ViewerLog.Category.Terrain,
            $"SQL spawns injected: {_externalMdxInstances.Count} MDX, {_externalSkyboxInstances.Count} skybox, {_externalWmoInstances.Count} WMO");

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

        ProcessDeferredAssetLoads();

        // Extract camera position for sky dome
        Matrix4x4.Invert(view, out var viewInvSky);
        var camPos = new Vector3(viewInvSky.M41, viewInvSky.M42, viewInvSky.M43);
        var lighting = _terrainManager.Lighting;
        Vector3 fogColor;
        float fogStart;
        float fogEnd;

        // 0. Render sky dome (before terrain, no depth write)
        // Update DBC lighting early so sky colors are available
        _lightService?.Update(camPos);
        if (_lightService != null && _lightService.ActiveLightId >= 0)
        {
            // Override sky dome colors from DBC Light data
            _skyDome.ZenithColor = _lightService.SkyTopColor;
            _skyDome.HorizonColor = _lightService.FogColor;
            _skyDome.SkyFogColor = _lightService.FogColor;
            fogColor = _lightService.FogColor;
            fogEnd = _lightService.FogEnd > 10f ? _lightService.FogEnd : lighting.FogEnd;
            fogStart = fogEnd * 0.25f;
        }
        else
        {
            _skyDome.UpdateFromLighting(_terrainManager.Lighting.GameTime);
            fogColor = lighting.FogColor;
            fogStart = lighting.FogStart;
            fogEnd = lighting.FogEnd;
        }
        _skyDome.Render(view, proj, camPos);

        // Also set clear color to horizon color so any gaps match the sky
        _gl.ClearColor(_skyDome.HorizonColor.X, _skyDome.HorizonColor.Y, _skyDome.HorizonColor.Z, 1f);

        RenderSkyboxBackdrop(view, proj, camPos, fogColor, fogStart, fogEnd, lighting);

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
                if (_assets.TryGetLoadedWmo(inst.ModelKey, out _)) wmoFound++;
                else { wmoMissing++; if (wmoMissing <= 3) ViewerLog.Debug(ViewerLog.Category.Wmo, $"NOT FOUND: \"{inst.ModelKey}\""); }
            }
            int mdxFound = 0, mdxMissing = 0;
            foreach (var inst in _mdxInstances)
            {
                if (_assets.TryGetLoadedMdx(inst.ModelKey, out _)) mdxFound++;
                else { mdxMissing++; if (mdxMissing <= 3) ViewerLog.Debug(ViewerLog.Category.Mdx, $"NOT FOUND: \"{inst.ModelKey}\""); }
            }
            ViewerLog.Info(ViewerLog.Category.Terrain, $"Render check: WMO {wmoFound} found / {wmoMissing} missing, MDX {mdxFound} found / {mdxMissing} missing");
        }

        // Extract camera position from view matrix (inverse of view translation)
        Matrix4x4.Invert(view, out var viewInv);
        var cameraPos = new Vector3(viewInv.M41, viewInv.M42, viewInv.M43);

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

                var renderer = TryGetQueuedWmo(inst.ModelKey);
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
                    _assets.TryGetLoadedMdx(inst.ModelKey, out var r);
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
                if (_assets.TryGetLoadedMdx(inst.ModelKey, out batchRenderer) && batchRenderer != null)
                    break;
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
                var renderer = TryGetQueuedMdx(inst.ModelKey);
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
                // Same frustum + distance cull as opaque pass (with NoCullRadius)
                var placementPos = inst.Transform.Translation;
                float dist = Vector3.DistanceSquared(cameraPos, placementPos);
                if (dist > NoCullRadiusSq && !_frustumCuller.TestAABB(inst.BoundsMin, inst.BoundsMax)) continue;
                var diag = (inst.BoundsMax - inst.BoundsMin).Length();
                if (diag < DoodadSmallThreshold && dist > DoodadCullDistanceSq) continue;
                if (TryGetQueuedMdx(inst.ModelKey) == null) continue;
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
                var renderer = TryGetQueuedMdx(inst.ModelKey);
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

        if (_wireframeRevealEnabled)
            RenderWireframeReveal(view, proj, cameraPos, fogColor, fogStart, fogEnd, lighting);

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
            _bbRenderer.BeginSolidBatch();

            _pm4VisibleObjectCount = 0;
            _pm4VisibleLineCount = 0;
            _pm4VisibleTriangleCount = 0;
            _pm4VisiblePositionRefCount = 0;

            if (_showPm4Overlay && _pm4TileObjects.Count > 0)
            {
                bool hasLoadedTiles = _terrainManager.LoadedTileCount > 0;
                Matrix4x4 pm4Transform = BuildPm4OverlayTransformMatrix();
                bool applyPm4Transform = _pm4OverlayTranslation != Vector3.Zero
                    || _pm4OverlayRotationDegrees.LengthSquared() > 0.0001f
                    || _pm4OverlayScale != Vector3.One;

                foreach (var (tileKey, objects) in _pm4TileObjects)
                {
                    if (hasLoadedTiles && !_terrainManager.IsTileLoaded(tileKey.tileX, tileKey.tileY))
                        continue;

                    if (_showPm4PositionRefs
                        && _pm4TilePositionRefs.TryGetValue(tileKey, out List<Vector3>? positionRefs)
                        && positionRefs.Count > 0)
                    {
                        for (int i = 0; i < positionRefs.Count; i++)
                        {
                            Vector3 marker = applyPm4Transform ? ApplyPm4OverlayTransform(positionRefs[i], pm4Transform) : positionRefs[i];
                            _bbRenderer.BatchPin(marker, 16f, 3f, new Vector3(0.20f, 0.90f, 1.00f));
                        }

                        _pm4VisiblePositionRefCount += positionRefs.Count;
                    }

                    foreach (Pm4OverlayObject obj in objects)
                    {
                        if (!ShouldRenderPm4ObjectType(obj.Ck24Type))
                            continue;

                        var objectKey = (tileKey.tileX, tileKey.tileY, obj.Ck24, obj.ObjectPartId);
                        Matrix4x4 objectTransform = BuildPm4ObjectTransform(objectKey, applyPm4Transform, pm4Transform, out bool applyObjectTransform);

                        _pm4VisibleObjectCount++;
                        Vector3 pm4Color = GetPm4ObjectColor(tileKey, obj);
                        if (_selectedPm4ObjectKey.HasValue && _selectedPm4ObjectKey.Value == objectKey)
                            pm4Color = new Vector3(1.0f, 1.0f, 0.2f);

                        if (_showPm4SolidOverlay && obj.Triangles.Count > 0)
                        {
                            for (int i = 0; i < obj.Triangles.Count; i++)
                            {
                                Pm4Triangle tri = obj.Triangles[i];
                                Vector3 a = applyObjectTransform ? ApplyPm4OverlayTransform(tri.A, objectTransform) : tri.A;
                                Vector3 b = applyObjectTransform ? ApplyPm4OverlayTransform(tri.B, objectTransform) : tri.B;
                                Vector3 c = applyObjectTransform ? ApplyPm4OverlayTransform(tri.C, objectTransform) : tri.C;
                                _bbRenderer.BatchTriangle(a, b, c, pm4Color, 0.20f);
                            }
                            _pm4VisibleTriangleCount += obj.Triangles.Count;
                        }

                        for (int i = 0; i < obj.Lines.Count; i++)
                        {
                            Pm4LineSegment line = obj.Lines[i];
                            Vector3 from = applyObjectTransform ? ApplyPm4OverlayTransform(line.From, objectTransform) : line.From;
                            Vector3 to = applyObjectTransform ? ApplyPm4OverlayTransform(line.To, objectTransform) : line.To;
                            _bbRenderer.BatchLine(from, to, pm4Color);
                        }

                        _pm4VisibleLineCount += obj.Lines.Count;

                        if (_showPm4ObjectCentroids)
                        {
                            Vector3 center = applyObjectTransform ? ApplyPm4OverlayTransform(obj.Center, objectTransform) : obj.Center;
                            _bbRenderer.BatchPin(center, 22f, 4f, pm4Color);
                        }
                    }
                }
            }

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

            if (_showPm4Overlay && _showPm4SolidOverlay && _pm4VisibleTriangleCount > 0)
            {
                _gl.Enable(EnableCap.Blend);
                _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
                _gl.Enable(EnableCap.DepthTest);
                _gl.DepthFunc(DepthFunction.Lequal);
                _gl.DepthMask(false);
                _gl.Disable(EnableCap.CullFace);
                _bbRenderer.FlushSolidBatch(view, proj);
                _gl.Enable(EnableCap.CullFace);
                _gl.DepthMask(true);
                _gl.Disable(EnableCap.Blend);
            }

            _bbRenderer.FlushBatch(view, proj);
        }
    }

    private void RenderSkyboxBackdrop(Matrix4x4 view, Matrix4x4 proj, Vector3 cameraPos,
        Vector3 fogColor, float fogStart, float fogEnd, TerrainLighting lighting)
    {
        if (_skyboxInstances.Count == 0)
            return;

        ObjectInstance? nearestSkybox = null;
        float nearestDistSq = float.MaxValue;
        foreach (var inst in _skyboxInstances)
        {
            float distSq = Vector3.DistanceSquared(cameraPos, inst.PlacementPosition);
            if (distSq >= nearestDistSq)
                continue;

            nearestDistSq = distSq;
            nearestSkybox = inst;
        }

        if (!nearestSkybox.HasValue)
            return;

        var skybox = nearestSkybox.Value;
        var renderer = TryGetQueuedMdx(skybox.ModelKey);
        if (renderer == null)
            return;

        renderer.UpdateAnimation();
        renderer.RenderBackdrop(CreateSkyboxBackdropTransform(skybox.Transform, cameraPos), view, proj,
            fogColor, fogStart, fogEnd, cameraPos,
            lighting.LightDirection, lighting.LightColor, lighting.AmbientColor);
    }

    private static Matrix4x4 CreateSkyboxBackdropTransform(Matrix4x4 placementTransform, Vector3 cameraPos)
    {
        placementTransform.M41 = cameraPos.X;
        placementTransform.M42 = cameraPos.Y;
        placementTransform.M43 = cameraPos.Z;
        return placementTransform;
    }

    internal static bool IsSkyboxModelPath(string modelPath)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            return false;

        string normalized = modelPath.Replace('\\', '/').ToLowerInvariant();
        if (!normalized.EndsWith(".m2", StringComparison.OrdinalIgnoreCase) &&
            !normalized.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase))
            return false;

        if (normalized.Contains("skylight"))
            return false;

        return normalized.Contains("environments/stars/") ||
               normalized.Contains("/skybox/") ||
               normalized.Contains("skybox") ||
               normalized.Contains("skybowl");
    }

    public void ToggleWireframe()
    {
        _wireframeRevealEnabled = !_wireframeRevealEnabled;
        _terrainManager.ToggleWireframe();
        if (!_wireframeRevealEnabled)
            ClearWireframeReveal();
    }

    public void UpdateWireframeReveal(Matrix4x4 view, Matrix4x4 proj,
        float mouseViewportX, float mouseViewportY, float viewportWidth, float viewportHeight)
    {
        if (!_wireframeRevealEnabled)
        {
            ClearWireframeReveal();
            return;
        }

        if (_instancesDirty)
            RebuildInstanceLists();

        _wireframeRevealWmoIndices.Clear();
        _wireframeRevealMdxIndices.Clear();

        if (_wmosVisible)
            PopulateWireframeRevealHits(_wmoInstances, _wireframeRevealWmoIndices,
                view, proj, mouseViewportX, mouseViewportY, viewportWidth, viewportHeight);
        if (_doodadsVisible)
            PopulateWireframeRevealHits(_mdxInstances, _wireframeRevealMdxIndices,
                view, proj, mouseViewportX, mouseViewportY, viewportWidth, viewportHeight);
    }

    public void ClearWireframeReveal()
    {
        _wireframeRevealWmoIndices.Clear();
        _wireframeRevealMdxIndices.Clear();
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

    public bool SelectPm4ObjectByRay(Vector3 rayOrigin, Vector3 rayDir)
    {
        if (!_showPm4Overlay || _pm4TileObjects.Count == 0)
        {
            _selectedPm4ObjectKey = null;
            return false;
        }

        bool hasLoadedTiles = _terrainManager.LoadedTileCount > 0;
        Matrix4x4 pm4Transform = BuildPm4OverlayTransformMatrix();
        bool applyPm4Transform = _pm4OverlayTranslation != Vector3.Zero
            || _pm4OverlayRotationDegrees.LengthSquared() > 0.0001f
            || _pm4OverlayScale != Vector3.One;

        float bestT = float.MaxValue;
        (int tileX, int tileY, uint ck24, int objectPart)? bestKey = null;

        foreach (var (tileKey, objects) in _pm4TileObjects)
        {
            if (hasLoadedTiles && !_terrainManager.IsTileLoaded(tileKey.tileX, tileKey.tileY))
                continue;

            foreach (Pm4OverlayObject obj in objects)
            {
                if (!ShouldRenderPm4ObjectType(obj.Ck24Type))
                    continue;

                var objectKey = (tileKey.tileX, tileKey.tileY, obj.Ck24, obj.ObjectPartId);
                Matrix4x4 objectTransform = BuildPm4ObjectTransform(objectKey, applyPm4Transform, pm4Transform, out bool applyObjectTransform);

                Vector3 boundsMin = obj.BoundsMin;
                Vector3 boundsMax = obj.BoundsMax;
                if (applyObjectTransform)
                    TransformBounds(boundsMin, boundsMax, objectTransform, out boundsMin, out boundsMax);

                Vector3 padding = new(2f, 2f, 2f);
                float t = RayAABBIntersect(rayOrigin, rayDir, boundsMin - padding, boundsMax + padding);
                if (t >= 0f && t < bestT)
                {
                    bestT = t;
                    bestKey = objectKey;
                }
            }
        }

        _selectedPm4ObjectKey = bestKey;
        return bestKey.HasValue;
    }

    public void ClearSelection()
    {
        _selectedObjectType = ObjectType.None;
        _selectedObjectIndex = -1;
    }

    public void ClearPm4ObjectSelection()
    {
        _selectedPm4ObjectKey = null;
    }

    public bool TryGetSelectedPm4ObjectDebugInfo(out Pm4ObjectDebugInfo info)
    {
        info = default;
        if (!_selectedPm4ObjectKey.HasValue)
            return false;

        var objectKey = _selectedPm4ObjectKey.Value;
        if (!_pm4ObjectLookup.TryGetValue(objectKey, out Pm4OverlayObject? obj))
            return false;

        Matrix4x4 pm4Transform = BuildPm4OverlayTransformMatrix();
        bool applyPm4Transform = _pm4OverlayTranslation != Vector3.Zero
            || _pm4OverlayRotationDegrees.LengthSquared() > 0.0001f
            || _pm4OverlayScale != Vector3.One;
        Matrix4x4 objectTransform = BuildPm4ObjectTransform(objectKey, applyPm4Transform, pm4Transform, out bool applyObjectTransform);

        Vector3 center = applyObjectTransform ? ApplyPm4OverlayTransform(obj.Center, objectTransform) : obj.Center;
        Vector3 boundsMin = obj.BoundsMin;
        Vector3 boundsMax = obj.BoundsMax;
        if (applyObjectTransform)
            TransformBounds(boundsMin, boundsMax, objectTransform, out boundsMin, out boundsMax);

        float nearestPositionRefDistance = float.NaN;
        if (_pm4TilePositionRefs.TryGetValue((objectKey.tileX, objectKey.tileY), out List<Vector3>? positionRefs)
            && positionRefs.Count > 0)
        {
            nearestPositionRefDistance = NearestPointDistance(center, positionRefs, applyPm4Transform, pm4Transform);
        }

        info = new Pm4ObjectDebugInfo(
            obj.Ck24,
            obj.Ck24Type,
            obj.Ck24ObjectId,
            obj.ObjectPartId,
            obj.LinkGroupObjectId,
            obj.LinkedPositionRefCount,
            objectKey.tileX,
            objectKey.tileY,
            obj.SurfaceCount,
            obj.DominantGroupKey,
            obj.DominantAttributeMask,
            obj.DominantMdosIndex,
            obj.AverageSurfaceHeight,
            boundsMin,
            boundsMax,
            center,
            nearestPositionRefDistance,
            obj.PlanarTransform.SwapPlanarAxes,
            obj.PlanarTransform.InvertU,
            obj.PlanarTransform.InvertV,
            obj.PlanarTransform.InvertsWinding);

        return true;
    }

    private static float NearestPointDistance(Vector3 point, IReadOnlyList<Vector3> candidates, bool applyPm4Transform, in Matrix4x4 pm4Transform)
    {
        float best = float.MaxValue;
        for (int i = 0; i < candidates.Count; i++)
        {
            Vector3 candidate = applyPm4Transform ? ApplyPm4OverlayTransform(candidates[i], pm4Transform) : candidates[i];
            float dist = Vector3.Distance(point, candidate);
            if (dist < best)
                best = dist;
        }

        return best;
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

    private void PopulateWireframeRevealHits(List<ObjectInstance> instances, List<int> hitIndices,
        Matrix4x4 view, Matrix4x4 proj, float mouseViewportX, float mouseViewportY,
        float viewportWidth, float viewportHeight)
    {
        for (int i = 0; i < instances.Count; i++)
        {
            if (ShouldRevealInstance(instances[i], view, proj, mouseViewportX, mouseViewportY, viewportWidth, viewportHeight))
                hitIndices.Add(i);
        }
    }

    private static bool ShouldRevealInstance(ObjectInstance inst, Matrix4x4 view, Matrix4x4 proj,
        float mouseViewportX, float mouseViewportY, float viewportWidth, float viewportHeight)
    {
        Vector3 center = (inst.BoundsMin + inst.BoundsMax) * 0.5f;
        if (!TryProjectToViewport(center, view, proj, viewportWidth, viewportHeight, out float sx, out float sy, out float depth))
            return false;

        float dx = sx - mouseViewportX;
        float dy = sy - mouseViewportY;
        float distanceSq = dx * dx + dy * dy;

        float worldRadius = MathF.Max((inst.BoundsMax - inst.BoundsMin).Length() * 0.5f, 4f);
        float projectedRadius = EstimateProjectedRadius(worldRadius, depth, proj, viewportHeight);
        float revealRadius = MathF.Min(WireframeRevealBrushPixels + projectedRadius, WireframeRevealMaxScreenRadius);
        return distanceSq <= revealRadius * revealRadius;
    }

    private static bool TryProjectToViewport(Vector3 worldPos, Matrix4x4 view, Matrix4x4 proj,
        float viewportWidth, float viewportHeight, out float sx, out float sy, out float depth)
    {
        var viewSpace = Vector4.Transform(new Vector4(worldPos, 1f), view);
        depth = MathF.Abs(viewSpace.Z);
        if (depth < 0.001f)
        {
            sx = sy = 0f;
            return false;
        }

        var clip = Vector4.Transform(new Vector4(worldPos, 1f), view * proj);
        if (clip.W <= 0f)
        {
            sx = sy = 0f;
            return false;
        }

        float ndcX = clip.X / clip.W;
        float ndcY = clip.Y / clip.W;
        sx = (ndcX * 0.5f + 0.5f) * viewportWidth;
        sy = (1f - (ndcY * 0.5f + 0.5f)) * viewportHeight;
        return true;
    }

    private static float EstimateProjectedRadius(float worldRadius, float depth, Matrix4x4 proj, float viewportHeight)
    {
        float yScale = MathF.Abs(proj.M22);
        if (yScale < 0.0001f)
            return 0f;

        return MathF.Min((worldRadius * yScale / depth) * (viewportHeight * 0.5f), WireframeRevealMaxScreenRadius);
    }

    private void RenderWireframeReveal(Matrix4x4 view, Matrix4x4 proj, Vector3 cameraPos,
        Vector3 fogColor, float fogStart, float fogEnd, TerrainLighting lighting)
    {
        if (_wireframeRevealWmoIndices.Count == 0 && _wireframeRevealMdxIndices.Count == 0)
            return;

        _gl.Enable(EnableCap.DepthTest);
        _gl.DepthFunc(DepthFunction.Lequal);
        _gl.DepthMask(false);
        _gl.Disable(EnableCap.Blend);

        foreach (int idx in _wireframeRevealWmoIndices)
        {
            if ((uint)idx >= (uint)_wmoInstances.Count)
                continue;

            var inst = _wmoInstances[idx];
            var renderer = TryGetQueuedWmo(inst.ModelKey);
            if (renderer == null)
                continue;

            renderer.RenderWireframeOverlay(inst.Transform, view, proj,
                fogColor, fogStart, fogEnd, cameraPos,
                lighting.LightDirection, lighting.LightColor, lighting.AmbientColor);
        }

        foreach (int idx in _wireframeRevealMdxIndices)
        {
            if ((uint)idx >= (uint)_mdxInstances.Count)
                continue;

            var inst = _mdxInstances[idx];
            var renderer = TryGetQueuedMdx(inst.ModelKey);
            if (renderer == null)
                continue;

            renderer.RenderWireframeOverlay(inst.Transform, view, proj,
                fogColor, fogStart, fogEnd, cameraPos,
                lighting.LightDirection, lighting.LightColor, lighting.AmbientColor);
        }

        _gl.DepthMask(true);
        _gl.DepthFunc(DepthFunction.Lequal);
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

    private Vector3 GetPm4ObjectColor((int tileX, int tileY) tileKey, Pm4OverlayObject obj)
    {
        return _pm4ColorMode switch
        {
            Pm4OverlayColorMode.Ck24ObjectId => ColorFromSeed(obj.Ck24ObjectId),
            Pm4OverlayColorMode.Ck24Key => ColorFromSeed(obj.Ck24),
            Pm4OverlayColorMode.Tile => ColorFromSeed((uint)HashCode.Combine(tileKey.tileX, tileKey.tileY)),
            Pm4OverlayColorMode.GroupKey => ColorFromSeed(obj.DominantGroupKey),
            Pm4OverlayColorMode.AttributeMask => ColorFromSeed(obj.DominantAttributeMask),
            Pm4OverlayColorMode.Height => ColorFromHeight(obj.Center.Z),
            _ => GetPm4TypeColor(obj.Ck24Type)
        };
    }

    private Vector3 ColorFromHeight(float z)
    {
        float denom = _pm4MaxObjectZ - _pm4MinObjectZ;
        float t = denom > 0.001f ? Math.Clamp((z - _pm4MinObjectZ) / denom, 0f, 1f) : 0.5f;
        return Vector3.Lerp(new Vector3(0.15f, 0.45f, 1.00f), new Vector3(1.00f, 0.35f, 0.15f), t);
    }

    private static Vector3 ColorFromSeed(uint seed)
    {
        uint golden = seed * 2654435761u;
        float hue = (golden & 0x00FFFFFF) / 16777215f;
        return HsvToRgb(hue, 0.75f, 0.95f);
    }

    private static Vector3 HsvToRgb(float h, float s, float v)
    {
        h = h - MathF.Floor(h);
        float c = v * s;
        float x = c * (1f - MathF.Abs((h * 6f) % 2f - 1f));
        float m = v - c;

        float r;
        float g;
        float b;
        int sector = (int)(h * 6f);
        switch (sector)
        {
            case 0:
                r = c; g = x; b = 0f;
                break;
            case 1:
                r = x; g = c; b = 0f;
                break;
            case 2:
                r = 0f; g = c; b = x;
                break;
            case 3:
                r = 0f; g = x; b = c;
                break;
            case 4:
                r = x; g = 0f; b = c;
                break;
            default:
                r = c; g = 0f; b = x;
                break;
        }

        return new Vector3(r + m, g + m, b + m);
    }

    private static Vector3 GetPm4TypeColor(byte ck24Type)
    {
        return ck24Type switch
        {
            0x40 => new Vector3(1.0f, 0.55f, 0.10f),
            0x80 => new Vector3(0.95f, 0.32f, 0.08f),
            _ => new Vector3(0.95f, 0.48f, 0.08f)
        };
    }

    private bool ShouldRenderPm4ObjectType(byte ck24Type)
    {
        return ck24Type switch
        {
            0x40 => _showPm4Type40,
            0x80 => _showPm4Type80,
            _ => _showPm4TypeOther
        };
    }

    private Matrix4x4 BuildPm4OverlayTransformMatrix()
    {
        float rotX = _pm4OverlayRotationDegrees.X * MathF.PI / 180f;
        float rotY = _pm4OverlayRotationDegrees.Y * MathF.PI / 180f;
        float rotZ = _pm4OverlayRotationDegrees.Z * MathF.PI / 180f;
        return Matrix4x4.CreateScale(_pm4OverlayScale)
            * Matrix4x4.CreateRotationX(rotX)
            * Matrix4x4.CreateRotationY(rotY)
            * Matrix4x4.CreateRotationZ(rotZ)
            * Matrix4x4.CreateTranslation(_pm4OverlayTranslation);
    }

    private static Vector3 ApplyPm4OverlayTransform(Vector3 position, in Matrix4x4 transform)
    {
        return Vector3.Transform(position, transform);
    }

    private Matrix4x4 BuildPm4ObjectTransform((int tileX, int tileY, uint ck24, int objectPart) objectKey,
        bool applyPm4Transform,
        in Matrix4x4 pm4Transform,
        out bool applyObjectTransform)
    {
        applyObjectTransform = false;
        Matrix4x4 transform = Matrix4x4.Identity;

        if (applyPm4Transform)
        {
            transform = pm4Transform;
            applyObjectTransform = true;
        }

        if (_pm4ObjectTranslations.TryGetValue(objectKey, out Vector3 objectTranslation)
            && objectTranslation.LengthSquared() > 0.0001f)
        {
            Matrix4x4 objectTranslationMatrix = Matrix4x4.CreateTranslation(objectTranslation);
            transform = applyObjectTransform
                ? transform * objectTranslationMatrix
                : objectTranslationMatrix;
            applyObjectTransform = true;
        }

        return transform;
    }

    private static void TransformBounds(Vector3 boundsMin, Vector3 boundsMax, in Matrix4x4 transform,
        out Vector3 transformedMin, out Vector3 transformedMax)
    {
        transformedMin = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
        transformedMax = new Vector3(float.MinValue, float.MinValue, float.MinValue);

        Span<Vector3> corners = stackalloc Vector3[8];
        corners[0] = new Vector3(boundsMin.X, boundsMin.Y, boundsMin.Z);
        corners[1] = new Vector3(boundsMax.X, boundsMin.Y, boundsMin.Z);
        corners[2] = new Vector3(boundsMin.X, boundsMax.Y, boundsMin.Z);
        corners[3] = new Vector3(boundsMax.X, boundsMax.Y, boundsMin.Z);
        corners[4] = new Vector3(boundsMin.X, boundsMin.Y, boundsMax.Z);
        corners[5] = new Vector3(boundsMax.X, boundsMin.Y, boundsMax.Z);
        corners[6] = new Vector3(boundsMin.X, boundsMax.Y, boundsMax.Z);
        corners[7] = new Vector3(boundsMax.X, boundsMax.Y, boundsMax.Z);

        for (int i = 0; i < corners.Length; i++)
        {
            Vector3 transformed = Vector3.Transform(corners[i], transform);
            transformedMin = Vector3.Min(transformedMin, transformed);
            transformedMax = Vector3.Max(transformedMax, transformed);
        }
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
        _skyboxInstances.Clear();
        _wmoInstances.Clear();
        _tileMdxInstances.Clear();
        _tileSkyboxInstances.Clear();
        _tileWmoInstances.Clear();
        _externalMdxInstances.Clear();
        _externalSkyboxInstances.Clear();
        _externalWmoInstances.Clear();
        _pm4TileObjects.Clear();
        _pm4TileStats.Clear();
        _pm4TilePositionRefs.Clear();
        _pm4ObjectLookup.Clear();
        _pm4ObjectTranslations.Clear();
    }
}

internal sealed class Pm4OverlayObject
{
    public Pm4OverlayObject(
        uint ck24,
        byte ck24Type,
        int objectPartId,
        uint linkGroupObjectId,
        int linkedPositionRefCount,
        List<Pm4LineSegment> lines,
        List<Pm4Triangle> triangles,
        int surfaceCount,
        byte dominantGroupKey,
        byte dominantAttributeMask,
        uint dominantMdosIndex,
        float averageSurfaceHeight,
        Pm4PlanarTransform planarTransform)
    {
        Ck24 = ck24;
        Ck24Type = ck24Type;
        ObjectPartId = objectPartId;
        LinkGroupObjectId = linkGroupObjectId;
        LinkedPositionRefCount = linkedPositionRefCount;
        Lines = lines;
        Triangles = triangles;
        SurfaceCount = surfaceCount;
        DominantGroupKey = dominantGroupKey;
        DominantAttributeMask = dominantAttributeMask;
        DominantMdosIndex = dominantMdosIndex;
        AverageSurfaceHeight = averageSurfaceHeight;
        PlanarTransform = planarTransform;
        (BoundsMin, BoundsMax) = ComputeBounds(lines, triangles);
        Center = (BoundsMin + BoundsMax) * 0.5f;
    }

    public uint Ck24 { get; }
    public byte Ck24Type { get; }
    public ushort Ck24ObjectId => (ushort)(Ck24 & 0xFFFF);
    public int ObjectPartId { get; }
    public uint LinkGroupObjectId { get; }
    public int LinkedPositionRefCount { get; }
    public List<Pm4LineSegment> Lines { get; }
    public List<Pm4Triangle> Triangles { get; }
    public int SurfaceCount { get; }
    public byte DominantGroupKey { get; }
    public byte DominantAttributeMask { get; }
    public uint DominantMdosIndex { get; }
    public float AverageSurfaceHeight { get; }
    public Pm4PlanarTransform PlanarTransform { get; }
    public Vector3 BoundsMin { get; }
    public Vector3 BoundsMax { get; }
    public Vector3 Center { get; }

    private static (Vector3 min, Vector3 max) ComputeBounds(List<Pm4LineSegment> lines, List<Pm4Triangle> triangles)
    {
        Vector3 min = new(float.MaxValue, float.MaxValue, float.MaxValue);
        Vector3 max = new(float.MinValue, float.MinValue, float.MinValue);
        bool hasData = false;

        for (int i = 0; i < lines.Count; i++)
        {
            min = Vector3.Min(min, lines[i].From);
            min = Vector3.Min(min, lines[i].To);
            max = Vector3.Max(max, lines[i].From);
            max = Vector3.Max(max, lines[i].To);
            hasData = true;
        }

        for (int i = 0; i < triangles.Count; i++)
        {
            min = Vector3.Min(min, triangles[i].A);
            min = Vector3.Min(min, triangles[i].B);
            min = Vector3.Min(min, triangles[i].C);
            max = Vector3.Max(max, triangles[i].A);
            max = Vector3.Max(max, triangles[i].B);
            max = Vector3.Max(max, triangles[i].C);
            hasData = true;
        }

        if (!hasData)
            return (Vector3.Zero, Vector3.Zero);

        return (min, max);
    }
}

internal readonly struct Pm4LineSegment
{
    public Pm4LineSegment(Vector3 from, Vector3 to)
    {
        From = from;
        To = to;
    }

    public Vector3 From { get; }
    public Vector3 To { get; }
}

internal readonly struct Pm4Triangle
{
    public Pm4Triangle(Vector3 a, Vector3 b, Vector3 c)
    {
        A = a;
        B = b;
        C = c;
    }

    public Vector3 A { get; }
    public Vector3 B { get; }
    public Vector3 C { get; }
}

public readonly struct Pm4OverlayTileStats
{
    public Pm4OverlayTileStats(int tileX, int tileY, int objectCount, int lineCount, int triangleCount)
    {
        TileX = tileX;
        TileY = tileY;
        ObjectCount = objectCount;
        LineCount = lineCount;
        TriangleCount = triangleCount;
    }

    public int TileX { get; }
    public int TileY { get; }
    public int ObjectCount { get; }
    public int LineCount { get; }
    public int TriangleCount { get; }
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
    /// <summary>True once bounds were derived from the loaded model instead of a temporary placement fallback.</summary>
    public bool BoundsResolved;
}

public enum ObjectType { None, Wmo, Mdx }
