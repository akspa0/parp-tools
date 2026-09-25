using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Text;
using System.Text.Json;
using WoWViewer.DataSources;
using WoWViewer.Logging;
using WoWViewer.Population;
using WoWViewer.Rendering;
using WoWViewer.Audio;
using WowViewer.Core.Audio;
using WowViewer.Core.Maps;
using Silk.NET.OpenGL;
using CorePm4AxisConvention = WowViewer.Core.PM4.Models.Pm4AxisConvention;
using CorePm4CorrelationCandidateScore = WowViewer.Core.PM4.Models.Pm4CorrelationCandidateScore;
using CorePm4CorrelationMetrics = WowViewer.Core.PM4.Models.Pm4CorrelationMetrics;
using CorePm4CorrelationObjectDescriptor = WowViewer.Core.PM4.Models.Pm4CorrelationObjectDescriptor;
using CorePm4CorrelationGeometryInput = WowViewer.Core.PM4.Models.Pm4CorrelationGeometryInput;
using CorePm4CorrelationObjectInput = WowViewer.Core.PM4.Models.Pm4CorrelationObjectInput;
using CorePm4CorrelationObjectState = WowViewer.Core.PM4.Models.Pm4CorrelationObjectState;
using CorePm4CorrelationMath = WowViewer.Core.PM4.Services.Pm4CorrelationMath;
using CorePm4ConnectorKey = WowViewer.Core.PM4.Models.Pm4ConnectorKey;
using CorePm4ConnectorMergeCandidate = WowViewer.Core.PM4.Models.Pm4ConnectorMergeCandidate;
using CorePm4CoordinateMode = WowViewer.Core.PM4.Models.Pm4CoordinateMode;
using CorePm4GeometryLineSegment = WowViewer.Core.PM4.Models.Pm4GeometryLineSegment;
using CorePm4GeometryTriangle = WowViewer.Core.PM4.Models.Pm4GeometryTriangle;
using CorePm4LinkedPositionRefSummary = WowViewer.Core.PM4.Models.Pm4LinkedPositionRefSummary;
using CorePm4MprlEntry = WowViewer.Core.PM4.Models.Pm4MprlEntry;
using CorePm4MshdGroupingService = WowViewer.Core.PM4.Services.Pm4MshdGroupingService;
using CorePm4MslkEntry = WowViewer.Core.PM4.Models.Pm4MslkEntry;
using CorePm4MsurEntry = WowViewer.Core.PM4.Models.Pm4MsurEntry;
using CorePm4CoordinateModeResolution = WowViewer.Core.PM4.Models.Pm4CoordinateModeResolution;
using CorePm4ObjectGroupKey = WowViewer.Core.PM4.Models.Pm4ObjectGroupKey;
using CorePm4CachedTile = WowViewer.Core.PM4.Caching.Pm4CachedTile;
using CorePm4CachedObject = WowViewer.Core.PM4.Caching.Pm4CachedObject;
using CorePm4CachedConnectorKey = WowViewer.Core.PM4.Caching.Pm4CachedConnectorKey;
using CorePm4CachedLineSegment = WowViewer.Core.PM4.Caching.Pm4CachedLineSegment;
using CorePm4CachedTriangle = WowViewer.Core.PM4.Caching.Pm4CachedTriangle;
using CorePm4PerFileCacheEntry = WowViewer.Core.PM4.Caching.Pm4PerFileCacheEntry;
using CorePm4PerFileCache = WowViewer.Core.PM4.Caching.Pm4PerFileCache;
using CorePm4PerFileCacheService = WowViewer.Core.PM4.Caching.Pm4PerFileCacheService;
using CorePm4PlacementContract = WowViewer.Core.PM4.Services.Pm4PlacementContract;
using CorePm4PlacementMath = WowViewer.Core.PM4.Services.Pm4PlacementMath;
using CorePm4PlacementSolution = WowViewer.Core.PM4.Models.Pm4PlacementSolution;
using Pm4PlanarTransform = WowViewer.Core.PM4.Models.Pm4PlanarTransform;
using CorePm4DocumentReader = WowViewer.Core.PM4.Services.Pm4ResearchReader;
using CorePm4DecodeAuditReport = WowViewer.Core.PM4.Models.Pm4DecodeAuditReport;
using CorePm4ExplorationSnapshot = WowViewer.Core.PM4.Models.Pm4ExplorationSnapshot;
using Pm4CoordinateService = WowViewer.Core.PM4.Services.Pm4CoordinateService;
using CorePm4ObjectHypothesis = WowViewer.Core.PM4.Models.Pm4ObjectHypothesis;
using MprlEntry = WowViewer.Core.PM4.Models.Pm4MprlEntry;
using MslkEntry = WowViewer.Core.PM4.Models.Pm4MslkEntry;
using Pm4VersionFormatter = WowViewer.Core.PM4.Services.Pm4VersionFormatter;
using MsurEntry = WowViewer.Core.PM4.Models.Pm4MsurEntry;
using Pm4File = WowViewer.Core.PM4.Research.Pm4ResearchDocument;
using CorePm4ReferenceAudit = WowViewer.Core.PM4.Models.Pm4ReferenceAudit;
using CorePm4ResearchAuditAnalyzer = WowViewer.Core.PM4.Research.Pm4ResearchAuditAnalyzer;
using CorePm4ResearchHierarchyAnalyzer = WowViewer.Core.PM4.Research.Pm4ResearchHierarchyAnalyzer;
using CorePm4ResearchSnapshotBuilder = WowViewer.Core.PM4.Research.Pm4ResearchSnapshotBuilder;
using CorePm4TileObjectHypothesisReport = WowViewer.Core.PM4.Models.Pm4TileObjectHypothesisReport;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;
using WorldFramePassCoordinator = WowViewer.Core.Runtime.World.Passes.WorldFramePassCoordinator;
using WorldFramePassOptions = WowViewer.Core.Runtime.World.Passes.WorldFramePassOptions;
using WorldFramePasses = WowViewer.Core.Runtime.World.Passes.WorldFramePasses;
using WorldObjectPassCoordinator = WowViewer.Core.Runtime.World.Passes.WorldObjectPassCoordinator;
using WorldObjectPassFrame = WowViewer.Core.Runtime.World.Passes.WorldObjectPassFrame;
using WorldModelBatchGate = WowViewer.Core.Runtime.World.Passes.WorldModelBatchGate;
using WorldModelRenderPath = WowViewer.Core.Runtime.World.Passes.WorldModelRenderPath;
using WorldModelSubmissionOutcome = WowViewer.Core.Runtime.World.Passes.WorldModelSubmissionOutcome;
using WorldModelSubmissionTally = WowViewer.Core.Runtime.World.Passes.WorldModelSubmissionTally;
using VisibleMdxInstance = WowViewer.Core.Runtime.World.Visibility.WorldVisibleMdxEntry;
using VisibleWmoInstance = WowViewer.Core.Runtime.World.Visibility.WorldVisibleWmoEntry;
using WowViewer.Core.Runtime.World;
using WowViewer.Core.Runtime.World.SceneGraph;
using WowViewer.Core.Runtime.World.Visibility;
using WowViewer.Core.World;
using static WoWViewer.Terrain.Pm4OverlayMatching;
using static WoWViewer.Terrain.Pm4OverlayCacheCodec;
using static WoWViewer.Terrain.Pm4OverlayGeometry;
using static WoWViewer.Terrain.Pm4OverlayCoordinates;
using static WoWViewer.Terrain.Pm4OverlayColors;

namespace WoWViewer.Terrain;

/// <summary>
/// PM4 navmesh debug overlay owned by <see cref="WorldScene"/> (Epic 251 U-01 step E1).
/// Moved verbatim out of <c>WorldScene.cs</c>; the scene keeps one field and delegates.
/// State the overlay reads from the world scene comes only through <see cref="IPm4OverlayHost"/>.
/// </summary>
/// <remarks>
/// Split across partial files by responsibility, each under the 2,000-line budget:
/// <c>Pm4OverlayScene.cs</c> (state, load, cache), <c>Pm4OverlayScene.Reports.cs</c>,
/// <c>Pm4OverlayScene.Selection.cs</c>. Pure static helpers live in their own static classes.
/// </remarks>
public sealed partial class Pm4OverlayScene
{
    private readonly IPm4OverlayHost _host;

    internal Pm4OverlayScene(IPm4OverlayHost host, Pm4OverlayCacheService? pm4OverlayCacheService)
    {
        _host = host;
        _pm4OverlayCacheService = pm4OverlayCacheService;
    }

    // Host bridge. These keep the names the moved code used when it lived in WorldScene, so
    // the move needed no body edits; each one reads through IPm4OverlayHost.
    private IDataSource? _dataSource => _host.DataSource;
    private TerrainManager _terrainManager => _host.TerrainManager;
    private WorldAssetManager _assets => _host.Assets;
    private bool _instancesDirty => _host.InstancesDirty;
    private void RebuildInstanceLists() => _host.RebuildInstanceLists();
    private Dictionary<(int, int), List<ObjectInstance>> _tileWmoInstances => _host.TileWmoInstances;
    private Dictionary<(int, int), List<ObjectInstance>> _tileMdxInstances => _host.TileMdxInstances;
    private List<ObjectInstance> _wmoInstances => _host.WmoInstances;
    private bool _hasLastRenderedCameraPosition => _host.HasLastRenderedCameraPosition;
    private Vector3 _lastRenderedCameraPosition => _host.LastRenderedCameraPosition;
    private bool IsHoverPickDistanceAllowed(float distance) => _host.IsHoverPickDistanceAllowed(distance);
    private bool IsHoverPickPositionAllowed(Vector3 worldPosition) => _host.IsHoverPickPositionAllowed(worldPosition);
    private FrustumCuller _frustumCuller => _host.FrustumCuller;
    private const float NoCullRadius = WorldScene.NoCullRadius;
    internal static Vector3 ConvertMprlPositionToWorld(Vector3 refPos) => WorldScene.ConvertMprlPositionToWorld(refPos);
    private static float RayAABBIntersect(Vector3 origin, Vector3 dir, Vector3 bmin, Vector3 bmax) => WorldScene.RayAABBIntersect(origin, dir, bmin, bmax);
    private static void TransformBounds(Vector3 min, Vector3 max, Matrix4x4 m, out Vector3 outMin, out Vector3 outMax)
        => WorldScene.TransformBounds(min, max, m, out outMin, out outMax);
    private static void TransformBounds(Vector3 boundsMin, Vector3 boundsMax, in Matrix4x4 transform, out Vector3 transformedMin, out Vector3 transformedMax)
        => WorldScene.TransformBounds(boundsMin, boundsMax, in transform, out transformedMin, out transformedMax);
    private static bool TryMeasureHoverInfoHit(Vector3 boundsMin, Vector3 boundsMax, Matrix4x4 view, Matrix4x4 proj, float mouseViewportX, float mouseViewportY, float viewportWidth, float viewportHeight, out float distanceSq, out float depth)
        => WorldScene.TryMeasureHoverInfoHit(boundsMin, boundsMax, view, proj, mouseViewportX, mouseViewportY, viewportWidth, viewportHeight, out distanceSq, out depth);

    private static float? JsonFiniteOrNull(float value) => float.IsFinite(value) ? value : null;
    private readonly Pm4OverlayCacheService? _pm4OverlayCacheService;
    // Spec 054: on-disk per-file PM4 overlay cache. Constructed lazily
    // on first PM4 load from the same cache root the per-window cache
    // uses. Holds small per-PM4-file gzip blobs; one entry per file.
    private CorePm4PerFileCacheService? _pm4PerFileDiskCache;
    private const float NoCullRadiusSq = NoCullRadius * NoCullRadius;

    // PM4 debug overlay
    private const int Pm4MaxLinesTotal = int.MaxValue;
    internal const int Pm4MaxLinesPerTile = int.MaxValue;
    private const int Pm4MaxTrianglesTotal = int.MaxValue;
    internal const int Pm4MaxTrianglesPerTile = int.MaxValue;
    private const int Pm4MaxPositionRefsTotal = int.MaxValue;
    private const int Pm4MaxPositionRefsPerTile = int.MaxValue;
    internal const float Pm4MaxEdgeLength = 512f;
    private const int Pm4MinCameraTileRadius = 1;
    private const int Pm4MaxCameraTileRadius = 2;
    private const double Pm4ExpandWindowThresholdMs = 120.0;
    private const double Pm4ShrinkWindowThresholdMs = 300.0;
    private const long Pm4ProgressStatusIntervalMs = 1000;
    private const long Pm4ProgressLogIntervalMs = 5000;
    internal bool _showPm4Overlay;
    internal bool _showPm4SolidOverlay = true;
    internal bool _showPm4ObjectBounds;
    internal bool _showPm4Ck24Bounds;
    internal bool _showPm4GeneratedPlacements;
    internal bool _pm4GeneratedPlacementsTerrainlessOnly = true;
    internal bool _showPm4PlacementZPlane;
    internal bool _showPm4PlacementZForAllObjects;
    internal bool _pm4OverlayIgnoreDepth;
    private bool _pm4FlipAllObjectsY;
    internal bool _showPm4PositionRefs;
    internal bool _showPm4ObjectCentroids;
    internal bool _showPm4MscnNodes;
    internal bool _showPm4MspvNodes;
    internal float _pm4MscnCubeSize = 0.8f;
    internal float _pm4MspvCubeSize = 1.0f;
    internal float _pm4MscnCubeAlpha = 0.95f;
    internal float _pm4MspvCubeAlpha = 0.95f;
    internal float _pm4WireframeLineWidth = 2.5f;
    internal bool _pm4RenderNodesAsCubes = true;
    private bool _pm4SplitCk24ByConnectivity;
    private bool _showPm4Type40 = true;
    private bool _showPm4Type80 = true;
    private bool _showPm4TypeOther = true;
    private bool _pm4SplitCk24ByMscnRef = true;
    // MSPV/MSPI path windows are a vertical planar quad mesh — the walls that stand between the
    // MSUR walkable surfaces. Measured corpus-wide: 98% of windows are exactly 4 indices, 99.6%
    // coplanar, and not one of 598,790 faces has Z as its dominant normal. The viewer has never
    // drawn them, so half the decoded geometry has been invisible.
    private bool _pm4ShowPathWalls = true;
    private Pm4OverlayColorMode _pm4ColorMode = Pm4OverlayColorMode.PlacementZ;
    internal Vector3 _pm4OverlayTranslation = Vector3.Zero;
    internal Vector3 _pm4OverlayRotationDegrees = Vector3.Zero;
    internal Vector3 _pm4OverlayScale = Vector3.One;
    private bool _pm4LoadAttempted;
    private string _pm4Status = "PM4 overlay not loaded.";
    private int _pm4TotalFiles;
    private int _pm4LoadedFiles;
    private int _pm4ObjectCount;
    private int _pm4LineCount;
    private int _pm4TriangleCount;
    private int _pm4RejectedLongEdges;
    internal int _pm4VisibleObjectCount;
    internal int _pm4VisibleLineCount;
    internal int _pm4VisibleTriangleCount;
    private int _pm4PositionRefCount;
    internal int _pm4VisiblePositionRefCount;
    private int _pm4TotalMsurCount;
    private int _pm4DroppedShortIndexCount;
    private int _pm4WallFaceCount;
    private int _pm4DroppedOutOfRangeMsviCount;
    private int _pm4DroppedEmptyComponentCount;
    private float _pm4MinObjectZ;
    private float _pm4MaxObjectZ;
    private int _pm4CameraTileRadius = Pm4MinCameraTileRadius;
    private double _pm4AverageLoadMs = -1.0;
    private (int minTileX, int minTileY, int maxTileX, int maxTileY)? _pm4LoadedCameraWindow;
    private ((int tileX, int tileY, uint ck24, int objectPart) key, (int tileX, int tileY, uint ck24) group)? _pm4GraphInfoCacheKey;
    private Pm4SelectedObjectGraphInfo? _pm4GraphInfoCacheValue;
    private bool _pm4GraphInfoCacheSplitByMscnRef;
    private bool _pm4GraphInfoCacheSplitByConnectivity;

    // Click-freeze / per-frame instrumentation. Toggle via the static
    // Pm4Profiling.Enabled flag. Counts and cumulative milliseconds per hot spot.
    // Logged on a coarser cadence than every call so we don't drown the log.
    private static readonly System.Diagnostics.Stopwatch s_pm4PickSw = new();
    private static long s_pm4PickCallCount;
    private static long s_pm4PickAabbHitCount;
    private static double s_pm4PickTotalMs;
    private static double s_pm4PickMaxMs;
    private static long s_pm4PickReportCount;

    private static readonly System.Diagnostics.Stopwatch s_pm4ResearchSw = new();
    private static long s_pm4ResearchCallCount;
    private static double s_pm4ResearchTotalMs;
    private static double s_pm4ResearchMaxMs;
    private static long s_pm4ResearchReportCount;

    private static readonly System.Diagnostics.Stopwatch s_pm4GraphBuildSw = new();
    private static long s_pm4GraphBuildCallCount;
    private static double s_pm4GraphBuildTotalMs;
    private static double s_pm4GraphBuildMaxMs;
    private static long s_pm4GraphBuildReportCount;
    private static long s_pm4GraphBuildLastObjectCount;
    private static long s_pm4GraphBuildLastRegionCount;

    /// <summary>
    /// Receiver for per-frame graph-build timings reported by the viewer-app
    /// side. Centralises the per-section log so all PM4 hot-spot reports
    /// share one cadence and one tag.
    /// </summary>
    public static class Pm4ProfilingAccumulator
    {
        public static void RecordGraphBuild(double elapsedMs, int walkedObjectCount, int regionCount)
        {
            if (!Pm4Profiling.Enabled) return;
            s_pm4GraphBuildCallCount++;
            s_pm4GraphBuildTotalMs += elapsedMs;
            if (elapsedMs > s_pm4GraphBuildMaxMs) s_pm4GraphBuildMaxMs = elapsedMs;
            s_pm4GraphBuildLastObjectCount = walkedObjectCount;
            s_pm4GraphBuildLastRegionCount = regionCount;
            s_pm4GraphBuildReportCount++;
            if (elapsedMs >= 50.0 || s_pm4GraphBuildReportCount >= 200)
            {
                ViewerLog.Info(ViewerLog.Category.Terrain,
                    $"[PM4-PROFILE] DrawPm4SceneGraph.Build: call={s_pm4GraphBuildCallCount} last={elapsedMs:0.0}ms max={s_pm4GraphBuildMaxMs:0.0}ms avg={s_pm4GraphBuildTotalMs / s_pm4GraphBuildCallCount:0.0}ms walked={walkedObjectCount} regions={regionCount}");
                s_pm4GraphBuildReportCount = 0;
            }
        }
    }

    private readonly HashSet<(int tileX, int tileY)> _pm4KnownMapTiles = new();
    private readonly HashSet<(int tileX, int tileY)> _pm4CoveredMapTiles = new();
    private Task<Pm4OverlayAsyncLoadResult>? _pm4LoadTask;
    private CancellationTokenSource? _pm4LoadCancellation;
    private int _pm4LoadRequestId;
    internal readonly Dictionary<(int tileX, int tileY), List<Pm4OverlayObject>> _pm4TileObjects = new();
    // Per-file in-memory PM4 cache (spec 054). Lets a camera shift or
    // re-visit reuse already-decoded PM4 payloads without re-running
    // BuildPm4TileObjects. Bounded by a simple LRU cap; cleared on
    // ReloadPm4Overlay().
    private readonly CorePm4PerFileCache _pm4PerFileInMemoryCache = new(capacity: 256);
    // MSCN = scene-graph connector anchors; one per MSUR surface (placed via MSUR.MscnRefIndex).
    // See wow-viewer/docs/architecture/pm4-chunk-semantics.md.
    internal readonly Dictionary<(int tileX, int tileY), List<Vector3>> _pm4TileMscnPoints = new();
    // MSPV = path-vertex positions reached via MSPI from MSLK link records. Only present when surfaces are connected.
    internal readonly Dictionary<(int tileX, int tileY), List<Vector3>> _pm4TileMspvPoints = new();
    internal readonly Dictionary<(int tileX, int tileY), Pm4OverlayTileStats> _pm4TileStats = new();
    internal readonly Dictionary<(int tileX, int tileY), List<Vector3>> _pm4TilePositionRefs = new();
    internal readonly Dictionary<string, Pm4ResearchContext> _pm4ResearchBySourcePath = new(StringComparer.OrdinalIgnoreCase);
    internal readonly HashSet<string> _pm4ResearchUnavailablePaths = new(StringComparer.OrdinalIgnoreCase);
    internal readonly Dictionary<(int tileX, int tileY, uint ck24, int objectPart), Pm4OverlayObject> _pm4ObjectLookup = new();
    internal readonly HashSet<(int tileX, int tileY, uint ck24, int objectPart)> _highlightedPm4ObjectKeys = new();
    internal readonly Dictionary<(int tileX, int tileY, uint ck24), (int tileX, int tileY, uint ck24)> _pm4MergedObjectGroupKeys = new();
    internal readonly Dictionary<(int tileX, int tileY, uint ck24), List<(int tileX, int tileY, uint ck24, int objectPart)>> _pm4GroupToObjectKeys = new();
    internal readonly Dictionary<(int tileX, int tileY, uint ck24), (Vector3 min, Vector3 max)> _pm4ObjectGroupBounds = new();
    internal readonly Dictionary<(int tileX, int tileY, uint ck24), (Vector3 min, Vector3 max)> _pm4TileCk24Bounds = new();
    internal readonly Dictionary<(int tileX, int tileY, uint ck24), Vector3> _pm4ObjectTranslations = new();
    internal readonly Dictionary<(int tileX, int tileY, uint ck24), Vector3> _pm4ObjectRotationsDegrees = new();
    internal readonly Dictionary<(int tileX, int tileY, uint ck24), Vector3> _pm4ObjectScales = new();
    internal readonly Dictionary<(int tileX, int tileY, uint ck24), Vector3> _pm4TileCk24Translations = new();
    internal readonly Dictionary<(int tileX, int tileY, uint ck24), Vector3> _pm4TileCk24RotationsDegrees = new();
    internal readonly Dictionary<(int tileX, int tileY, uint ck24), Vector3> _pm4TileCk24Scales = new();
    internal (int tileX, int tileY, uint ck24, int objectPart)? _selectedPm4ObjectKey;
    internal (int tileX, int tileY, uint ck24)? _selectedPm4ObjectGroupKey;
    public bool ShowPm4Overlay
    {
        get => _showPm4Overlay;
        set
        {
            if (_showPm4Overlay == value)
                return;

            _showPm4Overlay = value;
            if (_showPm4Overlay)
                BeginPm4OverlayLoad();
        }
    }

    public bool Pm4LoadAttempted => _pm4LoadAttempted;
    public bool IsPm4Loading => _pm4LoadTask != null && !_pm4LoadTask.IsCompleted;
    public string Pm4Status => _pm4Status;
    public int Pm4TotalFiles => _pm4TotalFiles;
    public int Pm4LoadedFiles => _pm4LoadedFiles;
    public int Pm4ObjectCount => _pm4ObjectCount;

    /// <summary>The tile coordinates of every PM4 currently loaded into the overlay.</summary>
    public IReadOnlyList<(int TileX, int TileY)> LoadedPm4Tiles
        => _pm4TileObjects.Keys.Select(static k => (TileX: k.tileX, TileY: k.tileY)).OrderBy(static t => t.TileX).ThenBy(static t => t.TileY).ToArray();

    public bool LoadLoosePm4File(string filePath)
    {
        if (!File.Exists(filePath))
            return false;

        try
        {
            byte[] bytes = File.ReadAllBytes(filePath);
            Pm4File pm4 = CorePm4DocumentReader.Read(bytes, filePath);

            Pm4CoordinateService.TryParseTileCoordinates(filePath, out int tileX, out int tileY);

            int lineBudget = int.MaxValue;
            int triBudget = int.MaxValue;
            int rejectedLong = 0;
            List<Pm4OverlayObject> objects = BuildPm4TileObjects(
                pm4,
                filePath,
                tileX,
                tileY,
                _pm4SplitCk24ByMscnRef,
                _pm4SplitCk24ByConnectivity,
                _pm4ShowPathWalls,
                ref lineBudget,
                ref triBudget,
                ref rejectedLong,
                out _);

            if (objects.Count == 0)
                return false;

            _pm4TileObjects[(tileX, tileY)] = objects;
            _pm4LoadedCameraWindow = (tileX, tileY, tileX, tileY);
            _pm4LoadAttempted = true;
            _showPm4Overlay = true;
            _pm4Status = $"Loaded loose PM4/PD4 '{Path.GetFileName(filePath)}' ({Pm4VersionFormatter.Format(pm4.Version)}): {objects.Count} objects, {pm4.KnownChunks.Msvt.Count} verts, {pm4.KnownChunks.Msur.Count} surfaces.";
            ViewerLog.Important(ViewerLog.Category.Terrain, "[PM4] " + _pm4Status);
            return true;
        }
        catch (Exception ex)
        {
            _pm4Status = $"Failed to load loose PM4/PD4 '{Path.GetFileName(filePath)}': {ex.Message}";
            ViewerLog.Important(ViewerLog.Category.Terrain, "[PM4] " + _pm4Status);
            return false;
        }
    }
    public int Pm4LineCount => _pm4LineCount;
    public int Pm4TriangleCount => _pm4TriangleCount;
    public int Pm4RejectedLongEdges => _pm4RejectedLongEdges;
    public int Pm4TotalMsurCount => _pm4TotalMsurCount;
    public int Pm4DroppedShortIndexCount => _pm4DroppedShortIndexCount;

    /// <summary>MSPV/MSPI wall faces emitted across the loaded tiles.</summary>
    public int Pm4WallFaceCount => _pm4WallFaceCount;
    public int Pm4DroppedOutOfRangeMsviCount => _pm4DroppedOutOfRangeMsviCount;
    public int Pm4DroppedEmptyComponentCount => _pm4DroppedEmptyComponentCount;
    public int Pm4VisibleObjectCount => _pm4VisibleObjectCount;
    public int Pm4VisibleLineCount => _pm4VisibleLineCount;
    public int Pm4VisibleTriangleCount => _pm4VisibleTriangleCount;
    public int Pm4PositionRefCount => _pm4PositionRefCount;
    public int Pm4VisiblePositionRefCount => _pm4VisiblePositionRefCount;
    public bool ShowPm4SolidOverlay { get => _showPm4SolidOverlay; set => _showPm4SolidOverlay = value; }
    public bool ShowPm4ObjectBounds { get => _showPm4ObjectBounds; set => _showPm4ObjectBounds = value; }
    public bool ShowPm4Ck24Bounds { get => _showPm4Ck24Bounds; set => _showPm4Ck24Bounds = value; }

    /// <summary>
    /// Draws a flat marker at each object's <c>MSUR._0x1C</c> read as a Z height, so the claim that
    /// the field is the producing placement's Z can be checked by eye rather than taken on trust.
    /// </summary>
    /// <remarks>
    /// What to look for. For an object that carries a value, the marker should sit at the BASE of
    /// the object - that is the claim, and if it floats or sinks the claim is wrong for that object.
    /// For an object whose value is 0, the marker drops to world Z = 0, typically far below its
    /// geometry: measured, 98.7% of that population has surface centroids nowhere near zero
    /// (range -656.4..596.2, mean 120.2, only 1.329% within one unit of zero). That visible gap IS
    /// the evidence that 0 is an absent value rather than a real placement height.
    /// </remarks>
    /// <summary>
    /// Draw the placements recovered from PM4 geometry by <c>pm4 generate-placements</c>.
    /// </summary>
    /// <remarks>
    /// These are boxes the file itself never carried: position and extents derived from the navmesh,
    /// with an asset name that is a ranked guess. Colour is confidence, so a firm identification and a
    /// shrug do not look alike. Defaults to showing only tiles with no surviving terrain, since on a
    /// tile that still has its ADT the real placements are already drawn and these merely double them.
    /// </remarks>
    public bool ShowPm4GeneratedPlacements
    {
        get => _showPm4GeneratedPlacements;
        set
        {
            _showPm4GeneratedPlacements = value;
            if (value)
                Pm4GeneratedPlacements.EnsureLoaded();
        }
    }

    public bool Pm4GeneratedPlacementsTerrainlessOnly
    {
        get => _pm4GeneratedPlacementsTerrainlessOnly;
        set => _pm4GeneratedPlacementsTerrainlessOnly = value;
    }

    public int Pm4GeneratedPlacementCount => Pm4GeneratedPlacements.Count;

    public string? Pm4GeneratedPlacementSource => Pm4GeneratedPlacements.SourcePath;

    public bool ShowPm4PlacementZPlane { get => _showPm4PlacementZPlane; set => _showPm4PlacementZPlane = value; }

    /// <summary>
    /// Draw a placement-Z marker for EVERY placed object rather than only the selected one.
    /// </summary>
    /// <remarks>
    /// Off by default and worth leaving off. The marker answers "does this object's recorded height
    /// sit at its base", which is a question about one object; drawing a thousand at once produces a
    /// field of identical cubes that answers nothing and hides the geometry underneath.
    /// </remarks>
    public bool ShowPm4PlacementZForAllObjects { get => _showPm4PlacementZForAllObjects; set => _showPm4PlacementZForAllObjects = value; }
    public bool Pm4OverlayIgnoreDepth { get => _pm4OverlayIgnoreDepth; set => _pm4OverlayIgnoreDepth = value; }
    public bool Pm4FlipAllObjectsY
    {
        get => _pm4FlipAllObjectsY;
        set
        {
            if (_pm4FlipAllObjectsY == value)
                return;

            _pm4FlipAllObjectsY = value;

            // Bake global Y-flip at PM4 decode time to avoid per-frame vertex transform cost.
            if (_pm4LoadAttempted)
                ReloadPm4Overlay();
        }
    }
    public bool ShowPm4PositionRefs { get => _showPm4PositionRefs; set => _showPm4PositionRefs = value; }
    public bool ShowPm4ObjectCentroids { get => _showPm4ObjectCentroids; set => _showPm4ObjectCentroids = value; }
    public bool ShowPm4MscnNodes { get => _showPm4MscnNodes; set => _showPm4MscnNodes = value; }
    public bool ShowPm4MspvNodes { get => _showPm4MspvNodes; set => _showPm4MspvNodes = value; }
    public float Pm4MscnCubeSize { get => _pm4MscnCubeSize; set => _pm4MscnCubeSize = MathF.Max(0.1f, value); }
    public float Pm4MspvCubeSize { get => _pm4MspvCubeSize; set => _pm4MspvCubeSize = MathF.Max(0.1f, value); }
    public float Pm4MscnCubeAlpha { get => _pm4MscnCubeAlpha; set => _pm4MscnCubeAlpha = Math.Clamp(value, 0.1f, 1f); }
    public float Pm4MspvCubeAlpha { get => _pm4MspvCubeAlpha; set => _pm4MspvCubeAlpha = Math.Clamp(value, 0.1f, 1f); }
    public float Pm4WireframeLineWidth { get => _pm4WireframeLineWidth; set => _pm4WireframeLineWidth = MathF.Max(1.0f, value); }
    public bool Pm4RenderNodesAsCubes { get => _pm4RenderNodesAsCubes; set => _pm4RenderNodesAsCubes = value; }
    public bool Pm4SplitCk24ByConnectivity { get => _pm4SplitCk24ByConnectivity; set => _pm4SplitCk24ByConnectivity = value; }
    public bool ShowPm4Type40 { get => _showPm4Type40; set => _showPm4Type40 = value; }
    public bool ShowPm4Type80 { get => _showPm4Type80; set => _showPm4Type80 = value; }
    public bool ShowPm4TypeOther { get => _showPm4TypeOther; set => _showPm4TypeOther = value; }
    public bool Pm4SplitCk24ByMscnRef { get => _pm4SplitCk24ByMscnRef; set => _pm4SplitCk24ByMscnRef = value; }
    public bool Pm4ShowPathWalls { get => _pm4ShowPathWalls; set => _pm4ShowPathWalls = value; }
    public Pm4OverlayColorMode Pm4ColorMode { get => _pm4ColorMode; set => _pm4ColorMode = value; }
    public Vector3 Pm4OverlayTranslation { get => _pm4OverlayTranslation; set => _pm4OverlayTranslation = value; }
    public Vector3 Pm4OverlayRotationDegrees { get => _pm4OverlayRotationDegrees; set => _pm4OverlayRotationDegrees = value; }
    public Vector3 Pm4OverlayScale { get => _pm4OverlayScale; set => _pm4OverlayScale = value; }
    public bool HasSelectedPm4Object => _selectedPm4ObjectKey.HasValue;
    public (int tileX, int tileY, uint ck24, int objectPart)? SelectedPm4ObjectKey => _selectedPm4ObjectKey;

    private const uint Pm4SyntheticZeroCk24GroupMask = 0x80000000u;

    private static (int tileX, int tileY, uint ck24) BuildPm4BaseObjectGroupKey(
        (int tileX, int tileY, uint ck24, int objectPart) objectKey)
    {
        uint groupKey = objectKey.ck24 != 0
            ? objectKey.ck24
            : Pm4SyntheticZeroCk24GroupMask | (uint)objectKey.objectPart;
        return (0, 0, groupKey);
    }

    private (int tileX, int tileY, uint ck24) ResolvePm4ObjectGroupKey((int tileX, int tileY, uint ck24, int objectPart) objectKey)
    {
        var baseGroupKey = BuildPm4BaseObjectGroupKey(objectKey);
        return _pm4MergedObjectGroupKeys.TryGetValue(baseGroupKey, out var mergedGroupKey)
            ? mergedGroupKey
            : baseGroupKey;
    }

    internal bool IsPm4ObjectInGroup(
        (int tileX, int tileY, uint ck24) groupKey,
        (int tileX, int tileY, uint ck24, int objectPart) objectKey)
    {
        return ResolvePm4ObjectGroupKey(objectKey) == groupKey;
    }

    public Vector3 SelectedPm4ObjectTranslation
    {
        get
        {
            if (!_selectedPm4ObjectGroupKey.HasValue)
                return Vector3.Zero;

            return _pm4ObjectTranslations.TryGetValue(_selectedPm4ObjectGroupKey.Value, out Vector3 translation)
                ? translation
                : Vector3.Zero;
        }
        set
        {
            if (!_selectedPm4ObjectGroupKey.HasValue)
                return;

            if (value.LengthSquared() < 0.0001f)
                _pm4ObjectTranslations.Remove(_selectedPm4ObjectGroupKey.Value);
            else
                _pm4ObjectTranslations[_selectedPm4ObjectGroupKey.Value] = value;
        }
    }
    public Vector3 SelectedPm4ObjectRotationDegrees
    {
        get
        {
            if (!_selectedPm4ObjectGroupKey.HasValue)
                return Vector3.Zero;

            return _pm4ObjectRotationsDegrees.TryGetValue(_selectedPm4ObjectGroupKey.Value, out Vector3 rotationDegrees)
                ? rotationDegrees
                : Vector3.Zero;
        }
        set
        {
            if (!_selectedPm4ObjectGroupKey.HasValue)
                return;

            if (IsNearZeroVector(value))
                _pm4ObjectRotationsDegrees.Remove(_selectedPm4ObjectGroupKey.Value);
            else
                _pm4ObjectRotationsDegrees[_selectedPm4ObjectGroupKey.Value] = value;
        }
    }
    public Vector3 SelectedPm4ObjectScale
    {
        get
        {
            if (!_selectedPm4ObjectGroupKey.HasValue)
                return Vector3.One;

            return _pm4ObjectScales.TryGetValue(_selectedPm4ObjectGroupKey.Value, out Vector3 scale)
                ? scale
                : Vector3.One;
        }
        set
        {
            if (!_selectedPm4ObjectGroupKey.HasValue)
                return;

            Vector3 sanitized = SanitizeScale(value);
            if (IsNearOneVector(sanitized))
                _pm4ObjectScales.Remove(_selectedPm4ObjectGroupKey.Value);
            else
                _pm4ObjectScales[_selectedPm4ObjectGroupKey.Value] = sanitized;
        }
    }
    public uint? SelectedPm4RawCk24 => _selectedPm4ObjectKey?.ck24;
    public (int tileX, int tileY, uint ck24)? SelectedPm4TileCk24Key
        => _selectedPm4ObjectKey.HasValue
            ? (_selectedPm4ObjectKey.Value.tileX, _selectedPm4ObjectKey.Value.tileY, _selectedPm4ObjectKey.Value.ck24)
            : null;
    public Vector3 SelectedPm4Ck24LayerTranslation
    {
        get
        {
            if (!SelectedPm4TileCk24Key.HasValue)
                return Vector3.Zero;

            return _pm4TileCk24Translations.TryGetValue(SelectedPm4TileCk24Key.Value, out Vector3 translation)
                ? translation
                : Vector3.Zero;
        }
        set
        {
            if (!SelectedPm4TileCk24Key.HasValue)
                return;

            if (value.LengthSquared() < 0.0001f)
                _pm4TileCk24Translations.Remove(SelectedPm4TileCk24Key.Value);
            else
                _pm4TileCk24Translations[SelectedPm4TileCk24Key.Value] = value;
        }
    }
    public Vector3 SelectedPm4Ck24LayerRotationDegrees
    {
        get
        {
            if (!SelectedPm4TileCk24Key.HasValue)
                return Vector3.Zero;

            return _pm4TileCk24RotationsDegrees.TryGetValue(SelectedPm4TileCk24Key.Value, out Vector3 rotationDegrees)
                ? rotationDegrees
                : Vector3.Zero;
        }
        set
        {
            if (!SelectedPm4TileCk24Key.HasValue)
                return;

            if (IsNearZeroVector(value))
                _pm4TileCk24RotationsDegrees.Remove(SelectedPm4TileCk24Key.Value);
            else
                _pm4TileCk24RotationsDegrees[SelectedPm4TileCk24Key.Value] = value;
        }
    }
    public Vector3 SelectedPm4Ck24LayerScale
    {
        get
        {
            if (!SelectedPm4TileCk24Key.HasValue)
                return Vector3.One;

            return _pm4TileCk24Scales.TryGetValue(SelectedPm4TileCk24Key.Value, out Vector3 scale)
                ? scale
                : Vector3.One;
        }
        set
        {
            if (!SelectedPm4TileCk24Key.HasValue)
                return;

            Vector3 sanitized = SanitizeScale(value);
            if (IsNearOneVector(sanitized))
                _pm4TileCk24Scales.Remove(SelectedPm4TileCk24Key.Value);
            else
                _pm4TileCk24Scales[SelectedPm4TileCk24Key.Value] = sanitized;
        }
    }
    public float Pm4OverlayYawDegrees
    {
        get => _pm4OverlayRotationDegrees.Z;
        set => _pm4OverlayRotationDegrees = new Vector3(_pm4OverlayRotationDegrees.X, _pm4OverlayRotationDegrees.Y, value);
    }
    public IReadOnlyCollection<Pm4OverlayTileStats> Pm4TileStats => _pm4TileStats.Values;

    public bool TryGetSelectedPm4Ck24LayerStats(out int tileCount, out int objectCount)
    {
        tileCount = 0;
        objectCount = 0;

        var tileCk24Key = SelectedPm4TileCk24Key;
        if (!tileCk24Key.HasValue)
            return false;

        foreach (var objectKey in _pm4ObjectLookup.Keys)
        {
            if (objectKey.tileX != tileCk24Key.Value.tileX
                || objectKey.tileY != tileCk24Key.Value.tileY
                || objectKey.ck24 != tileCk24Key.Value.ck24)
                continue;

            objectCount++;
        }

        tileCount = objectCount > 0 ? 1 : 0;
        return objectCount > 0;
    }

    /// <summary>
    /// Legacy match/correlation reports are scoped to this many tiles around the camera tile; an
    /// unscoped build walked every loaded tile and froze the render thread on whole-map loads.
    /// </summary>
    internal const int Pm4MatchCameraTileRadius = 1;

    private void BeginPm4OverlayLoad(bool ignoreCache = false)
    {
        if (_dataSource == null)
        {
            _pm4LoadAttempted = true;
            _pm4Status = "PM4 unavailable: no data source.";
            return;
        }

        if (!ignoreCache && _pm4LoadTask != null && !_pm4LoadTask.IsCompleted)
            return;

        ReleasePm4LoadCancellation(cancelPendingLoad: true);

        _pm4LoadAttempted = true;
        int requestId = ++_pm4LoadRequestId;
        var selectedObjectKey = _selectedPm4ObjectKey;
        var cancellation = new CancellationTokenSource();
        _pm4LoadCancellation = cancellation;
        _pm4Status = ignoreCache
            ? "PM4 reload queued: decoding map-wide overlay in background..."
            : "PM4 loading: decoding map-wide overlay in background...";
        _pm4LoadTask = Task.Run(() => LoadPm4OverlayAsync(requestId, ignoreCache, selectedObjectKey, cancellation.Token), cancellation.Token);
    }

    internal void TryFinalizePm4OverlayLoad()
    {
        Task<Pm4OverlayAsyncLoadResult>? loadTask = _pm4LoadTask;
        if (loadTask == null || !loadTask.IsCompleted)
            return;

        _pm4LoadTask = null;

        Pm4OverlayAsyncLoadResult result;
        try
        {
            result = loadTask.GetAwaiter().GetResult();
        }
        catch (Exception ex)
        {
            _pm4Status = $"PM4 load failed: {ex.Message}";
            ViewerLog.Important(ViewerLog.Category.Terrain, "[PM4] " + _pm4Status);
            return;
        }

        if (result.RequestId != _pm4LoadRequestId || result.Cancelled)
            return;

        bool replaceExisting = !_pm4LoadedCameraWindow.HasValue || _pm4TileObjects.Count == 0;

        if (result.KnownMapTiles.Count > 0)
            _pm4KnownMapTiles.UnionWith(result.KnownMapTiles);

        if (result.CacheData != null)
        {
            if (replaceExisting)
                ClearPm4OverlayRuntimeState();

            MergePm4OverlayFromCache(result.CacheData);
            RestoreSelectedPm4Object(result.SelectedObjectKey);
            UpdatePm4AdaptiveWindow(result.LoadElapsedMs);
        }

        if (result.CoveredMapTiles.Count > 0)
            _pm4CoveredMapTiles.UnionWith(result.CoveredMapTiles);

        if (result.LoadedCameraWindow.HasValue)
            ExpandPm4LoadedCameraWindow(result.LoadedCameraWindow.Value);

        _pm4Status = result.StatusMessage;
        LogPm4FinalStatus(_pm4Status);
    }

    private void LogPm4FinalStatus(string status)
    {
        if (string.IsNullOrWhiteSpace(status) || ShouldSuppressPm4FinalStatusLog(status))
            return;

        if (status.StartsWith("PM4 ready:", StringComparison.Ordinal)
            || status.StartsWith("PM4 load failed:", StringComparison.Ordinal)
            || status.StartsWith("PM4 unavailable:", StringComparison.Ordinal))
        {
            ViewerLog.Important(ViewerLog.Category.Terrain, "[PM4] " + status);
            return;
        }

        ViewerLog.Info(ViewerLog.Category.Terrain, "[PM4] " + status);
    }

    internal void ReleasePm4LoadCancellation(bool cancelPendingLoad)
    {
        CancellationTokenSource? cancellation = _pm4LoadCancellation;
        _pm4LoadCancellation = null;
        if (cancellation == null)
            return;

        if (cancelPendingLoad)
            cancellation.Cancel();

        cancellation.Dispose();
    }

    private void ReportPm4LoadProgress(
        int requestId,
        string phase,
        int processedFiles,
        int totalFiles,
        int loadedFiles,
        int objectCount,
        int lineCount,
        int triangleCount,
        int readFailed,
        int decodeFailed,
        int zeroObjectFiles,
        int memCacheHits,
        int diskCacheHits,
        string? currentPath,
        bool emitLog)
    {
        if (requestId != _pm4LoadRequestId)
            return;

        string currentFileSuffix = string.IsNullOrWhiteSpace(currentPath)
            ? string.Empty
            : $", file={Path.GetFileName(currentPath)}";
        string status =
            $"PM4 loading: {phase} {processedFiles}/{totalFiles} files, loaded={loadedFiles}, objects={objectCount}, lines={lineCount}, tris={triangleCount}, readFail={readFailed}, decodeFail={decodeFailed}, zero={zeroObjectFiles} (mem-cache {memCacheHits} hit, disk-cache {diskCacheHits} hit){currentFileSuffix}";
        _pm4Status = status;

        if (emitLog)
            ViewerLog.Info(ViewerLog.Category.Terrain, "[PM4] " + status);
    }

    private Pm4OverlayAsyncLoadResult LoadPm4OverlayAsync(
        int requestId,
        bool ignoreCache,
        (int tileX, int tileY, uint ck24, int objectPart)? selectedObjectKey,
        CancellationToken cancellationToken)
    {
        try
        {
            if (_dataSource == null)
                return new Pm4OverlayAsyncLoadResult(requestId, null, null, [], [], selectedObjectKey, 0.0, "PM4 unavailable: no data source.", cancelled: false);

            string mapName = _terrainManager.MapName;
            List<string> mapPm4Candidates = _dataSource
                .GetFileList(".pm4")
                .Where(path => IsMapPm4Path(path, mapName))
                .OrderBy(path => path, StringComparer.OrdinalIgnoreCase)
                .ToList();

            int mapPm4CandidateCount = mapPm4Candidates.Count;
            if (mapPm4CandidateCount == 0)
                return new Pm4OverlayAsyncLoadResult(requestId, null, null, [], [], selectedObjectKey, 0.0, $"PM4: no files found for map '{mapName}'.", cancelled: false);

            int tileParseRejected = 0;
            int tileRangeRejected = 0;
            var pm4Candidates = new List<(string path, int tileX, int tileY)>();
            foreach (string pm4Path in mapPm4Candidates)
            {
                cancellationToken.ThrowIfCancellationRequested();

                if (!Pm4CoordinateService.TryParseTileCoordinates(pm4Path, out int fileTileX, out int fileTileY))
                {
                    tileParseRejected++;
                    continue;
                }

                if (!TryMapPm4FileTileToTerrainTile(fileTileX, fileTileY, out int effectiveTileX, out int effectiveTileY))
                {
                    tileRangeRejected++;
                    continue;
                }

                pm4Candidates.Add((pm4Path, effectiveTileX, effectiveTileY));
            }

            HashSet<(int tileX, int tileY)> knownMapTiles = pm4Candidates
                .Select(static candidate => (candidate.tileX, candidate.tileY))
                .ToHashSet();

            int totalFiles = pm4Candidates.Count;
            if (totalFiles == 0)
            {
                return new Pm4OverlayAsyncLoadResult(
                    requestId,
                    null,
                    null,
                    knownMapTiles,
                    [],
                    selectedObjectKey,
                    0.0,
                    $"PM4: 0/{mapPm4CandidateCount} valid map files after tile mapping (tileParse={tileParseRejected}, tileRange={tileRangeRejected}).",
                    cancelled: false);
            }

            Vector3 loadAnchorCameraPosition = GetPm4LoadAnchorCameraPosition();
            var cameraWindow = GetPm4CameraWindow(loadAnchorCameraPosition, _pm4CameraTileRadius);
            List<(string path, int tileX, int tileY)> loadCandidates = pm4Candidates
                .Where(candidate => IsPm4TileInsideCameraWindow(candidate.tileX, candidate.tileY, cameraWindow))
                .ToList();

            if (loadCandidates.Count == 0)
            {
                return new Pm4OverlayAsyncLoadResult(
                    requestId,
                    null,
                    cameraWindow,
                    knownMapTiles,
                    [],
                    selectedObjectKey,
                    0.0,
                    $"PM4: no files intersect camera window ({cameraWindow.minTileX}..{cameraWindow.maxTileX}, {cameraWindow.minTileY}..{cameraWindow.maxTileY}) out of {totalFiles} valid map files.",
                    cancelled: false);
            }

            HashSet<(int tileX, int tileY)> loadCandidateTiles = loadCandidates
                .Select(static candidate => (candidate.tileX, candidate.tileY))
                .ToHashSet();

            if (ignoreCache && _pm4OverlayCacheService != null)
            {
                if (!_pm4OverlayCacheService.TryDelete(mapName, out string? cacheDeleteError) && !string.IsNullOrWhiteSpace(cacheDeleteError))
                    ViewerLog.Debug(ViewerLog.Category.Terrain, $"[PM4] {cacheDeleteError}");
            }

            string candidateSignature = Pm4OverlayCacheService.BuildCandidateSignature(
                _dataSource,
                loadCandidates.Select(static candidate => candidate.path).ToList(),
                _pm4SplitCk24ByMscnRef,
                _pm4SplitCk24ByConnectivity,
                _pm4ShowPathWalls);
            var loadStopwatch = Stopwatch.StartNew();
            string? cacheLoadError = null;
            if (!ignoreCache
                && _pm4OverlayCacheService != null
                && _pm4OverlayCacheService.TryLoad(mapName, candidateSignature, out Pm4OverlayCacheData? cachedOverlay, out cacheLoadError)
                && cachedOverlay != null)
            {
                loadStopwatch.Stop();
                return new Pm4OverlayAsyncLoadResult(
                    requestId,
                    cachedOverlay,
                    cameraWindow,
                    knownMapTiles,
                    loadCandidateTiles,
                    selectedObjectKey,
                    loadStopwatch.Elapsed.TotalMilliseconds,
                    $"PM4 ready: {cachedOverlay.LoadedFiles}/{cachedOverlay.TotalFiles} camera-window files restored from disk cache for ({cameraWindow.minTileX}..{cameraWindow.maxTileX}, {cameraWindow.minTileY}..{cameraWindow.maxTileY}), avg {_pm4AverageLoadMs:0} ms, next radius {_pm4CameraTileRadius}, from {mapPm4CandidateCount} map files, {cachedOverlay.ObjectCount} objects, {cachedOverlay.LineCount} lines, {cachedOverlay.TriangleCount} triangles, {cachedOverlay.PositionRefCount} refs, {cachedOverlay.RejectedLongEdges} long edges rejected, {loadStopwatch.ElapsedMilliseconds} ms.",
                    cancelled: false);
            }

            if (!string.IsNullOrWhiteSpace(cacheLoadError))
                ViewerLog.Debug(ViewerLog.Category.Terrain, $"[PM4] {cacheLoadError}");

            _pm4Status = $"PM4 loading: decoding {loadCandidates.Count} camera-window files (per-file cache active)...";

            int remainingLineBudget = Pm4MaxLinesTotal;
            int remainingTriangleBudget = Pm4MaxTrianglesTotal;
            int remainingPositionRefBudget = Pm4MaxPositionRefsTotal;
            int loadedFiles = 0;
            int objectCount = 0;
            int lineCount = 0;
            int triangleCount = 0;
            int positionRefCount = 0;
            int rejectedLongEdgesTotal = 0;
            int readFailed = 0;
            int decodeFailed = 0;
            int zeroObjectFiles = 0;
            int memCacheHits = 0;
            int memCacheMisses = 0;
            int diskCacheHits = 0;
            float minObjectZ = float.MaxValue;
            float maxObjectZ = float.MinValue;
            var tileCandidateCounts = loadCandidates
                .GroupBy(static candidate => (candidate.tileX, candidate.tileY))
                .ToDictionary(static group => group.Key, static group => group.Count());
            var tileSatisfiedCounts = tileCandidateCounts.Keys.ToDictionary(static tile => tile, static _ => 0);
            var tileObjects = new Dictionary<(int tileX, int tileY), List<Pm4OverlayObject>>();
            var tilePositionRefs = new Dictionary<(int tileX, int tileY), List<Vector3>>();
            var progressStopwatch = Stopwatch.StartNew();
            long lastStatusReportMs = -Pm4ProgressStatusIntervalMs;
            long lastLogReportMs = -Pm4ProgressLogIntervalMs;
            int processedFiles = 0;

            foreach (var candidate in loadCandidates)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (remainingLineBudget <= 0)
                    break;

                string pm4Path = candidate.path;
                int effectiveTileX = candidate.tileX;
                int effectiveTileY = candidate.tileY;
                processedFiles++;

                byte[]? bytes = _dataSource.ReadFile(pm4Path);
                if (bytes == null)
                {
                    readFailed++;
                    long readFailElapsedMs = progressStopwatch.ElapsedMilliseconds;
                    if (readFailElapsedMs - lastStatusReportMs >= Pm4ProgressStatusIntervalMs)
                    {
                        bool emitLog = readFailElapsedMs - lastLogReportMs >= Pm4ProgressLogIntervalMs;
                        ReportPm4LoadProgress(requestId, "reading", processedFiles, loadCandidates.Count, loadedFiles, objectCount, lineCount, triangleCount, readFailed, decodeFailed, zeroObjectFiles, memCacheHits, diskCacheHits, pm4Path, emitLog);
                        lastStatusReportMs = readFailElapsedMs;
                        if (emitLog)
                            lastLogReportMs = readFailElapsedMs;
                    }
                    continue;
                }

                if (bytes.Length == 0)
                {
                    tileSatisfiedCounts[(effectiveTileX, effectiveTileY)]++;
                    zeroObjectFiles++;
                    ViewerLog.Debug(ViewerLog.Category.Terrain,
                        $"[PM4] Skipping empty PM4 carrier '{pm4Path}' for tile ({effectiveTileX},{effectiveTileY}).");
                    continue;
                }

                // Spec 054: per-file in-memory cache check. A hit lets the
                // camera-window load skip the per-tile BuildPm4TileObjects
                // work entirely for files already decoded in this session.
                // The stamp folds (file length, split-flag bits) so a
                // content edit invalidates the entry and a split-flag
                // toggle does too. The on-disk per-file cache below uses
                // the loose-file write-tick for stronger stamp coverage
                // when the data source exposes it.
                string normalizedPm4Path = pm4Path.Replace('\\', '/');
                long memStamp = ((_pm4ShowPathWalls ? 1L : 0L) << 33)
                    | ((_pm4SplitCk24ByMscnRef ? 1L : 0L) << 32)
                    | (_pm4SplitCk24ByConnectivity ? 1L : 0L);
                if (_pm4PerFileInMemoryCache.TryGet(normalizedPm4Path, bytes.Length, memStamp, out CorePm4PerFileCacheEntry? cachedEntry)
                    && cachedEntry != null
                    && cachedEntry.Tiles.Count > 0)
                {
                    int cachedObjectCount = ApplyCachedTilesToTileDictionaries(
                        cachedEntry,
                        effectiveTileX,
                        effectiveTileY,
                        tileObjects,
                        tilePositionRefs,
                        ref minObjectZ,
                        ref maxObjectZ,
                        ref objectCount,
                        ref lineCount,
                        ref triangleCount,
                        ref positionRefCount);
                    if (cachedObjectCount > 0)
                    {
                        loadedFiles++;
                        memCacheHits++;
                        tileSatisfiedCounts[(effectiveTileX, effectiveTileY)]++;
                        ViewerLog.Debug(
                            ViewerLog.Category.Terrain,
                            $"[PM4] Per-file in-memory cache hit for '{pm4Path}' ({cachedObjectCount} objects, {cachedEntry.Tiles.Count} tiles).");
                        continue;
                    }
                }
                memCacheMisses++;

                // Spec 054: per-file on-disk cache check. Falls through
                // when the in-memory cache misses. We use the loose-file
                // write-tick as the file stamp when available (falls
                // back to 0 for MPQ-only data sources).
                CorePm4PerFileCacheService? onDiskCache = EnsurePerFileDiskCache(mapName);
                if (onDiskCache != null
                    && TryReadPerFileDiskCache(onDiskCache, normalizedPm4Path, bytes.Length, out CorePm4PerFileCacheEntry? diskCachedEntry)
                    && diskCachedEntry != null
                    && diskCachedEntry.Tiles.Count > 0)
                {
                    int cachedObjectCount = ApplyCachedTilesToTileDictionaries(
                        diskCachedEntry,
                        effectiveTileX,
                        effectiveTileY,
                        tileObjects,
                        tilePositionRefs,
                        ref minObjectZ,
                        ref maxObjectZ,
                        ref objectCount,
                        ref lineCount,
                        ref triangleCount,
                        ref positionRefCount);
                    if (cachedObjectCount > 0)
                    {
                        loadedFiles++;
                        diskCacheHits++;
                        tileSatisfiedCounts[(effectiveTileX, effectiveTileY)]++;
                        ViewerLog.Debug(
                            ViewerLog.Category.Terrain,
                            $"[PM4] Per-file on-disk cache hit for '{pm4Path}' ({cachedObjectCount} objects, {diskCachedEntry.Tiles.Count} tiles).");
                        continue;
                    }
                }

                try
                {
                    Pm4File pm4 = CorePm4DocumentReader.Read(bytes, pm4Path);
                    int rejectedLongEdges = 0;
                    List<Pm4OverlayObject> objects = BuildPm4TileObjects(
                        pm4,
                        pm4Path,
                        effectiveTileX,
                        effectiveTileY,
                _pm4SplitCk24ByMscnRef,
                        _pm4SplitCk24ByConnectivity,
                        _pm4ShowPathWalls,
                        ref remainingLineBudget,
                        ref remainingTriangleBudget,
                        ref rejectedLongEdges,
                        out Pm4TileBuildDiagnostics fileDiagnostics);
                    _pm4TotalMsurCount += fileDiagnostics.TotalMsurCount;
                    _pm4DroppedShortIndexCount += fileDiagnostics.DroppedShortIndexCount;
                    _pm4WallFaceCount += fileDiagnostics.WallFaceCount;
                    _pm4DroppedOutOfRangeMsviCount += fileDiagnostics.DroppedOutOfRangeMsviCount;
                    _pm4DroppedEmptyComponentCount += fileDiagnostics.DroppedEmptyComponentCount;
                    if (objects.Count == 0)
                    {
                        tileSatisfiedCounts[(effectiveTileX, effectiveTileY)]++;
                        zeroObjectFiles++;
                        continue;
                    }

                    if (tileObjects.TryGetValue((effectiveTileX, effectiveTileY), out List<Pm4OverlayObject>? existingObjects))
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
                        tileObjects[(effectiveTileX, effectiveTileY)] = objects;
                    }

                    foreach (Pm4OverlayObject obj in objects)
                    {
                        minObjectZ = MathF.Min(minObjectZ, obj.Center.Z);
                        maxObjectZ = MathF.Max(maxObjectZ, obj.Center.Z);
                    }

                    // Store MSCN points in global world space (X↔Y swapped relative to tile coords)
                    // MSCN/MSPV extraction removed from construction path — too slow.
                    // Extracted on-demand via EnsurePm4MscnData() / EnsurePm4MspvData() when the
                    // "MSCN Nodes" / "MSPV Nodes" checkboxes are enabled. See
                    // wow-viewer/docs/architecture/pm4-chunk-semantics.md for what these streams are.

                    if (remainingPositionRefBudget > 0)
                    {
                        List<Vector3> positionRefs = BuildPm4PositionRefMarkers(pm4, Math.Min(Pm4MaxPositionRefsPerTile, remainingPositionRefBudget));
                        if (positionRefs.Count > 0)
                        {
                            if (tilePositionRefs.TryGetValue((effectiveTileX, effectiveTileY), out List<Vector3>? existingPositionRefs))
                                existingPositionRefs.AddRange(positionRefs);
                            else
                                tilePositionRefs[(effectiveTileX, effectiveTileY)] = positionRefs;

                            positionRefCount += positionRefs.Count;
                            remainingPositionRefBudget -= positionRefs.Count;
                        }
                    }

                    tileSatisfiedCounts[(effectiveTileX, effectiveTileY)]++;
                    loadedFiles++;
                    objectCount += objects.Count;
                    lineCount += objects.Sum(obj => obj.Lines.Count);
                    triangleCount += objects.Sum(obj => obj.Triangles.Count);
                    rejectedLongEdgesTotal += rejectedLongEdges;

                    // Spec 054: store the per-file payload in the
                    // in-memory per-file cache so a future camera-window
                    // shift that touches this same file can skip the
                    // BuildPm4TileObjects + budget enforcement work.
                    StorePerFileInMemoryCache(
                        normalizedPm4Path,
                        bytes.Length,
                        effectiveTileX,
                        effectiveTileY,
                        objects,
                        tilePositionRefs,
                        _pm4SplitCk24ByMscnRef,
                        _pm4SplitCk24ByConnectivity);

                    // Spec 054: also persist to the on-disk per-file
                    // cache. Best-effort; a failure here does not break
                    // the load, it just means the next session's reload
                    // of this file will decode fresh instead of reading
                    // from disk. The on-disk entry uses the loose-file
                    // write-tick as its stamp (falls back to 0 for MPQ
                    // data sources where stamps are not available).
                    CorePm4PerFileCacheService? perFileDiskCache = EnsurePerFileDiskCache(mapName);
                    if (perFileDiskCache != null)
                    {
                        long diskStamp = 0L;
                        if (_dataSource != null
                            && Pm4OverlayCacheService.TryGetLooseFileStamp(_dataSource, pm4Path, out _, out long looseTicks))
                        {
                            diskStamp = looseTicks;
                        }

                        perFileDiskCache.Write(
                            normalizedPm4Path,
                            new CorePm4PerFileCacheEntry(
                                FileLength: bytes.Length,
                                LastWriteTicks: diskStamp,
                                Tiles: new[]
                                {
                                    new CorePm4CachedTile(
                                        TileX: effectiveTileX,
                                        TileY: effectiveTileY,
                                        PositionRefs: tilePositionRefs.TryGetValue((effectiveTileX, effectiveTileY), out List<Vector3>? refs)
                                            ? new List<Vector3>(refs)
                                            : new List<Vector3>(),
                                        Objects: BuildCachedObjectsForDiskWrite(objects))
                                }));
                    }
                }
                catch (Exception ex)
                {
                    decodeFailed++;
                    ViewerLog.Debug(ViewerLog.Category.Terrain, $"[PM4] Failed to decode '{pm4Path}': {ex.Message}");
                }

                long elapsedMs = progressStopwatch.ElapsedMilliseconds;
                if (elapsedMs - lastStatusReportMs >= Pm4ProgressStatusIntervalMs || processedFiles == loadCandidates.Count)
                {
                    bool emitLog = elapsedMs - lastLogReportMs >= Pm4ProgressLogIntervalMs || processedFiles == loadCandidates.Count;
                    ReportPm4LoadProgress(requestId, "decoding", processedFiles, loadCandidates.Count, loadedFiles, objectCount, lineCount, triangleCount, readFailed, decodeFailed, zeroObjectFiles, memCacheHits, diskCacheHits, pm4Path, emitLog);
                    lastStatusReportMs = elapsedMs;
                    if (emitLog)
                        lastLogReportMs = elapsedMs;
                }
            }

            HashSet<(int tileX, int tileY)> coveredMapTiles = tileCandidateCounts
                .Where(entry => tileSatisfiedCounts[entry.Key] >= entry.Value)
                .Select(static entry => entry.Key)
                .ToHashSet();

            if (loadedFiles == 0)
            {
                return new Pm4OverlayAsyncLoadResult(
                    requestId,
                    null,
                    cameraWindow,
                    knownMapTiles,
                    coveredMapTiles,
                    selectedObjectKey,
                    loadStopwatch.Elapsed.TotalMilliseconds,
                    $"PM4: {loadCandidates.Count}/{totalFiles} camera-window files found, none decoded into overlay data for ({cameraWindow.minTileX}..{cameraWindow.maxTileX}, {cameraWindow.minTileY}..{cameraWindow.maxTileY}) (tileParse={tileParseRejected}, tileRange={tileRangeRejected}, read={readFailed}, decode={decodeFailed}, zeroObjects={zeroObjectFiles}).",
                    cancelled: false);
            }

            if (minObjectZ > maxObjectZ)
            {
                minObjectZ = 0f;
                maxObjectZ = 1f;
            }

            loadStopwatch.Stop();
            Pm4OverlayCacheData cacheData = BuildPm4OverlayCacheData(
                mapName,
                candidateSignature,
                totalFiles,
                loadedFiles,
                objectCount,
                lineCount,
                triangleCount,
                positionRefCount,
                rejectedLongEdgesTotal,
                minObjectZ,
                maxObjectZ,
                tileObjects,
                tilePositionRefs);
            if (_pm4OverlayCacheService != null)
            {
                if (!_pm4OverlayCacheService.TrySave(cacheData, out string? cacheSaveError) && !string.IsNullOrWhiteSpace(cacheSaveError))
                    ViewerLog.Debug(ViewerLog.Category.Terrain, $"[PM4] {cacheSaveError}");
            }

            return new Pm4OverlayAsyncLoadResult(
                requestId,
                cacheData,
                cameraWindow,
                knownMapTiles,
                coveredMapTiles,
                selectedObjectKey,
                loadStopwatch.Elapsed.TotalMilliseconds,
                $"PM4 ready: {loadedFiles}/{loadCandidates.Count} camera-window files (mem-cache {memCacheHits} hit, {memCacheMisses} fresh-decode) for ({cameraWindow.minTileX}..{cameraWindow.maxTileX}, {cameraWindow.minTileY}..{cameraWindow.maxTileY}), avg {_pm4AverageLoadMs:0} ms, next radius {_pm4CameraTileRadius}, from {mapPm4CandidateCount} map files, {objectCount} objects, {lineCount} lines, {triangleCount} triangles, {positionRefCount} refs, {rejectedLongEdgesTotal} long edges rejected, {loadStopwatch.ElapsedMilliseconds} ms.",
                cancelled: false);
        }
        catch (OperationCanceledException)
        {
            return new Pm4OverlayAsyncLoadResult(requestId, null, null, [], [], selectedObjectKey, 0.0, "PM4 load cancelled.", cancelled: true);
        }
        catch (Exception ex)
        {
            return new Pm4OverlayAsyncLoadResult(requestId, null, null, [], [], selectedObjectKey, 0.0, $"PM4 load failed: {ex.Message}", cancelled: false);
        }
    }

    private void ClearPm4OverlayRuntimeState()
    {
        _pm4LoadedCameraWindow = null;
        _pm4CoveredMapTiles.Clear();
        _pm4KnownMapTiles.Clear();
        _pm4TileObjects.Clear();
        _pm4TileMscnPoints.Clear();
        _pm4TileMspvPoints.Clear();
        _pm4TileStats.Clear();
        _pm4TilePositionRefs.Clear();
        _pm4ResearchBySourcePath.Clear();
        _pm4ResearchUnavailablePaths.Clear();
        _pm4ObjectLookup.Clear();
        _pm4MergedObjectGroupKeys.Clear();
        _pm4GroupToObjectKeys.Clear();
        _pm4ObjectGroupBounds.Clear();
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
    }

    private static Pm4OverlayCacheData BuildPm4OverlayCacheData(
        string mapName,
        string candidateSignature,
        int totalFiles,
        int loadedFiles,
        int objectCount,
        int lineCount,
        int triangleCount,
        int positionRefCount,
        int rejectedLongEdges,
        float minObjectZ,
        float maxObjectZ,
        Dictionary<(int tileX, int tileY), List<Pm4OverlayObject>> tileObjects,
        Dictionary<(int tileX, int tileY), List<Vector3>> tilePositionRefs)
    {
        var tiles = new List<Pm4OverlayCacheTile>(tileObjects.Count);
        foreach (var tileEntry in tileObjects.OrderBy(static entry => entry.Key.tileX).ThenBy(static entry => entry.Key.tileY))
        {
            List<Vector3> positionRefs = tilePositionRefs.TryGetValue(tileEntry.Key, out List<Vector3>? existingPositionRefs)
                ? existingPositionRefs
                : new List<Vector3>();
            var objects = new List<Pm4OverlayCacheObject>(tileEntry.Value.Count);
            for (int i = 0; i < tileEntry.Value.Count; i++)
            {
                Pm4OverlayObject obj = tileEntry.Value[i];
                objects.Add(new Pm4OverlayCacheObject(
                    obj.SourcePath,
                    obj.MshdField00,
                    obj.MshdRegionId,
                    obj.MshdField08,
                    obj.Ck24,
                    obj.Ck24Type,
                    obj.ObjectPartId,
                    obj.LinkGroupObjectId,
                    obj.LinkedPositionRefCount,
                    obj.LinkedPositionRefSummary,
                    obj.Lines,
                    obj.Triangles,
                    obj.SurfaceCount,
                    obj.TotalIndexCount,
                    obj.DominantGroupKey,
                    obj.DominantAttributeMask,
                    obj.DominantMscnRefIndex,
                    obj.AverageSurfaceHeight,
                    obj.PlacementAnchor,
                    obj.BaseRotationRadians,
                    obj.PlanarTransform,
                    obj.BoundsMin,
                    obj.BoundsMax,
                    obj.ConnectorKeys.ToList()));
            }

            tiles.Add(new Pm4OverlayCacheTile(tileEntry.Key.tileX, tileEntry.Key.tileY, objects, positionRefs));
        }

        return new Pm4OverlayCacheData(
            mapName,
            candidateSignature,
            totalFiles,
            loadedFiles,
            objectCount,
            lineCount,
            triangleCount,
            positionRefCount,
            rejectedLongEdges,
            minObjectZ,
            maxObjectZ,
            tiles);
    }

    private void RestoreSelectedPm4Object((int tileX, int tileY, uint ck24, int objectPart)? selectedObjectKey)
    {
        if (!selectedObjectKey.HasValue)
        {
            _selectedPm4ObjectKey = null;
            _selectedPm4ObjectGroupKey = null;
            return;
        }

        if (_pm4ObjectLookup.ContainsKey(selectedObjectKey.Value))
        {
            _selectedPm4ObjectKey = selectedObjectKey;
            _selectedPm4ObjectGroupKey = ResolvePm4ObjectGroupKey(selectedObjectKey.Value);
            return;
        }

        _selectedPm4ObjectKey = null;
        _selectedPm4ObjectGroupKey = null;
    }

    internal Vector3 GetPm4LoadAnchorCameraPosition()
    {
        if (_hasLastRenderedCameraPosition)
            return _lastRenderedCameraPosition;

        return _terrainManager.GetInitialCameraPosition();
    }

    internal void EnsurePm4OverlayMatchesCameraWindow(Vector3 cameraPos)
    {
        if (!_showPm4Overlay)
            return;

        if (_pm4LoadTask != null && !_pm4LoadTask.IsCompleted)
            return;

        if (!_pm4LoadAttempted || !_pm4LoadedCameraWindow.HasValue)
        {
            BeginPm4OverlayLoad();
            return;
        }

        var desiredWindow = GetPm4CameraWindow(cameraPos, _pm4CameraTileRadius);
        if (!IsPm4CameraWindowCovered(desiredWindow))
            BeginPm4OverlayLoad();
    }

    private bool IsPm4CameraWindowCovered((int minTileX, int minTileY, int maxTileX, int maxTileY) cameraWindow)
    {
        if (_pm4KnownMapTiles.Count > 0)
        {
            bool hasKnownTileInWindow = false;
            foreach ((int tileX, int tileY) in _pm4KnownMapTiles)
            {
                if (!IsPm4TileInsideCameraWindow(tileX, tileY, cameraWindow))
                    continue;

                hasKnownTileInWindow = true;

                if (!_pm4CoveredMapTiles.Contains((tileX, tileY)))
                    return false;
            }

            if (hasKnownTileInWindow)
                return true;
        }

        if (!_pm4LoadedCameraWindow.HasValue)
            return false;

        var loadedWindow = _pm4LoadedCameraWindow.Value;
        return cameraWindow.minTileX >= loadedWindow.minTileX
            && cameraWindow.minTileY >= loadedWindow.minTileY
            && cameraWindow.maxTileX <= loadedWindow.maxTileX
            && cameraWindow.maxTileY <= loadedWindow.maxTileY;
    }

    private void ExpandPm4LoadedCameraWindow((int minTileX, int minTileY, int maxTileX, int maxTileY) window)
    {
        if (!_pm4LoadedCameraWindow.HasValue)
        {
            _pm4LoadedCameraWindow = window;
            return;
        }

        var existing = _pm4LoadedCameraWindow.Value;
        _pm4LoadedCameraWindow = (
            Math.Min(existing.minTileX, window.minTileX),
            Math.Min(existing.minTileY, window.minTileY),
            Math.Max(existing.maxTileX, window.maxTileX),
            Math.Max(existing.maxTileY, window.maxTileY));
    }

    private void UpdatePm4AdaptiveWindow(double loadElapsedMs)
    {
        _pm4AverageLoadMs = _pm4AverageLoadMs < 0.0
            ? loadElapsedMs
            : _pm4AverageLoadMs * 0.65 + loadElapsedMs * 0.35;

        int previousRadius = _pm4CameraTileRadius;
        if (_pm4AverageLoadMs >= Pm4ShrinkWindowThresholdMs && _pm4CameraTileRadius > Pm4MinCameraTileRadius)
            _pm4CameraTileRadius--;
        else if (_pm4AverageLoadMs <= Pm4ExpandWindowThresholdMs && _pm4CameraTileRadius < Pm4MaxCameraTileRadius)
            _pm4CameraTileRadius++;

        if (previousRadius != _pm4CameraTileRadius)
        {
            ViewerLog.Info(
                ViewerLog.Category.Terrain,
                $"[PM4] Adaptive window radius changed {previousRadius} -> {_pm4CameraTileRadius} (avg {_pm4AverageLoadMs:0} ms).");
        }
    }

    private Pm4OverlayCacheData BuildPm4OverlayCacheData(string mapName, string candidateSignature)
    {
        var tiles = new List<Pm4OverlayCacheTile>(_pm4TileObjects.Count);
        foreach (var tileEntry in _pm4TileObjects.OrderBy(static entry => entry.Key.tileX).ThenBy(static entry => entry.Key.tileY))
        {
            List<Vector3> positionRefs = _pm4TilePositionRefs.TryGetValue(tileEntry.Key, out List<Vector3>? existingPositionRefs)
                ? existingPositionRefs
                : new List<Vector3>();
            var objects = new List<Pm4OverlayCacheObject>(tileEntry.Value.Count);
            for (int i = 0; i < tileEntry.Value.Count; i++)
            {
                Pm4OverlayObject obj = tileEntry.Value[i];
                objects.Add(new Pm4OverlayCacheObject(
                    obj.SourcePath,
                    obj.MshdField00,
                    obj.MshdRegionId,
                    obj.MshdField08,
                    obj.Ck24,
                    obj.Ck24Type,
                    obj.ObjectPartId,
                    obj.LinkGroupObjectId,
                    obj.LinkedPositionRefCount,
                    obj.LinkedPositionRefSummary,
                    obj.Lines,
                    obj.Triangles,
                    obj.SurfaceCount,
                    obj.TotalIndexCount,
                    obj.DominantGroupKey,
                    obj.DominantAttributeMask,
                    obj.DominantMscnRefIndex,
                    obj.AverageSurfaceHeight,
                    obj.PlacementAnchor,
                    obj.BaseRotationRadians,
                    obj.PlanarTransform,
                    obj.BoundsMin,
                    obj.BoundsMax,
                    obj.ConnectorKeys.ToList()));
            }

            tiles.Add(new Pm4OverlayCacheTile(tileEntry.Key.tileX, tileEntry.Key.tileY, objects, positionRefs));
        }

        return new Pm4OverlayCacheData(
            mapName,
            candidateSignature,
            _pm4TotalFiles,
            _pm4LoadedFiles,
            _pm4ObjectCount,
            _pm4LineCount,
            _pm4TriangleCount,
            _pm4PositionRefCount,
            _pm4RejectedLongEdges,
            _pm4MinObjectZ,
            _pm4MaxObjectZ,
            tiles);
    }

    private void MergePm4OverlayFromCache(Pm4OverlayCacheData cacheData)
    {
        for (int tileIndex = 0; tileIndex < cacheData.Tiles.Count; tileIndex++)
        {
            Pm4OverlayCacheTile tile = cacheData.Tiles[tileIndex];
            var tileKey = (tile.TileX, tile.TileY);

            var objects = new List<Pm4OverlayObject>(tile.Objects.Count);
            for (int objectIndex = 0; objectIndex < tile.Objects.Count; objectIndex++)
            {
                Pm4OverlayCacheObject cachedObject = tile.Objects[objectIndex];
                Pm4OverlayObject restored = Pm4OverlayObject.FromCachedLocalized(
                    cachedObject.SourcePath,
                    cachedObject.MshdField00,
                    cachedObject.MshdRegionId,
                    cachedObject.MshdField08,
                    cachedObject.Ck24,
                    cachedObject.Ck24Type,
                    cachedObject.ObjectPartId,
                    cachedObject.LinkGroupObjectId,
                    cachedObject.LinkedPositionRefCount,
                    cachedObject.LinkedPositionRefSummary,
                    new List<Pm4LineSegment>(cachedObject.Lines),
                    new List<Pm4Triangle>(cachedObject.Triangles),
                    cachedObject.SurfaceCount,
                    cachedObject.TotalIndexCount,
                    cachedObject.DominantGroupKey,
                    cachedObject.DominantAttributeMask,
                    cachedObject.DominantMscnRefIndex,
                    cachedObject.AverageSurfaceHeight,
                    cachedObject.PlacementAnchor,
                    cachedObject.BaseRotationRadians,
                    cachedObject.PlanarTransform,
                    cachedObject.BoundsMin,
                    cachedObject.BoundsMax,
                    cachedObject.ConnectorKeys.ToList());
                objects.Add(restored);
                _pm4ObjectLookup[(tile.TileX, tile.TileY, restored.Ck24, restored.ObjectPartId)] = restored;
                var groupKey = BuildPm4BaseObjectGroupKey((tile.TileX, tile.TileY, restored.Ck24, restored.ObjectPartId));
                if (!_pm4GroupToObjectKeys.TryGetValue(groupKey, out var groupObjectKeys))
                {
                    groupObjectKeys = new List<(int, int, uint, int)>();
                    _pm4GroupToObjectKeys[groupKey] = groupObjectKeys;
                }
                groupObjectKeys.Add((tile.TileX, tile.TileY, restored.Ck24, restored.ObjectPartId));
            }

            _pm4TileObjects[tileKey] = objects;
            _pm4TileStats[tileKey] = new Pm4OverlayTileStats(
                tile.TileX,
                tile.TileY,
                objects.Count,
                objects.Sum(static obj => obj.Lines.Count),
                objects.Sum(static obj => obj.Triangles.Count));

            if (tile.PositionRefs.Count > 0)
                _pm4TilePositionRefs[tileKey] = new List<Vector3>(tile.PositionRefs);
        }

        if (_pm4MinObjectZ > _pm4MaxObjectZ)
        {
            _pm4MinObjectZ = 0f;
            _pm4MaxObjectZ = 1f;
        }

        _pm4TotalFiles = Math.Max(_pm4TotalFiles, cacheData.TotalFiles);
        RecalculatePm4OverlayRuntimeTotals();
    }

    /// <summary>
    /// Spec 054: After a successful per-file decode, store the payload in
    /// the in-memory per-file cache. The split flags are folded into the
    /// entry's stamp so an entry decoded with one set of split flags is
    /// treated as a miss when the user toggles a split-flag and re-loads.
    /// </summary>
    private void StorePerFileInMemoryCache(
        string normalizedPath,
        long fileLength,
        int effectiveTileX,
        int effectiveTileY,
        IReadOnlyList<Pm4OverlayObject> objects,
        IReadOnlyDictionary<(int tileX, int tileY), List<Vector3>> tilePositionRefs,
        bool splitByMscnRef,
        bool splitByConnectivity)
    {
        if (objects.Count == 0)
            return;

        List<CorePm4CachedObject> cachedObjects = new(objects.Count);
        for (int i = 0; i < objects.Count; i++)
        {
            Pm4OverlayObject obj = objects[i];
            cachedObjects.Add(new CorePm4CachedObject(
                SourcePath: obj.SourcePath,
                MshdField00: obj.MshdField00,
                MshdRegionId: obj.MshdRegionId,
                MshdField08: obj.MshdField08,
                Ck24: obj.Ck24,
                Ck24Type: obj.Ck24Type,
                ObjectPartId: obj.ObjectPartId,
                LinkGroupObjectId: obj.LinkGroupObjectId,
                LinkedPositionRefCount: obj.LinkedPositionRefCount,
                LinkedPositionRefSummary: new CorePm4LinkedPositionRefSummary(
                    obj.LinkedPositionRefSummary.TotalCount,
                    obj.LinkedPositionRefSummary.NormalCount,
                    obj.LinkedPositionRefSummary.TerminatorCount,
                    obj.LinkedPositionRefSummary.FloorMin,
                    obj.LinkedPositionRefSummary.FloorMax,
                    obj.LinkedPositionRefSummary.HeadingMinDegrees,
                    obj.LinkedPositionRefSummary.HeadingMaxDegrees,
                    obj.LinkedPositionRefSummary.HeadingMeanDegrees),
                SurfaceCount: obj.SurfaceCount,
                TotalIndexCount: obj.TotalIndexCount,
                DominantGroupKey: obj.DominantGroupKey,
                DominantAttributeMask: obj.DominantAttributeMask,
                DominantMscnRefIndex: obj.DominantMscnRefIndex,
                AverageSurfaceHeight: obj.AverageSurfaceHeight,
                PlacementAnchor: obj.PlacementAnchor,
                BaseRotationRadians: obj.BaseRotationRadians,
                PlanarSwapPlanarAxes: obj.PlanarTransform.SwapPlanarAxes,
                PlanarInvertU: obj.PlanarTransform.InvertU,
                PlanarInvertV: obj.PlanarTransform.InvertV,
                BoundsMin: obj.BoundsMin,
                BoundsMax: obj.BoundsMax,
                ConnectorKeys: obj.ConnectorKeys
                    .Select(static k => new CorePm4CachedConnectorKey(k.X, k.Y, k.Z))
                    .ToList(),
                Lines: obj.Lines
                    .Select(static seg => new CorePm4CachedLineSegment(seg.From, seg.To))
                    .ToList(),
                Triangles: obj.Triangles
                    .Select(static tri => new CorePm4CachedTriangle(tri.A, tri.B, tri.C))
                    .ToList()));
        }

        List<Vector3> positionRefs = tilePositionRefs.TryGetValue((effectiveTileX, effectiveTileY), out List<Vector3>? refs)
            ? new List<Vector3>(refs)
            : new List<Vector3>();

        CorePm4CachedTile cachedTile = new(
            TileX: effectiveTileX,
            TileY: effectiveTileY,
            PositionRefs: positionRefs,
            Objects: cachedObjects);

        long splitStamp = ((splitByMscnRef ? 1L : 0L) << 32) | (splitByConnectivity ? 1L : 0L);

        _pm4PerFileInMemoryCache.Set(
            normalizedPath,
            new CorePm4PerFileCacheEntry(
                FileLength: fileLength,
                LastWriteTicks: splitStamp,
                Tiles: new[] { cachedTile }));
    }

    /// <summary>
    /// Spec 054: lazy accessor for the on-disk per-file cache. The
    /// service is constructed once per (dataSource, mapName) and
    /// cached for the lifetime of <see cref="WorldScene"/>. The cache
    /// root is derived from the per-window cache service's
    /// <see cref="Pm4OverlayCacheService.CacheRoot"/> so the two layers
    /// share the on-disk parent.
    /// </summary>
    private CorePm4PerFileCacheService? EnsurePerFileDiskCache(string mapName)
    {
        if (_pm4PerFileDiskCache != null)
            return _pm4PerFileDiskCache;
        if (_pm4OverlayCacheService == null)
            return null;
        if (_dataSource == null)
            return null;
        if (string.IsNullOrWhiteSpace(mapName))
            return null;

        string identity = _dataSource.Name ?? "default";
        _pm4PerFileDiskCache = CorePm4PerFileCacheService.CreateForDataSource(
            Path.Combine(_pm4OverlayCacheService.CacheRoot, "files"),
            identity,
            mapName);
        return _pm4PerFileDiskCache;
    }

    private void RecalculatePm4OverlayRuntimeTotals()
    {
        _pm4LoadedFiles = _pm4TileObjects.Count;
        _pm4ObjectCount = 0;
        _pm4LineCount = 0;
        _pm4TriangleCount = 0;
        _pm4PositionRefCount = 0;
        _pm4RejectedLongEdges = 0;
        _pm4TotalMsurCount = 0;
        _pm4DroppedShortIndexCount = 0;
        _pm4WallFaceCount = 0;
        _pm4DroppedOutOfRangeMsviCount = 0;
        _pm4DroppedEmptyComponentCount = 0;
        _pm4MinObjectZ = float.MaxValue;
        _pm4MaxObjectZ = float.MinValue;

        foreach (var tileEntry in _pm4TileObjects)
        {
            List<Pm4OverlayObject> objects = tileEntry.Value;
            _pm4ObjectCount += objects.Count;
            _pm4LineCount += objects.Sum(static obj => obj.Lines.Count);
            _pm4TriangleCount += objects.Sum(static obj => obj.Triangles.Count);

            for (int i = 0; i < objects.Count; i++)
            {
                _pm4MinObjectZ = MathF.Min(_pm4MinObjectZ, objects[i].Center.Z);
                _pm4MaxObjectZ = MathF.Max(_pm4MaxObjectZ, objects[i].Center.Z);
            }
        }

        foreach (var refsEntry in _pm4TilePositionRefs)
            _pm4PositionRefCount += refsEntry.Value.Count;

        if (_pm4MinObjectZ > _pm4MaxObjectZ)
        {
            _pm4MinObjectZ = 0f;
            _pm4MaxObjectZ = 1f;
        }
    }

    internal bool ShouldRenderPm4Tile(int tileX, int tileY)
    {
        // PM4 overlay loading is already constrained by the PM4 camera window and object-level
        // culling. Gating PM4 by terrain AOI slices large structures across adjacent tiles,
        // which makes multi-tile WMO footprints like Stormwind Harbour disappear in pieces.
        return true;
    }

    internal readonly struct Pm4IndexedSurface
    {
        public Pm4IndexedSurface(int surfaceIndex, MsurEntry surface)
        {
            SurfaceIndex = surfaceIndex;
            Surface = surface;
        }

        public int SurfaceIndex { get; }
        public MsurEntry Surface { get; }
    }

    internal readonly struct Pm4OverlaySeedGroup
    {
        public Pm4OverlaySeedGroup(uint displayCk24, byte displayCk24Type, bool requiresConnectivitySeedSplit, List<Pm4IndexedSurface> surfaces)
        {
            DisplayCk24 = displayCk24;
            DisplayCk24Type = displayCk24Type;
            RequiresConnectivitySeedSplit = requiresConnectivitySeedSplit;
            Surfaces = surfaces;
        }

        public uint DisplayCk24 { get; }
        public byte DisplayCk24Type { get; }
        public bool RequiresConnectivitySeedSplit { get; }
        public List<Pm4IndexedSurface> Surfaces { get; }
    }

    /// <summary>
    /// The one frame PM4 geometry is drawn in. MSVT needs no fitting, so nothing is fitted.
    /// </summary>
    /// <remarks>
    /// MSVT is stored in ADT placement space — an origin-relative coordinate, like a raw MDDF
    /// position — so <c>WorldSpace</c> with the identity planar transform composes with
    /// <see cref="ConvertWorldToRenderer"/> into <c>(MapOrigin - X, MapOrigin - Y, Z)</c>. That is
    /// already, letter for letter, what <see cref="EnsurePm4MscnData"/> and
    /// <see cref="EnsurePm4MspvData"/> do to place MSCN and MSPV, and those land correctly. MSPV,
    /// MSVT and MSCN share one chunk frame, so the mesh has no business using a different one.
    /// </remarks>
    internal static readonly CorePm4CoordinateModeResolution CanonicalCoordinateModeResolution =
        new(
            CorePm4CoordinateMode.WorldSpace,
            CorePm4PlacementContract.GetDefaultPlanarTransform(CorePm4CoordinateMode.WorldSpace),
            0f,
            0f,
            false);

    private void RebuildPm4MergedObjectGroups()
    {
        _pm4MergedObjectGroupKeys.Clear();

        var groups = new List<CorePm4ConnectorMergeCandidate>();
        foreach (var tileEntry in _pm4TileObjects)
        {
            foreach (IGrouping<(int tileX, int tileY, uint ck24), Pm4OverlayObject> objectGroup in tileEntry.Value.GroupBy(obj => BuildPm4BaseObjectGroupKey((tileEntry.Key.tileX, tileEntry.Key.tileY, obj.Ck24, obj.ObjectPartId))))
            {
                var baseGroupKey = objectGroup.Key;
                Vector3 boundsMin = new(float.MaxValue, float.MaxValue, float.MaxValue);
                Vector3 boundsMax = new(float.MinValue, float.MinValue, float.MinValue);
                bool hasBounds = false;
                var connectorKeys = new HashSet<CorePm4ConnectorKey>();

                foreach (Pm4OverlayObject obj in objectGroup)
                {
                    IncludePointInBounds(obj.BoundsMin, ref boundsMin, ref boundsMax, ref hasBounds);
                    IncludePointInBounds(obj.BoundsMax, ref boundsMin, ref boundsMax, ref hasBounds);

                    for (int i = 0; i < obj.ConnectorKeys.Count; i++)
                        connectorKeys.Add(ToCorePm4ConnectorKey(obj.ConnectorKeys[i]));
                }

                if (!hasBounds)
                {
                    boundsMin = Vector3.Zero;
                    boundsMax = Vector3.Zero;
                }

                Vector3 center = (boundsMin + boundsMax) * 0.5f;
                groups.Add(new CorePm4ConnectorMergeCandidate(
                    new CorePm4ObjectGroupKey(baseGroupKey.tileX, baseGroupKey.tileY, baseGroupKey.ck24),
                    boundsMin,
                    boundsMax,
                    center,
                    connectorKeys));

                _pm4MergedObjectGroupKeys[baseGroupKey] = baseGroupKey;
            }
        }

        IReadOnlyDictionary<CorePm4ObjectGroupKey, CorePm4ObjectGroupKey> mergedGroupMap = CorePm4PlacementMath.BuildMergedGroupMap(groups);
        foreach ((CorePm4ObjectGroupKey sourceKey, CorePm4ObjectGroupKey mergedKey) in mergedGroupMap)
            _pm4MergedObjectGroupKeys[(sourceKey.TileX, sourceKey.TileY, sourceKey.Ck24)] = (mergedKey.TileX, mergedKey.TileY, mergedKey.Ck24);
    }

    internal enum Pm4AxisConvention
    {
        XZPlaneYUp,
        XYPlaneZUp,
        YZPlaneXUp
    }

    public void ReloadPm4Overlay()
    {
        ClearPm4OverlayRuntimeState();
        _pm4PerFileInMemoryCache.Clear();
        _pm4PerFileDiskCache?.ClearForMap();
        _pm4PerFileDiskCache = null;
        _pm4LoadAttempted = false;
        BeginPm4OverlayLoad(ignoreCache: true);
    }

    // ──────────────────────────────────────────────────────────────────────
    // PM4 color system: light pastels for containers, dark pastels for mesh,
    // saturated colors reserved for markers/highlights/selection.
    // See wow-viewer/docs/architecture/pm4-color-palette.md (to be written).
    // ──────────────────────────────────────────────────────────────────────

    /// <summary>Light pastel — PM4 object bounds (per-object, sub-merged).</summary>
    internal static readonly Vector3 Pm4ColorObjectBounds = new(1.00f, 0.75f, 0.80f);

    /// <summary>Light pastel — PM4 CK24 bounds (merged across sub-objects).</summary>
    private static readonly Vector3 Pm4ColorCk24Bounds = new(0.70f, 1.00f, 0.80f);

    /// <summary>Light pastel — selected bounds inner fill.</summary>
    internal static readonly Vector3 Pm4ColorSelectedBounds = new(1.00f, 1.00f, 0.95f);

    /// <summary>Light pastel — MDDF (M2) instance bounds.</summary>
    internal static readonly Vector3 Pm4ColorMddfBounds = new(1.00f, 0.70f, 1.00f);

    /// <summary>Light pastel — MODF (WMO) instance bounds.</summary>
    internal static readonly Vector3 Pm4ColorModfBounds = new(0.75f, 0.95f, 1.00f);

    /// <summary>Dark pastel — PM4 centroid pin (per-object center marker).</summary>
    internal static readonly Vector3 Pm4ColorCentroid = new(0.70f, 0.55f, 0.50f);

    /// <summary>Saturated — MSCN scene-graph connector anchor. One cube per MSUR surface (placed via MSUR.MscnRefIndex). Bright cyan, distinct from everything else. See wow-viewer/docs/architecture/pm4-chunk-semantics.md.</summary>
    internal static readonly Vector3 Pm4ColorMscn = new(0.10f, 0.95f, 1.00f);

    /// <summary>Saturated — MSPV path-vertex position. One cube per MSPI index reached from an MSLK link's path-vertex chain. Only present when surfaces are connected via MSLK. Bright magenta, distinct from MSCN and from pastel mesh. See wow-viewer/docs/architecture/pm4-chunk-semantics.md.</summary>
    internal static readonly Vector3 Pm4ColorMspv = new(1.00f, 0.20f, 0.80f);

    /// <summary>Medium pastel — MPRL position reference pin.</summary>
    internal static readonly Vector3 Pm4ColorMprl = new(0.40f, 0.80f, 0.85f);

    // Saturated signals (NOT in the pastel family — reserved for interactive signals)

    /// <summary>Saturated — search/highlight match (eyecatching on pastel mesh).</summary>
    internal static readonly Vector3 Pm4ColorHighlight = new(0.20f, 1.00f, 0.95f);

    /// <summary>Saturated — group selection (THE selection signal — must be unmistakable).</summary>
    internal static readonly Vector3 Pm4ColorSelection = new(1.00f, 0.95f, 0.20f);

    /// <summary>
    /// Per-surface-class visibility, keyed by <c>MSUR._0x00</c>.
    /// </summary>
    /// <remarks>
    /// Measured 2026-08-24 with `pm4 surface-class` over 309 files. The field is a genuine surface
    /// class, and the cleanest result is that <b>0x03 is the doodad marker</b>: 100.0% of its 184,356
    /// surfaces carry no placement height, while every other class is ~99.5% in the placed
    /// population. It is therefore a second, independent identifier for the M2 population.
    ///
    /// <para>The remaining classes stratify by height inside their own object rather than into
    /// roof/wall/floor: 0x10 sits lowest at 0.291 of the object's Z extent, 0x13 at 0.427, 0x12 at
    /// 0.474, 0x14 highest at 0.624. All are Z-dominant and up-facing (82-99%), because MSUR holds
    /// FLOORS - walls live in MSPV/MSPI, so no MSUR class is a wall.</para>
    ///
    /// <para>Filtering is by the object's DOMINANT class, since the overlay carries one class per
    /// object rather than per surface. That separates doodads from placed objects and low-floor from
    /// high-floor objects, but it cannot isolate one surface class WITHIN a single object - that
    /// needs per-surface class carried through the overlay build.</para>
    /// </remarks>
    private readonly HashSet<byte> _pm4HiddenSurfaceClasses = [];

    private sealed class Pm4OverlayAsyncLoadResult
    {
        public Pm4OverlayAsyncLoadResult(
            int requestId,
            Pm4OverlayCacheData? cacheData,
            (int minTileX, int minTileY, int maxTileX, int maxTileY)? loadedCameraWindow,
            IReadOnlyCollection<(int tileX, int tileY)> knownMapTiles,
            IReadOnlyCollection<(int tileX, int tileY)> coveredMapTiles,
            (int tileX, int tileY, uint ck24, int objectPart)? selectedObjectKey,
            double loadElapsedMs,
            string statusMessage,
            bool cancelled)
        {
            RequestId = requestId;
            CacheData = cacheData;
            LoadedCameraWindow = loadedCameraWindow;
            KnownMapTiles = knownMapTiles;
            CoveredMapTiles = coveredMapTiles;
            SelectedObjectKey = selectedObjectKey;
            LoadElapsedMs = loadElapsedMs;
            StatusMessage = statusMessage;
            Cancelled = cancelled;
        }

        public int RequestId { get; }
        public Pm4OverlayCacheData? CacheData { get; }
        public (int minTileX, int minTileY, int maxTileX, int maxTileY)? LoadedCameraWindow { get; }
        public IReadOnlyCollection<(int tileX, int tileY)> KnownMapTiles { get; }
        public IReadOnlyCollection<(int tileX, int tileY)> CoveredMapTiles { get; }
        public (int tileX, int tileY, uint ck24, int objectPart)? SelectedObjectKey { get; }
        public double LoadElapsedMs { get; }
        public string StatusMessage { get; }
        public bool Cancelled { get; }
    }
}
