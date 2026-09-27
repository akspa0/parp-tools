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
using static WoWViewer.Terrain.Pm4OverlayScene;
using static WoWViewer.Terrain.Pm4OverlayMatching;
using static WoWViewer.Terrain.Pm4OverlayCacheCodec;
using static WoWViewer.Terrain.Pm4OverlayGeometry;
using static WoWViewer.Terrain.Pm4OverlayCoordinates;
using static WoWViewer.Terrain.Pm4OverlayColors;
using WowViewer.Core.Runtime.World.Selection;

namespace WoWViewer.Terrain;

/// <summary>
/// Taxi paths in the world scene: lazy route loading, node/route selection state, and taxi actors riding routes.
/// Moved verbatim from <see cref="WorldScene"/> (Spec 255). Scene state it still needs comes
/// only through <see cref="IWorldSceneHost"/>; the bridge members below keep the names the moved
/// code used inside WorldScene, so no moved body was edited.
/// </summary>
public sealed class TaxiActorScene
{
    private readonly IWorldSceneHost _host;

    internal TaxiActorScene(IWorldSceneHost host)
    {
        _host = host;
    }

    // Host bridge (same names as the WorldScene members).
    private WorldAssetManager _assets => _host.Assets;
    private ref IDataSource? _dataSource => ref _host.DataSource;
    private ref string? _dbcBuild => ref _host.DbcBuild;
    private ref DBCD.Providers.IDBCProvider? _dbcProvider => ref _host.DbcProvider;
    private ref string? _dbdDir => ref _host.DbdDir;
    private ref int _mapId => ref _host.MapId;
    private List<ObjectInstance> _taxiActorInstances => _host.TaxiActorInstances;
    private static void TransformBounds(Vector3 min, Vector3 max, Matrix4x4 m, out Vector3 outMin, out Vector3 outMax) => WorldScene.TransformBounds(min, max, m, out outMin, out outMax);
    private static void TransformBounds(Vector3 boundsMin, Vector3 boundsMax, in Matrix4x4 transform, out Vector3 transformedMin, out Vector3 transformedMax) => WorldScene.TransformBounds(boundsMin, boundsMax, in transform, out transformedMin, out transformedMax);

    private const float TaxiActorHeadingSampleWindow = 18f;
    private const float TaxiActorHeadingSmoothingHz = 8f;
    public const float TaxiActorNormalSpeedSetting = 0.10f;
    public const float TaxiActorMinSpeedSetting = 0.01f;
    public const float TaxiActorMaxSpeedSetting = 0.50f;
    private static readonly string[] TaxiActorDefaultModelCandidates =
    {
        @"Creature\Gryphon\Gryphon.mdx",
        @"Creature\FelBat\BatTaxi.mdx",
    };

    public static IReadOnlyList<string> DefaultTaxiActorModelPaths => TaxiActorDefaultModelCandidates;

    // Taxi paths (lazy-loaded on first toggle)
    internal TaxiPathLoader? _taxiLoader;
    internal bool _showTaxi = false;
    private bool _taxiLoadAttempted = false;
    public bool ShowTaxi
    {
        get => _showTaxi;
        set { _showTaxi = value; if (value && !_taxiLoadAttempted) LazyLoadTaxi(); }
    }
    public TaxiPathLoader? TaxiLoader => _taxiLoader;
    public bool TaxiLoadAttempted => _taxiLoadAttempted;

    // Taxi selection: -1 = show all (or none if !_showTaxi)
    internal int _selectedTaxiNodeId = -1;
    internal int _selectedTaxiRouteId = -1;
    private readonly Dictionary<int, string> _taxiActorModelOverrideByPath = new();
    private readonly Dictionary<int, float> _taxiActorTravelByPath = new();
    private readonly Dictionary<int, TaxiActorPose> _taxiActorPoseByPath = new();
    private readonly Dictionary<int, Vector3> _taxiActorSmoothedForwardByPath = new();
    private long _lastTaxiActorTick;
    private bool _taxiActorClockInitialized;
    private int _activeTaxiRideRouteId = -1;
    private bool _showTaxiActors = true;
    private float _taxiActorSpeedMultiplier = TaxiActorNormalSpeedSetting;
    private float _taxiActorScaleMultiplier = 1.0f;
    private const float TaxiActorBaseUnitsPerSecond = 650f;
    private const float TaxiActorHoverOffset = 12f;
    public int SelectedTaxiNodeId { get => _selectedTaxiNodeId; set { _selectedTaxiNodeId = value; _selectedTaxiRouteId = -1; } }
    public int SelectedTaxiRouteId { get => _selectedTaxiRouteId; set { _selectedTaxiRouteId = value; _selectedTaxiNodeId = -1; } }
    public void ClearTaxiSelection() { _selectedTaxiNodeId = -1; _selectedTaxiRouteId = -1; }
    /// <summary>
    /// Route currently carrying the ride camera. This is deliberately separate
    /// from selection and visibility state: an active ride must keep its pose
    /// alive while the operator browses or hides taxi presentation controls.
    /// </summary>
    public int ActiveTaxiRideRouteId
    {
        get => _activeTaxiRideRouteId;
        set => _activeTaxiRideRouteId = value >= 0 ? value : -1;
    }

    public bool ShowTaxiActors { get => _showTaxiActors; set => _showTaxiActors = value; }
    public float TaxiActorSpeedMultiplier
    {
        get => _taxiActorSpeedMultiplier;
        set
        {
            float normalized = float.IsFinite(value) ? value : TaxiActorNormalSpeedSetting;
            _taxiActorSpeedMultiplier = Math.Clamp(normalized, TaxiActorMinSpeedSetting, TaxiActorMaxSpeedSetting);
        }
    }

    public float TaxiActorScaleMultiplier
    {
        get => _taxiActorScaleMultiplier;
        set => _taxiActorScaleMultiplier = Math.Max(0f, value);
    }

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

    public TaxiPathLoader.TaxiNode? GetTaxiNode(int nodeId)
        => _taxiLoader?.Nodes.FirstOrDefault(node => node.Id == nodeId);

    public TaxiPathLoader.TaxiRoute? GetTaxiRoute(int pathId)
        => _taxiLoader?.Routes.FirstOrDefault(route => route.PathId == pathId);

    public string? GetTaxiActorModelOverride(int pathId)
        => _taxiActorModelOverrideByPath.TryGetValue(pathId, out string? modelPath) ? modelPath : null;

    public void SetTaxiActorModelOverride(int pathId, string? modelPath)
    {
        string normalizedPath = string.IsNullOrWhiteSpace(modelPath)
            ? string.Empty
            : modelPath.Trim().Replace('/', '\\');

        if (string.IsNullOrWhiteSpace(normalizedPath))
        {
            _taxiActorModelOverrideByPath.Remove(pathId);
            return;
        }

        _taxiActorModelOverrideByPath[pathId] = normalizedPath;
        _assets.QueueMdxLoad(WorldAssetManager.NormalizeKey(normalizedPath));
    }

    public string? GetResolvedTaxiActorModelPath(int pathId)
    {
        if (_taxiActorModelOverrideByPath.TryGetValue(pathId, out string? overrideModelPath)
            && !string.IsNullOrWhiteSpace(overrideModelPath))
        {
            return overrideModelPath;
        }

        TaxiPathLoader.TaxiRoute? route = GetTaxiRoute(pathId);
        if (route == null)
            return ResolveDefaultTaxiActorModelPath();

        TaxiPathLoader.TaxiNode? mountNode = ResolveTaxiActorNode(route);
        if (!string.IsNullOrWhiteSpace(mountNode?.MountModelPath))
            return mountNode.MountModelPath.Replace('/', '\\');

        return ResolveDefaultTaxiActorModelPath();
    }

    private string ResolveDefaultTaxiActorModelPath()
    {
        if (_dataSource != null)
        {
            foreach (string candidate in TaxiActorDefaultModelCandidates)
            {
                if (_dataSource.FileExists(candidate))
                    return candidate;
            }
        }

        return TaxiActorDefaultModelCandidates[0];
    }

    public bool TryGetTaxiActorPose(int pathId, out TaxiActorPose pose)
        => _taxiActorPoseByPath.TryGetValue(pathId, out pose);

    public bool TryGetSelectedTaxiActorPose(out TaxiActorPose pose)
    {
        if (_selectedTaxiRouteId < 0)
        {
            pose = default;
            return false;
        }

        return _taxiActorPoseByPath.TryGetValue(_selectedTaxiRouteId, out pose);
    }

    public bool TryGetTaxiRouteSelectionPoint(int pathId, out Vector3 point)
    {
        TaxiPathLoader.TaxiRoute? route = GetTaxiRoute(pathId);
        if (route == null)
        {
            point = Vector3.Zero;
            return false;
        }

        return TryGetTaxiRouteSelectionPoint(route, out point);
    }

    private void LazyLoadTaxi()
    {
        _taxiLoadAttempted = true;
        if (_dbcProvider == null || _dbdDir == null || _dbcBuild == null || _mapId < 0) return;
        _taxiLoader = new TaxiPathLoader();
        var dbcd = new DBCD.DBCD(_dbcProvider, new DBCD.Providers.FilesystemDBDProvider(_dbdDir));
        _taxiLoader.Load(dbcd, _dbcBuild, _mapId);
        _taxiActorTravelByPath.Clear();
        _taxiActorPoseByPath.Clear();
        _taxiActorSmoothedForwardByPath.Clear();
        _taxiActorClockInitialized = false;
    }

    internal void UpdateTaxiActorInstances()
    {
        _taxiActorInstances.Clear();

        bool hasTaxiSelection = _selectedTaxiNodeId >= 0 || _selectedTaxiRouteId >= 0;
        if (_taxiLoader == null
            || !TaxiRideSimulationPolicy.ShouldSimulateRoute(
                _activeTaxiRideRouteId,
                _activeTaxiRideRouteId,
                _showTaxi,
                _showTaxiActors,
                hasTaxiSelection,
                routeVisible: true))
        {
            _taxiActorPoseByPath.Clear();
            _taxiActorSmoothedForwardByPath.Clear();
            _taxiActorClockInitialized = false;
            return;
        }

        long now = Stopwatch.GetTimestamp();
        float deltaSeconds = 0f;
        if (_taxiActorClockInitialized)
            deltaSeconds = (float)((now - _lastTaxiActorTick) / (double)Stopwatch.Frequency);
        _lastTaxiActorTick = now;
        _taxiActorClockInitialized = true;

        float distanceStep = TaxiActorBaseUnitsPerSecond * _taxiActorSpeedMultiplier * Math.Max(0f, deltaSeconds);
        var activePathIds = new HashSet<int>();

        foreach (var route in _taxiLoader.Routes)
        {
            bool routeVisible = IsTaxiRouteVisible(route);
            if (!TaxiRideSimulationPolicy.ShouldSimulateRoute(
                    route.PathId,
                    _activeTaxiRideRouteId,
                    _showTaxi,
                    _showTaxiActors,
                    hasTaxiSelection,
                    routeVisible)
                || route.Waypoints.Count < 2)
                continue;

            TaxiPathLoader.TaxiNode? mountNode = ResolveTaxiActorNode(route);
            float scale = mountNode?.MountScale > 0.01f ? mountNode.MountScale : 1.0f;

            scale *= _taxiActorScaleMultiplier;

            float routeLength = GetRouteLength(route.Waypoints);
            if (routeLength <= 1f)
                continue;

            activePathIds.Add(route.PathId);

            float travel = _taxiActorTravelByPath.TryGetValue(route.PathId, out float existingTravel)
                ? existingTravel
                : 0f;
            if (distanceStep > 0f)
                travel = (travel + distanceStep) % routeLength;
            _taxiActorTravelByPath[route.PathId] = travel;

            SampleRoute(route.Waypoints, travel, out Vector3 actorPosition, out Vector3 actorDirection);
            actorPosition.Z += TaxiActorHoverOffset;

            Vector3 sampledForward = SampleSmoothedTaxiRouteDirection(route.Waypoints, travel, routeLength);
            Vector3 actorForward = sampledForward;
            if (_taxiActorSmoothedForwardByPath.TryGetValue(route.PathId, out Vector3 previousForward)
                && previousForward.LengthSquared() > 0.0001f)
            {
                float blend = 1f - MathF.Exp(-TaxiActorHeadingSmoothingHz * Math.Max(0f, deltaSeconds));
                if (blend <= 0f)
                {
                    actorForward = previousForward;
                }
                else if (blend < 0.999f)
                {
                    Vector3 blendedForward = Vector3.Lerp(previousForward, sampledForward, blend);
                    actorForward = blendedForward.LengthSquared() > 0.0001f
                        ? Vector3.Normalize(blendedForward)
                        : sampledForward;
                }
            }

            if (actorForward.LengthSquared() <= 0.0001f)
            {
                actorForward = actorDirection.LengthSquared() > 0.0001f
                    ? Vector3.Normalize(actorDirection)
                    : Vector3.UnitX;
            }

            _taxiActorSmoothedForwardByPath[route.PathId] = actorForward;

            float yawRadians = ComputeTaxiActorYawRadians(actorForward);
            string modelPath = GetResolvedTaxiActorModelPath(route.PathId)?.Replace('/', '\\') ?? string.Empty;

            _taxiActorPoseByPath[route.PathId] = new TaxiActorPose(
                route.PathId,
                actorPosition,
                actorForward,
                yawRadians,
                scale,
                modelPath);

            // Pose simulation is independent of asset availability. A ride
            // camera can follow a route while its model is still queued or
            // unavailable; only the render instance needs a non-empty model key.
            if (string.IsNullOrWhiteSpace(modelPath))
                continue;

            string key = WorldAssetManager.NormalizeKey(modelPath);
            _assets.QueueMdxLoad(key);

            var transform = Matrix4x4.CreateScale(scale)
                * Matrix4x4.CreateRotationZ(yawRadians)
                * Matrix4x4.CreateTranslation(actorPosition);

            Vector3 boundsMin;
            Vector3 boundsMax;
            Vector3 localMin = Vector3.Zero;
            Vector3 localMax = Vector3.Zero;
            bool boundsResolved = false;
            if (_assets.TryGetMdxBounds(key, out Vector3 modelMin, out Vector3 modelMax))
            {
                localMin = modelMin;
                localMax = modelMax;
                boundsResolved = true;
                TransformBounds(modelMin, modelMax, transform, out boundsMin, out boundsMax);
            }
            else
            {
                boundsMin = actorPosition - new Vector3(2f);
                boundsMax = actorPosition + new Vector3(2f);
            }

            _taxiActorInstances.Add(new ObjectInstance
            {
                ModelKey = key,
                Transform = transform,
                BoundsMin = boundsMin,
                BoundsMax = boundsMax,
                LocalBoundsMin = localMin,
                LocalBoundsMax = localMax,
                BoundsResolved = boundsResolved,
                ModelName = Path.GetFileName(modelPath),
                ModelPath = modelPath,
                PlacementPosition = actorPosition,
                PlacementRotation = new Vector3(0f, 0f, yawRadians * (180f / MathF.PI)),
                PlacementScale = scale,
                UniqueId = -route.PathId
            });
        }

        foreach (int stalePathId in _taxiActorTravelByPath.Keys.Except(activePathIds).ToList())
            _taxiActorTravelByPath.Remove(stalePathId);

        foreach (int stalePathId in _taxiActorPoseByPath.Keys.Except(activePathIds).ToList())
            _taxiActorPoseByPath.Remove(stalePathId);

        foreach (int stalePathId in _taxiActorSmoothedForwardByPath.Keys.Except(activePathIds).ToList())
            _taxiActorSmoothedForwardByPath.Remove(stalePathId);
    }

    private TaxiPathLoader.TaxiNode? ResolveTaxiActorNode(TaxiPathLoader.TaxiRoute route)
    {
        if (_taxiLoader == null)
            return null;

        if (_selectedTaxiNodeId >= 0)
        {
            var selectedNode = GetTaxiNode(_selectedTaxiNodeId);
            if (selectedNode != null && (route.FromNodeId == selectedNode.Id || route.ToNodeId == selectedNode.Id))
                return selectedNode;
        }

        var fromNode = GetTaxiNode(route.FromNodeId);
        if (fromNode != null && !string.IsNullOrWhiteSpace(fromNode.MountModelPath))
            return fromNode;

        var toNode = GetTaxiNode(route.ToNodeId);
        if (toNode != null && !string.IsNullOrWhiteSpace(toNode.MountModelPath))
            return toNode;

        return fromNode ?? toNode;
    }

    internal static bool TryGetTaxiRouteSelectionPoint(TaxiPathLoader.TaxiRoute route, out Vector3 point)
    {
        if (route.Waypoints.Count == 0)
        {
            point = Vector3.Zero;
            return false;
        }

        float routeLength = GetRouteLength(route.Waypoints);
        if (routeLength <= 1f)
        {
            point = route.Waypoints[route.Waypoints.Count / 2];
            return true;
        }

        SampleRoute(route.Waypoints, routeLength * 0.5f, out point, out _);
        return true;
    }

    private static float GetRouteLength(List<Vector3> waypoints)
    {
        float total = 0f;
        for (int i = 0; i < waypoints.Count - 1; i++)
            total += Vector3.Distance(waypoints[i], waypoints[i + 1]);
        return total;
    }

    private static void SampleRoute(List<Vector3> waypoints, float distance, out Vector3 position, out Vector3 direction)
    {
        float remaining = distance;
        for (int i = 0; i < waypoints.Count - 1; i++)
        {
            Vector3 start = waypoints[i];
            Vector3 end = waypoints[i + 1];
            Vector3 segment = end - start;
            float segmentLength = segment.Length();
            if (segmentLength <= 0.001f)
                continue;

            if (remaining <= segmentLength)
            {
                float t = remaining / segmentLength;
                position = Vector3.Lerp(start, end, t);
                direction = Vector3.Normalize(segment);
                return;
            }

            remaining -= segmentLength;
        }

        position = waypoints[^1];
        direction = waypoints[^1] - waypoints[^2];
        if (direction.LengthSquared() > 0.0001f)
            direction = Vector3.Normalize(direction);
    }

    private static Vector3 SampleSmoothedTaxiRouteDirection(List<Vector3> waypoints, float distance, float routeLength)
    {
        if (waypoints.Count < 2)
            return Vector3.UnitX;

        float sampleWindow = MathF.Min(TaxiActorHeadingSampleWindow, MathF.Max(1f, routeLength * 0.1f));
        float behindDistance = WrapTaxiRouteDistance(distance - sampleWindow * 0.5f, routeLength);
        float aheadDistance = WrapTaxiRouteDistance(distance + sampleWindow * 0.5f, routeLength);

        SampleRoute(waypoints, behindDistance, out Vector3 behindPosition, out Vector3 behindDirection);
        SampleRoute(waypoints, aheadDistance, out Vector3 aheadPosition, out Vector3 aheadDirection);

        Vector3 tangent = aheadPosition - behindPosition;
        if (tangent.LengthSquared() > 0.0001f)
            return Vector3.Normalize(tangent);

        Vector3 fallback = aheadDirection.LengthSquared() > 0.0001f ? aheadDirection : behindDirection;
        if (fallback.LengthSquared() > 0.0001f)
            return Vector3.Normalize(fallback);

        return Vector3.UnitX;
    }

    private static float WrapTaxiRouteDistance(float distance, float routeLength)
    {
        if (routeLength <= 0f)
            return 0f;

        while (distance < 0f)
            distance += routeLength;

        while (distance >= routeLength)
            distance -= routeLength;

        return distance;
    }

    private static float ComputeTaxiActorYawRadians(Vector3 actorForward)
    {
        Vector3 horizontalForward = new Vector3(actorForward.X, actorForward.Y, 0f);
        if (horizontalForward.LengthSquared() <= 0.0001f)
            return 0f;

        horizontalForward = Vector3.Normalize(horizontalForward);
        return MathF.Atan2(horizontalForward.Y, horizontalForward.X);
    }
}
