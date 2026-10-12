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
/// World-object filters: UniqueId range/tile scope, object-path filters, and UniqueId archaeology layers.
/// Moved verbatim from <see cref="WorldScene"/> (Spec 255). Scene state it still needs comes
/// only through <see cref="IWorldSceneHost"/>; the bridge members below keep the names the moved
/// code used inside WorldScene, so no moved body was edited.
/// </summary>
public sealed class SceneObjectFilters
{
    private readonly IWorldSceneHost _host;

    internal SceneObjectFilters(IWorldSceneHost host)
    {
        _host = host;
    }

    // Host bridge (same names as the WorldScene members).
    private ref bool _instancesDirty => ref _host.InstancesDirty;
    private ref List<ObjectInstance> _mdxInstances => ref _host.MdxInstances;
    private ref List<ObjectInstance> _wmoInstances => ref _host.WmoInstances;
    private void RebuildInstanceLists() => _host.RebuildInstanceLists();

    private bool _objectPathFiltersEnabled = true;
    private const int UniqueIdLayerGapThreshold = 100;
    private bool _uniqueIdFilterEnabled;
    private UniqueIdVisibilityScope _uniqueIdVisibilityScope = UniqueIdVisibilityScope.PerMap;
    private int _uniqueIdFilterMin = -1;
    private int _uniqueIdFilterMax = -1;
    private (int tileX, int tileY)? _uniqueIdFilterTile;
    private readonly List<ObjectPathFilterEntry> _objectPathFilters = new();
    public bool ObjectPathFiltersEnabled { get => _objectPathFiltersEnabled; set => _objectPathFiltersEnabled = value; }
    public IReadOnlyList<ObjectPathFilterEntry> ObjectPathFilters => _objectPathFilters;
    public bool UniqueIdFilterEnabled { get => _uniqueIdFilterEnabled; set => _uniqueIdFilterEnabled = value; }
    public UniqueIdVisibilityScope UniqueIdVisibilityScope { get => _uniqueIdVisibilityScope; set => _uniqueIdVisibilityScope = value; }
    public int UniqueIdFilterMin { get => _uniqueIdFilterMin; set => _uniqueIdFilterMin = value; }
    public int UniqueIdFilterMax { get => _uniqueIdFilterMax; set => _uniqueIdFilterMax = value; }
    public (int tileX, int tileY)? UniqueIdFilterTile => _uniqueIdFilterTile;

    public void SetUniqueIdFilterTile(int tileX, int tileY)
    {
        _uniqueIdFilterTile = (tileX, tileY);
    }

    public void SetUniqueIdFilterRange(int minUniqueId, int maxUniqueId)
    {
        if (minUniqueId <= maxUniqueId)
        {
            _uniqueIdFilterMin = minUniqueId;
            _uniqueIdFilterMax = maxUniqueId;
            return;
        }

        _uniqueIdFilterMin = maxUniqueId;
        _uniqueIdFilterMax = minUniqueId;
    }

    public void ResetUniqueIdFilter()
    {
        _uniqueIdFilterEnabled = false;
        _uniqueIdFilterMin = -1;
        _uniqueIdFilterMax = -1;
    }

    public bool AddObjectPathFilter(string pathPrefix, bool appliesToWmo, bool appliesToMdx)
    {
        string normalizedPrefix = ObjectPathFilterEntry.NormalizePrefix(pathPrefix);
        if (string.IsNullOrWhiteSpace(normalizedPrefix) || (!appliesToWmo && !appliesToMdx))
            return false;

        ObjectPathFilterEntry entry = new(normalizedPrefix, appliesToWmo, appliesToMdx);
        if (_objectPathFilters.Contains(entry))
            return false;

        _objectPathFilters.Add(entry);
        _objectPathFilters.Sort(static (left, right) => string.Compare(left.PathPrefix, right.PathPrefix, StringComparison.OrdinalIgnoreCase));
        return true;
    }

    public bool RemoveObjectPathFilter(string pathPrefix, bool appliesToWmo, bool appliesToMdx)
    {
        string normalizedPrefix = ObjectPathFilterEntry.NormalizePrefix(pathPrefix);
        if (string.IsNullOrWhiteSpace(normalizedPrefix))
            return false;

        return _objectPathFilters.RemoveAll(entry =>
            string.Equals(entry.PathPrefix, normalizedPrefix, StringComparison.OrdinalIgnoreCase)
            && entry.AppliesToWmo == appliesToWmo
            && entry.AppliesToMdx == appliesToMdx) > 0;
    }

    public void ClearObjectPathFilters()
    {
        _objectPathFilters.Clear();
    }

    public bool TryGetUniqueIdFilterRange(out int minUniqueId, out int maxUniqueId, out int instanceCount)
    {
        if (_instancesDirty)
            RebuildInstanceLists();

        minUniqueId = int.MaxValue;
        maxUniqueId = int.MinValue;
        instanceCount = 0;

        AccumulateUniqueIdFilterRange(_wmoInstances, ref minUniqueId, ref maxUniqueId, ref instanceCount);
        AccumulateUniqueIdFilterRange(_mdxInstances, ref minUniqueId, ref maxUniqueId, ref instanceCount);

        if (instanceCount <= 0)
        {
            minUniqueId = 0;
            maxUniqueId = 0;
            return false;
        }

        return true;
    }

    public IReadOnlyList<UniqueIdArchaeologyLayer> GetUniqueIdArchaeologyLayers()
    {
        if (_instancesDirty)
            RebuildInstanceLists();

        var countsById = new SortedDictionary<int, (int wmoCount, int mdxCount)>();
        AccumulateUniqueIdLayerCandidates(_wmoInstances, isWmo: true, countsById);
        AccumulateUniqueIdLayerCandidates(_mdxInstances, isWmo: false, countsById);

        if (countsById.Count == 0)
            return Array.Empty<UniqueIdArchaeologyLayer>();

        var layers = new List<UniqueIdArchaeologyLayer>();
        int layerNumber = 1;
        int layerStart = 0;
        int layerEnd = 0;
        int previousId = 0;
        int placementCount = 0;
        int wmoCount = 0;
        int mdxCount = 0;
        bool hasLayer = false;

        foreach ((int uniqueId, (int layerWmoCount, int layerMdxCount) counts) in countsById)
        {
            if (!hasLayer)
            {
                layerStart = uniqueId;
                hasLayer = true;
            }
            else if (uniqueId - previousId > UniqueIdLayerGapThreshold)
            {
                layers.Add(new UniqueIdArchaeologyLayer(layerNumber++, layerStart, layerEnd, placementCount, wmoCount, mdxCount));
                layerStart = uniqueId;
                placementCount = 0;
                wmoCount = 0;
                mdxCount = 0;
            }

            layerEnd = uniqueId;
            previousId = uniqueId;
            placementCount += counts.layerWmoCount + counts.layerMdxCount;
            wmoCount += counts.layerWmoCount;
            mdxCount += counts.layerMdxCount;
        }

        if (hasLayer)
            layers.Add(new UniqueIdArchaeologyLayer(layerNumber, layerStart, layerEnd, placementCount, wmoCount, mdxCount));

        return layers;
    }

    private void AccumulateUniqueIdFilterRange(
        IReadOnlyList<ObjectInstance> instances,
        ref int minUniqueId,
        ref int maxUniqueId,
        ref int instanceCount)
    {
        for (int i = 0; i < instances.Count; i++)
        {
            ObjectInstance inst = instances[i];
            if (inst.UniqueId <= 0 || !MatchesUniqueIdFilterScope(inst))
                continue;

            minUniqueId = Math.Min(minUniqueId, inst.UniqueId);
            maxUniqueId = Math.Max(maxUniqueId, inst.UniqueId);
            instanceCount++;
        }
    }

    private void AccumulateUniqueIdLayerCandidates(
        IReadOnlyList<ObjectInstance> instances,
        bool isWmo,
        SortedDictionary<int, (int wmoCount, int mdxCount)> countsById)
    {
        for (int i = 0; i < instances.Count; i++)
        {
            ObjectInstance inst = instances[i];
            if (inst.UniqueId <= 0 || !MatchesUniqueIdFilterScope(inst))
                continue;

            countsById.TryGetValue(inst.UniqueId, out (int wmoCount, int mdxCount) counts);
            counts = isWmo
                ? (counts.wmoCount + 1, counts.mdxCount)
                : (counts.wmoCount, counts.mdxCount + 1);
            countsById[inst.UniqueId] = counts;
        }
    }

    private bool MatchesUniqueIdFilterScope(in ObjectInstance inst)
    {
        if (_uniqueIdVisibilityScope != UniqueIdVisibilityScope.CameraTile)
            return true;

        if (!_uniqueIdFilterTile.HasValue || !inst.HasTileCoordinate)
            return false;

        return inst.TileX == _uniqueIdFilterTile.Value.tileX
            && inst.TileY == _uniqueIdFilterTile.Value.tileY;
    }

    internal bool ShouldHideObjectInstanceByUniqueId(in ObjectInstance inst)
    {
        if (ShouldHideObjectInstanceByPathFilter(inst))
            return true;

        if (!_uniqueIdFilterEnabled
            || _uniqueIdFilterMin < 0
            || _uniqueIdFilterMax < 0
            || inst.UniqueId <= 0
            || !MatchesUniqueIdFilterScope(inst))
        {
            return false;
        }

        int minUniqueId = Math.Min(_uniqueIdFilterMin, _uniqueIdFilterMax);
        int maxUniqueId = Math.Max(_uniqueIdFilterMin, _uniqueIdFilterMax);
        return inst.UniqueId < minUniqueId || inst.UniqueId > maxUniqueId;
    }

    private bool ShouldHideObjectInstanceByPathFilter(in ObjectInstance inst)
    {
        if (!_objectPathFiltersEnabled || _objectPathFilters.Count == 0 || string.IsNullOrWhiteSpace(inst.ModelPath))
            return false;

        string normalizedPath = ObjectPathFilterEntry.NormalizePrefix(inst.ModelPath);
        if (string.IsNullOrWhiteSpace(normalizedPath))
            return false;

        for (int i = 0; i < _objectPathFilters.Count; i++)
        {
            ObjectPathFilterEntry entry = _objectPathFilters[i];
            if (entry.MatchesModelPath(normalizedPath))
                return true;
        }

        return false;
    }
}
