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
/// World-scene hover info, scene-object pick hits by ray, and the wireframe-reveal brush.
/// Moved verbatim from <see cref="WorldScene"/> (Spec 255). Scene state it still needs comes
/// only through <see cref="IWorldSceneHost"/>; the bridge members below keep the names the moved
/// code used inside WorldScene, so no moved body was edited.
/// </summary>
public sealed class SceneHoverPickController
{
    private readonly IWorldSceneHost _host;

    internal SceneHoverPickController(IWorldSceneHost host)
    {
        _host = host;
    }

    // Host bridge (same names as the WorldScene members).
    private WorldAssetManager _assets => _host.Assets;
    private ref bool _doodadsVisible => ref _host.DoodadsVisible;
    private GL _gl => _host.Gl;
    private ref bool _instancesDirty => ref _host.InstancesDirty;
    private ref float _lastHoverPickFogEnd => ref _host.LastHoverPickFogEnd;
    private ref List<ObjectInstance> _mdxInstances => ref _host.MdxInstances;
    private Pm4OverlayScene _pm4Overlay => _host.Pm4Overlay;
    private ref bool _showWlLiquids => ref _host.ShowWlLiquids;
    private TerrainManager _terrainManager => _host.TerrainManager;
    private ref bool _wireframeRevealEnabled => ref _host.WireframeRevealEnabledField;
    private ref WlLiquidLoader? _wlLoader => ref _host.WlLoader;
    private ref List<ObjectInstance> _wmoInstances => ref _host.WmoInstances;
    private ref bool _wmosVisible => ref _host.WmosVisible;
    private void RebuildInstanceLists() => _host.RebuildInstanceLists();
    private IModelRenderer? ResolveVisibleMdxRenderer(WorldRenderFrame frame, string modelKey) => _host.ResolveVisibleMdxRenderer(frame, modelKey);
    private WmoRenderer? ResolveVisibleWmoRenderer(WorldRenderFrame frame, string modelKey) => _host.ResolveVisibleWmoRenderer(frame, modelKey);
    private bool ShouldHideObjectInstanceByUniqueId(in ObjectInstance inst) => _host.ShouldHideObjectInstanceByUniqueId(in inst);
    private IModelRenderer? TryGetQueuedMdx(string modelKey) => _host.TryGetQueuedMdx(modelKey);
    private WmoRenderer? TryGetQueuedWmo(string modelKey) => _host.TryGetQueuedWmo(modelKey);
    private const float MaxWorldObjectViewDistance = WorldScene.MaxWorldObjectViewDistance;
    private static bool AreFiniteOrderedBounds(Vector3 min, Vector3 max) => WorldScene.AreFiniteOrderedBounds(min, max);
    private static Vector3 GetSceneObjectSelectionPoint(in ObjectInstance instance) => WorldScene.GetSceneObjectSelectionPoint(in instance);
    private static float RayAABBIntersect(Vector3 origin, Vector3 dir, Vector3 bmin, Vector3 bmax) => WorldScene.RayAABBIntersect(origin, dir, bmin, bmax);
    private static (Vector3 origin, Vector3 dir) ScreenToRay(float ndcX, float ndcY, Matrix4x4 view, Matrix4x4 proj) => WorldScene.ScreenToRay(ndcX, ndcY, view, proj);
    private static bool TryGetSceneObjectChunkKey(in ObjectInstance instance, out (int tileX, int tileY, int chunkX, int chunkY) key) => WorldScene.TryGetSceneObjectChunkKey(in instance, out key);
    private static bool TryGetTerrainChunkKey(float worldX, float worldY, out (int tileX, int tileY, int chunkX, int chunkY) key) => WorldScene.TryGetTerrainChunkKey(worldX, worldY, out key);

    private bool _limitHoveredAssetRange = true;
    private bool _useDynamicHoveredAssetRange = false;
    private float _hoveredAssetMaxDistance = 533.33f;
    private const float HoverInfoBrushPixels = 32f;
    private const float HoverInfoMaxScreenRadius = 96f;
    private const float WireframeRevealBrushPixels = 96f;
    private const float WireframeRevealMaxScreenRadius = 220f;
    private readonly List<int> _wireframeRevealWmoIndices = new();
    private readonly List<int> _wireframeRevealMdxIndices = new();
    private HoveredAssetInfo? _hoveredAssetInfo;
    private bool _showHoveredAssetTooltips = true;
    public bool WireframeRevealEnabled => _wireframeRevealEnabled;
    public HoveredAssetInfo? HoveredAssetInfo => _hoveredAssetInfo;
    public bool ShowHoveredAssetTooltips { get => _showHoveredAssetTooltips; set => _showHoveredAssetTooltips = value; }
    public bool LimitHoveredAssetRange { get => _limitHoveredAssetRange; set => _limitHoveredAssetRange = value; }
    public bool UseDynamicHoveredAssetRange { get => _useDynamicHoveredAssetRange; set => _useDynamicHoveredAssetRange = value; }
    public float HoveredAssetMaxDistance
    {
        get => _hoveredAssetMaxDistance;
        set => _hoveredAssetMaxDistance = Math.Clamp(value, 10f, MaxWorldObjectViewDistance);
    }
    public float EffectiveHoveredAssetMaxDistance => ComputeEffectiveHoveredAssetMaxDistance();

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

    public void UpdateHoveredAssetInfo(Matrix4x4 view, Matrix4x4 proj,
        float mouseViewportX, float mouseViewportY, float viewportWidth, float viewportHeight)
    {
        float safeViewportWidth = Math.Max(viewportWidth, 1f);
        float safeViewportHeight = Math.Max(viewportHeight, 1f);
        float ndcX = (mouseViewportX / safeViewportWidth) * 2f - 1f;
        float ndcY = 1f - (mouseViewportY / safeViewportHeight) * 2f;
        var (rayOrigin, rayDir) = ScreenToRay(ndcX, ndcY, view, proj);

        bool hasSceneRayHit = TryBuildHoveredSceneInfoByRay(rayOrigin, rayDir, out HoveredAssetInfo sceneRayInfo, out float sceneRayDistance);
        HoveredAssetInfo pm4RayInfo = default;
        float pm4RayDistance = float.MaxValue;
        bool hasPm4RayHit = _pm4Overlay._showPm4Overlay
            && _pm4Overlay.TryBuildHoveredPm4InfoByRay(rayOrigin, rayDir, out pm4RayInfo, out pm4RayDistance);

        WorldSceneHoverSource raySource = WorldSceneSelectionService.ChooseHoverRaySource(
            hasSceneRayHit, sceneRayDistance, hasPm4RayHit, pm4RayDistance, _pm4Overlay._pm4OverlayIgnoreDepth);
        if (raySource == WorldSceneHoverSource.Pm4)
        {
            _hoveredAssetInfo = pm4RayInfo.WithPreciseRayHit();
            return;
        }

        if (raySource == WorldSceneHoverSource.Scene)
        {
            _hoveredAssetInfo = sceneRayInfo.WithPreciseRayHit();
            return;
        }

        bool hasSceneBrushHit = TryBuildHoveredSceneInfo(
            view,
            proj,
            mouseViewportX,
            mouseViewportY,
            viewportWidth,
            viewportHeight,
            out HoveredAssetInfo sceneBrushInfo,
            out float sceneBrushDistanceSq,
            out float sceneBrushDepth);
        HoveredAssetInfo pm4BrushInfo = default;
        int hoveredPm4Count = 0;
        float pm4BrushDistanceSq = float.MaxValue;
        float pm4BrushDepth = float.MaxValue;
        bool hasPm4BrushHit = _pm4Overlay._showPm4Overlay
            && _pm4Overlay.TryBuildHoveredPm4Info(
                view,
                proj,
                mouseViewportX,
                mouseViewportY,
                viewportWidth,
                viewportHeight,
                out pm4BrushInfo,
                out hoveredPm4Count,
                out pm4BrushDistanceSq,
                out pm4BrushDepth);

        WorldSceneHoverSource brushSource = WorldSceneSelectionService.ChooseHoverBrushSource(
            hasSceneBrushHit, sceneBrushDistanceSq, sceneBrushDepth,
            hasPm4BrushHit, pm4BrushDistanceSq, pm4BrushDepth, _pm4Overlay._pm4OverlayIgnoreDepth);
        if (brushSource == WorldSceneHoverSource.Pm4)
        {
            _hoveredAssetInfo = new HoveredAssetInfo(
                pm4BrushInfo.AssetKind,
                pm4BrushInfo.DisplayName,
                pm4BrushInfo.SourcePath,
                pm4BrushInfo.DetailLine,
                pm4BrushInfo.WorldPosition,
                Math.Max(0, hoveredPm4Count - 1),
                pm4BrushInfo.Pm4ObjectKey,
                pm4BrushInfo.SceneObjectType,
                pm4BrushInfo.SceneObjectIndex,
                pm4BrushInfo.WlBodyKey);
            return;
        }

        if (hasSceneBrushHit)
        {
            _hoveredAssetInfo = sceneBrushInfo;
            return;
        }

        _hoveredAssetInfo = null;
    }

    public void ClearWireframeReveal()
    {
        _wireframeRevealWmoIndices.Clear();
        _wireframeRevealMdxIndices.Clear();
    }

    public void ClearHoveredAssetInfo()
    {
        _hoveredAssetInfo = null;
    }

    private bool TryBuildHoveredSceneInfo(
        Matrix4x4 view,
        Matrix4x4 proj,
        float mouseViewportX,
        float mouseViewportY,
        float viewportWidth,
        float viewportHeight,
        out HoveredAssetInfo info,
        out float bestDistanceSq,
        out float bestDepth)
    {
        info = default;
        bestDistanceSq = float.MaxValue;
        bestDepth = float.MaxValue;

        LiquidRenderer? liquidRenderer = _terrainManager?.LiquidRenderer;
        var candidateInfos = new List<HoveredAssetInfo>();
        var brushCandidates = new List<WorldSceneBrushCandidate>();

        void ConsiderCandidate(HoveredAssetInfo candidateInfo, float distanceSq, float depth)
        {
            brushCandidates.Add(new WorldSceneBrushCandidate(candidateInfos.Count, distanceSq, depth, candidateInfo.WorldPosition));
            candidateInfos.Add(candidateInfo);
        }

        if (_wmosVisible)
        {
            for (int i = 0; i < _wmoInstances.Count; i++)
            {
                ObjectInstance inst = _wmoInstances[i];
                if (ShouldHideObjectInstanceByUniqueId(inst))
                    continue;

                if (!TryMeasureHoverInfoHit(inst.BoundsMin, inst.BoundsMax, view, proj, mouseViewportX, mouseViewportY, viewportWidth, viewportHeight, out float distanceSq, out float depth))
                    continue;

                ConsiderCandidate(BuildHoveredObjectInfo("WMO", inst, ObjectType.Wmo, i), distanceSq, depth);
            }
        }

        if (_doodadsVisible)
        {
            for (int i = 0; i < _mdxInstances.Count; i++)
            {
                ObjectInstance inst = _mdxInstances[i];
                if (ShouldHideObjectInstanceByUniqueId(inst))
                    continue;

                if (!TryMeasureHoverInfoHit(inst.BoundsMin, inst.BoundsMax, view, proj, mouseViewportX, mouseViewportY, viewportWidth, viewportHeight, out float distanceSq, out float depth))
                    continue;

                ConsiderCandidate(BuildHoveredObjectInfo("MDX", inst, ObjectType.Mdx, i), distanceSq, depth);
            }
        }

        if (_showWlLiquids && _wlLoader != null)
        {
            for (int i = 0; i < _wlLoader.Bodies.Count; i++)
            {
                WlLiquidBody body = _wlLoader.Bodies[i];
                if (liquidRenderer != null && !liquidRenderer.IsWlBodyVisible(body.BodyKey))
                    continue;

                if (!TryMeasureHoverInfoHit(body.BoundsMin, body.BoundsMax, view, proj, mouseViewportX, mouseViewportY, viewportWidth, viewportHeight, out float distanceSq, out float depth))
                    continue;

                ConsiderCandidate(BuildHoveredWlLiquidInfo(body), distanceSq, depth);
            }
        }

        // Range check, cursor-distance/depth choice and the eligible count live in the Core selection
        // service (Spec 228; Epic 251 U-01 E4). Candidates keep their WMO, MDX, liquid order.
        WorldSceneBrushResult brush = WorldSceneSelectionService.SelectBrush(
            new WorldSceneSelectionPolicy(
                _limitHoveredAssetRange,
                ComputeEffectiveHoveredAssetMaxDistance(),
                _limitHoveredAssetRange ? _pm4Overlay.GetPm4LoadAnchorCameraPosition() : Vector3.Zero),
            brushCandidates);
        if (brush.Status != WorldSceneSelectionStatus.Hit)
            return false;

        HoveredAssetInfo bestCandidate = candidateInfos[brush.BestId];
        int hitCount = brush.EligibleCount;
        bestDistanceSq = brush.BestScreenDistanceSq;
        bestDepth = brush.BestDepth;
        info = new HoveredAssetInfo(
            bestCandidate.AssetKind,
            bestCandidate.DisplayName,
            bestCandidate.SourcePath,
            bestCandidate.DetailLine,
            bestCandidate.WorldPosition,
            Math.Max(0, hitCount - 1),
            bestCandidate.Pm4ObjectKey,
            bestCandidate.SceneObjectType,
            bestCandidate.SceneObjectIndex,
            bestCandidate.WlBodyKey);
        return true;
    }

    private bool TryBuildHoveredSceneInfoByRay(Vector3 rayOrigin, Vector3 rayDir, out HoveredAssetInfo info, out float distance)
    {
        info = default;
        distance = float.MaxValue;
        LiquidRenderer? liquidRenderer = _terrainManager?.LiquidRenderer;

        // Hover and click must agree about scene-object identity. Reuse the proven Spec 211 click
        // picker so WMO doodads participate in hover and an enclosing WMO AABB cannot hide them.
        var sceneHits = new List<SceneObjectPickHit>();
        CollectSceneObjectPickHits(rayOrigin, rayDir, sceneHits, logHits: false);

        var liquidTargets = new List<WorldSceneRayTarget>();
        if (_showWlLiquids && _wlLoader != null)
        {
            Vector3 padding = new(2f, 2f, 1f);
            for (int i = 0; i < _wlLoader.Bodies.Count; i++)
            {
                WlLiquidBody body = _wlLoader.Bodies[i];
                if (liquidRenderer != null && !liquidRenderer.IsWlBodyVisible(body.BodyKey))
                    continue;

                liquidTargets.Add(new WorldSceneRayTarget(i, RayAABBIntersect(rayOrigin, rayDir, body.BoundsMin - padding, body.BoundsMax + padding)));
            }
        }

        // Visibility, WMO container fall-through, range and nearest-first live in the Core
        // selection service (Spec 228; Epic 251 U-01 E4).
        WorldSceneHoverRayResult hover = WorldSceneSelectionAdapter.ResolveHoverRay(
            sceneHits,
            new WorldSceneSelectionPolicy(_limitHoveredAssetRange, ComputeEffectiveHoveredAssetMaxDistance(), Vector3.Zero),
            _wmosVisible,
            _doodadsVisible,
            liquidTargets);

        if (hover.Target == WorldSceneHoverRayTarget.SceneObject)
        {
            SceneObjectPickHit hit = sceneHits[hover.Id];
            info = hit.ObjectType == ObjectType.WmoDoodad
                ? BuildHoveredWmoDoodadInfo(hit)
                : BuildHoveredScenePickHitInfo(hit);
            distance = hit.Distance;
        }
        else if (hover.Target == WorldSceneHoverRayTarget.LiquidBody)
        {
            info = BuildHoveredWlLiquidInfo(_wlLoader!.Bodies[hover.Id]);
            distance = hover.Distance;
        }

        return distance < float.MaxValue;
    }

    private static HoveredAssetInfo BuildHoveredScenePickHitInfo(in SceneObjectPickHit hit)
    {
        return new HoveredAssetInfo(
            hit.KindLabel,
            hit.ModelName,
            hit.ModelPath,
            $"UniqueId: {hit.UniqueId}",
            hit.PlacementPosition,
            0,
            null,
            hit.ObjectType,
            hit.ObjectIndex,
            null);
    }

    private HoveredAssetInfo BuildHoveredWmoDoodadInfo(in SceneObjectPickHit hit)
    {
        string detail = $"Active doodad index: {hit.ObjectIndex}";
        string parentPath = string.Empty;
        string parentName = $"WMO [{hit.ParentWmoIndex}]";

        if (hit.ParentWmoIndex >= 0 && hit.ParentWmoIndex < _wmoInstances.Count)
        {
            ObjectInstance parent = _wmoInstances[hit.ParentWmoIndex];
            parentPath = parent.ModelPath;
            parentName = string.IsNullOrWhiteSpace(parent.ModelName) ? parent.ModelPath : parent.ModelName;

            if (_assets.TryGetLoadedWmo(parent.ModelKey, out WmoRenderer? renderer) && renderer != null
                && renderer.TryGetDoodadInfo(hit.ObjectIndex, out WmoDoodadInfo doodad))
            {
                string setName = renderer.GetDoodadSetName(renderer.ActiveDoodadSet);
                List<int> renderGroups = renderer.GetRenderGroupsForDoodadDef(doodad.DoodadDefIndex);
                string groupText = renderGroups.Count == 0
                    ? "no MODR group reference"
                    : string.Join(", ", renderGroups.Select(renderer.GetRenderGroupName));
                detail = $"MODD definition: {doodad.DoodadDefIndex}   MODN offset: {doodad.NameIndex}\n"
                    + $"Doodad set [{renderer.ActiveDoodadSet}]: {setName}\n"
                    + $"WMO groups: {groupText}\n"
                    + $"Parent WMO [{hit.ParentWmoIndex}]: {parentName}";
            }
            else
            {
                detail += $"\nParent WMO [{hit.ParentWmoIndex}]: {parentName}";
            }
        }

        return new HoveredAssetInfo(
            "WMO Doodad",
            hit.ModelName,
            hit.ModelPath,
            detail,
            hit.SelectionPoint,
            0,
            null,
            ObjectType.WmoDoodad,
            hit.ObjectIndex,
            null,
            parentWmoIndex: hit.ParentWmoIndex,
            parentSourcePath: parentPath);
    }

    public bool TryPickSceneObjectByRay(Vector3 rayOrigin, Vector3 rayDir, out ObjectType objectType, out int objectIndex, out float distance)
    {
        var hits = new List<SceneObjectPickHit>();
        CollectSceneObjectPickHits(rayOrigin, rayDir, hits, logHits: true);

        if (hits.Count == 0)
        {
            objectType = ObjectType.None;
            objectIndex = -1;
            distance = float.MaxValue;
            return false;
        }

        SceneObjectPickHit bestHit = hits[0];
        objectType = bestHit.ObjectType;
        objectIndex = bestHit.ObjectIndex;
        distance = bestHit.Distance;
        return true;
    }

    public bool TryPickSceneObjectsByRay(Vector3 rayOrigin, Vector3 rayDir, List<SceneObjectPickHit> hits)
    {
        return TryPickSceneObjectsByRay(rayOrigin, rayDir, hits, null, null);
    }

    public bool TryPickSceneObjectsByRay(
        Vector3 rayOrigin,
        Vector3 rayDir,
        List<SceneObjectPickHit> hits,
        (int tileX, int tileY, int chunkX, int chunkY)? clickedChunkKey,
        Vector3? clickedWorldPoint)
    {
        ArgumentNullException.ThrowIfNull(hits);
        CollectSceneObjectPickHits(rayOrigin, rayDir, hits, logHits: false, clickedChunkKey, clickedWorldPoint);
        return hits.Count > 0;
    }

    private void CollectSceneObjectPickHits(
        Vector3 rayOrigin,
        Vector3 rayDir,
        List<SceneObjectPickHit> hits,
        bool logHits,
        (int tileX, int tileY, int chunkX, int chunkY)? clickedChunkKey = null,
        Vector3? clickedWorldPoint = null)
    {
        hits.Clear();

        if (_instancesDirty)
            RebuildInstanceLists();

        // Pick padding is applied in the model's own local space, so it already scales with the
        // placement. It was 2 yd for WMOs and 1 yd for doodads, which inflated every click volume by
        // that much in all six directions: a nearby object then swallowed rays aimed past it, which
        // is what made objects close to the camera hard to inspect. Keep just enough forgiveness for
        // thin geometry such as fences and poles.
        AppendSceneObjectPickHits(rayOrigin, rayDir, hits, _wmoInstances, ObjectType.Wmo, new Vector3(0.25f, 0.25f, 0.25f), clickedChunkKey, clickedWorldPoint);
        AppendWmoDoodadPickHits(rayOrigin, rayDir, hits, clickedChunkKey, clickedWorldPoint);
        AppendSceneObjectPickHits(rayOrigin, rayDir, hits, _mdxInstances, ObjectType.Mdx, new Vector3(0.1f, 0.1f, 0.1f), clickedChunkKey, clickedWorldPoint);

        // Range limit, clicked-chunk filter and click ranking live in the Core selection service
        // (Spec 228; Epic 251 U-01 E4).
        WorldSceneSelectionAdapter.RankClickHits(
            hits,
            clickedChunkKey.HasValue,
            new WorldSceneSelectionPolicy(_limitHoveredAssetRange, ComputeEffectiveHoveredAssetMaxDistance(), Vector3.Zero));

        if (!logHits || hits.Count == 0)
            return;

        ViewerLog.Debug(ViewerLog.Category.Terrain, $"[ObjectPick] Ray hit {hits.Count} objects:");
        foreach (SceneObjectPickHit hit in hits.Take(5))
            ViewerLog.Debug(ViewerLog.Category.Terrain, $"  {hit.KindLabel}[{hit.ObjectIndex}] {hit.ModelName} @ dist={hit.Distance:F1}");
        if (hits.Count > 5)
            ViewerLog.Debug(ViewerLog.Category.Terrain, $"  ... and {hits.Count - 5} more");
    }

    private void AppendSceneObjectPickHits(
        Vector3 rayOrigin,
        Vector3 rayDir,
        List<SceneObjectPickHit> hits,
        List<ObjectInstance> instances,
        ObjectType objectType,
        Vector3 padding,
        (int tileX, int tileY, int chunkX, int chunkY)? clickedChunkKey,
        Vector3? clickedWorldPoint)
    {
        for (int i = 0; i < instances.Count; i++)
        {
            ObjectInstance instance = instances[i];
            if (ShouldHideObjectInstanceByUniqueId(instance))
                continue;

            if (!TryRayIntersectInstanceBounds(rayOrigin, rayDir, instance, padding, out float distance))
                continue;

            Vector3 selectionPoint = GetSceneObjectSelectionPoint(instance);
            bool sharesClickedChunk = clickedChunkKey.HasValue
                && TryGetSceneObjectChunkKey(instance, out var instanceChunkKey)
                && instanceChunkKey == clickedChunkKey.Value;
            int chunkGridDistance = clickedChunkKey.HasValue && TryGetSceneObjectChunkKey(instance, out instanceChunkKey)
                ? Math.Abs(instanceChunkKey.tileX - clickedChunkKey.Value.tileX)
                    + Math.Abs(instanceChunkKey.tileY - clickedChunkKey.Value.tileY)
                    + Math.Abs(instanceChunkKey.chunkX - clickedChunkKey.Value.chunkX)
                    + Math.Abs(instanceChunkKey.chunkY - clickedChunkKey.Value.chunkY)
                : int.MaxValue;
            float selectionPointDistanceSq = clickedWorldPoint.HasValue
                ? Vector3.DistanceSquared(selectionPoint, clickedWorldPoint.Value)
                : float.MaxValue;

            hits.Add(new SceneObjectPickHit(
                objectType,
                i,
                distance,
                instance.ModelName,
                instance.ModelPath,
                instance.UniqueId,
                instance.PlacementPosition,
                instance.BoundsMin,
                instance.BoundsMax,
                selectionPoint,
                selectionPointDistanceSq,
                sharesClickedChunk,
                chunkGridDistance));
        }
    }

    private void AppendWmoDoodadPickHits(
        Vector3 rayOrigin,
        Vector3 rayDir,
        List<SceneObjectPickHit> hits,
        (int tileX, int tileY, int chunkX, int chunkY)? clickedChunkKey,
        Vector3? clickedWorldPoint)
    {
        var doodadHitsScratch = new List<(int index, float distance, Vector3 hitPoint, Vector3 boundsMin, Vector3 boundsMax, WmoDoodadInfo info)>();
        for (int wmoIndex = 0; wmoIndex < _wmoInstances.Count; wmoIndex++)
        {
            ObjectInstance wmo = _wmoInstances[wmoIndex];
            if (ShouldHideObjectInstanceByUniqueId(wmo))
                continue;

            if (!TryRayIntersectInstanceBounds(rayOrigin, rayDir, wmo, new Vector3(5f, 5f, 5f), out _))
                continue;

            if (!_assets.TryGetLoadedWmo(wmo.ModelKey, out WmoRenderer? wmoRenderer) || wmoRenderer == null)
                continue;

            doodadHitsScratch.Clear();
            if (!wmoRenderer.TryPickDoodadsByRay(rayOrigin, rayDir, wmo.Transform, doodadHitsScratch))
                continue;

            foreach (var dh in doodadHitsScratch)
            {
                bool sharesClickedChunk = clickedChunkKey.HasValue
                    && TryGetTerrainChunkKey(dh.hitPoint.X, dh.hitPoint.Y, out var chunkKey)
                    && chunkKey == clickedChunkKey.Value;

                int chunkGridDistance = clickedChunkKey.HasValue && TryGetTerrainChunkKey(dh.hitPoint.X, dh.hitPoint.Y, out chunkKey)
                    ? Math.Abs(chunkKey.tileX - clickedChunkKey.Value.tileX)
                        + Math.Abs(chunkKey.tileY - clickedChunkKey.Value.tileY)
                        + Math.Abs(chunkKey.chunkX - clickedChunkKey.Value.chunkX)
                        + Math.Abs(chunkKey.chunkY - clickedChunkKey.Value.chunkY)
                    : int.MaxValue;

                float selectionPointDistanceSq = clickedWorldPoint.HasValue
                    ? Vector3.DistanceSquared(dh.hitPoint, clickedWorldPoint.Value)
                    : float.MaxValue;

                hits.Add(new SceneObjectPickHit(
                    ObjectType.WmoDoodad,
                    dh.index,
                    dh.distance,
                    Path.GetFileName(dh.info.ModelPath),
                    dh.info.ModelPath,
                    // MODD has no uniqueId; the def index is not one. Putting it in this slot made
                    // the disambiguation list report a uniqueId the record does not have.
                    UniqueId: 0,
                    dh.hitPoint,
                    dh.boundsMin,
                    dh.boundsMax,
                    dh.hitPoint,
                    selectionPointDistanceSq,
                    sharesClickedChunk,
                    chunkGridDistance,
                    ParentWmoIndex: wmoIndex));
            }
        }
    }

    private float ComputeEffectiveHoveredAssetMaxDistance()
    {
        if (!_limitHoveredAssetRange)
            return float.MaxValue;

        if (!_useDynamicHoveredAssetRange)
            return _hoveredAssetMaxDistance;

        float fogDrivenDistance = Math.Clamp(_lastHoverPickFogEnd * 0.4f, 533.33f, MaxWorldObjectViewDistance);
        return Math.Min(_hoveredAssetMaxDistance, fogDrivenDistance);
    }

    internal bool IsHoverPickDistanceAllowed(float distance)
    {
        if (!_limitHoveredAssetRange)
            return true;

        return new WorldSceneSelectionPolicy(true, ComputeEffectiveHoveredAssetMaxDistance(), Vector3.Zero).IsDistanceAllowed(distance);
    }

    internal bool IsHoverPickPositionAllowed(Vector3 worldPosition)
    {
        if (!_limitHoveredAssetRange)
            return true;

        Vector3 cameraPosition = _pm4Overlay.GetPm4LoadAnchorCameraPosition();
        return new WorldSceneSelectionPolicy(true, ComputeEffectiveHoveredAssetMaxDistance(), cameraPosition).IsPositionAllowed(worldPosition);
    }

    private static bool TryRayIntersectInstanceBounds(Vector3 origin, Vector3 dir, in ObjectInstance instance, Vector3 padding, out float distance)
    {
        if (instance.BoundsResolved
            && Matrix4x4.Invert(instance.Transform, out Matrix4x4 inverseTransform))
        {
            Vector3 localOrigin = Vector3.Transform(origin, inverseTransform);
            Vector3 localDirection = Vector3.TransformNormal(dir, inverseTransform);

            // Pick against the tight geometry box where one is available. Using the culling bounds
            // here meant an M2's declared animation extent was the click target, so a nearby object
            // swallowed rays aimed past it — the reason objects close to the camera were hard to
            // inspect.
            bool useSelectionBounds = instance.SelectionBoundsResolved
                && AreFiniteOrderedBounds(instance.SelectionLocalBoundsMin, instance.SelectionLocalBoundsMax);
            Vector3 pickMin = useSelectionBounds ? instance.SelectionLocalBoundsMin : instance.LocalBoundsMin;
            Vector3 pickMax = useSelectionBounds ? instance.SelectionLocalBoundsMax : instance.LocalBoundsMax;

            if (localDirection.LengthSquared() > 1e-10f)
            {
                float localT = RayAABBIntersect(
                    localOrigin,
                    localDirection,
                    pickMin - padding,
                    pickMax + padding);

                if (localT >= 0f)
                {
                    Vector3 localHit = localOrigin + (localDirection * localT);
                    Vector3 worldHit = Vector3.Transform(localHit, instance.Transform);
                    distance = Vector3.Distance(origin, worldHit);
                    return true;
                }
            }
        }

        distance = RayAABBIntersect(origin, dir, instance.BoundsMin - padding, instance.BoundsMax + padding);
        return distance >= 0f;
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
        return TryMeasureHoverBrushHit(inst.BoundsMin, inst.BoundsMax, view, proj, mouseViewportX, mouseViewportY, viewportWidth, viewportHeight, out _, out _);
    }

    internal static bool TryMeasureHoverInfoHit(Vector3 boundsMin, Vector3 boundsMax,
        Matrix4x4 view, Matrix4x4 proj, float mouseViewportX, float mouseViewportY,
        float viewportWidth, float viewportHeight, out float distanceSq, out float depth)
    {
        return TryMeasureScreenBrushHit(
            boundsMin,
            boundsMax,
            view,
            proj,
            mouseViewportX,
            mouseViewportY,
            viewportWidth,
            viewportHeight,
            HoverInfoBrushPixels,
            HoverInfoMaxScreenRadius,
            out distanceSq,
            out depth);
    }

    private static bool TryMeasureHoverBrushHit(Vector3 boundsMin, Vector3 boundsMax,
        Matrix4x4 view, Matrix4x4 proj, float mouseViewportX, float mouseViewportY,
        float viewportWidth, float viewportHeight, out float distanceSq, out float depth)
    {
        return TryMeasureScreenBrushHit(
            boundsMin,
            boundsMax,
            view,
            proj,
            mouseViewportX,
            mouseViewportY,
            viewportWidth,
            viewportHeight,
            WireframeRevealBrushPixels,
            WireframeRevealMaxScreenRadius,
            out distanceSq,
            out depth);
    }

    private static bool TryMeasureScreenBrushHit(Vector3 boundsMin, Vector3 boundsMax,
        Matrix4x4 view, Matrix4x4 proj, float mouseViewportX, float mouseViewportY,
        float viewportWidth, float viewportHeight, float brushPixels, float maxScreenRadius,
        out float distanceSq, out float depth)
    {
        Vector3 center = (boundsMin + boundsMax) * 0.5f;
        if (!TryProjectToViewport(center, view, proj, viewportWidth, viewportHeight, out float sx, out float sy, out depth))
        {
            distanceSq = 0f;
            return false;
        }

        float dx = sx - mouseViewportX;
        float dy = sy - mouseViewportY;
        distanceSq = dx * dx + dy * dy;

        float worldRadius = MathF.Max((boundsMax - boundsMin).Length() * 0.5f, 4f);
        float projectedRadius = EstimateProjectedRadius(worldRadius, depth, proj, viewportHeight);
        float revealRadius = MathF.Min(brushPixels + projectedRadius, maxScreenRadius);
        return distanceSq <= revealRadius * revealRadius;
    }

    private static HoveredAssetInfo BuildHoveredObjectInfo(string assetKind, in ObjectInstance inst, ObjectType objectType, int objectIndex)
    {
        return new HoveredAssetInfo(
            assetKind,
            inst.ModelName,
            inst.ModelPath,
            $"UniqueId: {inst.UniqueId}",
            inst.PlacementPosition,
            0,
            null,
            objectType,
                objectIndex,
                null);
    }

    private static HoveredAssetInfo BuildHoveredWlLiquidInfo(WlLiquidBody body)
    {
        Vector3 worldPosition = (body.BoundsMin + body.BoundsMax) * 0.5f;
        return new HoveredAssetInfo(
            "WL liquid",
            body.Name,
            body.SourcePath,
            $"{body.FileType} • {body.GroupLabel} • {body.BlockCount} blocks • Z {body.MinHeight:F1}..{body.MaxHeight:F1}",
            worldPosition,
            0,
                null,
                ObjectType.None,
                -1,
                body.BodyKey);
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

    internal (int PreparedPrimitiveCount, int SubmittedPrimitiveCount) RenderWireframeReveal(Matrix4x4 view, Matrix4x4 proj, Vector3 cameraPos,
        Vector3 fogColor, float fogStart, float fogEnd, TerrainLighting lighting)
    {
        if (_wireframeRevealWmoIndices.Count == 0 && _wireframeRevealMdxIndices.Count == 0)
            return (0, 0);

        int preparedPrimitiveCount = _wireframeRevealWmoIndices.Count + _wireframeRevealMdxIndices.Count;
        int submittedPrimitiveCount = 0;

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
            submittedPrimitiveCount++;
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
            submittedPrimitiveCount++;
        }

        _gl.DepthMask(true);
        _gl.DepthFunc(DepthFunction.Lequal);
        return (preparedPrimitiveCount, submittedPrimitiveCount);
    }

    internal (int PreparedPrimitiveCount, int SubmittedPrimitiveCount) RenderVisibleObjectWireframeOverlay(WorldRenderFrame frame, Matrix4x4 view, Matrix4x4 proj,
        Vector3 cameraPos, Vector3 fogColor, float fogStart, float fogEnd, TerrainLighting lighting)
    {
        if (frame.Visibility.VisibleWmos.Count == 0 && frame.Visibility.VisibleMdx.Count == 0)
            return (0, 0);

        int preparedPrimitiveCount = frame.Visibility.VisibleWmos.Count + frame.Visibility.VisibleMdx.Count;
        int submittedPrimitiveCount = 0;

        _gl.Enable(EnableCap.DepthTest);
        _gl.DepthFunc(DepthFunction.Lequal);
        _gl.DepthMask(false);
        _gl.Disable(EnableCap.Blend);

        foreach (VisibleWmoInstance visible in frame.Visibility.VisibleWmos)
        {
            WmoRenderer? renderer = ResolveVisibleWmoRenderer(frame, visible.Instance.ModelKey);
            if (renderer == null)
                continue;

            renderer.RenderWireframeOverlay(visible.Instance.Transform, view, proj,
                fogColor, fogStart, fogEnd, cameraPos,
                lighting.LightDirection, lighting.LightColor, lighting.AmbientColor);
            submittedPrimitiveCount++;
        }

        foreach (VisibleMdxInstance visible in frame.Visibility.VisibleMdx)
        {
            IModelRenderer? renderer = ResolveVisibleMdxRenderer(frame, visible.Instance.ModelKey);
            if (renderer == null)
                continue;

            renderer.RenderWireframeOverlay(visible.Instance.Transform, view, proj,
                fogColor, fogStart, fogEnd, cameraPos,
                lighting.LightDirection, lighting.LightColor, lighting.AmbientColor);
            submittedPrimitiveCount++;
        }

        _gl.DepthMask(true);
        _gl.DepthFunc(DepthFunction.Lequal);
        return (preparedPrimitiveCount, submittedPrimitiveCount);
    }
}
