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

namespace WoWViewer.Terrain;


/// <summary>
/// Combines terrain (WDT/ADT), WMO placements (MODF), and MDX placements (MDDF)
/// into a single world scene — the same way the game client renders a map.
/// 
/// Uses <see cref="WorldAssetManager"/> to ensure each model is loaded exactly once.
/// Instances are lightweight structs holding only a model key + transform.
/// </summary>
public readonly record struct TaxiActorPose(
    int RouteId,
    Vector3 Position,
    Vector3 Forward,
    float YawRadians,
    float Scale,
    string ModelPath);

public class WorldScene : ISceneRenderer, IPm4OverlayHost
{
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

    private readonly record struct SelectedSceneObjectKey(
        ObjectType ObjectType,
        int UniqueId,
        int PlacementEntryIndex,
        int TileX,
        int TileY,
        bool HasTileCoordinate,
        string ModelKey,
        Vector3 PlacementPosition);

    public static IReadOnlyList<string> DefaultTaxiActorModelPaths => TaxiActorDefaultModelCandidates;

    private static float DecodeRawMprlPackedAngleRadians(MprlEntry positionRef)
    {
        return positionRef.Unk04 * (2f * MathF.PI / 65536f);
    }

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
    // These buckets mirror the graph's tile/chunk partition, but remain a flat collector aid.
    // They reject whole candidate lists; the existing per-instance collector stays authoritative.
    private readonly Dictionary<(int, int), List<FlatVisibilityBucket>> _tileMdxVisibilityBuckets = new();
    private readonly Dictionary<(int, int), List<FlatVisibilityBucket>> _tileWmoVisibilityBuckets = new();
    private readonly Dictionary<(int, int), (Vector3 Min, Vector3 Max)> _tileMdxBounds = new();
    private readonly Dictionary<(int, int), (Vector3 Min, Vector3 Max)> _tileWmoBounds = new();
    private readonly List<ObjectInstance> _externalMdxInstances = new();
    private readonly List<ObjectInstance> _externalSkyboxInstances = new();
    private readonly List<ObjectInstance> _externalWmoInstances = new();
    private readonly List<ObjectInstance> _taxiActorInstances = new();
    private WorldSceneGraphBuildSet? _sceneGraphBuild;
    private readonly Dictionary<string, WorldScenePortalAdapterResult> _sceneGraphPortalAdapters = new(StringComparer.Ordinal);
    private readonly Dictionary<string, WorldScenePortalVisibilityResult> _sceneGraphPortalVisibility = new(StringComparer.Ordinal);
    private readonly List<ObjectInstance> _sceneGraphVisibleMdxInstances = new();
    private readonly List<ObjectInstance> _sceneGraphVisibleWmoInstances = new();
    private bool _sceneGraphFrameVisibilityPrepared;
    private WorldSceneTraversalDiagnostics _lastSceneGraphTraversalDiagnostics = new();
    // The hierarchical graph remains available as an explicit investigation path, but it is
    // not yet a proven replacement for the production flat visibility collectors. Real Azeroth
    // captures measured tens of milliseconds in graph traversal per object pass, so keep the
    // stable legacy path as the runtime default until the graph has a bounded cost budget.
    private bool _useHierarchicalSceneTraversal;
    private bool _instancesDirty = false;
    private readonly Dictionary<string, float> _pendingVisibleMdxLoadDistances = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, float> _pendingVisibleWmoLoadDistances = new(StringComparer.OrdinalIgnoreCase);
    private readonly List<KeyValuePair<string, float>> _pendingVisibleMdxLoadScratch = new();
    private readonly List<KeyValuePair<string, float>> _pendingVisibleWmoLoadScratch = new();

    private sealed class FlatVisibilityBucket
    {
        public List<ObjectInstance> Instances { get; } = new();
        public Vector3 Min { get; private set; } = new(float.MaxValue);
        public Vector3 Max { get; private set; } = new(float.MinValue);
        public bool BoundsKnown { get; private set; } = true;

        public void Add(in ObjectInstance instance)
        {
            Instances.Add(instance);
            if (!instance.BoundsResolved
                || !AreFiniteOrderedBounds(instance.BoundsMin, instance.BoundsMax))
            {
                BoundsKnown = false;
                return;
            }

            Min = Vector3.Min(Min, instance.BoundsMin);
            Max = Vector3.Max(Max, instance.BoundsMax);
        }
    }

    private bool _objectsVisible = true;
    private bool _wmosVisible = true;
    private bool _doodadsVisible = true;
    private bool _objectFogEnabled = true;
    private bool _objectPathFiltersEnabled = true;
    private bool _limitHoveredAssetRange = true;
    private bool _useDynamicHoveredAssetRange = false;
    private bool _showSelectedObjectBounds = true;
    private float _hoveredAssetMaxDistance = 533.33f;
    private float _lastHoverPickFogEnd = 1500f;
    private float _objectStreamingRangeMultiplier = 0.5f;
    private float _maxVisibleMdxBoundsHeight;
    private bool _hideTerrainOccludedMdx;
    private WorldObjectVisibilityProfile _objectVisibilityProfile = WorldObjectVisibilityProfile.Performance;

    // Frustum culling
    private readonly FrustumCuller _frustumCuller = new();
    private const float DoodadCullDistance = 16000f; // Hard ceiling for very small doodads when fog allows farther visibility
    private const float DoodadCullDistanceSq = DoodadCullDistance * DoodadCullDistance;
    private const float DoodadSmallThreshold = 10f; // AABB diagonal below this = "small" (relaxed — only cull tiny objects)
    private const float FadeStartFraction = 0.80f;  // Fade begins at 80% of cull distance
    private const float WmoCullDistance = 1600f;     // Default world-object visibility should stay close to terrain fog unless explicitly widened
    internal const float NoCullRadius = 512f;         // Objects within this radius are never frustum-culled
    private const float ObjectNearHoldRadius = 384f;
    private const float ObjectNearHoldRadiusSq = ObjectNearHoldRadius * ObjectNearHoldRadius;
    private const float VisionConeFrontDot = 0.15f;
    private const float VisionConeRearDot = -0.35f;
    private const float RearConeCullFraction = 0.45f;
    private const float MinOffFrustumConeFactor = 0.35f;
    private const float RearConeFadeFloor = 0.25f;
    private const float RearConeLoadPenalty = 2.5f;
    private const float HoverInfoBrushPixels = 32f;
    private const float HoverInfoMaxScreenRadius = 96f;
    private const float WireframeRevealBrushPixels = 96f;
    private const float WireframeRevealMaxScreenRadius = 220f;
    private const float MaxWorldObjectViewDistance = 20000f;
    private const float MaxWorldObjectViewDistanceSq = MaxWorldObjectViewDistance * MaxWorldObjectViewDistance;

    private readonly record struct TerrainAssetLoadPolicy(
        bool PrewarmTileAssets,
        int MaxNewMdxLoadsPerFrame,
        int MaxNewWmoLoadsPerFrame,
        int MaxDeferredLoadsPerFrame,
        double MaxDeferredLoadBudgetMs,
        int MaxPriorityLoadBacklog);

    private static readonly TerrainAssetLoadPolicy WmoOnlyAssetLoadPolicy = new(
        PrewarmTileAssets: false,
        MaxNewMdxLoadsPerFrame: 6,
        MaxNewWmoLoadsPerFrame: 3,
        MaxDeferredLoadsPerFrame: 2,
        MaxDeferredLoadBudgetMs: 6.0,
        MaxPriorityLoadBacklog: 8);

    private static readonly TerrainAssetLoadPolicy StreamingTerrainAssetLoadPolicy = new(
        PrewarmTileAssets: false,
        MaxNewMdxLoadsPerFrame: 12,
        MaxNewWmoLoadsPerFrame: 6,
        MaxDeferredLoadsPerFrame: 4,
        MaxDeferredLoadBudgetMs: 3.5,
        MaxPriorityLoadBacklog: 16);

    private sealed class WorldRenderFrame
    {
        public WorldVisibilityFrame Visibility { get; } = new();
        public WorldObjectPassFrame ObjectPasses { get; } = new();
        public Dictionary<string, WmoRenderer> VisibleWmoRendererCache { get; } = new(StringComparer.OrdinalIgnoreCase);
        public Dictionary<string, IModelRenderer> VisibleMdxRendererCache { get; } = new(StringComparer.OrdinalIgnoreCase);

        /// <summary>
        /// Applied render path per visible model key, resolved once per frame from the asset
        /// manager's route decisions. Spec 201 FR-002: attribution uses the applied route, because
        /// a model whose primary route failed and fell back drew on the fallback.
        /// </summary>
        public Dictionary<string, WorldModelRenderPath> VisibleMdxRenderPathCache { get; } = new(StringComparer.OrdinalIgnoreCase);

        // Opaque-pass scratch, reused across frames and cleared in place. These were allocated fresh
        // every frame inside the batching pass — the pass that exists to reduce work. Named scratch,
        // not cache: they are rebuilt every frame by design.
        public List<WorldObjectPassCoordinator.WorldWmoOpaqueBatchCandidate> WmoBatchCandidateScratch { get; } = [];
        public Dictionary<IModelRenderer, List<Matrix4x4>> WmoDoodadBatchGroupScratch { get; } = [];
        public List<WmoOpaqueDoodadBatchItem> WmoDoodadUnbatchedScratch { get; } = [];
        public HashSet<IModelRenderer> UpdatedRendererScratch { get; } = [];
        public HashSet<IGpuInstancedModelRenderer> GpuBatchRendererScratch { get; } = [];
        public HashSet<IModelRenderer> ImmediateBatchRendererScratch { get; } = [];

        // Distinct model keys submitted per pass. Per-model instancing cannot collapse opaque
        // draws below this count, so it is the floor the instanced counter converges on
        // (spec 202 research R2).
        public HashSet<string> OpaqueSubmittedModelKeyScratch { get; } = new(StringComparer.OrdinalIgnoreCase);
        public HashSet<string> TransparentSubmittedModelKeyScratch { get; } = new(StringComparer.OrdinalIgnoreCase);
        public List<(bool IsWmo, int Index, float DistanceSq)> TransparentSortScratch { get; } = [];
        public List<VisibleWmoInstance> WmoInstanceBatchScratch { get; } = [];

        // The batch-group values are themselves per-frame lists; pool them so clearing the group
        // dictionary does not drop a list per renderer per frame onto the heap.
        private readonly Stack<List<Matrix4x4>> _matrixListPool = new();

        public List<Matrix4x4> RentMatrixList()
            => _matrixListPool.Count > 0 ? _matrixListPool.Pop() : [];

        private void ReturnMatrixLists()
        {
            foreach (List<Matrix4x4> list in WmoDoodadBatchGroupScratch.Values)
            {
                list.Clear();
                _matrixListPool.Push(list);
            }

            WmoDoodadBatchGroupScratch.Clear();
        }

        private void ResetOpaquePassScratch()
        {
            WmoBatchCandidateScratch.Clear();
            ReturnMatrixLists();
            WmoDoodadUnbatchedScratch.Clear();
            UpdatedRendererScratch.Clear();
            GpuBatchRendererScratch.Clear();
            ImmediateBatchRendererScratch.Clear();
            OpaqueSubmittedModelKeyScratch.Clear();
            TransparentSubmittedModelKeyScratch.Clear();
            TransparentSortScratch.Clear();
            WmoInstanceBatchScratch.Clear();
        }

        public List<VisibleWmoInstance> VisibleWmoInstances => Visibility.VisibleWmos;
        public List<VisibleMdxInstance> VisibleMdxInstances => Visibility.VisibleMdx;
        public int VisibleTaxiMdxCount
        {
            get => Visibility.VisibleTaxiMdxCount;
            set => Visibility.VisibleTaxiMdxCount = value;
        }

        public int OpaqueBatchedMdxCount { get; set; }
        public int OpaqueUnbatchedMdxCount { get; set; }
        public int TransparentBatchedMdxCount { get; set; }
        public int TransparentUnbatchedMdxCount { get; set; }

        /// <summary>
        /// Specs 201 and 202 Phase 0. Decomposes the four aggregate counters above by render path
        /// and by the gate that stopped each instance short of GPU instancing, and carries the
        /// draw calls the pass actually issued. The aggregates are kept so the decomposition can
        /// be proved to sum to them (spec 201 FR-005).
        /// </summary>
        public WorldModelSubmissionTally OpaqueModelSubmission;
        public WorldModelSubmissionTally TransparentModelSubmission;
        public int WmoDrawCallCount { get; set; }
        public int WmoBatchDrawCallCount { get; set; }
        public int WmoOpaqueBatchInstanceCount { get; set; }
        public int WmoGroupFallbackDrawCallCount { get; set; }
        public int WmoLiquidDrawCallCount { get; set; }
        public int WmoDoodadSubmissionCount { get; set; }
        public int WmoVisibleGroupSubmissionCount { get; set; }

        /// <summary>
        /// Spec 151 admission accounting: which rule admitted each WMO placement and each group.
        /// Accumulated across every placement and submission pass in the frame.
        /// </summary>
        public WmoAdmissionTally WmoAdmission;

        public double DeferredAssetLoadMs { get; set; }
        public double TaxiActorUpdateMs { get; set; }
        public double LightingMs { get; set; }
        public double SkyMs { get; set; }
        public double SkyboxBackdropMs { get; set; }
        public double WdlMs { get; set; }
        public double TerrainMs { get; set; }
        public double WmoVisibilityMs { get; set; }
        public double WmoSubmissionMs { get; set; }
        public double WmoTransparentSubmissionMs { get; set; }
        public double MdxAnimationMs { get; set; }
        public double MdxVisibilityMs { get; set; }
        public double MdxOpaqueSubmissionMs { get; set; }
        public double LiquidMs { get; set; }
        public double MdxTransparentSortMs { get; set; }
        public double MdxTransparentSubmissionMs { get; set; }
        public double OverlayMs { get; set; }
        public double SceneMaintenanceMs { get; set; }
        public double PrepareObjectPhaseMs { get; set; }
        public List<WorldOverlayOwnerFrameStats> OverlayOwners { get; } = new(WorldOverlayOwners.All.Count);

        public void Reset()
        {
            Visibility.Reset();
            ObjectPasses.Reset();
            VisibleWmoRendererCache.Clear();
            VisibleMdxRendererCache.Clear();
            VisibleMdxRenderPathCache.Clear();
            ResetOpaquePassScratch();
            OpaqueModelSubmission.Reset();
            TransparentModelSubmission.Reset();
            OpaqueBatchedMdxCount = 0;
            OpaqueUnbatchedMdxCount = 0;
            TransparentBatchedMdxCount = 0;
            TransparentUnbatchedMdxCount = 0;
            WmoDrawCallCount = 0;
            WmoBatchDrawCallCount = 0;
            WmoOpaqueBatchInstanceCount = 0;
            WmoGroupFallbackDrawCallCount = 0;
            WmoLiquidDrawCallCount = 0;
            WmoDoodadSubmissionCount = 0;
            WmoVisibleGroupSubmissionCount = 0;
            WmoAdmission.Reset();
            DeferredAssetLoadMs = 0;
            TaxiActorUpdateMs = 0;
            LightingMs = 0;
            SkyMs = 0;
            SkyboxBackdropMs = 0;
            WdlMs = 0;
            TerrainMs = 0;
            WmoVisibilityMs = 0;
            WmoSubmissionMs = 0;
            WmoTransparentSubmissionMs = 0;
            MdxAnimationMs = 0;
            MdxVisibilityMs = 0;
            MdxOpaqueSubmissionMs = 0;
            LiquidMs = 0;
            MdxTransparentSortMs = 0;
            MdxTransparentSubmissionMs = 0;
            OverlayMs = 0;
            SceneMaintenanceMs = 0;
            PrepareObjectPhaseMs = 0;
            OverlayOwners.Clear();
            foreach (string ownerId in WorldOverlayOwners.All)
                OverlayOwners.Add(WorldOverlayOwnerFrameStats.Disabled(ownerId));
        }

        public void SetOverlayOwner(
            string ownerId,
            double durationMs,
            bool enabled,
            int preparedPrimitiveCount = 0,
            int submittedPrimitiveCount = 0,
            string cacheStatus = "not_cached",
            int deferredCount = 0)
        {
            WorldOverlayOwnerFrameStats stats = new(
                ownerId,
                Math.Max(0, durationMs),
                enabled,
                Math.Max(0, preparedPrimitiveCount),
                Math.Max(0, submittedPrimitiveCount),
                cacheStatus,
                Math.Max(0, deferredCount));

            for (int i = 0; i < OverlayOwners.Count; i++)
            {
                if (string.Equals(OverlayOwners[i].OwnerId, ownerId, StringComparison.Ordinal))
                {
                    OverlayOwners[i] = stats;
                    return;
                }
            }

            throw new InvalidOperationException($"Unknown world overlay owner '{ownerId}'.");
        }

        public double OverlayOwnerDurationSum => OverlayOwners.Sum(static owner => owner.DurationMs);

        public WorldRenderFrameStats ToStats(
            double totalCpuMs,
            int pendingAssetLoadCount,
            int terrainChunksRendered,
            int terrainChunksCulled,
            int wdlVisibleTileCount,
            int wdlHiddenTileCount)
        {
            int visibleMdxCount = Math.Max(0, VisibleMdxInstances.Count - VisibleTaxiMdxCount);
            return new WorldRenderFrameStats(
                totalCpuMs,
                pendingAssetLoadCount,
                terrainChunksRendered,
                terrainChunksCulled,
                wdlVisibleTileCount,
                wdlHiddenTileCount,
                VisibleWmoInstances.Count,
                visibleMdxCount,
                VisibleTaxiMdxCount,
                OpaqueBatchedMdxCount,
                OpaqueUnbatchedMdxCount,
                TransparentBatchedMdxCount,
                TransparentUnbatchedMdxCount,
                WmoDrawCallCount,
                WmoBatchDrawCallCount,
                WmoOpaqueBatchInstanceCount,
                WmoGroupFallbackDrawCallCount,
                WmoLiquidDrawCallCount,
                WmoDoodadSubmissionCount,
                WmoVisibleGroupSubmissionCount,
                new WorldRenderStageStats(DeferredAssetLoadMs),
                new WorldRenderStageStats(TaxiActorUpdateMs),
                new WorldRenderStageStats(LightingMs),
                new WorldRenderStageStats(SkyMs),
                new WorldRenderStageStats(SkyboxBackdropMs),
                new WorldRenderStageStats(WdlMs, wdlVisibleTileCount),
                new WorldRenderStageStats(TerrainMs, terrainChunksRendered),
                new WorldRenderStageStats(WmoVisibilityMs, VisibleWmoInstances.Count),
                new WorldRenderStageStats(WmoSubmissionMs, VisibleWmoInstances.Count, VisibleWmoInstances.Count),
                new WorldRenderStageStats(WmoTransparentSubmissionMs, VisibleWmoInstances.Count, WmoDrawCallCount),
                new WorldRenderStageStats(MdxAnimationMs),
                new WorldRenderStageStats(MdxVisibilityMs, VisibleMdxInstances.Count),
                new WorldRenderStageStats(MdxOpaqueSubmissionMs, VisibleMdxInstances.Count, OpaqueBatchedMdxCount + OpaqueUnbatchedMdxCount),
                new WorldRenderStageStats(LiquidMs),
                new WorldRenderStageStats(MdxTransparentSortMs, ObjectPasses.TransparentVisibleMdxRoutes.Count),
                new WorldRenderStageStats(MdxTransparentSubmissionMs, ObjectPasses.TransparentVisibleMdxRoutes.Count, TransparentBatchedMdxCount + TransparentUnbatchedMdxCount),
                new WorldRenderStageStats(OverlayMs),
                new WorldRenderStageStats(SceneMaintenanceMs),
                new WorldRenderStageStats(PrepareObjectPhaseMs))
            {
                OverlayOwners = OverlayOwners.ToArray(),
                WmoAdmission = WmoAdmission.ToStats(),
                OpaqueModelSubmission = OpaqueModelSubmission.ToStats(),
                TransparentModelSubmission = TransparentModelSubmission.ToStats(),
            };
        }
    }

    // Scratch collections reused every frame to avoid hot-path allocations.
    private readonly WorldRenderFrame _renderFrame = new();
    private readonly SceneLightManager _sceneLightManager = new();
    private readonly List<SceneLight> _sceneLightCollectScratch = new();
    private readonly HashSet<WmoRenderer> _worldFrameWmoRenderers = new();
    private readonly List<int> _wireframeRevealWmoIndices = new();
    private readonly List<int> _wireframeRevealMdxIndices = new();
    private readonly MinimapRenderer? _minimapRenderer;
    private HoveredAssetInfo? _hoveredAssetInfo;
    private TerrainAssetLoadPolicy _assetLoadPolicy = StreamingTerrainAssetLoadPolicy;
    private bool _wireframeRevealEnabled;
    private bool _showHoveredAssetTooltips = true;
    private const int UniqueIdLayerGapThreshold = 100;
    private bool _uniqueIdFilterEnabled;
    private UniqueIdVisibilityScope _uniqueIdVisibilityScope = UniqueIdVisibilityScope.PerMap;
    private int _uniqueIdFilterMin = -1;
    private int _uniqueIdFilterMax = -1;
    private (int tileX, int tileY)? _uniqueIdFilterTile;
    private readonly List<ObjectPathFilterEntry> _objectPathFilters = new();

    // PM4 debug overlay (Epic 251 U-01 E1): state and behaviour live in Pm4OverlayScene.
    private readonly Pm4OverlayScene _pm4Overlay;
    public Pm4OverlayScene Pm4Overlay => _pm4Overlay;
    IDataSource? IPm4OverlayHost.DataSource => _dataSource;
    TerrainManager IPm4OverlayHost.TerrainManager => _terrainManager;
    WorldAssetManager IPm4OverlayHost.Assets => _assets;
    bool IPm4OverlayHost.InstancesDirty => _instancesDirty;
    void IPm4OverlayHost.RebuildInstanceLists() => RebuildInstanceLists();
    Dictionary<(int, int), List<ObjectInstance>> IPm4OverlayHost.TileWmoInstances => _tileWmoInstances;
    Dictionary<(int, int), List<ObjectInstance>> IPm4OverlayHost.TileMdxInstances => _tileMdxInstances;
    List<ObjectInstance> IPm4OverlayHost.WmoInstances => _wmoInstances;
    bool IPm4OverlayHost.HasLastRenderedCameraPosition => _hasLastRenderedCameraPosition;
    Vector3 IPm4OverlayHost.LastRenderedCameraPosition => _lastRenderedCameraPosition;
    bool IPm4OverlayHost.IsHoverPickDistanceAllowed(float distance) => IsHoverPickDistanceAllowed(distance);
    bool IPm4OverlayHost.IsHoverPickPositionAllowed(Vector3 worldPosition) => IsHoverPickPositionAllowed(worldPosition);
    FrustumCuller IPm4OverlayHost.FrustumCuller => _frustumCuller;

    private Vector3 _lastRenderedCameraPosition;
    private bool _hasLastRenderedCameraPosition;

    // Culling stats (updated each frame)
    public int WmoRenderedCount { get; private set; }
    public int WmoCulledCount { get; private set; }
    public int MdxRenderedCount { get; private set; }
    public int MdxCulledCount { get; private set; }
    public int LastUnloadedWmoTileX { get; private set; } = -1;
    public int LastUnloadedWmoTileY { get; private set; } = -1;
    public int LastUnloadedWmoInstanceCount { get; private set; }
    public int WmoTileUnloadEventCount { get; private set; }
    public WorldRenderFrameStats LastRenderFrameStats { get; private set; } = WorldRenderFrameStats.Empty;

    /// <summary>
    /// Rolling per-frame timing history. <see cref="LastRenderFrameStats"/> holds a single frame,
    /// which cannot show a periodic hitch; this retains a bounded window so behavior is observable
    /// over time. Recording is always on and allocation-free.
    /// </summary>
    public WorldRenderFrameHistory FrameHistory { get; } = new();

    /// <summary>
    /// Detector-power check: stall one upcoming frame by a known amount. If the history does not
    /// flag that frame with that magnitude, the detector is not trusted and no renderer measurement
    /// derived from it is valid. Diagnostics only; never set during normal use.
    /// </summary>
    public double DebugInjectStallMs { get; set; }

    /// <summary>
    /// Opt-in per-kind scene-graph attribution. Off by default because collecting it requires
    /// recursively walking every rejected subtree — the work culling exists to avoid — on every
    /// frame. Enable it only while a diagnostics surface is actually reading the per-kind
    /// breakdown.
    /// </summary>
    public bool SceneGraphDetailedDiagnosticsEnabled { get; set; }

    /// <summary>
    /// Spec 153 US3. Routes opaque MDX that declare themselves batchable through the shared
    /// begin-once/submit-many path instead of a full per-instance state setup per draw.
    /// <para>
    /// Kept as a runtime switch so a before/after capture can be taken on one flight without a
    /// rebuild, and so the fix is revertible in place if the measured effect does not clear the
    /// noise floor (FR-010, FR-011). Turning it off restores the previous 100%-unbatched behaviour
    /// exactly.
    /// </para>
    /// </summary>
    public bool MdxOpaqueBatchingEnabled { get; set; } = true;

    /// <summary>
    /// Whether world doodads (ADT/MDDF placements) advance their animators each frame. WMO-internal
    /// doodads are driven separately by <c>WmoRenderer.UpdateDoodadAnimations</c> and are unaffected.
    /// </summary>
    /// <remarks>
    /// Default off, per the operator's rule that only WMO doodads should auto-animate in the world
    /// renderer. Kept as a toggle rather than a deletion because it is also the measurement: the
    /// MdxAnimation stage timer with this on versus off is the size of the world-doodad animation
    /// bill, which was previously an estimate.
    /// <para>
    /// Note this does **not** stop bone matrices being uploaded. A model that owns an animator still
    /// reports <c>ShouldUploadBoneMatrices</c>, so its pose is still sent — it simply stops changing.
    /// Skipping the upload as well would render the bind pose, which is usually but not always the
    /// same thing, and that is a separate decision from whether the animation advances.
    /// </para>
    /// </remarks>
    public bool WorldDoodadAnimationEnabled { get; set; } = true;

    private Vector3 _frameHistoryPreviousCameraPosition = new(float.NaN, float.NaN, float.NaN);
    private Vector3 _frameHistoryPreviousCameraForward = new(float.NaN, float.NaN, float.NaN);

    /// <summary>
    /// Localises unaccounted frame time — the cost inside <see cref="Render"/> that no stage timer
    /// covers. Measured directly rather than derived, and split three ways so a hitch points at a
    /// region instead of at "somewhere". Peaks are retained because hitches are transient and a
    /// live value will almost never be sampled on the bad frame.
    /// </summary>
    public double RenderPrologueMs { get; private set; }
    public double RenderPassGapMs { get; private set; }
    public double RenderEpilogueMs { get; private set; }
    public double RenderProloguePeakMs { get; private set; }
    public double RenderPassGapPeakMs { get; private set; }
    public double RenderEpiloguePeakMs { get; private set; }

    /// <summary>
    /// PrepareObjectPhase now has its own stage timer (Spec 153 FR-001), so its cost no longer hides
    /// in the pass gap. <see cref="ObjectPhasePrepareMs"/> is that same measurement; the sub-probes
    /// below attribute its internals. Peaks are retained because the suspected work (PM4 overlay
    /// window changes, audio residency) is periodic and a live reading almost never lands on it.
    /// </summary>
    public double ObjectPhasePrepareMs { get; private set; }
    public double ObjectPhasePreparePeakMs { get; private set; }
    public double AudioRuntimeUpdateMs { get; private set; }
    public double AudioRuntimeUpdatePeakMs { get; private set; }
    public double Pm4OverlayWindowMs { get; private set; }
    public double Pm4OverlayWindowPeakMs { get; private set; }

    public void ResetRenderRegionPeaks()
    {
        RenderProloguePeakMs = 0;
        RenderPassGapPeakMs = 0;
        RenderEpiloguePeakMs = 0;
        ObjectPhasePreparePeakMs = 0;
        AudioRuntimeUpdatePeakMs = 0;
        Pm4OverlayWindowPeakMs = 0;
    }

    // Per-frame scratch, reused rather than reallocated. Named scratch, not cache: they are rebuilt
    // every frame by design, so calling them a cache would misrepresent their lifetime.
    private readonly List<WorldSceneGraphBuildResult> _activeSceneGraphScratch = [];
    private readonly HashSet<WorldSceneGraphBuildResult> _activeSceneGraphSetScratch =
        new(ReferenceEqualityComparer.Instance);
    private readonly List<WorldSceneNode> _sceneGraphVisibleNodeScratch = [];
    private readonly List<WorldSceneNode> _sceneGraphRejectedNodeScratch = [];
    private readonly WorldSceneTraversalDiagnostics _sceneGraphTraversalScratchDiagnostics = new();
    public string RendererOptimizationHint => WorldRenderOptimizationAdvisor.BuildHint(LastRenderFrameStats);
    public bool UseHierarchicalSceneTraversal
    {
        get => _useHierarchicalSceneTraversal;
        set
        {
            if (_useHierarchicalSceneTraversal == value)
                return;

            _useHierarchicalSceneTraversal = value;
            _sceneGraphBuild = null;
            _sceneGraphFrameVisibilityPrepared = false;
            _instancesDirty = true;
        }
    }
    public bool IsHierarchicalSceneTraversalActive => UseHierarchicalSceneTraversal && _sceneGraphBuild is not null;
    public int SceneGraphResidentAdtCount => _sceneGraphBuild?.AdtGraphs.Count ?? 0;
    public bool SceneGraphHasExternalRoot => _sceneGraphBuild?.ExternalGraph is not null;
    public WorldSceneGraphSnapshot? SceneGraphSnapshot => _sceneGraphBuild?.CreateSnapshot();
    public WorldSceneTraversalDiagnostics SceneGraphTraversalDiagnostics => _lastSceneGraphTraversalDiagnostics;
    public IReadOnlyDictionary<string, WorldScenePortalAdapterResult> SceneGraphPortalAdapters => _sceneGraphPortalAdapters;
    public IReadOnlyDictionary<string, WorldScenePortalVisibilityResult> SceneGraphPortalVisibility => _sceneGraphPortalVisibility;

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
    public bool ShowSky { get; set; } = true;

    // Bounding box debug rendering
    private bool _showBoundingBoxes = false;
    private BoundingBoxRenderer? _bbRenderer;
    public bool ShowBoundingBoxes { get => _showBoundingBoxes; set => _showBoundingBoxes = value; }

    // Object selection
    private ObjectType _selectedObjectType = ObjectType.None;
    private int _selectedObjectIndex = -1;
    private int _selectedWmoParentIndex = -1;
    private SelectedSceneObjectKey? _selectedSceneObjectKey;
    public ObjectType SelectedObjectType => _selectedObjectType;
    public int SelectedObjectIndex => _selectedObjectIndex;
    public int SelectedWmoParentIndex => _selectedWmoParentIndex;
    public bool WireframeRevealEnabled => _wireframeRevealEnabled;
    public bool TerrainWireframeEnabled => _terrainManager.IsWireframe;
    public bool ObjectWireframeEnabled => _assets.ObjectWireframeEnabled;
    public HoveredAssetInfo? HoveredAssetInfo => _hoveredAssetInfo;
    public bool ShowHoveredAssetTooltips { get => _showHoveredAssetTooltips; set => _showHoveredAssetTooltips = value; }
    public bool LimitHoveredAssetRange { get => _limitHoveredAssetRange; set => _limitHoveredAssetRange = value; }
    public bool UseDynamicHoveredAssetRange { get => _useDynamicHoveredAssetRange; set => _useDynamicHoveredAssetRange = value; }
    public int PendingAssetLoadCount => _assets.PendingAssetLoadCount;
    public int PendingDeferredWmoDoodadLoadCount => _assets.PendingDeferredWmoDoodadLoadCount;
    public int PendingDeferredWmoMaterialTextureLoadCount => _assets.PendingDeferredWmoMaterialTextureLoadCount;
    public int PendingWorldObjectLoadCount => PendingAssetLoadCount + PendingDeferredWmoDoodadLoadCount;
    public int PendingCapturePreloadLoadCount => PendingWorldObjectLoadCount + PendingDeferredWmoMaterialTextureLoadCount;
    private bool _capturePreloadActive;
    private readonly HashSet<(int tileX, int tileY)> _capturePreloadTiles = new();
    public bool CapturePreloadActive
    {
        get => _capturePreloadActive;
        set
        {
            _capturePreloadActive = value;
            if (!value)
                _capturePreloadTiles.Clear();
        }
    }
    public float ObjectStreamingRangeMultiplier
    {
        get => _objectStreamingRangeMultiplier;
        set => _objectStreamingRangeMultiplier = Math.Clamp(value, 0.25f, 4.0f);
    }
    public float MaxVisibleMdxBoundsHeight
    {
        get => _maxVisibleMdxBoundsHeight;
        set => _maxVisibleMdxBoundsHeight = value > 0f ? value : 0f;
    }
    public bool HideTerrainOccludedMdx
    {
        get => _hideTerrainOccludedMdx;
        set => _hideTerrainOccludedMdx = value;
    }
    public string? SecondaryOverlayMap
    {
        get => _terrainManager?.OverlayMapName;
        set => _terrainManager?.SetOverlayMap(value);
    }

    /// <summary>
    /// The phase overlay stack, for consumers that need more than the first enabled layer.
    /// </summary>
    /// <remarks>
    /// <see cref="SecondaryOverlayMap"/> is a single-overlay shim and reports only the first enabled
    /// layer, with no tile offset. Anything that renders overlay content -- the minimap included --
    /// has to read the stack, or it silently shows one layer at the wrong coordinates.
    /// </remarks>
    public IReadOnlyList<PhaseLayerSettings> PhaseLayers =>
        _terrainManager?.PhaseLayers as IReadOnlyList<PhaseLayerSettings>
        ?? (_terrainManager?.PhaseLayers?.ToList() ?? new List<PhaseLayerSettings>());

    /// <summary>Cartography (Spec 222): which layer row is expanded/selected on the minimap; -1 = none.</summary>
    public int SelectedPhaseLayerIndex { get; set; } = -1;

    /// <summary>
    /// Cartography (Spec 222): true when the named map is WMO-based (a dungeon/global-WMO map) —
    /// such layers carry no terrain tiles and must not paint minimap textures or claim tiles.
    /// Measured 2026-09-04: Shadowfang's leftover MAIN entries painted minimap fragments at wrong
    /// coordinates and drove 667 missing/failed terrain loads.
    /// </summary>
    public bool IsWmoBasedMap(string mapName)
        => _terrainManager != null && _terrainManager.IsMapWmoBased(mapName);

    /// <summary>Spec 231 Phase 7: whether the named layer donor map has terrain content at this tile (own-grid coordinates).</summary>
    public bool LayerHasTile(string mapName, int tileX, int tileY)
        => _terrainManager != null && _terrainManager.LayerTileExists(mapName, tileX, tileY);

    /// <summary>
    /// Spec 232 FR-11: the channels the BASE map contributes to its own tiles — the operator can
    /// drop the base map's liquids, shadows, objects, etc. per channel. Changing this re-streams.
    /// </summary>
    public PhaseDataChannel BaseChannelKeep
    {
        get => _terrainManager?.BaseChannelKeep ?? PhaseDataChannel.All;
        set
        {
            if (_terrainManager != null)
                _terrainManager.BaseChannelKeep = value;
        }
    }

    /// <summary>
    /// Cartography (Spec 222): each enabled, resolved layer with its donor footprint in DONOR tile
    /// coordinates. The minimap overlay applies the layer's offset to show where the content will
    /// land; an unoffset layer's footprint therefore appears at its true coordinates — which is
    /// exactly how a non-overlapping map (Shadowfang over Azeroth) stays visible instead of
    /// silently contributing nothing.
    /// </summary>
    public IReadOnlyList<(PhaseLayerSettings Layer, IReadOnlyList<(int TileX, int TileY)> Tiles)> GetLayerFootprints()
    {
        var result = new List<(PhaseLayerSettings, IReadOnlyList<(int, int)>)>();
        if (_terrainManager == null)
            return result;

        foreach (PhaseLayerSettings layer in _terrainManager.PhaseLayers)
        {
            if (!layer.Enabled || string.IsNullOrWhiteSpace(layer.MapName))
                continue;
            if (layer.Resolution != PhaseLayerResolution.Resolved)
                continue;
            if (_terrainManager.IsMapWmoBased(layer.MapName))
                continue;

            if (layer.UsePlacedTilesOnly)
            {
                IReadOnlyList<(int TileX, int TileY)> placedTargets = layer.TilePlacements
                    .Where(static placement => placement.IsValid)
                    .Select(static placement => (placement.TargetTileX, placement.TargetTileY))
                    .Distinct()
                    .ToList();
                result.Add((layer, placedTargets));
                continue;
            }

            IReadOnlyList<(int TileX, int TileY)> donorTiles = _terrainManager.GetLayerFootprint(layer);

            // Spec 231 Phase 7: compose the footprint through rotation/mirror + offset so
            // minimap rendering, click hit-tests, and drag logic all see where the layer's
            // tiles actually land on the base map. Targets outside the 64x64 grid are dropped
            // (operator rule: a layer can never extend past the map grid).
            var composedTiles = new List<(int TileX, int TileY)>(donorTiles.Count);
            foreach ((int donorTileX, int donorTileY) in donorTiles)
            {
                int tx = donorTileX;
                int ty = donorTileY;
                if (layer.RotationDegrees != 0f || layer.MirrorHorizontal || layer.MirrorVertical)
                {
                    (tx, ty) = WowViewer.Core.Maps.PhaseCompositionPolicy.ForwardTransformTile(donorTileX, donorTileY, layer);
                }
                tx += layer.TileOffsetX;
                ty += layer.TileOffsetY;
                if (tx < 0 || tx > 63 || ty < 0 || ty > 63)
                    continue;
                composedTiles.Add((tx, ty));
            }

            result.Add((layer, composedTiles));
        }

        return result;
    }
    public bool EnableRuntimeWmoGroupVisibility
    {
        get => _assets.EnableRuntimeWmoGroupVisibility;
        set => _assets.EnableRuntimeWmoGroupVisibility = value;
    }
    public bool EnableRuntimeWmoGroupLiquids
    {
        get => _assets.EnableRuntimeWmoGroupLiquids;
        set => _assets.EnableRuntimeWmoGroupLiquids = value;
    }
    public WorldObjectVisibilityProfile ObjectVisibilityProfile
    {
        get => _objectVisibilityProfile;
        set => _objectVisibilityProfile = value;
    }
    public bool ObjectsVisible { get => _objectsVisible; set => _objectsVisible = value; }
    public bool WmosVisible { get => _wmosVisible; set => _wmosVisible = value; }
    public bool DoodadsVisible { get => _doodadsVisible; set => _doodadsVisible = value; }
    public bool ObjectPathFiltersEnabled { get => _objectPathFiltersEnabled; set => _objectPathFiltersEnabled = value; }
    public IReadOnlyList<ObjectPathFilterEntry> ObjectPathFilters => _objectPathFilters;
    public bool ShowSelectedObjectBounds { get => _showSelectedObjectBounds; set => _showSelectedObjectBounds = value; }
    public float HoveredAssetMaxDistance
    {
        get => _hoveredAssetMaxDistance;
        set => _hoveredAssetMaxDistance = Math.Clamp(value, 10f, MaxWorldObjectViewDistance);
    }
    public float EffectiveHoveredAssetMaxDistance => ComputeEffectiveHoveredAssetMaxDistance();
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

    public void ApplyTextureSamplingSettings()
    {
        _terrainManager.Renderer.ApplyTextureSamplingSettings();
        _assets.ApplyTextureSamplingSettings();
    }

    /// <summary>Get the currently selected object instance, or null if nothing selected.</summary>
    public ObjectInstance? SelectedInstance => TryGetSelectedSceneInstance(out ObjectInstance instance) ? instance : null;

    public bool TryGetSelectedPlacementSourceData(out string sourcePath, out byte[] sourceBytes)
    {
        sourcePath = string.Empty;
        sourceBytes = Array.Empty<byte>();

        ObjectInstance? selected = SelectedInstance;
        if (!selected.HasValue)
            return false;

        ObjectInstance instance = selected.Value;
        if (!instance.HasTileCoordinate || instance.PlacementEntryIndex < 0)
            return false;

        return _terrainManager.Adapter.TryGetPlacementSourceData(instance.TileX, instance.TileY, out sourcePath, out sourceBytes);
    }

    public bool TryGetSelectedPlacementWritablePath(out string? fullPath)
    {
        fullPath = null;

        ObjectInstance? selected = SelectedInstance;
        if (!selected.HasValue)
            return false;

        ObjectInstance instance = selected.Value;
        if (!instance.HasTileCoordinate || instance.PlacementEntryIndex < 0)
            return false;

        return _terrainManager.Adapter.TryGetPlacementWritablePath(instance.TileX, instance.TileY, out fullPath);
    }

    public bool TryUpdateSelectedPlacementPosition(Vector3 newPosition, out string error)
    {
        error = string.Empty;

        ObjectInstance? selected = SelectedInstance;
        if (!selected.HasValue)
        {
            error = "No world object is selected.";
            return false;
        }

        ObjectInstance current = selected.Value;
        if (!current.HasTileCoordinate || current.PlacementEntryIndex < 0)
        {
            error = "The selected object is not backed by a writable ADT placement entry.";
            return false;
        }

        if (_selectedObjectType is not (ObjectType.Mdx or ObjectType.Wmo))
        {
            error = "Only ADT MDDF and MODF placements are supported by the current save seam.";
            return false;
        }

        Dictionary<(int, int), List<ObjectInstance>> tileInstances = _selectedObjectType == ObjectType.Mdx
            ? _tileMdxInstances
            : _tileWmoInstances;

        if (!tileInstances.TryGetValue((current.TileX, current.TileY), out List<ObjectInstance>? instances))
        {
            error = $"Tile ({current.TileX}, {current.TileY}) is not currently loaded.";
            return false;
        }

        int instanceIndex = FindPlacementInstanceIndex(instances, current);
        if (instanceIndex < 0)
        {
            error = "The selected placement could not be matched back to the loaded tile instance list.";
            return false;
        }

        ObjectInstance updated = MovePlacementInstance(current, newPosition, _selectedObjectType);
        instances[instanceIndex] = updated;

        _terrainManager.TryUpdateCachedPlacementPosition(_selectedObjectType, current.TileX, current.TileY, current.PlacementEntryIndex, newPosition);
        UpdateAdapterPlacementPosition(_selectedObjectType, current, newPosition);

        _instancesDirty = true;
        RebuildInstanceLists();
        return true;
    }

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
    private bool _clientStarsProbeComplete;
    private string? _clientStarsFallbackModelPath;
    private string? _activeLightSkyboxSourcePath;
    private string? _activeLightSkyboxModelKey;
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
    private WorldAudioRuntime? _audioRuntime;
    private AreaLookupResult? _currentAreaLookup;
    private bool _audioMuted;
    private bool _showAudioEmitterMarkers;
    public string AudioStatus => _audioRuntime?.Status ?? "Audio runtime not configured.";
    public string AudioLastDiagnostic => _audioRuntime?.LastDiagnostic ?? "Audio runtime not configured.";
    public string AreaMusicStatus => _audioRuntime?.AreaMusicStatus ?? "Area music runtime not configured.";
    public bool AudioBackendReady => _audioRuntime?.BackendReady ?? false;
    public bool AudioMuted => _audioMuted;
    public string? AudioPreviewPath => _audioRuntime?.PreviewPath;
    public int ResidentAudioEmitterCount => _audioRuntime?.ResidentEmitterCount ?? 0;
    public int ActiveAudioEmitterCount => _audioRuntime?.ActiveEmitterCount ?? 0;
    public int ResolvedAudioSoundEntryCount => _audioRuntime?.ResolvedSoundEntryCount ?? 0;
    public int ResolvedAudioSoundWaterTypeCount => _audioRuntime?.ResolvedSoundWaterTypeCount ?? 0;
    public bool AudioWorldTriggersEnabled => _audioRuntime?.WorldTriggersEnabled ?? false;
    public bool AreaMusicPlaybackEnabled => _audioRuntime?.AreaMusicPlaybackEnabled ?? false;
    public IReadOnlyList<AudioTriggerDiagnostic> AudioEmitterDiagnostics
        => _audioRuntime?.EmitterDiagnostics ?? Array.Empty<AudioTriggerDiagnostic>();
    public IReadOnlyList<TerrainSoundEmitter> AudioEmitterMarkers
        => _audioRuntime?.ResidentEmitters ?? Array.Empty<TerrainSoundEmitter>();
    public IReadOnlyList<int> ResidentAudioSoundEntryIds
        => _audioRuntime?.ResidentSoundEntryIds ?? Array.Empty<int>();

    /// <summary>
    /// Opt-in world-space speaker markers. This affects only the debug overlay;
    /// it never enables playback or probes audio files.
    /// </summary>
    public bool ShowAudioEmitterMarkers
    {
        get => _showAudioEmitterMarkers;
        set => _showAudioEmitterMarkers = value;
    }

    public bool TryPreviewAudioSoundEntry(uint soundEntryId, bool loop, out string reason)
        => _audioRuntime?.TryPlaySoundEntry(soundEntryId, loop, out reason) ??
            FailAudioPreview("Audio runtime is not configured.", out reason);

    public void StopAudioPreview() => _audioRuntime?.StopPreview();

    public void RefreshAudioEmitterDiagnostics(bool probeFiles)
        => _audioRuntime?.RefreshEmitterDiagnostics(probeFiles);

    /// <summary>
    /// Call once per render of any surface that displays <see cref="AudioEmitterDiagnostics"/>.
    /// Without it the list is not kept current, because the periodic rebuild is gated on someone
    /// actually reading it — see <see cref="WorldAudioRuntime.NoteEmitterDiagnosticsObserved"/> for
    /// why (Spec 153 Defect A).
    /// </summary>
    public void NoteAudioEmitterDiagnosticsObserved()
        => _audioRuntime?.NoteEmitterDiagnosticsObserved();

    /// <summary>Measured coordinate frame of resident MCSE positions. See <see cref="McseFrameEvidence"/>.</summary>
    public McseFrameEvidence AudioMcseFrame => _audioRuntime?.McseFrame ?? McseFrameEvidence.Empty;

    /// <summary>Emitters considered by the last audio update, after the camera-tile window.</summary>
    public int AudioScannedEmitterCount => _audioRuntime?.ScannedEmitterCount ?? 0;

    /// <summary>Emitters that passed the distance test on the last audio update.</summary>
    public int AudioInRangeEmitterCount => _audioRuntime?.InRangeEmitterCount ?? 0;

    public void SetAudioMasterGain(float gain) => _audioRuntime?.SetMasterGain(gain);

    public void SetAudioMuted(bool muted)
    {
        _audioMuted = muted;
        _audioRuntime?.SetMuted(muted);
    }

    public void SetAudioEmitterGain(float gain) => _audioRuntime?.SetEmitterGain(gain);

    public void SetAudioWorldTriggersEnabled(bool enabled) => _audioRuntime?.SetWorldTriggersEnabled(enabled);

    public void SetAreaMusicPlaybackEnabled(bool enabled) => _audioRuntime?.SetAreaMusicPlaybackEnabled(enabled);

    public void PlayAreaMusicNow() => _audioRuntime?.PlayAreaMusicNow();

    public void StopAreaMusicNow() => _audioRuntime?.StopAreaMusicNow();

    public void SetExternalAudioEmitters(IReadOnlyList<TerrainSoundEmitter> emitters) => _audioRuntime?.SetExternalEmitters(emitters);

    /// <summary>
    /// Supplies the same Zone/SubZone resolution used by the viewer status bar
    /// to the audio runtime. This keeps packed Alpha AreaNumber handling in one
    /// lookup path instead of making audio reinterpret the raw chunk value.
    /// </summary>
    public void SetCurrentAreaLookup(AreaLookupResult? areaLookup)
        => _currentAreaLookup = areaLookup;

    private static bool FailAudioPreview(string message, out string reason)
    {
        reason = message;
        return false;
    }

    // DBC Lighting
    private LightService? _lightService;
    public LightService? LightService => _lightService;
    private long _lastAutomaticTimeTick;
    private bool _automaticTimeTickInitialized;

    // Alpha LIT lighting (lazy-loaded on first request)
    private LitLoader? _litLoader;
    private bool _showLitLights;
    private bool _showAreaRegionOverlay;
    private IReadOnlyList<AreaOverlayRegion> _areaOverlayRegions = Array.Empty<AreaOverlayRegion>();
    private int _areaOverlayResidentChunkCount;
    private int _areaOverlayUnresolvedChunkCount;
    private bool _showLitMinimapMarkers;
    private bool _litLoadAttempted;
    private bool _useLitFogOverride;
    private bool _litAutoFallback;
    private string _litAutoFallbackReason = string.Empty;
    // Local Light* spatial selection is retained for diagnostics, but its
    // renderer application remains opt-in until the native local-zone
    // transform/falloff contract is proven for the active build.
    private bool _useLocalDbcLightingOverlay;
    private bool _hasGlobalViewerFogRange;
    private float _globalViewerFogStart;
    private float _globalViewerFogEnd;
    private bool _hasPreLitFogRange;
    private float _preLitFogStart;
    private float _preLitFogEnd;
    private bool _hasUserFogRangeOverride;
    private float _userFogStart = TerrainLightingMath.DefaultFogStart;
    private float _userFogEnd = TerrainLightingMath.DefaultFogEnd;
    private float _activeFogStart = TerrainLightingMath.DefaultFogStart;
    private float _activeFogEnd = TerrainLightingMath.DefaultFogEnd;
    private string _activeFogRangeSource = "Fallback";
    private bool _activeFogRangeAdjusted;
    private string _litStatus = "LIT not loaded.";
    private int _selectedLitLightIndex = -1;
    private string? _selectedLitSourcePath;
    private LitLoader.LitLightingSample? _lastLitSample;
    public bool ShowLitLights
    {
        get => _showLitLights;
        set
        {
            _showLitLights = value;
            if (value && !_litLoadAttempted)
                LazyLoadLit();
        }
    }

    /// <summary>Shows loaded positional LIT entries on shared minimap surfaces without changing lighting.</summary>
    public bool ShowLitMinimapMarkers
    {
        get => _showLitMinimapMarkers;
        set
        {
            _showLitMinimapMarkers = value;
            if (value && !_litLoadAttempted)
                LazyLoadLit();
        }
    }

    public bool UseLitFogOverride
    {
        get => _useLitFogOverride;
        set
        {
            if (_useLitFogOverride == value)
                return;
            if (!value)
                RestorePreLitFogRange(_terrainManager?.Lighting);
            _useLitFogOverride = value;
            if (value && !_litLoadAttempted)
                LazyLoadLit();
        }
    }

    public bool ShowAreaRegionOverlay
    {
        get => _showAreaRegionOverlay;
        set => _showAreaRegionOverlay = value;
    }

    public IReadOnlyList<AreaOverlayRegion> AreaOverlayRegions => _areaOverlayRegions;
    public int AreaOverlayResidentChunkCount => _areaOverlayResidentChunkCount;
    public int AreaOverlayUnresolvedChunkCount => _areaOverlayUnresolvedChunkCount;

    public void SetAreaOverlay(AreaOverlayBuildResult result)
    {
        ArgumentNullException.ThrowIfNull(result);
        _areaOverlayRegions = result.Regions;
        _areaOverlayResidentChunkCount = result.ResidentChunkCount;
        _areaOverlayUnresolvedChunkCount = result.UnresolvedChunkCount;
    }

    public bool UseLocalDbcLightingOverlay
    {
        get => _useLocalDbcLightingOverlay;
        set => _useLocalDbcLightingOverlay = value;
    }

    public LitLoader? LitLoader => _litLoader;
    public bool LitLoadAttempted => _litLoadAttempted;
    public string LitStatus => _litStatus;
    public bool LitAutoFallbackActive => _litAutoFallback;
    public string LitAutoFallbackReason => _litAutoFallbackReason;
    public int SelectedLitLightIndex { get => _selectedLitLightIndex; set => _selectedLitLightIndex = value; }
    public string? SelectedLitSourcePath => _selectedLitSourcePath ?? _litLoader?.SourcePath;
    public IReadOnlyList<string> AvailableLitSourcePaths => _litLoader?.AvailableSourcePaths ?? Array.Empty<string>();
    public LitLoader.LitLightingSample? LastLitSample => _lastLitSample;

    /// <summary>User-selected fog range that is intentionally independent from lighting recommendations.</summary>
    public bool HasUserFogRangeOverride => _hasUserFogRangeOverride;

    public float UserFogStart => _userFogStart;

    public float UserFogEnd => _userFogEnd;

    public float ActiveFogStart => _activeFogStart;

    public float ActiveFogEnd => _activeFogEnd;

    public string ActiveFogRangeSource => _activeFogRangeSource;

    public bool ActiveFogRangeAdjusted => _activeFogRangeAdjusted;

    public void SetUserFogRangeOverride(float fogStart, float fogEnd)
    {
        (_userFogStart, _userFogEnd) = TerrainLightingMath.NormalizeFogRange(fogStart, fogEnd);
        _hasUserFogRangeOverride = true;
    }

    public void ClearUserFogRangeOverride()
    {
        _hasUserFogRangeOverride = false;
    }

    private void CapturePreLitFogRange(TerrainLighting lighting)
    {
        if (_hasPreLitFogRange)
            return;

        _preLitFogStart = lighting.FogStart;
        _preLitFogEnd = lighting.FogEnd;
        _hasPreLitFogRange = true;
    }

    private void RestoreGlobalViewerFogRange(TerrainLighting lighting)
    {
        if (!_hasGlobalViewerFogRange)
        {
            _globalViewerFogStart = lighting.FogStart;
            _globalViewerFogEnd = lighting.FogEnd;
            _hasGlobalViewerFogRange = true;
        }

        lighting.FogStart = _globalViewerFogStart;
        lighting.FogEnd = _globalViewerFogEnd;
    }

    private void RestorePreLitFogRange(TerrainLighting? lighting)
    {
        if (!_hasPreLitFogRange)
            return;
        if (lighting != null)
        {
            lighting.FogStart = _preLitFogStart;
            lighting.FogEnd = _preLitFogEnd;
        }
        _hasPreLitFogRange = false;
    }

    private void ResolveActiveFogRange(TerrainLighting lighting, string recommendationSource)
    {
        float rawRecommendedStart = lighting.FogStart;
        float rawRecommendedEnd = lighting.FogEnd;
        float fallbackStart = _hasPreLitFogRange ? _preLitFogStart : TerrainLightingMath.DefaultFogStart;
        float fallbackEnd = _hasPreLitFogRange ? _preLitFogEnd : TerrainLightingMath.DefaultFogEnd;
        (float recommendedStart, float recommendedEnd) = TerrainLightingMath.NormalizeFogRange(
            rawRecommendedStart,
            rawRecommendedEnd,
            fallbackStart,
            fallbackEnd);

        (float activeStart, float activeEnd) = _hasUserFogRangeOverride
            ? TerrainLightingMath.NormalizeFogRange(_userFogStart, _userFogEnd, recommendedStart, recommendedEnd)
            : (recommendedStart, recommendedEnd);

        _activeFogRangeAdjusted = !FogRangesEqual(rawRecommendedStart, rawRecommendedEnd, recommendedStart, recommendedEnd)
            || (_hasUserFogRangeOverride && !FogRangesEqual(_userFogStart, _userFogEnd, activeStart, activeEnd));
        _activeFogRangeSource = _hasUserFogRangeOverride ? "User override" : recommendationSource;
        _activeFogStart = activeStart;
        _activeFogEnd = activeEnd;
        lighting.FogStart = activeStart;
        lighting.FogEnd = activeEnd;
    }

    private static bool FogRangesEqual(float leftStart, float leftEnd, float rightStart, float rightEnd)
        => MathF.Abs(leftStart - rightStart) < 0.001f && MathF.Abs(leftEnd - rightEnd) < 0.001f;

    // Taxi selection: -1 = show all (or none if !_showTaxi)
    private int _selectedTaxiNodeId = -1;
    private int _selectedTaxiRouteId = -1;
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

        _audioRuntime?.Dispose();
        _audioRuntime = _dataSource is null ? null : new WorldAudioRuntime(_dataSource);
        if (_audioRuntime is not null)
        {
            _audioRuntime.Configure(dbcProvider, dbdDir, build);
            _audioRuntime.SetMuted(_audioMuted);
            foreach ((int tileX, int tileY) in _terrainManager.LoadedTiles.ToArray())
            {
                if (_terrainManager.TryGetTileLoadResult(tileX, tileY, out TileLoadResult result))
                    _audioRuntime.AddTile(tileX, tileY, result.SoundEmitters);
            }
        }
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

    private void LazyLoadLit()
    {
        _litLoadAttempted = true;
        _lastLitSample = null;

        if (_dataSource == null)
        {
            _litStatus = "LIT unavailable: no data source.";
            return;
        }

        _litLoader = new LitLoader(_dataSource, _terrainManager.MapName, _selectedLitSourcePath);
        if (_litLoader.Load())
        {
            _litStatus = _litLoader.Status;
            _selectedLitSourcePath = _litLoader.SourcePath;
            if (_selectedLitLightIndex < 0 && _litLoader.Lights.Count > 0)
                _selectedLitLightIndex = 0;
            return;
        }

        _litStatus = _litLoader.Status;
    }

    public void ReloadLit(string? sourcePath = null)
    {
        _selectedLitSourcePath = string.IsNullOrWhiteSpace(sourcePath) ? null : sourcePath;
        _selectedLitLightIndex = -1;
        _litLoader = null;
        _litLoadAttempted = false;
        _litStatus = "LIT reload queued.";
        LazyLoadLit();
    }

    /// <summary>
    /// Activates the map's LIT lighting source when no usable map-scoped Light DBC profile exists.
    /// This is an automatic default only; the user can still turn the override off in the UI.
    /// </summary>
    public void EnableLitFallback(string reason)
    {
        _litAutoFallback = true;
        _litAutoFallbackReason = string.IsNullOrWhiteSpace(reason)
            ? "No usable map-scoped Light DBC profile is available."
            : reason.Trim();

        if (!_useLitFogOverride)
            UseLitFogOverride = true;
        else if (!_litLoadAttempted)
            LazyLoadLit();
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
            // Keep MPRL low-16 orientation as a raw packed angle until its world-yaw semantics
            // are proven. Basis/sign ambiguity is handled later by the comparison fallback path.
            float angleRadians = DecodeRawMprlPackedAngleRadians(positionRefs[i]);
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

    private static float NormalizeSignedRadians(float radians)
    {
        while (radians > MathF.PI)
            radians -= 2f * MathF.PI;
        while (radians < -MathF.PI)
            radians += 2f * MathF.PI;

        return radians;
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

    private static float ComputeBestSignedYawDeltaWithBasisFallback(float candidateYaw, float expectedYaw)
    {
        const float quarterTurn = MathF.PI * 0.5f;
        float[] expectedCandidates =
        {
            expectedYaw,
            -expectedYaw,
            expectedYaw + quarterTurn,
            expectedYaw - quarterTurn,
            -expectedYaw + quarterTurn,
            -expectedYaw - quarterTurn,
        };

        float bestDelta = 0f;
        float bestAbsDelta = float.MaxValue;
        for (int i = 0; i < expectedCandidates.Length; i++)
        {
            float target = expectedCandidates[i];
            for (int parity = 0; parity < 2; parity++)
            {
                float orientedTarget = target + (parity == 0 ? 0f : MathF.PI);
                float delta = NormalizeSignedRadians(orientedTarget - candidateYaw);
                float absDelta = MathF.Abs(delta);
                if (absDelta < bestAbsDelta)
                {
                    bestAbsDelta = absDelta;
                    bestDelta = delta;
                }
            }
        }

        return bestDelta;
    }

    internal static Vector3 ConvertMprlPositionToWorld(Vector3 refPos)
    {
        // Older PM4 R&D exported MSVT in a fixed viewer/world basis of (Y, X, Z).
        // The raw forensic mapping on the development dataset was:
        //   MPRL X -> raw MSVT Y
        //   MPRL Z -> raw MSVT X
        //   MPRL Y -> raw MSVT Z
        // Folding those together gives viewer/world coordinates of (X, Z, Y).
        return new Vector3(refPos.X, refPos.Z, refPos.Y);
    }

    private static CorePm4CoordinateMode ToCoreCoordinateMode(bool useTileLocalCoordinates)
    {
        return useTileLocalCoordinates
            ? CorePm4CoordinateMode.TileLocal
            : CorePm4CoordinateMode.WorldSpace;
    }

    private static float NearestPositionRefDistanceSquared(IReadOnlyList<MprlEntry> positionRefs, Vector3 world)
    {
        float best = float.MaxValue;
        for (int i = 0; i < positionRefs.Count; i++)
        {
            Vector3 refWorld = ConvertMprlPositionToWorld(positionRefs[i].Position);
            float dx = refWorld.X - world.X;
            float dy = refWorld.Y - world.Y;
            float distSq = dx * dx + dy * dy;
            if (distSq < best)
                best = distSq;
        }

        return best;
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
        _taxiActorTravelByPath.Clear();
        _taxiActorPoseByPath.Clear();
        _taxiActorSmoothedForwardByPath.Clear();
        _taxiActorClockInitialized = false;
    }

    private void UpdateTaxiActorInstances()
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

    private static bool TryGetTaxiRouteSelectionPoint(TaxiPathLoader.TaxiRoute route, out Vector3 point)
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

    private void LazyLoadAreaTriggers()
    {
        _areaTriggerLoadAttempted = true;
        if (_dbcProvider == null || _dbdDir == null || _dbcBuild == null || _mapId < 0) return;
        _areaTriggerLoader = new AreaTriggerLoader();
        _areaTriggerLoader.Load(_dbcProvider, _dbdDir, _dbcBuild, _mapId);
    }

    /// <summary>
    /// Load the exact-build Light* DBC chain for zone-based lighting, with the flattened
    /// LightData table retained only as a later-build compatibility fallback.
    /// </summary>
    public void LoadLighting(DBCD.Providers.IDBCProvider dbcProvider, string dbdDir, string build, int mapId)
    {
        _lightService = new LightService();
        _lightService.Load(dbcProvider, dbdDir, build, mapId);
        if (!_lightService.HasUsableLightingForMap)
        {
            EnableLitFallback(
                $"No usable Light DBC profile exists for map {mapId}; LIT is enabled automatically.");
        }
    }

    public WorldScene(GL gl, string wdtPath, IDataSource? dataSource,
        ReplaceableTextureResolver? texResolver = null,
        string? buildVersion = null,
        MinimapRenderer? minimapRenderer = null,
        Action<string>? onStatus = null)
    {
        _gl = gl;
        _dataSource = dataSource;
        _dbcBuild = buildVersion;
        _minimapRenderer = minimapRenderer;
        _pm4Overlay = new Pm4OverlayScene(this, Pm4OverlayCacheService.CreateForDataSource(dataSource));
        _assets = new WorldAssetManager(gl, dataSource, texResolver, buildVersion);
        _bbRenderer = new BoundingBoxRenderer(gl);
        _skyDome = new SkyDomeRenderer(gl);

        // Create terrain manager (uses AOI-based lazy loading — tiles load as camera moves)
        onStatus?.Invoke("Loading WDT...");
        _terrainManager = new TerrainManager(gl, wdtPath, dataSource);

        // Spec 232 Phase 2 (FR-2): auto-load this map's saved layer project, so a
        // locked-in composition is live on every launch without manual re-entry.
        try
        {
            if (_terrainManager.LoadLayerProject())
                ViewerLog.Important(WoWViewer.Logging.ViewerLog.Category.Terrain,
                    $"[Cartography] Loaded saved layer project for '{_terrainManager.MapName}'.");
        }
        catch (Exception projectEx)
        {
            ViewerLog.Important(WoWViewer.Logging.ViewerLog.Category.Terrain,
                $"[Cartography] Layer project load failed: {projectEx.Message}");
        }

        InitFromAdapter(onStatus);
    }

    /// <summary>
    /// Create a WorldScene with a pre-built TerrainManager (for Standard WDT, etc.).
    /// </summary>
    public WorldScene(GL gl, TerrainManager terrainManager, IDataSource? dataSource,
        ReplaceableTextureResolver? texResolver = null,
        string? buildVersion = null,
        MinimapRenderer? minimapRenderer = null,
        Action<string>? onStatus = null)
    {
        _gl = gl;
        _dataSource = dataSource;
        _dbcBuild = buildVersion;
        _minimapRenderer = minimapRenderer;
        _pm4Overlay = new Pm4OverlayScene(this, Pm4OverlayCacheService.CreateForDataSource(dataSource));
        _assets = new WorldAssetManager(gl, dataSource, texResolver, buildVersion);
        _bbRenderer = new BoundingBoxRenderer(gl);
        _skyDome = new SkyDomeRenderer(gl);
        _terrainManager = terrainManager;

        // Spec 232 Phase 2 (FR-2): same auto-load for the pre-built-manager path (Standard WDT).
        try
        {
            if (_terrainManager.LoadLayerProject())
                ViewerLog.Important(WoWViewer.Logging.ViewerLog.Category.Terrain,
                    $"[Cartography] Loaded saved layer project for '{_terrainManager.MapName}'.");
        }
        catch (Exception projectEx)
        {
            ViewerLog.Important(WoWViewer.Logging.ViewerLog.Category.Terrain,
                $"[Cartography] Layer project load failed: {projectEx.Message}");
        }

        InitFromAdapter(onStatus);
    }

    private void InitFromAdapter(Action<string>? onStatus)
    {
        var adapter = _terrainManager.Adapter;
        _assetLoadPolicy = ResolveTerrainAssetLoadPolicy(adapter);

        if (adapter.ModfPlacements.Count > 0)
        {
            // Register WDT-global placements before any visibility pass. Terrain
            // maps will subsequently stream tile placements; WMO-only maps own
            // their complete placement list here and must not rely on a tile gate.
            var manifest = _assets.BuildManifest(
                adapter.MdxModelNames, adapter.WmoModelNames,
                adapter.MddfPlacements, adapter.ModfPlacements);
            _assets.LoadManifest(manifest);
            BuildInstances(adapter);
        }

        if (adapter.IsWmoBased)
        {
            // WMO-only maps have no resident ADT tile. Their global MODF
            // placements therefore cannot enter the normal _tileWmoInstances
            // collector (and used to remain in _wmoInstances only, which the
            // render admission path never visits). Treat them as an explicit
            // external scene source: still subject to the regular frustum and
            // object filters, but never to terrain-tile residency.
            if (adapter.ModfPlacements.Count > 0)
            {
                _externalWmoInstances.AddRange(_wmoInstances);
                _wmoInstances.Clear();
                _instancesDirty = true;
            }

            if (adapter.ModfPlacements.Count > 0)
            {
                var p = adapter.ModfPlacements[0];
                var bbCenter = (p.BoundsMin + p.BoundsMax) * 0.5f;
                var bbExtent = p.BoundsMax - p.BoundsMin;
                float dist = MathF.Max(bbExtent.Length() * 0.5f, 100f);
                _wmoCameraOverride = bbCenter + new Vector3(dist, 0, bbExtent.Z * 0.3f);
                ViewerLog.Info(ViewerLog.Category.Terrain, $"WMO-only map, camera at BB center: ({bbCenter.X:F1}, {bbCenter.Y:F1}, {bbCenter.Z:F1}), dist={dist:F0}");
            }

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
                _wdlTerrain = new WdlTerrainRenderer(_gl, _minimapRenderer);
                if (!_wdlTerrain.Load(_dataSource, _terrainManager.MapName))
                {
                    _wdlTerrain.Dispose();
                    _wdlTerrain = null;
                }
            }

            _terrainManager.OnTileLoaded += OnTileLoaded;
            _terrainManager.OnTileUnloaded += OnTileUnloaded;
            onStatus?.Invoke("World loaded (tiles stream as you move).");
        }

        if (!adapter.IsWmoBased)
        {
            AdtProfile adtProfile = FormatProfileRegistry.ResolveAdtProfile(_dbcBuild);
            ViewerLog.Info(
                ViewerLog.Category.Terrain,
                $"Terrain asset load policy: build={_dbcBuild ?? "unknown"}, adtProfile={adtProfile.ProfileId}, prewarmTileAssets={_assetLoadPolicy.PrewarmTileAssets}, visibleMdx={_assetLoadPolicy.MaxNewMdxLoadsPerFrame}, visibleWmo={_assetLoadPolicy.MaxNewWmoLoadsPerFrame}, deferredLoads={_assetLoadPolicy.MaxDeferredLoadsPerFrame}, deferredBudgetMs={_assetLoadPolicy.MaxDeferredLoadBudgetMs:F1}");
        }
        
        // Auto-load WL liquids if enabled
        if (_showWlLiquids && !_wlLoadAttempted)
        {
            LazyLoadWlLiquids();
        }
    }

    private TerrainAssetLoadPolicy ResolveTerrainAssetLoadPolicy(ITerrainAdapter adapter)
    {
        return adapter.IsWmoBased
            ? WmoOnlyAssetLoadPolicy
            : StreamingTerrainAssetLoadPolicy;
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
        // MDX and WMO placements share one renderer-space transform. Keeping
        // this in Core.Runtime is important because bounds and mesh submission
        // must agree about which side of a placement is in front of the camera.
        foreach (var p in adapter.MddfPlacements)
        {
            if (p.NameIndex < 0 || p.NameIndex >= mdxNames.Count) continue;

            string key = WorldAssetManager.NormalizeKey(mdxNames[p.NameIndex]);
            float scale = p.Scale > 0 ? p.Scale : 1.0f;
            var transform = WorldPlacementTransform.Build(p.Position, p.Rotation, scale);

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
            ResolveMdxSelectionBounds(key, localMin, localMax, boundsResolved,
                out Vector3 selectionMin, out Vector3 selectionMax, out bool selectionResolved);
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
                SelectionLocalBoundsMin = selectionMin,
                SelectionLocalBoundsMax = selectionMax,
                SelectionBoundsResolved = selectionResolved,
                ModelName = Path.GetFileName(modelPath),
                ModelPath = modelPath,
                PlacementPosition = p.Position,
                PlacementRotation = p.Rotation,
                PlacementScale = scale,
                UniqueId = p.UniqueId,
                PlacementEntryIndex = -1,
                TileX = -1,
                TileY = -1,
                HasTileCoordinate = false
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
            var transform = WorldPlacementTransform.Build(p.Position, p.Rotation);

            // Get geometry-tight local bounds for the WMO placement and transform them to world space.
            // Falls back to MODF file bounds if the model summary is unavailable.
            Vector3 localMin, localMax, worldMin, worldMax;
            if (_assets.TryGetWmoPlacementBounds(key, out localMin, out localMax))
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
                UniqueId = p.UniqueId,
                PlacementEntryIndex = -1,
                TileX = -1,
                TileY = -1,
                HasTileCoordinate = false
            });
        }

        ViewerLog.Important(ViewerLog.Category.Terrain, $"Instances: {_mdxInstances.Count} MDX, {_skyboxInstances.Count} skybox, {_wmoInstances.Count} WMO");
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
        _audioRuntime?.AddTile(tileX, tileY, result.SoundEmitters);
        var adapter = _terrainManager.Adapter;
        var mdxNames = adapter.MdxModelNames;
        var wmoNames = adapter.WmoModelNames;

        // Build MDX instances for this tile
        var tileMdx = new List<ObjectInstance>();
        var tileSkyboxes = new List<ObjectInstance>();
        int tileMddfEntryIndex = 0;
        foreach (var p in result.MddfPlacements)
        {
            if (p.NameIndex < 0 || p.NameIndex >= mdxNames.Count) continue;
            string key = WorldAssetManager.NormalizeKey(mdxNames[p.NameIndex]);
            float scale = p.Scale > 0 ? p.Scale : 1.0f;

            var transform = WorldPlacementTransform.Build(p.Position, p.Rotation, scale);
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
            ResolveMdxSelectionBounds(key, localMin, localMax, boundsResolved,
                out Vector3 selectionMin, out Vector3 selectionMax, out bool selectionResolved);
            string modelPath = mdxNames[p.NameIndex];
            var instance = new ObjectInstance
            {
                ModelKey = key, Transform = transform, BoundsMin = bbMin, BoundsMax = bbMax,
                LocalBoundsMin = localMin, LocalBoundsMax = localMax, BoundsResolved = boundsResolved,
                SelectionLocalBoundsMin = selectionMin,
                SelectionLocalBoundsMax = selectionMax,
                SelectionBoundsResolved = selectionResolved,
                ModelName = Path.GetFileName(modelPath), ModelPath = modelPath,
                PlacementPosition = p.Position, PlacementRotation = p.Rotation, PlacementScale = scale,
                UniqueId = p.UniqueId,
                PlacementEntryIndex = tileMddfEntryIndex,
                TileX = tileX,
                TileY = tileY,
                HasTileCoordinate = true
            };

            if (IsSkyboxModelPath(modelPath))
                tileSkyboxes.Add(instance);
            else
                tileMdx.Add(instance);

            tileMddfEntryIndex++;
        }

        // Build WMO instances for this tile
        var tileWmo = new List<ObjectInstance>();
        int tileModfEntryIndex = 0;
        foreach (var p in result.ModfPlacements)
        {
            if (p.NameIndex < 0 || p.NameIndex >= wmoNames.Count) continue;
            string key = WorldAssetManager.NormalizeKey(wmoNames[p.NameIndex]);
            var transform = WorldPlacementTransform.Build(p.Position, p.Rotation);

            // Get geometry-tight local bounds and transform to world space.
            Vector3 localMin, localMax, worldMin, worldMax;
            if (_assets.TryGetWmoPlacementBounds(key, out localMin, out localMax))
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
                UniqueId = p.UniqueId,
                PlacementEntryIndex = tileModfEntryIndex,
                TileX = tileX,
                TileY = tileY,
                HasTileCoordinate = true
            });

            tileModfEntryIndex++;
        }

        _tileMdxInstances[(tileX, tileY)] = tileMdx;
        _tileSkyboxInstances[(tileX, tileY)] = tileSkyboxes;
        _tileWmoInstances[(tileX, tileY)] = tileWmo;
        UpdateObjectBucketBounds(_tileMdxBounds, (tileX, tileY), tileMdx);
        UpdateObjectBucketBounds(_tileWmoBounds, (tileX, tileY), tileWmo);
        _instancesDirty = true;

        if (_assetLoadPolicy.PrewarmTileAssets
            || (CapturePreloadActive && _capturePreloadTiles.Contains((tileX, tileY))))
            QueueTileAssetLoads(tileMdx, tileSkyboxes, tileWmo);

        if ((tileMdx.Count > 0 || tileSkyboxes.Count > 0 || tileWmo.Count > 0) && ViewerLog.Verbose)
            ViewerLog.Trace($"[Terrain] Tile ({tileX},{tileY}) loaded: {tileMdx.Count} MDX, {tileSkyboxes.Count} skybox, {tileWmo.Count} WMO instances");
    }

    /// <summary>
    /// Called by TerrainManager when a tile leaves the AOI.
    /// </summary>
    private void OnTileUnloaded(int tileX, int tileY)
    {
        _audioRuntime?.RemoveTile(tileX, tileY);
        int wmoInstanceCount = _tileWmoInstances.TryGetValue((tileX, tileY), out List<ObjectInstance>? wmoInstances)
            ? wmoInstances.Count
            : 0;
        _tileMdxInstances.Remove((tileX, tileY));
        _tileSkyboxInstances.Remove((tileX, tileY));
        _tileWmoInstances.Remove((tileX, tileY));
        _tileMdxBounds.Remove((tileX, tileY));
        _tileWmoBounds.Remove((tileX, tileY));
        LastUnloadedWmoTileX = tileX;
        LastUnloadedWmoTileY = tileY;
        LastUnloadedWmoInstanceCount = wmoInstanceCount;
        WmoTileUnloadEventCount++;
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

        RebuildFlatVisibilityBuckets();

        RestoreSelectedSceneObjectAfterRebuild();
        if (UseHierarchicalSceneTraversal)
            RebuildSceneGraphObjectIndex();
        else
            _sceneGraphBuild = null;

        _instancesDirty = false;
    }

    private void RebuildFlatVisibilityBuckets()
    {
        RebuildFlatVisibilityBuckets(_tileMdxInstances, _tileMdxVisibilityBuckets);
        RebuildFlatVisibilityBuckets(_tileWmoInstances, _tileWmoVisibilityBuckets);
    }

    private static void RebuildFlatVisibilityBuckets(
        IReadOnlyDictionary<(int, int), List<ObjectInstance>> source,
        Dictionary<(int, int), List<FlatVisibilityBucket>> destination)
    {
        destination.Clear();
        foreach (KeyValuePair<(int, int), List<ObjectInstance>> tile in source)
        {
            Dictionary<(int chunkX, int chunkY), FlatVisibilityBucket> byChunk = new();
            FlatVisibilityBucket? fallback = null;
            foreach (ObjectInstance instance in tile.Value)
            {
                if (!TryGetSceneObjectChunkKey(instance, out (int tileX, int tileY, int chunkX, int chunkY) chunkKey)
                    || chunkKey.tileX != tile.Key.Item1
                    || chunkKey.tileY != tile.Key.Item2)
                {
                    fallback ??= new FlatVisibilityBucket();
                    fallback.Add(instance);
                    continue;
                }

                if (!byChunk.TryGetValue((chunkKey.chunkX, chunkKey.chunkY), out FlatVisibilityBucket? bucket))
                {
                    bucket = new FlatVisibilityBucket();
                    byChunk.Add((chunkKey.chunkX, chunkKey.chunkY), bucket);
                }

                bucket.Add(instance);
            }

            List<FlatVisibilityBucket> buckets = new(byChunk.Count + (fallback is null ? 0 : 1));
            buckets.AddRange(byChunk.Values);
            if (fallback is not null)
                buckets.Add(fallback);
            destination[tile.Key] = buckets;
        }
    }

    private void RebuildSceneGraphObjectIndex()
    {
        _sceneGraphPortalAdapters.Clear();
        _sceneGraphPortalVisibility.Clear();
        List<WorldSceneGraphObjectPlacement> placements = new(
            _mdxInstances.Count + _skyboxInstances.Count + _wmoInstances.Count);

        bool hasPartitionedSources = _tileMdxInstances.Count > 0
            || _tileSkyboxInstances.Count > 0
            || _tileWmoInstances.Count > 0
            || _externalMdxInstances.Count > 0
            || _externalSkyboxInstances.Count > 0
            || _externalWmoInstances.Count > 0;
        if (hasPartitionedSources)
        {
            AppendSceneGraphPlacements(placements, _tileMdxInstances, WorldSceneNodeKind.M2Placement, isSkybox: false, isExternal: false, requiresUpdate: true);
            AppendSceneGraphPlacements(placements, _tileSkyboxInstances, WorldSceneNodeKind.M2Placement, isSkybox: true, isExternal: false, requiresUpdate: true);
            AppendSceneGraphPlacements(placements, _tileWmoInstances, WorldSceneNodeKind.WmoPlacement, isSkybox: false, isExternal: false, requiresUpdate: false, childFactory: BuildWmoSceneGraphChildren);
            AppendSceneGraphPlacements(placements, _externalMdxInstances, WorldSceneNodeKind.M2Placement, isSkybox: false, isExternal: true, requiresUpdate: true);
            AppendSceneGraphPlacements(placements, _externalSkyboxInstances, WorldSceneNodeKind.M2Placement, isSkybox: true, isExternal: true, requiresUpdate: true);
            AppendSceneGraphPlacements(placements, _externalWmoInstances, WorldSceneNodeKind.WmoPlacement, isSkybox: false, isExternal: true, requiresUpdate: false, childFactory: BuildWmoSceneGraphChildren);
        }
        else
        {
            AppendSceneGraphPlacements(placements, _mdxInstances, WorldSceneNodeKind.M2Placement, isSkybox: false, isExternal: true, requiresUpdate: true);
            AppendSceneGraphPlacements(placements, _skyboxInstances, WorldSceneNodeKind.M2Placement, isSkybox: true, isExternal: true, requiresUpdate: true);
            AppendSceneGraphPlacements(placements, _wmoInstances, WorldSceneNodeKind.WmoPlacement, isSkybox: false, isExternal: true, requiresUpdate: false, childFactory: BuildWmoSceneGraphChildren);
        }

        _sceneGraphBuild = WorldSceneGraphObjectAdapter.BuildPerAdt(placements);
        _sceneGraphFrameVisibilityPrepared = false;
        _lastSceneGraphTraversalDiagnostics = new WorldSceneTraversalDiagnostics();
    }

    private static void AppendSceneGraphPlacements(
        List<WorldSceneGraphObjectPlacement> destination,
        IEnumerable<KeyValuePair<(int, int), List<ObjectInstance>>> tileInstances,
        WorldSceneNodeKind kind,
        bool isSkybox,
        bool isExternal,
        bool requiresUpdate,
        Func<string, ObjectInstance, IReadOnlyList<WorldSceneGraphChildNode>?>? childFactory = null)
    {
        foreach (KeyValuePair<(int, int), List<ObjectInstance>> tile in tileInstances)
        {
            AppendSceneGraphPlacements(destination, tile.Value, kind, isSkybox, isExternal, requiresUpdate, childFactory, tile.Key);
        }
    }

    private static void AppendSceneGraphPlacements(
        List<WorldSceneGraphObjectPlacement> destination,
        IReadOnlyList<ObjectInstance> instances,
        WorldSceneNodeKind kind,
        bool isSkybox,
        bool isExternal,
        bool requiresUpdate,
        Func<string, ObjectInstance, IReadOnlyList<WorldSceneGraphChildNode>?>? childFactory = null,
        (int tileX, int tileY)? tileKey = null)
    {
        for (int index = 0; index < instances.Count; index++)
        {
            ObjectInstance instance = instances[index];
            string sourceToken = isExternal
                ? "external"
                : $"tile/{tileKey!.Value.tileX:D2}/{tileKey.Value.tileY:D2}";
            string id = $"world/object/{GetSceneGraphKindToken(kind, isSkybox)}/{sourceToken}/{index:D6}";
            destination.Add(new WorldSceneGraphObjectPlacement(
                id,
                kind,
                instance,
                isExternal,
                WorldSceneRenderPass.Opaque,
                IsQueryable: true,
                RequiresUpdate: requiresUpdate,
                IsSkybox: isSkybox,
                Children: childFactory?.Invoke(id, instance),
                SpatialBucket: GetSceneGraphSpatialBucket(kind, isSkybox, isExternal, tileKey, instance)));
        }
    }

    private static WorldSceneGraphSpatialBucket? GetSceneGraphSpatialBucket(
        WorldSceneNodeKind kind,
        bool isSkybox,
        bool isExternal,
        (int tileX, int tileY)? tileKey,
        in ObjectInstance instance)
    {
        if (kind != WorldSceneNodeKind.M2Placement
            || isSkybox
            || isExternal
            || !tileKey.HasValue
            || !instance.HasTileCoordinate
            || !TryGetSceneObjectChunkKey(instance, out (int tileX, int tileY, int chunkX, int chunkY) chunkKey)
            || chunkKey.tileX != tileKey.Value.tileX
            || chunkKey.tileY != tileKey.Value.tileY)
        {
            return null;
        }

        return new WorldSceneGraphSpatialBucket(
            WorldSceneNodeKind.Chunk,
            $"{chunkKey.chunkX:D2}/{chunkKey.chunkY:D2}");
    }

    private IReadOnlyList<WorldSceneGraphChildNode>? BuildWmoSceneGraphChildren(
        string parentId,
        ObjectInstance instance)
    {
        if (_assets.TryGetLoadedWmo(instance.ModelKey, out WmoRenderer? renderer) && renderer is not null)
        {
            _sceneGraphPortalAdapters[parentId] = WorldScenePortalAdapter.Build(
                renderer.GetSceneGraphPortalGroups(),
                renderer.GetSceneGraphPortalReadModels(),
                parentId);
        }

        if (!_assets.TryGetCachedWmoMeshSummary(instance.ModelKey, out WmoMeshSummary summary)
            || summary.GroupSummaries is null
            || summary.GroupSummaries.Length == 0)
        {
            return null;
        }

        List<WorldSceneGraphChildNode> children = new(summary.GroupSummaries.Length);
        foreach (WmoGroupMeshSummary group in summary.GroupSummaries.OrderBy(group => group.GroupIndex))
        {
            bool boundsKnown = AreFiniteOrderedBounds(group.BoundsMin, group.BoundsMax);
            children.Add(new WorldSceneGraphChildNode(
                $"{parentId}/group/{group.GroupIndex:D4}",
                WorldSceneNodeKind.WmoGroup,
                Matrix4x4.Identity,
                boundsKnown ? group.BoundsMin : Vector3.Zero,
                boundsKnown ? group.BoundsMax : Vector3.Zero,
                BoundsKnown: boundsKnown,
                IsRenderable: true,
                IsQueryable: true,
                RequiresUpdate: false,
                AssetKey: $"{instance.ModelKey}#group/{group.GroupIndex:D4}",
                RenderPassMask: WorldSceneRenderPass.Opaque,
                PortalGroup: group.GroupIndex));
        }

        return children;
    }

    private static bool AreFiniteOrderedBounds(Vector3 min, Vector3 max)
    {
        return float.IsFinite(min.X) && float.IsFinite(min.Y) && float.IsFinite(min.Z)
            && float.IsFinite(max.X) && float.IsFinite(max.Y) && float.IsFinite(max.Z)
            && min.X <= max.X && min.Y <= max.Y && min.Z <= max.Z;
    }

    private static string GetSceneGraphKindToken(WorldSceneNodeKind kind, bool isSkybox)
    {
        if (kind == WorldSceneNodeKind.WmoPlacement)
            return "wmo";

        return isSkybox ? "m2-skybox" : "m2";
    }

    private bool TryGetSelectedSceneInstance(out ObjectInstance instance)
    {
        // Asset promotion can resolve bounds after the frame-maintenance pass and mark the
        // placement lists dirty. Never rebuild the full world synchronously from a read/query
        // accessor (the render-time selected-bounds overlay is one such accessor); let the next
        // frame's scene-maintenance pass perform the rebuild once.
        if (_instancesDirty)
        {
            instance = default;
            return false;
        }

        if (TryGetSceneObjectByIndex(_selectedObjectType, _selectedObjectIndex, out instance))
        {
            if (!_selectedSceneObjectKey.HasValue || !IsSameSceneObject(instance, _selectedSceneObjectKey.Value))
                _selectedSceneObjectKey = CreateSelectedSceneObjectKey(_selectedObjectType, instance);

            return true;
        }

        if (_selectedSceneObjectKey.HasValue
            && TryResolveSelectedSceneObject(_selectedSceneObjectKey.Value, out ObjectType resolvedType, out int resolvedIndex, out instance))
        {
            _selectedObjectType = resolvedType;
            _selectedObjectIndex = resolvedIndex;
            return true;
        }

        instance = default;
        return false;
    }

    private void RestoreSelectedSceneObjectAfterRebuild()
    {
        if (!_selectedSceneObjectKey.HasValue)
            return;

        if (TryResolveSelectedSceneObject(_selectedSceneObjectKey.Value, out ObjectType resolvedType, out int resolvedIndex, out _))
        {
            _selectedObjectType = resolvedType;
            _selectedObjectIndex = resolvedIndex;
            return;
        }

        _selectedObjectIndex = -1;
    }

    private bool TryResolveSelectedSceneObject(SelectedSceneObjectKey key, out ObjectType objectType, out int objectIndex, out ObjectInstance instance)
    {
        List<ObjectInstance> instances = key.ObjectType switch
        {
            ObjectType.Wmo => _wmoInstances,
            ObjectType.Mdx => _mdxInstances,
            _ => []
        };

        for (int index = 0; index < instances.Count; index++)
        {
            ObjectInstance candidate = instances[index];
            if (!IsSameSceneObject(candidate, key))
                continue;

            objectType = key.ObjectType;
            objectIndex = index;
            instance = candidate;
            return true;
        }

        objectType = ObjectType.None;
        objectIndex = -1;
        instance = default;
        return false;
    }

    /// <summary>
    /// Tests a world-space point against all placed WMO instances and their internal group bounding boxes.
    /// Returns true if the point falls inside a WMO group.
    /// </summary>
    public bool TryGetWmoGroupAt(Vector3 worldPos, out ObjectInstance wmoInstance, out WmoRenderer renderer, out int renderGroupIndex)
    {
        wmoInstance = default;
        renderer = null!;
        renderGroupIndex = -1;

        if (_wmoInstances == null || _wmoInstances.Count == 0)
            return false;

        float bestVolume = float.MaxValue;
        bool found = false;

        foreach (var inst in _wmoInstances)
        {
            if (inst.BoundsResolved &&
                (worldPos.X < inst.BoundsMin.X || worldPos.X > inst.BoundsMax.X ||
                 worldPos.Y < inst.BoundsMin.Y || worldPos.Y > inst.BoundsMax.Y ||
                 worldPos.Z < inst.BoundsMin.Z || worldPos.Z > inst.BoundsMax.Z))
            {
                continue;
            }

            if (!_assets.TryGetLoadedWmo(inst.ModelKey, out WmoRenderer? wmoRenderer) || wmoRenderer == null)
                continue;

            if (!Matrix4x4.Invert(inst.Transform, out Matrix4x4 invTransform))
                continue;

            Vector3 localPos = Vector3.Transform(worldPos, invTransform);
            int groupIdx = wmoRenderer.FindGroupContainingPoint(localPos);
            if (groupIdx >= 0)
            {
                if (wmoRenderer.TryGetGroupBounds(groupIdx, out Vector3 gMin, out Vector3 gMax))
                {
                    Vector3 size = gMax - gMin;
                    float vol = Math.Abs(size.X * size.Y * size.Z);
                    if (vol < bestVolume)
                    {
                        bestVolume = vol;
                        wmoInstance = inst;
                        renderer = wmoRenderer;
                        renderGroupIndex = groupIdx;
                        found = true;
                    }
                }
                else if (!found)
                {
                    wmoInstance = inst;
                    renderer = wmoRenderer;
                    renderGroupIndex = groupIdx;
                    found = true;
                }
            }
        }

        return found;
    }

    public bool TryGetSelectedWmoDoodad(out WmoDoodadInfo doodadInfo, out Vector3 worldPosition, out ObjectInstance parentWmo)
    {
        doodadInfo = default;
        worldPosition = Vector3.Zero;
        parentWmo = default;

        if (_selectedObjectType != ObjectType.WmoDoodad || _selectedWmoParentIndex < 0 || _selectedWmoParentIndex >= _wmoInstances.Count)
            return false;

        parentWmo = _wmoInstances[_selectedWmoParentIndex];
        if (!_assets.TryGetLoadedWmo(parentWmo.ModelKey, out WmoRenderer? wmo) || wmo == null)
            return false;

        if (!wmo.TryGetDoodadInfo(_selectedObjectIndex, out doodadInfo))
            return false;

        worldPosition = Vector3.Transform(doodadInfo.LocalPosition, parentWmo.Transform);
        return true;
    }

    /// <summary>
    /// Build the selected WMO doodad as an <see cref="ObjectInstance"/> for the inspector and the
    /// selection-bounds overlay.
    /// </summary>
    /// <remarks>
    /// This used to synthesise a placeholder: a hard-coded <c>position ± 1</c> cube for bounds, no
    /// model name or path, no rotation, no scale, and the MODD table index copied into
    /// <c>UniqueId</c>. Three separate defects followed from that.
    ///
    /// The bounds were the visible one. <see cref="WmoRenderer.TryGetDoodadBounds"/> already
    /// computes the doodad's real transformed AABB and the picker already carried it, but this
    /// method threw it away and substituted a 2-yard cube. The selection overlay then inflates
    /// whatever it is given by a further 0.75 yd or more for its accent box, so a scroll a third of
    /// a yard across was drawn inside about 3.8 yards of wireframe. The minimum-half-extent clamp
    /// added alongside the overlay could never have helped: 1.0 already exceeds the 0.75 floor, so
    /// the clamp was a no-op on exactly the objects it was meant to fix.
    ///
    /// <c>UniqueId</c> was the subtle one. MODD has no uniqueId field at all — uniqueId identifies
    /// an MDDF/MODF placement in an ADT. Copying the MODD index into it did not merely mislabel the
    /// inspector; <c>ShouldHideObjectInstanceByUniqueId</c> keys on that field, so a doodad could be
    /// hidden by an unrelated placement's id colliding with its table index. The def index is kept
    /// in <see cref="ObjectInstance.PlacementEntryIndex"/>, which is what it actually is, and
    /// UniqueId is left at 0 to mean "this kind of object does not have one".
    /// </remarks>
    private bool TryBuildSelectedWmoDoodadInstance(out ObjectInstance instance)
    {
        instance = default;

        if (!TryGetSelectedWmoDoodad(out WmoDoodadInfo dInfo, out Vector3 worldPosition, out ObjectInstance parentWmo))
            return false;

        if (!_assets.TryGetLoadedWmo(parentWmo.ModelKey, out WmoRenderer? wmoRenderer) || wmoRenderer == null)
            return false;

        // Real geometry bounds when the doodad's model is loaded. When it is not, say so rather
        // than presenting the placeholder cube as the object's extent.
        bool boundsResolved = false;
        Vector3 boundsMin = worldPosition;
        Vector3 boundsMax = worldPosition;
        if (wmoRenderer.TryGetDoodadBounds(dInfo.Index, parentWmo.Transform, out Vector3 bMin, out Vector3 bMax, out boundsResolved))
        {
            boundsMin = bMin;
            boundsMax = bMax;
        }

        // Local (model-space) geometry bounds drive the oriented tight selection box. When the
        // doodad's model has not streamed in yet, request it so real bounds resolve within a few
        // frames instead of the selection sitting on a centroid cube indefinitely.
        bool hasLocalBounds = wmoRenderer.TryGetDoodadLocalBounds(dInfo.Index, out Vector3 localMin, out Vector3 localMax);
        if (!boundsResolved)
            wmoRenderer.RequestDoodadModelLoad(dInfo.Index);

        if (!wmoRenderer.TryGetDoodadWorldTransform(dInfo.Index, parentWmo.Transform, out Matrix4x4 doodadTransform))
            doodadTransform = Matrix4x4.CreateTranslation(worldPosition);

        instance = new ObjectInstance
        {
            ModelKey = dInfo.ModelPath,
            ModelPath = dInfo.ModelPath,
            ModelName = System.IO.Path.GetFileName(dInfo.ModelPath),
            AssetKind = "WMO Doodad",
            PlacementPosition = worldPosition,
            PlacementRotation = QuaternionToEulerDegrees(dInfo.Orientation),
            PlacementScale = dInfo.Scale,
            BoundsMin = boundsMin,
            BoundsMax = boundsMax,
            BoundsResolved = boundsResolved,
            // MODD has no uniqueId. The def index is a MODD table index, meaningful only inside this
            // WMO; it belongs here, not in UniqueId. See the remarks above.
            PlacementEntryIndex = dInfo.DoodadDefIndex,
            UniqueId = 0,
            LocalBoundsMin = hasLocalBounds ? localMin : Vector3.Zero,
            LocalBoundsMax = hasLocalBounds ? localMax : Vector3.Zero,
            SelectionLocalBoundsMin = hasLocalBounds ? localMin : Vector3.Zero,
            SelectionLocalBoundsMax = hasLocalBounds ? localMax : Vector3.Zero,
            SelectionBoundsResolved = hasLocalBounds,
            Transform = doodadTransform
        };
        return true;
    }

    private static Vector3 QuaternionToEulerDegrees(Quaternion q)
    {
        float sinR = 2f * (q.W * q.X + q.Y * q.Z);
        float cosR = 1f - 2f * (q.X * q.X + q.Y * q.Y);
        float roll = MathF.Atan2(sinR, cosR);

        float sinP = Math.Clamp(2f * (q.W * q.Y - q.Z * q.X), -1f, 1f);
        float pitch = MathF.Asin(sinP);

        float sinY = 2f * (q.W * q.Z + q.X * q.Y);
        float cosY = 1f - 2f * (q.Y * q.Y + q.Z * q.Z);
        float yaw = MathF.Atan2(sinY, cosY);

        const float ToDegrees = 180f / MathF.PI;
        return new Vector3(roll * ToDegrees, pitch * ToDegrees, yaw * ToDegrees);
    }

    private bool TryGetSceneObjectByIndex(ObjectType objectType, int objectIndex, out ObjectInstance instance)
    {
        switch (objectType)
        {
            case ObjectType.Wmo when objectIndex >= 0 && objectIndex < _wmoInstances.Count:
                instance = _wmoInstances[objectIndex];
                return true;
            case ObjectType.Mdx when objectIndex >= 0 && objectIndex < _mdxInstances.Count:
                instance = _mdxInstances[objectIndex];
                return true;
            case ObjectType.WmoDoodad when TryBuildSelectedWmoDoodadInstance(out instance):
                return true;
            default:
                instance = default;
                return false;
        }
    }

    private static SelectedSceneObjectKey CreateSelectedSceneObjectKey(ObjectType objectType, ObjectInstance instance)
    {
        return new SelectedSceneObjectKey(
            objectType,
            instance.UniqueId,
            instance.PlacementEntryIndex,
            instance.TileX,
            instance.TileY,
            instance.HasTileCoordinate,
            instance.ModelKey,
            instance.PlacementPosition);
    }

    private static bool IsSameSceneObject(ObjectInstance candidate, SelectedSceneObjectKey key)
    {
        if (candidate.UniqueId != key.UniqueId || candidate.PlacementEntryIndex != key.PlacementEntryIndex)
            return false;

        if (candidate.HasTileCoordinate != key.HasTileCoordinate)
            return false;

        if (candidate.HasTileCoordinate)
            return candidate.TileX == key.TileX && candidate.TileY == key.TileY;

        if (!string.Equals(candidate.ModelKey, key.ModelKey, StringComparison.OrdinalIgnoreCase))
            return false;

        return Vector3.DistanceSquared(candidate.PlacementPosition, key.PlacementPosition) < 0.0001f;
    }

    private IModelRenderer? TryGetQueuedMdx(string modelKey)
    {
        if (_assets.TryGetLoadedMdx(modelKey, out var renderer))
            return renderer;
        return null;
    }

    private WmoRenderer? TryGetQueuedWmo(string modelKey)
    {
        if (_assets.TryGetLoadedWmo(modelKey, out var renderer))
            return renderer;
        return null;
    }

    private IModelRenderer? ResolveVisibleMdxRenderer(WorldRenderFrame frame, string modelKey)
    {
        if (frame.VisibleMdxRendererCache.TryGetValue(modelKey, out IModelRenderer? renderer))
            return renderer;

        renderer = TryGetQueuedMdx(modelKey);
        if (renderer != null)
            frame.VisibleMdxRendererCache[modelKey] = renderer;

        return renderer;
    }

    /// <summary>
    /// The render path that actually drew <paramref name="modelKey"/>, from the asset manager's
    /// recorded route decision.
    /// </summary>
    /// <remarks>
    /// Spec 201 FR-002. Reads <c>AppliedRoute</c>, not <c>PrimaryRoute</c>: a model whose primary
    /// route failed and fell back drew on the fallback, and that is what the metric has to say.
    /// A key with no recorded decision maps to <see cref="WorldModelRenderPath.Unknown"/> rather
    /// than being guessed at or dropped.
    /// </remarks>
    private WorldModelRenderPath ResolveVisibleMdxRenderPath(WorldRenderFrame frame, string modelKey)
    {
        if (frame.VisibleMdxRenderPathCache.TryGetValue(modelKey, out WorldModelRenderPath cached))
            return cached;

        M2RouteDecision? decision = _assets.GetRouteDecision(modelKey);
        WorldModelRenderPath path = decision is null
            ? WorldModelRenderPath.Unknown
            : decision.AppliedRoute switch
            {
                M2RouteType.AdapterSkin => WorldModelRenderPath.AdapterSkin,
                M2RouteType.AdapterEmbeddedProfile => WorldModelRenderPath.AdapterEmbeddedProfile,
                M2RouteType.NativeEmbeddedProfile => WorldModelRenderPath.NativeEmbeddedProfile,
                M2RouteType.ConversionFallback => WorldModelRenderPath.ConversionFallback,
                M2RouteType.MdxDirect => WorldModelRenderPath.MdxDirect,
                _ => WorldModelRenderPath.Unknown,
            };

        frame.VisibleMdxRenderPathCache[modelKey] = path;
        return path;
    }

    private WmoRenderer? ResolveVisibleWmoRenderer(WorldRenderFrame frame, string modelKey)
    {
        if (frame.VisibleWmoRendererCache.TryGetValue(modelKey, out WmoRenderer? renderer))
        {
            renderer?.SetRuntimeDoodadsVisible(_doodadsVisible);
            return renderer;
        }

        renderer = TryGetQueuedWmo(modelKey);
        if (renderer != null)
        {
            renderer.SetRuntimeDoodadsVisible(_doodadsVisible);
            frame.VisibleWmoRendererCache[modelKey] = renderer;
        }

        return renderer;
    }

    private void PlanVisibleMdxPasses(WorldRenderFrame frame)
    {
        WorldObjectPassCoordinator.PlanOpaqueMdxRoutes(
            frame.ObjectPasses,
            frame.Visibility,
            visible =>
            {
                // Spec 153 Defect B. This predicate used to `return true` unconditionally, so 100%
                // of opaque MDX took the per-instance RenderWithTransform route while WMO batched
                // 198/198 — the batching machinery was present and inert.
                //
                // The renderer already declares whether it can be batched, and the WMO-internal
                // doodad path has consumed that declaration all along. Use the same contract here
                // instead of overriding it at the planner.
                if (!MdxOpaqueBatchingEnabled)
                    return true;

                IModelRenderer? renderer = ResolveVisibleMdxRenderer(frame, visible.Instance.ModelKey);
                return renderer == null || renderer.RequiresUnbatchedWorldRender;
            });

        WorldObjectPassCoordinator.PlanTransparentMdxRoutes(
            frame.ObjectPasses,
            frame.Visibility,
            visible =>
            {
                IModelRenderer? renderer = ResolveVisibleMdxRenderer(frame, visible.Instance.ModelKey);
                return renderer != null && renderer.HasTransparentWorldPass;
            });
    }

    private IModelRenderer? ResolveFirstOpaqueBatchedVisibleMdxRenderer(WorldRenderFrame frame)
    {
        if (frame.ObjectPasses.FirstOpaqueBatchedVisibleMdxIndex < 0)
            return null;

        VisibleMdxInstance visible = frame.Visibility.VisibleMdx[frame.ObjectPasses.FirstOpaqueBatchedVisibleMdxIndex];
        return ResolveVisibleMdxRenderer(frame, visible.Instance.ModelKey);
    }

    private static void AccumulateWmoRenderStats(WorldRenderFrame frame, WmoRenderStats stats, in WmoAdmissionTally admission)
    {
        frame.WmoDrawCallCount += stats.DrawCalls;
        frame.WmoBatchDrawCallCount += stats.BatchDrawCalls;
        frame.WmoOpaqueBatchInstanceCount += stats.OpaqueBatchInstanceCount;
        frame.WmoGroupFallbackDrawCallCount += stats.GroupFallbackDrawCalls;
        frame.WmoLiquidDrawCallCount += stats.LiquidDrawCalls;
        frame.WmoDoodadSubmissionCount += stats.DoodadSubmissions;
        frame.WmoVisibleGroupSubmissionCount += stats.VisibleGroupSubmissions;
        frame.WmoAdmission.Add(admission);
    }

    private void TrackPendingVisibleLoad(Dictionary<string, float> pendingLoads, string modelKey, float distanceSq)
    {
        if (pendingLoads.TryGetValue(modelKey, out float existingDistanceSq) && existingDistanceSq <= distanceSq)
            return;

        pendingLoads[modelKey] = distanceSq;
    }

    private void QueueTileAssetLoads(List<ObjectInstance> tileMdx, List<ObjectInstance> tileSkyboxes, List<ObjectInstance> tileWmo)
    {
        for (int i = 0; i < tileWmo.Count; i++)
            _assets.QueueWmoLoad(tileWmo[i].ModelKey);

        for (int i = 0; i < tileMdx.Count; i++)
            _assets.QueueMdxLoad(tileMdx[i].ModelKey);

        for (int i = 0; i < tileSkyboxes.Count; i++)
            _assets.QueueMdxLoad(tileSkyboxes[i].ModelKey);
    }

    private void QueueNearFieldWmoAssetLoads()
    {
        int cameraTileX = _terrainManager.CameraTileX;
        int cameraTileY = _terrainManager.CameraTileY;
        if (cameraTileX < 0 || cameraTileY < 0)
            return;

        IReadOnlyList<(int tileX, int tileY)> retainedTiles = _terrainManager.LastRetainedTiles;
        for (int tileIndex = 0; tileIndex < retainedTiles.Count; tileIndex++)
        {
            (int tileX, int tileY) tile = retainedTiles[tileIndex];
            if (Math.Abs(tile.tileX - cameraTileX) > 1 || Math.Abs(tile.tileY - cameraTileY) > 1)
                continue;

            if (!_tileWmoInstances.TryGetValue(tile, out List<ObjectInstance>? wmoInstances))
                continue;

            for (int instanceIndex = 0; instanceIndex < wmoInstances.Count; instanceIndex++)
                _assets.PrioritizeWmoLoad(wmoInstances[instanceIndex].ModelKey);
        }
    }

    /// <summary>
    /// Queue every unique world asset referenced by the supplied resident tiles.
    /// This is the capture-path warmup seam: it uses the same placement lists and
    /// asset queues as normal streaming, but does not make the render path visit
    /// or submit the objects early.
    /// </summary>
    public void QueueCapturePreloadAssets(IEnumerable<(int tileX, int tileY)> tiles)
    {
        ArgumentNullException.ThrowIfNull(tiles);

        _capturePreloadTiles.Clear();
        foreach (var tile in tiles)
        {
            _capturePreloadTiles.Add(tile);
            if (_tileMdxInstances.TryGetValue(tile, out List<ObjectInstance>? mdx)
                && _tileSkyboxInstances.TryGetValue(tile, out List<ObjectInstance>? skyboxes)
                && _tileWmoInstances.TryGetValue(tile, out List<ObjectInstance>? wmo)
                )
            {
                QueueTileAssetLoads(mdx, skyboxes, wmo);
                continue;
            }

            if (_tileMdxInstances.TryGetValue(tile, out mdx))
                QueueTileAssetLoads(mdx, [], []);
            if (_tileSkyboxInstances.TryGetValue(tile, out skyboxes))
                QueueTileAssetLoads([], skyboxes, []);
            if (_tileWmoInstances.TryGetValue(tile, out wmo))
                QueueTileAssetLoads([], [], wmo);
        }
    }

    private void FlushPendingVisibleMdxLoads()
    {
        if (_pendingVisibleMdxLoadDistances.Count == 0)
            return;

        _pendingVisibleMdxLoadScratch.Clear();
        _pendingVisibleMdxLoadScratch.AddRange(_pendingVisibleMdxLoadDistances);
        _pendingVisibleMdxLoadScratch.Sort((left, right) => left.Value.CompareTo(right.Value));

        // Bound the backlog. The priority queue is a FIFO, so promoting more per frame than the
        // loader drains turns it into a stale insertion-ordered list: at 12 in and as few as 1 out
        // (the CPU throttle clamps to 1 whenever the previous frame exceeded 33 ms), an entry pops
        // many frames after it was queued, when the camera has moved on. That is what makes distant
        // objects appear before near ones. Anything not promoted this frame is re-derived from
        // visibility next frame and reconsidered against its distance then, so nothing is lost.
        int budget = Math.Min(
            _assetLoadPolicy.MaxNewMdxLoadsPerFrame,
            Math.Max(0, _assetLoadPolicy.MaxPriorityLoadBacklog - _assets.PriorityMdxLoadCount));

        int queued = 0;
        for (int i = 0; i < _pendingVisibleMdxLoadScratch.Count && queued < budget; i++)
        {
            _assets.PrioritizeMdxLoad(_pendingVisibleMdxLoadScratch[i].Key);
            queued++;
        }
    }

    private void FlushPendingVisibleWmoLoads()
    {
        if (_pendingVisibleWmoLoadDistances.Count == 0)
            return;

        _pendingVisibleWmoLoadScratch.Clear();
        _pendingVisibleWmoLoadScratch.AddRange(_pendingVisibleWmoLoadDistances);
        _pendingVisibleWmoLoadScratch.Sort((left, right) => left.Value.CompareTo(right.Value));

        int budget = Math.Min(
            _assetLoadPolicy.MaxNewWmoLoadsPerFrame,
            Math.Max(0, _assetLoadPolicy.MaxPriorityLoadBacklog - _assets.PriorityWmoLoadCount));

        int queued = 0;
        for (int i = 0; i < _pendingVisibleWmoLoadScratch.Count && queued < budget; i++)
        {
            _assets.PrioritizeWmoLoad(_pendingVisibleWmoLoadScratch[i].Key);
            queued++;
        }
    }

    private void ProcessDeferredAssetLoads()
    {
        int pendingLoadCount = _assets.PendingAssetLoadCount;
        int maxLoads = _assetLoadPolicy.MaxDeferredLoadsPerFrame;
        double maxBudgetMs = _assetLoadPolicy.MaxDeferredLoadBudgetMs;

        double previousFrameCpuMs = LastRenderFrameStats.TotalCpuMs;
        if (previousFrameCpuMs >= 33.0)
        {
            maxLoads = Math.Min(maxLoads, 1);
            maxBudgetMs = Math.Min(maxBudgetMs, 1.0);
        }
        else if (previousFrameCpuMs >= 20.0)
        {
            maxLoads = Math.Min(maxLoads, 2);
            maxBudgetMs = Math.Min(maxBudgetMs, 1.5);
        }

        if (!_assetLoadPolicy.PrewarmTileAssets && previousFrameCpuMs < 20.0)
        {
            if (pendingLoadCount >= 96)
            {
                maxLoads = Math.Max(maxLoads, 6);
                maxBudgetMs = Math.Max(maxBudgetMs, 4.0);
            }
            else if (pendingLoadCount >= 32)
            {
                maxLoads = Math.Max(maxLoads, 5);
                maxBudgetMs = Math.Max(maxBudgetMs, 3.0);
            }
        }

        if (_assetLoadPolicy.PrewarmTileAssets)
        {
            if (pendingLoadCount >= 96)
            {
                maxLoads = Math.Max(maxLoads, 16);
                maxBudgetMs = Math.Max(maxBudgetMs, 18.0);
            }
            else if (pendingLoadCount >= 32)
            {
                maxLoads = Math.Max(maxLoads, 12);
                maxBudgetMs = Math.Max(maxBudgetMs, 14.0);
            }
        }

        if (CapturePreloadActive)
        {
            maxLoads = Math.Max(maxLoads, 24);
            maxBudgetMs = Math.Max(maxBudgetMs, 16.0);
        }

        int processed = _assets.ProcessPendingLoads(maxLoads, maxBudgetMs);
        if (processed > 0)
        {
            bool flatVisibilityBucketsDirty = false;
            foreach (var pair in _tileMdxInstances)
            {
                if (!IsActiveObjectTile(pair.Key) && !_capturePreloadTiles.Contains(pair.Key))
                    continue;

                if (!RefreshMdxInstanceBounds(pair.Value, pair.Key, isSkybox: false, isExternal: false))
                    continue;

                UpdateObjectBucketBounds(_tileMdxBounds, pair.Key, pair.Value);
                flatVisibilityBucketsDirty = true;
            }

            foreach (var pair in _tileSkyboxInstances)
                RefreshMdxInstanceBounds(pair.Value, pair.Key, isSkybox: true, isExternal: false);

            foreach (var pair in _tileWmoInstances)
            {
                if (!IsActiveObjectTile(pair.Key) && !_capturePreloadTiles.Contains(pair.Key))
                    continue;

                if (!RefreshWmoInstanceBounds(pair.Value, pair.Key, isSkybox: false, isExternal: false))
                    continue;

                UpdateObjectBucketBounds(_tileWmoBounds, pair.Key, pair.Value);
                flatVisibilityBucketsDirty = true;
            }

            RefreshMdxInstanceBounds(_externalMdxInstances, tileKey: null, isSkybox: false, isExternal: true);
            RefreshMdxInstanceBounds(_externalSkyboxInstances, tileKey: null, isSkybox: true, isExternal: true);
            RefreshWmoInstanceBounds(_externalWmoInstances, tileKey: null, isSkybox: false, isExternal: true);

            if (flatVisibilityBucketsDirty)
                RebuildFlatVisibilityBuckets();
        }

        if (CapturePreloadActive)
        {
            _assets.ProcessDeferredWmoDoodadLoads(maxLoads: 24, maxBudgetMs: 12.0);
            _assets.ProcessDeferredWmoMaterialTextureLoads(maxLoads: 24, maxBudgetMs: 12.0);
        }
    }

    private static void UpdateObjectBucketBounds(
        Dictionary<(int, int), (Vector3 Min, Vector3 Max)> boundsByTile,
        (int, int) tileKey,
        IReadOnlyList<ObjectInstance> instances)
    {
        if (instances.Count == 0)
        {
            boundsByTile.Remove(tileKey);
            return;
        }

        Vector3 min = new(float.MaxValue);
        Vector3 max = new(float.MinValue);
        for (int i = 0; i < instances.Count; i++)
        {
            ObjectInstance instance = instances[i];
            min = Vector3.Min(min, instance.BoundsMin);
            max = Vector3.Max(max, instance.BoundsMax);
        }

        boundsByTile[tileKey] = (min, max);
    }

    private static bool AreMdxTileBoundsResolved(IReadOnlyList<ObjectInstance> instances)
    {
        for (int i = 0; i < instances.Count; i++)
        {
            if (!instances[i].BoundsResolved)
                return false;
        }

        return true;
    }

    private bool ShouldVisitObjectBucket(
        Vector3 bucketMin,
        Vector3 bucketMax,
        Vector3 cameraPos,
        Vector3 cameraForward,
        float fogEnd,
        bool isWmo,
        bool countAsTaxiActor)
    {
        float boundsDistSq = DistanceSquaredPointToAabb(cameraPos, bucketMin, bucketMax);
        float noCullDistanceSq = ComputeNoCullDistanceSq(bucketMin, bucketMax);
        bool frustumVisible = _frustumCuller.TestAABB(bucketMin, bucketMax);
        Vector3 bucketCenter = (bucketMin + bucketMax) * 0.5f;
        float centerDistanceSq = Vector3.DistanceSquared(cameraPos, bucketCenter);
        // Active-tile admission already bounds object residency. Do not shrink
        // resident object buckets by the camera cone; heading remains an
        // asset-load priority signal, not a second visibility gate.
        float loadConeFactor = ComputeVisionConeFactor(cameraPos, cameraForward, bucketCenter, centerDistanceSq);

        if (boundsDistSq > noCullDistanceSq && !frustumVisible && loadConeFactor < MinOffFrustumConeFactor)
            return false;

        float bucketDiagonal = (bucketMax - bucketMin).Length();
        float baseCullDistance = isWmo
            ? ComputeWmoCullDistance(fogEnd, _objectStreamingRangeMultiplier)
            : ComputeMdxCullDistance(fogEnd, bucketDiagonal, countAsTaxiActor, _objectStreamingRangeMultiplier);
        if (boundsDistSq > baseCullDistance * baseCullDistance)
            return false;

        return centerDistanceSq <= MaxWorldObjectViewDistanceSq;
    }

    private bool ShouldVisitFlatVisibilityBucket(
        FlatVisibilityBucket bucket,
        Vector3 cameraPos,
        float fogEnd,
        bool isWmo)
    {
        // Unresolved or malformed members remain fail-open. The bucket is only a coarse
        // accelerator and must never become a second correctness culler.
        if (!bucket.BoundsKnown || bucket.Instances.Count == 0)
            return true;

        float boundsDistSq = DistanceSquaredPointToAabb(cameraPos, bucket.Min, bucket.Max);
        bool frustumVisible = _frustumCuller.TestAABB(bucket.Min, bucket.Max);
        if (!frustumVisible && boundsDistSq > ComputeNoCullDistanceSq(bucket.Min, bucket.Max))
            return false;

        float bucketDiagonal = (bucket.Max - bucket.Min).Length();
        float baseCullDistance = isWmo
            ? ComputeWmoCullDistance(fogEnd, _objectStreamingRangeMultiplier)
            : ComputeMdxCullDistance(fogEnd, bucketDiagonal, isTaxiActor: false, _objectStreamingRangeMultiplier);
        if (boundsDistSq > baseCullDistance * baseCullDistance)
            return false;

        return boundsDistSq <= MaxWorldObjectViewDistanceSq;
    }

    /// <summary>
    /// Resolve the tight, geometry-derived selection bounds for a model, falling back to the
    /// supplied culling bounds when no tighter box is available.
    /// </summary>
    /// <remarks>
    /// For native MDX the two are the same box, so this is a no-op there. It exists for M2, whose
    /// culling bounds are a declared animation/collision extent much larger than the mesh.
    /// </remarks>
    private void ResolveMdxSelectionBounds(
        string modelKey,
        Vector3 fallbackMin,
        Vector3 fallbackMax,
        bool fallbackResolved,
        out Vector3 selectionMin,
        out Vector3 selectionMax,
        out bool selectionResolved)
    {
        if (_assets.TryGetMdxSelectionBounds(modelKey, out selectionMin, out selectionMax)
            && AreFiniteOrderedBounds(selectionMin, selectionMax))
        {
            selectionResolved = true;
            return;
        }

        selectionMin = fallbackMin;
        selectionMax = fallbackMax;
        selectionResolved = fallbackResolved && AreFiniteOrderedBounds(fallbackMin, fallbackMax);
    }

    private bool RefreshMdxInstanceBounds(
        List<ObjectInstance> instances,
        (int tileX, int tileY)? tileKey,
        bool isSkybox,
        bool isExternal)
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

            // Selection and picking use the tight geometry bounds; culling above keeps the
            // conservative ones. For M2 the two differ substantially because the declared header
            // extent is an animation/collision volume, not the mesh.
            ResolveMdxSelectionBounds(
                inst.ModelKey, localMin, localMax, fallbackResolved: true,
                out Vector3 selectionMin, out Vector3 selectionMax, out bool selectionResolved);
            inst.SelectionLocalBoundsMin = selectionMin;
            inst.SelectionLocalBoundsMax = selectionMax;
            inst.SelectionBoundsResolved = selectionResolved;

            instances[i] = inst;
            UpdateSceneGraphPlacementBounds(tileKey, WorldSceneNodeKind.M2Placement, i, inst, isSkybox, isExternal);
            changed = true;
        }

        return changed;
    }

    private bool RefreshWmoInstanceBounds(
        List<ObjectInstance> instances,
        (int tileX, int tileY)? tileKey,
        bool isSkybox,
        bool isExternal)
    {
        bool changed = false;

        for (int i = 0; i < instances.Count; i++)
        {
            var inst = instances[i];
            if (inst.BoundsResolved)
                continue;

            if (!_assets.TryGetWmoPlacementBounds(inst.ModelKey, out var localMin, out var localMax))
                continue;

            TransformBounds(localMin, localMax, inst.Transform, out var worldMin, out var worldMax);
            inst.LocalBoundsMin = localMin;
            inst.LocalBoundsMax = localMax;
            inst.BoundsMin = worldMin;
            inst.BoundsMax = worldMax;
            inst.BoundsResolved = true;

            // A WMO's placement bounds already describe the building itself, so there is no tighter
            // source to switch to. It still benefits from being drawn oriented rather than re-fitted
            // to the world axes.
            inst.SelectionLocalBoundsMin = localMin;
            inst.SelectionLocalBoundsMax = localMax;
            inst.SelectionBoundsResolved = true;

            instances[i] = inst;
            UpdateSceneGraphPlacementBounds(tileKey, WorldSceneNodeKind.WmoPlacement, i, inst, isSkybox, isExternal);
            changed = true;
        }

        return changed;
    }

    private void UpdateSceneGraphPlacementBounds(
        (int tileX, int tileY)? tileKey,
        WorldSceneNodeKind kind,
        int instanceIndex,
        in ObjectInstance instance,
        bool isSkybox,
        bool isExternal)
    {
        if (!UseHierarchicalSceneTraversal || _sceneGraphBuild is null)
            return;

        string kindToken = GetSceneGraphKindToken(kind, isSkybox);
        string sourceToken = isExternal
            ? "external"
            : tileKey.HasValue
                ? $"tile/{tileKey.Value.tileX:D2}/{tileKey.Value.tileY:D2}"
                : string.Empty;
        if (string.IsNullOrEmpty(sourceToken))
            return;

        string placementId = $"world/object/{kindToken}/{sourceToken}/{instanceIndex:D6}";
        if (!_sceneGraphBuild.TryGetGraphForPlacement(placementId, out WorldSceneGraphBuildResult? graph)
            || graph is null
            || !graph.Graph.TryGetNode(placementId, out WorldSceneNode? node)
            || node is null)
        {
            return;
        }

        // Bounds promotion changes the placement payload, not the scene topology. Update only
        // this placement node; the streaming-safe node API deliberately avoids refreshing every
        // sibling branch, and the authoritative tile root remains unchanged.
        graph.TryUpdatePlacementInstance(placementId, instance);
        node.UpdateLocalBoundsForStreaming(instance.LocalBoundsMin, instance.LocalBoundsMax, instance.BoundsResolved);
    }

    private static int FindPlacementInstanceIndex(List<ObjectInstance> instances, ObjectInstance current)
    {
        for (int index = 0; index < instances.Count; index++)
        {
            ObjectInstance candidate = instances[index];
            if (candidate.UniqueId == current.UniqueId
                && candidate.PlacementEntryIndex == current.PlacementEntryIndex
                && candidate.TileX == current.TileX
                && candidate.TileY == current.TileY)
            {
                return index;
            }
        }

        return -1;
    }

    private ObjectInstance MovePlacementInstance(ObjectInstance current, Vector3 newPosition, ObjectType objectType)
    {
        Vector3 delta = newPosition - current.PlacementPosition;
        current.PlacementPosition = newPosition;

        switch (objectType)
        {
            case ObjectType.Mdx:
            {
                var transform = WorldPlacementTransform.Build(
                    newPosition,
                    current.PlacementRotation,
                    current.PlacementScale);

                current.Transform = transform;
                if (_assets.TryGetMdxBounds(current.ModelKey, out Vector3 localMin, out Vector3 localMax))
                {
                    current.LocalBoundsMin = localMin;
                    current.LocalBoundsMax = localMax;
                    current.BoundsResolved = true;
                    TransformBounds(localMin, localMax, transform, out Vector3 worldMin, out Vector3 worldMax);
                    current.BoundsMin = worldMin;
                    current.BoundsMax = worldMax;
                }
                else
                {
                    current.BoundsMin += delta;
                    current.BoundsMax += delta;
                    current.BoundsResolved = false;
                }

                return current;
            }

            case ObjectType.Wmo:
            {
                var transform = WorldPlacementTransform.Build(
                    newPosition,
                    current.PlacementRotation);

                current.Transform = transform;
                if (_assets.TryGetWmoPlacementBounds(current.ModelKey, out Vector3 localMin, out Vector3 localMax))
                {
                    current.LocalBoundsMin = localMin;
                    current.LocalBoundsMax = localMax;
                    current.BoundsResolved = true;
                    TransformBounds(localMin, localMax, transform, out Vector3 worldMin, out Vector3 worldMax);
                    current.BoundsMin = worldMin;
                    current.BoundsMax = worldMax;
                }
                else
                {
                    current.BoundsMin += delta;
                    current.BoundsMax += delta;
                    current.BoundsResolved = false;
                }

                return current;
            }

            default:
                return current;
        }
    }

    private void UpdateAdapterPlacementPosition(ObjectType objectType, ObjectInstance current, Vector3 newPosition)
    {
        switch (objectType)
        {
            case ObjectType.Mdx:
                UpdateMddfPlacementPosition(current, newPosition);
                break;

            case ObjectType.Wmo:
                UpdateModfPlacementPosition(current, newPosition);
                break;
        }
    }

    private void UpdateMddfPlacementPosition(ObjectInstance current, Vector3 newPosition)
    {
        List<MddfPlacement> placements = _terrainManager.Adapter.MddfPlacements;
        int index = FindPlacementIndexByUniqueIdAndPosition(placements, current.UniqueId, current.PlacementPosition);
        if (index < 0)
            index = FindPlacementIndexByUniqueId(placements, current.UniqueId);
        if (index < 0)
            return;

        MddfPlacement updated = placements[index];
        updated.Position = newPosition;
        placements[index] = updated;
    }

    private void UpdateModfPlacementPosition(ObjectInstance current, Vector3 newPosition)
    {
        List<ModfPlacement> placements = _terrainManager.Adapter.ModfPlacements;
        int index = FindPlacementIndexByUniqueIdAndPosition(placements, current.UniqueId, current.PlacementPosition);
        if (index < 0)
            index = FindPlacementIndexByUniqueId(placements, current.UniqueId);
        if (index < 0)
            return;

        ModfPlacement updated = placements[index];
        Vector3 delta = newPosition - updated.Position;
        updated.Position = newPosition;
        updated.BoundsMin += delta;
        updated.BoundsMax += delta;
        placements[index] = updated;
    }

    private static int FindPlacementIndexByUniqueIdAndPosition(List<MddfPlacement> placements, int uniqueId, Vector3 position)
    {
        for (int index = 0; index < placements.Count; index++)
        {
            if (placements[index].UniqueId == uniqueId && Vector3.DistanceSquared(placements[index].Position, position) < 0.0001f)
                return index;
        }

        return -1;
    }

    private static int FindPlacementIndexByUniqueId(List<MddfPlacement> placements, int uniqueId)
    {
        for (int index = 0; index < placements.Count; index++)
        {
            if (placements[index].UniqueId == uniqueId)
                return index;
        }

        return -1;
    }

    private static int FindPlacementIndexByUniqueIdAndPosition(List<ModfPlacement> placements, int uniqueId, Vector3 position)
    {
        for (int index = 0; index < placements.Count; index++)
        {
            if (placements[index].UniqueId == uniqueId && Vector3.DistanceSquared(placements[index].Position, position) < 0.0001f)
                return index;
        }

        return -1;
    }

    private static int FindPlacementIndexByUniqueId(List<ModfPlacement> placements, int uniqueId)
    {
        for (int index = 0; index < placements.Count; index++)
        {
            if (placements[index].UniqueId == uniqueId)
                return index;
        }

        return -1;
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
            var (tileX, tileY) = ComputeTileCoordinates(pos);

            if (isWmo)
            {
                var transform = Matrix4x4.CreateRotationZ(finalYawRadians)
                    * Matrix4x4.CreateTranslation(pos);

                Vector3 localMin, localMax, worldMin, worldMax;
                if (_assets.TryGetWmoPlacementBounds(key, out localMin, out localMax))
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
                    UniqueId = spawn.SpawnId,
                    PlacementEntryIndex = -1,
                    TileX = tileX,
                    TileY = tileY,
                    HasTileCoordinate = true
                });
            }
            else
            {
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
                    UniqueId = spawn.SpawnId,
                    PlacementEntryIndex = -1,
                    TileX = tileX,
                    TileY = tileY,
                    HasTileCoordinate = true
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
    internal static void TransformBounds(Vector3 min, Vector3 max, Matrix4x4 m, out Vector3 outMin, out Vector3 outMax)
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

    private static float DistanceSquaredPointToAabb(Vector3 point, Vector3 min, Vector3 max)
    {
        float dx = point.X < min.X ? min.X - point.X : point.X > max.X ? point.X - max.X : 0f;
        float dy = point.Y < min.Y ? min.Y - point.Y : point.Y > max.Y ? point.Y - max.Y : 0f;
        float dz = point.Z < min.Z ? min.Z - point.Z : point.Z > max.Z ? point.Z - max.Z : 0f;
        return dx * dx + dy * dy + dz * dz;
    }

    private static float ComputeNoCullDistanceSq(Vector3 min, Vector3 max)
    {
        float halfDiagonal = (max - min).Length() * 0.5f;
        float graceRadius = MathF.Max(NoCullRadius, MathF.Min(halfDiagonal + 96f, 1024f));
        return graceRadius * graceRadius;
    }

    private static float ComputeObjectFogStart(float fogStart, float fogEnd)
    {
        if (fogEnd <= 0f)
            return fogStart;

        float delayedStart = fogEnd * 0.6f;
        return MathF.Min(fogEnd - 64f, MathF.Max(fogStart, delayedStart));
    }

    private static (float start, float end) ComputeObjectFogRange(float fogStart, float fogEnd, bool enabled)
    {
        if (enabled)
            return (ComputeObjectFogStart(fogStart, fogEnd), fogEnd);

        float disabledStart = MathF.Max(fogEnd, fogStart) + 100000f;
        return (disabledStart, disabledStart + 1f);
    }

    private static float ComputeWmoCullDistance(float fogEnd, float rangeMultiplier)
    {
        float clampedMultiplier = Math.Clamp(rangeMultiplier, 0.25f, 4.0f);
        if (fogEnd <= 0f)
            return MathF.Min(MaxWorldObjectViewDistance, MathF.Min(WmoCullDistance, MaxWorldObjectViewDistance) * clampedMultiplier);

        float baseDistance = MathF.Min(MaxWorldObjectViewDistance, MathF.Max(WmoCullDistance, fogEnd + 256f));
        return MathF.Min(MaxWorldObjectViewDistance, baseDistance * clampedMultiplier);
    }

    private static float ComputeMdxCullDistance(float fogEnd, float boundsDiagonal, bool isTaxiActor, float rangeMultiplier)
    {
        float clampedMultiplier = Math.Clamp(rangeMultiplier, 0.25f, 4.0f);
        if (isTaxiActor)
            return MathF.Min(MaxWorldObjectViewDistance, MathF.Max(1024f, fogEnd + 384f) * clampedMultiplier);

        if (fogEnd <= 0f)
            return MathF.Min(MaxWorldObjectViewDistance, MathF.Min(DoodadCullDistance, MaxWorldObjectViewDistance) * clampedMultiplier);

        float objectAllowance = MathF.Min(512f, boundsDiagonal * 0.5f + 96f);
        float baseDistance = MathF.Min(DoodadCullDistance, MathF.Max(1024f, fogEnd + objectAllowance));
        return MathF.Min(MaxWorldObjectViewDistance, baseDistance * clampedMultiplier);
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

    private bool ShouldHideObjectInstanceByUniqueId(in ObjectInstance inst)
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

    private bool ShouldHideVisibleMdxInstance(in ObjectInstance inst)
    {
        if (ShouldHideObjectInstanceByUniqueId(inst))
            return true;

        if (_maxVisibleMdxBoundsHeight > 0f)
        {
            float boundsHeight = MathF.Abs(inst.BoundsMax.Z - inst.BoundsMin.Z);
            if (float.IsFinite(boundsHeight) && boundsHeight > _maxVisibleMdxBoundsHeight)
                return true;
        }

        return _hideTerrainOccludedMdx && IsMdxFullyOccludedByTerrain(inst);
    }

    private bool IsMdxFullyOccludedByTerrain(in ObjectInstance inst)
    {
        if (!TrySampleLoadedTerrainHeight(inst.PlacementPosition.X, inst.PlacementPosition.Y, out float terrainHeight))
            return false;

        float objectTop = MathF.Max(inst.BoundsMin.Z, inst.BoundsMax.Z);
        if (!float.IsFinite(objectTop))
            return false;

        const float terrainOcclusionMargin = 1.0f;
        return terrainHeight >= objectTop + terrainOcclusionMargin;
    }

    private void RebuildSceneLights(WorldRenderFrame frame)
    {
        _sceneLightManager.Clear();
        _sceneLightCollectScratch.Clear();

        // Spec 236 T010: publish the frame outdoor ambient/sun representation alongside the
        // emitted point lights so consumers share one base-lighting contract.
        TerrainLighting baseLighting = _terrainManager.Lighting;
        _sceneLightManager.SetAmbient(baseLighting.LightDirection, baseLighting.LightColor, baseLighting.AmbientColor);

        for (int i = 0; i < frame.Visibility.VisibleWmos.Count; i++)
        {
            VisibleWmoInstance visible = frame.Visibility.VisibleWmos[i];
            WmoRenderer? renderer = ResolveVisibleWmoRenderer(frame, visible.Instance.ModelKey);
            renderer?.CollectSceneLights(visible.Instance.Transform, _sceneLightCollectScratch, visible.Instance.ModelKey);
        }

        for (int i = 0; i < _mdxInstances.Count; i++)
        {
            ObjectInstance instance = _mdxInstances[i];
            if (ShouldHideVisibleMdxInstance(instance))
                continue;

            IModelRenderer? renderer = ResolveVisibleMdxRenderer(frame, instance.ModelKey);
            if (renderer is not ISceneLightEmitter emitter)
                continue;

            emitter.CollectSceneLights(instance.Transform, _sceneLightCollectScratch, instance.ModelKey);
        }

        // Epic 249 R-10c: a light whose sphere lies wholly outside the view (side + near planes)
        // cannot light a drawn pixel. The margin covers terrain, which renders before this rebuild
        // and so uses the previous frame's set while the camera moves.
        _sceneLightManager.AddRange(_sceneLightCollectScratch,
            light => _frustumCuller.TestSphereIgnoringFarPlane(light.Position, light.AttenuationEnd + SceneLightManager.FrustumCullMargin));
        _sceneLightCollectScratch.Clear();
    }

    /// <summary>
    /// Resolves a camera-path sample against the loaded world. Terrain collision is
    /// heightfield-only; WMO collision uses the resident placement bounds as a
    /// conservative sweep volume. Both are deliberately opt-in because the viewer
    /// also supports free-fly inspection through geometry.
    /// </summary>
    public bool TryResolveCameraPathCollision(
        Vector3 previousPosition,
        Vector3 desiredPosition,
        float clearance,
        bool terrainCollision,
        bool wmoCollision,
        out Vector3 resolvedPosition)
    {
        resolvedPosition = desiredPosition;
        float safeClearance = float.IsFinite(clearance) ? Math.Clamp(clearance, 0f, 32f) : 0f;
        bool collided = false;

        if (terrainCollision && TrySampleLoadedTerrainHeight(desiredPosition.X, desiredPosition.Y, out float terrainHeight))
        {
            float minimumCameraZ = terrainHeight + safeClearance;
            if (resolvedPosition.Z < minimumCameraZ)
            {
                resolvedPosition.Z = minimumCameraZ;
                collided = true;
            }
        }

        if (wmoCollision)
        {
            if (_instancesDirty)
                RebuildInstanceLists();

            Vector3 segmentStart = previousPosition;
            Vector3 segmentEnd = resolvedPosition;
            foreach (ObjectInstance instance in _wmoInstances)
            {
                if (!AreFiniteOrderedBounds(instance.BoundsMin, instance.BoundsMax))
                    continue;

                Vector3 boundsMin = instance.BoundsMin - new Vector3(safeClearance);
                Vector3 boundsMax = instance.BoundsMax + new Vector3(safeClearance);
                if (!TrySegmentAabb(segmentStart, segmentEnd, boundsMin, boundsMax, out float entryT))
                    continue;

                bool startInside = IsPointInsideAabb(segmentStart, boundsMin, boundsMax);
                // A placement AABB is an exterior shell, not an indoor collision mesh.
                // Preserve paths that start inside a WMO instead of ejecting them from
                // the entire building; only stop an outside-to-inside sweep here.
                if (startInside)
                    continue;

                if (entryT > 0f)
                {
                    float stopT = Math.Clamp(entryT - 0.0025f, 0f, 1f);
                    resolvedPosition = Vector3.Lerp(segmentStart, segmentEnd, stopT);
                }
                else if (IsPointInsideAabb(segmentEnd, boundsMin, boundsMax))
                    resolvedPosition = segmentStart;

                collided = true;
                segmentEnd = resolvedPosition;
            }
        }

        return collided;
    }

    private static bool IsPointInsideAabb(Vector3 point, Vector3 min, Vector3 max)
        => point.X >= min.X && point.X <= max.X
            && point.Y >= min.Y && point.Y <= max.Y
            && point.Z >= min.Z && point.Z <= max.Z;

    private static bool TrySegmentAabb(Vector3 start, Vector3 end, Vector3 min, Vector3 max, out float entryT)
    {
        entryT = 0f;
        float exitT = 1f;
        Vector3 delta = end - start;
        for (int axis = 0; axis < 3; axis++)
        {
            float origin = start[axis];
            float direction = delta[axis];
            float axisMin = min[axis];
            float axisMax = max[axis];
            if (MathF.Abs(direction) < 0.000001f)
            {
                if (origin < axisMin || origin > axisMax)
                    return false;
                continue;
            }

            float inverse = 1f / direction;
            float near = (axisMin - origin) * inverse;
            float far = (axisMax - origin) * inverse;
            if (near > far)
                (near, far) = (far, near);
            entryT = MathF.Max(entryT, near);
            exitT = MathF.Min(exitT, far);
            if (entryT > exitT)
                return false;
        }

        return entryT >= 0f && entryT <= 1f;
    }

    private bool TrySampleLoadedTerrainHeight(float worldX, float worldY, out float height)
    {
        height = 0f;

        return TrySampleLoadedTerrainHeight(_terrainManager, _terrainManager.Renderer, worldX, worldY, out height);
    }

    private static bool TrySampleLoadedTerrainHeight(TerrainManager terrainManager, TerrainRenderer renderer, float worldX, float worldY, out float height)
    {
        height = 0f;

        TerrainRenderer.TerrainChunkInfo? chunkInfo = renderer.GetChunkInfoAt(worldX, worldY);
        if (!chunkInfo.HasValue)
            return false;

        if (!terrainManager.TryGetTileLoadResult(chunkInfo.Value.TileX, chunkInfo.Value.TileY, out TileLoadResult tile))
            return false;

        TerrainChunkData? chunk = tile.Chunks.FirstOrDefault(c => c.ChunkX == chunkInfo.Value.ChunkX && c.ChunkY == chunkInfo.Value.ChunkY);
        if (chunk == null || chunk.Heights == null || chunk.Heights.Length < 145)
            return false;

        float localX = chunk.WorldPosition.Y - worldY;
        float localY = chunk.WorldPosition.X - worldX;
        localX = Math.Clamp(localX, 0f, WoWConstants.ChunkSize);
        localY = Math.Clamp(localY, 0f, WoWConstants.ChunkSize);
        height = SampleHeightOuterGrid(chunk, localX, localY);
        return true;
    }

    private static float SampleHeightOuterGrid(TerrainChunkData chunk, float localX, float localY)
    {
        if (chunk.Heights == null || chunk.Heights.Length < 145)
            return chunk.WorldPosition.Z;

        float cellSize = WoWConstants.ChunkSize / 16f;
        float subCellSize = cellSize / 8f;

        Span<float> grid = stackalloc float[9 * 9];
        grid.Clear();

        for (int i = 0; i < 145; i++)
        {
            GetChunkVertexPosition(i, out int row, out int col, out bool isInner);
            if (isInner)
                continue;

            int gridY = row / 2;
            if ((uint)gridY >= 9u || (uint)col >= 9u)
                continue;

            grid[(gridY * 9) + col] = chunk.Heights[i];
        }

        float gridX = localX / subCellSize;
        float gridYFloat = localY / subCellSize;
        int ix = Math.Clamp((int)MathF.Floor(gridX), 0, 7);
        int iy = Math.Clamp((int)MathF.Floor(gridYFloat), 0, 7);
        float fx = Math.Clamp(gridX - ix, 0f, 1f);
        float fy = Math.Clamp(gridYFloat - iy, 0f, 1f);

        float h00 = grid[(iy * 9) + ix];
        float h10 = grid[(iy * 9) + (ix + 1)];
        float h01 = grid[((iy + 1) * 9) + ix];
        float h11 = grid[((iy + 1) * 9) + (ix + 1)];

        float h0 = h00 + ((h10 - h00) * fx);
        float h1 = h01 + ((h11 - h01) * fx);
        return h0 + ((h1 - h0) * fy);
    }

    private static void GetChunkVertexPosition(int index, out int row, out int col, out bool isInner)
    {
        int remaining = index;
        row = 0;
        col = 0;
        isInner = false;

        for (int currentRow = 0; currentRow < 17; currentRow++)
        {
            int rowSize = (currentRow % 2 == 0) ? 9 : 8;
            if (remaining < rowSize)
            {
                row = currentRow;
                col = remaining;
                isInner = (currentRow % 2 == 1);
                return;
            }

            remaining -= rowSize;
        }
    }

    private static (int tileX, int tileY) ComputeTileCoordinates(Vector3 rendererPosition)
    {
        int tileX = (int)MathF.Floor((WoWConstants.MapOrigin - rendererPosition.X) / WoWConstants.ChunkSize);
        int tileY = (int)MathF.Floor((WoWConstants.MapOrigin - rendererPosition.Y) / WoWConstants.ChunkSize);
        return (tileX, tileY);
    }

    private void PrepareSceneGraphFrameVisibility(
        Vector3 cameraPos,
        Vector3 cameraForward,
        float fogEnd,
        float verticalFieldOfViewRadians)
    {
        if (_sceneGraphFrameVisibilityPrepared)
            return;

        _sceneGraphVisibleMdxInstances.Clear();
        _sceneGraphVisibleWmoInstances.Clear();
        _sceneGraphPortalVisibility.Clear();
        _lastSceneGraphTraversalDiagnostics = new WorldSceneTraversalDiagnostics();

        if (_sceneGraphBuild is null)
        {
            _sceneGraphFrameVisibilityPrepared = true;
            return;
        }

        // Reused across frames and cleared in place. These were rebuilt every frame via LINQ plus a
        // fresh HashSet, which allocated in proportion to the resident tile set on the hot path.
        List<WorldSceneGraphBuildResult> activeGraphs = _activeSceneGraphScratch;
        HashSet<WorldSceneGraphBuildResult> activeGraphSet = _activeSceneGraphSetScratch;
        activeGraphs.Clear();
        activeGraphSet.Clear();
        foreach (WorldSceneGraphBuildResult activeGraph in EnumerateActiveSceneGraphs())
        {
            activeGraphs.Add(activeGraph);
            activeGraphSet.Add(activeGraph);
        }

        foreach ((string placementId, WorldScenePortalAdapterResult adapter) in _sceneGraphPortalAdapters)
        {
            if (_sceneGraphBuild.TryGetGraphForPlacement(placementId, out WorldSceneGraphBuildResult? placementGraph)
                && activeGraphSet.Contains(placementGraph)
                && placementGraph.Graph.TryGetNode(placementId, out WorldSceneNode? placementNode))
            {
                _sceneGraphPortalVisibility[placementId] = WorldScenePortalVisibilityEvaluator.Evaluate(
                    adapter,
                    placementNode,
                    cameraPos,
                    maximumDepth: 4);
            }
        }

        foreach (WorldSceneGraphBuildResult graphBuild in activeGraphs)
        {
            // Traverse into reused buffers. The allocating overload builds two lists, a diagnostics
            // object with four dictionaries, and a result record per graph per frame, which scales
            // with the resident tile set on the hot path.
            WorldSceneTraversal.TraverseInto(
                graphBuild.Graph,
                IsSceneGraphNodeVisible,
                _sceneGraphVisibleNodeScratch,
                _sceneGraphRejectedNodeScratch,
                _sceneGraphTraversalScratchDiagnostics,
                node => node.Kind is WorldSceneNodeKind.M2Placement or WorldSceneNodeKind.WmoPlacement,
                shouldEvaluateVisibility: static node =>
                    node.Kind != WorldSceneNodeKind.M2Placement
                    || node.Parent?.Kind != WorldSceneNodeKind.Chunk,
                validateGraph: false,
                collectDetailedDiagnostics: SceneGraphDetailedDiagnosticsEnabled);
            _lastSceneGraphTraversalDiagnostics.Accumulate(_sceneGraphTraversalScratchDiagnostics);

            foreach (WorldSceneNode node in _sceneGraphVisibleNodeScratch)
            {
                if (!graphBuild.PlacementsByNodeId.TryGetValue(node.Id, out WorldSceneGraphObjectPlacement placement)
                    || placement.IsSkybox)
                {
                    continue;
                }

                if (node.Kind == WorldSceneNodeKind.WmoPlacement)
                    _sceneGraphVisibleWmoInstances.Add(placement.Instance);
                else if (node.Kind == WorldSceneNodeKind.M2Placement)
                    _sceneGraphVisibleMdxInstances.Add(placement.Instance);
            }
        }

        _sceneGraphFrameVisibilityPrepared = true;
    }

    private IEnumerable<WorldSceneGraphBuildResult> EnumerateActiveSceneGraphs()
    {
        if (_sceneGraphBuild is null)
            yield break;

        foreach (KeyValuePair<(int TileX, int TileY), WorldSceneGraphBuildResult> entry in _sceneGraphBuild.AdtGraphs)
        {
            if (IsActiveObjectTile(entry.Key))
                yield return entry.Value;
        }

        // External spawns do not have an ADT coordinate and remain visible
        // through their dedicated graph.
        if (_sceneGraphBuild.ExternalGraph is not null)
            yield return _sceneGraphBuild.ExternalGraph;
    }

    private bool IsActiveObjectTile((int tileX, int tileY) tile)
    {
        // Detailed terrain remains directional, but resident neighbor tiles
        // must also be eligible for object admission. The actual frustum and
        // object bounds tests below decide what is submitted. Restricting
        // objects to the directional list made buildings disappear as soon as
        // the camera turned, even though their tile was still resident.
        if (_terrainManager.IsTileLoaded(tile.tileX, tile.tileY)
            || WorldObjectTileAdmission.IsResident(
                _terrainManager.LastSelectedTiles,
                _terrainManager.LastRetainedTiles,
                tile))
            return true;

        // Capture warmup is an explicit render-path lease. Keep its pinned
        // tiles eligible for object admission while normal navigation remains
        // bounded by the camera-centered residency window.
        return CapturePreloadActive && _capturePreloadTiles.Contains(tile);
    }

    private bool IsSceneGraphNodeVisible(WorldSceneNode node)
    {
        // Scene-graph traversal is a placement/object admission pass. WmoRenderer
        // owns the authoritative group portal decision and fails open for groups
        // in the camera frustum; applying graph portal results here as a second
        // culler caused spotty interiors when the two volumes disagreed.
        return _frustumCuller.TestAABB(node.WorldBoundsMin, node.WorldBoundsMax);
    }

    private void CollectVisibleWmoInstances(WorldRenderFrame frame, Vector3 cameraPos, Vector3 cameraForward, float fogEnd, float verticalFieldOfViewRadians)
    {
        if (UseHierarchicalSceneTraversal && _sceneGraphBuild is not null)
        {
            PrepareSceneGraphFrameVisibility(cameraPos, cameraForward, fogEnd, verticalFieldOfViewRadians);
            WmoCulledCount += WorldObjectVisibilityCollector.CollectVisibleWmos(
                frame.Visibility,
                _sceneGraphVisibleWmoInstances,
                new WorldObjectVisibilityContext(
                    cameraPos,
                    cameraForward,
                    fogEnd,
                    _objectStreamingRangeMultiplier,
                    CullSmallDoodadsOnly: false,
                    CountAsTaxiActor: false,
                    VerticalFieldOfViewRadians: verticalFieldOfViewRadians,
                    VisibilityProfile: _objectVisibilityProfile,
                    IgnoreVisionConeCulling: true),
                inst => ShouldHideObjectInstanceByUniqueId(inst),
                (min, max) => _frustumCuller.TestAABB(min, max),
                modelKey => ResolveVisibleWmoRenderer(frame, modelKey) != null,
                (modelKey, priorityScore) => TrackPendingVisibleLoad(_pendingVisibleWmoLoadDistances, modelKey, priorityScore),
                ref frame.WmoAdmission);
            return;
        }

        var context = new WorldObjectVisibilityContext(
            cameraPos,
            cameraForward,
            fogEnd,
            _objectStreamingRangeMultiplier,
            CullSmallDoodadsOnly: false,
            CountAsTaxiActor: false,
            VerticalFieldOfViewRadians: verticalFieldOfViewRadians,
            VisibilityProfile: _objectVisibilityProfile,
            IgnoreVisionConeCulling: true);

        WmoCulledCount = 0;
        foreach (var pair in _tileWmoInstances)
        {
            if (!IsActiveObjectTile(pair.Key))
            {
                WmoCulledCount += pair.Value.Count;
                continue;
            }

            if (_tileWmoBounds.TryGetValue(pair.Key, out var bounds)
                && !ShouldVisitObjectBucket(bounds.Min, bounds.Max, cameraPos, cameraForward, fogEnd, isWmo: true, countAsTaxiActor: false))
            {
                WmoCulledCount += pair.Value.Count;
                continue;
            }

            if (!_tileWmoVisibilityBuckets.TryGetValue(pair.Key, out List<FlatVisibilityBucket>? buckets))
            {
                WmoCulledCount += WorldObjectVisibilityCollector.CollectVisibleWmos(
                    frame.Visibility,
                    pair.Value,
                    context,
                    inst => ShouldHideObjectInstanceByUniqueId(inst),
                    (min, max) => _frustumCuller.TestAABB(min, max),
                    modelKey => ResolveVisibleWmoRenderer(frame, modelKey) != null,
                    (modelKey, priorityScore) => TrackPendingVisibleLoad(_pendingVisibleWmoLoadDistances, modelKey, priorityScore),
                    ref frame.WmoAdmission);
                continue;
            }

            foreach (FlatVisibilityBucket bucket in buckets)
            {
                if (!ShouldVisitFlatVisibilityBucket(bucket, cameraPos, fogEnd, isWmo: true))
                {
                    WmoCulledCount += bucket.Instances.Count;
                    continue;
                }

                WmoCulledCount += WorldObjectVisibilityCollector.CollectVisibleWmos(
                    frame.Visibility,
                    bucket.Instances,
                    context,
                    inst => ShouldHideObjectInstanceByUniqueId(inst),
                    (min, max) => _frustumCuller.TestAABB(min, max),
                    modelKey => ResolveVisibleWmoRenderer(frame, modelKey) != null,
                    (modelKey, priorityScore) => TrackPendingVisibleLoad(_pendingVisibleWmoLoadDistances, modelKey, priorityScore),
                    ref frame.WmoAdmission);
            }
        }

        if (_externalWmoInstances.Count > 0)
        {
            WmoCulledCount += WorldObjectVisibilityCollector.CollectVisibleWmos(
                frame.Visibility,
                _externalWmoInstances,
                context,
                inst => ShouldHideObjectInstanceByUniqueId(inst),
                (min, max) => _frustumCuller.TestAABB(min, max),
                modelKey => ResolveVisibleWmoRenderer(frame, modelKey) != null,
                (modelKey, priorityScore) => TrackPendingVisibleLoad(_pendingVisibleWmoLoadDistances, modelKey, priorityScore),
                ref frame.WmoAdmission);
        }
    }

    private void CollectVisibleMdxInstances(
        WorldRenderFrame frame,
        List<ObjectInstance> instances,
        Vector3 cameraPos,
        Vector3 cameraForward,
        float fogEnd,
        float verticalFieldOfViewRadians,
        bool cullSmallDoodadsOnly,
        bool countAsTaxiActor)
    {
        MdxCulledCount += WorldObjectVisibilityCollector.CollectVisibleMdx(
            frame.Visibility,
            instances,
            new WorldObjectVisibilityContext(
                cameraPos,
                cameraForward,
                fogEnd,
                _objectStreamingRangeMultiplier,
                cullSmallDoodadsOnly,
                countAsTaxiActor,
                verticalFieldOfViewRadians,
                _objectVisibilityProfile,
                IgnoreVisionConeCulling: true),
            inst => ShouldHideVisibleMdxInstance(inst),
            (min, max) => _frustumCuller.TestAABB(min, max),
            modelKey => ResolveVisibleMdxRenderer(frame, modelKey) != null,
            (modelKey, priorityScore) => TrackPendingVisibleLoad(_pendingVisibleMdxLoadDistances, modelKey, priorityScore));
    }

    private void CollectVisibleMdxBuckets(
        WorldRenderFrame frame,
        Vector3 cameraPos,
        Vector3 cameraForward,
        float fogEnd,
        float verticalFieldOfViewRadians)
    {
        if (UseHierarchicalSceneTraversal && _sceneGraphBuild is not null)
        {
            PrepareSceneGraphFrameVisibility(cameraPos, cameraForward, fogEnd, verticalFieldOfViewRadians);
            CollectVisibleMdxInstances(
                frame,
                _sceneGraphVisibleMdxInstances,
                cameraPos,
                cameraForward,
                fogEnd,
                verticalFieldOfViewRadians,
                cullSmallDoodadsOnly: true,
                countAsTaxiActor: false);
            return;
        }

        foreach (var pair in _tileMdxInstances)
        {
            if (!IsActiveObjectTile(pair.Key))
            {
                MdxCulledCount += pair.Value.Count;
                continue;
            }

            if (_tileMdxBounds.TryGetValue(pair.Key, out var bounds)
                && AreMdxTileBoundsResolved(pair.Value)
                && !ShouldVisitObjectBucket(bounds.Min, bounds.Max, cameraPos, cameraForward, fogEnd, isWmo: false, countAsTaxiActor: false))
            {
                MdxCulledCount += pair.Value.Count;
                continue;
            }

            if (!_tileMdxVisibilityBuckets.TryGetValue(pair.Key, out List<FlatVisibilityBucket>? buckets))
            {
                CollectVisibleMdxInstances(frame, pair.Value, cameraPos, cameraForward, fogEnd, verticalFieldOfViewRadians, cullSmallDoodadsOnly: true, countAsTaxiActor: false);
                continue;
            }

            foreach (FlatVisibilityBucket bucket in buckets)
            {
                if (!ShouldVisitFlatVisibilityBucket(bucket, cameraPos, fogEnd, isWmo: false))
                {
                    MdxCulledCount += bucket.Instances.Count;
                    continue;
                }

                CollectVisibleMdxInstances(frame, bucket.Instances, cameraPos, cameraForward, fogEnd, verticalFieldOfViewRadians, cullSmallDoodadsOnly: true, countAsTaxiActor: false);
            }
        }

        if (_externalMdxInstances.Count > 0)
            CollectVisibleMdxInstances(frame, _externalMdxInstances, cameraPos, cameraForward, fogEnd, verticalFieldOfViewRadians, cullSmallDoodadsOnly: true, countAsTaxiActor: false);
    }

    private static float ExtractVerticalFieldOfViewRadians(Matrix4x4 projection)
    {
        float inverseTanHalfFov = projection.M22;
        if (!float.IsFinite(inverseTanHalfFov) || inverseTanHalfFov <= 1e-6f)
            return MathF.PI / 3f;

        return 2f * MathF.Atan(1f / inverseTanHalfFov);
    }

    private static Vector3 ExtractCameraForward(Matrix4x4 viewInverse)
    {
        Vector3 forward = Vector3.TransformNormal(-Vector3.UnitZ, viewInverse);
        float lengthSq = forward.LengthSquared();
        if (lengthSq <= 1e-6f)
            return Vector3.UnitY;

        return forward / MathF.Sqrt(lengthSq);
    }

    private static float ComputeVisionConeFactor(Vector3 cameraPos, Vector3 cameraForward, Vector3 targetPos, float targetDistanceSq)
    {
        if (targetDistanceSq <= ObjectNearHoldRadiusSq)
            return 1.0f;

        float forwardLengthSq = cameraForward.LengthSquared();
        if (forwardLengthSq <= 1e-6f)
            return 1.0f;

        Vector3 toTarget = targetPos - cameraPos;
        float toTargetLengthSq = toTarget.LengthSquared();
        if (toTargetLengthSq <= 1e-6f)
            return 1.0f;

        float invTargetLength = 1.0f / MathF.Sqrt(toTargetLengthSq);
        float alignment = Vector3.Dot(toTarget * invTargetLength, cameraForward);
        float factor = (alignment - VisionConeRearDot) / MathF.Max(0.001f, VisionConeFrontDot - VisionConeRearDot);
        return Math.Clamp(factor, 0.0f, 1.0f);
    }

    private static float ComputeConeCullDistance(float baseCullDistance, float coneFactor)
    {
        if (baseCullDistance <= 0f)
            return ObjectNearHoldRadius;

        float scale = RearConeCullFraction + (1.0f - RearConeCullFraction) * coneFactor;
        return MathF.Max(ObjectNearHoldRadius, baseCullDistance * scale);
    }

    private static float ComputeConeFade(float coneFactor, float centerDistanceSq)
    {
        if (centerDistanceSq <= ObjectNearHoldRadiusSq)
            return 1.0f;

        return RearConeFadeFloor + (1.0f - RearConeFadeFloor) * coneFactor;
    }

    private static float ComputeLoadPriorityScore(float centerDistanceSq, float coneFactor)
    {
        float penalty = RearConeLoadPenalty - (RearConeLoadPenalty - 1.0f) * coneFactor;
        return centerDistanceSq * penalty;
    }

    private static double TicksToMs(long ticks) => ticks * 1000.0 / Stopwatch.Frequency;

    private static double MeasureDurationMs(Action action)
    {
        var stageTimer = Stopwatch.StartNew();
        action();
        return stageTimer.Elapsed.TotalMilliseconds;
    }

    private void AdvanceAutomaticTimeOfDay(TerrainLighting lighting)
    {
        long currentTick = Stopwatch.GetTimestamp();
        if (!_automaticTimeTickInitialized)
        {
            _lastAutomaticTimeTick = currentTick;
            _automaticTimeTickInitialized = true;
            return;
        }

        long previousTick = _lastAutomaticTimeTick;
        _lastAutomaticTimeTick = currentTick;
        double elapsedSeconds = (currentTick - previousTick) / (double)Stopwatch.Frequency;
        lighting.AdvanceAutomaticTime(elapsedSeconds);
    }

    /// <summary>
    /// Split the frame's untimed remainder into prologue / pass-gap / epilogue so an unaccounted
    /// hitch names a region. <paramref name="preCoordinatorMs"/> and
    /// <paramref name="postCoordinatorMs"/> are elapsed-since-frame-start boundary readings.
    /// </summary>
    private void RecordRenderRegionBreakdown(
        WorldRenderFrame frame, double preCoordinatorMs, double postCoordinatorMs, double totalMs)
    {
        // Stages timed before the coordinator runs.
        double timedPrologue = frame.SceneMaintenanceMs + frame.DeferredAssetLoadMs + frame.TaxiActorUpdateMs;

        // Stages timed inside the coordinator.
        double timedInCoordinator =
            frame.LightingMs + frame.SkyMs + frame.SkyboxBackdropMs + frame.WdlMs + frame.TerrainMs
            + frame.WmoVisibilityMs + frame.WmoSubmissionMs + frame.WmoTransparentSubmissionMs
            + frame.MdxAnimationMs + frame.MdxVisibilityMs + frame.MdxOpaqueSubmissionMs
            + frame.LiquidMs + frame.MdxTransparentSortMs + frame.MdxTransparentSubmissionMs
            + frame.OverlayMs + frame.PrepareObjectPhaseMs;

        RenderPrologueMs = Math.Max(0, preCoordinatorMs - timedPrologue);
        RenderPassGapMs = Math.Max(0, (postCoordinatorMs - preCoordinatorMs) - timedInCoordinator);
        RenderEpilogueMs = Math.Max(0, totalMs - postCoordinatorMs);

        RenderProloguePeakMs = Math.Max(RenderProloguePeakMs, RenderPrologueMs);
        RenderPassGapPeakMs = Math.Max(RenderPassGapPeakMs, RenderPassGapMs);
        RenderEpiloguePeakMs = Math.Max(RenderEpiloguePeakMs, RenderEpilogueMs);
    }

    private void FinalizeRenderFrameStats(WorldRenderFrame frame, Stopwatch frameTimer, Matrix4x4 view)
    {
        int terrainChunksRendered = _terrainManager.Renderer.ChunksRendered;
        int terrainChunksCulled = _terrainManager.Renderer.ChunksCulled;
        int wdlVisibleTiles = _wdlTerrain?.VisibleTiles ?? 0;
        int wdlHiddenTiles = _wdlTerrain?.HiddenTiles ?? 0;
        LastRenderFrameStats = frame.ToStats(
            frameTimer.Elapsed.TotalMilliseconds,
            _assets.PendingAssetLoadCount,
            terrainChunksRendered,
            terrainChunksCulled,
            wdlVisibleTiles,
            wdlHiddenTiles) with
        {
            SceneLighting = _sceneLightManager.TakeFrameCounters(),
        };

        // Retain the frame. Without this the stats are produced and immediately discarded, which is
        // why a periodic hitch was invisible to every consumer.
        // Camera pose comes from the view matrix so this does not depend on where in Render the
        // local cameraPos happens to be resolved.
        Matrix4x4.Invert(view, out Matrix4x4 frameHistoryViewInverse);
        Vector3 cameraPosition = frameHistoryViewInverse.Translation;
        Vector3 cameraForward = new(
            -frameHistoryViewInverse.M31, -frameHistoryViewInverse.M32, -frameHistoryViewInverse.M33);
        bool cameraMoved =
            cameraPosition != _frameHistoryPreviousCameraPosition
            || cameraForward != _frameHistoryPreviousCameraForward;
        _frameHistoryPreviousCameraPosition = cameraPosition;
        _frameHistoryPreviousCameraForward = cameraForward;
        FrameHistory.Record(LastRenderFrameStats, cameraMoved);
    }

    // ── ISceneRenderer ──────────────────────────────────────────────────

    private bool _renderDiagPrinted = false;
    public void Render(Matrix4x4 view, Matrix4x4 proj)
    {
        WorldRenderFrame frame = _renderFrame;
        frame.Reset();
        var frameTimer = Stopwatch.StartNew();

        // Detector-power check (diagnostics only): stall this frame by a known amount so the frame
        // history can be verified to flag it. Inside the timer so the stall is measured as real work.
        if (DebugInjectStallMs > 0)
        {
            double stallMs = DebugInjectStallMs;
            DebugInjectStallMs = 0;
            long stallUntil = Stopwatch.GetTimestamp() + (long)(stallMs / 1000.0 * Stopwatch.Frequency);
            while (Stopwatch.GetTimestamp() < stallUntil)
            {
                // Busy-wait: a sleep would yield the thread and measure scheduler latency instead of
                // frame cost, which is not what the check is proving.
            }
        }
        _pendingVisibleMdxLoadDistances.Clear();
        _pendingVisibleWmoLoadDistances.Clear();
        _sceneGraphFrameVisibilityPrepared = false;
        _sceneGraphVisibleMdxInstances.Clear();
        _sceneGraphVisibleWmoInstances.Clear();

        frame.SceneMaintenanceMs = MeasureDurationMs(() =>
        {
            _pm4Overlay.TryFinalizePm4OverlayLoad();

            // Rebuild flat instance lists if tiles changed.
            if (_instancesDirty)
                RebuildInstanceLists();
            else if (UseHierarchicalSceneTraversal && _sceneGraphBuild is null)
                RebuildSceneGraphObjectIndex();

            // Keep WMO admission ahead of visibility. The near retained window
            // is still rendered through the selected-tile gate, but its models
            // must be prioritized before a camera turn exposes the tile.
            QueueNearFieldWmoAssetLoads();
        });

        frame.DeferredAssetLoadMs = MeasureDurationMs(() =>
        {
            ProcessDeferredAssetLoads();
            _assets.ProcessDeferredWmoDoodadLoads();
        });
        frame.TaxiActorUpdateMs = MeasureDurationMs(UpdateTaxiActorInstances);

        // Extract camera position for sky dome
        Matrix4x4.Invert(view, out var viewInvSky);
        var camPos = new Vector3(viewInvSky.M41, viewInvSky.M42, viewInvSky.M43);
        var lighting = _terrainManager.Lighting;
        Vector3 fogColor;
        float fogStart;
        float fogEnd;

        // 0. Resolve frame lighting before any world pass so terrain, WDL, liquids,
        // skybackdrops, WMOs, and MDXs all sample one lighting state.
        fogColor = Vector3.Zero;
        fogStart = 0f;
        fogEnd = 0f;
        float objectFogStart = 0f;
        float objectFogEnd = 0f;
        Vector3 cameraPos = Vector3.Zero;
        Vector3 cameraForward = Vector3.UnitZ;
        float verticalFieldOfViewRadians = ExtractVerticalFieldOfViewRadians(proj);
        double overlayElapsedMs = 0;
        double objectWireframeMs = 0;
        int objectWireframePreparedCount = 0;
        int objectWireframeSubmittedCount = 0;
        bool objectWireframeEnabled = false;
        double selectionBoundsMs = 0;
        int selectionBoundsPreparedCount = 0;
        double pm4BoundsMs = 0;
        int pm4BoundsPreparedCount = 0;
        double pm4GeometryPrepareMs = 0;
        double pm4GeometrySubmitMs = 0;
        double pm4NodesMs = 0;
        int pm4GeometryPreparedCount = 0;
        int pm4GeometrySubmittedCount = 0;
        int pm4NodesPreparedCount = 0;
        int poiTaxiPreparedCount = 0;
        double poiTaxiMs = 0;
        int areaTriggerPreparedCount = 0;
        double areaTriggersMs = 0;
        int audioEmitterMarkerPreparedCount = 0;
        double audioEmitterMarkersMs = 0;

        // Boundary probe: everything before this point that is not one of the three timed prologue
        // stages is untimed setup.
        double preCoordinatorMs = frameTimer.Elapsed.TotalMilliseconds;

        bool continuedPastTerrain = WorldFramePassCoordinator.Execute(
            new WorldFramePassOptions(_objectsVisible, _wmosVisible, _doodadsVisible),
            new WorldFramePasses(
                () =>
                {
                    frame.LightingMs = MeasureDurationMs(() =>
                    {
                        LitLoader.LitLightingSample? litSample = null;
                        string fogRecommendationSource;
                        AdvanceAutomaticTimeOfDay(lighting);
                        if (_lightService != null)
                        {
                            // Light.dbc and LIT use the same native 0..2880 clock. Keep the DBC
                            // overlay on the same frame/time as the global terrain lighting.
                            _lightService.TimeOfDay = lighting.TimeOfDayUnits;
                            _lightService.Update(camPos);
                        }
                        UpdateActiveSkyboxModel();

                        if (_litLoader != null && _litLoader.HasData)
                            litSample = _litLoader.EvaluateLighting(camPos, lighting.GameTime);

                        // The viewer global sun is unconditional. DBC/LightData colors and fog
                        // are spatial overlays; a missing record is therefore an identity case,
                        // never a reason to darken the terrain or retain a departed zone's fog.
                        RestoreGlobalViewerFogRange(lighting);
                        lighting.ClearExternalLighting();
                        lighting.Update();
                        _skyDome.UpdateFromLighting(lighting.GameTime, lighting.LightDirection);
                        fogRecommendationSource = "Global viewer light";

                        if (_useLocalDbcLightingOverlay
                            && _lightService is { HasActiveLocalOverlay: true } localLighting)
                        {
                            (float dbcFogStart, float dbcFogEnd) =
                                TerrainLightingMath.ComputeClientFogRange(
                                    localLighting.FogEnd,
                                    localLighting.FogScaler);

                            var globalState = new TerrainViewerLightingState(
                                lighting.LightColor,
                                lighting.AmbientColor,
                                lighting.FogColor,
                                lighting.FogStart,
                                lighting.FogEnd);
                            var localState = new TerrainViewerLightingState(
                                localLighting.DirectColor,
                                localLighting.AmbientColor,
                                localLighting.FogColor,
                                dbcFogStart,
                                dbcFogEnd);
                            TerrainViewerLightingState composed =
                                TerrainViewerLightingComposer.ComposeGlobalWithLocal(
                                    globalState,
                                    localState,
                                    localLighting.ActiveLocalWeight);

                            Vector3 globalSkyTop = _skyDome.ZenithColor;
                            Vector3 globalSkyHorizon = _skyDome.HorizonColor;
                            lighting.ApplyExternalLighting(
                                composed.DirectionalColor,
                                composed.AmbientColor,
                                composed.FogColor);
                            lighting.FogStart = composed.FogStart;
                            lighting.FogEnd = composed.FogEnd;
                            lighting.Update();
                            fogRecommendationSource = "Global viewer light + local DBC overlay";

                            _skyDome.ZenithColor = Vector3.Lerp(
                                globalSkyTop,
                                localLighting.SkyTopColor,
                                localLighting.ActiveLocalWeight);
                            _skyDome.HorizonColor = Vector3.Lerp(
                                globalSkyHorizon,
                                lighting.FogColor,
                                localLighting.ActiveLocalWeight);
                            _skyDome.SkyFogColor = lighting.FogColor;
                        }

                        if (_useLitFogOverride && litSample != null)
                        {
                            CapturePreLitFogRange(lighting);
                            // LIT tracks 0/1/7 are the global diffuse, ambient, and fog colors.
                            // Apply the profile as one coherent source; silently mixing DBC colors
                            // with LIT fog produced a profile that no client file actually authored.
                            lighting.ApplyExternalLighting(
                                litSample.DirectColor,
                                litSample.AmbientColor,
                                litSample.FogColor);
                            lighting.FogStart = litSample.FogStart;
                            lighting.FogEnd = litSample.FogEnd;
                            lighting.Update();
                            fogRecommendationSource = "LIT lighting";

                            _skyDome.ZenithColor = litSample.SkyTopColor;
                            _skyDome.HorizonColor = litSample.SkyHorizonColor;
                            _skyDome.SkyFogColor = litSample.FogColor;
                        }

                        ResolveActiveFogRange(lighting, fogRecommendationSource);
                        _skyDome.UpdateFromLighting(lighting.GameTime, lighting.LightDirection);
                        fogColor = lighting.FogColor;
                        fogStart = lighting.FogStart;
                        fogEnd = lighting.FogEnd;
                        _lastLitSample = litSample;
                    });

                    _lastHoverPickFogEnd = fogEnd;
                    (objectFogStart, objectFogEnd) = ComputeObjectFogRange(fogStart, fogEnd, _objectFogEnabled);
                },
                () =>
                {
                    if (!ShowSky)
                    {
                        frame.SkyMs = 0;
                        return;
                    }

                    frame.SkyMs = MeasureDurationMs(() => _skyDome.Render(view, proj, camPos));
                },
                () =>
                {
                    if (!ShowSky)
                    {
                        _gl.ClearColor(0f, 0f, 0f, 1f);
                        frame.SkyboxBackdropMs = 0;
                        return;
                    }

                    // Also set clear color to horizon color so any gaps match the sky
                    _gl.ClearColor(_skyDome.HorizonColor.X, _skyDome.HorizonColor.Y, _skyDome.HorizonColor.Z, 1f);
                    frame.SkyboxBackdropMs = MeasureDurationMs(() => RenderSkyboxBackdrop(view, proj, camPos, fogColor, fogStart, fogEnd, lighting));
                },
                () =>
                {
                    // 0. Render WDL low-res terrain (far background — hidden tiles replaced by detailed ADTs)
                    frame.WdlMs = MeasureDurationMs(() =>
                    {
                        if (ShowWdlTerrain && _wdlTerrain != null)
                        {
                            // WDL suppression follows actual detailed ADT submission,
                            // not merely GPU residency. Retained neighbors stay loaded
                            // for streaming but keep their low-resolution underlay until
                            // the directional detail set submits them.
                            _wdlTerrain.SetDetailedTileSubmission(
                                _terrainManager.LastSelectedTiles,
                                _terrainManager.IsTileLoaded);
                            bool renderWdlAsOpaqueFallback = _terrainManager.LoadedTileCount == 0;
                            _wdlTerrain.Render(view, proj, camPos, _terrainManager.Lighting, _frustumCuller, renderWdlAsOpaqueFallback);
                        }
                    });
                },
                () =>
                {
                    // 1. Render terrain (with frustum culling) and nearby emitted-light evaluation
                    frame.TerrainMs = MeasureDurationMs(() => _terrainManager.Render(view, proj, camPos, _frustumCuller, _sceneLightManager));

                    // Reset GL state after terrain
                    _gl.DepthFunc(DepthFunction.Lequal);
                    _gl.DepthMask(true);
                    _gl.Disable(EnableCap.Blend);
                    _gl.Enable(EnableCap.DepthTest);
                    _gl.UseProgram(0); // unbind terrain shader
                },
                () =>
                {
                    // FR-001: this pass previously had no stage timer, so everything below it —
                    // including the periodic stall — landed in the unaccounted pass gap. The timer
                    // spans the whole pass; the sub-probes inside it attribute the parts.
                    long objectPhaseStart = Stopwatch.GetTimestamp();

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
                    cameraPos = new Vector3(viewInv.M41, viewInv.M42, viewInv.M43);
                    cameraForward = ExtractCameraForward(viewInv);
                    _lastRenderedCameraPosition = cameraPos;
                    _hasLastRenderedCameraPosition = true;
                    int terrainAreaId = _terrainManager.Renderer.GetChunkInfoAt(cameraPos.X, cameraPos.Y)?.AreaId ?? 0;
                    int audioAreaId = (_currentAreaLookup?.Reason == WowViewer.Core.World.AreaResolutionReason.Resolved)
                        ? (_currentAreaLookup.CanonicalAreaId ?? _currentAreaLookup.RawAreaId)
                        : terrainAreaId;

                    // Sub-probes inside the timed pass. Both do periodic residency work and are the
                    // prime suspects for the ~212 ms recurring stall.
                    long audioStart = Stopwatch.GetTimestamp();
                    _audioRuntime?.Update(
                        cameraPos,
                        cameraForward,
                        audioAreaId,
                        lighting.GameTime,
                        _mapId,
                        _currentAreaLookup,
                        // Only audio near the camera matters. Pass the terrain manager's own camera
                        // tile rather than re-deriving it here, so the audio window can never drift
                        // out of agreement with the tile the renderer considers current.
                        _terrainManager.CameraTileX,
                        _terrainManager.CameraTileY);
                    AudioRuntimeUpdateMs = TicksToMs(Stopwatch.GetTimestamp() - audioStart);
                    AudioRuntimeUpdatePeakMs = Math.Max(AudioRuntimeUpdatePeakMs, AudioRuntimeUpdateMs);

                    long pm4Start = Stopwatch.GetTimestamp();
                    _pm4Overlay.EnsurePm4OverlayMatchesCameraWindow(cameraPos);
                    Pm4OverlayWindowMs = TicksToMs(Stopwatch.GetTimestamp() - pm4Start);
                    Pm4OverlayWindowPeakMs = Math.Max(Pm4OverlayWindowPeakMs, Pm4OverlayWindowMs);

                    // Update frustum planes for culling
                    var vp = view * proj;
                    _frustumCuller.Update(vp, cameraPos);

                    // ── PASS 1: OPAQUE ──────────────────────────────────────────────
                    // Render all opaque geometry first with depth write ON.
                    // This ensures correct depth buffer before any transparent rendering.
                    _gl.Enable(EnableCap.DepthTest);
                    _gl.DepthFunc(DepthFunction.Less);
                    _gl.DepthMask(true);
                    _gl.Disable(EnableCap.Blend);

                    WmoRenderedCount = 0;
                    WmoCulledCount = 0;
                    MdxRenderedCount = 0;
                    MdxCulledCount = 0;

                    ObjectPhasePrepareMs = TicksToMs(Stopwatch.GetTimestamp() - objectPhaseStart);
                    ObjectPhasePreparePeakMs = Math.Max(ObjectPhasePreparePeakMs, ObjectPhasePrepareMs);
                    frame.PrepareObjectPhaseMs = ObjectPhasePrepareMs;
                },
                () =>
                {
                    frame.WmoVisibilityMs = MeasureDurationMs(() => CollectVisibleWmoInstances(frame, cameraPos, cameraForward, fogEnd, verticalFieldOfViewRadians));
                    FlushPendingVisibleWmoLoads();
                    RebuildSceneLights(frame);

                    // State is constant for this pass; set once to reduce per-instance churn and
                    // keep WMO submission running through one explicit visible-instance bucket.
                    frame.WmoSubmissionMs = MeasureDurationMs(() =>
                    {
                        _gl.Disable(EnableCap.Blend);
                        _gl.DepthMask(true);

                        var visibleWmoRenderers = new WmoRenderer?[frame.Visibility.VisibleWmos.Count];
                        List<WorldObjectPassCoordinator.WorldWmoOpaqueBatchCandidate> wmoBatchCandidates =
                            frame.WmoBatchCandidateScratch;
                        _worldFrameWmoRenderers.Clear();
                        WmoRenderedCount = 0;
                        for (int visibleIndex = 0; visibleIndex < frame.Visibility.VisibleWmos.Count; visibleIndex++)
                        {
                            VisibleWmoInstance visible = frame.Visibility.VisibleWmos[visibleIndex];
                            WmoRenderedCount++;
                            WmoRenderer? renderer = ResolveVisibleWmoRenderer(frame, visible.Instance.ModelKey);
                            visibleWmoRenderers[visibleIndex] = renderer;
                            if (renderer != null)
                                _worldFrameWmoRenderers.Add(renderer);

                            // Epic 249 R-10b: the batch decision is per placement. A placement a scene
                            // light reaches keeps the per-placement path and its own light set; one no
                            // light reaches is instanced. (Previously any active light anywhere disabled
                            // WMO instancing for the whole scene.)
                            bool canBatch = renderer is IGpuInstancedWmoRenderer gpuRenderer
                                && gpuRenderer.SupportsGpuInstancedOpaque;
                            bool reachedByLight = false;
                            if (canBatch && _sceneLightManager.Count > 0)
                            {
                                renderer!.GetWorldBounds(visible.Instance.Transform, out Vector3 placementMin, out Vector3 placementMax);
                                reachedByLight = _sceneLightManager.AnyAffecting(placementMin, placementMax);
                            }

                            wmoBatchCandidates.Add(new(
                                visible.Instance.ModelKey,
                                canBatch,
                                visibleIndex,
                                reachedByLight,
                                renderer?.EmitsSceneLights ?? false));
                        }

                        foreach (WmoRenderer renderer in _worldFrameWmoRenderers)
                            renderer.BeginWorldFrame();

                        WorldObjectPassCoordinator.WorldWmoOpaqueBatchPlan wmoBatchPlan =
                            WorldObjectPassCoordinator.PlanOpaqueWmoBatches(wmoBatchCandidates);
                        _sceneLightManager.RecordWmoPartition(
                            wmoBatchPlan.BatchedPlacementCount,
                            wmoBatchPlan.LitFallbackCount,
                            wmoBatchPlan.SelfLitFallbackCount);
                        Dictionary<IModelRenderer, List<Matrix4x4>> wmoDoodadBatchGroups = frame.WmoDoodadBatchGroupScratch;
                        List<WmoOpaqueDoodadBatchItem> wmoDoodadUnbatched = frame.WmoDoodadUnbatchedScratch;
                        foreach (int visibleIndex in wmoBatchPlan.FallbackVisibleIndices)
                        {
                            VisibleWmoInstance visible = frame.Visibility.VisibleWmos[visibleIndex];
                            WmoRenderer? renderer = visibleWmoRenderers[visibleIndex];
                            if (renderer == null)
                                continue;

                            renderer.RenderWithTransform(visible.Instance.Transform, view, proj, WmoRenderPass.Opaque,
                                fogColor, objectFogStart, objectFogEnd, cameraPos,
                                lighting.LightDirection, lighting.LightColor, lighting.AmbientColor,
                                _sceneLightManager);
                            AccumulateWmoRenderStats(frame, renderer.LastRenderStats, renderer.LastGroupAdmission);
                        }

                        foreach (WorldObjectPassCoordinator.WorldWmoOpaqueBatch batch in wmoBatchPlan.Batches)
                        {
                            int firstVisibleIndex = batch.VisibleIndices[0];
                            WmoRenderer renderer = visibleWmoRenderers[firstVisibleIndex]!;
                            IGpuInstancedWmoRenderer gpuRenderer = (IGpuInstancedWmoRenderer)renderer;
                            // Reused across batches within the frame, not one list per batch.
                            List<VisibleWmoInstance> instances = frame.WmoInstanceBatchScratch;
                            instances.Clear();
                            foreach (int visibleIndex in batch.VisibleIndices)
                                instances.Add(frame.Visibility.VisibleWmos[visibleIndex]);

                            // Only placements no scene light reaches are batched, so the exact local
                            // light set for every instance here is empty.
                            gpuRenderer.BeginGpuInstanceBatch(
                                 view, proj, fogColor, objectFogStart, objectFogEnd, cameraPos,
                                 lighting.LightDirection, lighting.LightColor, lighting.AmbientColor,
                                 sceneLights: null);
                            foreach (VisibleWmoInstance visible in instances)
                                gpuRenderer.QueueGpuInstance(visible.Instance.Transform);
                            gpuRenderer.EndGpuInstanceBatch();

                            // The shell is shared across placements, but WMO-internal doodads
                            // retain placement-local visibility, animation, and M2 fallback rules.
                            foreach (VisibleWmoInstance visible in instances)
                            {
                                gpuRenderer.CollectOpaqueDoodadsForPlacement(
                                    visible.Instance.Transform,
                                    cameraPos, objectFogEnd,
                                    item =>
                                    {
                                        if (item.Renderer.RequiresUnbatchedWorldRender)
                                        {
                                            wmoDoodadUnbatched.Add(item);
                                            return;
                                        }

                                        if (!wmoDoodadBatchGroups.TryGetValue(item.Renderer, out List<Matrix4x4>? transforms))
                                        {
                                            transforms = frame.RentMatrixList();
                                            wmoDoodadBatchGroups.Add(item.Renderer, transforms);
                                        }

                                        transforms.Add(item.ModelMatrix);
                                    });
                            }

                            AccumulateWmoRenderStats(frame, gpuRenderer.LastRenderStats, gpuRenderer.LastGroupAdmission);
                        }

                        foreach (WmoOpaqueDoodadBatchItem item in wmoDoodadUnbatched)
                        {
                            item.Renderer.RenderWithTransform(
                                item.ModelMatrix,
                                view,
                                proj,
                                RenderPass.Opaque,
                                1.0f,
                                fogColor,
                                objectFogStart,
                                objectFogEnd,
                                cameraPos,
                                lighting.LightDirection,
                                lighting.LightColor,
                                lighting.AmbientColor,
                                _sceneLightManager);
                        }

                        foreach ((IModelRenderer renderer, List<Matrix4x4> transforms) in wmoDoodadBatchGroups)
                        {
                            if (renderer is IGpuInstancedModelRenderer gpuDoodadRenderer
                                && gpuDoodadRenderer.SupportsGpuInstancedOpaque)
                            {
                                gpuDoodadRenderer.BeginGpuInstanceBatch(
                                    view,
                                    proj,
                                    fogColor,
                                    objectFogStart,
                                    objectFogEnd,
                                    cameraPos,
                                    lighting.LightDirection,
                                    lighting.LightColor,
                                    lighting.AmbientColor);
                                foreach (Matrix4x4 transform in transforms)
                                    gpuDoodadRenderer.QueueGpuInstance(transform);
                                gpuDoodadRenderer.EndGpuInstanceBatch();
                            }
                            else
                            {
                                renderer.BeginBatch(
                                    view,
                                    proj,
                                    fogColor,
                                    objectFogStart,
                                    objectFogEnd,
                                    cameraPos,
                                    lighting.LightDirection,
                                    lighting.LightColor,
                                    lighting.AmbientColor,
                                    _sceneLightManager);
                                foreach (Matrix4x4 transform in transforms)
                                    renderer.RenderInstance(transform, RenderPass.Opaque, 1.0f);
                            }
                        }
                    });
                    if (!_renderDiagPrinted) ViewerLog.Info(ViewerLog.Category.Wmo, $"WMO render: {WmoRenderedCount} drawn, {WmoCulledCount} culled");
                },
                () =>
                {
                    frame.MdxVisibilityMs = MeasureDurationMs(() =>
                    {
                        CollectVisibleMdxBuckets(frame, cameraPos, cameraForward, fogEnd, verticalFieldOfViewRadians);
                        CollectVisibleMdxInstances(frame, _taxiActorInstances, cameraPos, cameraForward, fogEnd, verticalFieldOfViewRadians, cullSmallDoodadsOnly: false, countAsTaxiActor: true);
                    });
                    FlushPendingVisibleMdxLoads();

                    // Advance animation only for renderers that survived visibility admission.
                    // The previous path scanned every placed MDX instance every frame, which was
                    // pure idle CPU cost on large maps even when only a fraction were visible.
                    frame.MdxAnimationMs = MeasureDurationMs(() =>
                    {
                        if (!WorldDoodadAnimationEnabled)
                            return;

                        HashSet<IModelRenderer> updatedRenderers = frame.UpdatedRendererScratch;
                        WorldObjectPassCoordinator.ExecuteVisibleMdxAnimation(frame.ObjectPasses, frame.Visibility, visible =>
                        {
                            IModelRenderer? renderer = ResolveVisibleMdxRenderer(frame, visible.Instance.ModelKey);
                            if (renderer != null && updatedRenderers.Add(renderer))
                            {
                                renderer.UpdateAnimation();
                            }
                        });
                    });

                    frame.MdxTransparentSortMs = MeasureDurationMs(() => PlanVisibleMdxPasses(frame));

                    frame.MdxOpaqueSubmissionMs = MeasureDurationMs(() =>
                    {
                        HashSet<IGpuInstancedModelRenderer> gpuBatchRenderers = frame.GpuBatchRendererScratch;
                        HashSet<IModelRenderer> immediateBatchRenderers = frame.ImmediateBatchRendererScratch;

                        // Spec 202 T003. Draw calls, not instances, are what batching exists to
                        // reduce, and the two cannot be converted: a model draws once per geoset or
                        // section, so this has to be read from the GL call sites. Sampled across the
                        // whole pass including EndGpuInstanceBatch, which is where the instanced
                        // draws are actually issued.
                        long opaqueDrawCallStart = ModelDrawCallCounter.Count;

                        try
                        {
                            (frame.OpaqueBatchedMdxCount, frame.OpaqueUnbatchedMdxCount) =
                            WorldObjectPassCoordinator.ExecutePlannedOpaqueMdx(
                                frame.ObjectPasses,
                                frame.Visibility,
                                visible =>
                                {
                                    IModelRenderer? renderer = ResolveVisibleMdxRenderer(frame, visible.Instance.ModelKey);
                                    WorldModelRenderPath renderPath = ResolveVisibleMdxRenderPath(frame, visible.Instance.ModelKey);
                                    if (renderer == null)
                                    {
                                        // Counted, not silently absent: an instance that resolved no
                                        // renderer never reached the GPU at all, while the aggregate
                                        // counter above still counts it as an unbatched draw.
                                        frame.OpaqueModelSubmission.RecordRendererUnavailable();
                                        return;
                                    }

                                    frame.OpaqueSubmittedModelKeyScratch.Add(visible.Instance.ModelKey);

                                    // Which of the three gates (spec 202 research R3) sent this
                                    // instance down the per-instance path. The toggle and the route
                                    // are different problems with different fixes, and the aggregate
                                    // counter cannot tell them apart.
                                    frame.OpaqueModelSubmission.Record(
                                        renderPath,
                                        WorldModelSubmissionOutcome.Unbatched,
                                        !MdxOpaqueBatchingEnabled
                                            ? WorldModelBatchGate.BatchingDisabled
                                            : WorldModelBatchGate.RouteRequiresUnbatchedRender);

                                    if (!_renderDiagPrinted)
                                    {
                                        ViewerLog.Info(ViewerLog.Category.Mdx,
                                            $"[M2-WORLD-DIAG] opaqueUnbatched key=\"{visible.Instance.ModelKey}\" renderer={renderer?.GetType().Name} hasTransparent={renderer?.HasTransparentWorldPass} requiresUnbatched={renderer?.RequiresUnbatchedWorldRender} bounds=({visible.Instance.BoundsMin.X:F1},{visible.Instance.BoundsMin.Y:F1},{visible.Instance.BoundsMin.Z:F1})-({visible.Instance.BoundsMax.X:F1},{visible.Instance.BoundsMax.Y:F1},{visible.Instance.BoundsMax.Z:F1}) pos={visible.Instance.Transform.Translation}");
                                    }

                                    renderer.RenderWithTransform(visible.Instance.Transform, view, proj, RenderPass.Opaque, visible.OpaqueFade,
                                        fogColor, objectFogStart, objectFogEnd, cameraPos,
                                        lighting.LightDirection, lighting.LightColor, lighting.AmbientColor,
                                        _sceneLightManager);
                                    MdxRenderedCount++;
                                },
                                visible =>
                                {
                                    IModelRenderer? renderer = ResolveVisibleMdxRenderer(frame, visible.Instance.ModelKey);
                                    WorldModelRenderPath renderPath = ResolveVisibleMdxRenderPath(frame, visible.Instance.ModelKey);
                                    if (renderer == null)
                                    {
                                        frame.OpaqueModelSubmission.RecordRendererUnavailable();
                                        return;
                                    }

                                    frame.OpaqueSubmittedModelKeyScratch.Add(visible.Instance.ModelKey);

                                    // Spec 207 US1: the OpaqueFade >= 0.999 condition that used to
                                    // stand here forced every distance-faded instance onto the
                                    // one-draw-per-instance path -- roughly a third of the visible
                                    // disc, and exactly the population instancing helps most. The
                                    // renderer now batches faded instances separately so they blend
                                    // correctly, so fade no longer disqualifies instancing.
                                    if (renderer is IGpuInstancedModelRenderer gpuRenderer
                                        && gpuRenderer.SupportsGpuInstancedOpaque)
                                    {
                                        if (gpuBatchRenderers.Add(gpuRenderer))
                                        {
                                            gpuRenderer.BeginGpuInstanceBatch(
                                                view, proj, fogColor, objectFogStart, objectFogEnd, cameraPos,
                                                lighting.LightDirection, lighting.LightColor, lighting.AmbientColor);
                                        }

                                        gpuRenderer.QueueGpuInstance(visible.Instance.Transform, visible.OpaqueFade);

                                        // The only outcome on this callback that reduces draw calls.
                                        frame.OpaqueModelSubmission.Record(
                                            renderPath,
                                            WorldModelSubmissionOutcome.Instanced,
                                            WorldModelBatchGate.None);

                                        // Track the faded share separately: it is the population
                                        // spec 207 moved onto this path, so its size is the measure
                                        // of whether that was worth doing.
                                        if (visible.OpaqueFade < 0.999f)
                                            frame.OpaqueModelSubmission.RecordFadedInstanced();
                                    }
                                    else
                                    {
                                        if (immediateBatchRenderers.Add(renderer))
                                        {
                                            renderer.BeginBatch(
                                                view, proj, fogColor, objectFogStart, objectFogEnd, cameraPos,
                                                lighting.LightDirection, lighting.LightColor, lighting.AmbientColor,
                                                _sceneLightManager);
                                        }

                                        renderer.RenderInstance(visible.Instance.Transform, RenderPass.Opaque, visible.OpaqueFade);

                                        // Spec 202 research R1: this arm hoists state setup but still
                                        // issues one draw per instance. Counting it as "batched"
                                        // alongside the instanced arm is what made the batched counter
                                        // unreadable, so it gets its own outcome and the gate that put
                                        // it here.
                                        frame.OpaqueModelSubmission.Record(
                                            renderPath,
                                            WorldModelSubmissionOutcome.StateHoisted,
                                            renderer is IGpuInstancedModelRenderer { SupportsGpuInstancedOpaque: true }
                                                ? WorldModelBatchGate.OpaqueFadeBelowInstancingThreshold
                                                : WorldModelBatchGate.GpuInstancingUnsupported);
                                    }

                                    MdxRenderedCount++;
                                });
                        }
                        finally
                        {
                            foreach (IGpuInstancedModelRenderer gpuRenderer in gpuBatchRenderers)
                                gpuRenderer.EndGpuInstanceBatch();

                            frame.OpaqueModelSubmission.RecordDrawCalls(ModelDrawCallCounter.Since(opaqueDrawCallStart));

                            // The floor per-model instancing converges on (spec 202 research R2).
                            // Recorded next to the instanced count so the gap between them is
                            // readable without a second flight.
                            frame.OpaqueModelSubmission.DistinctModelCount = frame.OpaqueSubmittedModelKeyScratch.Count;
                        }
                    });

                    if (!_renderDiagPrinted) ViewerLog.Info(ViewerLog.Category.Mdx, $"MDX opaque: {MdxRenderedCount} drawn, {MdxCulledCount} culled");
                },
                () =>
                {
                    // ── PASS 2: LIQUID ──────────────────────────────────────────────
                    // Render liquid after opaque geometry has established the depth buffer,
                    // but before transparent MDX layers so reflective/translucent model
                    // surfaces are composited on top instead of being overpainted by water.
                    _gl.Disable(EnableCap.Blend);
                    _gl.DepthMask(true);
                    _gl.Enable(EnableCap.DepthTest);
                    _gl.DepthFunc(DepthFunction.Lequal);
                    frame.LiquidMs = MeasureDurationMs(() => _terrainManager.RenderLiquid(view, proj, cameraPos));
                },
                () =>
                {
                    // ── PASS 3: TRANSPARENT (back-to-front, frustum-culled) ─────────
                    // Render transparent/blended object layers sorted by distance to camera.
                    // Depth test ON but depth write OFF so transparent objects don't
                    // occlude each other incorrectly.
                    MeasureDurationMs(() =>
                    {
                        _gl.Enable(EnableCap.DepthTest);
                        _gl.DepthFunc(DepthFunction.Lequal);

                        List<(bool IsWmo, int Index, float DistanceSq)> transparentObjectSort =
                            frame.TransparentSortScratch;

                        for (int i = 0; i < frame.Visibility.VisibleWmos.Count; i++)
                            transparentObjectSort.Add((true, i, frame.Visibility.VisibleWmos[i].CenterDistanceSq));

                        for (int i = 0; i < frame.ObjectPasses.TransparentVisibleMdxRoutes.Count; i++)
                        {
                            var route = frame.ObjectPasses.TransparentVisibleMdxRoutes[i];
                            var visible = frame.Visibility.VisibleMdx[route.VisibleMdxIndex];
                            transparentObjectSort.Add((false, route.VisibleMdxIndex, visible.CenterDistanceSq));
                        }

                        transparentObjectSort.Sort((left, right) => right.DistanceSq.CompareTo(left.DistanceSq));

                        frame.TransparentBatchedMdxCount = 0;
                        frame.TransparentUnbatchedMdxCount = 0;
                        frame.WmoTransparentSubmissionMs = 0;
                        frame.MdxTransparentSubmissionMs = 0;

                        foreach (var entry in transparentObjectSort)
                        {
                            if (entry.IsWmo)
                            {
                                var visibleWmo = frame.Visibility.VisibleWmos[entry.Index];
                                WmoRenderer? renderer = ResolveVisibleWmoRenderer(frame, visibleWmo.Instance.ModelKey);
                                if (renderer == null)
                                    continue;

                                double wmoTransparentMs = MeasureDurationMs(() =>
                                {
                                    renderer.RenderWithTransform(visibleWmo.Instance.Transform, view, proj, WmoRenderPass.Transparent,
                                        fogColor, objectFogStart, objectFogEnd, cameraPos,
                                        lighting.LightDirection, lighting.LightColor, lighting.AmbientColor,
                                        _sceneLightManager);
                                });
                                frame.WmoTransparentSubmissionMs += wmoTransparentMs;
                                AccumulateWmoRenderStats(frame, renderer.LastRenderStats, renderer.LastGroupAdmission);
                                continue;
                            }

                            var visibleMdx = frame.Visibility.VisibleMdx[entry.Index];
                            IModelRenderer? mdxRenderer = ResolveVisibleMdxRenderer(frame, visibleMdx.Instance.ModelKey);
                            WorldModelRenderPath transparentRenderPath = ResolveVisibleMdxRenderPath(frame, visibleMdx.Instance.ModelKey);
                            if (mdxRenderer == null)
                            {
                                frame.TransparentModelSubmission.RecordRendererUnavailable();
                                continue;
                            }

                            frame.TransparentSubmittedModelKeyScratch.Add(visibleMdx.Instance.ModelKey);
                            long transparentDrawCallStart = ModelDrawCallCounter.Count;

                            double mdxTransparentMs = MeasureDurationMs(() =>
                            {
                                mdxRenderer.RenderWithTransform(visibleMdx.Instance.Transform, view, proj, RenderPass.Transparent, visibleMdx.TransparentFade,
                                    fogColor, objectFogStart, objectFogEnd, cameraPos,
                                    lighting.LightDirection, lighting.LightColor, lighting.AmbientColor,
                                    _sceneLightManager);
                            });
                            frame.MdxTransparentSubmissionMs += mdxTransparentMs;
                            frame.TransparentUnbatchedMdxCount++;

                            // The transparent pass has no batch path at all: draw order is
                            // observable here, so every instance is submitted on its own. That is a
                            // property of the pass, not a gate any instance failed, and
                            // TransparentBatchedMdxCount is consequently never incremented.
                            frame.TransparentModelSubmission.Record(
                                transparentRenderPath,
                                WorldModelSubmissionOutcome.Unbatched,
                                WorldModelBatchGate.PassHasNoBatchPath);
                            frame.TransparentModelSubmission.RecordDrawCalls(ModelDrawCallCounter.Since(transparentDrawCallStart));
                        }

                        frame.TransparentModelSubmission.DistinctModelCount = frame.TransparentSubmittedModelKeyScratch.Count;

                        foreach (WmoRenderer renderer in _worldFrameWmoRenderers)
                            renderer.EndWorldFrame();
                        _worldFrameWmoRenderers.Clear();
                    });
                    if (!_renderDiagPrinted) _renderDiagPrinted = true;
                },
                () =>
                {
                    overlayElapsedMs = MeasureDurationMs(() =>
                    {
                        objectWireframeEnabled = _assets.ObjectWireframeEnabled || _wireframeRevealEnabled;
                        if (_assets.ObjectWireframeEnabled)
                        {
                            objectWireframeMs = MeasureDurationMs(() =>
                            {
                                (objectWireframePreparedCount, objectWireframeSubmittedCount) =
                                    RenderVisibleObjectWireframeOverlay(frame, view, proj, cameraPos, fogColor, fogStart, fogEnd, lighting);
                            });
                        }
                        else if (_wireframeRevealEnabled)
                        {
                            objectWireframeMs = MeasureDurationMs(() =>
                            {
                                (objectWireframePreparedCount, objectWireframeSubmittedCount) =
                                    RenderWireframeReveal(view, proj, cameraPos, fogColor, fogStart, fogEnd, lighting);
                            });
                        }

                        // Reset GL state before bounding boxes
                        _gl.Disable(EnableCap.Blend);
                        _gl.DepthMask(true);
                        _gl.Enable(EnableCap.DepthTest);
                        _gl.DepthFunc(DepthFunction.Lequal);
                        _gl.UseProgram(0);
                        _gl.BindVertexArray(0);

                        // 4. Debug bounding boxes for camera-admitted placements.
                        //
                        // The visibility pass has already paid the placement admission cost.
                        // Walking _mdxInstances/_wmoInstances here made the debug overlay rescan
                        // every loaded placement on every frame, including placements that could
                        // never produce a box. On full-map scenes that scan was the dominant
                        // overlay cost. Keep selected bounds independent, but consume the
                        // visibility frame for the global debug boxes.
                        if ((_showSelectedObjectBounds || _showBoundingBoxes || _pm4Overlay._showPm4ObjectBounds) && _bbRenderer != null)
                        {
                // Depth test ON so boxes behind terrain/objects are hidden,
                // depth write OFF so box lines don't occlude models
                _gl.Enable(EnableCap.DepthTest);
                _gl.DepthFunc(DepthFunction.Lequal);
                _gl.DepthMask(false);
                _bbRenderer.BeginBatch();

                float selectedBoundsTime = (float)(Stopwatch.GetTimestamp() / (double)Stopwatch.Frequency);
                Vector3 selectedBoundsInnerColor = Pm4ColorSelectedBounds;
                Vector3 selectedBoundsAccentA = Pm4ColorSelection;       // saturated yellow
                Vector3 selectedBoundsAccentB = Pm4ColorHighlight;       // saturated teal

                selectionBoundsMs = MeasureDurationMs(() =>
                {
                    if (_showSelectedObjectBounds)
                    {
                        if (SelectedInstance is ObjectInstance selectedInstance && !ShouldHideObjectInstanceByUniqueId(selectedInstance))
                        {
                            // A WMO doodad whose model has not streamed in yet has placeholder
                            // bounds; flag that in the accent color so the operator reads the
                            // box as "centroid marker", never as the object's extent.
                            bool doodadPlaceholderBounds = selectedInstance.AssetKind == "WMO Doodad"
                                && !selectedInstance.BoundsResolved;
                            Vector3 selectedBoundsAccentBResolved = doodadPlaceholderBounds
                                ? new Vector3(1.0f, 0.55f, 0.10f)   // orange = placeholder, not real extent
                                : Pm4ColorHighlight;
                            // Guard against a degenerate (zero-volume) box only. This floor used to
                            // be 0.75 yd, which made the box stop describing the object: a scroll a
                            // third of a yard across was drawn inside 1.5 yd of wireframe, and the
                            // accent pass then inflated that further. A selection box that does not
                            // match the thing it selects is worse than a small one — at a distance
                            // where a tiny doodad is genuinely hard to see, the operator is not
                            // trying to select it anyway.
                            const float minHalfExtent = 0.02f;

                            // Draw the model's own box carried through its placement transform, so
                            // the outline is oriented with the object. Re-fitting a rotated box to
                            // the world axes inflates it by up to 1.73x, and picking already tests
                            // the oriented box — so the drawn box used to be larger than the
                            // clickable one.
                            // Prefer the tight geometry box, but fall back to the instance's own
                            // local box before falling back to the world AABB. That matters for
                            // native MDX, where LocalBounds is ALREADY the geometry box — most
                            // instances are constructed with bounds resolved inline and never pass
                            // through the lazy refresh, so keying only on SelectionBoundsResolved
                            // sent the common case to the axis-aligned path and changed nothing.
                            bool hasTightBox = selectedInstance.SelectionBoundsResolved
                                && AreFiniteOrderedBounds(
                                    selectedInstance.SelectionLocalBoundsMin,
                                    selectedInstance.SelectionLocalBoundsMax);
                            bool hasLocalBox = selectedInstance.BoundsResolved
                                && AreFiniteOrderedBounds(
                                    selectedInstance.LocalBoundsMin,
                                    selectedInstance.LocalBoundsMax);
                            bool useOrientedBox = hasTightBox || hasLocalBox;

                            Vector3 bbMin = hasTightBox
                                ? selectedInstance.SelectionLocalBoundsMin
                                : hasLocalBox ? selectedInstance.LocalBoundsMin : selectedInstance.BoundsMin;
                            Vector3 bbMax = hasTightBox
                                ? selectedInstance.SelectionLocalBoundsMax
                                : hasLocalBox ? selectedInstance.LocalBoundsMax : selectedInstance.BoundsMax;
                            Vector3 center = (bbMin + bbMax) * 0.5f;
                            Vector3 halfExtent = (bbMax - bbMin) * 0.5f;
                            halfExtent = new Vector3(
                                MathF.Max(halfExtent.X, minHalfExtent),
                                MathF.Max(halfExtent.Y, minHalfExtent),
                                MathF.Max(halfExtent.Z, minHalfExtent));
                            bbMin = center - halfExtent;
                            bbMax = center + halfExtent;

                            // Selection highlight: draw the model's own mesh as a red wireframe
                            // overlay instead of a bounding box (operator request). Fall back to
                            // the oriented box only when no renderer is available yet (model not
                            // streamed in), and for WMO-doodad placement markers, whose mesh
                            // belongs to the parent WMO's doodad cache.
                            bool drewSelectionWireframe = false;
                            if (!doodadPlaceholderBounds && selectedInstance.AssetKind != "WMO Doodad")
                            {
                                IModelRenderer? selectedMdxRenderer =
                                    ResolveVisibleMdxRenderer(frame, selectedInstance.ModelKey);
                                if (selectedMdxRenderer != null)
                                {
                                    selectedMdxRenderer.RenderWireframeOverlay(
                                        selectedInstance.Transform, view, proj,
                                        fogColor, fogStart, fogEnd, cameraPos,
                                        lighting.LightDirection, lighting.LightColor, lighting.AmbientColor,
                                        Pm4ColorHighlight);
                                    drewSelectionWireframe = true;
                                }
                                else
                                {
                                    WmoRenderer? selectedWmoRenderer =
                                        ResolveVisibleWmoRenderer(frame, selectedInstance.ModelKey);
                                    if (selectedWmoRenderer != null)
                                    {
                                        selectedWmoRenderer.RenderWireframeOverlay(
                                            selectedInstance.Transform, view, proj,
                                            fogColor, fogStart, fogEnd, cameraPos,
                                            lighting.LightDirection, lighting.LightColor, lighting.AmbientColor,
                                            Pm4ColorHighlight);
                                        drewSelectionWireframe = true;
                                    }
                                }
                            }

                            if (!drewSelectionWireframe)
                            {
                                if (useOrientedBox)
                                {
                                    _bbRenderer.BatchHighlightedBoxOriented(
                                        bbMin,
                                        bbMax,
                                        selectedInstance.Transform,
                                        selectedBoundsTime,
                                        selectedBoundsInnerColor,
                                        selectedBoundsAccentA,
                                        selectedBoundsAccentBResolved);
                                }
                                else
                                {
                                    _bbRenderer.BatchHighlightedBoxMinMax(
                                        bbMin,
                                        bbMax,
                                        selectedBoundsTime,
                                        selectedBoundsInnerColor,
                                        selectedBoundsAccentA,
                                        selectedBoundsAccentBResolved);
                                }
                            }

                            // 3D selection aids for WMO doodads: an origin jewel + gold position
                            // pin mark the MODD placement point, and the RGB axis tripod shows the
                            // placement's orientation and scale direction. Sized from the box so
                            // they stay readable on both a torch and a chandelier.
                            if (selectedInstance.AssetKind == "WMO Doodad")
                            {
                                Vector3 anchor = selectedInstance.PlacementPosition;
                                float boxRadius = MathF.Max(
                                    halfExtent.X,
                                    MathF.Max(halfExtent.Y, halfExtent.Z));
                                float aidScale = Math.Clamp(boxRadius * 1.5f, 0.35f, 4f);

                                _bbRenderer.BatchOctahedron(
                                    anchor,
                                    Math.Clamp(boxRadius * 0.3f, 0.06f, 0.5f),
                                    new Vector3(0.20f, 0.95f, 1.00f));
                                _bbRenderer.BatchPin(
                                    anchor,
                                    aidScale,
                                    Math.Clamp(boxRadius * 0.25f, 0.05f, 0.4f),
                                    new Vector3(1.00f, 0.85f, 0.25f));

                                Vector3 axisX = Vector3.Normalize(Vector3.TransformNormal(Vector3.UnitX, selectedInstance.Transform));
                                Vector3 axisY = Vector3.Normalize(Vector3.TransformNormal(Vector3.UnitY, selectedInstance.Transform));
                                Vector3 axisZ = Vector3.Normalize(Vector3.TransformNormal(Vector3.UnitZ, selectedInstance.Transform));
                                _bbRenderer.BatchLine(anchor, anchor + axisX * aidScale, new Vector3(1.00f, 0.25f, 0.25f));
                                _bbRenderer.BatchLine(anchor, anchor + axisY * aidScale, new Vector3(0.30f, 1.00f, 0.35f));
                                _bbRenderer.BatchLine(anchor, anchor + axisZ * aidScale, new Vector3(0.40f, 0.55f, 1.00f));
                            }

                            selectionBoundsPreparedCount++;
                        }

                        if (_pm4Overlay._showPm4Overlay
                            && _pm4Overlay._selectedPm4ObjectKey.HasValue
                            && _pm4Overlay._pm4ObjectLookup.TryGetValue(_pm4Overlay._selectedPm4ObjectKey.Value, out Pm4OverlayObject? selectedPm4Object))
                        {
                            Matrix4x4 pm4Transform = _pm4Overlay.BuildPm4OverlayTransformMatrix();
                            bool applyPm4Transform = _pm4Overlay._pm4OverlayTranslation != Vector3.Zero
                                || _pm4Overlay._pm4OverlayRotationDegrees.LengthSquared() > 0.0001f
                                || _pm4Overlay._pm4OverlayScale != Vector3.One;
                            Matrix4x4 objectTransform = _pm4Overlay.BuildPm4ObjectTransform(_pm4Overlay._selectedPm4ObjectKey.Value, applyPm4Transform, pm4Transform, out bool applyObjectTransform);
                            Vector3 boundsMin = selectedPm4Object.BoundsMin;
                            Vector3 boundsMax = selectedPm4Object.BoundsMax;
                            if (applyObjectTransform)
                                TransformBounds(boundsMin, boundsMax, objectTransform, out boundsMin, out boundsMax);

                            _bbRenderer.BatchHighlightedBoxMinMax(
                                boundsMin,
                                boundsMax,
                                selectedBoundsTime,
                                selectedBoundsInnerColor,
                                selectedBoundsAccentA,
                                selectedBoundsAccentB);
                            selectionBoundsPreparedCount++;
                        }
                    }

                    if (_showBoundingBoxes)
                    {
                        var adapter = _terrainManager.Adapter;
                        if (!_renderDiagPrinted)
                            ViewerLog.Debug(ViewerLog.Category.Terrain, $"BB render: {adapter.MddfPlacements.Count} MDDF + {adapter.ModfPlacements.Count} MODF markers");

                        // MDDF bounding boxes (light pastel magenta)
                        foreach (VisibleMdxInstance visible in frame.Visibility.VisibleMdx)
                        {
                            ObjectInstance inst = visible.Instance;
                            _bbRenderer.BatchBoxMinMax(inst.BoundsMin, inst.BoundsMax, Pm4ColorMddfBounds);
                            selectionBoundsPreparedCount++;
                        }
                        // MODF bounding boxes (light pastel cyan)
                        foreach (VisibleWmoInstance visible in frame.Visibility.VisibleWmos)
                        {
                            ObjectInstance inst = visible.Instance;
                            _bbRenderer.BatchBoxMinMax(inst.BoundsMin, inst.BoundsMax, Pm4ColorModfBounds);
                            selectionBoundsPreparedCount++;
                        }
                    }
                });

                if (_pm4Overlay._showPm4ObjectBounds && _pm4Overlay._showPm4Overlay && _pm4Overlay._pm4TileObjects.Count > 0)
                {
                    pm4BoundsMs += MeasureDurationMs(() =>
                    {
                    Matrix4x4 pm4Transform = _pm4Overlay.BuildPm4OverlayTransformMatrix();
                    bool applyPm4Transform = _pm4Overlay._pm4OverlayTranslation != Vector3.Zero
                        || _pm4Overlay._pm4OverlayRotationDegrees.LengthSquared() > 0.0001f
                        || _pm4Overlay._pm4OverlayScale != Vector3.One;

                    foreach (var (tileKey, objects) in _pm4Overlay._pm4TileObjects)
                    {
                        if (!_pm4Overlay.ShouldRenderPm4Tile(tileKey.tileX, tileKey.tileY))
                            continue;

                        foreach (Pm4OverlayObject obj in objects)
                        {
                            if (!_pm4Overlay.ShouldRenderPm4Object(obj))
                                continue;

                            var objectKey = (tileKey.tileX, tileKey.tileY, obj.Ck24, obj.ObjectPartId);
                            Matrix4x4 objectTransform = _pm4Overlay.BuildPm4ObjectTransform(objectKey, applyPm4Transform, pm4Transform, out bool applyObjectTransform);
                            if (!_pm4Overlay.ShouldRenderPm4Object(obj, objectTransform, applyObjectTransform, cameraPos, out _))
                                continue;

                            Vector3 boundsMin = obj.BoundsMin;
                            Vector3 boundsMax = obj.BoundsMax;
                            if (applyObjectTransform)
                                TransformBounds(boundsMin, boundsMax, objectTransform, out boundsMin, out boundsMax);

                            Vector3 boxColor = Pm4ColorObjectBounds;  // light pastel — container
                            if (_pm4Overlay._highlightedPm4ObjectKeys.Contains(objectKey))
                                boxColor = Pm4ColorHighlight;        // saturated — search hit
                            if (_pm4Overlay._selectedPm4ObjectGroupKey.HasValue
                                && _pm4Overlay.IsPm4ObjectInGroup(_pm4Overlay._selectedPm4ObjectGroupKey.Value, objectKey))
                                boxColor = Pm4ColorSelection;        // saturated — selection

                            _bbRenderer.BatchBoxMinMax(boundsMin, boundsMax, boxColor);
                            pm4BoundsPreparedCount++;
                        }
                    }
                    });
                }

                // Placement-Z markers: a flat slab at Z = MSUR._0x1C-as-float under each object.
                if (_pm4Overlay._showPm4PlacementZPlane && _pm4Overlay._showPm4Overlay && _bbRenderer != null && _pm4Overlay._pm4TileObjects.Count > 0)
                {
                    pm4BoundsMs += MeasureDurationMs(() =>
                    {
                        Matrix4x4 markerTransform = _pm4Overlay.BuildPm4OverlayTransformMatrix();
                        bool applyMarkerTransform = _pm4Overlay._pm4OverlayTranslation != Vector3.Zero
                            || _pm4Overlay._pm4OverlayRotationDegrees.LengthSquared() > 0.0001f
                            || _pm4Overlay._pm4OverlayScale != Vector3.One;

                        foreach (var tileEntry in _pm4Overlay._pm4TileObjects)
                        {
                            foreach (Pm4OverlayObject obj in tileEntry.Value)
                            {
                                if (!_pm4Overlay.ShouldRenderPm4Object(obj))
                                    continue;

                                // No recorded height means there is nothing to mark; those objects
                                // would all stack at world Z=0 and say nothing.
                                if (obj.Ck24 == 0)
                                    continue;

                                // Follow the selection unless explicitly asked for all of them. The
                                // marker answers a question about ONE object, and a thousand cubes
                                // at once answers none of them while hiding the geometry.
                                var thisKey = (tileEntry.Key.tileX, tileEntry.Key.tileY, obj.Ck24, obj.ObjectPartId);
                                if (!_pm4Overlay._showPm4PlacementZForAllObjects)
                                {
                                    if (!_pm4Overlay._selectedPm4ObjectGroupKey.HasValue
                                        || !_pm4Overlay.IsPm4ObjectInGroup(_pm4Overlay._selectedPm4ObjectGroupKey.Value, thisKey))
                                    {
                                        continue;
                                    }
                                }

                                var markerKey = (tileEntry.Key.tileX, tileEntry.Key.tileY, obj.Ck24, obj.ObjectPartId);
                                Matrix4x4 objTransform = _pm4Overlay.BuildPm4ObjectTransform(markerKey, applyMarkerTransform, markerTransform, out bool applyObj);

                                Vector3 bMin = obj.BoundsMin;
                                Vector3 bMax = obj.BoundsMax;
                                if (applyObj)
                                    TransformBounds(bMin, bMax, objTransform, out bMin, out bMax);

                                // A SMALL marker at the object's centre, deliberately not the size of
                                // the object. An earlier version spanned the object's XY footprint,
                                // which made every marker look like the object's bounding box and
                                // invited reading _0x1C as encoded bounds. Only the HEIGHT here comes
                                // from the data; any width would come from geometry we already have,
                                // so the marker carries no width worth showing.
                                float z = BitConverter.UInt32BitsToSingle(obj.Ck24 << 8);
                                Vector3 centre = (bMin + bMax) * 0.5f;
                                const float MarkerHalfWidth = 1.25f;
                                Vector3 min = new(centre.X - MarkerHalfWidth, centre.Y - MarkerHalfWidth, z - 0.1f);
                                Vector3 max = new(centre.X + MarkerHalfWidth, centre.Y + MarkerHalfWidth, z + 0.1f);
                                _bbRenderer.BatchBoxMinMax(min, max, new Vector3(0.35f, 0.95f, 0.45f));
                                pm4BoundsPreparedCount++;
                            }
                        }
                    });
                }

                // CK24-level bounding boxes: one merged box per CK24 object across all sub-objects.
                if (_pm4Overlay._showPm4Ck24Bounds && _pm4Overlay._showPm4Overlay && _pm4Overlay._pm4TileObjects.Count > 0)
                {
                    pm4BoundsMs += MeasureDurationMs(() =>
                    {
                    Matrix4x4 pm4Transform = _pm4Overlay.BuildPm4OverlayTransformMatrix();
                    bool applyPm4Transform = _pm4Overlay._pm4OverlayTranslation != Vector3.Zero
                        || _pm4Overlay._pm4OverlayRotationDegrees.LengthSquared() > 0.0001f
                        || _pm4Overlay._pm4OverlayScale != Vector3.One;

                    // Group objects by (tileX, tileY, Ck24) to merge sub-objects into one box per CK24.
                    var ck24Groups = new Dictionary<(int tileX, int tileY, uint ck24), (Vector3 min, Vector3 max, byte ck24Type, int count)>();

                    foreach (var (tileKey, objects) in _pm4Overlay._pm4TileObjects)
                    {
                        if (!_pm4Overlay.ShouldRenderPm4Tile(tileKey.tileX, tileKey.tileY))
                            continue;

                        foreach (Pm4OverlayObject obj in objects)
                        {
                            if (!_pm4Overlay.ShouldRenderPm4ObjectType(obj.Ck24Type))
                                continue;

                            var ck24Key = (tileKey.tileX, tileKey.tileY, obj.Ck24);
                            var objectKey = (tileKey.tileX, tileKey.tileY, obj.Ck24, obj.ObjectPartId);
                            Matrix4x4 objectTransform = _pm4Overlay.BuildPm4ObjectTransform(objectKey, applyPm4Transform, pm4Transform, out bool applyObjectTransform);
                            if (!_pm4Overlay.ShouldRenderPm4Object(obj, objectTransform, applyObjectTransform, cameraPos, out _))
                                continue;

                            Vector3 boundsMin = obj.BoundsMin;
                            Vector3 boundsMax = obj.BoundsMax;
                            if (applyObjectTransform)
                                TransformBounds(boundsMin, boundsMax, objectTransform, out boundsMin, out boundsMax);

                            if (ck24Groups.TryGetValue(ck24Key, out var existing))
                            {
                                ck24Groups[ck24Key] = (
                                    Vector3.Min(existing.min, boundsMin),
                                    Vector3.Max(existing.max, boundsMax),
                                    existing.ck24Type,
                                    existing.count + 1);
                            }
                            else
                            {
                                ck24Groups[ck24Key] = (boundsMin, boundsMax, obj.Ck24Type, 1);
                            }
                        }
                    }

                    // Render one box per CK24 object.
                    foreach (var ((tileX, tileY, ck24), (boundsMin, boundsMax, ck24Type, count)) in ck24Groups)
                    {
                        // Light pastel — CK24 container color, varied by ck24Type for at-a-glance discrimination
                        Vector3 boxColor = ck24Type switch
                        {
                            0x00 => new Vector3(0.65f, 0.65f, 0.75f),  // nav mesh: light pastel blue-gray
                            0x40 or 0x41 => new Vector3(1.00f, 0.95f, 0.65f),  // M2: light pastel yellow
                            0x42 or 0x43 => new Vector3(0.65f, 0.95f, 0.95f),  // WMO: light pastel cyan
                            0xC0 or 0xC1 or 0xC2 or 0xC3 => new Vector3(1.00f, 0.75f, 0.60f),  // M2 exterior: light pastel orange
                            _ => new Vector3(0.80f, 0.80f, 0.80f)  // unknown: light pastel gray
                        };

                        // Highlight the selected object's CK24 group.
                        if (_pm4Overlay._selectedPm4ObjectKey.HasValue
                            && _pm4Overlay._selectedPm4ObjectKey.Value.tileX == tileX
                            && _pm4Overlay._selectedPm4ObjectKey.Value.tileY == tileY
                            && _pm4Overlay._selectedPm4ObjectKey.Value.ck24 == ck24)
                        {
                            boxColor = Pm4ColorSelectedBounds;  // light pastel white — clear container
                        }

                        _bbRenderer.BatchBoxMinMax(boundsMin, boundsMax, boxColor);
                        pm4BoundsPreparedCount++;
                    }
                    });
                }

                if (_pm4Overlay._showPm4GeneratedPlacements)
                {
                    foreach (Pm4GeneratedPlacements.RecoveredBox box in Pm4GeneratedPlacements.Boxes)
                    {
                        if (_pm4Overlay._pm4GeneratedPlacementsTerrainlessOnly && !box.OnTileWithoutTerrain)
                            continue;

                        // Never box an object that is already there. This overlay exists to show what
                        // is MISSING; drawing a cage around a WMO the scene has already placed and
                        // named adds nothing and buries the geometry underneath it.
                        if (HasRealPlacementInside(box.Min, box.Max))
                            continue;

                        // Green where the shape match is tight, amber where it is loose. A recovered
                        // name is a guess and the colour has to say so without a label being read.
                        Vector3 color = box.Score <= 0.05 ? new Vector3(0.35f, 0.95f, 0.45f)
                            : box.Score <= 0.15 ? new Vector3(0.75f, 0.95f, 0.40f)
                            : new Vector3(0.98f, 0.75f, 0.30f);

                        _bbRenderer.BatchBoxMinMax(box.Min, box.Max, color);
                        pm4BoundsPreparedCount++;
                    }
                }

                _bbRenderer.FlushBatch(view, proj);

                _gl.DepthMask(true);
            }

            // 5+6. Batched overlay rendering (POI pins + taxi paths) — single draw call
            if (_bbRenderer != null)
            {
                _bbRenderer.BeginBatch();
                _bbRenderer.BeginSolidBatch();

                _pm4Overlay._pm4VisibleObjectCount = 0;
                _pm4Overlay._pm4VisibleLineCount = 0;
                _pm4Overlay._pm4VisibleTriangleCount = 0;
                _pm4Overlay._pm4VisiblePositionRefCount = 0;

                pm4GeometryPrepareMs = MeasureDurationMs(() =>
                {
                if (_pm4Overlay._showPm4Overlay && _pm4Overlay._pm4TileObjects.Count > 0)
                {
                    Matrix4x4 pm4Transform = _pm4Overlay.BuildPm4OverlayTransformMatrix();
                    bool applyPm4Transform = _pm4Overlay._pm4OverlayTranslation != Vector3.Zero
                        || _pm4Overlay._pm4OverlayRotationDegrees.LengthSquared() > 0.0001f
                        || _pm4Overlay._pm4OverlayScale != Vector3.One;

                    foreach (var (tileKey, objects) in _pm4Overlay._pm4TileObjects)
                    {
                        if (!_pm4Overlay.ShouldRenderPm4Tile(tileKey.tileX, tileKey.tileY))
                            continue;

                        if (_pm4Overlay._showPm4PositionRefs
                            && _pm4Overlay._pm4TilePositionRefs.TryGetValue(tileKey, out List<Vector3>? positionRefs)
                            && positionRefs.Count > 0)
                        {
                            for (int i = 0; i < positionRefs.Count; i++)
                            {
                                Vector3 marker = applyPm4Transform ? ApplyPm4OverlayTransform(positionRefs[i], pm4Transform) : positionRefs[i];
                                _bbRenderer.BatchPin(marker, 16f, 3f, Pm4ColorMprl);
                            }

                            _pm4Overlay._pm4VisiblePositionRefCount += positionRefs.Count;
                        }

                        foreach (Pm4OverlayObject obj in objects)
                        {
                            if (!_pm4Overlay.ShouldRenderPm4ObjectType(obj.Ck24Type))
                                continue;

                            var objectKey = (tileKey.tileX, tileKey.tileY, obj.Ck24, obj.ObjectPartId);
                            Matrix4x4 objectTransform = _pm4Overlay.BuildPm4ObjectTransform(objectKey, applyPm4Transform, pm4Transform, out bool applyObjectTransform);
                            Matrix4x4 geometryTransform = BuildPm4GeometryTransform(obj, objectTransform, applyObjectTransform);

                            if (!_pm4Overlay.ShouldRenderPm4Object(obj, objectTransform, applyObjectTransform, cameraPos, out Vector3 transformedCenter))
                                continue;

                            _pm4Overlay._pm4VisibleObjectCount++;
                            Vector3 pm4Color = _pm4Overlay.GetPm4ObjectColor(tileKey, obj);
                            if (_pm4Overlay._highlightedPm4ObjectKeys.Contains(objectKey))
                                pm4Color = Pm4ColorHighlight;  // saturated teal — search hit
                            if (_pm4Overlay._selectedPm4ObjectGroupKey.HasValue
                                && _pm4Overlay.IsPm4ObjectInGroup(_pm4Overlay._selectedPm4ObjectGroupKey.Value, objectKey))
                                pm4Color = Pm4ColorSelection;  // saturated yellow — selection

                            if (_pm4Overlay._showPm4SolidOverlay && obj.Triangles.Count > 0)
                            {
                                for (int i = 0; i < obj.Triangles.Count; i++)
                                {
                                    Pm4Triangle tri = obj.Triangles[i];
                                    Vector3 a = ApplyPm4OverlayTransform(tri.A, geometryTransform);
                                    Vector3 b = ApplyPm4OverlayTransform(tri.B, geometryTransform);
                                    Vector3 c = ApplyPm4OverlayTransform(tri.C, geometryTransform);
                                    _bbRenderer.BatchTriangle(a, b, c, pm4Color, 0.20f);
                                }
                                _pm4Overlay._pm4VisibleTriangleCount += obj.Triangles.Count;
                            }

                            for (int i = 0; i < obj.Lines.Count; i++)
                            {
                                Pm4LineSegment line = obj.Lines[i];
                                Vector3 from = ApplyPm4OverlayTransform(line.From, geometryTransform);
                                Vector3 to = ApplyPm4OverlayTransform(line.To, geometryTransform);
                                _bbRenderer.BatchLine(from, to, pm4Color);
                            }

                            _pm4Overlay._pm4VisibleLineCount += obj.Lines.Count;

                            if (_pm4Overlay._showPm4ObjectCentroids)
                            {
                                // Centroid is a per-object marker, not mesh — use dedicated dark pastel
                                _bbRenderer.BatchPin(transformedCenter, 22f, 4f, Pm4ColorCentroid);
                            }
                        }
                    }
                }
                });
                pm4GeometryPreparedCount = _pm4Overlay._pm4VisibleLineCount + _pm4Overlay._pm4VisibleTriangleCount;
                pm4GeometrySubmittedCount = pm4GeometryPreparedCount;

                if (_pm4Overlay._showPm4Overlay)
                {
                    bool pm4IgnoreDepth = _pm4Overlay._pm4OverlayIgnoreDepth;

                    if (_pm4Overlay._showPm4SolidOverlay && _pm4Overlay._pm4VisibleTriangleCount > 0)
                    {
                        _gl.Enable(EnableCap.Blend);
                        _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
                        if (pm4IgnoreDepth)
                        {
                            _gl.Disable(EnableCap.DepthTest);
                        }
                        else
                        {
                            _gl.Enable(EnableCap.DepthTest);
                            _gl.DepthFunc(DepthFunction.Lequal);
                        }

                        _gl.DepthMask(false);
                        _gl.Disable(EnableCap.CullFace);
                        pm4GeometrySubmitMs += MeasureDurationMs(() => _bbRenderer.FlushSolidBatch(view, proj));
                        _gl.Enable(EnableCap.CullFace);
                        _gl.Disable(EnableCap.Blend);
                    }

                    // MSCN/MSPV node markers — solid filled cubes. Bright saturated colors
                    // (cyan/magenta) so they pop against the pastel mesh.
                    pm4NodesMs += MeasureDurationMs(() =>
                    {
                    if (_pm4Overlay._pm4RenderNodesAsCubes && (_pm4Overlay._showPm4MscnNodes || _pm4Overlay._showPm4MspvNodes))
                    {
                        if (_pm4Overlay._showPm4MscnNodes)
                            _pm4Overlay.EnsurePm4MscnData();
                        if (_pm4Overlay._showPm4MspvNodes)
                            _pm4Overlay.EnsurePm4MspvData();

                        if (_pm4Overlay._showPm4SolidOverlay || _pm4Overlay._pm4VisibleTriangleCount == 0)
                        {
                            _gl.Enable(EnableCap.Blend);
                            _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
                            if (pm4IgnoreDepth)
                                _gl.Disable(EnableCap.DepthTest);
                            else
                            {
                                _gl.Enable(EnableCap.DepthTest);
                                _gl.DepthFunc(DepthFunction.Lequal);
                            }
                            _gl.DepthMask(false);
                            _gl.Disable(EnableCap.CullFace);

                            int mscnDrawn = 0;
                            if (_pm4Overlay._showPm4MscnNodes && _pm4Overlay._pm4TileMscnPoints.Count > 0)
                            {
                                int limit = 15000;
                                foreach (var kv in _pm4Overlay._pm4TileMscnPoints)
                                {
                                    var pts = kv.Value;
                                    if (pts == null) continue;
                                    for (int i = 0; i < pts.Count && mscnDrawn < limit; i++)
                                    {
                                        _bbRenderer.BatchSolidCube(pts[i], _pm4Overlay._pm4MscnCubeSize, Pm4ColorMscn, _pm4Overlay._pm4MscnCubeAlpha);
                                        mscnDrawn++;
                                    }
                                }
                            }

                            int mspvDrawn = 0;
                            if (_pm4Overlay._showPm4MspvNodes && _pm4Overlay._pm4TileMspvPoints.Count > 0)
                            {
                                int limit = 8000;
                                foreach (var kv in _pm4Overlay._pm4TileMspvPoints)
                                {
                                    var pts = kv.Value;
                                    if (pts == null) continue;
                                    for (int i = 0; i < pts.Count && mspvDrawn < limit; i++)
                                    {
                                        _bbRenderer.BatchSolidCube(pts[i], _pm4Overlay._pm4MspvCubeSize, Pm4ColorMspv, _pm4Overlay._pm4MspvCubeAlpha);
                                        mspvDrawn++;
                                    }
                                }
                            }

                            if (mscnDrawn > 0 || mspvDrawn > 0)
                            {
                                _bbRenderer.FlushSolidBatch(view, proj);
                                _pm4Overlay._pm4VisiblePositionRefCount += mscnDrawn + mspvDrawn;
                                pm4NodesPreparedCount += mscnDrawn + mspvDrawn;
                            }
                            _gl.Enable(EnableCap.CullFace);
                            _gl.Disable(EnableCap.Blend);
                        }
                    }
                    });

                    bool hasPm4LineGeometry = _pm4Overlay._pm4VisibleLineCount > 0
                        || _pm4Overlay._pm4VisiblePositionRefCount > 0
                        || (_pm4Overlay._showPm4ObjectCentroids && _pm4Overlay._pm4VisibleObjectCount > 0);
                    if (hasPm4LineGeometry)
                    {
                        _gl.LineWidth(_pm4Overlay._pm4WireframeLineWidth);
                        if (pm4IgnoreDepth)
                        {
                            _gl.Disable(EnableCap.DepthTest);
                        }
                        else
                        {
                            _gl.Enable(EnableCap.DepthTest);
                            _gl.DepthFunc(DepthFunction.Lequal);
                        }

                        // MSCN node overlay (legacy wireframe pin mode)
                        if (!_pm4Overlay._pm4RenderNodesAsCubes && _pm4Overlay._showPm4MscnNodes && _pm4Overlay._pm4TileMscnPoints.Count > 0)
                        {
                            pm4NodesMs += MeasureDurationMs(() =>
                            {
                                _pm4Overlay.EnsurePm4MscnData();
                                int limit = 15000, drawn = 0;
                                foreach (var kv in _pm4Overlay._pm4TileMscnPoints)
                                {
                                    var pts = kv.Value;
                                    if (pts == null) continue;
                                    for (int i = 0; i < pts.Count && drawn < limit; i++)
                                    {
                                        _bbRenderer.BatchPin(pts[i], _pm4Overlay._pm4MscnCubeSize * 2.0f, _pm4Overlay._pm4MscnCubeSize * 0.5f, Pm4ColorMscn);
                                        drawn++;
                                    }
                                }
                                _pm4Overlay._pm4VisiblePositionRefCount += drawn;
                                pm4NodesPreparedCount += drawn;
                            });
                        }
                        if (!_pm4Overlay._pm4RenderNodesAsCubes && _pm4Overlay._showPm4MspvNodes && _pm4Overlay._pm4TileMspvPoints.Count > 0)
                        {
                            pm4NodesMs += MeasureDurationMs(() =>
                            {
                                _pm4Overlay.EnsurePm4MspvData();
                                int limit = 8000, drawn = 0;
                                foreach (var kv in _pm4Overlay._pm4TileMspvPoints)
                                {
                                    var pts = kv.Value;
                                    if (pts == null) continue;
                                    for (int i = 0; i < pts.Count && drawn < limit; i++)
                                    {
                                        _bbRenderer.BatchPin(pts[i], _pm4Overlay._pm4MspvCubeSize * 2.0f, _pm4Overlay._pm4MspvCubeSize * 0.5f, Pm4ColorMspv);
                                        drawn++;
                                    }
                                }
                                _pm4Overlay._pm4VisiblePositionRefCount += drawn;
                                pm4NodesPreparedCount += drawn;
                            });
                        }

                        _gl.DepthMask(false);
                        pm4GeometrySubmitMs += MeasureDurationMs(() => _bbRenderer.FlushBatch(view, proj));
                        _gl.LineWidth(1.0f);
                    }

                    // Reset default state and clear PM4 primitives so other overlays use their normal pass.
                    _gl.Enable(EnableCap.DepthTest);
                    _gl.DepthFunc(DepthFunction.Lequal);
                    _gl.DepthMask(true);
                    _gl.Disable(EnableCap.Blend);

                    _bbRenderer.BeginBatch();
                    _bbRenderer.BeginSolidBatch();
                }

                poiTaxiMs = MeasureDurationMs(() =>
                {
                // POI pin markers (magenta)
                if (_showPoi && _poiLoader != null && _poiLoader.Entries.Count > 0)
                {
                    var poiColor = new Vector3(1f, 0f, 1f);
                    foreach (var poi in _poiLoader.Entries)
                    {
                        _bbRenderer.BatchPin(poi.Position, 56f, 9f, poiColor);
                        poiTaxiPreparedCount++;
                    }
                }

                // Taxi paths — filtered by selection
                if (_showTaxi && _taxiLoader != null)
                {
                    var nodeColor = new Vector3(1f, 1f, 0f);
                    var lineColor = new Vector3(0f, 1f, 1f);
                    var routeHandleColor = new Vector3(1f, 0.65f, 0f);
                    var selectedRouteColor = new Vector3(1f, 1f, 1f);
                    var nodeBoxColor = new Vector3(1f, 0.92f, 0.35f);
                    var routeBoxColor = new Vector3(1f, 0.78f, 0.28f);
                    int visibleRouteCount = _taxiLoader.Routes.Count(IsTaxiRouteVisible);
                    bool showRouteHandles = _selectedTaxiNodeId >= 0 || _selectedTaxiRouteId >= 0 || visibleRouteCount <= 32;

                    foreach (var node in _taxiLoader.Nodes)
                    {
                        if (!IsTaxiNodeVisible(node)) continue;
                        _bbRenderer.BatchPin(node.Position, 64f, 12f, nodeColor);
                        poiTaxiPreparedCount++;
                        _bbRenderer.BatchBoxMinMax(
                            node.Position - new Vector3(36f, 36f, 18f),
                            node.Position + new Vector3(36f, 36f, 96f),
                            nodeBoxColor);
                        poiTaxiPreparedCount++;
                    }

                    foreach (var route in _taxiLoader.Routes)
                    {
                        if (!IsTaxiRouteVisible(route)) continue;
                        Vector3 routeColor = route.PathId == _selectedTaxiRouteId ? selectedRouteColor : lineColor;
                        for (int i = 0; i < route.Waypoints.Count - 1; i++)
                        {
                            _bbRenderer.BatchLine(route.Waypoints[i], route.Waypoints[i + 1], routeColor);
                            poiTaxiPreparedCount++;
                        }

                        if (showRouteHandles && TryGetTaxiRouteSelectionPoint(route, out Vector3 selectionPoint))
                        {
                            float pinHeight = route.PathId == _selectedTaxiRouteId ? 64f : 52f;
                            float headSize = route.PathId == _selectedTaxiRouteId ? 12f : 10f;
                            _bbRenderer.BatchPin(selectionPoint, pinHeight, headSize,
                                route.PathId == _selectedTaxiRouteId ? selectedRouteColor : routeHandleColor);
                            poiTaxiPreparedCount++;
                            _bbRenderer.BatchBoxMinMax(
                                selectionPoint - new Vector3(34f, 34f, 20f),
                                selectionPoint + new Vector3(34f, 34f, 72f),
                                route.PathId == _selectedTaxiRouteId ? selectedRouteColor : routeBoxColor);
                            poiTaxiPreparedCount++;
                        }
                    }
                }
                });

                // AreaTriggers (green wireframe shapes for portals and event markers)
                areaTriggersMs = MeasureDurationMs(() =>
                {
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
                            areaTriggerPreparedCount += segments * 3;
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
                            areaTriggerPreparedCount += 12;
                        }
                    }
                }
                });

                if (_showLitLights && _litLoader != null && _litLoader.HasData)
                {
                    int highlightedLightIndex = _selectedLitLightIndex >= 0
                        ? _selectedLitLightIndex
                        : _lastLitSample?.DominantLightIndex ?? -1;

                    for (int lightIndex = 0; lightIndex < _litLoader.Lights.Count; lightIndex++)
                    {
                        LitLoader.LitLight light = _litLoader.Lights[lightIndex];
                        if (!light.HasMeaningfulPosition)
                            continue;

                        Vector3 lightColor = _litLoader.EvaluateOverlayColor(light, lighting.GameTime);
                        bool isHighlighted = lightIndex == highlightedLightIndex;
                        float pinHeight = isHighlighted ? 60f : 36f;
                        float headSize = isHighlighted ? 8f : 5f;
                        _bbRenderer.BatchPin(light.Position, pinHeight, headSize,
                            isHighlighted ? new Vector3(1f, 1f, 1f) : lightColor);

                        float footprintRadius = Math.Max(light.Radius, 6f);
                        float footprintHeight = Math.Max(8f, Math.Min(light.Dropoff, 80f));
                        var min = new Vector3(light.Position.X - footprintRadius, light.Position.Y - footprintRadius, light.Position.Z - footprintHeight * 0.25f);
                        var max = new Vector3(light.Position.X + footprintRadius, light.Position.Y + footprintRadius, light.Position.Z + footprintHeight * 0.25f);
                        _bbRenderer.BatchBoxMinMax(min, max,
                            isHighlighted ? new Vector3(1f, 1f, 1f) : lightColor);
                    }
                }

                if (_showAreaRegionOverlay && _areaOverlayRegions.Count > 0)
                {
                    foreach (AreaOverlayRegion region in _areaOverlayRegions)
                    {
                        foreach (AreaOverlayFootprintCell cell in region.Cells)
                        {
                            Vector3 min = cell.BoundsMin;
                            Vector3 max = cell.BoundsMax;
                            float z = MathF.Max(min.Z, max.Z) + 1.5f;
                            Vector3 v0 = new(MathF.Min(min.X, max.X), MathF.Min(min.Y, max.Y), z);
                            Vector3 v1 = new(MathF.Max(min.X, max.X), MathF.Min(min.Y, max.Y), z);
                            Vector3 v2 = new(MathF.Max(min.X, max.X), MathF.Max(min.Y, max.Y), z);
                            Vector3 v3 = new(MathF.Min(min.X, max.X), MathF.Max(min.Y, max.Y), z);
                            _bbRenderer.BatchLine(v0, v1, region.Color);
                            _bbRenderer.BatchLine(v1, v2, region.Color);
                            _bbRenderer.BatchLine(v2, v3, region.Color);
                            _bbRenderer.BatchLine(v3, v0, region.Color);
                        }

                        _bbRenderer.BatchPin(region.LabelPosition, 30f, 5f, region.Color);
                    }
                }

                audioEmitterMarkersMs = MeasureDurationMs(() =>
                {
                    if (!_showAudioEmitterMarkers || _audioRuntime is null)
                        return;

                    foreach (TerrainSoundEmitter emitter in _audioRuntime.ResidentEmitters)
                    {
                        Vector3 position = emitter.Position;
                        if (!float.IsFinite(position.X)
                            || !float.IsFinite(position.Y)
                            || !float.IsFinite(position.Z))
                            continue;

                        Vector3 color = emitter.TriggerKind switch
                        {
                            AudioTriggerKind.McnkLiquid when emitter.LiquidFamily >= 0
                                => new Vector3(0.15f, 0.8f, 1.0f),
                            AudioTriggerKind.McnkLiquid
                                => new Vector3(0.7f, 0.35f, 1.0f),
                            _ => new Vector3(1.0f, 0.62f, 0.12f)
                        };

                        float pinHeight = emitter.TriggerKind == AudioTriggerKind.McnkLiquid ? 24f : 20f;
                        float headSize = emitter.TriggerKind == AudioTriggerKind.McnkLiquid ? 4.5f : 4f;
                        _bbRenderer.BatchPin(position, pinHeight, headSize, color);
                        audioEmitterMarkerPreparedCount++;
                    }
                });

                _gl.LineWidth(5.0f);
                _bbRenderer.FlushBatch(view, proj);
                _gl.LineWidth(1.0f);
            }
                    });

                    frame.SetOverlayOwner(
                        WorldOverlayOwners.ObjectWireframe,
                        objectWireframeMs,
                        objectWireframeEnabled,
                        objectWireframePreparedCount,
                        objectWireframeSubmittedCount,
                        objectWireframeEnabled ? "not_cached" : "disabled");
                    frame.SetOverlayOwner(
                        WorldOverlayOwners.SelectionBounds,
                        selectionBoundsMs,
                        _bbRenderer != null && (_showSelectedObjectBounds || _showBoundingBoxes),
                        selectionBoundsPreparedCount,
                        selectionBoundsPreparedCount,
                        _bbRenderer != null && (_showSelectedObjectBounds || _showBoundingBoxes) ? "not_cached" : "disabled");
                    frame.SetOverlayOwner(
                        WorldOverlayOwners.Pm4Bounds,
                        pm4BoundsMs,
                        _bbRenderer != null && _pm4Overlay._showPm4Overlay && (_pm4Overlay._showPm4ObjectBounds || _pm4Overlay._showPm4Ck24Bounds || _pm4Overlay._showPm4PlacementZPlane),
                        pm4BoundsPreparedCount,
                        pm4BoundsPreparedCount,
                        _bbRenderer != null && _pm4Overlay._showPm4Overlay && (_pm4Overlay._showPm4ObjectBounds || _pm4Overlay._showPm4Ck24Bounds || _pm4Overlay._showPm4PlacementZPlane) ? "not_cached" : "disabled");
                    frame.SetOverlayOwner(
                        WorldOverlayOwners.Pm4GeometryPrepare,
                        pm4GeometryPrepareMs,
                        _bbRenderer != null && _pm4Overlay._showPm4Overlay,
                        pm4GeometryPreparedCount,
                        0,
                        _bbRenderer != null && _pm4Overlay._showPm4Overlay ? "not_cached" : "disabled");
                    frame.SetOverlayOwner(
                        WorldOverlayOwners.Pm4GeometrySubmit,
                        pm4GeometrySubmitMs,
                        _bbRenderer != null && _pm4Overlay._showPm4Overlay,
                        pm4GeometryPreparedCount,
                        pm4GeometrySubmittedCount,
                        _bbRenderer != null && _pm4Overlay._showPm4Overlay ? "not_cached" : "disabled");
                    frame.SetOverlayOwner(
                        WorldOverlayOwners.Pm4Nodes,
                        pm4NodesMs,
                        _bbRenderer != null && _pm4Overlay._showPm4Overlay && (_pm4Overlay._showPm4MscnNodes || _pm4Overlay._showPm4MspvNodes),
                        pm4NodesPreparedCount,
                        pm4NodesPreparedCount,
                        _bbRenderer != null && _pm4Overlay._showPm4Overlay && (_pm4Overlay._showPm4MscnNodes || _pm4Overlay._showPm4MspvNodes) ? "not_cached" : "disabled");
                    frame.SetOverlayOwner(
                        WorldOverlayOwners.PoiTaxi,
                        poiTaxiMs,
                        _bbRenderer != null && (_showPoi || _showTaxi),
                        poiTaxiPreparedCount,
                        poiTaxiPreparedCount,
                        _bbRenderer != null && (_showPoi || _showTaxi) ? "not_cached" : "disabled");
                    frame.SetOverlayOwner(
                        WorldOverlayOwners.AreaTriggers,
                        areaTriggersMs,
                        _bbRenderer != null && _showAreaTriggers,
                        areaTriggerPreparedCount,
                        areaTriggerPreparedCount,
                        _bbRenderer != null && _showAreaTriggers ? "not_cached" : "disabled");
                    frame.SetOverlayOwner(
                        WorldOverlayOwners.AudioEmitters,
                        audioEmitterMarkersMs,
                        _bbRenderer != null && _showAudioEmitterMarkers,
                        audioEmitterMarkerPreparedCount,
                        audioEmitterMarkerPreparedCount,
                        _bbRenderer != null && _showAudioEmitterMarkers ? "not_cached" : "disabled");

                    double accountedOverlayMs = frame.OverlayOwnerDurationSum;
                    double otherOverlayMs = Math.Max(0, overlayElapsedMs - accountedOverlayMs);
                    frame.SetOverlayOwner(
                        WorldOverlayOwners.OtherOverlay,
                        otherOverlayMs,
                        otherOverlayMs > 0,
                        cacheStatus: otherOverlayMs > 0 ? "not_applicable" : "disabled");
                    frame.OverlayMs = frame.OverlayOwnerDurationSum;
                }));

        // Boundary probe: the coordinator has returned, so anything after this is epilogue.
        double postCoordinatorMs = frameTimer.Elapsed.TotalMilliseconds;

        if (!continuedPastTerrain)
        {
            RecordRenderRegionBreakdown(
                frame, preCoordinatorMs, frameTimer.Elapsed.TotalMilliseconds,
                frameTimer.Elapsed.TotalMilliseconds);
            FinalizeRenderFrameStats(frame, frameTimer, view);
            return;
        }

        if (!_doodadsVisible && !_renderDiagPrinted)
            _renderDiagPrinted = true;

        RecordRenderRegionBreakdown(
            frame, preCoordinatorMs, postCoordinatorMs, frameTimer.Elapsed.TotalMilliseconds);
        FinalizeRenderFrameStats(frame, frameTimer, view);
    }

    private void RenderSkyboxBackdrop(Matrix4x4 view, Matrix4x4 proj, Vector3 cameraPos,
        Vector3 fogColor, float fogStart, float fogEnd, TerrainLighting lighting)
    {
        bool renderedActiveClientSky = false;
        if (_skyDome.NightVisibility > 0.001f
            && TryGetQueuedMdx(_activeLightSkyboxModelKey ?? string.Empty) is { } lightSkyboxRenderer)
        {
            lightSkyboxRenderer.UpdateAnimation();
            lightSkyboxRenderer.RenderBackdrop(Matrix4x4.CreateTranslation(cameraPos), view, proj,
                fogColor, fogStart, fogEnd, cameraPos,
                lighting.LightDirection, lighting.LightColor, lighting.AmbientColor);
            renderedActiveClientSky = true;
        }

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
        if (renderedActiveClientSky
            && string.Equals(skybox.ModelKey, _activeLightSkyboxModelKey, StringComparison.OrdinalIgnoreCase))
        {
            return;
        }

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
        return WorldSkyboxBackdropClassifier.IsBackdropModelPath(modelPath);
    }

    private void UpdateActiveSkyboxModel()
    {
        string? sourcePath = _lightService?.ActiveSkyboxModelPath;
        sourcePath = ResolveClientSkyboxPath(sourcePath);
        if (string.IsNullOrWhiteSpace(sourcePath))
            sourcePath = ResolveClientStarsFallback();

        if (string.Equals(sourcePath, _activeLightSkyboxSourcePath, StringComparison.OrdinalIgnoreCase))
            return;

        _activeLightSkyboxSourcePath = sourcePath;
        _activeLightSkyboxModelKey = string.IsNullOrWhiteSpace(sourcePath)
            ? null
            : WorldAssetManager.NormalizeKey(sourcePath);

        if (string.IsNullOrWhiteSpace(_activeLightSkyboxModelKey))
            return;

        _assets.PrioritizeMdxLoad(_activeLightSkyboxModelKey);
        ViewerLog.Info(
            ViewerLog.Category.Mdx,
            $"[Sky] Active client sky model: {sourcePath} (source={(_lightService?.ActiveSkyboxModelPath is null ? "client-stars fallback" : "LightSkybox DBC")})");
    }

    private string? ResolveClientSkyboxPath(string? sourcePath)
    {
        if (string.IsNullOrWhiteSpace(sourcePath) || _dataSource == null)
            return null;

        if (_dataSource.FileExists(sourcePath))
            return sourcePath;

        if (!string.IsNullOrWhiteSpace(Path.GetExtension(sourcePath)))
            return null;

        foreach (string extension in new[] { ".m2", ".mdx", ".mdl" })
        {
            string candidate = sourcePath + extension;
            if (_dataSource.FileExists(candidate))
                return candidate;
        }

        return null;
    }

    private string? ResolveClientStarsFallback()
    {
        if (_clientStarsProbeComplete)
            return _clientStarsFallbackModelPath;

        _clientStarsProbeComplete = true;
        if (_dataSource == null)
            return null;

        string[] candidates =
        [
            @"Environments\Stars\Stars.m2",
            @"Environments\Stars\Stars.mdx",
            @"Environments\Stars\Stars.mdl",
        ];
        foreach (string path in candidates)
        {
            if (!_dataSource.FileExists(path))
                continue;

            _clientStarsFallbackModelPath = path;
            ViewerLog.Info(ViewerLog.Category.Mdx, $"[Sky] Discovered client stars fallback: {path}");
            return path;
        }

        // Some extracted clients retain a World prefix around the same asset.
        foreach (string path in new[]
        {
            @"World\Environments\Stars\Stars.m2",
            @"World\Environments\Stars\Stars.mdx",
            @"World\Environments\Stars\Stars.mdl",
        })
        {
            if (_dataSource.FileExists(path))
            {
                _clientStarsFallbackModelPath = path;
                ViewerLog.Info(ViewerLog.Category.Mdx, $"[Sky] Discovered client stars fallback: {path}");
                return path;
            }
        }

        ViewerLog.Debug(ViewerLog.Category.Mdx, "[Sky] Client stars fallback was not present in the data source file list.");
        return null;
    }

    public void ToggleWireframe()
    {
        bool enable = !IsWireframe;
        SetTerrainWireframeEnabled(enable);
        SetObjectWireframeEnabled(enable);
    }

    public void SetTerrainWireframeEnabled(bool enabled)
    {
        if (_terrainManager.IsWireframe == enabled)
            return;

        _terrainManager.ToggleWireframe();
    }

    public void SetObjectWireframeEnabled(bool enabled)
    {
        if (_assets.ObjectWireframeEnabled == enabled && !_wireframeRevealEnabled)
            return;

        _wireframeRevealEnabled = false;
        _assets.SetObjectWireframeEnabled(enabled);
        ClearWireframeReveal();
    }

    public bool IsWireframe => TerrainWireframeEnabled || ObjectWireframeEnabled;

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

        if (hasSceneRayHit || hasPm4RayHit)
        {
            const float rayDistanceEpsilon = 0.01f;
            if (hasPm4RayHit && (!hasSceneRayHit || _pm4Overlay._pm4OverlayIgnoreDepth || pm4RayDistance < sceneRayDistance - rayDistanceEpsilon))
            {
                _hoveredAssetInfo = pm4RayInfo.WithPreciseRayHit();
                return;
            }

            if (hasSceneRayHit)
            {
                _hoveredAssetInfo = sceneRayInfo.WithPreciseRayHit();
                return;
            }
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

        if (hasPm4BrushHit && (!hasSceneBrushHit || _pm4Overlay.ShouldPreferPm4HoverBrush(pm4BrushDistanceSq, pm4BrushDepth, sceneBrushDistanceSq, sceneBrushDepth)))
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
        float currentBestDistanceSq = float.MaxValue;
        float currentBestDepth = float.MaxValue;
        bestDistanceSq = float.MaxValue;
        bestDepth = float.MaxValue;

        HoveredAssetInfo? bestInfo = null;
        int hitCount = 0;
        LiquidRenderer? liquidRenderer = _terrainManager?.LiquidRenderer;

        void ConsiderCandidate(HoveredAssetInfo candidateInfo, float distanceSq, float depth)
        {
            if (!IsHoverPickPositionAllowed(candidateInfo.WorldPosition))
                return;

            hitCount++;

            const float distanceEpsilon = 0.01f;
            if (!bestInfo.HasValue
                || distanceSq < currentBestDistanceSq - distanceEpsilon
                || (MathF.Abs(distanceSq - currentBestDistanceSq) <= distanceEpsilon && depth < currentBestDepth))
            {
                bestInfo = candidateInfo;
                currentBestDistanceSq = distanceSq;
                currentBestDepth = depth;
            }
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

        if (!bestInfo.HasValue)
            return false;

        HoveredAssetInfo bestCandidate = bestInfo.Value;
        bestDistanceSq = currentBestDistanceSq;
        bestDepth = currentBestDepth;
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
        var visibleSceneHits = sceneHits
            .Where(hit => hit.ObjectType switch
            {
                ObjectType.Wmo or ObjectType.WmoDoodad => _wmosVisible,
                ObjectType.Mdx => _doodadsVisible,
                _ => false,
            })
            .ToList();
        IReadOnlyList<SceneObjectPickHit> filteredSceneHits = ApplyWmoContainerFallThrough(visibleSceneHits);
        SceneObjectPickHit? bestSceneHit = filteredSceneHits
            .Where(hit => IsHoverPickDistanceAllowed(hit.Distance))
            .OrderBy(static hit => hit.Distance)
            .Cast<SceneObjectPickHit?>()
            .FirstOrDefault();

        if (bestSceneHit.HasValue)
        {
            SceneObjectPickHit hit = bestSceneHit.Value;
            info = hit.ObjectType == ObjectType.WmoDoodad
                ? BuildHoveredWmoDoodadInfo(hit)
                : BuildHoveredScenePickHitInfo(hit);
            distance = hit.Distance;
        }

        if (_showWlLiquids && _wlLoader != null)
        {
            Vector3 padding = new(2f, 2f, 1f);
            for (int i = 0; i < _wlLoader.Bodies.Count; i++)
            {
                WlLiquidBody body = _wlLoader.Bodies[i];
                if (liquidRenderer != null && !liquidRenderer.IsWlBodyVisible(body.BodyKey))
                    continue;

                float t = RayAABBIntersect(rayOrigin, rayDir, body.BoundsMin - padding, body.BoundsMax + padding);
                if (t < 0f || !IsHoverPickDistanceAllowed(t) || t >= distance)
                    continue;

                info = BuildHoveredWlLiquidInfo(body);
                distance = t;
            }
        }

        return distance < float.MaxValue;
    }

    private static IReadOnlyList<SceneObjectPickHit> ApplyWmoContainerFallThrough(IReadOnlyList<SceneObjectPickHit> hits)
    {
        if (hits.Count <= 1)
            return hits;

        var candidates = new WmoContainerFallThroughFilter.CandidateObject[hits.Count];
        for (int index = 0; index < hits.Count; index++)
        {
            SceneObjectPickHit hit = hits[index];
            candidates[index] = new WmoContainerFallThroughFilter.CandidateObject(
                index,
                hit.ObjectType == ObjectType.Wmo,
                hit.BoundsMin,
                hit.BoundsMax,
                hit.SelectionPoint);
        }

        IReadOnlyList<WmoContainerFallThroughFilter.CandidateObject> filtered =
            WmoContainerFallThroughFilter.ApplyFallThrough(candidates);
        if (filtered.Count == hits.Count)
            return hits;

        var result = new List<SceneObjectPickHit>(filtered.Count);
        foreach (WmoContainerFallThroughFilter.CandidateObject candidate in filtered)
            result.Add(hits[candidate.Id]);
        return result;
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

    public void ToggleObjects() => _objectsVisible = !_objectsVisible;
    public void ToggleWmos() => _wmosVisible = !_wmosVisible;
    public void ToggleDoodads() => _doodadsVisible = !_doodadsVisible;
    public bool ObjectFogEnabled
    {
        get => _objectFogEnabled;
        set => _objectFogEnabled = value;
    }

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
        if (TryPickSceneObjectByRay(rayOrigin, rayDir, out ObjectType bestType, out int bestIndex, out _))
        {
            _selectedObjectType = bestType;
            _selectedObjectIndex = bestIndex;
            if (TryGetSceneObjectByIndex(bestType, bestIndex, out ObjectInstance selectedInstance))
                _selectedSceneObjectKey = CreateSelectedSceneObjectKey(bestType, selectedInstance);
            return;
        }

        _selectedObjectType = ObjectType.None;
        _selectedObjectIndex = -1;
        _selectedSceneObjectKey = null;
    }

    public bool SelectSceneObject(ObjectType objectType, int objectIndex, int parentWmoIndex = -1)
    {
        if (_instancesDirty)
            RebuildInstanceLists();

        switch (objectType)
        {
            case ObjectType.Wmo when objectIndex >= 0 && objectIndex < _wmoInstances.Count:
                _selectedObjectType = objectType;
                _selectedObjectIndex = objectIndex;
                _selectedWmoParentIndex = -1;
                _selectedSceneObjectKey = CreateSelectedSceneObjectKey(objectType, _wmoInstances[objectIndex]);
                return true;
            case ObjectType.Mdx when objectIndex >= 0 && objectIndex < _mdxInstances.Count:
                _selectedObjectType = objectType;
                _selectedObjectIndex = objectIndex;
                _selectedWmoParentIndex = -1;
                _selectedSceneObjectKey = CreateSelectedSceneObjectKey(objectType, _mdxInstances[objectIndex]);
                return true;
            case ObjectType.WmoDoodad when parentWmoIndex >= 0 && parentWmoIndex < _wmoInstances.Count:
                _selectedObjectType = objectType;
                _selectedObjectIndex = objectIndex;
                _selectedWmoParentIndex = parentWmoIndex;
                _selectedSceneObjectKey = null;
                return true;
            default:
                return false;
        }
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

        if (clickedChunkKey.HasValue && hits.Any(static hit => hit.SharesClickedChunk))
            hits.RemoveAll(static hit => !hit.SharesClickedChunk);

        hits.Sort(static (left, right) =>
        {
            int clickedChunkCompare = right.SharesClickedChunk.CompareTo(left.SharesClickedChunk);
            if (clickedChunkCompare != 0)
                return clickedChunkCompare;

            int chunkDistanceCompare = left.ChunkGridDistance.CompareTo(right.ChunkGridDistance);
            if (chunkDistanceCompare != 0)
                return chunkDistanceCompare;

            int centroidCompare = left.SelectionPointDistanceSq.CompareTo(right.SelectionPointDistanceSq);
            if (centroidCompare != 0)
                return centroidCompare;

            return left.Distance.CompareTo(right.Distance);
        });

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

            if (!TryRayIntersectInstanceBounds(rayOrigin, rayDir, instance, padding, out float distance) || !IsHoverPickDistanceAllowed(distance))
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
                if (!IsHoverPickDistanceAllowed(dh.distance))
                    continue;

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

    private static Vector3 GetSceneObjectSelectionPoint(in ObjectInstance instance)
    {
        if (instance.BoundsResolved)
        {
            Vector3 boundsCenter = (instance.BoundsMin + instance.BoundsMax) * 0.5f;
            if (float.IsFinite(boundsCenter.X) && float.IsFinite(boundsCenter.Y) && float.IsFinite(boundsCenter.Z))
                return boundsCenter;
        }

        return instance.PlacementPosition;
    }

    private static bool TryGetSceneObjectChunkKey(in ObjectInstance instance, out (int tileX, int tileY, int chunkX, int chunkY) key)
    {
        Vector3 selectionPoint = GetSceneObjectSelectionPoint(instance);
        return TryGetTerrainChunkKey(selectionPoint.X, selectionPoint.Y, out key);
    }

    private static bool TryGetTerrainChunkKey(float worldX, float worldY, out (int tileX, int tileY, int chunkX, int chunkY) key)
    {
        key = default;

        float dx = WoWConstants.MapOrigin - worldX;
        float dy = WoWConstants.MapOrigin - worldY;
        if (float.IsNaN(dx) || float.IsNaN(dy) || float.IsInfinity(dx) || float.IsInfinity(dy))
            return false;

        int tileX = (int)MathF.Floor(dx / WoWConstants.ChunkSize);
        int tileY = (int)MathF.Floor(dy / WoWConstants.ChunkSize);
        if (tileX < 0 || tileX >= WoWConstants.TilesPerMapEdge || tileY < 0 || tileY >= WoWConstants.TilesPerMapEdge)
            return false;

        float localX = dx - tileX * WoWConstants.ChunkSize;
        float localY = dy - tileY * WoWConstants.ChunkSize;
        float chunkSize = WoWConstants.ChunkSize / WoWConstants.ChunksPerTileEdge;

        int chunkY = Math.Clamp((int)MathF.Floor(localX / chunkSize), 0, WoWConstants.ChunksPerTileEdge - 1);
        int chunkX = Math.Clamp((int)MathF.Floor(localY / chunkSize), 0, WoWConstants.ChunksPerTileEdge - 1);

        key = (tileX, tileY, chunkX, chunkY);
        return true;
    }

    public void ClearSelection()
    {
        _selectedObjectType = ObjectType.None;
        _selectedObjectIndex = -1;
        _selectedWmoParentIndex = -1;
        _selectedSceneObjectKey = null;
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

    private bool IsHoverPickDistanceAllowed(float distance)
    {
        if (!_limitHoveredAssetRange)
            return true;

        return distance <= ComputeEffectiveHoveredAssetMaxDistance();
    }

    private bool IsHoverPickPositionAllowed(Vector3 worldPosition)
    {
        if (!_limitHoveredAssetRange)
            return true;

        Vector3 cameraPosition = _pm4Overlay.GetPm4LoadAnchorCameraPosition();
        return Vector3.Distance(cameraPosition, worldPosition) <= ComputeEffectiveHoveredAssetMaxDistance();
    }

    /// <summary>
    /// Ray-AABB slab intersection test. Returns distance along ray, or -1 if no hit.
    /// </summary>
    public static float RayAABBIntersect(Vector3 origin, Vector3 dir, Vector3 bmin, Vector3 bmax)
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

    /// <summary>
    /// Resolves a PM4 object to the placed asset that produced it.
    /// </summary>
    /// <remarks>
    /// The join key is the object's own <c>MSUR._0x1C</c> read as a float, which equals the
    /// producing placement's Z, combined with the placement standing inside the object's horizontal
    /// footprint. The stored <c>Ck24</c> is the top 24 bits of that float, so the reconstruction
    /// carries roughly 0.003% relative error and the tolerance is sized for it.
    /// </remarks>
    /// <summary>
    /// True when the scene already has a placed WMO standing inside these bounds.
    /// </summary>
    private bool HasRealPlacementInside(Vector3 boundsMin, Vector3 boundsMax)
    {
        foreach (ObjectInstance inst in _wmoInstances)
        {
            Vector3 p = inst.PlacementPosition;
            if (p.X >= boundsMin.X - 1f && p.X <= boundsMax.X + 1f
                && p.Y >= boundsMin.Y - 1f && p.Y <= boundsMax.Y + 1f
                && p.Z >= boundsMin.Z - 1f && p.Z <= boundsMax.Z + 1f)
            {
                return true;
            }
        }

        return false;
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

    private (int PreparedPrimitiveCount, int SubmittedPrimitiveCount) RenderWireframeReveal(Matrix4x4 view, Matrix4x4 proj, Vector3 cameraPos,
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

    private (int PreparedPrimitiveCount, int SubmittedPrimitiveCount) RenderVisibleObjectWireframeOverlay(WorldRenderFrame frame, Matrix4x4 view, Matrix4x4 proj,
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

    private static string FormatAttributeMaskLabel(byte value)
    {
        if (value == 0) return "AttrMask 0x00 (none)";
        List<string> bits = [];
        if ((value & 0x01) != 0) bits.Add("bit0");
        if ((value & 0x02) != 0) bits.Add("bit1");
        if ((value & 0x04) != 0) bits.Add("bit2");
        if ((value & 0x08) != 0) bits.Add("bit3");
        if ((value & 0x10) != 0) bits.Add("bit4");
        if ((value & 0x20) != 0) bits.Add("bit5");
        if ((value & 0x40) != 0) bits.Add("bit6");
        if ((value & 0x80) != 0) bits.Add("bit7");
        return $"AttrMask 0x{value:X2} ({string.Join("|", bits)})";
    }

    private static byte PickPrimaryTypeFlag(uint mask)
    {
        // Prefer known flags in priority order; fall back to lowest set bit
        if ((mask & (1u << 0x12)) != 0) return 0x12;  // exterior solid
        if ((mask & (1u << 0x10)) != 0) return 0x10;  // interior floor
        if ((mask & (1u << 0x03)) != 0) return 0x03;  // M2 top
        // Pick the lowest set bit for unknown flags
        for (int bit = 1; bit < 32; bit++)
        {
            if ((mask & (1u << bit)) != 0)
                return (byte)bit;
        }
        return 0;
    }

    internal static void TransformBounds(Vector3 boundsMin, Vector3 boundsMax, in Matrix4x4 transform,
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
        _pm4Overlay.ReleasePm4LoadCancellation(cancelPendingLoad: true);
        _audioRuntime?.Dispose();
        _audioRuntime = null;
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
        _tileMdxVisibilityBuckets.Clear();
        _tileWmoVisibilityBuckets.Clear();
        _externalMdxInstances.Clear();
        _externalSkyboxInstances.Clear();
        _externalWmoInstances.Clear();
        _pm4Overlay._pm4TileObjects.Clear();
        _pm4Overlay._pm4TileMscnPoints.Clear();
        _pm4Overlay._pm4TileMspvPoints.Clear();
        _pm4Overlay._pm4TileStats.Clear();
        _pm4Overlay._pm4TilePositionRefs.Clear();
        _pm4Overlay._pm4ResearchBySourcePath.Clear();
        _pm4Overlay._pm4ResearchUnavailablePaths.Clear();
        _pm4Overlay._pm4ObjectLookup.Clear();
        _pm4Overlay._highlightedPm4ObjectKeys.Clear();
        _pm4Overlay._pm4MergedObjectGroupKeys.Clear();
        _pm4Overlay._pm4GroupToObjectKeys.Clear();
        _pm4Overlay._pm4ObjectGroupBounds.Clear();
        _pm4Overlay._pm4TileCk24Bounds.Clear();
        _pm4Overlay._pm4ObjectTranslations.Clear();
        _pm4Overlay._pm4ObjectRotationsDegrees.Clear();
        _pm4Overlay._pm4ObjectScales.Clear();
        _pm4Overlay._pm4TileCk24Translations.Clear();
        _pm4Overlay._pm4TileCk24RotationsDegrees.Clear();
        _pm4Overlay._pm4TileCk24Scales.Clear();
    }
}

/// <summary>
/// Lightweight placement instance — just a model key and world transform.
/// The actual renderer is looked up from WorldAssetManager at render time.
/// </summary>
public enum ObjectType { None, Wmo, Mdx, WmoDoodad }

public enum UniqueIdVisibilityScope
{
    PerMap,
    CameraTile
}

public readonly struct UniqueIdArchaeologyLayer
{
    public UniqueIdArchaeologyLayer(int layerNumber, int minUniqueId, int maxUniqueId, int placementCount, int wmoCount, int mdxCount)
    {
        LayerNumber = layerNumber;
        MinUniqueId = minUniqueId;
        MaxUniqueId = maxUniqueId;
        PlacementCount = placementCount;
        WmoCount = wmoCount;
        MdxCount = mdxCount;
    }

    public int LayerNumber { get; }
    public int MinUniqueId { get; }
    public int MaxUniqueId { get; }
    public int PlacementCount { get; }
    public int WmoCount { get; }
    public int MdxCount { get; }
}

    public readonly struct HoveredAssetInfo
{
    public HoveredAssetInfo(
        string assetKind,
        string displayName,
        string sourcePath,
        string detailLine,
        Vector3 worldPosition,
        int additionalHitCount,
        (int tileX, int tileY, uint ck24, int objectPart)? pm4ObjectKey,
        ObjectType sceneObjectType = ObjectType.None,
        int sceneObjectIndex = -1,
        string? wlBodyKey = null,
        bool isPreciseRayHit = false,
        int parentWmoIndex = -1,
        string? parentSourcePath = null)
    {
        AssetKind = assetKind ?? string.Empty;
        DisplayName = displayName ?? string.Empty;
        SourcePath = sourcePath ?? string.Empty;
        DetailLine = detailLine ?? string.Empty;
        WorldPosition = worldPosition;
        AdditionalHitCount = Math.Max(0, additionalHitCount);
        Pm4ObjectKey = pm4ObjectKey;
        SceneObjectType = sceneObjectType;
        SceneObjectIndex = sceneObjectIndex;
        WlBodyKey = wlBodyKey ?? string.Empty;
        IsPreciseRayHit = isPreciseRayHit;
        ParentWmoIndex = parentWmoIndex;
        ParentSourcePath = parentSourcePath ?? string.Empty;
    }

    public string AssetKind { get; }
    public string DisplayName { get; }
    public string SourcePath { get; }
    public string DetailLine { get; }
    public Vector3 WorldPosition { get; }
    public int AdditionalHitCount { get; }
    public (int tileX, int tileY, uint ck24, int objectPart)? Pm4ObjectKey { get; }
    public ObjectType SceneObjectType { get; }
    public int SceneObjectIndex { get; }
    public string WlBodyKey { get; }
    public bool IsPreciseRayHit { get; }
    public int ParentWmoIndex { get; }
    public string ParentSourcePath { get; }
    public bool HasSceneObject => SceneObjectType is ObjectType.Mdx or ObjectType.Wmo or ObjectType.WmoDoodad && SceneObjectIndex >= 0;

    public HoveredAssetInfo WithPreciseRayHit() => new(
        AssetKind, DisplayName, SourcePath, DetailLine, WorldPosition, AdditionalHitCount, Pm4ObjectKey,
        SceneObjectType, SceneObjectIndex, WlBodyKey, isPreciseRayHit: true, ParentWmoIndex, ParentSourcePath);
}

public readonly record struct SceneObjectPickHit(
    ObjectType ObjectType,
    int ObjectIndex,
    float Distance,
    string ModelName,
    string ModelPath,
    int UniqueId,
    Vector3 PlacementPosition,
    Vector3 BoundsMin,
    Vector3 BoundsMax,
    Vector3 SelectionPoint,
    float SelectionPointDistanceSq,
    bool SharesClickedChunk,
    int ChunkGridDistance,
    int ParentWmoIndex = -1)
{
    public string KindLabel => ObjectType switch
    {
        ObjectType.Wmo => "WMO",
        ObjectType.WmoDoodad => "WMO Doodad",
        _ => "MDX"
    };
}
